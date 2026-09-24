"""T4-3c  Numeric-Jacobian IK grasp trial.

Why: RmpFlow reported convergence while the arm never moved, and hand-crafted
joint guesses land in unpredictable places (7-DOF redundancy). So we MEASURE:
nudge j2/j4/j6 one at a time, see how the hand moves, build a numeric Jacobian,
and solve for the correction. Two passes get us onto the box.

Pipeline (each phase auto-advances, the callback is removed at the end):
    raise box (it is buried under the table)
    measure -> solve -> verify
    close gripper
    lift
    judge (did the box rise?)

Manual stop at any time: _trial.stop()
"""

from collections import deque
import numpy as np
import omni.usd
import omni.timeline
import omni.physx
from isaacsim.core.experimental.prims import Articulation
from pxr import UsdGeom, Gf

ROBOT, BOX_PRIM = "/World/Franka", "/World/GraspObject"
TABLE_Z = 0.025
HAND_CLEARANCE = 0.13      # hand centre sits this far above the fingertips
DOF_IDX = [1, 3, 5]        # j2, j4, j6
DELTA = 0.10
GRIP_OPEN, GRIP_CLOSE = 0.04, 0.005
LIFT = 0.08

art = Articulation(ROBOT)
names = list(art.dof_names)
grip_idx = [i for i, n in enumerate(names) if "finger" in n]

stage = omni.usd.get_context().get_stage()
hand_prim = None
for p in stage.Traverse():
    if p.GetName() == "panda_hand":
        hand_prim = p
print("hand prim:", hand_prim.GetPath() if hand_prim else "NOT FOUND")


def hand_pos():
    if hand_prim is None:
        return None
    t = UsdGeom.XformCache(0.0).GetLocalToWorldTransform(hand_prim).ExtractTranslation()
    return np.array([t[0], t[1], t[2]])


def box_bounds():
    prim = stage.GetPrimAtPath(BOX_PRIM)
    r = UsdGeom.Boundable(prim).ComputeWorldBound(0.0, "default").ComputeAlignedRange()
    return np.array(r.GetMin()), np.array(r.GetMax())


def raise_box():
    mn, mx = box_bounds()
    dz = (TABLE_Z + 0.002) - mn[2]
    if dz <= 0.001:
        print(f"box already above table (bottom {mn[2]:.3f})")
        return
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(BOX_PRIM))
    for op in xf.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate":
            v = op.Get()
            op.Set(Gf.Vec3d(v[0], v[1], v[2] + dz))
            print(f"raised box by {dz:.3f} m")
            return
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, dz))
    print(f"raised box by {dz:.3f} m (new op)")


class Trial:
    def __init__(self):
        raise_box()
        self.mn, self.mx = box_bounds()
        self.target = np.array([
            (self.mn[0] + self.mx[0]) / 2.0,
            (self.mn[1] + self.mx[1]) / 2.0,
            self.mx[2] + HAND_CLEARANCE,
        ])
        print("box bbox:", np.round(self.mn, 3), np.round(self.mx, 3))
        print("hand target:", np.round(self.target, 3))

        self.base = None        # filled on the first physics step (tensors
                                # are only valid once the timeline is playing)
        self.queue = deque()
        self.hands = {}
        self.pass_n = 0
        self.cur = None
        self.step = 0
        self.settled = 0
        self.box_z0 = self.box_pos()[2]   # baseline BEFORE any grasping
        print(f"box centre z before grasp: {self.box_z0:.4f}")
        self.warm = 20
        tl = omni.timeline.get_timeline_interface()
        tl.stop()
        tl.play()
        self.sub = omni.physx.get_physx_interface().subscribe_physics_step_events(self.on_step)
        print("[trial] started - it will unsubscribe itself when done")

    # ---------- queue helpers ----------
    def q(self, delta3, grip):
        out = self.base.copy()
        out[DOF_IDX] = out[DOF_IDX] + delta3
        out[grip_idx] = grip
        return out

    def enqueue_measure(self):
        self.queue.append({"k": "goto", "tag": "h0", "d": np.zeros(3),
                           "g": GRIP_OPEN, "q": self.q(np.zeros(3), GRIP_OPEN)})
        for i in range(3):
            d = np.zeros(3)
            d[i] = DELTA
            self.queue.append({"k": "goto", "tag": f"h{i+1}", "d": d,
                               "g": GRIP_OPEN, "q": self.q(d, GRIP_OPEN)})
        self.queue.append({"k": "solve"})

    def stop(self):
        if self.sub is not None:
            try:
                self.sub.unsubscribe()
                print("[trial] unsubscribed - nothing left running")
            except Exception as e:
                print("[trial] unsubscribe error:", e)
            self.sub = None

    def finish(self, msg="done"):
        print(f"[trial] {msg}")
        self.stop()

    # ---------- loop ----------
    def on_step(self, dt):
        try:
            self._tick(dt)
        except Exception:
            import traceback
            traceback.print_exc()
            self.finish("CRASHED - unsubscribed")

    def _tick(self, dt):
        if self.warm > 0:
            self.warm -= 1
            return
        if self.base is None:
            # first physics step after play - tensors are valid now
            try:
                self.base = np.asarray(
                    art.get_dof_positions().numpy(), dtype=np.float64
                ).reshape(-1).copy()
            except AssertionError:
                return
            print("base joints:", np.round(self.base, 3))
            self.enqueue_measure()
        if not self.queue:
            self.finish("queue empty")
            return
        if self.cur is None:
            self.cur = self.queue.popleft()
            self.step, self.settled = 0, 0
            if self.cur["k"] in ("goto", "hold"):
                try:
                    art.set_dof_position_targets(self.cur["q"])
                except Exception as e:
                    print("set targets failed:", e)
                    self.finish("aborted")
                    return

        self.step += 1
        c = self.cur

        if c["k"] == "solve":
            self.do_solve()
            self.cur = None
            return

        if c["k"] == "grip":
            if self.step == 1:
                full = self.base.copy()
                full[grip_idx] = c["v"]
                art.set_dof_position_targets(full)
            if self.step >= c["frames"]:
                self.cur = None
            return

        if c["k"] == "judge":
            if self.step >= 30:
                z = self.box_pos()[2]
                rose = z - self.box_z0
                print(f"\nbox z {z:.4f} (start {self.box_z0}, rose {rose:+.4f} m)")
                print("GRASP RESULT:", "SUCCESS" if rose > 0.05 else "FAILURE")
                self.cur = None
                self.finish("trial complete")
            return

        # goto / hold: wait for the joints to settle
        try:
            qnow = np.asarray(art.get_dof_positions().numpy(), dtype=np.float64).reshape(-1)
        except AssertionError:
            return
        err = float(np.abs(qnow - c["q"]).max())
        self.settled = self.settled + 1 if err < 0.02 else 0
        if self.step % 120 == 0:
            print(f"  [{c.get('tag', c['k'])}] step {self.step} err={err:.3f}")
        if self.settled >= 12 or self.step > 360:
            if "tag" in c:
                self.hands[c["tag"]] = hand_pos()
                print(f"  {c['tag']}: hand {np.round(self.hands[c['tag']], 3)}")
            self.cur = None

    def do_solve(self):
        h0, hs = self.hands.get("h0"), [self.hands.get(f"h{i+1}") for i in range(3)]
        if h0 is None or any(h is None for h in hs):
            self.finish("missing hand samples")
            return
        J = np.zeros((3, 3))
        for k in range(3):
            J[:, k] = (hs[k] - h0) / DELTA
        e = self.target - h0
        dq = np.linalg.pinv(J) @ e
        self.base[DOF_IDX] = self.base[DOF_IDX] + dq
        self.pass_n += 1
        print(f"  pass {self.pass_n}: |err| {np.linalg.norm(e)*1000:.0f} mm, dq {np.round(dq, 3)}")
        if self.pass_n >= 3 or np.linalg.norm(e) < 0.02:
            # go to the solved pose, then close -> lift -> judge
            self.queue.append({"k": "goto", "tag": "final", "d": np.zeros(3),
                               "g": GRIP_OPEN, "q": self.q(np.zeros(3), GRIP_OPEN)})
            self.queue.append({"k": "grip", "v": GRIP_CLOSE, "frames": 40})
            dz = np.array([0.0, 0.0, LIFT])
            dq_up = np.linalg.pinv(J) @ dz
            self.queue.append({"k": "goto", "tag": "lifted", "d": dq_up,
                               "g": GRIP_CLOSE, "q": self.q(dq_up, GRIP_CLOSE)})
            self.queue.append({"k": "judge"})
        else:
            self.enqueue_measure()

    def box_pos(self):
        mn, mx = box_bounds()
        return (mx + mn) / 2.0


_prev = globals().get("_trial")
if _prev is not None and getattr(_prev, "sub", None) is not None:
    try:
        _prev.sub.unsubscribe()
    except Exception:
        pass

_trial = Trial()
