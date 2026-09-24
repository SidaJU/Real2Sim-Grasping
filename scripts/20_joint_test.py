"""T4-3b  Joint-space calibration (v2, crash-proof).

v1 died on four things at once; all fixed here:
  * physics tensors need a few frames after play() -> tick() tolerates the
    warm-up frames instead of asserting
  * finish() accepts the message argument
  * name/target are initialised before the first callback fires
  * hand prim matched EXACTLY ("panda_hand"), not the joint "panda_hand_joint"
"""

import numpy as np
import omni.usd
import omni.timeline
import omni.physx
from isaacsim.core.experimental.prims import Articulation
from pxr import UsdGeom

ROBOT = "/World/Franka"
GRIP_OPEN = 0.04

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
    if hand_prim is None or not hand_prim.IsValid():
        return None
    try:
        t = UsdGeom.XformCache(0.0).GetLocalToWorldTransform(hand_prim).ExtractTranslation()
        return (t[0], t[1], t[2])
    except Exception:
        return None


def full_targets(arm7, grip):
    full = np.asarray(art.get_dof_positions().numpy(), dtype=np.float64).reshape(-1).copy()
    arm_idx = [i for i in range(len(full)) if i not in grip_idx]
    full[arm_idx] = arm7
    full[grip_idx] = grip
    return full


POSES = [
    # extrapolated from the ready/reach1/reach2 calibration:
    # each +0.77 rad on j2 moved the hand y +0.165, z -0.232
    ("aim1", [0.00,  0.560, 0.00, -1.770, 0.00, 1.370, 0.780]),   # predicted (0.30, 0.00, 0.13)
    ("aim2", [0.00,  0.450, 0.00, -1.900, 0.00, 1.550, 0.780]),   # predicted (0.30, -0.02, 0.16)
]


class JointTest:
    def __init__(self):
        self.sub = None
        self.i = -1
        self.step = 0
        self.settled = 0
        self.target = None
        self.name = "(init)"
        self.results = []
        self.warmup = 30
        omni.timeline.get_timeline_interface().play()
        self.sub = omni.physx.get_physx_interface().subscribe_physics_step_events(self.on_step)
        print("[test] started - auto-unsubscribes after the last pose")

    def stop(self):
        if self.sub is not None:
            try:
                self.sub.unsubscribe()
                print("[test] unsubscribed - nothing left running")
            except Exception as e:
                print("[test] unsubscribe error:", e)
            self.sub = None

    def next_pose(self):
        self.i += 1
        if self.i >= len(POSES):
            self.finish()
            return
        self.name, arm = POSES[self.i]
        try:
            self.target = full_targets(arm, GRIP_OPEN)
            art.set_dof_position_targets(self.target)
            self.step, self.settled = 0, 0
            print(f"[{self.name}] targets sent: {np.round(arm, 2)}")
        except Exception as e:
            print(f"[{self.name}] set targets failed: {e}")
            self.finish()

    def finish(self, msg="done"):
        for name, fp in self.results:
            print(f"  RESULT {name}: hand at {np.round(fp, 3)}")
        print(f"[test] {msg}")
        self.stop()

    def on_step(self, dt):
        try:
            self._tick(dt)
        except Exception:
            import traceback
            traceback.print_exc()
            self.finish("CRASHED - unsubscribed")

    def _tick(self, dt):
        if self.warmup > 0:
            self.warmup -= 1
            return                                  # let the tensors come alive
        if self.target is None:
            self.next_pose()
            return
        try:
            q = np.asarray(art.get_dof_positions().numpy(), dtype=np.float64).reshape(-1)
        except AssertionError:
            return                                  # tensors not ready yet - skip
        self.step += 1
        err = float(np.abs(q - self.target).max())
        self.settled = self.settled + 1 if err < 0.02 else 0
        if self.step % 120 == 0:
            print(f"  [{self.name}] step {self.step} err={err:.3f}")
        if self.settled >= 20 or self.step > 480:
            fp = hand_pos()
            self.results.append((self.name, fp if fp else (np.nan,) * 3))
            self.next_pose()


_prev = globals().get("_test")
if _prev is not None and getattr(_prev, "sub", None) is not None:
    try:
        _prev.sub.unsubscribe()
    except Exception:
        pass

_test = JointTest()
