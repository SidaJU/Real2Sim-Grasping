"""T4-3  Grasp trial via PHYSICS CALLBACK state machine (self-terminating).

Why not a coroutine: Kit's async_engine does not reliably drive asyncio.sleep,
so the coroutine hung on its first await and nothing after it ever ran.

This version subscribes to physics step events and walks a state machine:
    open -> hover -> descend -> close -> lift -> judge -> AUTO-UNSUBSCRIBE
It ALWAYS unsubscribes when finished (success, failure or timeout), so no
ghost logic is ever left running.

Stop it manually at any time with:   _fsm.stop()
"""

import numpy as np
import omni.usd
import omni.timeline
import omni.physx
import warp as wp
import isaacsim.robot_motion.experimental.motion_generation as mg
from isaacsim.robot_motion.cumotion import (
    CumotionWorldInterface,
    RmpFlowController,
    load_cumotion_supported_robot,
)
from isaacsim.core.experimental.prims import Articulation, GeomPrim
from pxr import UsdGeom

ROBOT, BOX_PRIM = "/World/Franka", "/World/GraspObject"
# these get OVERWRITTEN from the real world bounding box at FSM start
# (the obj->usd converter re-arranged vertex coordinates, so the prim
#  translate is NOT the visual box position)
BOX_TOP_Z, BOX_CX, BOX_CY = 0.072, 0.30, 0.0


def box_bounds():
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(BOX_PRIM)
    bbox = UsdGeom.Boundable(prim).ComputeWorldBound(0.0, "default")
    rng = bbox.ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    return (mn[0], mn[1], mn[2]), (mx[0], mx[1], mx[2])
HOVER_H, DESCEND_GAP, LIFT_DZ = 0.15, 0.015, 0.10
GRIP_OPEN, GRIP_CLOSE = 0.04, 0.005
SUCCESS_Z, MAX_STEPS, TOL, HOLD = 0.10, 1200, 0.015, 15

art = Articulation(ROBOT)
box = GeomPrim(BOX_PRIM)
cr = load_cumotion_supported_robot("franka")
tool = cr.robot_description.tool_frame_names()[0]
names = list(art.dof_names)
grip_idx = [i for i, n in enumerate(names) if "finger" in n]

controller = RmpFlowController(
    cumotion_robot=cr,
    cumotion_world_interface=CumotionWorldInterface(),
    robot_joint_space=names,
    robot_site_space=[tool],
    tool_frame=tool,
)
print("controller ready | tool:", tool)

# ---- kill any previous FSM left over from an earlier run ----
_prev = globals().get("_fsm")
if _prev is not None and getattr(_prev, "sub", None) is not None:
    try:
        _prev.sub.unsubscribe()
        print("unsubscribed stale FSM from a previous run")
    except Exception:
        pass


def est():
    q = art.get_dof_positions()
    return mg.RobotState(joints=mg.JointState.from_name(
        robot_joint_space=names, positions=(names, q), velocities=(names, art.get_dof_velocities()),
    ))


def setpoint(pos):
    p = wp.array([list(pos)], dtype=wp.float32)
    o = wp.array([[0.0, 1.0, 0.0, 0.0]], dtype=wp.float32)
    return mg.RobotState(sites=mg.SpatialState.from_name(
        spatial_space=[tool], positions=([tool], p), orientations=([tool], o),
    ))


def apply(desired):
    if desired is None or desired.joints.positions is None:
        return None
    # DEBUG: dump the real structure once so we stop guessing
    if not globals().get("_dbg_printed"):
        globals()["_dbg_printed"] = True
        print("DBG positions:", desired.joints.positions.numpy().shape,
              "indices:", desired.joints.position_indices.numpy().shape)
        print("DBG positions[:3]:", np.asarray(
            desired.joints.positions.numpy()).reshape(-1)[:3])
    # set the FULL dof vector without dof_indices - the indexed variant
    # internally calls ApplyMultipleApplyAPI and blows up on this robot
    full = np.asarray(art.get_dof_positions().numpy(), dtype=np.float64).reshape(-1).copy()
    q = np.asarray(desired.joints.positions.numpy(), dtype=np.float64).reshape(-1)
    # position_indices turned out to be UNRELIABLE (its shape morphs between
    # calls: (7,) then (7,9)). DBG proved positions are already in plain DOF
    # order, so map them positionally and ignore the indices entirely.
    if q.size == len(full):
        full[:] = q
    else:
        full[: len(q)] = q
    art.set_dof_position_targets(full)
    return full


def set_grip(open_):
    v = GRIP_OPEN if open_ else GRIP_CLOSE
    full = np.asarray(art.get_dof_positions().numpy(), dtype=np.float64).reshape(-1).copy()
    full[grip_idx] = v
    art.set_dof_position_targets(full)


def box_z():
    return float(box.get_world_poses()[0].numpy()[0][2])


def box_xy():
    """Live XY of the reconstructed mesh (never hard-code it - the scene has
    a second, NON-physical YCB box that we must not target)."""
    p = box.get_world_poses()[0].numpy()[0]
    return float(p[0]), float(p[1])


def fingertip_pos():
    stage = omni.usd.get_context().get_stage()
    xc = UsdGeom.XformCache(0.0)
    found = []
    for p in stage.Traverse():
        n = p.GetName().lower()
        if "finger" in n or "tip" in n or "hand" in n:
            found.append(str(p.GetPath()))
    if not found:
        print("  [warn] no finger/hand prim found in the stage")
        return None
    pick = None
    for f in found:
        if "leftfingertip" in f.lower():
            pick = f
            break
    if pick is None:
        pick = found[0]
        print("  [warn] leftfingertip not found, using:", pick, "| all:", found)
    t = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(pick)).ExtractTranslation()
    return (t[0], t[1], t[2])


def hover_pos():
    x, y = box_xy()
    return (x, y, BOX_TOP_Z + HOVER_H)


def lift_pos():
    x, y = box_xy()
    return (x, y, BOX_TOP_Z + HOVER_H + LIFT_DZ)


class GraspFSM:
    """Self-terminating grasp state machine driven by physics step events."""

    def __init__(self):
        self.sub = None
        self.phase = "open"
        self.step = 0
        self.stable = 0
        self.t = 0.0
        self.target = None
        # locate the box from its REAL world bounding box (prim translate lies)
        global BOX_CX, BOX_CY, BOX_TOP_Z
        mn, mx = box_bounds()
        BOX_CX = (mn[0] + mx[0]) / 2.0
        BOX_CY = (mn[1] + mx[1]) / 2.0
        BOX_TOP_Z = mx[2]
        print(f"box world bbox: min {tuple(round(v, 3) for v in mn)} "
              f"max {tuple(round(v, 3) for v in mx)}")
        print(f"grasp target: ({BOX_CX:.3f}, {BOX_CY:.3f})  top z={BOX_TOP_Z:.3f}")
        # stop->play resets physics: the box goes back to its authored spot
        tl = omni.timeline.get_timeline_interface()
        tl.stop()
        tl.play()
        self.box_z0 = box_z()   # baseline for the success test
        # NOTE: get_dof_positions only works AFTER the timeline plays, so the
        # first set_grip happens inside on_step (physics live by then).
        self.sub = omni.physx.get_physx_interface().subscribe_physics_step_events(self.on_step)
        print("[FSM] started - it WILL unsubscribe itself when done")

    def stop(self):
        if self.sub is not None:
            try:
                self.sub.unsubscribe()
                print("[FSM] unsubscribed - callback fully removed")
            except Exception as e:
                print("[FSM] unsubscribe error:", e)
            self.sub = None

    def enter(self, phase, target=None):
        print(f"[{phase}]")
        self.phase, self.step, self.stable, self.t = phase, 0, 0, 0.0
        self.target = target
        if target is not None:
            ok = controller.reset(est(), setpoint(target), t=0.0)
            print(f"  reset ok={ok}")

    def on_step(self, dt):
        # ANY crash inside the callback must also unsubscribe, otherwise a
        # broken callback keeps firing forever (the "ghost process" problem).
        try:
            self._tick(dt)
        except Exception:
            import traceback
            traceback.print_exc()
            self.finish("CRASHED - unsubscribed automatically")

    def _tick(self, dt):
        self.t += dt
        self.step += 1

        if self.phase == "open":
            if self.step == 0:
                set_grip(True)          # physics is live here
            if self.step >= 30:
                self.enter("hover", hover_pos())

        elif self.phase in ("hover", "descend", "lift"):
            if self.step % 240 == 1:
                print(f"  [{self.phase}] still approaching, step {self.step}")
            qt = apply(controller.forward(est(), setpoint(self.target), self.t))
            if qt is None:
                if self.step % 120 == 0:
                    print(f"  [{self.phase}] forward returned None (step {self.step})")
            else:
                q = np.asarray(art.get_dof_positions().numpy(), dtype=np.float64)
                err = float(np.abs(q - qt).max())
                self.stable = self.stable + 1 if err < TOL else 0
                if self.step % 120 == 0:
                    print(f"  [{self.phase}] step {self.step} joint err={err:.3f}")
                if self.stable >= HOLD:
                    self.on_phase_done()
            if self.step > MAX_STEPS:
                self.finish(f"{self.phase} TIMEOUT (err never converged)")

        elif self.phase == "close":
            apply(controller.forward(est(), setpoint(self.target), self.t))
            if self.step >= 45:
                self.enter("lift", lift_pos())

        elif self.phase == "judge":
            if self.step >= 15:
                z = box_z()
                rose = z - self.box_z0
                ok = rose > 0.05
                print(f"\nbox z: {z:.4f} (baseline {self.box_z0:.4f}, rose {rose:+.4f} m)")
                print("GRASP RESULT:", "SUCCESS" if ok else "FAILURE")
                set_grip(True)
                self.finish("trial complete")

    def on_phase_done(self):
        if self.phase == "hover":
            # NO compensation: the finger PRIM origin sits ~10 cm above the
            # real TCP, so prim-based offsets are garbage (they once pushed a
            # target below the floor). IK z is accurate on its own.
            x, y = box_xy()
            print(f"  target box (reconstructed mesh) at x={x:.3f} y={y:.3f}")
            ideal = np.array([x, y, BOX_TOP_Z + DESCEND_GAP])
            self.descend_target = tuple(ideal)
            self._desc_retried = 0
            self.enter("descend", self.descend_target)

        elif self.phase == "descend":
            fp = fingertip_pos()
            ideal = np.array(self.descend_target)
            if fp is not None:
                d = fp - ideal
                # only the xy residual is meaningful (prim z is not the TCP)
                print(f"  at descend: fingertip {np.round(fp, 3)} residual xy "
                      f"{np.round((d[0], d[1]), 3)} m")
                if np.abs(d[:2]).max() > 0.008 and self._desc_retried < 2:
                    self._desc_retried += 1
                    print(f"  re-approach #{self._desc_retried} with xy correction")
                    cur = np.array(self.target)
                    self.enter("descend", tuple(cur - np.array((d[0], d[1], 0.0))))
                    return
            else:
                print("  no fingertip reading at descend")
            print("  closing gripper")
            set_grip(False)
            self.enter("close", tuple(ideal))

        elif self.phase == "close":
            self.enter("lift", tuple(
                np.array(self.descend_target)
                + (0.0, 0.0, HOVER_H + LIFT_DZ - DESCEND_GAP)
            ))

        elif self.phase == "lift":
            self.enter("judge", None)

    def finish(self, msg):
        print(f"[FSM] {msg}")
        self.stop()


def desired_idx(desired):
    return desired.joints.position_indices.numpy()


_fsm = GraspFSM()
