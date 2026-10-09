"""T4-3  BACKUP / TOP-DOWN grasp (GRASP_SIDE = False) - the version that ran
end-to-end and produced a GRASP RESULT. Kept as a known-good fallback:
side grasping lives in 19d_grasp_side.py.


Why not a coroutine: Kit's async_engine does not reliably drive asyncio.sleep,
so the coroutine hung on its first await and nothing after it ever ran.

This version subscribes to physics step events and walks a state machine:
    open -> hover -> descend -> close -> lift -> judge -> AUTO-UNSUBSCRIBE
It ALWAYS unsubscribes when finished (success, failure or timeout), so no
ghost logic is ever left running.

HOW TO RUN: just Run this file in the Script Editor - the last line
constructs GraspFSM(), and __init__ starts it immediately. There is nothing
to call afterwards.

Stop it manually at any time with:   _fsm.stop()
"""

import math

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
from pxr import UsdGeom, Gf

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
# Franka finger joints: prismatic, range 0..0.04 m (0 = fully closed).
# The old CLOSE of 0.005 left a visible gap and the box slipped out during the
# lift: a finger target is only a POSITION command, so unless it is pushed
# past what the object physically allows, the drive settles early and the grip
# is loose. Commanding 0.0 (even beyond the 0.021 m the 42 mm-thick box
# permits) makes the drive saturate -> maximum squeeze for this drive gain.
GRIP_OPEN, GRIP_CLOSE = 0.04, 0.0
SUCCESS_Z, MAX_STEPS, TOL, HOLD = 0.10, 1200, 0.015, 15
# descend exits either by full convergence (err < TOL) or, once in CONTACT
# with the object, by step count: touching the box pins some joint ~0.1 rad
# away from the command, so err never reaches TOL and the trial would run to
# TIMEOUT and be judged a failure even though the grasp is perfectly placed.
DESCEND_STEPS, DESCEND_ERR_MAX = 90, 0.30
# TCP sinks this far BELOW the top face so the finger pads WRAP the top edge.
# At top+0.015 (the old value) the pads hover ABOVE the edge: closing then
# slaps the pads onto the top rim and shoves the box over instead of gripping.
DESCEND_SINK = 0.015

# ---- SIDE GRASP (flat slab on a pillar) -----------------------------------
# Approach HORIZONTALLY with the finger-opening axis VERTICAL: the upper pad
# presses the top face near the rim, the lower pad presses the bottom face -
# a clothespin grip on the 42 mm rim. Approach direction = from the robot
# base toward the object centre, stopping SIDE_INSET inside the near edge
# so both pads sit fully on the faces. Lift is straight up.
GRASP_SIDE = False
SIDE_INSET = 0.015

# Fallback only: distance from the panda_hand prim origin to the real TCP
# (between the fingertips), in metres. Used ONLY when the robot USD has no
# fingertip prim at all. Verify it: with the gripper open the TCP should sit
# between the two fingertips, not inside the palm.
TCP_FROM_HAND_Z = -0.1034

# Yaw of the finger-opening direction about the vertical axis, degrees.
# yaw=0 was the original (0,1,0,0) quaternion. For the box standing on its
# edge, the opening must straddle the THIN (0.042 m) pair of faces - if the
# fingers land on the wide faces instead, change this (0 / 90 / 180 / -90).
GRIP_YAW_DEG = 90.0

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


# ---- controller target frame ----------------------------------------------
# Empirically RmpFlowController interprets the setpoint in the ROBOT BASE
# frame, not the world frame. Evidence: base at (0.2,-0.6) rotZ 90, world
# target (0.021,-0.052,0.19) -> the arm curled up at its own chest, which is
# exactly where base+(0.021,-0.052,0.19) lands in world coordinates
# (= (0.252,-0.579), 2 cm in front of the base).
# So: convert world -> base before handing a target to the controller.
# If a future Isaac build interprets world frames, flip this to False.
TARGET_IN_BASE_FRAME = True


def world_to_base(pos):
    """Map a world-frame point into the robot base frame."""
    stage = omni.usd.get_context().get_stage()
    m = UsdGeom.XformCache(0.0).GetLocalToWorldTransform(
        stage.GetPrimAtPath(ROBOT))
    lp = m.GetInverse().Transform(Gf.Vec3d(*pos))
    return (lp[0], lp[1], lp[2])


def world_quat_to_base(q_w):
    """Rotate a world-frame tool orientation into the base frame (w,x,y,z)."""
    stage = omni.usd.get_context().get_stage()
    m = UsdGeom.XformCache(0.0).GetLocalToWorldTransform(
        stage.GetPrimAtPath(ROBOT))
    q_bw = m.ExtractRotationQuat()             # base -> world (Matrix4d builtin)
    q = q_bw.GetInverse() * Gf.Quatd(*q_w)     # world -> base
    im = q.GetImaginary()
    return (q.GetReal(), im[0], im[1], im[2])


def quat_from_axes(x_ax, y_ax, z_ax):
    """Tool axes (rotation-matrix columns) -> quaternion (w, x, y, z).

    Plain numpy (Shepperd's method) - no Gf.Rotation, whose matrix ctor is
    missing/strict in some pxr builds.
    """
    R = np.array([[x_ax[0], y_ax[0], z_ax[0]],
                  [x_ax[1], y_ax[1], z_ax[1]],
                  [x_ax[2], y_ax[2], z_ax[2]]], dtype=np.float64)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w, x = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        y, z = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return tuple(q / np.linalg.norm(q))


def side_quat(d):
    """Side-grasp orientation: tool z along the approach direction, tool y
    (the finger-opening axis) vertical."""
    z_ax = Gf.Vec3d(float(d[0]), float(d[1]), 0.0)
    y_ax = Gf.Vec3d(0.0, 0.0, 1.0)
    x_ax = Gf.Cross(y_ax, z_ax)
    return quat_from_axes(x_ax, y_ax, z_ax)


def side_targets():
    """(approach dir d, TCP point on the near rim, pull-back hover point)."""
    stage = omni.usd.get_context().get_stage()
    m = UsdGeom.XformCache(0.0).GetLocalToWorldTransform(
        stage.GetPrimAtPath(ROBOT))
    b = m.ExtractTranslation()
    cx, cy = box_xy()
    d = np.array([cx - b[0], cy - b[1], 0.0])
    d = d / np.linalg.norm(d)
    mn, mx = box_bounds()
    hx = (mx[0] - mn[0]) / 2.0
    hy = (mx[1] - mn[1]) / 2.0
    # distance from centre to the rim along d (support of the bbox)
    support = 0.5 * (hx * abs(d[0]) + hy * abs(d[1]))
    tcp = (np.array([cx, cy, (mn[2] + mx[2]) / 2.0])
           - d * (support - SIDE_INSET))
    back = tcp - d * HOVER_H                   # approach starts from here
    return d, tcp, back


def setpoint(pos, quat=None):
    if TARGET_IN_BASE_FRAME:
        pos = world_to_base(pos)
        quat = world_quat_to_base(quat) if quat is not None else None
    p = wp.array([list(pos)], dtype=wp.float32)
    if quat is None:
        # default: gripper straight down; GRIP_YAW_DEG spins the opening dir.
        t = math.radians(GRIP_YAW_DEG) / 2.0
        quat = (0.0, math.cos(t), math.sin(t), 0.0)
    o = wp.array([list(quat)], dtype=wp.float32)
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


# ---- SINGLE SOURCE OF TRUTH for where the object actually is --------------
# The obj->usd asset converter re-arranged the vertex coordinates, so the prim
# translate (box.get_world_poses()) is NOT where the mesh is. Measured on this
# scene: prim said (0.000, 0.377) while the geometry really sat at
# (0.001, -0.052) - a 0.43 m error that sent the arm to the wrong place.
# EVERY target below must come from the measured bbox, never from the prim.
_GEOM = {"cx": BOX_CX, "cy": BOX_CY, "top": BOX_TOP_Z}


def refresh_geom():
    """Re-read the object's real visual bbox and cache centre + top."""
    mn, mx = box_bounds()
    _GEOM["cx"] = (mn[0] + mx[0]) / 2.0
    _GEOM["cy"] = (mn[1] + mx[1]) / 2.0
    _GEOM["top"] = mx[2]
    print(f"geometry bbox: min {tuple(round(v, 3) for v in mn)} "
          f"max {tuple(round(v, 3) for v in mx)}")
    print(f"  -> target xy ({_GEOM['cx']:.3f}, {_GEOM['cy']:.3f}) "
          f"| top z {_GEOM['top']:.3f}")
    return _GEOM


def box_xy():
    """XY of the object, taken from the measured bbox (never the prim)."""
    return _GEOM["cx"], _GEOM["cy"]


def box_z():
    """Live height of the object = centre of its bbox (prim translate lies)."""
    mn, mx = box_bounds()
    return (mn[2] + mx[2]) / 2.0


def _prims_named(*keys):
    """All prims whose name contains any of `keys` (case-insensitive)."""
    stage = omni.usd.get_context().get_stage()
    out = []
    for p in stage.Traverse():
        n = p.GetName().lower()
        if any(k in n for k in keys):
            out.append(p.GetPath())
    return out


def fingertip_pos():
    """World position of the REAL tool centre point (TCP).

    cumotion controls `panda_leftfingertip`, but not every Franka USD ships
    that prim. The old code then silently fell back to `panda_hand`, whose
    origin sits ~10 cm away from the TCP - every xy "residual" it printed was
    off by that much, and the descend corrections chased a ghost.
    """
    stage = omni.usd.get_context().get_stage()
    xc = UsdGeom.XformCache(0.0)

    def world(path):
        tr = xc.GetLocalToWorldTransform(
            stage.GetPrimAtPath(path)).ExtractTranslation()
        return (tr[0], tr[1], tr[2])

    # 1) a real fingertip / tcp / tool prim: prefer the one cumotion controls
    tips = _prims_named("fingertip", "tcp", "tool")
    if tips:
        for t in tips:
            s = str(t).lower()
            if "left" in s or "tcp" in s or "tool" in s:
                return world(t)
        return world(tips[0])

    # 2) no fingertip prim at all: hand origin + a fixed TCP offset
    hands = _prims_named("hand")
    if not hands:
        print("  [warn] no finger/hand prim found in the stage")
        return None
    hp = hands[0]
    print(f"  [warn] no fingertip prim; using {hp} + TCP offset "
          f"{TCP_FROM_HAND_Z} m  <-- VERIFY this against your USD")
    m = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(hp))
    p = m.Transform(Gf.Vec3d(0.0, 0.0, TCP_FROM_HAND_Z))
    return (p[0], p[1], p[2])


def hover_pos():
    x, y = box_xy()
    return (x, y, _GEOM["top"] + HOVER_H)


def lift_pos():
    x, y = box_xy()
    return (x, y, _GEOM["top"] + HOVER_H + LIFT_DZ)


class GraspFSM:
    """Self-terminating grasp state machine driven by physics step events."""

    def __init__(self):
        self.sub = None
        self.phase = "open"
        self.step = 0
        self.stable = 0
        self.t = 0.0
        self.target = None
        self.quat = None
        self.side_q = None
        # locate the box from its REAL world bounding box (prim translate lies)
        # ORDER MATTERS: stop->play first. It resets physics and puts the box
        # back on its authored spot - reading the bbox BEFORE it measured a
        # box that had already toppled (z 0.001-0.038 instead of 0.030-0.072),
        # which then became the grasp target.
        tl = omni.timeline.get_timeline_interface()
        tl.stop()
        tl.play()
        refresh_geom()
        _m = UsdGeom.XformCache(0.0).GetLocalToWorldTransform(
            omni.usd.get_context().get_stage().GetPrimAtPath(ROBOT))
        print("robot base at",
              tuple(round(v, 3) for v in _m.ExtractTranslation()),
              "| targets given in",
              "BASE frame" if TARGET_IN_BASE_FRAME else "WORLD frame")
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

    def start(self):
        """No-op alias: `GraspFSM()` already starts itself (the physics
        subscription happens in __init__). Kept only so that typing
        `_fsm.start()` after a Run does not raise AttributeError."""
        print("[FSM] already running - constructing the FSM starts it")

    def enter(self, phase, target=None, quat=None):
        print(f"[{phase}]")
        self.phase, self.step, self.stable, self.t = phase, 0, 0, 0.0
        self.target = target
        self.quat = quat
        if target is not None:
            ok = controller.reset(est(), setpoint(target, quat), t=0.0)
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
                if GRASP_SIDE:
                    d, tcp, back = side_targets()
                    q = side_quat(d)
                    print(f"  [side] approach d=({d[0]:.2f},{d[1]:.2f}) "
                          f"tcp={np.round(tcp, 3).tolist()}")
                    self.side_q = q
                    self.enter("hover", tuple(back), q)
                else:
                    self.enter("hover", hover_pos())

        elif self.phase in ("hover", "descend", "lift"):
            if self.step % 240 == 1:
                print(f"  [{self.phase}] still approaching, step {self.step}")
            qt = apply(controller.forward(est(), setpoint(self.target, self.quat), self.t))
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
                elif (self.phase == "descend" and self.step >= DESCEND_STEPS
                        and err < DESCEND_ERR_MAX):
                    print(f"  [descend] in contact (err {err:.3f}) -> closing")
                    self.on_phase_done()
            if self.step > MAX_STEPS:
                self.finish(f"{self.phase} TIMEOUT (err never converged)")

        elif self.phase == "close":
            # HOLD POSITION: sending RMP targets here kept the arm pressing
            # INTO the object while the fingers closed. Freezing the last
            # joint targets lets the gripper close on a stationary arm.
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
            if GRASP_SIDE:
                d, tcp, back = side_targets()
                self.descend_target = tuple(tcp)
                self.enter("descend", tuple(tcp), self.side_q)
                return
            # NO compensation: the finger PRIM origin sits ~10 cm above the
            # real TCP, so prim-based offsets are garbage (they once pushed a
            # target below the floor). IK z is accurate on its own.
            x, y = box_xy()
            print(f"  target box (reconstructed mesh) at x={x:.3f} y={y:.3f}")
            ideal = np.array([x, y, _GEOM["top"] - DESCEND_SINK])
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
                if np.abs(d[:2]).max() > 0.008:
                    # RE-APPROACH DISABLED. In contact, the fingertip reading
                    # measures the object pushing back (once 0.1 m - exactly
                    # the hand-origin-vs-TCP offset), NOT a positioning error.
                    # "Correcting" by it shoved the target INTO the box and
                    # every trial timed out. bbox targeting + the contact exit
                    # above are the source of truth now.
                    print(f"  (residual {np.abs(d[:2]).max():.3f} m - ignored, "
                          "re-approach disabled)")
            else:
                print("  no fingertip reading at descend")
            print("  closing gripper")
            set_grip(False)
            self.enter("close", tuple(ideal),
                       self.side_q if GRASP_SIDE else None)

        elif self.phase == "close":
            if GRASP_SIDE:
                # straight up, keeping the side orientation
                self.enter("lift", tuple(
                    np.array(self.descend_target) + (0.0, 0.0, LIFT_DZ)),
                    self.side_q)
            else:
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


# Run this file and it starts on its own - no follow-up call needed.
# Re-running is safe: the block at line ~63 unsubscribes any stale FSM first.
_fsm = GraspFSM()
