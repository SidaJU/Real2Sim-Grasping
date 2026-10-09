"""Side grasp for a flat object on a pillar.

Run in Isaac Sim Script Editor; starts automatically. Stop with _fsm.stop().
The gripper remains closed after the result. Top-down backup: 19_grasp_trial_topdown.py.
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


def box_bounds():
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(BOX_PRIM)
    bbox = UsdGeom.Boundable(prim).ComputeWorldBound(0.0, "default")
    rng = bbox.ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    return (mn[0], mn[1], mn[2]), (mx[0], mx[1], mx[2])


HOVER_H, LIFT_DZ = 0.15, 0.10
GRIP_OPEN, GRIP_CLOSE = 0.04, 0.0
MAX_STEPS, HOLD = 1200, 15
SIDE_INSET = 0.06
SIDE_Z_FRAC = 0.35
APPROACH_STEPS, SETTLE_STEPS = 180, 30
JUDGE_HOLD = 90

art = None
box = None
controller = None
cr = load_cumotion_supported_robot("franka")
tool = cr.robot_description.tool_frame_names()[0]
names = []
grip_idx = []

# ---- kill any previous FSM left over from an earlier run ----
_prev = globals().get("_fsm")
if _prev is not None and getattr(_prev, "sub", None) is not None:
    try:
        _prev.sub.unsubscribe()
        print("unsubscribed stale FSM from a previous run")
    except Exception:
        pass


def est():
    assert art is not None, "Physics initialization must finish before reading joints"
    q = art.get_dof_positions()
    return mg.RobotState(joints=mg.JointState.from_name(
        robot_joint_space=names, positions=(names, q), velocities=(names, art.get_dof_velocities()),
    ))


MODEL_FINGER_OFFSET = 0.025
FINGER_TIP_Z = 0.045
TCP_POSITION_TOL = 0.012
TCP_ANGLE_TOL_DEG = 8.0


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
    q = q_bw.GetInverse() * Gf.Quatd(float(q_w[0]), Gf.Vec3d(*map(float, q_w[1:])))
    im = q.GetImaginary()
    return (q.GetReal(), im[0], im[1], im[2])


def quat_from_axes(x_ax, y_ax, z_ax):
    """Rotation-matrix columns to a (w, x, y, z) quaternion."""
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
    n = np.linalg.norm(d)
    if n < 1e-6:
        raise RuntimeError(
            "side grasp: object centre sits on the robot base xy - no "
            "approach direction. Move the object away from the base.")
    mn, mx = box_bounds()
    z_tcp = mn[2] + SIDE_Z_FRAC * (mx[2] - mn[2])
    rim = np.array([cx, cy, z_tcp])
    if abs(d[0]) >= abs(d[1]):
        d = np.array([np.sign(d[0]), 0.0, 0.0])
        rim[0] = mn[0] if d[0] > 0.0 else mx[0]
    else:
        d = np.array([0.0, np.sign(d[1]), 0.0])
        rim[1] = mn[1] if d[1] > 0.0 else mx[1]
    tcp = rim + d * SIDE_INSET
    back = tcp - d * HOVER_H
    print(f"  [side-v2] rim={np.round(rim, 3).tolist()} "
          f"grasp={np.round(tcp, 3).tolist()} "
          f"pregrasp={np.round(back, 3).tolist()}")
    return d, tcp, back


def setpoint(pos, quat):
    """Convert the jaw-centre pose to the model's left-fingertip pose."""
    hand_quat = Gf.Quatd(float(quat[0]), Gf.Vec3d(*map(float, quat[1:])))
    pos = np.asarray(pos, dtype=np.float64) + np.asarray(
        hand_quat.Transform(Gf.Vec3d(0.0, MODEL_FINGER_OFFSET, 0.0)))
    tip_quat = hand_quat * Gf.Quatd(0.0, Gf.Vec3d(0.0, 0.0, 1.0))
    im = tip_quat.GetImaginary()
    quat = (tip_quat.GetReal(), im[0], im[1], im[2])
    pos = world_to_base(pos)
    quat = world_quat_to_base(quat)
    p = wp.array([list(pos)], dtype=wp.float32)
    o = wp.array([list(quat)], dtype=wp.float32)
    return mg.RobotState(sites=mg.SpatialState.from_name(
        spatial_space=[tool], positions=([tool], p), orientations=([tool], o),
    ))


_GRIP_STATE = {"target": GRIP_OPEN, "hold_q": None}


def apply(desired):
    assert art is not None, "Physics initialization must finish before commanding joints"
    if desired is None or desired.joints.positions is None:
        return None
    full = np.asarray(art.get_dof_positions().numpy(), dtype=np.float64).reshape(-1).copy()
    q = np.asarray(desired.joints.positions.numpy(), dtype=np.float64).reshape(-1)
    if q.size == len(full):
        full[:] = q
    else:
        full[: len(q)] = q
    full[grip_idx] = _GRIP_STATE["target"]
    art.set_dof_position_targets(full)
    return full


def set_grip(open_):
    assert art is not None, "Physics initialization must finish before commanding fingers"
    _GRIP_STATE["target"] = GRIP_OPEN if open_ else GRIP_CLOSE
    full = np.asarray(art.get_dof_positions().numpy(), dtype=np.float64).reshape(-1).copy()
    full[grip_idx] = _GRIP_STATE["target"]
    if not open_:
        _GRIP_STATE["hold_q"] = full.copy()
    art.set_dof_position_targets(full)


GRIP_STIFFNESS_MULT = 20.0
GRIP_DAMPING_MULT = 10.0
GRIP_EFFORT_MULT = 5.0


def boost_gripper():
    """Raise the finger drive gains so the grasp does not slip out."""
    assert art is not None, "Physics initialization must finish before setting gains"
    def flat(x):
        return np.asarray(x.numpy() if hasattr(x, "numpy") else x,
                          dtype=np.float64).ravel()
    try:
        kp, kd = art.get_dof_gains(dof_indices=grip_idx)
        eff = art.get_dof_max_efforts(dof_indices=grip_idx)
        print("gripper before: kp", flat(kp), "kd", flat(kd),
              "max_effort", flat(eff))
        art.set_dof_gains(
            stiffnesses=flat(kp) * GRIP_STIFFNESS_MULT,
            dampings=flat(kd) * GRIP_DAMPING_MULT,
            dof_indices=grip_idx,
        )
        art.set_dof_max_efforts(flat(eff) * GRIP_EFFORT_MULT,
                                dof_indices=grip_idx)
        kp2, kd2 = art.get_dof_gains(dof_indices=grip_idx)
        eff2 = art.get_dof_max_efforts(dof_indices=grip_idx)
        print(f"gripper after : kp {flat(kp2)} kd {flat(kd2)} "
              f"max_effort {flat(eff2)}  (x{GRIP_STIFFNESS_MULT}/"
              f"x{GRIP_DAMPING_MULT}/x{GRIP_EFFORT_MULT})")
    except Exception as e:
        print("[warn] could not boost gripper gains:", e)


_GEOM = {}


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


_JAW_LINKS = {}


def gripper_target_error(target, quat):
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.XformCache(0.0)
    matrices = []
    for name in ("panda_leftfinger", "panda_rightfinger"):
        path = _JAW_LINKS.get(name)
        if path is None:
            prim = stage.GetPrimAtPath(f"{ROBOT}/{name}")
            if not prim.IsValid():
                candidates = [p for p in stage.Traverse()
                              if str(p.GetPath()).startswith(ROBOT + "/")
                              and p.GetName() == name and p.GetTypeName() == "Xform"]
                if not candidates:
                    raise RuntimeError(f"Cannot locate finger link {name}; grip not attempted")
                prim = candidates[0]
            path = prim.GetPath()
            _JAW_LINKS[name] = path
        matrices.append(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)))
    tips = [np.asarray(m.Transform(Gf.Vec3d(0.0, 0.0, FINGER_TIP_Z))) for m in matrices]
    centre = (tips[0] + tips[1]) / 2.0
    expected = Gf.Quatd(float(quat[0]), Gf.Vec3d(*map(float, quat[1:])))
    angle = 0.0
    for axis in (Gf.Vec3d(0.0, 1.0, 0.0), Gf.Vec3d(0.0, 0.0, 1.0)):
        actual_axis = np.asarray(matrices[0].TransformDir(axis))
        actual_axis /= np.linalg.norm(actual_axis)
        desired_axis = np.asarray(expected.Transform(axis))
        angle = max(angle, math.degrees(math.acos(float(np.clip(
            np.dot(actual_axis, desired_axis), -1.0, 1.0)))))
    delta = centre - np.asarray(target)
    return centre, delta, float(np.linalg.norm(delta)), angle


class GraspFSM:
    """Self-terminating grasp state machine driven by physics step events."""

    def __init__(self):
        self.sub = None
        self.phase = "initialize"
        self.step = 0
        self.stable = 0
        self.t = 0.0
        self.target = None
        self.quat = None
        self.side_q = None
        self.approach_from = None
        self.obj_center0 = None
        self.descend_target = None
        tl = omni.timeline.get_timeline_interface()
        tl.stop()
        tl.play()
        self.box_z0 = None
        self.sub = omni.physx.get_physx_interface().subscribe_physics_step_events(self.on_step)
        print("[FSM] waiting for physics initialization")

    def stop(self):
        if self.sub is not None:
            try:
                self.sub.unsubscribe()
                print("[FSM] unsubscribed - callback fully removed")
            except Exception as e:
                print("[FSM] unsubscribe error:", e)
            self.sub = None

    def enter(self, phase, target=None, quat=None):
        print(f"[{phase}]")
        self.phase, self.step, self.stable, self.t = phase, 0, 0, 0.0
        self.target = target
        self.quat = quat
        if target is not None and phase not in ("close", "judge"):
            assert controller is not None, "Controller initialization must finish before motion"
            reset_target = self.approach_from if phase == "descend" else target
            ok = controller.reset(est(), setpoint(reset_target, quat), t=0.0)
            print(f"  reset ok={ok}")

    def on_step(self, dt):
        try:
            self._tick(dt)
        except Exception:
            import traceback
            traceback.print_exc()
            self.finish("CRASHED - unsubscribed automatically")

    def _tick(self, dt):
        global art, box, controller, names, grip_idx
        self.t += dt
        self.step += 1

        if self.phase == "initialize":
            if self.step < 2:
                return
            if art is None:
                art = Articulation(ROBOT)
            if not art.is_physics_tensor_entity_valid():
                if self.step >= 120:
                    self.finish("physics initialization TIMEOUT; no joint commands sent")
                return
            art.get_dof_positions()
            box = GeomPrim(BOX_PRIM)
            names = list(art.dof_names)
            grip_idx = [i for i, n in enumerate(names) if "finger" in n]
            controller = RmpFlowController(
                cumotion_robot=cr,
                cumotion_world_interface=CumotionWorldInterface(),
                robot_joint_space=names,
                robot_site_space=[tool],
                tool_frame=tool,
            )
            refresh_geom()
            boost_gripper()
            self.box_z0 = box_z()
            print("[FSM] physics ready | controller tool:", tool)
            self.enter("open")
            return

        if self.phase == "open":
            if self.step == 1:
                set_grip(True)
            if self.step >= 30:
                d, tcp, back = side_targets()
                self.side_q = side_quat(d)
                self.enter("hover", tuple(back), self.side_q)

        elif self.phase in ("hover", "descend", "lift"):
            assert art is not None and controller is not None, "Physics initialization incomplete"
            tgt = self.target
            if self.phase == "descend" and self.approach_from is not None:
                a = min(1.0, self.step / float(APPROACH_STEPS))
                tgt = tuple(np.asarray(self.approach_from) + a * (
                    np.asarray(self.target) - np.asarray(self.approach_from)))
            qt = apply(controller.forward(est(), setpoint(tgt, self.quat), self.t))
            if qt is None:
                if self.step % 120 == 0:
                    print(f"  [{self.phase}] forward returned None (step {self.step})")
            else:
                at_final_target = self.phase != "descend" or (
                    self.step >= APPROACH_STEPS + SETTLE_STEPS
                )
                centre, delta, distance, angle = gripper_target_error(tgt, self.quat)
                pose_ready = distance < TCP_POSITION_TOL and angle < TCP_ANGLE_TOL_DEG
                self.stable = self.stable + 1 if at_final_target and pose_ready else 0

                if (self.phase == "descend"
                        and self.obj_center0 is not None and self.step % 5 == 0):
                    mn, mx = box_bounds()
                    c_now = np.array([(mn[0] + mx[0]) / 2,
                                      (mn[1] + mx[1]) / 2])
                    disp = float(np.linalg.norm(c_now - self.obj_center0))
                    if disp > 0.005:
                        self.finish(f"object displaced {disp*1000:.0f} mm; approach stopped")
                        return

                if self.step % 60 == 0:
                    print(f"  [{self.phase}] step {self.step} "
                          f"delta_mm={np.round(delta*1000, 1).tolist()} "
                          f"angle_deg={angle:.1f} ready={pose_ready}")
                if at_final_target and self.stable >= HOLD:
                    self.on_phase_done()
                    return
            if self.step > MAX_STEPS:
                self.finish(f"{self.phase} TIMEOUT (err never converged)")

        elif self.phase == "close":
            assert art is not None, "Physics initialization incomplete"
            art.set_dof_position_targets(_GRIP_STATE["hold_q"])
            if self.step >= 45:
                self.on_phase_done()

        elif self.phase == "judge":
            if self.step >= JUDGE_HOLD:
                z = box_z()
                rose = z - self.box_z0
                ok = rose > 0.05
                print(f"\nbox z: {z:.4f} (baseline {self.box_z0:.4f}, rose {rose:+.4f} m)")
                print("GRASP RESULT:", "SUCCESS" if ok else "FAILURE")
                self.finish("trial complete")

    def on_phase_done(self):
        if self.phase == "hover":
            d, tcp, back = side_targets()
            self.descend_target = tuple(tcp)
            self.approach_from = back
            mn, mx = box_bounds()
            self.obj_center0 = np.array(
                [(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2])
            self.enter("descend", tuple(tcp), self.side_q)

        elif self.phase == "descend":
            ideal = np.array(self.descend_target)
            centre, delta, distance, angle = gripper_target_error(ideal, self.side_q)
            if distance >= TCP_POSITION_TOL or angle >= TCP_ANGLE_TOL_DEG:
                self.finish(f"jaw not aligned: {distance*1000:.1f} mm, {angle:.1f} deg")
                return
            print(f"  jaw aligned: centre={np.round(centre, 3).tolist()} "
                  f"delta_mm={np.round(delta*1000, 1).tolist()}")
            set_grip(False)
            self.enter("close", tuple(ideal), self.side_q)

        elif self.phase == "close":
            self.enter("lift", tuple(
                np.array(self.descend_target) + (0.0, 0.0, LIFT_DZ)),
                self.side_q)

        elif self.phase == "lift":
            self.enter("judge", None)

    def finish(self, msg):
        print(f"[FSM] {msg}")
        self.stop()


_fsm = GraspFSM()
