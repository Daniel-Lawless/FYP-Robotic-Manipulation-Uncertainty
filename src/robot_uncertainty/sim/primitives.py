from __future__ import annotations

import time
from dataclasses import dataclass, field

import mujoco
import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.environments.base import MujocoEnv

from robot_uncertainty.sim.actions import execution_heading

GRIPPER_OPEN, GRIPPER_CLOSED = -1.0, 1.0

# Grip-site orientation for a top grasp at yaw 0: approach axis (site z) pointing straight
# down and fingers closing along world x. Other yaws rotate this about world z.
TOP_DOWN = np.diag([1.0, -1.0, -1.0])

# Distance from the grip site back along the approach axis to the nearest point of the
# Panda hand's palm collision mesh (31.0 mm, measured from the robosuite 1.5.2 model;
# checked by tests/sim/test_tabletop.py). An object taller than this, gripped at its
# centre, would have its top face under the palm.
PALM_DEPTH = 0.031


def wrap_angle(angle: float, period: float = 2 * np.pi) -> float:
    """Wrap an angle into [-period/2, period/2)."""
    return float((angle + period / 2) % period - period / 2)


def heading(rot: np.ndarray) -> float:
    """Yaw of a frame's x axis projected onto the world xy-plane."""
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


def box_yaw(rot: np.ndarray) -> float:
    """Yaw of a horizontal face normal of a box resting on any of its faces.

    The body axis closest to world z is the vertical one; the next body axis is then
    horizontal and normal to a pair of opposing side faces.
    """
    vertical = int(np.argmax(np.abs(rot[2])))
    side = rot[:, (vertical + 1) % 3]
    return float(np.arctan2(side[1], side[0]))


def top_down_rotation(yaw: float) -> np.ndarray:
    """Top-down grip-site orientation with the fingers closing along heading yaw."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ TOP_DOWN


def aligned_grasp_yaw(object_yaw: float, current_yaw: float, symmetry: float = np.pi / 2) -> float:
    """Gripper yaw that closes on opposing faces with the least wrist rotation.

    Closing along object_yaw + k * symmetry grips the same pair of faces (or the other
    pair, for a square cross-section) for any integer k; pick the k nearest the current
    gripper yaw, so the wrist turns at most symmetry / 2.
    """
    return current_yaw + wrap_angle(object_yaw - current_yaw, symmetry)


def top_grasp_height(object_height: float, palm_clearance: float = 0.006) -> float:
    """Height of a top grasp point above an upright object's centre.

    Grip at the centre unless that would bring the palm within palm_clearance of the
    object's top face; then grip just high enough to keep that gap. Zero for objects up
    to 50 mm tall (the Lift cube), 15 mm for the 80 mm block. The descend phase may stop
    up to its 5 mm tolerance short of the grasp point, which only adds clearance.
    """
    return max(0.0, object_height / 2 - PALM_DEPTH + palm_clearance)


@dataclass
class PhaseResult:
    """One attempted movement; reached=False means it timed out."""

    name: str
    reached: bool
    steps: int
    pos_error: float  # metres, at the end of the phase
    ori_error: float  # radians, at the end of the phase


@dataclass
class GraspResult:
    """Outcome of one top-grasp attempt, measured from the true simulator state."""

    success: bool
    phases: list[PhaseResult] = field(default_factory=list)
    object_start: np.ndarray | None = None
    object_final: np.ndarray | None = None
    eef_final: np.ndarray | None = None
    finger_gap: float = 0.0  # metres between the fingers after lifting
    # Grasp selection (yaws in radians, as headings in the world xy-plane).
    true_yaw: float = 0.0  # object's true face yaw at the start
    observed_yaw: float = 0.0  # face yaw the grasp was planned from
    start_yaw: float = 0.0  # gripper yaw at the start
    target_yaw: float = 0.0  # selected gripper yaw
    face_axis: int = -1  # object body axis (0, 1, 2) the fingers closed along
    wrist_rotation: float = 0.0  # target_yaw - start_yaw
    yaw_error: float = 0.0  # fingers vs true object faces as the grip closes
    final_yaw_error: float = 0.0  # the same after the hold
    control_steps: int = 0
    sim_time: float = 0.0  # simulated seconds
    wall_time: float = 0.0  # seconds
    grasp_height: float = 0.0  # grasp point above the observed object centre, metres
    # True horizontal body axis (0 or 1) most aligned with the closing heading decided
    # before the grasp (the legacy selection when no heading was given), at the start.
    # Privileged evaluation information: not for policy or outcome-model inputs.
    commanded_face_axis: int = -1

    @property
    def lift_height(self) -> float:
        return float(self.object_final[2] - self.object_start[2])


class OSCArm:
    """Closed-loop pose control of one Panda arm through robosuite's OSC_POSE."""

    def __init__(self, env: MujocoEnv) -> None:
        self.env = env
        robot = env.robots[0]
        controller = robot.part_controllers["right"]
        self.model: mujoco.MjModel = env.sim.model._model
        self.data: mujoco.MjData = env.sim.data._data
        self.controller = controller
        self.site_id = self.model.site(controller.ref_name).id
        # Full-scale OSC deltas per control step (m, rad), for an input of 1.
        self.pos_scale = float(controller.output_max[0])
        self.rot_scale = float(controller.output_max[3])
        self.finger_qpos_adr = [
            self.model.joint(name).qposadr[0] for name in robot.gripper["right"].joints
        ]
        self.steps = 0

    def eef_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.site_id].copy()

    def eef_rot(self) -> np.ndarray:
        return self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def finger_gap(self) -> float:
        left, right = self.data.qpos[self.finger_qpos_adr]
        return float(left - right)

    def errors(self, target_pos: np.ndarray, target_rot: np.ndarray) -> tuple[np.ndarray, ...]:
        """Position error and rotation error (axis-angle), both in the world frame."""
        pos_err = target_pos - self.eef_pos()
        # World-frame rotation taking the current orientation to the target.
        rot_err = T.quat2axisangle(T.mat2quat(target_rot @ self.eef_rot().T))
        return pos_err, rot_err

    def step(self, target_pos, target_rot, gripper: float, max_speed: float) -> None:
        pos_err, rot_err = self.errors(target_pos, target_rot)
        pos_delta = np.clip(pos_err, -max_speed, max_speed)
        # Clip the rotation by its magnitude, not per component, so the axis is kept.
        rot_norm = np.linalg.norm(rot_err)
        rot_delta = rot_err * min(1.0, self.rot_scale / rot_norm) if rot_norm > 0 else rot_err
        # OSC_POSE deltas are in the robot base frame (goal = R_delta @ current, there),
        # so rotate both world-frame deltas by the inverse of the base orientation.
        to_base = self.controller.origin_ori.T
        action = np.concatenate(
            [to_base @ pos_delta / self.pos_scale, to_base @ rot_delta / self.rot_scale, [gripper]]
        )
        self.env.step(np.clip(action, -1.0, 1.0))
        self.steps += 1

    def move_to(
        self,
        name: str,
        target_pos: np.ndarray,
        target_rot: np.ndarray = TOP_DOWN,
        gripper: float = GRIPPER_OPEN,
        pos_tol: float = 0.005,
        ori_tol: float = 0.05,
        max_speed: float = 0.05,
        max_steps: int = 100,
    ) -> PhaseResult:
        """Servo the grip site to a pose; max_speed caps the commanded step in metres."""
        for steps in range(max_steps + 1):
            pos_err, rot_err = self.errors(target_pos, target_rot)
            reached = np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(rot_err) < ori_tol
            if reached or steps == max_steps:
                break
            self.step(target_pos, target_rot, gripper, max_speed)
        return self._phase_result(name, bool(reached), steps, target_pos, target_rot)

    def hold(self, name: str, gripper: float, steps: int) -> PhaseResult:
        """Hold the current pose for a fixed number of steps while actuating the gripper."""
        target_pos, target_rot = self.eef_pos(), self.eef_rot()
        for _ in range(steps):
            self.step(target_pos, target_rot, gripper, max_speed=0.05)
        return self._phase_result(name, True, steps, target_pos, target_rot)

    def _phase_result(self, name, reached, steps, target_pos, target_rot) -> PhaseResult:
        pos_err, rot_err = self.errors(target_pos, target_rot)
        return PhaseResult(
            name, reached, steps, float(np.linalg.norm(pos_err)), float(np.linalg.norm(rot_err))
        )


def settle(env: MujocoEnv, steps: int = 5) -> None:
    """Hold the robot still with the gripper open while objects come to rest.

    Lift spawns the cube 1 cm above the table; it is at rest within 5 control steps
    (0.25 s, the settle time in the proposal), so observe objects only after this.
    """
    OSCArm(env).hold("settle", GRIPPER_OPEN, steps)


def top_grasp(
    env: MujocoEnv,
    object_pos: np.ndarray,
    object_yaw: float,
    object_body: str,
    symmetry: float = np.pi / 2,
    pregrasp_height: float = 0.10,
    lift_height: float = 0.10,
    min_lift: float = 0.05,
    grasp_height: float = 0.0,
    closing_heading: float | None = None,
) -> GraspResult:
    """Grasp an object from above at its observed pose and lift it.

    Phases: open the gripper; move above the object while turning the fingers onto a
    pair of opposing faces; descend to the grasp point; close; lift; hold. object_pos
    and object_yaw (a face-normal heading, see box_yaw) are what the robot believes (for
    now the true values); symmetry is the object's rotational symmetry about vertical
    (pi/2 for a square cross-section). The grasp point is grasp_height above object_pos:
    zero, the centre, for short objects; see top_grasp_height for tall ones. Success is
    judged from the true pose of object_body: after the hold, the object has risen at
    least min_lift, is still between the fingers, and the fingers have not closed on
    nothing.

    closing_heading, if given, is a finger-closing heading decided beforehand (an
    actions.GraspAction); the wrist then follows actions.execution_heading on
    object_yaw instead of the least-rotation face pair.
    """
    arm = OSCArm(env)
    body_id = arm.model.body(object_body).id
    start_wall, start_sim = time.perf_counter(), arm.data.time
    object_start = arm.data.xpos[body_id].copy()

    def true_object_rot() -> np.ndarray:
        return arm.data.xmat[body_id].reshape(3, 3)

    true_start_yaw = box_yaw(true_object_rot())
    start_yaw = heading(arm.eef_rot())
    if closing_heading is None:
        grasp_yaw = aligned_grasp_yaw(object_yaw, start_yaw, symmetry)
        commanded_heading = grasp_yaw
    else:
        grasp_yaw = execution_heading(closing_heading, object_yaw, start_yaw)
        commanded_heading = closing_heading
    grasp_rot = top_down_rotation(grasp_yaw)
    commanded = [np.cos(commanded_heading), np.sin(commanded_heading), 0.0]
    commanded_face_axis = int(np.argmax(np.abs(true_object_rot()[:, :2].T @ commanded)))

    # Adding a zero offset is exact, so grasp_height=0 reproduces the Lift results.
    grasp_pos = object_pos + [0.0, 0.0, grasp_height]
    above = grasp_pos + [0.0, 0.0, pregrasp_height]
    phases = [
        arm.hold("open", GRIPPER_OPEN, steps=10),
        arm.move_to("pregrasp", above, grasp_rot, pos_tol=0.01, max_steps=100),
        arm.move_to("descend", grasp_pos, grasp_rot, max_speed=0.02, max_steps=80),
    ]
    # Alignment actually achieved, against the object's true pose, as the grip closes.
    yaw_error = wrap_angle(heading(arm.eef_rot()) - box_yaw(true_object_rot()), symmetry)
    finger_axis = arm.eef_rot()[:, 0]
    face_axis = int(np.argmax(np.abs(true_object_rot().T @ finger_axis)))
    lifted = grasp_pos + [0.0, 0.0, lift_height]
    phases += [
        arm.hold("close", GRIPPER_CLOSED, steps=15),
        arm.move_to("lift", lifted, grasp_rot, gripper=GRIPPER_CLOSED, pos_tol=0.01),
        arm.hold("hold", GRIPPER_CLOSED, steps=10),
    ]
    final_yaw_error = wrap_angle(heading(arm.eef_rot()) - box_yaw(true_object_rot()), symmetry)

    object_final = arm.data.xpos[body_id].copy()
    eef_final = arm.eef_pos()
    finger_gap = arm.finger_gap()
    # "Still between the fingers": the true grasp point stays within 2 cm of the grip site.
    success = (
        object_final[2] - object_start[2] >= min_lift
        and np.linalg.norm(object_final + [0.0, 0.0, grasp_height] - eef_final) < 0.02
        and finger_gap > 0.005
    )
    return GraspResult(
        success=bool(success),
        phases=phases,
        object_start=object_start,
        object_final=object_final,
        eef_final=eef_final,
        finger_gap=finger_gap,
        true_yaw=true_start_yaw,
        observed_yaw=float(object_yaw),
        start_yaw=start_yaw,
        target_yaw=grasp_yaw,
        face_axis=face_axis,
        wrist_rotation=grasp_yaw - start_yaw,
        yaw_error=yaw_error,
        final_yaw_error=final_yaw_error,
        control_steps=arm.steps,
        sim_time=float(arm.data.time - start_sim),
        wall_time=time.perf_counter() - start_wall,
        grasp_height=float(grasp_height),
        commanded_face_axis=commanded_face_axis,
    )
