from __future__ import annotations

import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.base import MujocoEnv

from robot_uncertainty.sim.primitives import box_yaw


def make_lift_env(seed: int, hard_reset: bool = True) -> MujocoEnv:
    """Build a headless Lift environment with a Panda under OSC_POSE control.

    robosuite seeds one generator (env.rng) here, at construction; it drives the cube
    size, cube placement and the robot's initial joint noise on every later reset.
    With hard_reset=True (robosuite's default) each reset rebuilds the model from XML
    and so redraws the cube size; with hard_reset=False the size is fixed at
    construction and only the pose is resampled.
    """
    # The Panda's default composite controller is "BASIC": OSC_POSE on the arm (delta
    # position and axis-angle rotation in the robot base frame) and a one-dimensional
    # open/close gripper command.
    controller_config = load_composite_controller_config(robot="Panda")
    arm_type = controller_config["body_parts"]["right"]["type"]
    assert arm_type == "OSC_POSE", f"expected OSC_POSE arm controller, got {arm_type}"

    return robosuite.make(
        "Lift",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        hard_reset=hard_reset,
        seed=seed,
    )


def observe_cube(env: MujocoEnv) -> tuple[np.ndarray, float]:
    """The robot's estimate of the cube's position and face yaw (exact for now)."""
    data = env.sim.data
    rot = data.body_xmat[env.cube_body_id].reshape(3, 3)
    return data.body_xpos[env.cube_body_id].copy(), box_yaw(rot)
