"""Smoke tests for the stock robosuite Lift task with a Panda (needs MuJoCo).

CI skips tests/sim/; run locally with `pytest tests/sim`.
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("robosuite")
from robot_uncertainty.sim.lift import make_lift_env  # noqa: E402


@pytest.fixture(scope="module")
def env():
    env = make_lift_env(seed=0)
    yield env
    env.close()


def test_panda_uses_osc_pose_with_7d_action(env) -> None:
    assert env.robots[0].part_controllers["right"].name == "OSC_POSE"
    assert env.action_dim == 7  # 6 pose deltas + 1 gripper


def test_raw_mujoco_objects_match_observations(env) -> None:
    obs = env.reset()
    model, data = env.sim.model._model, env.sim.data._data
    assert isinstance(model, mujoco.MjModel)
    assert isinstance(data, mujoco.MjData)
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, env.cube.root_body)
    np.testing.assert_array_equal(obs["cube_pos"], data.xpos[cube_id])


def test_upward_command_moves_end_effector_up(env) -> None:
    obs = env.reset()
    eef_start = obs["robot0_eef_pos"].copy()
    action = np.zeros(env.action_dim)
    action[2] = 0.2
    for _ in range(10):
        obs, _, _, _ = env.step(action)
    assert np.all(np.isfinite(obs["robot0_eef_pos"]))
    assert obs["robot0_eef_pos"][2] - eef_start[2] > 0.01


def test_seed_controls_initial_cube_position() -> None:
    first, repeat, other = (make_lift_env(s).reset()["cube_pos"] for s in (0, 0, 1))
    np.testing.assert_array_equal(first, repeat)
    assert not np.array_equal(first, other)
