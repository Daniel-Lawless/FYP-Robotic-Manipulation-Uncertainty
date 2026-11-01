import numpy as np
import pytest

pytest.importorskip("robosuite")
from robot_uncertainty.sim.lift import make_lift_env, observe_cube  # noqa: E402
from robot_uncertainty.sim.primitives import (  # noqa: E402
    aligned_grasp_yaw,
    box_yaw,
    heading,
    settle,
    top_down_rotation,
    top_grasp,
    wrap_angle,
)

DEG = np.pi / 180


def grasp_episode(env, offset=(0.0, 0.0, 0.0)):
    """Settle, observe the cube (optionally offset to fake a bad estimate), grasp."""
    settle(env)
    cube_pos, cube_yaw = observe_cube(env)
    return top_grasp(env, cube_pos + np.asarray(offset), cube_yaw, env.cube.root_body)


def set_cube_yaw(env, yaw: float) -> None:
    """Rotate the cube about world z in place (the free joint stores a wxyz quaternion)."""
    joint = env.cube.joints[0]
    qpos = env.sim.data.get_joint_qpos(joint).copy()
    qpos[3:] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    env.sim.data.set_joint_qpos(joint, qpos)
    env.sim.forward()


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize("yaw_deg", [0, 30, 90, -135, 180])
def test_top_down_rotation_is_valid_and_points_down(yaw_deg: float) -> None:
    rot = top_down_rotation(yaw_deg * DEG)
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(rot), 1.0)
    np.testing.assert_allclose(rot[:, 2], [0.0, 0.0, -1.0], atol=1e-12)  # approach down
    assert np.isclose(wrap_angle(heading(rot) - yaw_deg * DEG), 0.0)


def test_box_yaw_handles_any_resting_face() -> None:
    lying_on_side = rot_z(0.3) @ np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    yaw = box_yaw(lying_on_side)
    assert np.isclose(wrap_angle(yaw - 0.3, np.pi / 2), 0.0)


@pytest.mark.parametrize(
    ("object_deg", "current_deg", "expected_deg"),
    [(0, 90, 90), (3, 90, 93), (44, 90, 134), (46, 90, 46), (90, 90, 90), (-30, 90, 60)],
)
def test_aligned_yaw_turns_the_wrist_at_most_45_degrees(object_deg, current_deg, expected_deg):
    yaw = aligned_grasp_yaw(object_deg * DEG, current_deg * DEG)
    assert np.isclose(yaw, expected_deg * DEG)
    assert np.isclose(wrap_angle(yaw - object_deg * DEG, np.pi / 2), 0.0)  # on a face pair


@pytest.mark.parametrize("cube_deg", [0, 2, 44, 45, 46, 90, -30])
def test_grasp_aligns_with_cube_faces(cube_deg: float) -> None:
    env = make_lift_env(seed=0)
    env.reset()
    set_cube_yaw(env, cube_deg * DEG)
    result = grasp_episode(env)
    assert all(p.reached for p in result.phases)
    assert result.success
    assert abs(result.yaw_error) < 2 * DEG
    assert abs(result.wrist_rotation) <= 45 * DEG + 1e-9
    # Closed on two faces, not across a diagonal: the cube is 40-44 mm wide.
    assert result.finger_gap < 0.046


def test_top_grasp_is_verified_on_seed_0() -> None:
    env = make_lift_env(seed=0)
    env.reset()
    result = grasp_episode(env)
    assert all(p.reached for p in result.phases)
    assert result.success
    assert result.lift_height > 0.08


def test_yaw_switch_at_45_degrees_changes_face_pair_but_not_outcome() -> None:
    # Same physical state; only the observed yaw differs, by 0.1 deg either side of the
    # boundary 45 deg from the gripper. See scripts/diagnose_yaw_switch.py.
    results, states = [], []
    for eps in (-0.1 * DEG, 0.1 * DEG):
        env = make_lift_env(seed=0, hard_reset=False)
        env.reset()
        settle(env)
        start_yaw = heading(
            env.sim.data.site_xmat[env.robots[0].eef_site_id["right"]].reshape(3, 3)
        )
        set_cube_yaw(env, start_yaw + 45 * DEG)
        states.append(env.sim.data.qpos.copy())
        cube_pos = env.sim.data.body_xpos[env.cube_body_id].copy()
        results.append(top_grasp(env, cube_pos, start_yaw + 45 * DEG + eps, env.cube.root_body))
    np.testing.assert_array_equal(states[0], states[1])
    below, above = results
    assert np.isclose(below.wrist_rotation, 44.9 * DEG) and np.isclose(
        above.wrist_rotation, -44.9 * DEG
    )
    assert below.face_axis != above.face_axis
    for r in results:
        assert r.success and all(p.reached for p in r.phases)
        assert abs(r.yaw_error) < 0.5 * DEG
    assert abs(below.control_steps - above.control_steps) <= 2
    assert abs(below.lift_height - above.lift_height) < 0.002


def test_missed_grasp_is_not_verified() -> None:
    # Every movement completes, but the gripper closes 8 cm to the side of the cube.
    env = make_lift_env(seed=0)
    env.reset()
    result = grasp_episode(env, offset=(0.0, 0.08, 0.0))
    assert all(p.reached for p in result.phases)
    assert not result.success
    assert abs(result.lift_height) < 0.01


def test_same_seed_gives_identical_grasp() -> None:
    runs = []
    for _ in range(2):
        env = make_lift_env(seed=3)
        env.reset()
        runs.append((grasp_episode(env), env.sim.data.qpos.copy()))
    (first, first_qpos), (second, second_qpos) = runs
    assert first.control_steps == second.control_steps
    np.testing.assert_array_equal(first.object_final, second.object_final)
    np.testing.assert_array_equal(first_qpos, second_qpos)


def test_soft_reset_leaves_no_state_from_previous_episode() -> None:
    dirty, clean = (make_lift_env(seed=1, hard_reset=False) for _ in range(2))
    dirty.reset()
    grasp_episode(dirty)
    clean.reset()
    dirty.reset()
    clean.reset()
    assert dirty.sim.data.time == 0.0 and not dirty.sim.data.qvel.any()
    np.testing.assert_array_equal(dirty.sim.data.qpos, clean.sim.data.qpos)
    first, second = grasp_episode(dirty), grasp_episode(clean)
    assert first.control_steps == second.control_steps
    np.testing.assert_array_equal(dirty.sim.data.qpos, clean.sim.data.qpos)
