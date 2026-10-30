"""Tests for the custom tabletop environment (needs MuJoCo).

CI skips tests/sim/; run locally with `pytest tests/sim`.
"""

import dataclasses

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("robosuite")
from robot_uncertainty.sim.contacts import record_grasp  # noqa: E402
from robot_uncertainty.sim.primitives import (  # noqa: E402
    PALM_DEPTH,
    box_yaw,
    heading,
    settle,
    top_grasp_height,
    wrap_angle,
)
from robot_uncertainty.sim.scene import Placement, footprint_corners  # noqa: E402
from robot_uncertainty.sim.tabletop import (  # noqa: E402
    Observation,
    grasp_block_from_above,
    make_tabletop_env,
)

DEG = np.pi / 180


@pytest.fixture(scope="module")
def env():
    env = make_tabletop_env(seed=0)
    yield env
    env.close()


def raw(env):
    return env.sim.model._model, env.sim.data._data


def block_contacts(env, name: str) -> set[str]:
    """Names of the geoms currently touching a block."""
    model, data = raw(env)
    body = env.block_body_ids[name]
    names = set()
    for contact in data.contact[: data.ncon]:
        bodies = model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]
        if body in bodies:
            other = contact.geom2 if bodies[0] == body else contact.geom1
            names.add(model.geom(other).name)
    return names


def test_environment_uses_panda_osc_pose_without_model_rebuilds(env) -> None:
    assert env.robots[0].part_controllers["right"].name == "OSC_POSE"
    assert env.action_dim == 7
    model_before = env.sim.model._model
    env.reset(scene_seed=1)
    assert env.sim.model._model is model_before  # soft reset: no XML recompile


def test_robot_starts_at_exact_home_pose(env) -> None:
    home = np.array(env.robots[0].init_qpos)
    for seed in (0, 1):
        env.reset(scene_seed=seed)
        np.testing.assert_array_equal(env.observe().joint_positions, home)
        assert not env.sim.data.qvel.any()


def test_scene_seed_alone_determines_the_layout(env) -> None:
    env.reset(scene_seed=4)
    first = env.sim.data.qpos.copy()
    settle(env)
    grasp_block_from_above(env, "target")  # dirty the state between the two resets
    env.reset(scene_seed=4)
    np.testing.assert_array_equal(env.sim.data.qpos, first)
    other = make_tabletop_env(seed=99)
    other.reset(scene_seed=4)
    np.testing.assert_array_equal(other.sim.data.qpos, first)
    other.reset(scene_seed=5)
    assert not np.array_equal(other.sim.data.qpos, first)


def test_unseeded_resets_follow_the_env_seed() -> None:
    a, b = make_tabletop_env(seed=3), make_tabletop_env(seed=3)
    for _ in range(3):
        a.reset(), b.reset()
        assert a.scene_seed == b.scene_seed
        np.testing.assert_array_equal(a.sim.data.qpos, b.sim.data.qpos)


@pytest.mark.parametrize("scene_seed", range(10))
def test_block_is_placed_upright_at_rest_inside_the_workspace(env, scene_seed) -> None:
    env.reset(scene_seed=scene_seed)
    cfg, block = env.config, env.config.blocks[0]
    placed = env.layout["target"]
    state = env.true_state().object("target")
    expected = env.table_offset + [placed.x, placed.y, block.half_size[2]]
    np.testing.assert_allclose(state.position, expected, atol=1e-12)
    assert all(cfg.workspace.contains(c) for c in footprint_corners(block, placed))

    settle(env)  # 0.25 s: the block must not move, tip or touch anything but the table
    state = env.true_state().object("target")
    assert np.linalg.norm(state.position - expected) < 5e-4
    rot = env.sim.data.body_xmat[env.block_body_ids["target"]].reshape(3, 3)
    assert rot[2, 2] > np.cos(0.5 * DEG)
    assert np.isclose(wrap_angle(box_yaw(rot) - placed.yaw, block.symmetry), 0.0, atol=1e-3)
    assert np.linalg.norm(state.linear_velocity) < 1e-3
    assert block_contacts(env, "target") == {"table_collision"}


def test_observation_matches_simulator_values(env) -> None:
    env.reset(scene_seed=2)
    settle(env)
    obs = env.observe()
    _, data = raw(env)
    body = env.block_body_ids["target"]
    target = obs.object("target")
    np.testing.assert_array_equal(target.position, data.xpos[body])
    assert target.yaw == box_yaw(data.xmat[body].reshape(3, 3))
    np.testing.assert_array_equal(target.size, env.config.blocks[0].size)

    robot = env.robots[0]
    site = robot.eef_site_id["right"]
    np.testing.assert_array_equal(obs.eef_position, data.site_xpos[site])
    assert obs.eef_yaw == heading(data.site_xmat[site].reshape(3, 3))
    left, right = data.qpos[robot._ref_gripper_joint_pos_indexes["right"]]
    assert obs.gripper_width == left - right
    np.testing.assert_array_equal(obs.joint_positions, data.qpos[robot._ref_joint_pos_indexes])

    goal = env.config.goal
    np.testing.assert_allclose(obs.goal_position[:2], goal.center, atol=1e-12)
    assert np.isclose(obs.goal_position[2], env.config.table_height)
    assert obs.goal_radius == goal.radius


def test_observation_carries_no_outcome_information() -> None:
    fields = {f.name for f in dataclasses.fields(Observation)}
    assert fields == {
        "objects",
        "goal_position",
        "goal_radius",
        "eef_position",
        "eef_yaw",
        "gripper_width",
        "joint_positions",
    }


def test_true_state_matches_raw_mujoco(env) -> None:
    env.reset(scene_seed=6)
    state = env.true_state()
    _, data = raw(env)
    body = env.block_body_ids["target"]
    np.testing.assert_array_equal(state.object("target").quaternion, data.xquat[body])
    assert state.scene_seed == 6 and state.layout == env.layout
    assert state.time == data.time


def test_goal_is_visual_only(env) -> None:
    model, _ = raw(env)
    assert model.site(env.goal_site_id).name == "goal"  # a site: no collision geometry
    assert not any("goal" in model.geom(i).name for i in range(model.ngeom))


def test_palm_depth_matches_the_robot_model(env) -> None:
    env.reset(scene_seed=0)
    model, data = raw(env)
    geom = model.geom("gripper0_right_hand_collision").id
    mesh = model.geom_dataid[geom]
    start = model.mesh_vertadr[mesh]
    verts = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
    world = verts @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    site = env.robots[0].eef_site_id["right"]
    approach = data.site_xmat[site].reshape(3, 3)[:, 2]
    depth = -((world - data.site_xpos[site]) @ approach).max()
    assert abs(depth - PALM_DEPTH) < 5e-4


def test_grasp_height_is_zero_for_the_lift_cube_and_raised_for_the_block() -> None:
    assert top_grasp_height(0.044) == 0.0  # the largest Lift cube
    assert np.isclose(top_grasp_height(0.08), 0.015)


@pytest.mark.parametrize("scene_seed", range(6))
def test_top_grasp_lifts_the_block(env, scene_seed) -> None:
    env.reset(scene_seed=scene_seed)
    settle(env)
    result = grasp_block_from_above(env, "target")
    assert all(p.reached for p in result.phases)
    assert result.success
    assert result.lift_height > 0.08
    assert abs(result.finger_gap - 0.04) < 0.002  # closed on two faces, not a diagonal
    assert abs(result.yaw_error) < 2 * DEG


@pytest.mark.parametrize("x", [-0.07, 0.07])
@pytest.mark.parametrize("y", [-0.17, 0.03])
def test_workspace_corners_are_graspable(env, x, y) -> None:
    # The extreme centres a 4 x 4 cm footprint can take, at the worst-case 45 deg yaw.
    env.reset(scene_seed=0)
    env.set_block_pose("target", Placement(x, y, 45 * DEG))
    settle(env)
    result = grasp_block_from_above(env, "target")
    assert result.success and all(p.reached for p in result.phases)


def palm_touches_block_during_grasp(env, grasp_height) -> bool:
    touched = []
    step = env.step

    def recording_step(action):
        out = step(action)
        touched.append("gripper0_right_hand_collision" in block_contacts(env, "target"))
        return out

    env.step = recording_step
    try:
        result = grasp_block_from_above(env, "target", grasp_height)
    finally:
        del env.step
    assert result.success
    return any(touched)


def test_raised_grasp_point_keeps_the_palm_off_the_block(env) -> None:
    # Gripping the 8 cm block at its centre puts the palm on its top face; it still
    # lifts, but the palm presses the block into the table first.
    env.reset(scene_seed=3)
    settle(env)
    assert palm_touches_block_during_grasp(env, grasp_height=0.0)
    env.reset(scene_seed=3)
    settle(env)
    assert not palm_touches_block_during_grasp(env, grasp_height=None)


@pytest.mark.parametrize(
    "scene_seed, placement",
    [(0, None), (3, None), (7, None), (0, Placement(-0.07, 0.03, 45 * DEG))],
    ids=["seed0", "seed3", "seed7", "corner-45deg"],  # the corner had the least clearance
)
def test_raised_grasp_keeps_palm_clearance_and_grips_with_both_pads(
    env, scene_seed, placement
) -> None:
    # Clearance is MuJoCo's signed palm-block distance, contacts its contact list; see
    # robot_uncertainty.sim.contacts.
    env.reset(scene_seed=scene_seed)
    if placement is not None:
        env.set_block_pose("target", placement)
    settle(env)
    result, samples = record_grasp(env, "target")
    assert result.success and all(p.reached for p in result.phases)

    # The palm never touches the block, and MuJoCo's geometry keeps it at least 5 mm away.
    assert not any(s.palm_contact for s in samples)
    assert min(s.palm_clearance for s in samples) >= 0.005

    # From the end of the close through lift and hold, both pads press on the block,
    # one on each of a pair of opposite faces (half the 40 mm width from the centre).
    steps = {p.name: p.steps for p in result.phases}
    gripping = samples[sum(list(steps.values())[:4]) - 1 :]  # last close step onwards
    assert len(gripping) == 1 + steps["lift"] + steps["hold"]
    for sample in gripping:
        sides = []
        for force, points in zip(sample.pad_forces, sample.pad_points, strict=True):
            assert len(points) > 0, "a pad lost contact"
            assert force > 5.0
            offsets = points[:, result.face_axis]
            np.testing.assert_allclose(np.abs(offsets), 0.02, atol=0.002)
            sides.append(np.sign(offsets.mean()))
        assert sides[0] == -sides[1]


def test_same_scene_seed_gives_identical_grasp(env) -> None:
    # The module env has already run many grasps; a fresh env must match it exactly.
    runs = []
    for current in (env, env, make_tabletop_env(seed=0)):
        current.reset(scene_seed=8)
        settle(current)
        result = grasp_block_from_above(current, "target")
        runs.append((result, current.sim.data.qpos.copy()))
    first, first_qpos = runs[0]
    for result, qpos in runs[1:]:
        assert result.control_steps == first.control_steps
        np.testing.assert_array_equal(result.object_final, first.object_final)
        np.testing.assert_array_equal(qpos, first_qpos)
