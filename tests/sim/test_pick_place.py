import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("robosuite")
import robot_uncertainty.sim.pick_place as pick_place  # noqa: E402
from robot_uncertainty.sim.pick_place import (  # noqa: E402
    FINGERTIP_DEPTH,
    PlacementTolerances,
    assess_placement,
    is_unintended,
)
from robot_uncertainty.sim.primitives import GRIPPER_CLOSED, GRIPPER_OPEN, settle  # noqa: E402
from robot_uncertainty.sim.scene import Placement  # noqa: E402
from robot_uncertainty.sim.tabletop import grasp_block_from_above, make_tabletop_env  # noqa: E402

DEG = np.pi / 180
TARGET = "target"
TOL = PlacementTolerances()


@pytest.fixture(scope="module")
def env():
    env = make_tabletop_env(seed=0)
    yield env
    env.close()


def start(env, scene_seed: int, placement: Placement | None = None) -> None:
    env.reset(scene_seed=scene_seed)
    if placement is not None:
        env.set_block_pose(TARGET, placement)
    settle(env)


def set_block_state(env, position, quaternion, velocity=None) -> None:
    joint = env.blocks[TARGET].joints[0]
    env.sim.data.set_joint_qpos(joint, np.concatenate([position, quaternion]))
    env.sim.data.set_joint_qvel(joint, np.zeros(6) if velocity is None else velocity)
    env.sim.forward()


def tilted(angle: float) -> np.ndarray:
    """Quaternion (w, x, y, z) for a tilt about world x."""
    return np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])


def test_fingertip_depth_matches_the_robot_model(env) -> None:
    env.reset(scene_seed=0)
    model, data = env.sim.model._model, env.sim.data._data
    site = env.robots[0].eef_site_id["right"]
    approach = data.site_xmat[site].reshape(3, 3)[:, 2]
    reach = []
    for name in ("gripper0_right_finger1_collision", "gripper0_right_finger2_collision"):
        geom = model.geom(name).id
        mesh = model.geom_dataid[geom]
        start_vert = model.mesh_vertadr[mesh]
        verts = model.mesh_vert[start_vert : start_vert + model.mesh_vertnum[mesh]]
        world = verts @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        reach.append(((world - data.site_xpos[site]) @ approach).max())
    assert abs(max(reach) - FINGERTIP_DEPTH) < 5e-4


@pytest.mark.parametrize(
    "scene_seed, placement",
    [(0, None), (3, None), (0, Placement(-0.07, -0.17, 45 * DEG))],
    ids=["seed0", "seed3", "far-corner-45deg"],  # the corner is furthest from the goal
)
def test_pick_and_place_puts_the_block_upright_in_the_goal(env, scene_seed, placement) -> None:
    start(env, scene_seed, placement)
    result = pick_place.pick_and_place(env, TARGET)
    assert result.success and result.failure_reason == ""
    assert result.grasp_success and result.released and not result.dropped
    names = [p.name for p in result.phases]
    assert names == [
        *("open", "pregrasp", "descend", "close", "lift", "hold"),
        *("carry", "lower", "release", "retract", "settle"),
    ]
    assert not result.timeouts
    assert result.unintended_contacts == ()
    p = result.placement
    assert p.in_goal and p.upright and p.on_table and p.at_rest and p.stable
    assert p.contacts == ("table",)
    assert p.centre_error < 0.005  # aimed at the centre; 21.7 mm would still fit
    assert result.control_steps == sum(ph.steps for ph in result.phases)
    assert np.isclose(result.sim_time, result.control_steps * env.control_timestep)


def test_placement_is_released_retracted_and_settled(env) -> None:
    start(env, 5)
    poses = []
    step = env.step

    def recording_step(action):
        out = step(action)
        poses.append(env.true_state().object(TARGET).position)
        return out

    env.step = recording_step
    try:
        result = pick_place.pick_and_place(env, TARGET)
    finally:
        del env.step
    assert result.success
    steps = {p.name: p.steps for p in result.phases}
    released_at = sum(p.steps for p in result.phases[: names_index(result, "release") + 1])
    # The block landed during the release and did not move during retract and settle.
    assert poses[released_at - 1][2] < env.config.table_height + 0.041
    drift = np.linalg.norm(np.array(poses[released_at:]) - poses[released_at - 1], axis=1)
    assert len(drift) == steps["retract"] + steps["settle"] and drift.max() < 1e-4
    # The fingers ended open and above the block's top.
    obs = env.observe()
    assert obs.gripper_width > 0.06
    top = env.config.table_height + env.config.blocks[0].size[2]
    assert obs.eef_position[2] - FINGERTIP_DEPTH > top + 0.04
    p = result.placement
    assert p.linear_speed < TOL.max_linear_speed and p.angular_speed < TOL.max_angular_speed
    assert abs(p.support_gap) < TOL.max_support_gap and p.tilt < TOL.max_tilt


def names_index(result, name: str) -> int:
    return [p.name for p in result.phases].index(name)


@pytest.mark.parametrize(
    "dx, yaw_deg, inside",
    [
        (0.0, 30.0, True),
        (0.021, 45.0, True),  # outermost corner 49.3 mm from the goal centre
        (0.023, 45.0, False),  # 51.3 mm
        (0.025, 0.0, True),  # 49.2 mm: beyond the 21.7 mm any-yaw bound, yet inside
        (0.027, 0.0, False),  # 51.1 mm
    ],
)
def test_goal_containment_uses_the_actual_footprint(env, dx, yaw_deg, inside) -> None:
    env.reset(scene_seed=0)
    goal = env.config.goal
    env.set_block_pose(TARGET, Placement(goal.center[0] + dx, goal.center[1], yaw_deg * DEG))
    check = assess_placement(env, TARGET)
    yaw = yaw_deg * DEG
    corners = [
        np.array([dx, 0.0])
        + 0.02
        * np.array([sx * np.cos(yaw) - sy * np.sin(yaw), sx * np.sin(yaw) + sy * np.cos(yaw)])
        for sx in (-1, 1)
        for sy in (-1, 1)
    ]
    expected_margin = goal.radius - max(np.hypot(*c) for c in corners)
    assert check.in_goal == inside
    assert abs(check.goal_margin - expected_margin) < 1e-6


def test_stability_tolerances(env) -> None:
    env.reset(scene_seed=0)
    # A resting block sinks 0.11 mm into the table (MuJoCo's soft contact); exactly at
    # the surface, MuJoCo reports no contact.
    sink = 1e-4
    goal = np.array([*env.config.goal.center, env.config.table_height + 0.04 - sink])
    upright = tilted(0.0)
    set_block_state(env, goal, upright)
    check = assess_placement(env, TARGET)
    assert check.in_goal and check.upright and check.on_table and check.at_rest

    # Tilt: judged against max_tilt (2 deg) from the pose alone.
    assert assess_placement(env, TARGET).upright
    set_block_state(env, goal, tilted(1.5 * DEG))
    assert assess_placement(env, TARGET).upright
    set_block_state(env, goal, tilted(2.5 * DEG))
    assert not assess_placement(env, TARGET).upright
    # Lying on its side: on the table and inside the goal, but not upright.
    set_block_state(env, goal - [0, 0, 0.02], tilted(90 * DEG))
    check = assess_placement(env, TARGET)
    assert not check.upright and check.on_table and np.isclose(check.support_gap, -sink)

    # Support: 2 mm above the table is neither touching nor resting on it.
    set_block_state(env, goal + [0, 0, 0.002 + sink], upright)
    check = assess_placement(env, TARGET)
    assert not check.on_table and "table" not in check.contacts

    # Rest: 1 cm/s or 0.1 rad/s is moving; zero is at rest.
    set_block_state(env, goal, upright, [0.01, 0, 0, 0, 0, 0])
    assert not assess_placement(env, TARGET).at_rest
    set_block_state(env, goal, upright, [0, 0, 0, 0, 0, 0.1])
    assert not assess_placement(env, TARGET).at_rest
    set_block_state(env, goal, upright)
    settle(env)
    assert assess_placement(env, TARGET).stable


def test_placing_outside_the_goal_fails(env) -> None:
    start(env, 1)
    outside = env.goal_position[:2] + [0.04, 0.0]
    result = pick_place.pick_and_place(env, TARGET, place_at=outside)
    assert not result.success
    assert result.failure_reason.startswith("outside the goal by")
    p = result.placement
    assert not p.in_goal and p.goal_margin < 0
    # Everything else went right: released, standing upright and at rest.
    assert result.grasp_success and result.released and p.stable and not result.timeouts
    assert abs(p.centre_error - 0.04) < 0.005


def test_a_dropped_block_fails(env, monkeypatch) -> None:
    # The gripper opens instead of staying closed once the carry starts.
    monkeypatch.setattr(pick_place, "GRIPPER_CLOSED", GRIPPER_OPEN)
    start(env, 2)
    result = pick_place.pick_and_place(env, TARGET)
    assert result.grasp_success  # the grasp itself (its own module) still held the block
    assert result.dropped and not result.success
    assert result.failure_reason == "dropped during carry"
    assert not result.placement.in_goal


def test_a_block_that_is_not_released_fails(env, monkeypatch) -> None:
    # The release keeps the gripper closed, so the retract lifts the block again.
    monkeypatch.setattr(pick_place, "GRIPPER_OPEN", GRIPPER_CLOSED)
    start(env, 2)
    result = pick_place.pick_and_place(env, TARGET)
    assert not result.released and not result.success
    assert result.failure_reason.startswith("not released")
    p = result.placement
    assert p.touching_robot and not p.on_table


def test_unintended_contacts_are_flagged_separately_from_success(env, monkeypatch) -> None:
    # Gripping the block at its centre puts the palm on its top face (M1.1): the block is
    # still placed, so the task succeeds, but the palm contact is reported.
    centre_grasp = grasp_block_from_above

    def grasp_at_centre(env, name, lift_height):
        return centre_grasp(env, name, grasp_height=0.0, lift_height=lift_height)

    monkeypatch.setattr(pick_place, "grasp_block_from_above", grasp_at_centre)
    start(env, 3)
    result = pick_place.pick_and_place(env, TARGET)
    assert result.success and result.collided
    pairs = {e.pair for e in result.unintended_contacts}
    assert pairs == {"palm-target"}
    phases = {e.phase for e in result.unintended_contacts}
    assert {"descend", "carry", "lower"} <= phases


def test_contact_classification() -> None:
    assert is_unintended(("palm", "target"), "carry", TARGET)
    assert is_unintended(("arm", "table"), "lower", TARGET)
    assert is_unintended(("fingers", "table"), "lower", TARGET)
    assert is_unintended(("floor", "target"), "settle", TARGET)
    assert not is_unintended(("fingers", "target"), "carry", TARGET)
    assert is_unintended(("fingers", "target"), "descend", TARGET)  # brushed on approach
    assert is_unintended(("fingers", "target"), "retract", TARGET)  # disturbed after release
    assert not is_unintended(("table", "target"), "settle", TARGET)
    assert is_unintended(("table", "target"), "carry", TARGET)
    assert not is_unintended(("fingers", "fingers"), "close", TARGET)


def test_same_scene_seed_gives_identical_pick_and_place(env) -> None:
    runs = []
    for current in (env, env, make_tabletop_env(seed=0)):
        start(current, 7)
        result = pick_place.pick_and_place(current, TARGET)
        runs.append((result, current.sim.data.qpos.copy()))
    first, first_qpos = runs[0]
    for result, qpos in runs[1:]:
        assert result.control_steps == first.control_steps
        assert [p.steps for p in result.phases] == [p.steps for p in first.phases]
        np.testing.assert_array_equal(result.placement.position, first.placement.position)
        np.testing.assert_array_equal(result.placement.quaternion, first.placement.quaternion)
        np.testing.assert_array_equal(qpos, first_qpos)
        assert result.max_slip == first.max_slip


def test_default_lift_keeps_the_m1_1_grasp_unchanged(env) -> None:
    # grasp_block_from_above's new lift_height argument defaults to the old 10 cm lift.
    results = []
    for lift in ({}, {"lift_height": 0.10}):
        start(env, 4)
        result = grasp_block_from_above(env, TARGET, **lift)
        results.append((result, env.sim.data.qpos.copy()))
    (a, qa), (b, qb) = results
    assert a.success and a.control_steps == b.control_steps
    np.testing.assert_array_equal(qa, qb)
