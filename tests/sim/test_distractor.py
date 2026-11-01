from dataclasses import fields

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("robosuite")
from robot_uncertainty.sim import distractor_trials as trials  # noqa: E402
from robot_uncertainty.sim import pick_place  # noqa: E402
from robot_uncertainty.sim.outcome import (  # noqa: E402
    OutcomeThresholds,
    object_outcome,
    quat_to_matrix,
)
from robot_uncertainty.sim.pick_place import (  # noqa: E402
    StepMonitor,
    object_record,
    pick_and_place,
    table_surface,
)
from robot_uncertainty.sim.primitives import settle  # noqa: E402
from robot_uncertainty.sim.scene import (  # noqa: E402
    DISTRACTOR_TALL,
    BlockSpec,
    Placement,
    TabletopConfig,
    place_beside,
)
from robot_uncertainty.sim.tabletop import ObjectObservation, make_tabletop_env  # noqa: E402

TARGET, DISTRACTOR = "target", "distractor"
CONFIG = TabletopConfig(blocks=(BlockSpec(TARGET), DISTRACTOR_TALL))
THRESHOLDS = OutcomeThresholds()


@pytest.fixture(scope="module")
def env():
    env = make_tabletop_env(CONFIG, seed=0)
    yield env
    env.close()


def valid_scene(env, layout) -> None:
    env.reset(scene_seed=0)
    env.set_layout(layout)
    settle(env)
    assert env.check_initial_state() == []


def set_free_body(env, name, position, quaternion) -> None:
    joint = env.blocks[name].joints[0]
    env.sim.data.set_joint_qpos(joint, np.concatenate([position, quaternion]))
    env.sim.data.set_joint_qvel(joint, np.zeros(6))
    env.sim.forward()


# --- observations and layouts ------------------------------------------------------


def test_object_observation_fields_are_pinned() -> None:
    names = [f.name for f in fields(ObjectObservation)]
    assert names == ["name", "position", "yaw", "size", "role"]


def test_observation_reports_each_blocks_role(env) -> None:
    env.reset(scene_seed=0)
    roles = {o.name: o.role for o in env.observe().objects}
    assert roles == {TARGET: "target", DISTRACTOR: "distractor"}


@pytest.mark.parametrize(
    "layout, problem",
    [
        ({TARGET: Placement(0, -0.1, 0), DISTRACTOR: Placement(0.04, -0.1, 0)}, "overlaps"),
        (
            {TARGET: Placement(0, -0.1, 0), DISTRACTOR: Placement(0.29, 0.0, 0.3)},
            "beyond the table",
        ),
        (
            {TARGET: Placement(0, -0.1, 0), DISTRACTOR: Placement(0.0, 0.11, 0)},
            "intersects the goal",
        ),
        ({"ghost": Placement(0, 0, 0)}, "no such blocks"),
    ],
)
def test_invalid_layouts_are_rejected_and_change_nothing(env, layout, problem) -> None:
    env.reset(scene_seed=0)
    before = env.sim.data.qpos.copy()
    with pytest.raises(ValueError, match=problem):
        env.set_layout(layout)
    assert np.array_equal(env.sim.data.qpos, before)


def test_touching_layout_is_valid_and_every_block_starts_upright(env) -> None:
    target = Placement(0.0, -0.1, 0.3)
    beside = place_beside(target, CONFIG.blocks[0], DISTRACTOR_TALL, 0.0, 0.3, 0.3)
    layout = {TARGET: target, DISTRACTOR: beside}
    valid_scene(env, layout)
    table = table_surface(env)
    for name in (TARGET, DISTRACTOR):
        outcome = object_outcome(object_record(env, name), object_record(env, name), table=table)
        assert outcome.resting_face_start == "-z"
    assert env.layout == layout


def test_initial_state_check_reports_motion_and_tilt(env) -> None:
    env.reset(scene_seed=0)
    env.set_layout({TARGET: Placement(0, -0.1, 0), DISTRACTOR: Placement(0.1, -0.1, 0)})
    settle(env)
    assert env.check_initial_state() == []
    joint = env.blocks[DISTRACTOR].joints[0]
    env.sim.data.set_joint_qvel(joint, [0.02, 0, 0, 0, 0, 0])
    env.sim.forward()
    assert any("moving" in p for p in env.check_initial_state())
    pos = env.sim.data.get_joint_qpos(joint)[:3]
    set_free_body(env, DISTRACTOR, pos + [0, 0, 0.01], [np.cos(0.05), np.sin(0.05), 0, 0])
    assert any("tilted" in p for p in env.check_initial_state())


# --- invalid starts are rejected before anything moves ----------------------------------

SETTLED = {TARGET: Placement(0, -0.1, 0), DISTRACTOR: Placement(0.1, -0.1, 0)}


def sim_state(env) -> tuple:
    data = env.sim.data
    return data.time, data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()


def nothing_may_move(env, monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("pick_and_place moved before validating the start")

    monkeypatch.setattr(env, "step", forbidden)
    monkeypatch.setattr(pick_place, "grasp_block_from_above", forbidden)


@pytest.mark.parametrize("name", [TARGET, DISTRACTOR])
def test_moving_block_is_rejected_before_any_step(env, monkeypatch, name) -> None:
    valid_scene(env, SETTLED)
    env.sim.data.set_joint_qvel(env.blocks[name].joints[0], [0.006, 0, 0, 0, 0, 0])
    env.sim.forward()
    before = sim_state(env)
    nothing_may_move(env, monkeypatch)
    with pytest.raises(ValueError, match=f"{name} moving at 0.0060 m/s"):
        pick_and_place(env, TARGET)
    for a, b in zip(before, sim_state(env), strict=True):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("name", [TARGET, DISTRACTOR])
def test_stationary_unsupported_block_is_rejected_before_any_step(env, monkeypatch, name):
    valid_scene(env, SETTLED)
    pos = env.sim.data.get_joint_qpos(env.blocks[name].joints[0])[:3]
    set_free_body(env, name, pos + [0, 0, 0.02], [1.0, 0, 0, 0])
    # At rest, upright and apart: only the resting-face check can reject it.
    assert env.check_initial_state() == []
    before = sim_state(env)
    nothing_may_move(env, monkeypatch)
    with pytest.raises(ValueError, match=f"{name}: initial resting face is unresolved"):
        pick_and_place(env, TARGET)
    for a, b in zip(before, sim_state(env), strict=True):
        assert np.array_equal(a, b)


def test_touching_layout_still_runs(env) -> None:
    target = Placement(0.0, -0.1, 0.3)
    beside = place_beside(target, CONFIG.blocks[0], DISTRACTOR_TALL, 0.0, 0.3, 0.3)
    valid_scene(env, {TARGET: target, DISTRACTOR: beside})
    result = pick_and_place(env, TARGET)
    assert [o.resting_face_start for o in result.objects] == ["-z", "-z"]
    assert result.label is not None


# --- per-object tracking (Test 7) ----------------------------------------------------


def test_every_block_is_tracked_on_every_step(env) -> None:
    layout = {TARGET: Placement(0.0, -0.12, 0.4), DISTRACTOR: Placement(0.2, -0.2, 0.1)}
    valid_scene(env, layout)
    with StepMonitor(env, TARGET) as outer:
        result = pick_and_place(env, TARGET)
    assert result.success and result.label.y == 1
    state = env.true_state()
    outcomes = {o.name: o for o in result.objects}
    assert list(outcomes) == [TARGET, DISTRACTOR]
    for name in (TARGET, DISTRACTOR):
        poses = outer.poses[name]
        assert len(poses) == result.control_steps
        assert np.array_equal(poses[-1][0], state.object(name).position)
        assert np.array_equal(outcomes[name].final_position, state.object(name).position)
        start = np.array([layout[name].x, layout[name].y])
        assert np.allclose(outcomes[name].start_position[:2], start, atol=1e-4)
        planar = [np.linalg.norm(p[:2] - outcomes[name].start_position[:2]) for p, _ in poses]
        assert np.isclose(outcomes[name].max_displacement, max(planar))
    assert outcomes[TARGET].max_displacement > 0.25  # carried to the goal
    assert outcomes[DISTRACTOR].max_displacement < 1e-4


# --- distractor experiments (Tests 4, 5, 6, 9) -------------------------------------------


def check_consistent(result) -> None:
    """Flags agree with the per-object outcomes and contacts they summarise."""
    label = result.label
    d = next(o for o in result.objects if o.name == DISTRACTOR)
    assert ("displaced" in label.flags) == (d.max_displacement > THRESHOLDS.disturbance_threshold)
    assert ("toppled" in label.flags) == any(o.toppled for o in result.objects)
    assert d.toppled == (d.resting_face_final not in (None, d.resting_face_start))
    if any(DISTRACTOR in c.pair.split("-") and c.unintended for c in result.evidence.contacts):
        assert {"collision_object", "collision_environment"} & set(label.flags)
    scene_failed = bool({"displaced", "toppled", "knocked_off"} & set(label.flags))
    assert label.y == int(result.success and not scene_failed)
    assert (label.failure_stage == "") == (label.y == 1)


@pytest.mark.slow
def test_far_distractor_is_inert(env) -> None:
    """Test 4: 20 scenes with the tall distractor >= 15 cm from the swept hand."""
    scenes = trials.far_scenes(env)
    assert len(scenes) == 20
    for scene, distance in scenes:
        assert distance >= trials.FAR_DISTANCE
        result, _ = trials.attempt(env, scene, trials.legacy_action)
        d = result.objects[1]
        assert result.success and result.label.y == 1, scene.config
        assert not any(DISTRACTOR in c.pair.split("-") and c.pair != "distractor-table"
                       for c in result.evidence.contacts)  # fmt: skip
        assert d.max_displacement < 0.001
        assert d.resting_face_start == d.resting_face_final == "-z"
        check_consistent(result)


def test_grasp_axis_contrast_is_recorded(env) -> None:
    """Test 5 (diagnostic): the tall distractor 4 cm along candidate A's closing axis."""
    for seed in (0, 1):
        scene, a = trials.beside_scene(env, seed, axis_candidate=seed % 2)
        for index in (seed % 2, 1 - seed % 2):
            result, action = trials.attempt(env, scene, trials.candidate(index))
            assert result.grasp.commanded_face_axis == result.grasp.face_axis
            check_consistent(result)
            for c in result.evidence.contacts:
                assert c.peak_force > 0 and c.steps > 0


def test_carry_path_collision_is_detected(env) -> None:
    """Test 6 (diagnostic): the tall distractor halfway along the carry path."""
    scene = trials.carry_path_scenes(count=1)[0]
    result, _ = trials.attempt(env, scene, trials.legacy_action)
    check_consistent(result)
    hits = [c for c in result.evidence.contacts if c.pair == "distractor-target"]
    assert hits and hits[0].phase == "carry" and hits[0].unintended
    assert "collision_object" in result.label.flags


def test_same_scene_and_action_repeat_bitwise(env) -> None:
    """Test 9: same environment twice, then a fresh one."""
    scene, _ = trials.beside_scene(env, 2, axis_candidate=0)
    finals, labels = [], []
    for e in (env, env, None):
        fresh = e is None
        e = make_tabletop_env(CONFIG, seed=0) if fresh else e
        result, _ = trials.attempt(e, scene, trials.candidate(1))
        finals.append(e.sim.data.qpos.copy())
        labels.append(result.label)
        if fresh:
            e.close()
    assert all(np.array_equal(finals[0], f) for f in finals[1:])
    assert labels[0] == labels[1] == labels[2]


# --- physical toppling and leaning (Test 12) -------------------------------------------


def hold_still(env, steps: int) -> None:
    settle(env, steps)


def upright_start(env):
    """A valid upright scene and the distractor's start record, kept for judging."""
    layout = {TARGET: Placement(0.0, -0.1, 0.0), DISTRACTOR: Placement(0.15, -0.15, 0.0)}
    valid_scene(env, layout)
    return object_record(env, DISTRACTOR)


def judge_after_settling(env, start, steps: int = 40):
    hold_still(env, steps)
    return object_outcome(start, object_record(env, DISTRACTOR), table=table_surface(env))


def test_block_on_its_side_is_toppled(env) -> None:
    start = upright_start(env)
    assert start.position[2] > 0.85  # standing (11 cm tall)
    # Lay it on a side face, just above the table, and let it come to rest.
    q = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])  # 90 deg about world x
    set_free_body(env, DISTRACTOR, [0.15, -0.15, 0.8 + 0.025 + 0.002], q)
    o = judge_after_settling(env, start)
    assert o.resting_face_start == "-z" and o.resting_face_final in ("+y", "-y")
    assert o.toppled and not o.tilted and not o.leaning and not o.unsettled
    assert np.isclose(o.tilt_final, np.pi / 2, atol=np.radians(1))


def test_upside_down_block_is_toppled(env) -> None:
    start = upright_start(env)
    set_free_body(env, DISTRACTOR, [0.15, -0.15, 0.8 + 0.055 + 0.002], [0.0, 1.0, 0.0, 0.0])
    o = judge_after_settling(env, start)
    assert o.resting_face_final == "+z" and o.toppled
    assert np.isclose(o.tilt_final, np.pi, atol=np.radians(1))


def leaning_pose(target: Placement, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Tall block on the target's -x side, tipped by angle towards it about world y, its
    +x face resting on the target's top edge (8 cm up, 2 cm from the target's centre)."""
    w, h = DISTRACTOR_TALL.half_size[0], DISTRACTOR_TALL.half_size[2]
    face_x = target.x - 0.02
    pivot = np.array([face_x - 0.08 * np.tan(angle) - 0.0005, target.y, 0.8])
    q = np.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0])
    centre = pivot + quat_to_matrix(q) @ np.array([-w, 0.0, h])
    return centre + [0.0, 0.0, 0.0005], q


def test_block_leaning_on_the_target_is_not_toppled(env) -> None:
    start = upright_start(env)
    target = Placement(0.0, -0.1, 0.0)
    position, q = leaning_pose(target, np.radians(27))
    set_free_body(env, DISTRACTOR, position, q)
    o = judge_after_settling(env, start, steps=60)
    assert not o.unsettled, "the leaning block did not come to rest"
    assert o.resting_face_final is None and not o.toppled
    assert o.leaning and o.tilted and not o.unresolved_pose
    assert TARGET in o.final_contacts and "table" in o.final_contacts
    assert np.radians(24) < o.tilt_final < np.radians(31)
    assert o.com_outside_base  # beyond its 24.4-degree edge tipping angle, held up
    t = object_outcome(
        object_record(env, TARGET), object_record(env, TARGET), table=table_surface(env)
    )
    assert t.resting_face_final == "-z"  # the target still stands flat
