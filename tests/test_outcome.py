from dataclasses import replace

import numpy as np
import pytest

from robot_uncertainty.sim.outcome import (
    FLAGS,
    ContactEvent,
    ObjectRecord,
    OutcomeEvidence,
    OutcomeThresholds,
    PlacementCheck,
    PlacementTolerances,
    TableSurface,
    com_outside_base,
    object_outcome,
    outcome,
    quat_to_matrix,
    resting_face,
)

DEG = np.pi / 180
TABLE = TableSurface(height=0.8, half_x=0.3, half_y=0.3)
TALL = np.array([0.05, 0.05, 0.11])
SHORT = np.array([0.05, 0.05, 0.06])
TARGET = np.array([0.04, 0.04, 0.08])
SINK = 1e-4  # resting blocks sink about 0.1 mm into the table in MuJoCo
ALL_PHASES = (
    "open", "pregrasp", "descend", "close", "lift", "hold",
    "carry", "lower", "release", "retract", "settle",
)  # fmt: skip


def quat(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    return np.array([np.cos(angle / 2), *(np.sin(angle / 2) * axis)])


def qmul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def record(name="distractor", size=TALL, position=(0.1, 0.0, None), q=(1, 0, 0, 0), **kw):
    size = np.asarray(size, dtype=float)
    position = np.array(
        [position[0], position[1], 0.8 + size[2] / 2 - SINK if position[2] is None else position[2]]
    )
    fields = dict(
        name=name,
        role="target" if name == "target" else "distractor",
        size=size,
        position=position,
        quaternion=np.asarray(q, dtype=float),
        linear_speed=0.0,
        angular_speed=0.0,
        contacts=("table",),
    )
    return ObjectRecord(**{**fields, **kw})


def upright(name="distractor", size=TALL, x=0.1, y=0.0, yaw=0.0, **kw) -> ObjectRecord:
    return record(name, size, (x, y, None), quat((0, 0, 1), yaw), **kw)


def pivoted(start: ObjectRecord, pivot_xy, axis, angle: float, **kw) -> ObjectRecord:
    """start rotated by angle about a horizontal axis through a point of its base."""
    pivot = np.array([*pivot_xy, 0.8])
    q = quat(axis, angle)
    position = pivot + quat_to_matrix(q) @ (start.position - pivot)
    return replace(start, position=position, quaternion=qmul(q, start.quaternion), **kw)


def on_side(start: ObjectRecord, **kw) -> ObjectRecord:
    """Lying on its -y face (rotated 90 degrees about world x), resting on the table."""
    q = qmul(quat((1, 0, 0), np.pi / 2), start.quaternion)
    z = 0.8 + start.size[1] / 2 - SINK
    return replace(start, position=np.array([*start.position[:2], z]), quaternion=q, **kw)


def upside_down(start: ObjectRecord, **kw) -> ObjectRecord:
    q = qmul(quat((1, 0, 0), np.pi), start.quaternion)
    return replace(start, quaternion=q, **kw)


def judge(start, final, positions=None, quaternions=None):
    return object_outcome(start, final, positions, quaternions, TABLE)


# --- resting faces -----------------------------------------------------------------


def test_upright_block_rests_on_its_bottom_face() -> None:
    assert resting_face(upright(), TABLE) == "-z"
    assert resting_face(upright(yaw=1.1), TABLE) == "-z"


def test_side_and_upside_down_faces() -> None:
    assert resting_face(on_side(upright()), TABLE) == "-y"
    assert resting_face(upside_down(upright()), TABLE) == "+z"
    side_x = replace(
        upright(),
        position=np.array([0.1, 0.0, 0.8 + 0.025 - SINK]),
        quaternion=quat((0, 1, 0), np.pi / 2),  # body +x now points down
    )
    assert resting_face(side_x, TABLE) == "+x"


def test_tilted_block_has_no_flat_face() -> None:
    start = upright()
    for angle in (10 * DEG, 30 * DEG, 45 * DEG):
        assert resting_face(pivoted(start, (0.125, 0.0), (0, 1, 0), angle), TABLE) is None


def test_raised_corner_within_and_beyond_the_support_tolerance() -> None:
    start = upright()
    # 0.5 deg about an edge raises the far edge 0.44 mm: still flat on the table.
    assert resting_face(pivoted(start, (0.125, 0.0), (0, 1, 0), 0.5 * DEG), TABLE) == "-z"
    # 3 deg about a corner raises the far corner 3.7 mm, though the face normal is
    # within 5 degrees of straight down.
    raised = pivoted(start, (0.125, 0.025), (-1, 1, 0), 3 * DEG)
    assert resting_face(raised, TABLE) is None


def test_face_needs_table_contact_and_rest() -> None:
    assert resting_face(upright(contacts=()), TABLE) is None
    assert resting_face(upright(contacts=("target",)), TABLE) is None
    assert resting_face(upright(linear_speed=0.01), TABLE) is None
    assert resting_face(upright(angular_speed=0.1), TABLE) is None


# --- centre of mass over the base ----------------------------------------------------


@pytest.mark.parametrize(
    "pivot, axis, angle, outside",
    [
        ((0.125, 0.0), (0, 1, 0), 24.0 * DEG, False),  # edge tipping angle 24.4 deg
        ((0.125, 0.0), (0, 1, 0), 26.0 * DEG, True),
        ((0.125, 0.025), (-1, 1, 0), 26.0 * DEG, False),  # corner tipping angle 32.7 deg
        ((0.125, 0.025), (-1, 1, 0), 34.0 * DEG, True),
    ],
)
def test_com_outside_base_is_direction_aware_for_the_tall_block(
    pivot, axis, angle, outside
) -> None:
    tipped = pivoted(upright(), pivot, axis, angle)
    assert com_outside_base(tipped, 2, -1.0) is outside


def test_com_over_the_original_base_for_side_and_upside_down_blocks() -> None:
    start = upright()
    assert com_outside_base(on_side(start), 2, -1.0)  # base face is now vertical
    assert not com_outside_base(upside_down(start), 2, -1.0)  # base above the centre


def test_short_block_tipping_angles() -> None:
    start = upright(size=SHORT)
    edge = (0.125, 0.0), (0, 1, 0)
    assert not com_outside_base(pivoted(start, *edge, 39.0 * DEG), 2, -1.0)  # 39.8 deg
    assert com_outside_base(pivoted(start, *edge, 41.0 * DEG), 2, -1.0)


# --- per-object outcomes ---------------------------------------------------------------


def path(start: ObjectRecord, offsets_x) -> tuple[np.ndarray, np.ndarray]:
    positions = np.array([start.position + [dx, 0.0, 0.0] for dx in offsets_x])
    return positions, np.tile(start.quaternion, (len(offsets_x), 1))


def test_undisturbed_block() -> None:
    start = upright()
    o = judge(start, start, *path(start, [0.0, 0.0]))
    assert o.resting_face_start == o.resting_face_final == "-z"
    assert o.max_displacement == o.final_displacement == 0.0
    assert not (o.toppled or o.tilted or o.leaning or o.unresolved_pose or o.unsettled)
    assert not (o.off_table or o.com_outside_base)
    assert o.tilt_start == o.tilt_final == o.tilt_max == 0.0


def test_sustained_and_temporary_displacement() -> None:
    start = upright()
    moved = replace(start, position=start.position + [0.029, 0.0, 0.0])
    o = judge(start, moved, *path(start, np.linspace(0.0, 0.029, 10)))
    assert np.isclose(o.max_displacement, 0.029) and np.isclose(o.final_displacement, 0.029)
    back = replace(start, position=start.position + [0.005, 0.0, 0.0])
    o = judge(start, back, *path(start, [0.01, 0.025, 0.04, 0.02, 0.005]))
    assert np.isclose(o.max_displacement, 0.04) and np.isclose(o.final_displacement, 0.005)


def test_yaw_change_respects_the_square_symmetry() -> None:
    start = upright(yaw=0.2)
    o = judge(start, upright(yaw=0.2 + np.pi / 2 + 0.05))
    assert np.isclose(o.yaw_change, 0.05)


def test_side_topple_against_the_preserved_upright_start() -> None:
    start = upright()
    o = judge(start, on_side(start))
    assert o.resting_face_start == "-z" and o.resting_face_final == "-y"
    assert o.toppled and not o.tilted and not o.leaning and not o.unresolved_pose
    assert np.isclose(o.tilt_final, np.pi / 2) and o.com_outside_base


def test_starting_sideways_would_hide_the_topple() -> None:
    # Why the start record must be the upright block: judged against a sideways start,
    # the same final state is not a topple.
    side = on_side(upright())
    assert not judge(side, side).toppled


def test_upside_down_topple_against_the_preserved_upright_start() -> None:
    start = upright()
    o = judge(start, upside_down(start))
    assert o.resting_face_final == "+z" and o.toppled
    assert np.isclose(o.tilt_final, np.pi) and not o.com_outside_base


def test_supported_lean_is_not_a_topple() -> None:
    start = upright()
    lean = pivoted(start, (0.125, 0.0), (0, 1, 0), 27 * DEG, contacts=("table", "target"))
    o = judge(start, lean, *path(start, [0.0]))
    assert o.resting_face_final is None and not o.toppled
    assert o.leaning and o.tilted and not o.unresolved_pose
    assert np.isclose(o.tilt_final, 27 * DEG) and o.com_outside_base
    held = pivoted(start, (0.125, 0.0), (0, 1, 0), 10 * DEG, contacts=("palm", "table"))
    assert judge(start, held).leaning  # the robot can hold a block up too


def test_tilted_unsupported_block_is_unresolved() -> None:
    start = upright()
    tilted = pivoted(start, (0.125, 0.0), (0, 1, 0), 10 * DEG)
    o = judge(start, tilted)
    assert o.tilted and o.unresolved_pose and not o.leaning and not o.toppled
    assert not o.com_outside_base


def test_moving_block_is_unsettled_and_unresolved() -> None:
    start = upright()
    o = judge(start, replace(on_side(start), angular_speed=0.5))
    assert o.unsettled and o.resting_face_final is None and o.unresolved_pose
    assert not o.toppled  # not until it comes to rest on a face


def test_small_tilt_is_not_tilted_and_max_tilt_covers_the_path() -> None:
    start = upright()
    final = pivoted(start, (0.125, 0.0), (0, 1, 0), 0.5 * DEG)
    mid = pivoted(start, (0.125, 0.0), (0, 1, 0), 12 * DEG)
    o = judge(start, final, mid.position[None], mid.quaternion[None])
    assert not o.tilted and o.resting_face_final == "-z"
    assert np.isclose(o.tilt_max, 12 * DEG) and np.isclose(o.tilt_final, 0.5 * DEG)


def test_off_table() -> None:
    start = upright()
    fallen = replace(start, position=np.array([0.35, 0.0, 0.03]), contacts=("floor",))
    assert judge(start, fallen).off_table
    beyond = replace(start, position=np.array([0.31, 0.0, 0.85]), contacts=())
    assert judge(start, beyond).off_table


def test_object_outcome_requires_a_table() -> None:
    with pytest.raises(ValueError):
        object_outcome(upright(), upright())


# --- invalid initial evidence ----------------------------------------------------------


def test_start_slightly_too_fast_is_rejected_not_toppled() -> None:
    # Regression: an unresolved start face once made this unchanged block "toppled".
    start = upright(linear_speed=0.006)
    with pytest.raises(ValueError, match="distractor: initial resting face is unresolved"):
        judge(start, upright())


@pytest.mark.parametrize(
    "start",
    [
        upright(contacts=()),  # not touching the table
        upright(angular_speed=0.06),  # turning
        record(position=(0.1, 0.0, 0.8 + TALL[2] / 2 + 0.002)),  # 2 mm above the table
        pivoted(upright(), (0.125, 0.0), (0, 1, 0), 10 * DEG),  # on one edge
    ],
    ids=["no-table-contact", "angular-speed", "raised", "on-edge"],
)
def test_start_without_a_resting_face_is_rejected(start) -> None:
    with pytest.raises(ValueError, match="distractor: initial resting face is unresolved"):
        judge(start, upright())


def test_known_side_start_on_the_same_face_is_not_toppled() -> None:
    side = on_side(upright())
    o = judge(side, side)
    assert o.resting_face_start == o.resting_face_final == "-y" and not o.toppled


def test_start_face_uses_the_supplied_tolerances_and_thresholds() -> None:
    fast = upright(linear_speed=0.006)
    loose = PlacementTolerances(max_linear_speed=0.01)
    assert object_outcome(fast, upright(), table=TABLE, tolerances=loose).resting_face_start
    # A 3-degree tilt, with the support gap widened so only the face tilt decides.
    tilted = pivoted(upright(), (0.125, 0.0), (0, 1, 0), 3 * DEG)
    wide = PlacementTolerances(max_support_gap=0.005)
    o = object_outcome(tilted, tilted, table=TABLE, tolerances=wide)
    assert o.resting_face_start == "-z"
    strict = OutcomeThresholds(face_flat_tilt=2 * DEG)
    with pytest.raises(ValueError, match="unresolved"):
        object_outcome(tilted, tilted, table=TABLE, thresholds=strict, tolerances=wide)


# --- attempt labels -------------------------------------------------------------------


def placement(**kw) -> PlacementCheck:
    fields = dict(
        position=np.array([0.0, 0.18, 0.84 - SINK]),
        quaternion=np.array([1.0, 0, 0, 0]),
        centre_error=0.002,
        goal_margin=0.019,
        tilt=0.0,
        support_gap=-SINK,
        linear_speed=0.0,
        angular_speed=0.0,
        contacts=("table",),
        in_goal=True,
        upright=True,
        on_table=True,
        at_rest=True,
    )
    return PlacementCheck(**{**fields, **kw})


TARGET_START = upright("target", TARGET, 0.0, -0.1)
TARGET_PLACED = upright("target", TARGET, 0.0, 0.18)
DISTRACTOR_START = upright("distractor", TALL, 0.15, -0.05)


def evidence(distractor_final=None, distractor_path=None, **kw) -> OutcomeEvidence:
    target = judge(TARGET_START, TARGET_PLACED, *path(TARGET_START, [0.0, 0.1, 0.28]))
    final = distractor_final or DISTRACTOR_START
    distractor = judge(DISTRACTOR_START, final, *(distractor_path or (None, None)))
    fields = dict(
        target="target",
        grasp=None,
        grasp_success=True,
        phases=tuple((p, True) for p in ALL_PHASES),
        abort="",
        gripper_empty=False,
        dropped=False,
        drop_phase="",
        released=True,
        placement=placement(),
        objects=(target, distractor),
        contacts=(),
        table=TABLE,
    )
    return OutcomeEvidence(**{**fields, **kw})


def phases_with(timeout: str, until: str | None = None) -> tuple:
    names = ALL_PHASES[: ALL_PHASES.index(until) + 1] if until else ALL_PHASES
    return tuple((p, p != timeout) for p in names) + ((("settle", True),) if until else ())


def test_success() -> None:
    label = outcome(evidence())
    assert label.y == 1 and label.failure_stage == "" and label.flags == {}


@pytest.mark.parametrize("index, name", [(0, "target"), (1, "distractor")])
def test_supplied_outcome_without_a_start_face_is_rejected(index, name) -> None:
    # Otherwise a successful attempt: invalid evidence must not become y = 1 (or 0).
    e = evidence()
    objects = list(e.objects)
    objects[index] = replace(objects[index], resting_face_start=None)
    with pytest.raises(ValueError, match=f"{name}: resting_face_start is None"):
        outcome(replace(e, objects=tuple(objects)))


def test_target_moved_far_is_not_a_disturbance() -> None:
    e = evidence()
    assert e.object("target").max_displacement > 0.25
    assert outcome(e).y == 1


@pytest.mark.parametrize(
    "kw, stage, flag",
    [
        (dict(grasp_success=False), "grasp", "grasp_miss"),
        (dict(gripper_empty=True, phases=phases_with("", "hold")), "grasp", "grasp_miss"),
        (
            dict(phases=phases_with("descend", "hold"), grasp_success=False),
            "grasp",
            "grasp_timeout",
        ),
        (dict(dropped=True, drop_phase="carry"), "transport", "dropped"),
        (dict(dropped=True, drop_phase="lower"), "transport", "dropped"),
        (dict(phases=phases_with("lower", "lower")), "transport", "transport_timeout"),
        (dict(released=False), "release", "release_failure"),
        (dict(phases=phases_with("retract")), "release", "release_failure"),
        (
            dict(placement=placement(in_goal=False, goal_margin=-0.01)),
            "placement",
            "placement_failure",
        ),
        (dict(placement=placement(at_rest=False)), "placement", "placement_failure"),
        (
            dict(placement=placement(contacts=("fingers", "table"))),
            "placement",
            "placement_failure",
        ),
    ],
)
def test_stage_failures(kw, stage, flag) -> None:
    label = outcome(evidence(**kw))
    assert label.y == 0 and label.failure_stage == stage and flag in label.flags


def test_drop_phase_is_recorded() -> None:
    assert outcome(evidence(dropped=True, drop_phase="lower")).flags["dropped"] == ("lower",)


def test_disturbance_uses_the_maximum_displacement() -> None:
    start = DISTRACTOR_START

    def moved(final_dx, offsets):
        final = replace(start, position=start.position + [final_dx, 0.0, 0.0])
        return evidence(distractor_final=final, distractor_path=path(start, offsets))

    assert outcome(moved(0.029, [0.01, 0.029])).y == 1
    label = outcome(moved(0.031, [0.01, 0.031]))
    assert label.y == 0 and label.failure_stage == "scene"
    assert label.flags["displaced"] == ("distractor",)
    # Pushed 4 cm, then back to 5 mm from its start: still displaced.
    e = moved(0.005, [0.02, 0.04, 0.01])
    assert e.object("distractor").final_displacement < 0.03
    assert outcome(e).flags["displaced"] == ("distractor",)


def test_toppled_distractor_fails_the_scene() -> None:
    label = outcome(evidence(distractor_final=on_side(DISTRACTOR_START)))
    assert label.y == 0 and label.failure_stage == "scene"
    assert label.flags["toppled"] == ("distractor",)
    assert "displaced" not in label.flags  # it tipped in place


def test_knocked_off_fails_the_scene() -> None:
    fallen = replace(DISTRACTOR_START, position=np.array([0.4, 0.0, 0.03]), contacts=("floor",))
    label = outcome(evidence(distractor_final=fallen))
    assert label.failure_stage == "scene" and "knocked_off" in label.flags


def test_diagnostic_flags_do_not_fail() -> None:
    start = DISTRACTOR_START
    lean = pivoted(start, (0.175, -0.05), (0, 1, 0), 27 * DEG, contacts=("table", "target"))
    label = outcome(evidence(distractor_final=lean))
    assert label.y == 1
    assert {"tilted", "leaning", "com_outside_base"} <= set(label.flags)
    unsettled = replace(start, linear_speed=0.02)
    label = outcome(evidence(distractor_final=unsettled))
    assert label.y == 1 and label.flags["unsettled"] == ("distractor",)
    assert "unresolved_pose" in label.flags


def test_unintended_contacts_are_recorded_without_failing() -> None:
    contacts = (
        ContactEvent("palm-target", "descend", 40, 12, peak_force=20.0, kind="object"),
        ContactEvent("arm-table", "lower", 120, 2, peak_force=3.0, kind="environment"),
        ContactEvent("fingers-target", "close", 50, 15, unintended=False),
    )
    label = outcome(evidence(contacts=contacts))
    assert label.y == 1
    assert label.flags == {
        "collision_object": ("palm-target@descend",),
        "collision_environment": ("arm-table@lower",),
    }


def test_simultaneous_failures_report_the_earliest_stage() -> None:
    label = outcome(
        evidence(
            dropped=True,
            drop_phase="carry",
            released=True,
            placement=placement(in_goal=False, on_table=False),
            distractor_final=on_side(DISTRACTOR_START),
            contacts=(ContactEvent("distractor-target", "carry", 80, 3, peak_force=4.0),),
        )
    )
    assert label.failure_stage == "transport" and label.y == 0
    assert {"dropped", "placement_failure", "toppled", "collision_object"} <= set(label.flags)
    assert list(label.flags) == [f for f in FLAGS if f in label.flags]  # table order
