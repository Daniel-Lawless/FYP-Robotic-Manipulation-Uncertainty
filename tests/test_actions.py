from types import SimpleNamespace

import numpy as np

from robot_uncertainty.sim.actions import (
    GraspAction,
    execution_heading,
    grasp_candidates,
    legacy_equivalent,
    wrap_angle,
)

# Drawn once at import, in this order, so no test depends on another having run first.
RNG = np.random.default_rng(20261031)
CANDIDATE_YAWS = RNG.uniform(-10, 10, 200)
SYMMETRY_YAWS = RNG.uniform(-np.pi, np.pi, 50)
EXECUTION_TRIALS = 500
DEG = np.pi / 180


def observed(yaw: float) -> SimpleNamespace:
    return SimpleNamespace(name="target", yaw=yaw)


def same_axis(a: float, b: float, tol: float = 1e-9) -> bool:
    """Headings along the same line (equal modulo pi)."""
    return abs(wrap_angle(a - b, np.pi)) < tol


def legacy_yaw(object_yaw: float, current_yaw: float) -> float:
    """primitives.aligned_grasp_yaw for a square block, written out (CI has no MuJoCo);
    tests/sim/test_explicit_actions.py checks it against the real function."""
    return current_yaw + wrap_angle(object_yaw - current_yaw, np.pi / 2)


def test_candidates_are_perpendicular_with_a_canonical_first_heading() -> None:
    for i, yaw in enumerate(CANDIDATE_YAWS):
        case = f"CANDIDATE_YAWS[{i}], yaw={yaw!r}"
        first, second = grasp_candidates(observed(yaw))
        assert first.object == second.object == "target", case
        assert -np.pi / 4 <= first.closing_heading < np.pi / 4, (
            f"{case}: first heading {first.closing_heading!r}"
        )
        assert np.isclose(second.closing_heading - first.closing_heading, np.pi / 2), (
            f"{case}: headings {first.closing_heading!r}, {second.closing_heading!r}"
        )
        # Both close along face normals of the observed block.
        for c in (first, second):
            assert abs(wrap_angle(c.closing_heading - yaw, np.pi / 2)) < 1e-9, (
                f"{case}: heading {c.closing_heading!r} is not a face normal"
            )


def test_candidates_respect_the_square_blocks_quarter_turn_symmetry() -> None:
    for i, yaw in enumerate(SYMMETRY_YAWS):
        base = grasp_candidates(observed(yaw))
        for k in (-2, -1, 1, 2, 3):
            turned = grasp_candidates(observed(yaw + k * np.pi / 2))
            if abs(abs(wrap_angle(yaw, np.pi / 2)) - np.pi / 4) < 1e-9:
                continue  # at the canonical boundary the order may swap (see below)
            for a, b in zip(base, turned, strict=True):
                assert np.isclose(a.closing_heading, b.closing_heading, atol=1e-9), (
                    f"SYMMETRY_YAWS[{i}], yaw={yaw!r}, k={k}: "
                    f"{a.closing_heading!r} vs {b.closing_heading!r}"
                )


def test_candidate_index_swaps_where_the_canonical_yaw_crosses_45_degrees() -> None:
    below = grasp_candidates(observed(np.pi / 4 - 1e-6))
    above = grasp_candidates(observed(np.pi / 4 + 1e-6))
    # The same physical face pair moves from index 0 to index 1.
    assert same_axis(below[0].closing_heading, above[1].closing_heading, tol=1e-5)
    assert same_axis(below[1].closing_heading, above[0].closing_heading, tol=1e-5)


def test_execution_keeps_the_face_pair_and_turns_the_wrist_at_most_90_degrees() -> None:
    for trial in range(EXECUTION_TRIALS):
        rng = np.random.default_rng(trial)
        yaw, gripper = rng.uniform(-np.pi, np.pi, 2)
        action = grasp_candidates(observed(yaw))[trial % 2]
        perturbation = rng.uniform(-44.9, 44.9) * DEG
        executed = execution_heading(action.closing_heading, yaw + perturbation, gripper)
        case = (
            f"trial {trial} (seed {trial}, candidate {trial % 2}): yaw={yaw!r}, "
            f"gripper={gripper!r}, perturbation={perturbation!r}, executed={executed!r}"
        )
        # Same face pair: the decided heading turned with the block.
        assert same_axis(executed, action.closing_heading + perturbation), (
            f"{case}, expected axis {action.closing_heading + perturbation!r}"
        )
        assert abs(executed - gripper) <= np.pi / 2 + 1e-12, case


def test_execution_without_perturbation_is_the_decided_axis() -> None:
    for yaw in np.linspace(-3, 3, 61):
        for action in grasp_candidates(observed(yaw)):
            for gripper in np.linspace(-3, 3, 13):
                executed = execution_heading(action.closing_heading, yaw, gripper)
                assert same_axis(executed, action.closing_heading)


def test_legacy_equivalent_matches_the_least_rotation_rule_on_random_inputs() -> None:
    rng = np.random.default_rng(7)
    checked = 0
    for yaw, gripper in rng.uniform(-np.pi, np.pi, (10_000, 2)):
        legacy = legacy_yaw(yaw, gripper)
        if abs(abs(wrap_angle(yaw - gripper, np.pi / 2)) - np.pi / 4) < 1e-9:
            continue  # ties are tested separately
        action = legacy_equivalent(grasp_candidates(observed(yaw)), gripper)
        assert same_axis(action.closing_heading, legacy)
        assert abs(execution_heading(action.closing_heading, yaw, gripper) - legacy) < 1e-9
        checked += 1
    assert checked == 10_000


def test_legacy_equivalent_tie_takes_the_minus_45_degree_turn() -> None:
    # Block at exactly 45 degrees to the fingers: both face pairs need a 45-degree turn.
    for gripper in (0.0, 0.5, -2.0):
        candidates = (
            GraspAction("target", gripper - np.pi / 4),
            GraspAction("target", gripper + np.pi / 4),
        )
        for order in (candidates, candidates[::-1]):
            chosen = legacy_equivalent(order, gripper)
            assert np.isclose(chosen.closing_heading, gripper - np.pi / 4)
    # Through grasp_candidates, with exactly representable angles.
    chosen = legacy_equivalent(grasp_candidates(observed(np.pi / 4)), 0.0)
    assert chosen.closing_heading == -np.pi / 4 == legacy_yaw(np.pi / 4, 0.0)
