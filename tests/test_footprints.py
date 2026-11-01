import numpy as np
import pytest

from robot_uncertainty.sim.layouts import convex_hull, swept_hand_footprint
from robot_uncertainty.sim.scene import (
    DISTRACTOR_SHORT,
    DISTRACTOR_TALL,
    BlockSpec,
    Placement,
    TabletopConfig,
    footprint_circle_distance,
    footprint_corners,
    footprint_distance,
    footprints_overlap,
    place_beside,
)

TARGET = BlockSpec("target")  # 4 x 4 cm footprint
TALL = DISTRACTOR_TALL  # 5 x 5 cm footprint


def corners(spec: BlockSpec, x: float, y: float, yaw: float = 0.0) -> np.ndarray:
    return footprint_corners(spec, Placement(x, y, yaw))


def test_presets_are_distinct_distractors_and_the_default_scene_is_unchanged() -> None:
    assert DISTRACTOR_SHORT.size == (0.05, 0.05, 0.06)
    assert DISTRACTOR_TALL.size == (0.05, 0.05, 0.11)
    for preset in (DISTRACTOR_SHORT, DISTRACTOR_TALL):
        assert preset.role == "distractor"
        assert preset.rgba != TARGET.rgba
    assert DISTRACTOR_SHORT.rgba != DISTRACTOR_TALL.rgba
    assert TabletopConfig().blocks == (BlockSpec("target"),)


def test_separated_touching_and_overlapping_axis_aligned_footprints() -> None:
    a = corners(TARGET, 0.0, 0.0)
    # Faces 2 cm and 2.5 cm from the centres: 1 cm apart at x = 0.055.
    separated = corners(TALL, 0.055, 0.0)
    assert np.isclose(footprint_distance(a, separated), 0.01)
    assert not footprints_overlap(a, separated)
    touching = corners(TALL, 0.045, 0.0)
    assert abs(footprint_distance(a, touching)) < 1e-12
    assert not footprints_overlap(a, touching)
    overlapping = corners(TALL, 0.040, 0.0)
    assert np.isclose(footprint_distance(a, overlapping), -0.005)
    assert footprints_overlap(a, overlapping)


def test_corner_to_corner_distance_is_euclidean() -> None:
    a = corners(TARGET, 0.0, 0.0)
    b = corners(TARGET, 0.07, 0.07)  # nearest corners (0.02, 0.02) and (0.05, 0.05)
    assert np.isclose(footprint_distance(a, b), np.hypot(0.03, 0.03))


def test_rotated_footprints_overlap_where_their_enclosing_squares_would_not_say() -> None:
    a = corners(TARGET, 0.0, 0.0)
    # A 45-degree block reaches 25 sqrt(2) = 35.4 mm along x from its centre.
    b = corners(TALL, 0.054, 0.0, np.pi / 4)
    assert footprints_overlap(a, b)
    assert np.isclose(footprint_distance(a, b), -(0.02 + 0.025 * np.sqrt(2) - 0.054))
    c = corners(TALL, 0.056, 0.0, np.pi / 4)
    assert not footprints_overlap(a, c)
    assert np.isclose(footprint_distance(a, c), 0.056 - 0.02 - 0.025 * np.sqrt(2))


def test_clearance_counts_near_misses_as_overlap() -> None:
    a = corners(TARGET, 0.0, 0.0)
    b = corners(TALL, 0.055, 0.0)  # 1 cm apart
    assert not footprints_overlap(a, b, clearance=0.0099)
    assert footprints_overlap(a, b, clearance=0.0101)


def test_goal_circle_distance() -> None:
    a = corners(TARGET, 0.0, 0.0)
    assert np.isclose(footprint_circle_distance(a, (0.1, 0.0), 0.05), 0.03)
    assert np.isclose(footprint_circle_distance(a, (0.05, 0.05), 0.01), np.hypot(0.03, 0.03) - 0.01)
    assert footprint_circle_distance(a, (0.06, 0.0), 0.05) < 0  # overlaps the edge
    assert footprint_circle_distance(a, (0.0, 0.0), 0.005) < 0  # circle inside
    assert footprint_circle_distance(a[::-1], (0.0, 0.0), 0.005) < 0  # either winding


@pytest.mark.parametrize("gap", [0.0, 0.004, 0.04, 0.1])
@pytest.mark.parametrize("yaw", [0.0, 0.3, -1.2, 2.5])
@pytest.mark.parametrize("face", [0, 1, 2, 3])
def test_place_beside_gives_the_face_to_face_gap(gap, yaw, face) -> None:
    target = Placement(0.02, -0.1, yaw)
    heading = yaw + face * np.pi / 2  # a face normal of the target
    other = place_beside(target, TARGET, TALL, gap, heading, other_yaw=yaw)
    assert other.yaw == yaw
    a, b = footprint_corners(TARGET, target), footprint_corners(TALL, other)
    assert np.isclose(footprint_distance(a, b), gap, atol=1e-12)
    direction = np.arctan2(*(other.xy - target.xy)[::-1])
    assert abs((direction - heading + np.pi) % (2 * np.pi) - np.pi) < 1e-9
    assert np.isclose(np.linalg.norm(other.xy - target.xy), 0.02 + gap + 0.025)


def test_place_beside_with_a_turned_distractor_keeps_at_least_the_gap() -> None:
    target = Placement(0.0, 0.0, 0.0)
    other = place_beside(target, TARGET, TALL, 0.01, 0.0, other_yaw=np.pi / 4)
    a, b = footprint_corners(TARGET, target), footprint_corners(TALL, other)
    assert np.isclose(footprint_distance(a, b), 0.01)  # its corner points at the face
    other = place_beside(target, TARGET, TALL, 0.01, 0.3, other_yaw=0.0)
    assert footprint_distance(a, footprint_corners(TALL, other)) >= 0.01 - 1e-12


def test_swept_hand_footprint_is_the_hull_of_both_ends() -> None:
    hull = swept_hand_footprint((0.0, 0.0), (0.0, 0.2), 0.0, (0.1, 0.03))
    assert len(hull) == 4
    assert np.allclose(
        sorted(map(tuple, hull)), [(-0.1, -0.03), (-0.1, 0.23), (0.1, -0.03), (0.1, 0.23)]
    )
    square = convex_hull(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]], dtype=float))
    assert len(square) == 4
