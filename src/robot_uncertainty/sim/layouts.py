from __future__ import annotations

import numpy as np

from robot_uncertainty.sim.scene import (
    BlockSpec,
    GoalRegion,
    Placement,
    footprint_circle_distance,
    footprint_corners,
    footprint_distance,
    footprints_overlap,
    place_beside,
)


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull of 2D points (Andrew's monotone chain)."""
    pts = sorted(map(tuple, np.asarray(points, dtype=float)))

    def half(seq):
        out: list[tuple[float, float]] = []
        for p in seq:
            while len(out) >= 2:
                (ax, ay), (bx, by) = out[-2], out[-1]
                if (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax) > 0:
                    break
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(reversed(pts))
    return np.array(lower[:-1] + upper[:-1])


# The hand is modelled by its top-down footprint: a rectangle around the grip site, long
# along the finger-closing axis (tabletop.hand_footprint_half_extent measures it from the
# model).
def hand_rectangle(center, heading: float, half_extent) -> np.ndarray:
    """Corners of the hand's footprint at a grip-site position, closing along heading."""
    spec = BlockSpec("hand", size=(2 * half_extent[0], 2 * half_extent[1], 0.0))
    return footprint_corners(spec, Placement(float(center[0]), float(center[1]), heading))


def swept_hand_footprint(start, end, heading: float, half_extent) -> np.ndarray:
    """Region covered by the hand footprint moved in a straight line without turning."""
    return convex_hull(
        np.concatenate(
            [hand_rectangle(start, heading, half_extent), hand_rectangle(end, heading, half_extent)]
        )
    )


def hand_clearance(
    spec: BlockSpec, placement: Placement, start, end, heading: float, half_extent
) -> float:
    """Distance from a block's footprint to the swept hand footprint (negative: overlap)."""
    swept = swept_hand_footprint(start, end, heading, half_extent)
    return footprint_distance(footprint_corners(spec, placement), swept)


def placement_fits(
    spec: BlockSpec,
    placement: Placement,
    table_half: tuple[float, float],
    goal: GoalRegion,
    others: list[tuple[BlockSpec, Placement]],
) -> bool:
    """The same checks as TabletopEnv.set_layout, for one block against others."""
    corners = footprint_corners(spec, placement)
    if np.any(np.abs(corners) > np.asarray(table_half)):
        return False
    if footprint_circle_distance(corners, goal.center, goal.radius) <= 0.0:
        return False
    return not any(footprints_overlap(corners, footprint_corners(s, p)) for s, p in others)


# These place a distractor relative to a target for targeted checks (far away, beside
# one grasp axis, on the carry path); they are not a clutter generator.
def far_placement(
    target_spec: BlockSpec,
    target: Placement,
    spec: BlockSpec,
    heading: float,
    half_extent,
    goal: GoalRegion,
    table_half: tuple[float, float],
    step: float = 0.005,
    min_x: float = -np.inf,
) -> tuple[Placement, float]:
    """The distractor placement (yaw 0, on a grid) furthest from the hand's footprint
    swept in a straight line from the target to the goal, and that distance. min_x
    limits the search (e.g. 0 keeps to the half of the table away from the robot, whose
    arm this footprint does not model).

    The hand's grip site is taken to move from the target's centre to the goal's centre
    with the fingers closing along heading; the true path differs by the block's offset
    in the grip (a few millimetres).
    """
    half = np.asarray(table_half) - spec.half_size[:2]
    xs = np.arange(-half[0], half[0] + 1e-9, step)
    xs = xs[xs >= min_x]
    ys = np.arange(-half[1], half[1] + 1e-9, step)
    swept = swept_hand_footprint(target.xy, goal.center, heading, half_extent)
    best, best_distance = None, -np.inf
    for x in xs:
        for y in ys:
            p = Placement(float(x), float(y), 0.0)
            if not placement_fits(spec, p, table_half, goal, [(target_spec, target)]):
                continue
            distance = footprint_distance(footprint_corners(spec, p), swept)
            if distance > best_distance:
                best, best_distance = p, distance
    return best, float(best_distance)


def beside_axis_placement(
    target_spec: BlockSpec,
    target: Placement,
    spec: BlockSpec,
    gap: float,
    closing_heading: float,
    goal: GoalRegion,
    table_half: tuple[float, float],
) -> Placement:
    """A distractor gap metres face-to-face from the target along a closing axis, its
    faces parallel to the target's.

    Of the two sides along the axis, the one facing away from the goal is used (so the
    carry does not start over the distractor), unless only the other fits on the table.
    """
    u = np.array([np.cos(closing_heading), np.sin(closing_heading)])
    towards_goal = u @ (np.asarray(goal.center) - target.xy) > 0
    sides = [closing_heading + np.pi, closing_heading]
    if not towards_goal:
        sides.reverse()
    for heading in sides:
        p = place_beside(target, target_spec, spec, gap, heading, target.yaw)
        if placement_fits(spec, p, table_half, goal, [(target_spec, target)]):
            return p
    raise ValueError("no side of the closing axis fits the distractor on the table")


def carry_path_placement(target: Placement, goal: GoalRegion, fraction: float = 0.5) -> Placement:
    """A distractor centred on the straight line from the target to the goal centre,
    fraction of the way along, with a face across the path."""
    start, end = target.xy, np.asarray(goal.center, dtype=float)
    x, y = start + fraction * (end - start)
    yaw = float(np.arctan2(*(end - start)[::-1]))
    return Placement(float(x), float(y), yaw)
