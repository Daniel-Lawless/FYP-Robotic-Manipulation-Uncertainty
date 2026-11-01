from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from robot_uncertainty.sim.tabletop import ObjectObservation

QUARTER = np.pi / 2


def wrap_angle(angle: float, period: float = 2 * np.pi) -> float:
    """Wrap an angle into [-period/2, period/2); the same as primitives.wrap_angle."""
    return float((angle + period / 2) % period - period / 2)


# A high-level decision picks which pair of opposing side faces the fingers close on, by
# fixing the world-frame heading of the finger-closing axis at decision time
# (GraspAction.closing_heading). Headings follow primitives.heading and
# primitives.top_down_rotation: the grip site's x axis is the closing axis, and a closing
# heading and that heading + pi close on the same faces.
@dataclass(frozen=True)
class GraspAction:
    """Top grasp of one block, closing the fingers along a world-frame heading (radians).

    Headings are physically equivalent modulo pi (the same pair of faces).
    """

    object: str
    closing_heading: float


# Candidate identity: a square block's observed yaw is only defined modulo pi/2, so
# grasp_candidates canonicalises it into [-pi/4, pi/4) and returns the two perpendicular
# face-pair choices. Which physical face pair is candidate 0 swaps whenever the
# canonical yaw crosses +-pi/4, so the candidate index is not a stable model input; the
# physical closing heading of the action is what identifies it.
def grasp_candidates(obs: ObjectObservation) -> tuple[GraspAction, GraspAction]:
    """The two face-pair grasps of an observed block: closing along its canonical yaw
    psi_c = wrap_angle(yaw, pi/2) in [-pi/4, pi/4), and along psi_c + pi/2.

    Face normals of a box are psi + k pi/2 for any k, so the pair covers both face pairs
    of any box; for a square block the index swaps where psi_c crosses +-pi/4 (see the
    comment above).
    """
    canonical = wrap_angle(obs.yaw, QUARTER)
    return GraspAction(obs.name, canonical), GraspAction(obs.name, canonical + QUARTER)


def legacy_equivalent(
    candidates: tuple[GraspAction, GraspAction], gripper_yaw: float
) -> GraspAction:
    """The candidate that primitives.aligned_grasp_yaw would grip: the face pair needing
    the least wrist rotation from gripper_yaw.

    Each candidate needs a wrist turn of wrap_angle(heading - gripper_yaw, pi), in
    [-pi/2, pi/2); the smaller turn wins. Ties (both turns +-pi/4, the block at exactly
    45 degrees to the fingers) go to the candidate needing -pi/4, which is what
    aligned_grasp_yaw gives in exact arithmetic, because wrap_angle maps +pi/4 to -pi/4.
    In floating point aligned_grasp_yaw resolves a tie by rounding (it turned +45
    degrees in data/validation/yaw_switch.csv), so only non-tie inputs are guaranteed to
    match.
    """
    turns = [wrap_angle(c.closing_heading - gripper_yaw, np.pi) for c in candidates]
    sizes = [abs(t) for t in turns]
    if sizes[0] == sizes[1]:
        return candidates[0] if turns[0] < turns[1] else candidates[1]
    return candidates[int(np.argmin(sizes))]


# Execution: at execution time the robot re-observes the block and execution_heading
# refines the commanded heading by at most pi/4, so the intended face pair is kept for any
# orientation change under 45 degrees, and turns the wrist by at most pi/2 by choosing
# between the two equivalent wrist orientations. It is deterministic for now.
def execution_heading(closing_heading: float, observed_yaw: float, gripper_yaw: float) -> float:
    """Gripper yaw to execute a decided closing heading on a re-observed block yaw.

    theta = closing_heading + wrap_angle(observed_yaw - closing_heading, pi/2) is the face
    normal of the re-observed block nearest the decided heading, so it stays on the same
    face pair for any orientation change under pi/4. Of theta and theta + pi, the one
    nearest gripper_yaw is returned, so the wrist turns by at most pi/2.
    """
    theta = closing_heading + wrap_angle(observed_yaw - closing_heading, QUARTER)
    return gripper_yaw + wrap_angle(theta - gripper_yaw, np.pi)
