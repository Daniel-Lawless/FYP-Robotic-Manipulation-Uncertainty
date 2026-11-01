from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

ROBOT = frozenset({"fingers", "palm", "arm"})
GRASP_PHASES = ("open", "pregrasp", "descend", "close", "lift", "hold")
TRANSPORT_PHASES = ("carry", "lower")
STAGES = ("grasp", "transport", "release", "placement", "scene")
FLAGS = {
    "grasp_miss": "grasp",
    "grasp_timeout": "grasp",
    "dropped": "transport",
    "transport_timeout": "transport",
    "release_failure": "release",
    "placement_failure": "placement",
    "collision_object": "",
    "collision_environment": "",
    "displaced": "scene",
    "toppled": "scene",
    "knocked_off": "scene",
    "tilted": "",
    "leaning": "",
    "unresolved_pose": "",
    "unsettled": "",
    "com_outside_base": "",
}  # flag -> the stage it fails, or "" if it does not decide the outcome
# Contacts that can hold a block up other than the surface it stands on.
NOT_SUPPORTS = frozenset({"table", "floor"})


@dataclass(frozen=True)
class PlacementTolerances:
    """When a placed block counts as upright, supported by the table and at rest."""

    max_tilt: float = np.radians(2.0)  # vertical axis from world z
    max_support_gap: float = 0.001  # |lowest corner - table top|, metres
    max_linear_speed: float = 0.005  # m/s
    max_angular_speed: float = 0.05  # rad/s


@dataclass(frozen=True)
class PlacementCheck:
    """A block's state against the goal region, from the true simulator state."""

    position: np.ndarray  # world, metres
    quaternion: np.ndarray  # world orientation, (w, x, y, z)
    centre_error: float  # horizontal distance of the block centre from the goal centre
    goal_margin: float  # how far the outermost box corner is inside the goal circle
    tilt: float  # angle of the block's vertical axis from world z, radians
    support_gap: float  # lowest box corner above the table top (negative: sunk in)
    linear_speed: float  # m/s
    angular_speed: float  # rad/s
    contacts: tuple[str, ...]  # roles of everything touching the block (see geom_role)
    in_goal: bool  # every corner of the box lies over the goal circle
    upright: bool
    on_table: bool  # touching the table, resting on its lowest face
    at_rest: bool

    @property
    def touching_robot(self) -> bool:
        return bool(ROBOT & set(self.contacts))

    @property
    def stable(self) -> bool:
        return self.upright and self.on_table and self.at_rest


@dataclass(frozen=True)
class ContactEvent:
    """A contact pair during one phase.

    unintended follows pick_place.is_unintended; kind is "environment" when the pair
    includes the table, floor or robot mount and "object" otherwise.

    Two streams use this record (see pick_place.pick_and_place):
    - endpoint events (PickPlaceResult.unintended_contacts and endpoint_contacts) come
      from the contact list as it stands after each control step: steps counts the
      control steps whose endpoint showed the pair, and peak_force is the largest total
      normal force of the pair's contacts at one endpoint, N. first_substep and
      substeps are None.
    - historical events (OutcomeEvidence.contacts) come from every physics substep:
      steps counts the control intervals in which the pair touched in at least one
      substep, substeps the substeps in which it touched, and peak_force is the largest
      per-substep total normal force, N.
    """

    pair: str  # e.g. "palm-target"
    phase: str
    first_step: int  # control step of the whole sequence, from 1
    steps: int  # control steps (endpoint) or control intervals (historical) with the pair
    peak_force: float = 0.0
    unintended: bool = True
    kind: str = "object"
    first_substep: int | None = None  # physics substep of the whole sequence, from 1
    substeps: int | None = None  # physics substeps in which the pair was in contact


@dataclass(frozen=True)
class TableSurface:
    """The table top: height of its surface and its extent, centred on the world origin."""

    height: float
    half_x: float
    half_y: float


@dataclass(frozen=True)
class OutcomeThresholds:
    """disturbance_threshold applies to the maximum planar displacement of non-target
    blocks; significant_tilt marks a block as tilted; face_flat_tilt is how far a face
    normal may be from straight down for the block to rest flat on that face."""

    disturbance_threshold: float = 0.03  # metres
    significant_tilt: float = np.radians(5.0)
    face_flat_tilt: float = np.radians(5.0)


# Everything below is computed from explicit records, so it runs without MuJoCo and tests
# can supply the records directly. pick_place.pick_and_place builds the records from the
# true simulator state:
# - an ObjectRecord per block at the start of the attempt and after settling, plus the
#   block's pose after every 20 Hz control step (a peak between control steps is not
#   seen);
# - object_outcome turns them into an ObjectOutcome: displacement, tilt, resting faces and
#   the toppled / tilted / leaning / unresolved / unsettled / COM diagnostics;
# - outcome labels the whole attempt from an OutcomeEvidence: the binary outcome Y, the
#   earliest failure stage and non-exclusive flags.
@dataclass(frozen=True)
class ObjectRecord:
    """One block's true state at one moment."""

    name: str
    role: str
    size: np.ndarray  # full (x, y, z) extent of the box, metres
    position: np.ndarray  # world, centre of mass (the box centre)
    quaternion: np.ndarray  # world orientation, (w, x, y, z)
    linear_speed: float  # m/s
    angular_speed: float  # rad/s
    contacts: tuple[str, ...]  # roles of everything touching the block


@dataclass(frozen=True)
class ObjectOutcome:
    """What happened to one block over an attempt (see object_outcome)."""

    name: str
    role: str
    start_position: np.ndarray
    start_quaternion: np.ndarray
    final_position: np.ndarray
    final_quaternion: np.ndarray
    final_displacement: float  # planar distance from the start centre, metres
    max_displacement: float  # largest planar distance from the start centre at any record
    vertical_displacement: float  # final - start centre height, metres
    yaw_change: float  # radians, modulo the block's rotational symmetry about vertical
    off_table: bool
    at_rest: bool
    resting_face_start: str | None  # e.g. "-z": the body face resting flat on the table
    resting_face_final: str | None
    tilt_start: float  # original vertical body axis from world vertical, radians
    tilt_final: float
    tilt_max: float
    toppled: bool
    tilted: bool
    leaning: bool
    unresolved_pose: bool
    unsettled: bool
    com_outside_base: bool
    final_contacts: tuple[str, ...]


def quat_to_matrix(quaternion) -> np.ndarray:
    """Rotation matrix of a unit quaternion (w, x, y, z)."""
    w, x, y, z = np.asarray(quaternion, dtype=float) / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quaternion_tilts(quaternions, axis: int, sign: float) -> np.ndarray:
    """Angle between sign * body axis `axis` and world up, for each (w, x, y, z) row.

    The same arithmetic as quat_to_matrix's bottom row, element by element, so each
    value is bitwise equal to the scalar computation; vectorised for long trajectories.
    """
    q = np.asarray(quaternions, dtype=float).reshape(-1, 4)
    w, x, y, z = (q / np.linalg.norm(q, axis=1)[:, None]).T
    row = (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))[axis]
    return np.arccos(np.clip(sign * row, -1.0, 1.0))


def _wrap(angle: float, period: float) -> float:
    return float((angle + period / 2) % period - period / 2)


def _box_yaw(rot: np.ndarray) -> float:
    """primitives.box_yaw: heading of the body axis after the most vertical one."""
    side = rot[:, (int(np.argmax(np.abs(rot[2]))) + 1) % 3]
    return float(np.arctan2(side[1], side[0]))


def _face_corners(record: ObjectRecord, rot: np.ndarray, axis: int, sign: float) -> np.ndarray:
    """World corners of the face whose outward normal is sign * body axis, (4, 3)."""
    half = np.asarray(record.size, dtype=float) / 2
    i, j = [k for k in range(3) if k != axis]
    centre = record.position + sign * half[axis] * rot[:, axis]
    # Counter-clockwise in the face's (i, j) coordinates.
    signs = ((1, 1), (-1, 1), (-1, -1), (1, -1))
    return np.array([centre + a * half[i] * rot[:, i] + b * half[j] * rot[:, j] for a, b in signs])


def is_at_rest(record: ObjectRecord, tolerances: PlacementTolerances) -> bool:
    return (
        record.linear_speed <= tolerances.max_linear_speed
        and record.angular_speed <= tolerances.max_angular_speed
    )


def resting_face(
    record: ObjectRecord,
    table: TableSurface,
    thresholds: OutcomeThresholds | None = None,
    tolerances: PlacementTolerances | None = None,
) -> str | None:
    """The face a block rests flat on, as a signed body axis ("-z" is the bottom face of
    a block standing as built), or None.

    A face counts only if its outward normal is within face_flat_tilt of straight down,
    all four of its corners are within max_support_gap of the table top, the block
    touches the table and the block is at rest.
    """
    thresholds = thresholds or OutcomeThresholds()
    tolerances = tolerances or PlacementTolerances()
    if "table" not in record.contacts or not is_at_rest(record, tolerances):
        return None
    rot = quat_to_matrix(record.quaternion)
    for axis in range(3):
        for sign in (1.0, -1.0):
            if -sign * rot[2, axis] < np.cos(thresholds.face_flat_tilt):
                continue
            corners = _face_corners(record, rot, axis, sign)
            if np.all(np.abs(corners[:, 2] - table.height) <= tolerances.max_support_gap):
                return ("+" if sign > 0 else "-") + "xyz"[axis]
    return None


def initial_resting_face(
    record: ObjectRecord,
    table: TableSurface,
    thresholds: OutcomeThresholds | None = None,
    tolerances: PlacementTolerances | None = None,
) -> str:
    """resting_face of a block's start record, which must have one.

    Raises ValueError if it has none: an attempt that does not start with every block
    resting on a face is invalid input, neither a success nor a manipulation failure.
    """
    face = resting_face(record, table, thresholds, tolerances)
    if face is None:
        raise ValueError(
            f"{record.name}: initial resting face is unresolved (it must be at rest, touch "
            f"the table and lie flat on a face): contacts {record.contacts}, linear speed "
            f"{record.linear_speed:.4g} m/s, angular speed {record.angular_speed:.4g} rad/s"
        )
    return face


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Whether a point lies inside or on a convex polygon of either winding (2D).

    A degenerate polygon (a face seen edge-on) contains only points on its segment.
    """
    edges = np.roll(polygon, -1, axis=0) - polygon
    rel = point - polygon
    cross = edges[:, 0] * rel[:, 1] - edges[:, 1] * rel[:, 0]
    scale = max(np.abs(edges).max(), 1e-12) ** 2
    cross = np.where(np.abs(cross) <= 1e-12 * scale, 0.0, cross)
    if np.all(cross == 0.0):  # collinear: inside only on the segment
        lo, hi = polygon.min(axis=0), polygon.max(axis=0)
        return bool(np.all(point >= lo - 1e-12) and np.all(point <= hi + 1e-12))
    return bool(np.all(cross >= 0.0) or np.all(cross <= 0.0))


def up_axis(rot: np.ndarray) -> tuple[int, float]:
    """The body axis closest to world vertical, and the sign that makes it point up."""
    axis = int(np.argmax(np.abs(rot[2])))
    return axis, float(np.sign(rot[2, axis]) or 1.0)


def com_outside_base(record: ObjectRecord, base_axis: int, base_sign: float) -> bool:
    """Whether the centre of mass, projected straight down, falls outside the projection
    of the base face (the face with outward normal base_sign * body axis base_axis) at
    the block's current pose. A direction-aware tipping diagnostic, not a topple label.
    """
    rot = quat_to_matrix(record.quaternion)
    corners = _face_corners(record, rot, base_axis, base_sign)[:, :2]
    return not _point_in_polygon(np.asarray(record.position, dtype=float)[:2], corners)


def is_off_table(record: ObjectRecord, table: TableSurface) -> bool:
    """Centre below the table top, centre beyond the table's edge, or on the floor."""
    x, y, z = record.position
    return bool(
        z < table.height
        or abs(x) > table.half_x
        or abs(y) > table.half_y
        or "floor" in record.contacts
    )


# Toppling is a final-state classification, not a tilt threshold: a block has toppled
# when it ends resting flat on a different face from the one it started on. Tilt and
# centre-of-mass measures are diagnostics only. The definitions are provisional
# additions to proposal Section 3.1, documented in data/validation/README.md.
def object_outcome(
    start: ObjectRecord,
    final: ObjectRecord,
    positions: np.ndarray | None = None,
    quaternions: np.ndarray | None = None,
    table: TableSurface | None = None,
    thresholds: OutcomeThresholds | None = None,
    tolerances: PlacementTolerances | None = None,
) -> ObjectOutcome:
    """One block's outcome from its start record, its final record and, optionally, its
    recorded poses in between (positions (n, 3), quaternions (n, 4) in w, x, y, z).

    start must describe the beginning of the attempt: the original vertical body axis,
    the original base face and the starting resting face all come from it, so a block
    moved onto its side afterwards is judged against how it stood at the start.
    Maximum displacement and maximum tilt are taken over the start, every recorded pose
    and the final record. Raises ValueError if the start has no resting face (see
    initial_resting_face); the final record may have none.
    """
    if table is None:
        raise ValueError("table is required")
    thresholds = thresholds or OutcomeThresholds()
    tolerances = tolerances or PlacementTolerances()
    face_start = initial_resting_face(start, table, thresholds, tolerances)
    start_pos = np.asarray(start.position, dtype=float)
    final_pos = np.asarray(final.position, dtype=float)
    pos = [start_pos[None], final_pos[None]]
    quats = [np.asarray(start.quaternion)[None], np.asarray(final.quaternion)[None]]
    if positions is not None and len(positions):
        pos.insert(1, np.asarray(positions, dtype=float).reshape(-1, 3))
    if quaternions is not None and len(quaternions):
        quats.insert(1, np.asarray(quaternions, dtype=float).reshape(-1, 4))
    planar = np.linalg.norm(np.concatenate(pos)[:, :2] - start_pos[:2], axis=1)

    start_rot, final_rot = quat_to_matrix(start.quaternion), quat_to_matrix(final.quaternion)
    axis, sign = up_axis(start_rot)
    tilt_start, tilt_final = (
        float(np.arccos(np.clip(sign * rot[2, axis], -1.0, 1.0))) for rot in (start_rot, final_rot)
    )
    tilt_max = max(tilt_start, tilt_final)
    if len(quats) > 2:  # the recorded poses between the start and the final record
        tilt_max = max(tilt_max, float(quaternion_tilts(quats[1], axis, sign).max()))
    face_final = resting_face(final, table, thresholds, tolerances)
    toppled = face_final is not None and face_final != face_start
    leaning = face_final is None and bool(set(final.contacts) - NOT_SUPPORTS)
    at_rest = is_at_rest(final, tolerances)
    size = np.asarray(start.size, dtype=float)
    symmetry = np.pi / 2 if np.isclose(size[0], size[1]) else np.pi
    return ObjectOutcome(
        name=final.name,
        role=final.role,
        start_position=start_pos,
        start_quaternion=np.asarray(start.quaternion, dtype=float),
        final_position=final_pos,
        final_quaternion=np.asarray(final.quaternion, dtype=float),
        final_displacement=float(planar[-1]),
        max_displacement=float(planar.max()),
        vertical_displacement=float(final_pos[2] - start_pos[2]),
        yaw_change=_wrap(_box_yaw(final_rot) - _box_yaw(start_rot), symmetry),
        off_table=is_off_table(final, table),
        at_rest=at_rest,
        resting_face_start=face_start,
        resting_face_final=face_final,
        tilt_start=tilt_start,
        tilt_final=tilt_final,
        tilt_max=tilt_max,
        toppled=toppled,
        tilted=tilt_final > thresholds.significant_tilt and not toppled,
        leaning=leaning,
        unresolved_pose=face_final is None and not leaning,
        unsettled=not at_rest,
        com_outside_base=com_outside_base(final, axis, -sign),
        final_contacts=tuple(final.contacts),
    )


@dataclass(frozen=True)
class OutcomeEvidence:
    """What one pick-and-place attempt observed, as pick_and_place recorded it.

    grasp and placement are the primitive's own GraspResult and PlacementCheck; the
    label reads only the fields below them that it needs, so it can be tested without
    MuJoCo.

    released, dropped and placement are endpoint measurements (the state after a control
    step or after settling); contacts and the objects' maxima can come from every
    physics substep (see ContactEvent and pick_place.pick_and_place).
    """

    target: str
    grasp: Any  # primitives.GraspResult
    grasp_success: bool
    phases: tuple[tuple[str, bool], ...]  # (name, reached), in order
    abort: str  # pick_and_place's reason for skipping the remaining movements, or ""
    gripper_empty: bool  # the fingers closed on nothing after the grasp
    dropped: bool
    drop_phase: str  # phase in which the block first left the grip, or ""
    released: bool
    placement: PlacementCheck  # the target after settling
    objects: tuple[ObjectOutcome, ...]
    contacts: tuple[ContactEvent, ...]  # every pair, per phase, in order of first contact
    table: TableSurface

    def object(self, name: str) -> ObjectOutcome:
        return next(o for o in self.objects if o.name == name)


@dataclass(frozen=True)
class OutcomeLabel:
    """y = 1 for a successful attempt with an undisturbed scene; failure_stage is the
    earliest stage that failed ("" on success). flags maps each raised flag (see FLAGS)
    to what raised it: object names, or "pair@phase" for contact flags.
    """

    y: int
    failure_stage: str
    flags: dict[str, tuple[str, ...]]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.flags)


def target_placed(placement: PlacementCheck, table_height: float) -> bool:
    """pick_and_place's placement checks on the target after settling."""
    return (
        placement.position[2] > table_height
        and placement.in_goal
        and placement.upright
        and placement.on_table
        and placement.at_rest
        and not placement.touching_robot
    )


def outcome(evidence: OutcomeEvidence, thresholds: OutcomeThresholds | None = None) -> OutcomeLabel:
    """Label an attempt from its evidence.

    y = 1 needs: a verified grasp with the gripper not empty, no grasp or transport
    timeout, no drop, the target released and the retract completed, the target validly
    placed (pick_and_place's checks), no block off the table, no non-target block whose
    maximum planar displacement exceeds disturbance_threshold, and no toppled block.
    Contact and diagnostic flags are recorded but do not decide y. The tilt and face
    thresholds act through object_outcome, which computed evidence.objects.

    Raises ValueError if any object has no starting resting face: the attempt had
    invalid initial evidence and gets no label.
    """
    for obj in evidence.objects:
        if obj.resting_face_start is None:
            raise ValueError(
                f"{obj.name}: resting_face_start is None; an attempt with an unresolved "
                "initial resting face cannot be labelled"
            )
    thresholds = thresholds or OutcomeThresholds()
    reached = dict(evidence.phases)
    ran = [name for name, _ in evidence.phases]
    raised: dict[str, list[str]] = {}

    def flag(name: str, what: str) -> None:
        raised.setdefault(name, []).append(what)

    if not evidence.grasp_success or evidence.gripper_empty:
        flag("grasp_miss", evidence.target)
    for phase in GRASP_PHASES:
        if phase in reached and not reached[phase]:
            flag("grasp_timeout", phase)
    if evidence.dropped:
        flag("dropped", evidence.drop_phase)
    for phase in TRANSPORT_PHASES:
        if phase in reached and not reached[phase]:
            flag("transport_timeout", phase)
    if "release" in ran and not evidence.released:
        flag("release_failure", "release")
    if "retract" in reached and not reached["retract"]:
        flag("release_failure", "retract")
    if not target_placed(evidence.placement, evidence.table.height):
        flag("placement_failure", evidence.target)

    for event in evidence.contacts:
        if event.unintended:
            kind = "collision_environment" if event.kind == "environment" else "collision_object"
            flag(kind, f"{event.pair}@{event.phase}")

    for obj in evidence.objects:
        if obj.name != evidence.target and obj.max_displacement > thresholds.disturbance_threshold:
            flag("displaced", obj.name)
        for name in ("toppled", "tilted", "leaning", "unresolved_pose", "unsettled"):
            if getattr(obj, name):
                flag(name, obj.name)
        if obj.off_table:
            flag("knocked_off", obj.name)
        if obj.com_outside_base:
            flag("com_outside_base", obj.name)

    flags = {name: tuple(raised[name]) for name in FLAGS if name in raised}
    failed = {FLAGS[name] for name in flags} - {""}
    stage = next((s for s in STAGES if s in failed), "")
    return OutcomeLabel(y=int(not stage), failure_stage=stage, flags=flags)
