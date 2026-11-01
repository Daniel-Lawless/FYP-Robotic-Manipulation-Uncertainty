from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Stream identifiers for np.random.SeedSequence, so each source of randomness in the
# proposal (Section 2.1) gets its own independent stream from one scene seed. Only the
# layout stream is used so far; adding streams later does not change existing layouts.
LAYOUT_STREAM = 0


def stream_rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(stream,)))


def cross_section_symmetry(size) -> float:
    """Rotational symmetry of an upright box about vertical: pi/2 if square, else pi."""
    return np.pi / 2 if np.isclose(size[0], size[1]) else np.pi


@dataclass(frozen=True)
class BlockSpec:
    """A rigid box. size is the full (x, y, z) extent in metres, z vertical at rest."""

    name: str
    size: tuple[float, float, float] = (0.04, 0.04, 0.08)
    role: str = "target"
    density: float = 1000.0  # kg/m^3 (robosuite's default); 128 g for 4 x 4 x 8 cm
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)  # sliding, torsion, rolling
    rgba: tuple[float, float, float, float] = (0.85, 0.2, 0.15, 1.0)

    @property
    def half_size(self) -> np.ndarray:
        return np.asarray(self.size) / 2

    @property
    def footprint_radius(self) -> float:
        """Radius of the circle enclosing the block's footprint at any yaw."""
        return float(np.hypot(*self.half_size[:2]))

    @property
    def symmetry(self) -> float:
        return cross_section_symmetry(self.size)


# Distractors for explicit diagnostic scenes (M1.3a), with the target's default density
# and friction; dataclasses.replace(preset, name=...) gives further copies. Hypotheses to
# check against contact data, not guaranteed effects: the short one (6 cm) reaches into
# the fingers' corridor beside a gripped 8 cm target but stays under the palm, whose
# underside is about 8.6 cm above the table at the raised grasp point; the tall one
# (11 cm) reaches the palm and the upper hand.
DISTRACTOR_SHORT = BlockSpec(
    "distractor", size=(0.05, 0.05, 0.06), role="distractor", rgba=(0.2, 0.45, 0.85, 1.0)
)
DISTRACTOR_TALL = BlockSpec(
    "distractor", size=(0.05, 0.05, 0.11), role="distractor", rgba=(0.9, 0.75, 0.15, 1.0)
)


@dataclass(frozen=True)
class Region:
    """Axis-aligned rectangle on the table: x_range and y_range are (min, max)."""

    x_range: tuple[float, float]
    y_range: tuple[float, float]

    def contains(self, xy: np.ndarray, margin: float = 0.0) -> bool:
        x, y = xy
        return bool(
            self.x_range[0] + margin <= x <= self.x_range[1] - margin
            and self.y_range[0] + margin <= y <= self.y_range[1] - margin
        )

    def distance_to(self, xy: np.ndarray) -> float:
        """Distance from a point to the rectangle (zero inside)."""
        dx = max(self.x_range[0] - xy[0], 0.0, xy[0] - self.x_range[1])
        dy = max(self.y_range[0] - xy[1], 0.0, xy[1] - self.y_range[1])
        return float(np.hypot(dx, dy))


@dataclass(frozen=True)
class GoalRegion:
    """Circular placement goal on the table, marked visually but not physical.

    Reachability (checked in M1.1): the Panda reaches collision-free poses over the
    whole goal with the 8 cm block held, its bottom 2-5 mm above the table, at least
    51 mm between robot and table and 26 deg from every joint limit. But a 4 x 4 cm
    footprint fits inside this circle only with its centre within 21.7 mm of the goal
    centre, and the arm stops within its 5 mm positioning tolerance, so a block aimed
    near that boundary can end with part of its footprint outside (13 of 24 trials).
    Placement-success verification (M1.2) must account for this.
    """

    center: tuple[float, float] = (0.0, 0.18)
    radius: float = 0.05

    def contains(self, xy: np.ndarray) -> bool:
        return bool(np.hypot(*(np.asarray(xy) - self.center)) <= self.radius)


@dataclass(frozen=True)
class TabletopConfig:
    """Everything that defines a scene distribution; the defaults are the M1.1 scene."""

    table_size: tuple[float, float, float] = (0.6, 0.6, 0.05)  # full x, y, thickness
    table_height: float = 0.8  # world z of the table's top surface
    table_friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    # Where object footprints may be placed. Chosen to be reachable for a top grasp:
    # 36-56 cm in front of the Panda's base, which sits 16 cm behind the table edge.
    workspace: Region = Region(x_range=(-0.10, 0.10), y_range=(-0.20, 0.06))
    yaw_range: tuple[float, float] = (-np.pi, np.pi)
    goal: GoalRegion = field(default_factory=GoalRegion)
    # Free space kept between the goal circle and any footprint, and between footprints.
    clearance: float = 0.02
    blocks: tuple[BlockSpec, ...] = (BlockSpec("target"),)
    max_placement_tries: int = 1000

    def __post_init__(self) -> None:
        half_x, half_y = self.table_size[0] / 2, self.table_size[1] / 2
        table = Region((-half_x, half_x), (-half_y, half_y))
        corners = [(x, y) for x in self.workspace.x_range for y in self.workspace.y_range]
        if not all(table.contains(c) for c in corners):
            raise ValueError(f"workspace {self.workspace} extends beyond the table")
        if not table.contains(self.goal.center, margin=self.goal.radius):
            raise ValueError(f"goal {self.goal} extends beyond the table")
        # Every footprint lies inside the workspace, so this keeps the goal always clear.
        if self.workspace.distance_to(self.goal.center) < self.goal.radius + self.clearance:
            raise ValueError("goal region must lie outside the workspace, with clearance")
        names = [b.name for b in self.blocks]
        if len(set(names)) != len(names):
            raise ValueError(f"block names must be unique: {names}")


@dataclass(frozen=True)
class Placement:
    """Planar pose of one block: centre (x, y) on the table and yaw in radians."""

    x: float
    y: float
    yaw: float

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])


def footprint_corners(block: BlockSpec, placement: Placement) -> np.ndarray:
    """The four (x, y) corners of a block's footprint, shape (4, 2)."""
    hx, hy = block.half_size[:2]
    local = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
    c, s = np.cos(placement.yaw), np.sin(placement.yaw)
    return local @ np.array([[c, s], [-s, c]]) + placement.xy


def _edge_normals(polygon: np.ndarray) -> np.ndarray:
    edges = np.roll(polygon, -1, axis=0) - polygon
    normals = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


def _edges(polygon: np.ndarray):
    return zip(polygon, np.roll(polygon, -1, axis=0), strict=True)


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def footprint_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Signed distance between two convex footprints given as (n, 2) vertex arrays.

    Positive: the shortest distance between them (exact). Zero: touching. Negative:
    they overlap, by minus the smallest separating-axis penetration depth (a lower bound
    on how far one must move to separate them). The separating-axis test checks the edge
    normals of both polygons, which is exact for convex polygons such as rotated
    rectangles.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    gaps = []
    for normal in np.concatenate([_edge_normals(a), _edge_normals(b)]):
        pa, pb = a @ normal, b @ normal
        gaps.append(max(pb.min() - pa.max(), pa.min() - pb.max()))
    if max(gaps) < 0.0:
        return float(max(gaps))
    # Separated (or touching): the closest points include a vertex of one polygon.
    return min(
        min(_point_segment_distance(p, *e) for p in a for e in _edges(b)),
        min(_point_segment_distance(p, *e) for p in b for e in _edges(a)),
    )


# Distances below this count as touching, so rounding in rotated corners cannot turn
# touching footprints into overlapping ones (or a requested gap into a violation).
FOOTPRINT_TOLERANCE = 1e-9


def footprints_overlap(a: np.ndarray, b: np.ndarray, clearance: float = 0.0) -> bool:
    """Whether two convex footprints ((n, 2) vertices, e.g. footprint_corners) come
    closer than clearance. Touching footprints do not overlap at zero clearance.

    Exact for oriented rectangles (separating-axis test, then the true distance), unlike
    the enclosing-circle check of the random sampler; meant for explicit layouts.
    """
    return footprint_distance(a, b) < clearance - FOOTPRINT_TOLERANCE


def footprint_circle_distance(polygon: np.ndarray, center, radius: float) -> float:
    """Distance from a convex footprint ((n, 2) vertices in either winding) to a disc;
    zero or less where they intersect (-radius if the disc's centre is inside)."""
    polygon, center = np.asarray(polygon, dtype=float), np.asarray(center, dtype=float)
    sides = np.einsum("ij,ij->i", center - polygon, _edge_normals(polygon))
    if np.all(sides >= 0.0) or np.all(sides <= 0.0):
        return -radius
    return min(_point_segment_distance(center, *e) for e in _edges(polygon)) - radius


def place_beside(
    target_placement: Placement,
    target_spec: BlockSpec,
    other_spec: BlockSpec,
    gap: float,
    direction_heading: float,
    other_yaw: float,
) -> Placement:
    """Placement of other_spec beside a target, gap metres away along a heading.

    The other block's centre lies on the ray from the target's centre at
    direction_heading, far enough that the two footprints, projected onto that
    direction, are gap apart. When the facing sides are perpendicular to the direction
    (e.g. direction_heading a face normal of the target and other_yaw equal to the
    target's yaw, modulo pi/2), that is the face-to-face gap and the true distance
    between the footprints; otherwise the true distance is at least gap.
    """
    u = np.array([np.cos(direction_heading), np.sin(direction_heading)])
    reach_target = (
        footprint_corners(target_spec, Placement(0.0, 0.0, target_placement.yaw)) @ u
    ).max()
    reach_other = -(footprint_corners(other_spec, Placement(0.0, 0.0, other_yaw)) @ u).min()
    x, y = target_placement.xy + (reach_target + gap + reach_other) * u
    return Placement(float(x), float(y), float(other_yaw))


def placement_is_valid(
    config: TabletopConfig,
    block: BlockSpec,
    placement: Placement,
    placed: dict[str, Placement],
) -> bool:
    """Footprint inside the workspace and clear of every block already placed."""
    if not all(config.workspace.contains(c) for c in footprint_corners(block, placement)):
        return False
    specs = {b.name: b for b in config.blocks}
    for name, other in placed.items():
        # Enclosing circles: conservative, and exact enough until clutter (M2).
        min_gap = block.footprint_radius + specs[name].footprint_radius + config.clearance
        if np.hypot(*(placement.xy - other.xy)) < min_gap:
            return False
    return True


def sample_layout(config: TabletopConfig, scene_seed: int) -> dict[str, Placement]:
    """Draw a valid placement for every block from the scene seed's layout stream.

    Positions are uniform over the workspace and yaw uniform over yaw_range, rejecting
    footprints that leave the workspace or touch an earlier block. The same config and
    seed always give the same layout.
    """
    rng = stream_rng(scene_seed, LAYOUT_STREAM)
    layout: dict[str, Placement] = {}
    for block in config.blocks:
        for _ in range(config.max_placement_tries):
            candidate = Placement(
                x=float(rng.uniform(*config.workspace.x_range)),
                y=float(rng.uniform(*config.workspace.y_range)),
                yaw=float(rng.uniform(*config.yaw_range)),
            )
            if placement_is_valid(config, block, candidate, layout):
                layout[block.name] = candidate
                break
        else:
            raise RuntimeError(
                f"no valid placement for {block.name!r} after {config.max_placement_tries} tries"
            )
    return layout
