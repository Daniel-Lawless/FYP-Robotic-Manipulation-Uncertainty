from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import new_site

from robot_uncertainty.sim.primitives import (
    GraspResult,
    box_yaw,
    heading,
    top_grasp,
    top_grasp_height,
)
from robot_uncertainty.sim.scene import (
    Placement,
    TabletopConfig,
    cross_section_symmetry,
    footprint_circle_distance,
    footprint_corners,
    footprints_overlap,
    sample_layout,
)

# Limits for a valid initial scene after settling (check_initial_state); the speeds are
# the at-rest tolerances of the placement check.
INITIAL_MAX_LINEAR_SPEED = 0.005  # m/s
INITIAL_MAX_ANGULAR_SPEED = 0.05  # rad/s
INITIAL_MAX_BLOCK_PENETRATION = 0.0001  # m, between two blocks
INITIAL_MAX_TILT = np.radians(2.0)


@dataclass(frozen=True)
class ObjectObservation:
    name: str
    position: np.ndarray  # world (x, y, z) of the block's centre, metres
    yaw: float  # heading of a side-face normal (see primitives.box_yaw), radians
    size: np.ndarray  # full (x, y, z) extent, metres
    role: str  # BlockSpec.role, e.g. "target" or "distractor"


@dataclass(frozen=True)
class Observation:
    """The robot's view of the scene (exact for now)."""

    objects: tuple[ObjectObservation, ...]
    goal_position: np.ndarray  # world (x, y, z) of the goal centre on the table
    goal_radius: float
    eef_position: np.ndarray  # grip site, world frame
    eef_yaw: float  # heading of the finger-closing axis
    gripper_width: float  # distance between the fingers, metres
    joint_positions: np.ndarray  # the arm's 7 joints, radians

    def object(self, name: str) -> ObjectObservation:
        return next(o for o in self.objects if o.name == name)


@dataclass(frozen=True)
class ObjectState:
    name: str
    position: np.ndarray  # world, metres
    quaternion: np.ndarray  # world orientation, (w, x, y, z)
    linear_velocity: np.ndarray  # world, m/s
    angular_velocity: np.ndarray  # world, rad/s
    size: np.ndarray


@dataclass(frozen=True)
class TrueState:
    """Exact simulator state, for validation and outcome labels only."""

    scene_seed: int
    layout: dict[str, Placement]  # what the generator drew, relative to the table
    objects: tuple[ObjectState, ...]
    eef_position: np.ndarray
    eef_rotation: np.ndarray
    finger_gap: float
    joint_positions: np.ndarray
    time: float

    def object(self, name: str) -> ObjectState:
        return next(o for o in self.objects if o.name == name)


class TabletopEnv(ManipulationEnv):
    """A Panda under OSC_POSE at a table with configurable blocks and a goal region.

    Resets never rebuild the model (robosuite's hard_reset is off): block sizes and
    physical parameters are fixed by the config, and only their poses change. The
    robot starts every episode at its exact home configuration (no joint noise), the
    canonical decision state of the proposal (Section 2.1).
    """

    def __init__(self, config: TabletopConfig | None = None, seed: int = 0, **kwargs) -> None:
        self.config = config or TabletopConfig()
        self.table_offset = np.array([0.0, 0.0, self.config.table_height])
        # Scene seeds for resets that do not name one, drawn from the env's own seed.
        self._scene_seeds = np.random.default_rng(seed)
        self.scene_seed = int(self._scene_seeds.integers(2**31))
        self.layout: dict[str, Placement] = {}

        controller_config = load_composite_controller_config(robot="Panda")
        arm_type = controller_config["body_parts"]["right"]["type"]
        assert arm_type == "OSC_POSE", f"expected OSC_POSE arm controller, got {arm_type}"
        settings = dict(
            robots="Panda",
            controller_configs=controller_config,
            initialization_noise=None,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            hard_reset=False,
            ignore_done=True,
            seed=seed,
        )
        super().__init__(**{**settings, **kwargs})

    def reset(self, scene_seed: int | None = None):
        """Start a new scene; scene_seed alone determines its layout."""
        if scene_seed is None:
            scene_seed = int(self._scene_seeds.integers(2**31))
        self.scene_seed = int(scene_seed)
        return super().reset()

    # --- model construction --------------------------------------------------------

    def _load_model(self) -> None:
        super()._load_model()
        cfg = self.config
        robot_model = self.robots[0].robot_model
        robot_model.set_base_xpos(robot_model.base_xpos_offset["table"](cfg.table_size[0]))

        arena = TableArena(
            table_full_size=cfg.table_size,
            table_friction=cfg.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])
        # The goal is a flat visual disc on the table top (sites never collide), placed
        # in the table body's frame, whose origin is the middle of the table slab.
        arena.table_body.append(
            new_site(
                "goal",
                rgba=(0.2, 0.7, 0.3, 0.4),
                pos=(*cfg.goal.center, cfg.table_size[2] / 2),
                size=(cfg.goal.radius, 0.0005),
                type="cylinder",
            )
        )

        self.blocks = {
            spec.name: BoxObject(
                name=spec.name,
                size=list(spec.half_size),
                density=spec.density,
                friction=list(spec.friction),
                rgba=list(spec.rgba),
            )
            for spec in cfg.blocks
        }
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot_model],
            mujoco_objects=list(self.blocks.values()),
        )

    def _setup_references(self) -> None:
        super()._setup_references()
        self.block_body_ids = {
            name: self.sim.model.body_name2id(obj.root_body) for name, obj in self.blocks.items()
        }
        self.goal_site_id = self.sim.model.site_name2id("goal")

    def _reset_internal(self) -> None:
        super()._reset_internal()
        self.layout = sample_layout(self.config, self.scene_seed)
        for spec in self.config.blocks:
            self.set_block_pose(spec.name, self.layout[spec.name])

    def set_block_pose(self, name: str, placement: Placement) -> None:
        """Stand a block upright, at rest, on the table at a planar pose."""
        spec = next(b for b in self.config.blocks if b.name == name)
        pos = self.table_offset + [placement.x, placement.y, spec.half_size[2]]
        quat = [np.cos(placement.yaw / 2), 0.0, 0.0, np.sin(placement.yaw / 2)]  # wxyz
        joint = self.blocks[name].joints[0]
        self.sim.data.set_joint_qpos(joint, np.concatenate([pos, quat]))
        self.sim.data.set_joint_qvel(joint, np.zeros(6))
        self.sim.forward()

    def set_layout(self, placements: dict[str, Placement]) -> None:
        """Place blocks explicitly (diagnostic scenes), upright and at rest.

        Blocks not named keep their current placement in self.layout. The resulting
        layout is checked first, exactly (oriented rectangles, see scene.footprint_distance):
        every footprint on the table, none overlapping another (touching is allowed),
        none intersecting the goal circle. Raises ValueError and changes nothing if a
        check fails; nothing is adjusted. Settle afterwards, then check_initial_state.
        """
        unknown = set(placements) - set(self.blocks)
        if unknown:
            raise ValueError(f"no such blocks in the scene: {sorted(unknown)}")
        layout = {**self.layout, **placements}
        specs = {b.name: b for b in self.config.blocks}
        corners = {n: footprint_corners(specs[n], p) for n, p in layout.items()}
        half = np.asarray(self.config.table_size[:2]) / 2
        goal = self.config.goal
        problems = []
        for n, c in corners.items():
            if np.any(np.abs(c) > half):
                problems.append(f"{n} extends beyond the table")
            if footprint_circle_distance(c, goal.center, goal.radius) <= 0.0:
                problems.append(f"{n} intersects the goal")
        names = list(corners)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                if footprints_overlap(corners[a], corners[b]):
                    problems.append(f"{a} overlaps {b}")
        if problems:
            raise ValueError("invalid layout: " + "; ".join(problems))
        for n, p in placements.items():
            self.set_block_pose(n, p)
        self.layout = layout

    def check_initial_state(self) -> list[str]:
        """Problems that make the current state an invalid start for an attempt; empty
        if valid: every block at rest (INITIAL_MAX_*_SPEED), upright within
        INITIAL_MAX_TILT, and no two blocks interpenetrating by more than
        INITIAL_MAX_BLOCK_PENETRATION. Call after settling.
        """
        model, data = self.sim.model._model, self.sim.data._data
        problems = []
        for obj in self.true_state().objects:
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, obj.quaternion)
            tilt = float(np.arccos(np.clip(rot[8], -1.0, 1.0)))
            if np.linalg.norm(obj.linear_velocity) > INITIAL_MAX_LINEAR_SPEED:
                problems.append(
                    f"{obj.name} moving at {np.linalg.norm(obj.linear_velocity):.4f} m/s"
                )
            if np.linalg.norm(obj.angular_velocity) > INITIAL_MAX_ANGULAR_SPEED:
                problems.append(
                    f"{obj.name} turning at {np.linalg.norm(obj.angular_velocity):.4f} rad/s"
                )
            if tilt > INITIAL_MAX_TILT:
                problems.append(f"{obj.name} tilted {np.degrees(tilt):.2f} deg")
        owner = {model.geom(g).id: n for n, obj in self.blocks.items() for g in obj.contact_geoms}
        for contact in data.contact[: data.ncon]:
            a, b = owner.get(contact.geom1), owner.get(contact.geom2)
            if a and b and a != b and -contact.dist > INITIAL_MAX_BLOCK_PENETRATION:
                problems.append(f"{a} and {b} interpenetrate by {-1000 * contact.dist:.3f} mm")
        return problems

    # --- scene queries -------------------------------------------------------------

    def block_body(self, name: str) -> str:
        return self.blocks[name].root_body

    @property
    def goal_position(self) -> np.ndarray:
        return self.sim.data.site_xpos[self.goal_site_id].copy()

    def _robot_state(self) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        robot = self.robots[0]
        data = self.sim.data
        site = robot.eef_site_id["right"]
        fingers = data.qpos[robot._ref_gripper_joint_pos_indexes["right"]]
        return (
            data.site_xpos[site].copy(),
            data.site_xmat[site].reshape(3, 3).copy(),
            float(fingers[0] - fingers[1]),
            data.qpos[robot._ref_joint_pos_indexes].copy(),
        )

    def observe(self) -> Observation:
        """The robot's observation of the scene: exact, without noise, for now."""
        data = self.sim.data
        objects = tuple(
            ObjectObservation(
                name=spec.name,
                position=data.body_xpos[self.block_body_ids[spec.name]].copy(),
                yaw=box_yaw(data.body_xmat[self.block_body_ids[spec.name]].reshape(3, 3)),
                size=np.asarray(spec.size, dtype=float),
                role=spec.role,
            )
            for spec in self.config.blocks
        )
        eef_pos, eef_rot, gap, joints = self._robot_state()
        return Observation(
            objects=objects,
            goal_position=self.goal_position,
            goal_radius=self.config.goal.radius,
            eef_position=eef_pos,
            eef_yaw=heading(eef_rot),
            gripper_width=gap,
            joint_positions=joints,
        )

    def true_state(self) -> TrueState:
        """Exact simulator state, read from MuJoCo directly."""
        model, data = self.sim.model._model, self.sim.data._data
        objects = []
        for spec in self.config.blocks:
            body = self.block_body_ids[spec.name]
            velocity = np.zeros(6)  # (angular, linear) in the world frame
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
            objects.append(
                ObjectState(
                    name=spec.name,
                    position=data.xpos[body].copy(),
                    quaternion=data.xquat[body].copy(),
                    linear_velocity=velocity[3:],
                    angular_velocity=velocity[:3],
                    size=np.asarray(spec.size, dtype=float),
                )
            )
        eef_pos, eef_rot, gap, joints = self._robot_state()
        return TrueState(
            scene_seed=self.scene_seed,
            layout=dict(self.layout),
            objects=tuple(objects),
            eef_position=eef_pos,
            eef_rotation=eef_rot,
            finger_gap=gap,
            joint_positions=joints,
            time=float(data.time),
        )

    def reward(self, action=None) -> float:
        return 0.0  # tasks and outcome labels come with the full primitive (M1)


def grasp_block_from_above(
    env: TabletopEnv,
    name: str,
    grasp_height: float | None = None,
    lift_height: float = 0.10,
    closing_heading: float | None = None,
) -> GraspResult:
    """Observe a block and run the scripted top grasp on it, planned from the observation.

    The grasp height comes from the observed block height (top_grasp_height) unless
    given; lift_height is how far the grasp point is raised; closing_heading, if given,
    is a decided finger-closing heading (see top_grasp). Success is still judged from
    the true state inside top_grasp.
    """
    target = env.observe().object(name)
    if grasp_height is None:
        grasp_height = top_grasp_height(target.size[2])
    return top_grasp(
        env,
        target.position,
        target.yaw,
        env.block_body(name),
        symmetry=cross_section_symmetry(target.size),
        grasp_height=grasp_height,
        lift_height=lift_height,
        closing_heading=closing_heading,
    )


def make_tabletop_env(config: TabletopConfig | None = None, seed: int = 0, **kwargs) -> TabletopEnv:
    """A headless TabletopEnv; kwargs override robosuite settings, e.g. to add a viewer."""
    return TabletopEnv(config, seed=seed, **kwargs)


def hand_footprint_half_extent(env: TabletopEnv) -> tuple[float, float]:
    """Half-extents of the hand's top-down footprint in the grip-site frame, metres:
    along the finger-closing axis (site x) and across it (site y), over the hand's and
    fingers' collision meshes at the current finger opening (about 104 x 32 mm for the
    robosuite 1.5.2 Panda hand, set by the palm; open fingers reach about 66 mm).
    """
    model, data = env.sim.model._model, env.sim.data._data
    site = env.robots[0].eef_site_id["right"]
    origin, rot = data.site_xpos[site], data.site_xmat[site].reshape(3, 3)
    extent = np.zeros(2)
    for g in range(model.ngeom):
        collides = model.geom_contype[g] or model.geom_conaffinity[g]
        if not (model.geom(g).name.startswith("gripper0_") and collides):
            continue
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[g]
            first = model.mesh_vertadr[mesh]
            verts = model.mesh_vert[first : first + model.mesh_vertnum[mesh]]
        else:  # boxes (the finger pads): their eight corners
            signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
            verts = signs * model.geom_size[g]
        world = verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        extent = np.maximum(extent, np.abs((world - origin) @ rot)[:, :2].max(axis=0))
    return float(extent[0]), float(extent[1])
