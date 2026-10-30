from __future__ import annotations

import time
from dataclasses import dataclass

import mujoco
import numpy as np

from robot_uncertainty.sim.primitives import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    GraspResult,
    OSCArm,
    PhaseResult,
    top_down_rotation,
)
from robot_uncertainty.sim.scene import BlockSpec
from robot_uncertainty.sim.tabletop import TabletopEnv, grasp_block_from_above

# Distance the Panda's fingertips (the finger collision meshes) reach beyond the grip
# site along the approach axis: 9.2 mm, measured from the robosuite 1.5.2 model like
# primitives.PALM_DEPTH, and checked by tests/sim/test_pick_place.py.
FINGERTIP_DEPTH = 0.0092
# A block is held while its grasp point stays this close to the grip site (the same
# criterion as top_grasp's success check).
HELD_RADIUS = 0.02

ROBOT = frozenset({"fingers", "palm", "arm"})
GRASP_PHASES = ("open", "pregrasp", "descend", "close", "lift", "hold")
# Phases in which the fingers are meant to touch the target, and in which the target
# is meant to be off the table.
GRIPPING = frozenset({"close", "lift", "hold", "carry", "lower", "release"})
AIRBORNE = frozenset({"hold", "carry", "lower"})


@dataclass(frozen=True)
class PlaceParams:
    """Heights are clearances between surfaces; pick_and_place turns them into grip-site
    targets using the observed block size and its observed position in the grip.

    Speeds cap the commanded grip-site step per 20 Hz control step; tolerances end a
    movement once the grip site is that close to its target; step counts are timeouts.
    """

    carry_clearance: float = 0.10  # block bottom above the table while carried (lift 10 cm)
    # Block bottom above the table when the fingers open: the lowering tolerance plus
    # 1 mm for the block's tilt in the grip (under 1.5 deg, about 0.5 mm at a corner).
    release_gap: float = 0.003
    retract_clearance: float = 0.05  # fingertips above the placed block's top
    carry_speed: float = 0.03  # m per step
    lower_speed: float = 0.01
    carry_tol: float = 0.01
    place_tol: float = 0.002
    carry_steps: int = 150
    lower_steps: int = 100
    release_steps: int = 10
    retract_steps: int = 60
    settle_steps: int = 5  # 0.25 s, the proposal's settle time


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
    """An unintended contact pair during one phase (see is_unintended)."""

    pair: str  # e.g. "palm-target"
    phase: str
    first_step: int  # control step of the whole sequence, from 1
    steps: int  # control steps on which the pair was in contact


@dataclass
class PickPlaceResult:
    """Outcome of one pick-and-place attempt.

    success is the task outcome from the true state after settling; contacts that
    should not happen are listed separately in unintended_contacts.
    """

    success: bool
    failure_reason: str  # "" on success; otherwise the first check that failed
    grasp: GraspResult
    phases: list[PhaseResult]  # the grasp's phases, then carry, lower, release, ...
    released: bool  # the block was free of the robot at the end of the release phase
    dropped: bool  # the block left the grip while being carried or lowered
    placement: PlacementCheck  # after settling
    max_slip: float  # metres the block shifted in the grip while carried and lowered
    max_penetration: float  # deepest contact penetration of the whole sequence, metres
    unintended_contacts: tuple[ContactEvent, ...]
    control_steps: int
    sim_time: float  # simulated seconds
    wall_time: float  # seconds

    @property
    def grasp_success(self) -> bool:
        return self.grasp.success

    @property
    def timeouts(self) -> list[str]:
        return [p.name for p in self.phases if not p.reached]

    @property
    def collided(self) -> bool:
        return bool(self.unintended_contacts)


def block_spec(env: TabletopEnv, name: str) -> BlockSpec:
    return next(b for b in env.config.blocks if b.name == name)


def geom_role(model: mujoco.MjModel, geom: int, blocks: dict[int, str]) -> str:
    """What a geom belongs to: a block's name, table, floor, fingers, palm, arm or mount."""
    if geom in blocks:
        return blocks[geom]
    name = model.geom(geom).name
    if name == "table_collision":
        return "table"
    if name == "floor":
        return "floor"
    if name.startswith("gripper0_"):
        return "fingers" if "finger" in name else "palm"
    if name.startswith("robot0_"):
        return "arm"
    if name.startswith("fixed_mount0_"):
        return "mount"
    return name or f"geom{geom}"


def is_unintended(pair: tuple[str, str], phase: str, target: str) -> bool:
    """Whether a contact pair (sorted roles) should not occur during a phase.

    Expected: the target on the table except while lifted and carried; the fingers on
    the target from closing until the release; the fingers on each other. Everything
    else involving the robot or the target is unintended: the palm or arm touching
    anything, the robot touching the table or its mount, the fingers brushing the
    target while approaching or after letting go, the target falling to the floor.
    """
    if pair == tuple(sorted(("table", target))):
        return phase in AIRBORNE
    if pair == tuple(sorted(("fingers", target))):
        return phase not in GRIPPING
    if pair == ("fingers", "fingers"):
        return False
    return bool(ROBOT & set(pair)) or target in pair


class StepMonitor:
    """Records contacts and the target's pose after every env.step inside a with block."""

    def __init__(self, env: TabletopEnv, target: str) -> None:
        self.env = env
        model = env.sim.model._model
        blocks = {
            model.geom(geom).id: name
            for name, obj in env.blocks.items()
            for geom in obj.contact_geoms
        }
        self.roles = [geom_role(model, g, blocks) for g in range(model.ngeom)]
        self.body = env.block_body_ids[target]
        self.site = env.robots[0].eef_site_id["right"]
        self.pairs: list[frozenset[tuple[str, str]]] = []
        self.block_positions: list[np.ndarray] = []
        self.eef_positions: list[np.ndarray] = []
        self.penetration = 0.0

    def __enter__(self) -> StepMonitor:
        self._outer = self.env.__dict__.get("step")  # e.g. the viewer's paced step
        step = self.env.step

        def recording_step(action):
            out = step(action)
            self._record()
            return out

        self.env.step = recording_step
        return self

    def __exit__(self, *exc) -> None:
        if self._outer is None:
            del self.env.step
        else:
            self.env.step = self._outer

    def _record(self) -> None:
        data = self.env.sim.data._data
        pairs = set()
        for contact in data.contact[: data.ncon]:
            pairs.add(tuple(sorted((self.roles[contact.geom1], self.roles[contact.geom2]))))
            self.penetration = max(self.penetration, -contact.dist)
        self.pairs.append(frozenset(pairs))
        self.block_positions.append(data.xpos[self.body].copy())
        self.eef_positions.append(data.site_xpos[self.site].copy())

    def touching(self, name: str, roles: frozenset[str]) -> bool:
        """Whether, after the last step, the named block touched anything in roles."""
        return any(name in pair and roles & set(pair) for pair in self.pairs[-1])


def assess_placement(
    env: TabletopEnv, name: str, tolerances: PlacementTolerances | None = None
) -> PlacementCheck:
    """Judge a block's current state against the goal region from the true state.

    The box's eight corners, from its true pose and size, must all lie over the goal
    circle (the circle is convex, so then the whole box does). The block is upright if
    its vertical axis is within max_tilt of world z, on the table if it touches the
    table with its lowest corner within max_support_gap of the table top, and at rest
    if its speeds are below the tolerances.
    """
    tolerances = tolerances or PlacementTolerances()
    spec = block_spec(env, name)
    state = env.true_state().object(name)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, state.quaternion)
    rot = rot.reshape(3, 3)
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    corners = state.position + (signs * spec.half_size) @ rot.T
    goal_xy = env.goal_position[:2]
    goal_margin = env.config.goal.radius - np.hypot(*(corners[:, :2] - goal_xy).T).max()
    tilt = float(np.arccos(np.clip(abs(rot[2, 2]), -1.0, 1.0)))
    support_gap = float(corners[:, 2].min() - env.config.table_height)

    model, data = env.sim.model._model, env.sim.data._data
    geoms = set(env.blocks[name].contact_geoms)
    blocks = {model.geom(g).id: n for n, obj in env.blocks.items() for g in obj.contact_geoms}
    contacts = set()
    for contact in data.contact[: data.ncon]:
        names = model.geom(contact.geom1).name, model.geom(contact.geom2).name
        if names[0] in geoms or names[1] in geoms:
            other = contact.geom2 if names[0] in geoms else contact.geom1
            contacts.add(geom_role(model, other, blocks))
    linear_speed = float(np.linalg.norm(state.linear_velocity))
    angular_speed = float(np.linalg.norm(state.angular_velocity))
    return PlacementCheck(
        position=state.position,
        quaternion=state.quaternion,
        centre_error=float(np.hypot(*(state.position[:2] - goal_xy))),
        goal_margin=float(goal_margin),
        tilt=tilt,
        support_gap=support_gap,
        linear_speed=linear_speed,
        angular_speed=angular_speed,
        contacts=tuple(sorted(contacts)),
        in_goal=bool(goal_margin >= 0.0),
        upright=tilt <= tolerances.max_tilt,
        on_table="table" in contacts and abs(support_gap) <= tolerances.max_support_gap,
        at_rest=linear_speed <= tolerances.max_linear_speed
        and angular_speed <= tolerances.max_angular_speed,
    )


def pick_and_place(
    env: TabletopEnv,
    name: str,
    place_at: np.ndarray | None = None,
    params: PlaceParams | None = None,
    tolerances: PlacementTolerances | None = None,
) -> PickPlaceResult:
    """Grasp a block from above, carry it to the goal, set it down, let go and retract.

    Planned from env.observe() alone: the grasp from the block's observed pose, and the
    placement from the goal's observed position and the block's observed position in
    the grip after the grasp, so the grip site is aimed to put the block's centre (not
    the grip site) over the goal centre, or over place_at (world x, y) if given. A
    timeout, or a gripper closed on nothing after the grasp, skips the remaining
    movements; the robot then holds still for the settle phase. The outcome is judged
    from the true state after settling; see PickPlaceResult.
    """
    params = params or PlaceParams()
    start_wall, start_sim = time.perf_counter(), env.sim.data.time
    with StepMonitor(env, name) as monitor:
        grasp = grasp_block_from_above(env, name, lift_height=params.carry_clearance)
        phases = list(grasp.phases)
        arm = OSCArm(env)
        obs = env.observe()
        held = obs.object(name)
        abort = ""
        if not all(p.reached for p in phases):
            abort = "timeout in " + next(p.name for p in phases if not p.reached)
        elif obs.gripper_width < held.size[:2].min() / 2:
            abort = "grasp failed: the gripper closed on nothing"

        released = False
        if not abort:
            table_z = obs.goal_position[2]  # the goal is marked on the table top
            offset = held.position - obs.eef_position  # block centre from the grip site
            grip_above_bottom = held.size[2] / 2 - offset[2]
            target_xy = obs.goal_position[:2] if place_at is None else np.asarray(place_at)
            grip_xy = target_xy - offset[:2]
            rot = top_down_rotation(grasp.target_yaw)
            moves = [
                (
                    "carry",
                    table_z + params.carry_clearance + grip_above_bottom,
                    params.carry_tol,
                    params.carry_speed,
                    params.carry_steps,
                ),
                (
                    "lower",
                    table_z + params.release_gap + grip_above_bottom,
                    params.place_tol,
                    params.lower_speed,
                    params.lower_steps,
                ),
            ]
            for phase, z, tol, speed, steps in moves:
                phases.append(
                    arm.move_to(
                        phase,
                        np.array([*grip_xy, z]),
                        rot,
                        gripper=GRIPPER_CLOSED,
                        pos_tol=tol,
                        max_speed=speed,
                        max_steps=steps,
                    )
                )
                if not phases[-1].reached:
                    abort = f"timeout in {phase}"
                    break

        if not abort:
            phases.append(arm.hold("release", GRIPPER_OPEN, params.release_steps))
            released = not monitor.touching(name, ROBOT)
            # Straight up until the fingertips clear the placed block's top.
            block_top = table_z + held.size[2]
            rise = block_top + params.retract_clearance - (arm.eef_pos()[2] - FINGERTIP_DEPTH)
            phases.append(
                arm.move_to(
                    "retract",
                    arm.eef_pos() + [0.0, 0.0, rise],
                    rot,
                    gripper=GRIPPER_OPEN,
                    pos_tol=params.carry_tol,
                    max_speed=params.carry_speed,
                    max_steps=params.retract_steps,
                )
            )
            if not phases[-1].reached:
                abort = "timeout in retract"
        gripper = GRIPPER_OPEN if released else GRIPPER_CLOSED
        phases.append(arm.hold("settle", gripper, params.settle_steps))

    placement = assess_placement(env, name, tolerances)
    phase_of = [p.name for p in phases for _ in range(p.steps)]
    assert len(phase_of) == len(monitor.pairs)

    # Drops and slip: the block's position relative to the grip site while it is carried
    # and lowered, against its position there at the end of the grasp.
    offsets = np.array(monitor.block_positions) - np.array(monitor.eef_positions)
    carried = [i for i, phase in enumerate(phase_of) if phase in ("carry", "lower")]
    lost = [
        i
        for i in carried
        if np.linalg.norm(offsets[i] + [0.0, 0.0, grasp.grasp_height]) >= HELD_RADIUS
    ]
    slips = [np.linalg.norm(offsets[i] - offsets[carried[0] - 1]) for i in carried]
    max_slip = float(max(slips, default=0.0))

    events: dict[tuple[str, str], list[int]] = {}
    for i, (phase, pairs) in enumerate(zip(phase_of, monitor.pairs, strict=True)):
        for pair in pairs:
            if is_unintended(pair, phase, name):
                events.setdefault(("-".join(pair), phase), []).append(i + 1)
    unintended = tuple(
        ContactEvent(pair, phase, steps[0], len(steps)) for (pair, phase), steps in events.items()
    )

    table_z = env.config.table_height
    checks = [
        (grasp.success, "grasp failed"),
        (not abort, abort),
        (not lost, f"dropped during {phase_of[lost[0]]}" if lost else ""),
        (released, "not released: the block still touches the gripper"),
        (placement.position[2] > table_z, "the block left the table"),
        (placement.in_goal, f"outside the goal by {-1000 * placement.goal_margin:.1f} mm"),
        (placement.upright, f"not upright: tilted {np.degrees(placement.tilt):.1f} deg"),
        (placement.on_table, "not resting on the table"),
        (placement.at_rest, "still moving after settling"),
        (not placement.touching_robot, "still touching the robot after settling"),
    ]
    failure = next((reason for ok, reason in checks if not ok), "")
    return PickPlaceResult(
        success=not failure,
        failure_reason=failure,
        grasp=grasp,
        phases=phases,
        released=released,
        dropped=bool(lost),
        placement=placement,
        max_slip=max_slip,
        max_penetration=monitor.penetration,
        unintended_contacts=unintended,
        control_steps=len(monitor.pairs),
        sim_time=float(env.sim.data.time - start_sim),
        wall_time=time.perf_counter() - start_wall,
    )
