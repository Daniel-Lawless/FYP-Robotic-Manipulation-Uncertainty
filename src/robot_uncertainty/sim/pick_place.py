from __future__ import annotations

import time
from dataclasses import dataclass, replace

import mujoco
import numpy as np

from robot_uncertainty.sim.actions import GraspAction
from robot_uncertainty.sim.outcome import (  # noqa: F401 (re-exported)
    GRASP_PHASES,
    ROBOT,
    ContactEvent,
    ObjectOutcome,
    ObjectRecord,
    OutcomeEvidence,
    OutcomeLabel,
    OutcomeThresholds,
    PlacementCheck,
    PlacementTolerances,
    TableSurface,
    initial_resting_face,
    object_outcome,
    outcome,
)
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

# Roles that are fixed parts of the scene (ContactEvent.kind "environment").
ENVIRONMENT = frozenset({"table", "floor", "mount"})
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


@dataclass
class PickPlaceResult:
    """Outcome of one pick-and-place attempt.

    success is the task outcome from the true state after settling; contacts that
    should not happen are listed separately in unintended_contacts. objects, evidence
    and label add the whole scene (M1.3a): label.y also requires that no other block was
    displaced, toppled or knocked off; see outcome.outcome.

    The legacy fields (success to control_steps) come from endpoint measurements, after
    each control step, exactly as in M1.2: unintended_contacts is built from the
    endpoint contact stream, not from evidence.contacts. objects, evidence and label use
    every physics substep too (M1.3a.1): historical contact events and maxima over the
    substep poses. endpoint_contacts, endpoint_objects and endpoint_label keep the
    control-step assessment of M1.3a for comparison.
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
    objects: tuple[ObjectOutcome, ...] = ()  # every block in the scene, in config order
    evidence: OutcomeEvidence | None = None
    label: OutcomeLabel | None = None
    endpoint_contacts: tuple[ContactEvent, ...] = ()  # every endpoint event, all pairs
    endpoint_objects: tuple[ObjectOutcome, ...] = ()  # maxima over control steps only
    endpoint_label: OutcomeLabel | None = None  # from endpoint_contacts and _objects

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


def is_unintended(
    pair: tuple[str, str], phase: str, target: str, others: frozenset[str] = frozenset()
) -> bool:
    """Whether a contact pair (sorted roles) should not occur during a phase.

    Expected: the target on the table except while lifted and carried; the fingers on
    the target from closing until the release; the fingers on each other. Everything
    else involving the robot or the target is unintended: the palm or arm touching
    anything, the robot touching the table or its mount, the fingers brushing the
    target while approaching or after letting go, the target falling to the floor.

    others names the scene's other blocks: they are expected on the table, and any other
    contact of theirs (with the robot, the target, each other or the floor) is
    unintended. With none, the classification is the single-block one above.
    """
    if pair == tuple(sorted(("table", target))):
        return phase in AIRBORNE
    if pair == tuple(sorted(("fingers", target))):
        return phase not in GRIPPING
    if pair == ("fingers", "fingers"):
        return False
    if others & set(pair):
        return "table" not in pair
    return bool(ROBOT & set(pair)) or target in pair


# Integrators whose step leaves the contact list, xpos/xquat and the constraint forces
# describing the configuration before integration, and qpos/qvel the one after (checked
# by tests/sim/test_substep_evidence.py). RK4 recomputes them at intermediate stages.
SUBSTEP_INTEGRATORS = frozenset(
    {
        mujoco.mjtIntegrator.mjINT_EULER,
        mujoco.mjtIntegrator.mjINT_IMPLICIT,
        mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    }
)
_UNSET = object()


class StepMonitor:
    """Records contacts and poses inside a with block, in two separate streams.

    Endpoint stream, after every env.step (one control step): the contact pairs and the
    target's and grip site's positions (pairs, block_positions, eef_positions), the
    deepest penetration, every block's position and orientation (poses[name], one
    (position, wxyz quaternion) per step, from xpos/xquat) and each contact pair's total
    normal force (forces, one {pair: N} per step). After env.step, the contact list,
    xpos and xquat describe the configuration at the start of its last physics substep.

    Substep stream, after every physics substep (each sim.step2, or sim.step on the
    non-lite path): each role pair's total normal force over its contact points
    (substep_pairs, {pair key: N}; see role_names), read from the contact list, which
    describes the configuration before that substep's integration, and every block's
    pose after the integration, read from its free-joint qpos (substep_poses, one array
    of (x, y, z, w, qx, qy, qz) rows in block_names order). substep_bounds[i] is the
    number of substeps recorded by the end of control step i + 1.

    Recording reads existing arrays (and mj_contactForce) only, so it never changes the
    motion. The wrappers are instance attributes of env and env.sim, removed or
    restored on exit, so monitors nest and compose with other wrappers such as the
    viewer's paced step.
    """

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
        self.forces: list[dict[tuple[str, str], float]] = []
        self.poses: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {n: [] for n in env.blocks}
        self._force = np.zeros(6)

        if model.opt.integrator not in SUBSTEP_INTEGRATORS:
            integrator = mujoco.mjtIntegrator(model.opt.integrator).name
            raise ValueError(f"substep recording is not verified for {integrator}")
        # Role ids in sorted name order, so a pair key lo * n + hi sorts like the pair.
        self.role_names = tuple(sorted(set(self.roles)))
        index = {name: i for i, name in enumerate(self.role_names)}
        ids = np.array([index[r] for r in self.roles], dtype=np.int64)
        # The pair key of every geom pair, so a substep needs one lookup per contact.
        self._pair_keys = np.minimum.outer(ids, ids) * len(self.role_names) + np.maximum.outer(
            ids, ids
        )
        self.block_names = tuple(env.blocks)
        addresses = []
        for name in self.block_names:
            body = env.block_body_ids[name]
            joint = model.body_jntadr[body]
            free = model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
            if model.body_parentid[body] != 0 or model.body_jntnum[body] != 1 or not free:
                raise ValueError(f"{name}: substep poses need a world child with one free joint")
            addresses.append(model.jnt_qposadr[joint])
        self._pose_index = (np.array(addresses)[:, None] + np.arange(7)).ravel()
        self._model, self._data = model, env.sim.data._data
        self.substep_pairs: list[dict[int, float]] = []
        self.substep_poses: list[np.ndarray] = []
        self.substep_bounds: list[int] = []

    def pair_of(self, key: int) -> tuple[str, str]:
        """The sorted role pair of a substep_pairs key."""
        lo, hi = divmod(key, len(self.role_names))
        return self.role_names[lo], self.role_names[hi]

    def __enter__(self) -> StepMonitor:
        self._outer = self.env.__dict__.get("step")  # e.g. the viewer's paced step
        step = self.env.step

        def recording_step(action):
            out = step(action)
            self._record()
            return out

        self.env.step = recording_step
        self._sim = self.env.sim
        self._sim_saved = {}
        for name in ("step", "step2"):
            self._sim_saved[name] = self._sim.__dict__.get(name, _UNSET)
            original = getattr(self._sim, name)

            def recording_substep(*args, _original=original, **kwargs):
                out = _original(*args, **kwargs)
                self._record_substep()
                return out

            setattr(self._sim, name, recording_substep)
        return self

    def __exit__(self, *exc) -> None:
        if self._outer is None:
            del self.env.step
        else:
            self.env.step = self._outer
        for name, saved in self._sim_saved.items():
            if saved is _UNSET:
                del self._sim.__dict__[name]
            else:
                setattr(self._sim, name, saved)

    def _record_substep(self) -> None:
        data = self._data
        self.substep_poses.append(data.qpos[self._pose_index])
        totals: dict[int, float] = {}
        if data.ncon:
            geom = data.contact.geom
            keys = self._pair_keys[geom[:, 0], geom[:, 1]].tolist()
            model, force = self._model, self._force
            for i, key in enumerate(keys):
                mujoco.mj_contactForce(model, data, i, force)
                totals[key] = totals.get(key, 0.0) + abs(float(force[0]))
        self.substep_pairs.append(totals)

    def _record(self) -> None:
        model, data = self.env.sim.model._model, self.env.sim.data._data
        pairs = set()
        forces: dict[tuple[str, str], float] = {}
        for i, contact in enumerate(data.contact[: data.ncon]):
            pair = tuple(sorted((self.roles[contact.geom1], self.roles[contact.geom2])))
            pairs.add(pair)
            self.penetration = max(self.penetration, -contact.dist)
            mujoco.mj_contactForce(model, data, i, self._force)
            forces[pair] = forces.get(pair, 0.0) + abs(float(self._force[0]))
        self.pairs.append(frozenset(pairs))
        self.forces.append(forces)
        self.block_positions.append(data.xpos[self.body].copy())
        self.eef_positions.append(data.site_xpos[self.site].copy())
        for name, poses in self.poses.items():
            body = self.env.block_body_ids[name]
            poses.append((data.xpos[body].copy(), data.xquat[body].copy()))
        self.substep_bounds.append(len(self.substep_pairs))

    def touching(self, name: str, roles: frozenset[str]) -> bool:
        """Whether, at the endpoint of the last control step, the named block touched
        anything in roles: an endpoint query over pairs[-1] only, not over the substeps
        of that interval."""
        return any(name in pair and roles & set(pair) for pair in self.pairs[-1])

    def substep_controls(self) -> list[int]:
        """The control step (from 1) of each recorded substep: that of the first env.step
        to end after it (the env.step it ran in, unless the physics was stepped directly).
        """
        if len(self.substep_pairs) != (self.substep_bounds[-1] if self.substep_bounds else 0):
            raise ValueError("substeps were recorded outside env.step")
        controls, start = [], 0
        for step, end in enumerate(self.substep_bounds, start=1):
            controls += [step] * (end - start)
            start = end
        return controls


def historical_events(
    monitor: StepMonitor, phase_of: list[str], target: str, others: frozenset[str]
) -> tuple[ContactEvent, ...]:
    """Every contact pair per phase from a monitor's substep stream (see ContactEvent).

    Each substep takes the phase of the control step it ran in. Events are in order of
    first substep, pairs sorted within a substep; classification as for the endpoint
    events (is_unintended, ENVIRONMENT).
    """
    # (key, phase) -> [first_step, steps, last_step, first_substep, substeps, peak_force]
    events: dict[tuple[int, str], list] = {}
    for substep, (step, totals) in enumerate(
        zip(monitor.substep_controls(), monitor.substep_pairs, strict=True), start=1
    ):
        phase = phase_of[step - 1]
        for key in sorted(totals):
            event = events.get((key, phase))
            if event is None:
                events[(key, phase)] = [step, 1, step, substep, 1, totals[key]]
                continue
            if event[2] != step:
                event[1] += 1
                event[2] = step
            event[4] += 1
            event[5] = max(event[5], totals[key])
    out = []
    for (key, phase), (first, steps, _, first_sub, subs, peak) in events.items():
        pair = monitor.pair_of(key)
        out.append(
            ContactEvent(
                "-".join(pair),
                phase,
                first,
                steps,
                peak_force=peak,
                unintended=is_unintended(pair, phase, target, others),
                kind="environment" if ENVIRONMENT & set(pair) else "object",
                first_substep=first_sub,
                substeps=subs,
            )
        )
    return tuple(out)


def block_contacts(env: TabletopEnv, name: str) -> tuple[str, ...]:
    """Roles (see geom_role) of everything touching a block now, sorted."""
    model, data = env.sim.model._model, env.sim.data._data
    blocks = {model.geom(g).id: n for n, obj in env.blocks.items() for g in obj.contact_geoms}
    own = {model.geom(g).id for g in env.blocks[name].contact_geoms}
    roles = set()
    for contact in data.contact[: data.ncon]:
        if contact.geom1 in own and contact.geom2 not in own:
            roles.add(geom_role(model, contact.geom2, blocks))
        elif contact.geom2 in own and contact.geom1 not in own:
            roles.add(geom_role(model, contact.geom1, blocks))
    return tuple(sorted(roles))


def object_record(env: TabletopEnv, name: str) -> ObjectRecord:
    """A block's true state now, for outcome.object_outcome."""
    spec = block_spec(env, name)
    state = env.true_state().object(name)
    return ObjectRecord(
        name=name,
        role=spec.role,
        size=np.asarray(spec.size, dtype=float),
        position=state.position,
        quaternion=state.quaternion,
        linear_speed=float(np.linalg.norm(state.linear_velocity)),
        angular_speed=float(np.linalg.norm(state.angular_velocity)),
        contacts=block_contacts(env, name),
    )


def table_surface(env: TabletopEnv) -> TableSurface:
    size = env.config.table_size
    return TableSurface(env.config.table_height, size[0] / 2, size[1] / 2)


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
    action: GraspAction | None = None,
    thresholds: OutcomeThresholds | None = None,
) -> PickPlaceResult:
    """Grasp a block from above, carry it to the goal, set it down, let go and retract.

    Planned from env.observe() alone: the grasp from the block's observed pose, and the
    placement from the goal's observed position and the block's observed position in
    the grip after the grasp, so the grip site is aimed to put the block's centre (not
    the grip site) over the goal centre, or over place_at (world x, y) if given. A
    timeout, or a gripper closed on nothing after the grasp, skips the remaining
    movements; the robot then holds still for the settle phase. The outcome is judged
    from the true state after settling; see PickPlaceResult.

    action, if given, fixes the face pair to grip (its closing heading, chosen at
    decision time; see actions.execution_heading); otherwise the grasp takes the face
    pair needing the least wrist rotation, as before. Every block is recorded from the
    start of the call to the end of the settle phase for the scene outcome
    (PickPlaceResult.objects and label).

    Raises ValueError, before anything moves, if the start is invalid: any problem from
    env.check_initial_state(), or a block with no resting face (see
    outcome.initial_resting_face). The check only reads the state; settle first.
    """
    if action is not None and action.object != name:
        raise ValueError(f"action is for {action.object!r}, not {name!r}")
    params = params or PlaceParams()
    start_wall, start_sim = time.perf_counter(), env.sim.data.time
    problems = env.check_initial_state()
    if problems:
        raise ValueError(f"invalid initial state: {'; '.join(problems)}")
    table = table_surface(env)
    start = {n: object_record(env, n) for n in env.blocks}
    for record in start.values():
        initial_resting_face(record, table, thresholds, tolerances)
    # Without an action, the M1.2 call exactly (closing_heading is not even passed).
    explicit = {} if action is None else {"closing_heading": action.closing_heading}
    with StepMonitor(env, name) as monitor:
        grasp = grasp_block_from_above(env, name, lift_height=params.carry_clearance, **explicit)
        phases = list(grasp.phases)
        arm = OSCArm(env)
        obs = env.observe()
        held = obs.object(name)
        gripper_empty = bool(obs.gripper_width < held.size[:2].min() / 2)
        abort = ""
        if not all(p.reached for p in phases):
            abort = "timeout in " + next(p.name for p in phases if not p.reached)
        elif gripper_empty:
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

    # Endpoint events: every contact pair at the control-step endpoints, per phase, in
    # order of first appearance (sorted within a step). The legacy unintended_contacts
    # are these, filtered; the historical events (evidence.contacts) are built apart.
    others = frozenset(env.blocks) - {name}
    events: dict[tuple[tuple[str, str], str], tuple[list[int], list[float]]] = {}
    for i, (phase, pairs, forces) in enumerate(
        zip(phase_of, monitor.pairs, monitor.forces, strict=True)
    ):
        for pair in sorted(pairs):
            steps, peaks = events.setdefault((pair, phase), ([], []))
            steps.append(i + 1)
            peaks.append(forces[pair])
    endpoint_contacts = tuple(
        ContactEvent(
            "-".join(pair),
            phase,
            steps[0],
            len(steps),
            peak_force=max(peaks),
            unintended=is_unintended(pair, phase, name, others),
            kind="environment" if ENVIRONMENT & set(pair) else "object",
        )
        for (pair, phase), (steps, peaks) in events.items()
    )
    unintended = tuple(e for e in endpoint_contacts if e.unintended)
    contacts = historical_events(monitor, phase_of, name, others)

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

    # Maxima: endpoint_objects over the start, the control-step poses and the final
    # record, as in M1.3a; objects also over the pose after every physics substep.
    final = {n: object_record(env, n) for n in env.blocks}
    substeps = np.array(monitor.substep_poses).reshape(len(monitor.substep_poses), -1, 7)

    def assess(n: str, dense: bool) -> ObjectOutcome:
        positions = np.array([p for p, _ in monitor.poses[n]]).reshape(-1, 3)
        quaternions = np.array([q for _, q in monitor.poses[n]]).reshape(-1, 4)
        if dense:
            poses = substeps[:, monitor.block_names.index(n)]
            positions = np.concatenate([positions, poses[:, :3]])
            quaternions = np.concatenate([quaternions, poses[:, 3:]])
        return object_outcome(
            start[n], final[n], positions, quaternions, table, thresholds, tolerances
        )

    endpoint_objects = tuple(assess(n, dense=False) for n in env.blocks)
    objects = tuple(assess(n, dense=True) for n in env.blocks)
    endpoint_evidence = OutcomeEvidence(
        target=name,
        grasp=grasp,
        grasp_success=grasp.success,
        phases=tuple((p.name, p.reached) for p in phases),
        abort=abort,
        gripper_empty=gripper_empty,
        dropped=bool(lost),
        drop_phase=phase_of[lost[0]] if lost else "",
        released=released,
        placement=placement,
        objects=endpoint_objects,
        contacts=endpoint_contacts,
        table=table,
    )
    evidence = replace(endpoint_evidence, objects=objects, contacts=contacts)
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
        objects=objects,
        evidence=evidence,
        label=outcome(evidence, thresholds),
        endpoint_contacts=endpoint_contacts,
        endpoint_objects=endpoint_objects,
        endpoint_label=outcome(endpoint_evidence, thresholds),
    )
