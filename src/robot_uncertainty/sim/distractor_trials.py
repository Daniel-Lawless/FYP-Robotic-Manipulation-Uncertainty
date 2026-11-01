from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_uncertainty.sim.actions import GraspAction, grasp_candidates, legacy_equivalent
from robot_uncertainty.sim.layouts import (
    beside_axis_placement,
    carry_path_placement,
    far_placement,
    hand_clearance,
)
from robot_uncertainty.sim.pick_place import PickPlaceResult, pick_and_place
from robot_uncertainty.sim.primitives import settle
from robot_uncertainty.sim.scene import BlockSpec, Placement, TabletopConfig, sample_layout
from robot_uncertainty.sim.tabletop import TabletopEnv, hand_footprint_half_extent

TARGET = "target"
DISTRACTOR = "distractor"
CORNERS = [(x, y) for x in (-0.07, 0.07) for y in (-0.17, 0.03)]  # as run_pick_place.py
FAR_DISTANCE = 0.15  # metres from the swept hand footprint (Test 4)
BESIDE_GAP = 0.04  # face-to-face gap of the grasp-axis contrast (Test 5)


@dataclass(frozen=True)
class Scene:
    """An explicit scene: a seed for the reset, and placements for set_layout."""

    config: str  # a name for reports
    scene_seed: int
    layout: dict[str, Placement]


def isolated_scenes(seeds=range(50)) -> list[Scene]:
    """The target alone: the sampled layouts of the seeds, then the workspace corners."""
    single = TabletopConfig()
    scenes = [Scene(f"seed{s}", s, dict(sample_layout(single, s))) for s in seeds]
    scenes += [
        Scene(f"corner{100 * x:+.0f}{100 * y:+.0f}", 0, {TARGET: Placement(x, y, np.pi / 4)})
        for x, y in CORNERS
    ]
    return scenes


def start(env: TabletopEnv, scene: Scene) -> list[str]:
    """Reset to the scene, settle, and return check_initial_state's problems."""
    env.reset(scene_seed=scene.scene_seed)
    env.set_layout(scene.layout)
    settle(env)
    return env.check_initial_state()


def candidates_now(env: TabletopEnv) -> tuple[GraspAction, GraspAction]:
    return grasp_candidates(env.observe().object(TARGET))


def legacy_action(env: TabletopEnv) -> GraspAction:
    """The explicit action the legacy least-rotation rule would take now."""
    return legacy_equivalent(candidates_now(env), env.observe().eef_yaw)


# Shared by scripts/run_distractor_trials.py and the simulation tests. Each attempt
# resets to a scene seed, applies an explicit layout (validated by
# TabletopEnv.set_layout), settles, checks the initial state, chooses a grasp action
# from the observation and runs the unchanged pick_and_place.
def attempt(env: TabletopEnv, scene: Scene, choose) -> tuple[PickPlaceResult, GraspAction | None]:
    """Run one attempt; choose(env) returns the GraspAction, or None for the legacy path.

    Raises RuntimeError if the settled initial state is invalid (nothing is adjusted).
    """
    problems = start(env, scene)
    if problems:
        raise RuntimeError(f"{scene.config}: invalid initial state: {problems}")
    action = choose(env)
    return pick_and_place(env, TARGET, action=action), action


def candidate(index: int):
    return lambda env: candidates_now(env)[index]


def target_spec(env: TabletopEnv) -> BlockSpec:
    return next(b for b in env.config.blocks if b.name == TARGET)


def distractor_spec(env: TabletopEnv) -> BlockSpec:
    return next(b for b in env.config.blocks if b.name == DISTRACTOR)


def table_half(env: TabletopEnv) -> tuple[float, float]:
    return env.config.table_size[0] / 2, env.config.table_size[1] / 2


def far_scenes(env: TabletopEnv, count: int = 20, max_seed: int = 200) -> list[tuple[Scene, float]]:
    """Scenes whose distractor is at least FAR_DISTANCE from the hand's footprint swept
    from the target to the goal along the legacy grasp heading, searched on the half of
    the table away from the robot, with that distance. Targets come from the seeds'
    sampled layouts, in order, skipping any where no such place exists.
    """
    env.reset(scene_seed=0)
    half_extent = hand_footprint_half_extent(env)
    yaw = env.observe().eef_yaw  # the home configuration's gripper yaw, as every reset
    single = TabletopConfig()
    out = []
    for seed in range(max_seed):
        target = sample_layout(single, seed)[TARGET]
        heading = legacy_equivalent(
            grasp_candidates(ObservedYaw(TARGET, target.yaw)), yaw
        ).closing_heading
        place, distance = far_placement(
            target_spec(env),
            target,
            distractor_spec(env),
            heading,
            half_extent,
            env.config.goal,
            table_half(env),
            step=0.01,
            min_x=0.0,
        )
        if distance >= FAR_DISTANCE:
            out.append((Scene(f"far{seed}", seed, {TARGET: target, DISTRACTOR: place}), distance))
        if len(out) == count:
            break
    return out


@dataclass(frozen=True)
class ObservedYaw:
    """Just what grasp_candidates reads from an ObjectObservation."""

    name: str
    yaw: float


def beside_scene(env: TabletopEnv, seed: int, axis_candidate: int) -> tuple[Scene, GraspAction]:
    """The seed's target with the distractor BESIDE_GAP face-to-face along the closing
    axis of one of its two candidates (candidate A, returned with the scene)."""
    target = sample_layout(TabletopConfig(), seed)[TARGET]
    action = grasp_candidates(ObservedYaw(TARGET, target.yaw))[axis_candidate]
    place = beside_axis_placement(
        target_spec(env),
        target,
        distractor_spec(env),
        BESIDE_GAP,
        action.closing_heading,
        env.config.goal,
        table_half(env),
    )
    return Scene(f"beside{seed}", seed, {TARGET: target, DISTRACTOR: place}), action


CARRY_MIN_PATH = 0.30  # metres from the target to the goal centre (Test 6)


def carry_path_scenes(count: int = 5, max_seed: int = 200) -> list[Scene]:
    """Scenes with the distractor halfway along the straight path from the target to the
    goal centre, for the first seeds whose path is at least CARRY_MIN_PATH long, so the
    distractor (15 cm from both ends) is clear of the hand at the grasp and of the goal.
    """
    config, out = TabletopConfig(), []
    for seed in range(max_seed):
        target = sample_layout(config, seed)[TARGET]
        if np.hypot(*(np.asarray(config.goal.center) - target.xy)) < CARRY_MIN_PATH:
            continue
        place = carry_path_placement(target, config.goal)
        out.append(Scene(f"carry{seed}", seed, {TARGET: target, DISTRACTOR: place}))
        if len(out) == count:
            break
    return out


def swept_clearance(env: TabletopEnv, scene: Scene, heading: float) -> float:
    """Distance from the scene's distractor to the hand footprint swept to the goal."""
    target = scene.layout[TARGET]
    return hand_clearance(
        distractor_spec(env),
        scene.layout[DISTRACTOR],
        target.xy,
        env.config.goal.center,
        heading,
        hand_footprint_half_extent(env),
    )


def row(
    scene: Scene,
    choice: str,
    result: PickPlaceResult,
    action: GraspAction | None,
    extra: dict | None = None,
) -> dict:
    """One CSV row describing an attempt and the distractor's outcome (if any).

    extra holds an experiment's own columns, placed after the M1.3a columns and before
    the M1.3a.1 ones."""
    g, label = result.grasp, result.label
    objects = {o.name: o for o in result.objects}
    target = scene.layout[TARGET]
    out = dict(
        config=scene.config,
        scene_seed=scene.scene_seed,
        x_cm=round(100 * target.x, 3),
        y_cm=round(100 * target.y, 3),
        yaw_deg=round(float(np.degrees(target.yaw)), 3),
        choice=choice,
        closing_deg="" if action is None else round(float(np.degrees(action.closing_heading)), 3),
        success=result.success,
        failure_reason=result.failure_reason,
        y=label.y,
        failure_stage=label.failure_stage,
        flags=";".join(label.names),
        wrist_deg=round(float(np.degrees(g.wrist_rotation)), 3),
        commanded_face_axis=g.commanded_face_axis,
        face_axis=g.face_axis,
        timeouts=";".join(result.timeouts),
        unintended_contacts=";".join(
            f"{e.pair}@{e.phase}x{e.steps}:{e.peak_force:.1f}N" for e in result.unintended_contacts
        ),
        centre_error_mm=round(1000 * result.placement.centre_error, 3),
        steps=result.control_steps,
        sim_s=round(result.sim_time, 3),
        wall_s=round(result.wall_time, 3),
    )
    d = objects.get(DISTRACTOR)
    if d is not None:
        place = scene.layout[DISTRACTOR]
        out.update(
            d_x_cm=round(100 * place.x, 3),
            d_y_cm=round(100 * place.y, 3),
            d_yaw_deg=round(float(np.degrees(place.yaw)), 3),
            d_max_mm=round(1000 * d.max_displacement, 3),
            d_final_mm=round(1000 * d.final_displacement, 3),
            d_tilt_max_deg=round(float(np.degrees(d.tilt_max)), 3),
            d_tilt_final_deg=round(float(np.degrees(d.tilt_final)), 3),
            d_face_start=d.resting_face_start,
            d_face_final=d.resting_face_final,
            d_toppled=d.toppled,
            d_leaning=d.leaning,
            d_unsettled=d.unsettled,
            d_com_outside_base=d.com_outside_base,
            d_contacts=";".join(d.final_contacts),
        )
    out.update(extra or {})
    # M1.3a.1: the substep-resolution evidence behind y, flags and the d_ maxima above,
    # and the control-step (endpoint) assessment of M1.3a for comparison.
    historical = result.evidence.contacts
    out.update(
        historical_pairs=";".join(sorted({e.pair for e in historical})),
        historical_unintended=";".join(
            f"{e.pair}@{e.phase}x{e.steps}/{e.substeps}:{e.peak_force:.1f}N"
            for e in historical
            if e.unintended
        ),
        endpoint_y=result.endpoint_label.y,
        endpoint_failure_stage=result.endpoint_label.failure_stage,
        endpoint_flags=";".join(result.endpoint_label.names),
    )
    if d is not None:
        e = next(o for o in result.endpoint_objects if o.name == DISTRACTOR)
        out.update(
            d_max_endpoint_mm=round(1000 * e.max_displacement, 3),
            d_tilt_max_endpoint_deg=round(float(np.degrees(e.tilt_max)), 3),
        )
    return out


# Sampled-layout seeds (tall distractor, legacy-equivalent action) where M1.3a's
# control-step contacts missed a target-distractor impact that toppled the distractor.
KNOWN_MISSES = frozenset({3, 8, 30, 49})


def sampling_row(scene: Scene, choice: str, result: PickPlaceResult) -> dict:
    """Endpoint (control-step) against historical (substep) evidence for one attempt.

    Pair counts are of distinct role pairs over the whole attempt, whatever the phase,
    from all events of each stream (not only the unintended ones).
    """
    endpoint = {e.pair for e in result.endpoint_contacts}
    historical = {e.pair for e in result.evidence.contacts}
    hit = f"{DISTRACTOR}-{TARGET}"
    d = next(o for o in result.objects if o.name == DISTRACTOR)
    e = next(o for o in result.endpoint_objects if o.name == DISTRACTOR)
    return dict(
        config=scene.config,
        scene_seed=scene.scene_seed,
        choice=choice,
        known_miss=scene.config == f"sampled{scene.scene_seed}"
        and scene.scene_seed in KNOWN_MISSES,
        endpoint_pairs=len(endpoint),
        historical_pairs=len(historical),
        missed_pairs=";".join(sorted(historical - endpoint)),
        endpoint_collision_object="collision_object" in result.endpoint_label.flags,
        historical_collision_object="collision_object" in result.label.flags,
        endpoint_target_distractor=hit in endpoint,
        historical_target_distractor=hit in historical,
        d_max_endpoint_mm=round(1000 * e.max_displacement, 3),
        d_max_mm=round(1000 * d.max_displacement, 3),
        d_tilt_max_endpoint_deg=round(float(np.degrees(e.tilt_max)), 3),
        d_tilt_max_deg=round(float(np.degrees(d.tilt_max)), 3),
        endpoint_y=result.endpoint_label.y,
        y=result.label.y,
        endpoint_failure_stage=result.endpoint_label.failure_stage,
        failure_stage=result.label.failure_stage,
    )
