import copy
from dataclasses import fields, replace

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("robosuite")
import mujoco  # noqa: E402

from robot_uncertainty.sim import distractor_trials as trials  # noqa: E402
from robot_uncertainty.sim import pick_place  # noqa: E402
from robot_uncertainty.sim.outcome import ROBOT, ContactEvent  # noqa: E402
from robot_uncertainty.sim.pick_place import (  # noqa: E402
    StepMonitor,
    historical_events,
    pick_and_place,
)
from robot_uncertainty.sim.primitives import GRIPPER_CLOSED, settle  # noqa: E402
from robot_uncertainty.sim.scene import (  # noqa: E402
    DISTRACTOR_TALL,
    BlockSpec,
    Placement,
    TabletopConfig,
    sample_layout,
)
from robot_uncertainty.sim.tabletop import grasp_block_from_above, make_tabletop_env  # noqa: E402

TARGET, DISTRACTOR = "target", "distractor"
CONFIG = TabletopConfig(blocks=(BlockSpec(TARGET), DISTRACTOR_TALL))
LAYOUT = {TARGET: Placement(0.0, -0.1, 0.0), DISTRACTOR: Placement(0.15, -0.15, 0.0)}
# Baseline M1.3a labels of the four impacts its control-step contacts missed
# (results of the pre-patch capture at 9f7ce6b; see data/validation/README.md).
KNOWN_MISSES = {3: (0, "scene"), 8: (0, "scene"), 30: (0, "scene"), 49: (0, "scene")}


@pytest.fixture(scope="module")
def env():
    env = make_tabletop_env(CONFIG, seed=0)
    yield env
    env.close()


@pytest.fixture(scope="module")
def full_env():
    """The alternate stepping path: sim.forward, _pre_action, sim.step."""
    env = make_tabletop_env(CONFIG, seed=0, lite_physics=False)
    yield env
    env.close()


def settled(env) -> None:
    env.reset(scene_seed=0)
    env.set_layout(LAYOUT)
    settle(env)
    assert env.check_initial_state() == []


def hold(env, steps: int = 1) -> None:
    for _ in range(steps):
        env.step(np.zeros(env.action_dim))


def substeps_per_step(env) -> int:
    return int(env.control_timestep / env.model_timestep)


def copy_data(model, data):
    out = mujoco.MjData(model)
    mujoco.mj_copyData(out, model, data)
    return out


def contact_state(data) -> tuple:
    n = data.ncon
    return (
        n,
        data.contact.geom[:n].copy(),
        data.contact.dist[:n].copy(),
        data.contact.pos[:n].copy(),
        data.contact.efc_address[:n].copy(),
    )


def forces(model, data) -> np.ndarray:
    out, f = [], np.zeros(6)
    for i in range(data.ncon):
        mujoco.mj_contactForce(model, data, i, f)
        out.append(f.copy())
    return np.array(out)


def same(a, b) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def key(monitor, a: str, b: str) -> int:
    lo, hi = sorted(monitor.role_names.index(r) for r in (a, b))
    return lo * len(monitor.role_names) + hi


# --- physics timing, verified on copies of the live data ------------------------------


@pytest.mark.parametrize("integrator", ["mjINT_EULER", "mjINT_IMPLICIT", "mjINT_IMPLICITFAST"])
def test_substep_timing_on_copied_data(env, integrator) -> None:
    """After step2 (and after mj_step), the contact list, xpos/xquat and the contact
    forces describe the configuration before integration; qpos/qvel the one after."""
    settled(env)
    model = copy.deepcopy(env.sim.model._model)
    model.opt.integrator = getattr(mujoco.mjtIntegrator, integrator)
    start = copy_data(model, env.sim.data._data)
    joint = model.body_jntadr[env.block_body_ids[DISTRACTOR]]
    start.qvel[model.jnt_dofadr[joint] : model.jnt_dofadr[joint] + 3] = [0.2, 0.0, 0.0]
    lite = copy_data(model, start)
    mujoco.mj_step1(model, lite)
    mujoco.mj_step2(model, lite)
    full = copy_data(model, start)
    mujoco.mj_step(model, full)
    before = copy_data(model, start)
    mujoco.mj_forward(model, before)  # the configuration before integration
    for d in (lite, full):
        assert same(contact_state(d), contact_state(before))
        assert np.array_equal(d.xpos, before.xpos) and np.array_equal(d.xquat, before.xquat)
        assert np.array_equal(forces(model, d), forces(model, before))
    assert np.array_equal(lite.qpos, full.qpos) and np.array_equal(lite.qvel, full.qvel)
    assert not np.array_equal(lite.qpos, start.qpos)  # integrated: the block moved
    after = copy_data(model, lite)
    mujoco.mj_kinematics(model, after)  # the configuration after integration
    assert not np.array_equal(after.xpos, lite.xpos)


@pytest.mark.parametrize("path", ["lite", "full"])
def test_env_step_returns_the_last_substeps_state(env, full_env, path) -> None:
    """Nothing after the last substep (_update_observables, _post_action, the
    observations) changes the arrays the monitors read."""
    e = env if path == "lite" else full_env
    settled(e)
    model, data = e.sim.model._model, e.sim.data._data
    name = "step2" if path == "lite" else "step"
    original, snapshots = getattr(e.sim, name), []

    def snapshot(*args):
        original(*args)
        snapshots.append(copy_data(model, data))

    setattr(e.sim, name, snapshot)
    try:
        with StepMonitor(e, TARGET) as monitor:
            hold(e, 3)
    finally:
        delattr(e.sim, name)
    n = substeps_per_step(e)
    assert n == round(e.control_timestep / e.model_timestep) == 25
    assert len(snapshots) == len(monitor.substep_pairs) == 3 * n
    assert monitor.substep_bounds == [n, 2 * n, 3 * n]
    assert monitor.substep_controls() == [1] * n + [2] * n + [3] * n
    last = snapshots[-1]
    assert same(contact_state(data), contact_state(last))
    for field in ("qpos", "qvel", "xpos", "xquat", "efc_force", "time"):
        assert np.array_equal(getattr(data, field), getattr(last, field)), field


def test_substep_poses_are_the_free_joint_world_poses(env) -> None:
    settled(env)
    model = env.sim.model._model
    for name in (TARGET, DISTRACTOR):
        body = env.block_body_ids[name]
        joint = model.body_jntadr[body]
        assert model.body_parentid[body] == 0 and model.body_jntnum[body] == 1
        assert model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
    joint = env.blocks[DISTRACTOR].joints[0]
    env.sim.data.set_joint_qvel(joint, [0.3, -0.1, 0.0, 0.0, 0.0, 2.0])  # slide and spin
    with StepMonitor(env, TARGET) as monitor:
        hold(env, 2)
    check = copy_data(model, env.sim.data._data)
    mujoco.mj_kinematics(model, check)  # on the copy only
    poses = monitor.substep_poses[-1].reshape(-1, 7)
    for k, name in enumerate(monitor.block_names):
        body = env.block_body_ids[name]
        assert np.max(np.abs(poses[k, :3] - check.xpos[body])) <= 1e-12
        assert np.max(np.abs(poses[k, 3:] - check.xquat[body])) <= 1e-12


# --- wrapper lifetime -----------------------------------------------------------------


def hooks(env) -> tuple:
    return (
        env.__dict__.get("step"),
        env.sim.__dict__.get("step"),
        env.sim.__dict__.get("step2"),
    )


def test_hooks_are_removed_after_use_and_after_an_exception(env) -> None:
    settled(env)
    assert hooks(env) == (None, None, None)
    for _ in range(3):  # repeated contexts do not accumulate wrappers
        with StepMonitor(env, TARGET) as monitor:
            hold(env)
        assert hooks(env) == (None, None, None)
        assert len(monitor.substep_pairs) == substeps_per_step(env)
    with pytest.raises(RuntimeError), StepMonitor(env, TARGET):
        hold(env)
        raise RuntimeError("inside the monitor")
    assert hooks(env) == (None, None, None)


def test_existing_wrappers_are_kept_and_composed(env) -> None:
    """E.g. the viewer's paced env.step, and a wrapper on sim.step2."""
    settled(env)
    calls = {"step": 0, "step2": 0}
    step, step2 = env.step, env.sim.step2

    def paced_step(action):
        calls["step"] += 1
        return step(action)

    def counted_step2():
        calls["step2"] += 1
        return step2()

    env.step, env.sim.step2 = paced_step, counted_step2
    try:
        with StepMonitor(env, TARGET) as monitor:
            hold(env, 2)
        assert env.step is paced_step and env.sim.step2 is counted_step2
        assert "step" not in env.sim.__dict__
        assert calls == {"step": 2, "step2": 2 * substeps_per_step(env)}
        assert len(monitor.substep_pairs) == calls["step2"]
    finally:
        del env.step, env.sim.step2


def test_nested_monitors_see_each_substep_once(env) -> None:
    settled(env)
    n = substeps_per_step(env)
    with StepMonitor(env, TARGET) as outer:
        hold(env)
        with StepMonitor(env, TARGET) as inner:
            hold(env, 2)
        hold(env)
    assert hooks(env) == (None, None, None)
    assert len(outer.substep_pairs) == 4 * n and outer.substep_bounds[-1] == 4 * n
    assert len(inner.substep_pairs) == 2 * n and inner.substep_bounds == [n, 2 * n]
    assert same(outer.substep_poses[n : 3 * n], inner.substep_poses)
    assert outer.substep_pairs[n : 3 * n] == inner.substep_pairs


def test_the_alternate_stepping_path_is_recorded(full_env) -> None:
    settled(full_env)
    with StepMonitor(full_env, TARGET) as monitor:
        hold(full_env, 2)
    assert hooks(full_env) == (None, None, None)
    assert len(monitor.substep_pairs) == 2 * substeps_per_step(full_env)
    assert monitor.substep_pairs[-1]  # the blocks on the table


def test_unverified_integrators_are_refused(env) -> None:
    model = env.sim.model._model
    saved = model.opt.integrator
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    try:
        with pytest.raises(ValueError, match="mjINT_RK4"):
            StepMonitor(env, TARGET)
    finally:
        model.opt.integrator = saved


# --- historical events from a controlled substep record -----------------------------------


def controlled(env, phases, substeps: dict[int, dict], per_step: int = 3):
    """A monitor with a hand-written record: substeps maps a 1-based substep to
    {(role, role): total force}; the endpoint pairs are those of each step's last
    substep, as the live recording would show them."""
    monitor = StepMonitor(env, TARGET)
    total = per_step * len(phases)
    monitor.substep_pairs = [
        {key(monitor, *pair): f for pair, f in substeps.get(s, {}).items()}
        for s in range(1, total + 1)
    ]
    monitor.substep_bounds = [per_step * (i + 1) for i in range(len(phases))]
    monitor.pairs = [
        frozenset(tuple(sorted(p)) for p in substeps.get(per_step * (i + 1), {}))
        for i in range(len(phases))
    ]
    return monitor


def test_events_count_distinct_substeps_and_control_intervals(env) -> None:
    ft, pt = ("fingers", TARGET), ("palm", TARGET)
    phases = ["carry", "carry", "carry", "release"]
    # Substeps 1-3 are control step 1, 4-6 step 2, and so on. The pair touches in steps 1
    # and 3 but not 2 (non-consecutive), twice within step 1.
    record = {1: {ft: 5.0}, 3: {ft: 7.0}, 8: {ft: 6.5}, 10: {ft: 2.0, pt: 1.0}, 11: {ft: 4.0}}
    monitor = controlled(env, phases, record)
    events = historical_events(monitor, phases, TARGET, frozenset({DISTRACTOR}))
    assert events == (
        ContactEvent("fingers-target", "carry", 1, 2, 7.0, False, "object", 1, 3),
        ContactEvent("fingers-target", "release", 4, 1, 4.0, False, "object", 10, 2),
        ContactEvent("palm-target", "release", 4, 1, 1.0, True, "object", 10, 1),
    )


def test_release_contact_within_the_last_interval_is_history_not_endpoint(env) -> None:
    """The fingers leave the block during the final release interval: the history has
    the contact, the endpoint (touching, hence released) does not."""
    ft = ("fingers", TARGET)
    phases = ["lower", "release", "release"]
    record = {3: {ft: 30.0}, 6: {ft: 25.0}, 7: {ft: 3.0}, 8: {ft: 0.5}}
    monitor = controlled(env, phases, record)
    assert not monitor.touching(TARGET, ROBOT)  # released = True
    (lower, release) = historical_events(monitor, phases, TARGET, frozenset())
    assert (release.phase, release.first_step, release.steps, release.substeps) == (
        "release",
        2,
        2,
        3,
    )
    assert not release.unintended  # expected while releasing: no collision flag either
    # Converse: still touching at the release endpoint, so not released.
    record[9] = {ft: 0.2}
    assert controlled(env, phases, record).touching(TARGET, ROBOT)


def test_release_failure_comes_from_the_endpoint_only(env) -> None:
    settled(env)
    result = pick_and_place(env, TARGET)
    assert result.released and "release_failure" not in result.label.flags
    history = {(c.pair, c.phase): c for c in result.evidence.contacts}
    endpoint = {(c.pair, c.phase): c for c in result.endpoint_contacts}
    assert (
        history[("fingers-target", "release")].steps
        >= endpoint[("fingers-target", "release")].steps
    )
    # Even a historical contact through the whole release leaves the label unchanged.
    extra = ContactEvent("fingers-target", "release", 1, 10, 40.0, False, "object", 1, 250)
    evidence = replace(result.evidence, contacts=result.evidence.contacts + (extra,))
    assert pick_place.outcome(evidence) == result.label


def test_prevented_release_is_still_a_release_failure(env, monkeypatch) -> None:
    monkeypatch.setattr(pick_place, "GRIPPER_OPEN", GRIPPER_CLOSED)
    settled(env)
    with StepMonitor(env, TARGET) as outer:
        result = pick_and_place(env, TARGET)
    release_end = sum(
        p.steps for p in result.phases[: [p.name for p in result.phases].index("release") + 1]
    )
    assert ("fingers", TARGET) in outer.pairs[release_end - 1]  # touching at the endpoint
    assert not result.released and result.label.failure_stage == "release"
    assert "release_failure" in result.label.flags


# --- forces -------------------------------------------------------------------------------


def test_pad_contacts_are_summed_per_substep(env) -> None:
    """Several contact points of one role pair (the finger pads on the target) sum to
    the substep total, which equals the endpoint total at the end of a step."""
    settled(env)
    with StepMonitor(env, TARGET) as monitor:
        grasp = grasp_block_from_above(env, TARGET)
    assert grasp.success
    model, data = env.sim.model._model, env.sim.data._data
    roles, total, points, f = monitor.roles, 0.0, 0, np.zeros(6)
    for i in range(data.ncon):  # the live contact list, read only
        g1, g2 = data.contact.geom[i]
        if sorted((roles[g1], roles[g2])) == ["fingers", TARGET]:
            mujoco.mj_contactForce(model, data, i, f)
            total += abs(float(f[0]))
            points += 1
    assert points >= 2 and total > 0  # deliberately loaded
    ft = key(monitor, "fingers", TARGET)
    assert monitor.substep_pairs[-1][ft] == total == monitor.forces[-1][("fingers", TARGET)]
    peak = max(s.get(ft, 0.0) for s in monitor.substep_pairs)
    assert peak >= max(f.get(("fingers", TARGET), 0.0) for f in monitor.forces)


# --- coverage, determinism and the missed impacts -----------------------------------------


def sampled(seed: int):
    return trials.Scene(f"sampled{seed}", seed, dict(sample_layout(CONFIG, seed)))


def check_coverage(result) -> None:
    history = {(c.pair, c.phase): c for c in result.evidence.contacts}
    for c in result.endpoint_contacts:
        h = history[(c.pair, c.phase)]
        assert (h.unintended, h.kind) == (c.unintended, c.kind)
        assert h.first_step <= c.first_step and h.steps >= c.steps and h.substeps >= h.steps
        assert h.peak_force >= c.peak_force
    for o, e in zip(result.objects, result.endpoint_objects, strict=True):
        assert o.max_displacement >= e.max_displacement and o.tilt_max >= e.tilt_max
        assert o.final_position.tobytes() == e.final_position.tobytes()
        assert (o.resting_face_final, o.toppled, o.tilt_final) == (
            e.resting_face_final,
            e.toppled,
            e.tilt_final,
        )


@pytest.mark.parametrize("seed", sorted(KNOWN_MISSES))
def test_known_missed_impacts_are_recorded(env, seed) -> None:
    result, _ = trials.attempt(env, sampled(seed), trials.legacy_action)
    check_coverage(result)
    hits = [c for c in result.evidence.contacts if c.pair == "distractor-target"]
    assert hits and all(c.unintended and c.substeps > 0 for c in hits)
    assert "collision_object" in result.label.flags
    assert not any(c.pair == "distractor-target" for c in result.endpoint_contacts)
    assert "collision_object" not in result.endpoint_label.flags
    assert (result.label.y, result.label.failure_stage) == KNOWN_MISSES[seed]
    assert (result.endpoint_label.y, result.endpoint_label.failure_stage) == KNOWN_MISSES[seed]


def same_outcome(a, b) -> bool:
    return all(np.array_equal(getattr(a, f.name), getattr(b, f.name)) for f in fields(type(a)))


def test_same_scene_and_action_repeat_with_identical_history(env) -> None:
    scene = sampled(3)
    runs = []
    for e in (env, env, None):
        fresh = e is None
        e = make_tabletop_env(CONFIG, seed=0) if fresh else e
        result, _ = trials.attempt(e, scene, trials.legacy_action)
        runs.append((result, e.sim.data.qpos.copy()))
        if fresh:
            e.close()
    first, qpos = runs[0]
    for result, q in runs[1:]:
        assert np.array_equal(q, qpos)
        assert result.evidence.contacts == first.evidence.contacts  # values and order
        assert result.endpoint_contacts == first.endpoint_contacts
        assert all(same_outcome(a, b) for a, b in zip(result.objects, first.objects, strict=True))
        assert result.label == first.label


def test_a_transient_excursion_between_endpoints_is_measured(env) -> None:
    """The distractor is lifted, shifted 5 cm and turned 30 degrees for one substep in
    the middle of a control step, then put back: only the substep stream sees it."""
    settled(env)
    data = env.sim.data
    joint = env.blocks[DISTRACTOR].joints[0]
    step2, count, saved = env.sim.step2, [0], {}
    k = 2 * substeps_per_step(env) + 10  # mid-interval of control step 3 (the open phase)

    def excursion():
        step2()
        count[0] += 1
        if count[0] == k:
            saved["qpos"] = data.get_joint_qpos(joint).copy()
            saved["qvel"] = data.get_joint_qvel(joint).copy()
            turn = np.radians(30)
            pose = saved["qpos"] + [0.05, 0.0, 0.05, 0, 0, 0, 0]
            pose[3:] = [np.cos(turn / 2), np.sin(turn / 2), 0.0, 0.0]
            data.set_joint_qpos(joint, pose)
        elif count[0] == k + 1:
            data.set_joint_qpos(joint, saved["qpos"])
            data.set_joint_qvel(joint, saved["qvel"])

    env.sim.step2 = excursion
    try:
        result = pick_and_place(env, TARGET)
    finally:
        del env.sim.step2
    d = next(o for o in result.objects if o.name == DISTRACTOR)
    e = next(o for o in result.endpoint_objects if o.name == DISTRACTOR)
    assert e.max_displacement < 1e-3 and e.tilt_max < np.radians(1)
    assert d.max_displacement > 0.049 and d.tilt_max > np.radians(29.9)
    assert d.final_displacement < 1e-3 and not d.toppled
    assert "displaced" in result.label.flags and result.label.y == 0
    assert "displaced" not in result.endpoint_label.flags


def test_invalid_start_is_rejected_before_any_substep(env) -> None:
    settled(env)
    env.sim.data.set_joint_qvel(env.blocks[DISTRACTOR].joints[0], [0.006, 0, 0, 0, 0, 0])
    env.sim.forward()  # test setup: make the velocity visible to the state readers

    def forbidden(*args):
        raise AssertionError("a substep ran before the start was validated")

    env.sim.step2 = env.sim.step = forbidden
    try:
        with pytest.raises(ValueError, match="distractor moving at 0.0060 m/s"):
            pick_and_place(env, TARGET)
    finally:
        del env.sim.step2, env.sim.step
    assert hooks(env) == (None, None, None)
