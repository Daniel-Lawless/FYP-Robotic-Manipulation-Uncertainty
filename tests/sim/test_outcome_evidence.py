import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("robosuite")
import robot_uncertainty.sim.pick_place as pick_place  # noqa: E402
from robot_uncertainty.sim.primitives import GRIPPER_CLOSED, GRIPPER_OPEN, settle  # noqa: E402
from robot_uncertainty.sim.tabletop import make_tabletop_env  # noqa: E402

TARGET = "target"


@pytest.fixture(scope="module")
def env():
    env = make_tabletop_env(seed=0)
    yield env
    env.close()


def run(env, seed: int = 0, **kwargs):
    env.reset(scene_seed=seed)
    settle(env)
    return pick_place.pick_and_place(env, TARGET, **kwargs)


def check_evidence(result) -> None:
    """The evidence restates what pick_and_place measured."""
    e = result.evidence
    assert e.grasp is result.grasp and e.grasp_success == result.grasp_success
    assert e.phases == tuple((p.name, p.reached) for p in result.phases)
    assert e.placement is result.placement
    assert e.released == result.released and e.dropped == result.dropped
    assert tuple(c for c in e.contacts if c.unintended) == result.unintended_contacts
    assert e.objects == result.objects
    assert all(c.peak_force > 0 for c in e.contacts)


def test_success(env) -> None:
    result = run(env)
    check_evidence(result)
    assert result.success and result.label.y == 1 and result.label.failure_stage == ""
    assert result.label.flags == {}
    expected = {("fingers-target", "close"), ("table-target", "open"), ("table-target", "release")}
    assert expected <= {(c.pair, c.phase) for c in result.evidence.contacts if not c.unintended}


def test_off_target_placement_is_a_placement_failure(env) -> None:
    env.reset(scene_seed=0)
    settle(env)
    result = pick_place.pick_and_place(env, TARGET, place_at=env.goal_position[:2] + [0.04, 0.0])
    check_evidence(result)
    assert result.failure_reason.startswith("outside the goal by")
    assert result.label.failure_stage == "placement" and result.label.y == 0
    assert set(result.label.flags) == {"placement_failure"}


def test_gripper_opened_during_transport_is_a_transport_failure(env, monkeypatch) -> None:
    monkeypatch.setattr(pick_place, "GRIPPER_CLOSED", GRIPPER_OPEN)
    result = run(env)
    check_evidence(result)
    assert result.failure_reason == "dropped during carry"
    assert result.label.failure_stage == "transport"
    assert result.label.flags["dropped"] == ("carry",)
    assert result.evidence.drop_phase == "carry"
    assert "collision_environment" in result.label.flags  # the falling block hits the table


def test_gripper_not_released_is_a_release_failure(env, monkeypatch) -> None:
    monkeypatch.setattr(pick_place, "GRIPPER_OPEN", GRIPPER_CLOSED)
    result = run(env)
    check_evidence(result)
    assert result.failure_reason == "not released: the block still touches the gripper"
    assert result.label.failure_stage == "release"
    assert {"release_failure", "placement_failure"} <= set(result.label.flags)


def test_every_block_starts_on_its_bottom_face(env) -> None:
    for seed in (0, 3):
        result = run(env, seed)
        (target,) = result.objects
        assert target.resting_face_start == target.resting_face_final == "-z"
        assert np.isclose(target.final_position[2], result.placement.position[2])
