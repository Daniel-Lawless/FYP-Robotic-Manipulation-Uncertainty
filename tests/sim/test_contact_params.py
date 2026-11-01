import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("robosuite")
import mujoco  # noqa: E402

from robot_uncertainty.sim.contact_params import (  # noqa: E402
    ContactParamRecorder,
    expected_contact_params,
    geom_friction,
    matches_expected,
    measured_contacts,
    model_contact_settings,
)
from robot_uncertainty.sim.pick_place import pick_and_place  # noqa: E402
from robot_uncertainty.sim.primitives import settle  # noqa: E402
from robot_uncertainty.sim.tabletop import make_tabletop_env  # noqa: E402

TARGET = "target"
PAD = "gripper0_right_finger1_pad_collision"

# Boxes 1 mm into a floor (so their contacts are active constraints), each set up to
# exercise one rule.
RULES_XML = """
<mujoco>
  <option {option}/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1" friction="0.7 0.01 0.001"
          solref="0.02 1" solimp="0.9 0.95 0.001 0.5 2" solmix="1"/>
    <body name="hi" pos="-0.6 0 0.049"><freejoint/>
      <geom name="hi" type="box" size="0.05 0.05 0.05" priority="1" condim="1"
            friction="0.2 0.02 0.002" solref="0.05 0.8" solimp="0.8 0.9 0.002 0.4 3"/>
    </body>
    <body name="mix" pos="-0.3 0 0.049"><freejoint/>
      <geom name="mix" type="box" size="0.05 0.05 0.05" condim="6" solmix="3"
            friction="1.5 0.001 0.0001" solref="0.01 0.5" solimp="0.95 0.99 0.003 0.6 4"/>
    </body>
    <body name="direct" pos="0 0 0.049"><freejoint/>
      <geom name="direct" type="box" size="0.05 0.05 0.05" solref="-2000 -40"/>
    </body>
    <body name="paired" pos="0.3 0 0.049"><freejoint/>
      <geom name="paired" type="box" size="0.05 0.05 0.05"/>
    </body>
    <body name="excluded" pos="0.6 0 0.049"><freejoint/>
      <geom name="excluded" type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
  <contact>
    <pair geom1="paired" geom2="floor" condim="4" friction="0.3 0.3 0.01 0.002 0.002"
          solref="0.03 0.9" solimp="0.85 0.9 0.001 0.5 2"/>
    <exclude body1="world" body2="excluded"/>
  </contact>
</mujoco>
"""


def rules_model(option: str = ""):
    model = mujoco.MjModel.from_xml_string(RULES_XML.format(option=option))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def by_body(model, data) -> dict[str, list]:
    out: dict[str, list] = {}
    for c in measured_contacts(data):
        out.setdefault(model.body(model.geom_bodyid[c.geom2]).name, []).append(c)
    return out


@pytest.mark.parametrize(
    "option",
    [
        "",
        'cone="elliptic" impratio="7"',
        'o_solref="0.04 1.2" o_friction="0.4 0.4 0.004 0.0004 0.0004" o_margin="0"',
    ],
)
def test_rules_match_mujoco_on_a_synthetic_model(option) -> None:
    model, data = rules_model(option)
    if "o_" in option:  # the override must also be switched on
        model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_OVERRIDE
        mujoco.mj_forward(model, data)
    contacts = by_body(model, data)
    assert set(contacts) == {"hi", "mix", "direct", "paired"}  # "excluded": no contact
    for cs in contacts.values():
        for c in cs:
            assert matches_expected(model, c) == []
    floor, hi, mix = (model.geom(n).id for n in ("floor", "hi", "mix"))
    excluded = model.geom("excluded").id
    assert expected_contact_params(model, floor, excluded).source == "excluded"
    for cs in contacts.values():
        assert all(c.active for c in cs)
    if "o_" not in option:
        assert expected_contact_params(model, floor, hi).condim == 1  # priority wins
        p = expected_contact_params(model, floor, mix)
        assert p.condim == 6 and np.allclose(p.friction, [1.5, 1.5, 0.01, 0.001, 0.001])
        assert np.allclose(p.solref, 0.25 * np.array([0.02, 1]) + 0.75 * np.array([0.01, 0.5]))
        direct = expected_contact_params(model, floor, model.geom("direct").id)
        assert np.allclose(direct.solref, [-2000, -40])  # element-wise min with 0.02, 1
        assert expected_contact_params(model, floor, model.geom("paired").id).source == "pair"
    else:
        assert expected_contact_params(model, floor, hi).source == "geom+override"


@pytest.fixture(scope="module")
def env():
    env = make_tabletop_env(seed=0)
    yield env
    env.close()


def record_attempt(env):
    env.reset(scene_seed=0)
    settle(env)
    with ContactParamRecorder(env) as recorder:
        result = pick_and_place(env, TARGET)
    return result, recorder


def test_settings_are_measured_from_the_compiled_model(env) -> None:
    model = env.sim.model._model
    settings = model_contact_settings(model)
    assert settings["npair"] == model.npair and settings["nexclude"] == model.nexclude
    expected = bool(model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_OVERRIDE)
    assert settings["override"] is expected


def test_every_contact_of_a_pick_and_place_follows_the_rules(env) -> None:
    result, recorder = record_attempt(env)
    assert result.success
    categories = {r.category for r in recorder.pairs.values()}
    assert {"finger pad-target", "finger mesh-target", "table-target"} <= categories
    for record in recorder.pairs.values():
        assert record.mismatches == (), record.names


def test_target_friction_change_is_inert_and_restored(env) -> None:
    model = env.sim.model._model
    target = model.geom(env.blocks[TARGET].contact_geoms[0]).id
    original = model.geom_friction.copy()
    _, baseline = record_attempt(env)
    with geom_friction(model, [target], (0.5, 0.005, 0.0001)):
        assert np.allclose(model.geom_friction[target], [0.5, 0.005, 0.0001])
        _, low = record_attempt(env)
    assert np.array_equal(model.geom_friction, original)
    for names in (tuple(sorted((PAD, "target_g0"))), ("table_collision", "target_g0")):
        assert np.array_equal(low.pairs[names].first.friction, baseline.pairs[names].first.friction)
        assert low.pairs[names].mismatches == ()
    with pytest.raises(RuntimeError), geom_friction(model, [target], (0.1, 0.0, 0.0)):
        raise RuntimeError("restored even on an error")
    assert np.array_equal(model.geom_friction, original)
