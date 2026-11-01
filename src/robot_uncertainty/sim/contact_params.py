from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import mujoco
import numpy as np

MINVAL = mujoco.mjMINVAL


@dataclass(frozen=True)
class ContactParams:
    source: str  # "geom", "pair" or "excluded"; "+override" if the global override applies
    condim: int
    friction: np.ndarray  # (5,)
    solref: np.ndarray  # (2,)
    solreffriction: np.ndarray  # (2,)
    solimp: np.ndarray  # (5,)
    margin: float
    gap: float


def override_enabled(model: mujoco.MjModel) -> bool:
    return bool(model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_OVERRIDE)


def model_contact_settings(model: mujoco.MjModel) -> dict:
    """The model-wide settings that decide how contact parameters are formed."""
    return dict(
        npair=int(model.npair),
        nexclude=int(model.nexclude),
        override=override_enabled(model),
        cone="elliptic" if model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC else "pyramidal",
        impratio=float(model.opt.impratio),
    )


def _expand(friction: np.ndarray) -> np.ndarray:
    return np.array([friction[0], friction[0], friction[1], friction[2], friction[2]])


def _excluded(model: mujoco.MjModel, g1: int, g2: int) -> bool:
    """Whether the geoms' bodies form an <exclude> pair (signature (b1 << 16) + b2)."""
    b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
    signatures = {(b1 << 16) + b2, (b2 << 16) + b1}
    return bool(set(model.exclude_signature[: model.nexclude].tolist()) & signatures)


def _pair(model: mujoco.MjModel, g1: int, g2: int) -> int | None:
    for i in range(model.npair):
        if {int(model.pair_geom1[i]), int(model.pair_geom2[i])} == {g1, g2}:
            return i
    return None


# MuJoCo does not use one geom's friction or solver settings for a contact; it combines
# both geoms' (and can replace them entirely), so changing one object's friction may
# change nothing. expected_contact_params reproduces the rules of the installed MuJoCo,
# checked in this order:
# 1. global override (opt.enableflags & mjENBL_OVERRIDE): margin, solref, solimp and
#    friction come from opt.o_margin, o_solref, o_solimp and o_friction for every
#    contact (condim and gap are kept);
# 2. explicit <pair> definitions: the pair's own condim, friction, solref,
#    solreffriction, solimp, margin and gap, with no mixing;
# 3. excluded body pairs (<exclude>): no contact at all;
# 4. ordinary geom mixing (mj_contactParam): if the priorities differ, the
#    higher-priority geom's condim, friction, solref and solimp; otherwise condim is the
#    larger, friction the element-wise maximum, solimp the solmix-weighted mean
#    (mix = solmix1 / (solmix1 + solmix2), 0.5 if both are zero, all of the non-zero one
#    if one is), and solref the same weighted mean if both are in standard form
#    (solref[0] > 0) but the element-wise minimum otherwise (direct form). Margin and gap
#    are the larger of the two either way. Geom contacts have no solreffriction (zeros).
def expected_contact_params(model: mujoco.MjModel, g1: int, g2: int) -> ContactParams:
    """The parameters MuJoCo should give a contact between geoms g1 and g2 (see the
    comment above for the rules and their order)."""
    pair = _pair(model, g1, g2)
    if pair is not None:
        params = ContactParams(
            "pair",
            int(model.pair_dim[pair]),
            model.pair_friction[pair].copy(),
            model.pair_solref[pair].copy(),
            model.pair_solreffriction[pair].copy(),
            model.pair_solimp[pair].copy(),
            float(model.pair_margin[pair]),
            float(model.pair_gap[pair]),
        )
    elif _excluded(model, g1, g2):
        none = np.full(5, np.nan)
        return ContactParams("excluded", 0, none, none[:2], none[:2], none, 0.0, 0.0)
    else:
        params = _mix(model, g1, g2)
    if override_enabled(model):
        o = model.opt
        params = ContactParams(
            params.source + "+override",
            params.condim,
            np.asarray(o.o_friction, dtype=float).copy(),
            np.asarray(o.o_solref, dtype=float).copy(),
            np.asarray(o.o_solref, dtype=float).copy(),  # measured: also solreffriction
            np.asarray(o.o_solimp, dtype=float).copy(),
            float(o.o_margin),
            params.gap,
        )
    return params


def _mix(model: mujoco.MjModel, g1: int, g2: int) -> ContactParams:
    p1, p2 = int(model.geom_priority[g1]), int(model.geom_priority[g2])
    margin = float(max(model.geom_margin[g1], model.geom_margin[g2]))
    gap = float(max(model.geom_gap[g1], model.geom_gap[g2]))
    if p1 != p2:
        g = g1 if p1 > p2 else g2
        return ContactParams(
            "geom",
            int(model.geom_condim[g]),
            _expand(model.geom_friction[g]),
            model.geom_solref[g].copy(),
            np.zeros(2),
            model.geom_solimp[g].copy(),
            margin,
            gap,
        )
    s1, s2 = float(model.geom_solmix[g1]), float(model.geom_solmix[g2])
    if s1 >= MINVAL and s2 >= MINVAL:
        mix = s1 / (s1 + s2)
    elif s1 < MINVAL and s2 < MINVAL:
        mix = 0.5
    else:
        mix = 0.0 if s1 < MINVAL else 1.0
    r1, r2 = model.geom_solref[g1], model.geom_solref[g2]
    if r1[0] > 0 and r2[0] > 0:
        solref = mix * r1 + (1 - mix) * r2
    else:
        solref = np.minimum(r1, r2)
    return ContactParams(
        "geom",
        int(max(model.geom_condim[g1], model.geom_condim[g2])),
        _expand(np.maximum(model.geom_friction[g1], model.geom_friction[g2])),
        np.asarray(solref, dtype=float),
        np.zeros(2),
        mix * model.geom_solimp[g1] + (1 - mix) * model.geom_solimp[g2],
        margin,
        gap,
    )


# A contact's friction has five entries: sliding, sliding, torsional, rolling, rolling,
# expanded from a geom's three. mjContact.mu is the friction coefficient of the
# regularised cone that the solver uses; contact_mu records it as measured.
def expected_mu(model: mujoco.MjModel, params: ContactParams) -> float:
    """mjContact.mu for these parameters once the contact is an active constraint (it
    is zero before): zero for a frictionless contact (condim 1), otherwise the sliding
    friction scaled by 1 / sqrt(opt.impratio), as measured on MuJoCo 3.9.0 with both
    cone types (impratio is 20 in robosuite's Panda scenes, so mu is 0.224 for
    friction 1)."""
    if params.condim == 1:
        return 0.0
    return float(params.friction[0] / np.sqrt(model.opt.impratio))


@dataclass(frozen=True)
class MeasuredContact:
    """One contact's parameters as MuJoCo formed them."""

    geom1: int
    geom2: int
    condim: int
    friction: np.ndarray
    solref: np.ndarray
    solreffriction: np.ndarray
    solimp: np.ndarray
    mu: float
    includemargin: float
    active: bool  # instantiated as a constraint (efc_address >= 0); only then is mu set


def measured_contacts(data: mujoco.MjData) -> list[MeasuredContact]:
    return [
        MeasuredContact(
            int(c.geom1),
            int(c.geom2),
            int(c.dim),
            np.array(c.friction, dtype=float),
            np.array(c.solref, dtype=float),
            np.array(c.solreffriction, dtype=float),
            np.array(c.solimp, dtype=float),
            float(c.mu),
            float(c.includemargin),
            bool(c.efc_address >= 0),
        )
        for c in data.contact[: data.ncon]
    ]


def matches_expected(
    model: mujoco.MjModel, contact: MeasuredContact, atol: float = 1e-12
) -> list[str]:
    """Fields of a measured contact that differ from expected_contact_params (empty if
    all agree)."""
    expected = expected_contact_params(model, contact.geom1, contact.geom2)
    diffs = []
    if expected.source == "excluded":
        return ["contact exists between excluded bodies"]
    if contact.condim != expected.condim:
        diffs.append(f"condim {contact.condim} vs {expected.condim}")
    for field in ("friction", "solref", "solreffriction", "solimp"):
        got, want = getattr(contact, field), getattr(expected, field)
        if not np.allclose(got, want, rtol=0.0, atol=atol):
            diffs.append(f"{field} {got} vs {want}")
    mu = expected_mu(model, expected)
    if contact.active and not np.isclose(contact.mu, mu, rtol=0.0, atol=atol):
        diffs.append(f"mu {contact.mu} vs {mu}")
    margin = expected.margin - expected.gap
    if not np.isclose(contact.includemargin, margin, rtol=0.0, atol=atol):
        diffs.append(f"includemargin {contact.includemargin} vs {margin}")
    return diffs


@contextmanager
def geom_friction(model: mujoco.MjModel, geoms, friction):
    """Temporarily set the friction of some geoms; always restored on exit."""
    geoms = list(geoms)
    saved = model.geom_friction[geoms].copy()
    try:
        model.geom_friction[geoms] = friction
        yield
    finally:
        model.geom_friction[geoms] = saved


PADS = ("gripper0_right_finger1_pad_collision", "gripper0_right_finger2_pad_collision")
FINGER_MESHES = ("gripper0_right_finger1_collision", "gripper0_right_finger2_collision")


def contact_category(names: tuple[str, str], roles: dict[str, str]) -> str:
    """A label for a contact between two named geoms; roles maps block geom names to
    their block's role ("target" or "distractor")."""

    def kind(name: str) -> str:
        if name in roles:
            return roles[name]
        if name in PADS:
            return "finger pad"
        if name in FINGER_MESHES:
            return "finger mesh"
        if name == "table_collision":
            return "table"
        return name

    return "-".join(sorted(kind(n) for n in names))


@dataclass
class PairRecord:
    """Contacts between one pair of geoms over a run."""

    names: tuple[str, str]
    category: str
    first: MeasuredContact
    first_step: int
    steps: int = 0
    mismatches: tuple[str, ...] = ()


class ContactParamRecorder:
    """Inside a with block, records every geom pair's contact parameters after each
    env.step: the first contact seen per pair, and on how many steps it occurred.
    Composes with other env.step wrappers (it restores whatever was there)."""

    def __init__(self, env) -> None:
        self.env = env
        self.model = env.sim.model._model
        self.roles = {
            geom: spec.role
            for spec in env.config.blocks
            for geom in env.blocks[spec.name].contact_geoms
        }
        self.pairs: dict[tuple[str, str], PairRecord] = {}
        self.step_count = 0

    def __enter__(self) -> ContactParamRecorder:
        self._outer = self.env.__dict__.get("step")
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
        self.step_count += 1
        seen = set()
        for contact in measured_contacts(self.env.sim.data._data):
            names = tuple(
                sorted((self.model.geom(contact.geom1).name, self.model.geom(contact.geom2).name))
            )
            if names in seen:
                continue
            seen.add(names)
            record = self.pairs.get(names)
            if record is None:
                record = self.pairs[names] = PairRecord(
                    names,
                    contact_category(names, self.roles),
                    contact,
                    self.step_count,
                    mismatches=tuple(matches_expected(self.model, contact)),
                )
            record.steps += 1
