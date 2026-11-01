from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from robot_uncertainty.sim import distractor_trials as trials
from robot_uncertainty.sim.actions import grasp_candidates, legacy_equivalent
from robot_uncertainty.sim.contact_params import (
    ContactParamRecorder,
    geom_friction,
    model_contact_settings,
)
from robot_uncertainty.sim.pick_place import pick_and_place
from robot_uncertainty.sim.primitives import settle
from robot_uncertainty.sim.scene import (
    DISTRACTOR_SHORT,
    BlockSpec,
    Placement,
    TabletopConfig,
    place_beside,
    sample_layout,
)
from robot_uncertainty.sim.tabletop import make_tabletop_env
from robot_uncertainty.utils.datasets import write_rows

TARGET = "target"
LOW_FRICTION = (0.5, 0.005, 0.0001)
CATEGORIES = (
    "finger pad-target",
    "finger mesh-target",
    "table-target",
    "distractor-table",
    "distractor-target",
)


def fmt(values) -> str:
    return ";".join(f"{v:g}" for v in np.ravel(values))


def touching_layout(env, seed: int = 0) -> dict[str, Placement]:
    """The seed's target with the short distractor touching it, on the face pair the
    legacy grasp does not close on."""
    target = sample_layout(TabletopConfig(), seed)[TARGET]
    env.reset(scene_seed=seed)
    candidates = grasp_candidates(trials.ObservedYaw(TARGET, target.yaw))
    legacy = legacy_equivalent(candidates, env.observe().eef_yaw)
    other = candidates[1] if legacy is candidates[0] else candidates[0]
    specs = {b.name: b for b in env.config.blocks}
    place = place_beside(
        target, specs[TARGET], specs["distractor"], 0.0, other.closing_heading, target.yaw
    )
    return {TARGET: target, "distractor": place}


def run(env, name: str, layout: dict[str, Placement] | None) -> list[dict]:
    env.reset(scene_seed=0)
    if layout:
        env.set_layout(layout)
    settle(env)
    problems = env.check_initial_state()
    if problems:
        raise RuntimeError(f"{name}: invalid initial state: {problems}")
    model = env.sim.model._model
    with ContactParamRecorder(env) as recorder:
        result = pick_and_place(env, TARGET)
    settings = model_contact_settings(model)
    rows = []
    for record in recorder.pairs.values():
        c = record.first
        row = dict(
            run=name,
            category=record.category,
            geom1=record.names[0],
            geom2=record.names[1],
            first_step=record.first_step,
            steps=record.steps,
            condim=c.condim,
            friction=fmt(c.friction),
            solref=fmt(c.solref),
            solreffriction=fmt(c.solreffriction),
            solimp=fmt(c.solimp),
            mu=round(c.mu, 6),
            includemargin=c.includemargin,
            matches_rules=not record.mismatches,
            mismatch="; ".join(record.mismatches),
        )
        for k, geom in enumerate(record.names, start=1):
            g = model.geom(geom).id
            row.update(
                {
                    f"geom{k}_friction": fmt(model.geom_friction[g]),
                    f"geom{k}_solmix": float(model.geom_solmix[g]),
                    f"geom{k}_priority": int(model.geom_priority[g]),
                    f"geom{k}_condim": int(model.geom_condim[g]),
                    f"geom{k}_solref": fmt(model.geom_solref[g]),
                }
            )
        row.update(settings)
        row["attempt_success"] = result.success
        row["attempt_y"] = result.label.y
        rows.append(row)
    return rows


# Checks the measured parameters against the mixing rules
# (robot_uncertainty.sim.contact_params). Runs: the legacy pick-and-place of scene seed 0;
# the same target with a short distractor touching it face to face (on the axis the
# fingers do not close along); and both again with the target's friction temporarily set
# to LOW_FRICTION, which is restored afterwards. One CSV row per run and geom pair.
def main() -> None:
    parser = argparse.ArgumentParser(description="Measure effective MuJoCo contact parameters.")
    parser.add_argument("--out", type=Path, default=Path("results/contact_params.csv"))
    parser.add_argument("--overwrite", action="store_true", help="replace an existing --out")
    args = parser.parse_args()
    if args.out.exists() and not args.overwrite:
        parser.error(f"{args.out} exists; pass --overwrite to replace it")

    single = make_tabletop_env(seed=0)
    pair = make_tabletop_env(TabletopConfig(blocks=(BlockSpec(TARGET), DISTRACTOR_SHORT)), seed=0)
    layout = touching_layout(pair)
    rows = run(single, "legacy", None) + run(pair, "distractor", layout)
    for env, name, lay in ((single, "legacy", None), (pair, "distractor", layout)):
        model = env.sim.model._model
        geoms = [model.geom(g).id for g in env.blocks[TARGET].contact_geoms]
        before = model.geom_friction.copy()
        with geom_friction(model, geoms, LOW_FRICTION):
            rows += run(env, f"{name}_target_friction_0.5", lay)
        assert np.array_equal(model.geom_friction, before), "friction not restored"
    single.close()
    pair.close()
    write_rows(args.out, rows, overwrite=args.overwrite)

    print(
        f"npair {rows[0]['npair']}, nexclude {rows[0]['nexclude']}, global override "
        f"{rows[0]['override']}, cone {rows[0]['cone']}, impratio {rows[0]['impratio']}"
    )
    for r in rows:
        if r["category"] in CATEGORIES:
            print(
                f"{r['run']:<32} {r['category']:<20} {r['geom1']}/{r['geom2']}: condim "
                f"{r['condim']}, friction {r['friction']}, mu {r['mu']}, solref {r['solref']}, "
                f"{r['steps']} steps, matches rules {r['matches_rules']}"
            )
    missing = {
        run: sorted(set(CATEGORIES) - {r["category"] for r in rows if r["run"] == run})
        for run in dict.fromkeys(r["run"] for r in rows)
    }
    print(f"Categories not seen: {missing}")
    print(f"Every contact matches the rules: {all(r['matches_rules'] for r in rows)}")
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
