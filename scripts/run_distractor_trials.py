from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np

from robot_uncertainty.sim import distractor_trials as trials
from robot_uncertainty.sim.scene import (
    DISTRACTOR_SHORT,
    DISTRACTOR_TALL,
    BlockSpec,
    TabletopConfig,
    sample_layout,
)
from robot_uncertainty.sim.tabletop import make_tabletop_env
from robot_uncertainty.utils.datasets import write_rows

EXPERIMENTS = ("axes", "far", "beside", "carry", "throughput")


def env_with(distractor: BlockSpec | None):
    blocks = (BlockSpec("target"),) + ((distractor,) if distractor else ())
    return make_tabletop_env(TabletopConfig(blocks=blocks), seed=0)


def axes() -> list[dict]:
    """Both explicit candidates on every isolated scene (Test 3)."""
    env, rows = env_with(None), []
    for scene in trials.isolated_scenes():
        for index in (0, 1):
            result, action = trials.attempt(env, scene, trials.candidate(index))
            rows.append(trials.row(scene, f"candidate{index}", result, action))
    env.close()
    return rows


def far() -> list[dict]:
    """A tall distractor far from the hand's swept footprint, legacy-equivalent action."""
    env, rows = env_with(DISTRACTOR_TALL), []
    for scene, distance in trials.far_scenes(env):
        result, action = trials.attempt(env, scene, trials.legacy_action)
        r = trials.row(scene, "legacy", result, action)
        r["hand_clearance_mm"] = round(1000 * distance, 1)
        rows.append(r)
    env.close()
    return rows


def beside(seeds=range(10)) -> list[dict]:
    """A distractor 4 cm along candidate A's closing axis; both candidates (Test 5)."""
    rows = []
    for name, spec in (("tall", DISTRACTOR_TALL), ("short", DISTRACTOR_SHORT)):
        env = env_with(spec)
        for seed in seeds:
            scene, a = trials.beside_scene(env, seed, axis_candidate=seed % 2)
            for label, index in (("A", seed % 2), ("B", 1 - seed % 2)):
                result, action = trials.attempt(env, scene, trials.candidate(index))
                assert label != "A" or np.isclose(action.closing_heading, a.closing_heading)
                rows.append(trials.row(scene, f"{name}-{label}", result, action))
        env.close()
    return rows


def carry() -> list[dict]:
    """A tall distractor halfway along the straight carry path (Test 6)."""
    env, rows = env_with(DISTRACTOR_TALL), []
    for scene in trials.carry_path_scenes():
        result, action = trials.attempt(env, scene, trials.legacy_action)
        rows.append(trials.row(scene, "legacy", result, action))
    env.close()
    return rows


def throughput(attempts: int = 50) -> list[dict]:
    """Attempts per second with one tall distractor placed by the existing random
    sampler (seeds 0-49), legacy action, no rendering."""
    env, rows = env_with(DISTRACTOR_TALL), []
    start = time.perf_counter()
    for seed in range(attempts):
        scene = trials.Scene(f"sampled{seed}", seed, dict(sample_layout(env.config, seed)))
        result, action = trials.attempt(env, scene, trials.legacy_action)
        rows.append(trials.row(scene, "legacy", result, action))
    elapsed = time.perf_counter() - start
    env.close()
    wall = np.array([r["wall_s"] for r in rows])
    print(
        f"Throughput: {attempts} attempts in {elapsed:.1f} s, {attempts / elapsed:.2f} per "
        f"second including reset and settle; pick_and_place alone {1 / wall.mean():.2f} per "
        f"second (median {np.median(wall):.2f} s)"
    )
    return rows


def summarise(name: str, rows: list[dict]) -> None:
    n = len(rows)
    print(f"\n== {name}: {n} attempts ==")
    by_choice: dict[str, list[dict]] = {}
    for r in rows:
        by_choice.setdefault(r["choice"], []).append(r)
    for choice, group in by_choice.items():
        ok = sum(r["success"] for r in group)
        y = sum(r["y"] for r in group)
        stages = Counter(r["failure_stage"] or "-" for r in group)
        flags = Counter(f for r in group for f in r["flags"].split(";") if f)
        match = sum(r["commanded_face_axis"] == r["face_axis"] for r in group)
        wall = np.array([r["wall_s"] for r in group])
        print(
            f"{choice}: success {ok}/{len(group)}, y=1 {y}/{len(group)}, stages {dict(stages)}, "
            f"commanded = actual face axis {match}/{len(group)}"
        )
        print(f"  flags {dict(flags) or '{}'}")
        print(
            f"  |wrist| {np.abs([r['wrist_deg'] for r in group]).max():.1f} deg max, steps "
            f"{min(r['steps'] for r in group)}-{max(r['steps'] for r in group)}, wall "
            f"{wall.min():.2f}-{wall.max():.2f} s"
        )
        if "d_max_mm" in group[0]:
            d = np.array([r["d_max_mm"] for r in group])
            f = np.array([r["d_final_mm"] for r in group])
            print(
                f"  distractor max displacement {d.min():.1f}-{d.max():.1f} mm, final "
                f"{f.min():.1f}-{f.max():.1f} mm, final faces "
                f"{dict(Counter(r['d_face_final'] for r in group))}"
            )


# Writes one CSV per experiment (default results/m1_3a/<experiment>.csv) and prints a
# summary. Scenes are explicit and noise-free; see robot_uncertainty.sim.distractor_trials.
def main() -> None:
    parser = argparse.ArgumentParser(description="M1.3a grasp-axis and distractor diagnostics.")
    parser.add_argument(
        "--only", action="append", choices=EXPERIMENTS, help="run only this (repeatable)"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/m1_3a"))
    parser.add_argument("--overwrite", action="store_true", help="replace existing CSVs")
    args = parser.parse_args()
    chosen = args.only or list(EXPERIMENTS)
    for name in chosen:
        out = args.out_dir / f"{name}.csv"
        if out.exists() and not args.overwrite:
            parser.error(f"{out} exists; pass --overwrite to replace it")
    for name in chosen:
        rows = globals()[name]()
        summarise(name, rows)
        write_rows(args.out_dir / f"{name}.csv", rows, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
