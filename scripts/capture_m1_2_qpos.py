from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import mujoco
import numpy as np
import robosuite

from robot_uncertainty.sim.pick_place import pick_and_place
from robot_uncertainty.sim.primitives import settle
from robot_uncertainty.sim.tabletop import make_tabletop_env

ROOT = Path(__file__).resolve().parents[1]
TARGET = "target"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


# A regression reference for the M1.2 primitive (tests/sim/test_m1_2_regression.py).
# Each seed runs as in run_pick_place.py: reset to the scene seed, settle, pick and
# place, in one environment built with seed 0. The file records the MuJoCo, robosuite
# and NumPy versions and the git commit, because bitwise results hold only for the same
# build on the same kind of CPU.
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save final qpos after pick-and-place for regression checks."
    )
    parser.add_argument("--first-seed", type=int, default=1000, help="first scene seed")
    parser.add_argument("--seeds", type=int, default=10, help="number of consecutive seeds")
    parser.add_argument("--out", type=Path, default=Path("results/m1_2_reference_qpos.npz"))
    parser.add_argument("--overwrite", action="store_true", help="replace an existing --out")
    args = parser.parse_args()
    if args.out.exists() and not args.overwrite:
        parser.error(f"{args.out} exists; pass --overwrite to replace it")

    seeds = np.arange(args.first_seed, args.first_seed + args.seeds)
    env = make_tabletop_env(seed=0)
    qpos = []
    for seed in seeds:
        env.reset(scene_seed=int(seed))
        settle(env)
        result = pick_and_place(env, TARGET)
        qpos.append(env.sim.data.qpos.copy())
        print(f"seed {seed}: success {result.success}, {result.control_steps} steps")
    env.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        seeds=seeds,
        qpos=np.array(qpos),
        mujoco_version=mujoco.__version__,
        robosuite_version=robosuite.__version__,
        numpy_version=np.__version__,
        commit=git("rev-parse", "HEAD"),
        # Uncommitted changes to tracked code at capture time (empty for a clean tree).
        dirty=git("status", "--porcelain", "--untracked-files=no", "--", "src", "scripts", "tests"),
    )
    print(f"Wrote {len(seeds)} final states to {args.out}")


if __name__ == "__main__":
    main()
