"""Paired comparison of fixed-orientation and yaw-aligned top grasps on the Lift cube.

For each seed the same initial scene is grasped twice, from two fresh environments:

- fixed: fingers always close along world x (target yaw 0, the original controller's
  TOP_DOWN target), whatever the cube's yaw;
- aligned: fingers close on the face pair needing the least wrist rotation.

The fixed controller is a reconstruction of the original: it uses the original fixed
target, but the current controller code (rotation commands clipped by magnitude rather
than per component), so it is not an exact replay of the original implementation.

The cube's recorded dimensions and true yaw are saved with each row, so the geometric
span the fingers should close to can be computed for either controller. The committed
dataset is data/validation/grasp_alignment.csv (see the README there); by default this
script writes to the git-ignored results/ and never overwrites without --overwrite.

Usage:
    python scripts/compare_grasp_alignment.py [--seeds 50] [--out PATH] [--overwrite]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from robot_uncertainty.sim.lift import make_lift_env, observe_cube
from robot_uncertainty.sim.primitives import settle, top_grasp
from robot_uncertainty.utils.datasets import write_rows


def grasp(seed: int, controller: str) -> dict:
    env = make_lift_env(seed)
    env.reset()
    settle(env)
    cube_pos, cube_yaw = observe_cube(env)
    if controller == "fixed":
        # A full-turn symmetry leaves yaw 0 as the only candidate: the TOP_DOWN target.
        result = top_grasp(env, cube_pos, 0.0, env.cube.root_body, symmetry=2 * np.pi)
    else:
        result = top_grasp(env, cube_pos, cube_yaw, env.cube.root_body)
    width_x, width_y = (2 * s for s in env.cube.size[:2])
    env.close()
    return dict(
        seed=seed,
        controller=controller,
        cube_width_x_mm=1000 * width_x,
        cube_width_y_mm=1000 * width_y,
        cube_yaw_deg=np.degrees(result.true_yaw),
        target_yaw_deg=np.degrees(result.target_yaw),
        wrist_deg=np.degrees(result.wrist_rotation),
        face_axis=result.face_axis,
        success=result.success,
        gap_mm=1000 * result.finger_gap,
        lift_cm=100 * result.lift_height,
        steps=result.control_steps,
        wall_s=result.wall_time,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed vs yaw-aligned top grasps, paired.")
    parser.add_argument("--seeds", type=int, default=50, help="seeds 0..N-1")
    parser.add_argument("--out", type=Path, default=Path("results/grasp_alignment.csv"))
    parser.add_argument("--overwrite", action="store_true", help="replace an existing --out")
    args = parser.parse_args()
    if args.out.exists() and not args.overwrite:
        parser.error(f"{args.out} exists; pass --overwrite or choose another --out")

    rows = [grasp(seed, c) for seed in range(args.seeds) for c in ("fixed", "aligned")]
    write_rows(args.out, rows, overwrite=args.overwrite)

    for controller in ("fixed", "aligned"):
        group = [r for r in rows if r["controller"] == controller]
        gaps = [r["gap_mm"] for r in group]
        print(
            f"{controller:>7}: verified {sum(r['success'] for r in group)}/{len(group)}, "
            f"gap {min(gaps):.1f}-{max(gaps):.1f} mm, "
            f"steps {np.mean([r['steps'] for r in group]):.1f}"
        )
    print(f"Rows written to {args.out}")


if __name__ == "__main__":
    main()
