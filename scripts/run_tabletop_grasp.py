from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from robot_uncertainty.sim.primitives import settle
from robot_uncertainty.sim.tabletop import grasp_block_from_above, make_tabletop_env
from robot_uncertainty.utils.datasets import write_rows

TARGET = "target"


def trial(env, scene_seed: int, grasp_height: float | None) -> dict:
    """One seeded scene: reset, settle, observe, top-grasp the target."""
    start = time.perf_counter()
    env.reset(scene_seed=scene_seed)
    reset_s = time.perf_counter() - start
    settle(env)
    placed = env.layout[TARGET]
    result = grasp_block_from_above(env, TARGET, grasp_height)
    return dict(
        scene_seed=scene_seed,
        x_cm=100 * placed.x,
        y_cm=100 * placed.y,
        yaw_deg=np.degrees(placed.yaw),
        grasp_height_mm=1000 * result.grasp_height,
        success=result.success,
        timeouts=";".join(p.name for p in result.phases if not p.reached),
        wrist_deg=np.degrees(result.wrist_rotation),
        yaw_err_close_deg=np.degrees(result.yaw_error),
        gap_mm=1000 * result.finger_gap,
        lift_cm=100 * result.lift_height,
        steps=result.control_steps,
        wall_s=result.wall_time,
        reset_wall_s=reset_s,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seeded top grasps of the 4 x 4 x 8 cm target in the custom tabletop scene."
    )
    parser.add_argument("--seeds", type=int, default=50, help="scene seeds 0..N-1")
    parser.add_argument(
        "--grasp-height",
        type=float,
        default=None,
        help="grasp point above the block centre in metres (default: from the observed "
        "block height, see top_grasp_height; 0 grips at the centre)",
    )
    parser.add_argument("--out", type=Path, default=Path("results/tabletop_grasp.csv"))
    parser.add_argument("--overwrite", action="store_true", help="replace an existing --out")
    args = parser.parse_args()
    if args.out.exists() and not args.overwrite:
        parser.error(f"{args.out} exists; pass --overwrite to replace it")

    env = make_tabletop_env(seed=0)
    rows = [trial(env, seed, args.grasp_height) for seed in range(args.seeds)]
    env.close()
    write_rows(args.out, rows, overwrite=args.overwrite)

    ok = [r for r in rows if r["success"]]
    failed = [r["scene_seed"] for r in rows if not r["success"]]
    timeouts = [r["scene_seed"] for r in rows if r["timeouts"]]

    def span(key: str, fmt: str = ".1f") -> str:
        values = np.array([r[key] for r in rows])
        return f"{values.min():{fmt}}-{values.max():{fmt}} (median {np.median(values):{fmt}})"

    print(f"Grasp height {rows[0]['grasp_height_mm']:.1f} mm above the block centre")
    print(f"Verified grasps: {len(ok)}/{len(rows)}; failed seeds: {failed or 'none'}")
    print(f"Scenes with phase timeouts: {timeouts or 'none'}")
    print(f"Lift (cm): {span('lift_cm', '.2f')}; finger gap (mm): {span('gap_mm')}")
    print(f"Control steps: {span('steps', '.0f')}; wall s per grasp: {span('wall_s', '.2f')}")
    print(f"Reset wall s: {span('reset_wall_s', '.4f')}")
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
