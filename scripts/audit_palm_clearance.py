from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from robot_uncertainty.sim.contacts import record_grasp
from robot_uncertainty.sim.primitives import settle
from robot_uncertainty.sim.scene import Placement
from robot_uncertainty.sim.tabletop import make_tabletop_env
from robot_uncertainty.utils.datasets import write_rows

TARGET = "target"
# Seeded layouts, then the extreme footprint centres of the workspace at 45 deg yaw.
CORNERS = [(x, y) for x in (-0.07, 0.07) for y in (-0.17, 0.03)]
CONFIGS = [(f"seed{s:02d}", s, None) for s in range(12)] + [
    (f"corner{100 * x:+.0f}{100 * y:+.0f}", 0, Placement(x, y, np.pi / 4)) for x, y in CORNERS
]
STRATEGIES = {"centre": 0.0, "raised": None}  # None: top_grasp_height(observed height)
# 5 cm as in the regression test; 50 cm, used only beyond 5 cm, covers the whole grasp.
PROBE_MARGINS = (0.05, 0.5)


# Grasps the 4 x 4 x 8 cm target in the custom tabletop scene in 16 configurations (the
# layouts of scene seeds 0-11, and the four extreme workspace centres at 45 deg yaw), each
# with two grasp points: the block centre (grasp_height 0, the original behaviour) and
# the raised point from top_grasp_height (+15 mm). After every control step it records
# MuJoCo's signed palm-block distance and the actual palm and finger-pad contacts, using
# robot_uncertainty.sim.contacts (the method of the palm-clearance regression test).
def audit(env, config: str, scene_seed: int, placement, strategy: str) -> list[dict]:
    env.reset(scene_seed=scene_seed)
    if placement is not None:
        env.set_block_pose(TARGET, placement)
    placed = placement or env.layout[TARGET]
    settle(env)
    result, samples = record_grasp(env, TARGET, STRATEGIES[strategy], PROBE_MARGINS)
    phases = [p.name for p in result.phases for _ in range(p.steps)]
    assert len(phases) == len(samples)
    return [
        dict(
            config=config,
            scene_seed=scene_seed,
            x_cm=round(100 * placed.x, 3),
            y_cm=round(100 * placed.y, 3),
            yaw_deg=round(float(np.degrees(placed.yaw)), 3),
            strategy=strategy,
            grasp_height_mm=round(1000 * result.grasp_height, 3),
            success=result.success,
            step=i + 1,
            phase=phase,
            clearance_mm=round(1000 * s.palm_clearance, 4),
            palm_contact=s.palm_contact,
            palm_force_n=round(s.palm_force, 4),
            pad1_force_n=round(s.pad_forces[0], 4),
            pad2_force_n=round(s.pad_forces[1], 4),
        )
        for i, (phase, s) in enumerate(zip(phases, samples, strict=True))
    ]


# The committed dataset is data/validation/palm_clearance.csv (see the README there); by
# default this script writes to the git-ignored results/ and never overwrites without
# --overwrite. There is no wall-clock column, so a rerun must match it exactly.
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step-by-step palm clearance and contacts, centre vs raised top grasp."
    )
    parser.add_argument("--out", type=Path, default=Path("results/palm_clearance.csv"))
    parser.add_argument("--overwrite", action="store_true", help="replace an existing --out")
    args = parser.parse_args()
    if args.out.exists() and not args.overwrite:
        parser.error(f"{args.out} exists; pass --overwrite to replace it")

    env = make_tabletop_env(seed=0)
    rows = []
    for config, scene_seed, placement in CONFIGS:
        for strategy in STRATEGIES:
            grasp = audit(env, config, scene_seed, placement, strategy)
            rows += grasp
            clearance = min(r["clearance_mm"] for r in grasp)
            contacts = sum(r["palm_contact"] for r in grasp)
            print(
                f"{config:13s} {strategy:6s} success={grasp[0]['success']!s:5s} "
                f"steps={len(grasp):3d} min clearance {clearance:6.2f} mm, "
                f"palm-contact steps {contacts}",
                flush=True,
            )
    env.close()
    if not all(np.isfinite(r["clearance_mm"]) for r in rows):
        raise RuntimeError("palm left the widest probe margin; widen PROBE_MARGINS")
    write_rows(args.out, rows, overwrite=args.overwrite)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
