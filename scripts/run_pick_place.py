from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from robot_uncertainty.sim.pick_place import pick_and_place
from robot_uncertainty.sim.primitives import settle
from robot_uncertainty.sim.scene import Placement
from robot_uncertainty.sim.tabletop import make_tabletop_env
from robot_uncertainty.utils.datasets import write_rows

TARGET = "target"
CORNERS = [(x, y) for x in (-0.07, 0.07) for y in (-0.17, 0.03)]


def trial(env, config: str, scene_seed: int, placement: Placement | None) -> dict:
    env.reset(scene_seed=scene_seed)
    if placement is not None:
        env.set_block_pose(TARGET, placement)
    placed = placement or env.layout[TARGET]
    settle(env)
    r = pick_and_place(env, TARGET)
    p = r.placement
    steps = {phase.name: phase.steps for phase in r.phases}
    return dict(
        config=config,
        scene_seed=scene_seed,
        x_cm=round(100 * placed.x, 3),
        y_cm=round(100 * placed.y, 3),
        yaw_deg=round(float(np.degrees(placed.yaw)), 3),
        success=r.success,
        failure_reason=r.failure_reason,
        grasp_success=r.grasp_success,
        timeouts=";".join(r.timeouts),
        dropped=r.dropped,
        released=r.released,
        in_goal=p.in_goal,
        upright=p.upright,
        on_table=p.on_table,
        at_rest=p.at_rest,
        centre_error_mm=round(1000 * p.centre_error, 3),
        goal_margin_mm=round(1000 * p.goal_margin, 3),
        tilt_deg=round(float(np.degrees(p.tilt)), 4),
        support_gap_mm=round(1000 * p.support_gap, 3),
        linear_speed_mm_s=round(1000 * p.linear_speed, 4),
        angular_speed_rad_s=round(p.angular_speed, 5),
        max_slip_mm=round(1000 * r.max_slip, 3),
        max_penetration_mm=round(1000 * r.max_penetration, 3),
        unintended_contacts=";".join(f"{e.pair}@{e.phase}" for e in r.unintended_contacts),
        carry_steps=steps.get("carry", 0),
        lower_steps=steps.get("lower", 0),
        steps=r.control_steps,
        sim_s=round(r.sim_time, 3),
        wall_s=round(r.wall_time, 3),
    )


def summarise(rows: list[dict]) -> None:
    def span(key: str, fmt: str = ".1f") -> str:
        values = np.array([r[key] for r in rows], dtype=float)
        return f"{values.min():{fmt}}-{values.max():{fmt}} (median {np.median(values):{fmt}})"

    def count(key: str) -> int:
        return sum(bool(r[key]) for r in rows)

    n = len(rows)
    print(f"Placements: {count('success')}/{n} succeeded; grasps {count('grasp_success')}/{n}")
    failures = [(r["config"], r["failure_reason"]) for r in rows if not r["success"]]
    print(f"Failures: {failures or 'none'}")
    print(
        f"In goal {count('in_goal')}/{n}, upright {count('upright')}/{n}, on table "
        f"{count('on_table')}/{n}, at rest {count('at_rest')}/{n}, released "
        f"{count('released')}/{n}, dropped {count('dropped')}/{n}"
    )
    print(f"Phase timeouts: {[r['config'] for r in rows if r['timeouts']] or 'none'}")
    contacts = [(r["config"], r["unintended_contacts"]) for r in rows if r["unintended_contacts"]]
    print(f"Unintended contacts: {contacts or 'none'}")
    print(f"Centre error (mm): {span('centre_error_mm', '.2f')}")
    print(f"Goal margin (mm, outermost corner inside the circle): {span('goal_margin_mm')}")
    print(f"Tilt (deg): {span('tilt_deg', '.3f')}")
    print(f"Lowest corner above the table (mm): {span('support_gap_mm', '.2f')}")
    print(
        f"Final speed: linear {span('linear_speed_mm_s', '.3f')} mm/s, "
        f"angular {span('angular_speed_rad_s', '.4f')} rad/s"
    )
    print(f"Slip in the grip (mm): {span('max_slip_mm', '.2f')}")
    print(f"Deepest contact, incl. release impact (mm): {span('max_penetration_mm', '.2f')}")
    print(f"Control steps: {span('steps', '.0f')}; simulated s: {span('sim_s', '.2f')}")
    print(f"Wall s per attempt: {span('wall_s', '.2f')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seeded pick-and-place of the target block into the goal region."
    )
    parser.add_argument("--first-seed", type=int, default=1000, help="first scene seed")
    parser.add_argument("--seeds", type=int, default=200, help="number of consecutive seeds")
    parser.add_argument(
        "--corners", action="store_true", help="add the four workspace corners at 45 deg"
    )
    parser.add_argument("--out", type=Path, default=Path("results/pick_place.csv"))
    parser.add_argument("--overwrite", action="store_true", help="replace an existing --out")
    args = parser.parse_args()
    if args.out.exists() and not args.overwrite:
        parser.error(f"{args.out} exists; pass --overwrite to replace it")

    configs = [(f"seed{s}", s, None) for s in range(args.first_seed, args.first_seed + args.seeds)]
    if args.corners:
        configs += [
            (f"corner{100 * x:+.0f}{100 * y:+.0f}", 0, Placement(x, y, np.pi / 4))
            for x, y in CORNERS
        ]
    env = make_tabletop_env(seed=0)
    rows = [trial(env, *config) for config in configs]
    env.close()
    write_rows(args.out, rows, overwrite=args.overwrite)
    summarise(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
