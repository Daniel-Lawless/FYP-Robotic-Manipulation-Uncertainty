from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from robot_uncertainty.sim.lift import make_lift_env
from robot_uncertainty.sim.primitives import GraspResult, OSCArm, heading, settle, top_grasp
from robot_uncertainty.utils.datasets import write_rows

DELTAS_DEG = [40.0, 44.0, 45.0, 46.0, 50.0]
EPS_DEG = [-2.0, -1.0, -0.5, -0.1, -0.01, 0.0, 0.01, 0.1, 0.5, 1.0, 2.0]


def set_cube_yaw(env, yaw: float) -> None:
    """Rotate the resting cube about world z in place and zero its velocity."""
    joint = env.cube.joints[0]
    qpos = env.sim.data.get_joint_qpos(joint).copy()
    qpos[3:] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]  # wxyz
    env.sim.data.set_joint_qpos(joint, qpos)
    env.sim.data.set_joint_qvel(joint, np.zeros(6))
    env.sim.forward()


def fingerprint(env) -> str:
    data = env.sim.data._data
    raw = b"".join(a.tobytes() for a in (data.qpos, data.qvel, data.act, data.qacc_warmstart))
    return hashlib.sha1(raw).hexdigest()[:12]


def trial(seed: int, delta: float, eps: float) -> tuple[GraspResult, str, float]:
    """Cube at gripper yaw + delta; grasp planned from cube yaw + eps (radians)."""
    env = make_lift_env(seed, hard_reset=False)
    env.reset()
    settle(env)
    cube_yaw = heading(OSCArm(env).eef_rot()) + delta
    set_cube_yaw(env, cube_yaw)
    state = fingerprint(env)
    cube_pos = env.sim.data.body_xpos[env.cube_body_id].copy()
    result = top_grasp(env, cube_pos, cube_yaw + eps, env.cube.root_body)
    face_width = 2 * env.cube.size[result.face_axis]
    env.close()
    return result, state, face_width


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Does the 45-degree grasp-yaw ambiguity change grasp performance?"
    )
    parser.add_argument("--seeds", type=int, default=3, help="seeds 0..N-1")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/yaw_switch.csv"),
        help="per-trial CSV (committed copy: data/validation/yaw_switch.csv)",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing --out")
    args = parser.parse_args()
    if args.out.exists() and not args.overwrite:
        parser.error(f"{args.out} exists; pass --overwrite or choose another --out")

    rows, states = [], defaultdict(set)
    for seed in range(args.seeds):
        for delta_deg in DELTAS_DEG:
            for eps_deg in EPS_DEG:
                r, state, width = trial(seed, np.radians(delta_deg), np.radians(eps_deg))
                states[seed, delta_deg].add(state)
                rows.append(
                    dict(
                        seed=seed,
                        delta_deg=delta_deg,
                        eps_deg=eps_deg,
                        true_rel_deg=np.degrees(r.true_yaw - r.start_yaw),
                        observed_rel_deg=np.degrees(r.observed_yaw - r.start_yaw),
                        wrist_deg=np.degrees(r.wrist_rotation),
                        face_axis=r.face_axis,
                        face_width_mm=1000 * width,
                        success=r.success,
                        yaw_err_close_deg=np.degrees(r.yaw_error),
                        yaw_err_final_deg=np.degrees(r.final_yaw_error),
                        gap_mm=1000 * r.finger_gap,
                        lift_cm=100 * r.lift_height,
                        steps=r.control_steps,
                        wall_s=r.wall_time,
                        timeouts=sum(not p.reached for p in r.phases),
                    )
                )

    # Controls: identical start state across eps, and an exact repeat of one trial.
    controlled = all(len(s) == 1 for s in states.values())
    a, _, _ = trial(0, np.radians(45.0), 0.0)
    b, _, _ = trial(0, np.radians(45.0), 0.0)
    repeat_ok = a.control_steps == b.control_steps and np.array_equal(
        a.object_final, b.object_final
    )
    print(f"Start state identical across eps for every (seed, delta): {controlled}")
    print(f"Repeat of (seed 0, delta 45, eps 0) bit-identical: {repeat_ok}\n")

    header = (
        f"{'delta':>5} {'eps':>6} {'wrist':>7} {'face':>4} {'ok':>5} {'|err| close':>11} "
        f"{'|err| final':>11} {'gap mm':>7} {'lift cm':>7} {'steps':>5} {'wall s':>6}"
    )
    print(f"Means over seeds 0-{args.seeds - 1}; angles in degrees, wrist = turn commanded")
    print(header)
    groups = defaultdict(list)
    for row in rows:
        groups[row["delta_deg"], row["eps_deg"]].append(row)
    for (delta_deg, eps_deg), g in groups.items():

        def mean(key, group=g):
            return np.mean([row[key] for row in group])

        faces = "/".join(str(f) for f in sorted({row["face_axis"] for row in g}))
        print(
            f"{delta_deg:5.0f} {eps_deg:+6.2f} {mean('wrist_deg'):+7.2f} {faces:>4} "
            f"{sum(row['success'] for row in g)}/{len(g):<3} "
            f"{np.mean([abs(row['yaw_err_close_deg']) for row in g]):11.3f} "
            f"{np.mean([abs(row['yaw_err_final_deg']) for row in g]):11.3f} "
            f"{mean('gap_mm'):7.1f} {mean('lift_cm'):7.2f} {mean('steps'):5.1f} "
            f"{mean('wall_s'):6.3f}"
        )
    failures = [row for row in rows if not row["success"] or row["timeouts"]]
    print(f"\nTrials: {len(rows)}; failed or timed out: {len(failures)}")

    write_rows(args.out, rows, overwrite=args.overwrite)
    print(f"Per-trial rows written to {args.out}")


if __name__ == "__main__":
    main()
