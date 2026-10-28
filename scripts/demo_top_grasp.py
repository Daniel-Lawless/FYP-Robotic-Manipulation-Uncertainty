from __future__ import annotations

import argparse

import numpy as np

from robot_uncertainty.sim.lift import make_lift_env, observe_cube
from robot_uncertainty.sim.primitives import GraspResult, settle, top_grasp


def run_episode(seed: int) -> tuple[GraspResult, np.ndarray]:
    """One grasp from a fresh environment; returns the result and final full qpos."""
    env = make_lift_env(seed)
    env.reset()
    settle(env)
    cube_pos, cube_yaw = observe_cube(env)
    result = top_grasp(env, cube_pos, cube_yaw, env.cube.root_body)
    final_qpos = env.sim.data.qpos.copy()
    env.close()
    return result, final_qpos


def report(seed: int, result: GraspResult) -> None:
    verdict = "VERIFIED GRASP" if result.success else "GRASP NOT VERIFIED"
    print(f"\nSeed {seed}: {verdict}")
    for p in result.phases:
        status = "reached" if p.reached else "TIMED OUT"
        print(
            f"  {p.name:<9} {status:<9} {p.steps:>3} steps, "
            f"pos err {1000 * p.pos_error:5.1f} mm, ori err {np.degrees(p.ori_error):4.1f} deg"
        )
    print(f"  cube start z {result.object_start[2]:.4f} m, final z {result.object_final[2]:.4f} m")
    print(f"  lift height {result.lift_height:.4f} m, finger gap {1000 * result.finger_gap:.1f} mm")
    print(
        f"  wrist turned {np.degrees(result.wrist_rotation):+.1f} deg, "
        f"finger-to-face yaw error at close {np.degrees(result.yaw_error):+.2f} deg"
    )
    print(f"  final end effector {np.round(result.eef_final, 4)} m")
    print(
        f"  {result.control_steps} control steps, {result.sim_time:.2f} s simulated, "
        f"{result.wall_time:.2f} s wall"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scripted top grasp on the Lift cube, with a reproducibility check."
    )
    parser.add_argument("--seed", type=int, default=0, help="first seed")
    parser.add_argument("--num-seeds", type=int, default=1, help="consecutive seeds to run")
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    args = parser.parse_args()

    results, final_qpos = {}, {}
    for seed in range(args.seed, args.seed + args.num_seeds):
        results[seed], final_qpos[seed] = run_episode(seed)
        if not args.quiet:
            report(seed, results[seed])

    runs = list(results.values())
    successes = sum(r.success for r in runs)
    timeouts = sum(not p.reached for r in runs for p in r.phases)
    gaps = 1000 * np.array([r.finger_gap for r in runs])
    steps = [r.control_steps for r in runs]
    yaw_errors = np.degrees(np.abs([r.yaw_error for r in runs]))
    walls = [r.wall_time for r in runs]
    print(f"\nVerified grasps: {successes}/{len(runs)}; phase timeouts: {timeouts}")
    print(
        f"Finger gap (mm): min {gaps.min():.1f}, median {np.median(gaps):.1f}, "
        f"max {gaps.max():.1f}; |yaw error| at close: max {yaw_errors.max():.2f} deg"
    )
    print(
        f"Control steps: {min(steps)}-{max(steps)}; wall time per grasp: "
        f"mean {np.mean(walls):.2f} s, max {np.max(walls):.2f} s"
    )

    # Reproducibility: rerun the first seed from scratch and compare exactly.
    first, reference_qpos = results[args.seed], final_qpos[args.seed]
    repeat, repeat_qpos = run_episode(args.seed)
    identical = (
        repeat.success == first.success
        and repeat.control_steps == first.control_steps
        and [p.steps for p in repeat.phases] == [p.steps for p in first.phases]
        and np.array_equal(repeat.object_final, first.object_final)
        and np.array_equal(repeat.eef_final, first.eef_final)
        and np.array_equal(repeat_qpos, reference_qpos)
    )
    max_diff = np.max(np.abs(repeat_qpos - reference_qpos))
    print(
        f"Reproducibility (seed {args.seed}, fresh environment): "
        f"{'IDENTICAL' if identical else 'DIFFERENT'}, max |qpos difference| {max_diff:.1e}"
    )


if __name__ == "__main__":
    main()
