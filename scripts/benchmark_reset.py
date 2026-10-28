from __future__ import annotations

import argparse
import time

import numpy as np

from robot_uncertainty.sim.lift import make_lift_env, observe_cube
from robot_uncertainty.sim.primitives import settle, top_grasp


def grasp(env):
    settle(env)
    cube_pos, cube_yaw = observe_cube(env)
    return top_grasp(env, cube_pos, cube_yaw, env.cube.root_body)


def reset_time(hard_reset: bool, seed: int, resets: int) -> float:
    env = make_lift_env(seed, hard_reset=hard_reset)
    env.reset()
    start = time.perf_counter()
    for _ in range(resets):
        env.reset()
    return (time.perf_counter() - start) / resets


def leftover_state(hard_reset: bool, seed: int) -> dict[str, bool]:
    """Run episode 2 after a dirty episode 1 and after a clean one, and compare."""
    dirty, clean = make_lift_env(seed, hard_reset), make_lift_env(seed, hard_reset)
    dirty.reset()
    grasp(dirty)  # episode 1: the robot moves and the cube is lifted
    clean.reset()  # episode 1: no steps at all
    dirty.reset()
    clean.reset()

    data = dirty.sim.data._data
    controller = dirty.robots[0].part_controllers["right"]
    site_pos = data.site_xpos[dirty.sim.model._model.site(controller.ref_name).id]
    checks = {
        "time, qvel, act, warm-start all zero": (
            data.time == 0.0
            and not data.qvel.any()
            and not data.act.any()
            and not data.qacc_warmstart.any()
        ),
        "controller goal at current pose": np.allclose(controller.goal_pos, site_pos),
        "initial qpos identical": np.array_equal(dirty.sim.data.qpos, clean.sim.data.qpos),
    }
    result_dirty, result_clean = grasp(dirty), grasp(clean)
    checks["episode-2 grasp identical"] = (
        result_dirty.control_steps == result_clean.control_steps
        and np.array_equal(result_dirty.object_final, result_clean.object_final)
        and np.array_equal(dirty.sim.data.qpos, clean.sim.data.qpos)
    )
    return checks


def cube_sizes(hard_reset: bool, seed: int, resets: int = 4) -> list[float]:
    """Cube x half-size (mm) after each reset."""
    env = make_lift_env(seed, hard_reset)
    sizes = []
    for _ in range(resets):
        env.reset()
        geom_id = env.sim.model._model.geom(env.cube.contact_geoms[0]).id
        sizes.append(round(1000 * float(env.sim.model._model.geom_size[geom_id][0]), 2))
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare hard and soft robosuite resets on Lift.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resets", type=int, default=20)
    args = parser.parse_args()

    for hard_reset in (True, False):
        label = f"hard_reset={hard_reset}"
        print(f"\n{label}")
        mean = reset_time(hard_reset, args.seed, args.resets)
        print(f"  reset time: {1000 * mean:.1f} ms (mean of {args.resets})")
        for check, passed in leftover_state(hard_reset, args.seed).items():
            print(f"  {'PASS' if passed else 'FAIL'}  {check}")
        print(f"  cube half-size over 4 resets (mm): {cube_sizes(hard_reset, args.seed)}")


if __name__ == "__main__":
    main()
