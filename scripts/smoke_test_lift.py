from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np
import robosuite

from robot_uncertainty.sim.lift import make_lift_env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Headless smoke test for the stock robosuite Lift task with a Panda."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20, help="control steps to apply")
    args = parser.parse_args()

    print(f"MuJoCo    {mujoco.__version__}")
    print(f"robosuite {robosuite.__version__}")

    env = make_lift_env(args.seed)
    obs = env.reset()

    # robosuite wraps MuJoCo in env.sim: sim.model / sim.data are thin binding objects,
    # and the raw mujoco.MjModel / mujoco.MjData sit behind them as _model / _data.
    model: mujoco.MjModel = env.sim.model._model
    data: mujoco.MjData = env.sim.data._data
    substeps = round(1.0 / (env.control_freq * model.opt.timestep))
    print(
        f"\nLift, seed {args.seed}: action dim {env.action_dim}, "
        f"control {env.control_freq} Hz, timestep {model.opt.timestep} s "
        f"({substeps} physics steps per control step)"
    )

    cube_start = obs["cube_pos"].copy()
    eef_start = obs["robot0_eef_pos"].copy()
    print(f"Initial cube position (m): {np.round(cube_start, 4)}")
    print(f"Initial end-effector (m):  {np.round(eef_start, 4)}")

    # Cube free-joint qpos read straight from MjData: position (3) then quaternion (4).
    cube_qpos = env.sim.data.get_joint_qpos(env.cube.joints[0])
    print(f"Cube free-joint qpos:      {np.round(cube_qpos, 4)}")

    # A gentle upward command with the gripper open. Arm inputs lie in [-1, 1] and are
    # scaled to at most 5 cm / 0.5 rad per control step, so 0.2 asks for 1 cm per step.
    action = np.zeros(env.action_dim)
    action[2] = 0.2
    action[-1] = -1.0  # gripper: -1 open, +1 closed
    low, high = env.action_spec
    assert np.all(action >= low) and np.all(action <= high), "action outside action_spec"

    start_time, start_sim_time = time.perf_counter(), data.time
    for _ in range(args.steps):
        obs, reward, done, _ = env.step(action)
        assert np.all(np.isfinite(obs["robot0_eef_pos"])), "non-finite end-effector state"
    elapsed = time.perf_counter() - start_time

    eef_delta = obs["robot0_eef_pos"] - eef_start
    print(f"\nAfter {args.steps} steps (+z, gripper open):")
    print(f"End-effector displacement (m): {np.round(eef_delta, 4)}")
    print(f"Cube displacement (m):         {np.round(obs['cube_pos'] - cube_start, 4)}")
    print(f"Last reward {reward:.3f}, done {done}")
    print(
        f"Simulated {data.time - start_sim_time:.2f} s in {elapsed:.3f} s wall "
        f"({args.steps / elapsed:.0f} control steps/s)"
    )

    assert eef_delta[2] > 0.0, "end effector did not move up under a +z command"

    # Same seed, fresh environment: the initial scene should be identical.
    repeat_cube = make_lift_env(args.seed).reset()["cube_pos"]
    assert np.array_equal(repeat_cube, cube_start), "same seed gave a different cube position"
    print("\nSeed check: a second environment with the same seed placed the cube identically.")

    env.close()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
