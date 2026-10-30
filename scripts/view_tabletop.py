from __future__ import annotations

import argparse
import os
import signal
import threading
import time

import numpy as np

from robot_uncertainty.sim.pick_place import PickPlaceResult, pick_and_place
from robot_uncertainty.sim.primitives import GraspResult, settle
from robot_uncertainty.sim.tabletop import TabletopEnv, grasp_block_from_above, make_tabletop_env

TARGET = "target"
# Free camera looking at the table from in front of the robot (robosuite's default sits
# 2 m from a point 20 cm above the table).
CAMERA = {"lookat": [-0.12, 0.0, 0.88], "distance": 1.5, "azimuth": 150, "elevation": -25}
START_PAUSE = 1.0  # wall seconds showing the scene before the robot moves


class ViewerClosed(Exception):
    """The viewer window was closed."""


def make_viewer_env() -> TabletopEnv:
    return make_tabletop_env(
        seed=0,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        renderer="mjviewer",
        renderer_config={"cam_config": CAMERA},
        render_camera=None,  # free camera, moved with the mouse
    )


def viewer_is_open(env: TabletopEnv) -> bool:
    handle = env.viewer.viewer if env.viewer is not None else None
    return handle is not None and handle.is_running()


def pace(env: TabletopEnv, speed: float) -> None:
    """Make env.step wait so simulated time advances at speed x wall time.

    robosuite's step already syncs the viewer after the physics; this only adds the
    wait, and raises ViewerClosed once the window has been closed.
    """
    step = env.step
    start_sim, start_wall = env.sim.data.time, time.perf_counter()

    def paced_step(action):
        out = step(action)
        if not viewer_is_open(env):
            raise ViewerClosed
        ahead = (env.sim.data.time - start_sim) / speed - (time.perf_counter() - start_wall)
        if ahead > 0:
            time.sleep(ahead)
        return out

    env.step = paced_step


def open_viewer(env: TabletopEnv) -> None:
    """Open the viewer window on the current state.

    robosuite closes the mjviewer window on every reset. A Ctrl+C while MuJoCo creates
    the window would leave it without a handle to close it, so it is held back until
    the window is open.
    """
    interrupted = []
    previous = signal.signal(signal.SIGINT, lambda *_: interrupted.append(True))
    try:
        env.initialize_renderer()
        env.viewer.update()
    finally:
        signal.signal(signal.SIGINT, previous)
    if interrupted:
        raise KeyboardInterrupt


def show_scene(env: TabletopEnv, seconds: float) -> None:
    """Open the viewer on the current state and keep it there for a while."""
    open_viewer(env)
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        if not viewer_is_open(env):
            raise ViewerClosed
        time.sleep(0.02)


def report(result: GraspResult) -> None:
    verdict = "VERIFIED GRASP" if result.success else "GRASP NOT VERIFIED"
    print(f"\nResult: {verdict}")
    for p in result.phases:
        status = "reached" if p.reached else "TIMED OUT"
        print(
            f"  {p.name:<9} {status:<9} {p.steps:>3} steps, "
            f"pos err {1000 * p.pos_error:5.1f} mm, ori err {np.degrees(p.ori_error):4.1f} deg"
        )
    print(
        f"  grasp point {1000 * result.grasp_height:.0f} mm above the block centre; "
        f"wrist turned {np.degrees(result.wrist_rotation):+.1f} deg"
    )
    print(f"  lift {100 * result.lift_height:.2f} cm, finger gap {1000 * result.finger_gap:.1f} mm")
    print(f"  {result.control_steps} control steps, {result.sim_time:.2f} s simulated")


def report_pick_place(result: PickPlaceResult) -> None:
    verdict = "PLACED IN THE GOAL" if result.success else f"FAILED: {result.failure_reason}"
    print(f"\nResult: {verdict}")
    for p in result.phases:
        status = "reached" if p.reached else "TIMED OUT"
        print(f"  {p.name:<9} {status:<9} {p.steps:>3} steps, pos err {1000 * p.pos_error:5.1f} mm")
    pl = result.placement
    print(
        f"  grasp {'verified' if result.grasp_success else 'NOT verified'}, "
        f"released {result.released}, dropped {result.dropped}"
    )
    print(
        f"  block centre {1000 * pl.centre_error:.1f} mm from the goal centre; every corner "
        f"inside the circle by at least {1000 * pl.goal_margin:.1f} mm"
        if pl.in_goal
        else f"  block outside the goal circle by {-1000 * pl.goal_margin:.1f} mm"
    )
    print(
        f"  tilt {np.degrees(pl.tilt):.2f} deg, upright {pl.upright}, on the table "
        f"{pl.on_table}, at rest {pl.at_rest}"
    )
    contacts = [f"{e.pair} during {e.phase}" for e in result.unintended_contacts]
    print(f"  unintended contacts: {', '.join(contacts) or 'none'}")
    print(f"  {result.control_steps} control steps, {result.sim_time:.2f} s simulated")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Watch the top grasp or pick-and-place in the tabletop scene in MuJoCo's "
        "viewer."
    )
    parser.add_argument("--seed", type=int, default=0, help="scene seed (default 0)")
    parser.add_argument(
        "--task",
        choices=("grasp", "pick-place"),
        default="grasp",
        help="grasp and lift only (default), or the full pick-and-place into the goal",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="playback speed; 1 is real time, 0.5 half"
    )
    args = parser.parse_args()
    if not args.speed > 0:
        parser.error("--speed must be positive")
    if os.name == "posix" and not os.environ.get("DISPLAY"):
        parser.error("DISPLAY is not set, so there is no window to open (on WSL2 this needs WSLg)")

    env = None
    threads = set(threading.enumerate())
    try:
        env = make_viewer_env()
        env.reset(scene_seed=args.seed)
        placed = env.layout[TARGET]
        goal = env.config.goal
        print(
            f"Scene seed {args.seed}: block at ({100 * placed.x:+.1f}, {100 * placed.y:+.1f}) cm "
            f"from the table centre, yaw {np.degrees(placed.yaw):+.1f} deg; goal circle at "
            f"({100 * goal.center[0]:+.1f}, {100 * goal.center[1]:+.1f}) cm, "
            f"radius {100 * goal.radius:.0f} cm"
        )
        show_scene(env, START_PAUSE)
        pace(env, args.speed)
        print(f"Settling and running the {args.task} at {args.speed:g}x real time ...")
        settle(env)
        if args.task == "grasp":
            report(grasp_block_from_above(env, TARGET))
        else:
            report_pick_place(pick_and_place(env, TARGET))
        print("\nFinal state shown; close the window or press Ctrl+C to exit.")
        while viewer_is_open(env):
            time.sleep(0.05)
    except ViewerClosed:
        print("\nViewer window closed.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if env is not None:
            env.close()
        # MuJoCo's viewer renders in its own thread, which close() only asks to stop. Let
        # it finish, or glfw's exit hook can terminate GLFW under it (a segfault).
        for thread in set(threading.enumerate()) - threads:
            thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
