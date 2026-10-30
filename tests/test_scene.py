"""Scene configuration and seeded layout sampling (pure NumPy; runs in CI)."""

import numpy as np
import pytest

from robot_uncertainty.sim.scene import (
    BlockSpec,
    GoalRegion,
    Placement,
    Region,
    TabletopConfig,
    footprint_corners,
    sample_layout,
    stream_rng,
)

CONFIG = TabletopConfig()
SEEDS = range(500)


def test_default_target_is_the_proposal_block() -> None:
    (block,) = CONFIG.blocks
    assert block.size == (0.04, 0.04, 0.08) and block.role == "target"
    assert np.isclose(block.symmetry, np.pi / 2)


def test_same_seed_gives_same_layout_and_seeds_differ() -> None:
    assert sample_layout(CONFIG, 7) == sample_layout(CONFIG, 7)
    layouts = {sample_layout(CONFIG, s)["target"] for s in range(20)}
    assert len(layouts) == 20


def test_streams_are_independent() -> None:
    assert stream_rng(3, 0).random() != stream_rng(3, 1).random()
    assert stream_rng(3, 0).random() == stream_rng(3, 0).random()


def test_every_footprint_lies_inside_the_workspace_and_off_the_goal() -> None:
    block = CONFIG.blocks[0]
    for seed in SEEDS:
        placement = sample_layout(CONFIG, seed)["target"]
        corners = footprint_corners(block, placement)
        assert all(CONFIG.workspace.contains(c) for c in corners), seed
        assert not any(CONFIG.goal.contains(c) for c in corners), seed
        assert CONFIG.yaw_range[0] <= placement.yaw <= CONFIG.yaw_range[1]


def test_layouts_spread_over_the_workspace() -> None:
    # Guards against a sampler that ignores its ranges or the seed.
    xy = np.array([sample_layout(CONFIG, s)["target"].xy for s in SEEDS])
    reach = CONFIG.blocks[0].footprint_radius
    for axis, (lo, hi) in enumerate([CONFIG.workspace.x_range, CONFIG.workspace.y_range]):
        assert xy[:, axis].min() < lo + reach + 0.01
        assert xy[:, axis].max() > hi - reach - 0.01


def test_footprint_corners_rotate_with_yaw() -> None:
    block = BlockSpec("b", size=(0.04, 0.02, 0.08))
    flat = footprint_corners(block, Placement(0.0, 0.0, 0.0))
    np.testing.assert_allclose(np.abs(flat), [[0.02, 0.01]] * 4)
    turned = footprint_corners(block, Placement(0.1, 0.0, np.pi / 2))
    np.testing.assert_allclose(np.abs(turned - [0.1, 0.0]), [[0.01, 0.02]] * 4, atol=1e-12)


def test_goal_and_regions() -> None:
    goal = GoalRegion(center=(0.0, 0.18), radius=0.05)
    assert goal.contains([0.0, 0.2]) and not goal.contains([0.0, 0.1])
    region = Region((-0.1, 0.1), (-0.2, 0.06))
    assert region.contains([0.1, 0.06]) and not region.contains([0.1, 0.06], margin=0.01)
    assert np.isclose(region.distance_to([0.0, 0.18]), 0.12)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(workspace=Region((-0.35, 0.1), (-0.2, 0.06))),  # off the table
        dict(goal=GoalRegion(center=(0.0, 0.08), radius=0.05)),  # inside the workspace
        dict(goal=GoalRegion(center=(0.0, 0.28), radius=0.05)),  # off the table
        dict(blocks=(BlockSpec("a"), BlockSpec("a"))),  # duplicate names
    ],
)
def test_invalid_configs_are_rejected(overrides) -> None:
    with pytest.raises(ValueError):
        TabletopConfig(**overrides)


def test_impossible_placement_raises_instead_of_looping() -> None:
    config = TabletopConfig(blocks=(BlockSpec("huge", size=(0.3, 0.3, 0.08)),))
    with pytest.raises(RuntimeError, match="no valid placement"):
        sample_layout(config, 0)


def test_sampler_keeps_several_blocks_apart() -> None:
    # Only the sampler is exercised with more than one block; the environment still
    # has a single target in M1.1.
    config = TabletopConfig(blocks=(BlockSpec("a"), BlockSpec("b")))
    gap = 2 * config.blocks[0].footprint_radius + config.clearance
    for seed in range(200):
        layout = sample_layout(config, seed)
        assert np.hypot(*(layout["a"].xy - layout["b"].xy)) >= gap
