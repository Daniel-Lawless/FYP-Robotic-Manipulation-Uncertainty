import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("robosuite")
import mujoco  # noqa: E402
import robosuite  # noqa: E402

from robot_uncertainty.sim.pick_place import pick_and_place  # noqa: E402
from robot_uncertainty.sim.primitives import settle  # noqa: E402
from robot_uncertainty.sim.tabletop import make_tabletop_env  # noqa: E402
from robot_uncertainty.utils.datasets import diff_rows, read_rows  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CSV = ROOT / "data" / "validation" / "pick_place_m1_2.csv"
REFERENCE_QPOS = Path(__file__).parent / "data" / "m1_2_reference_qpos.npz"
REFERENCE_COMMIT = "c3f88e2"


def test_references_were_captured_from_the_clean_reference_commit() -> None:
    ref = np.load(REFERENCE_QPOS)
    assert str(ref["commit"]).startswith(REFERENCE_COMMIT)
    assert str(ref["dirty"]) == ""
    assert list(ref["seeds"]) == list(range(1000, 1010))
    rows = read_rows(REFERENCE_CSV)
    configs = [r["config"] for r in rows]
    assert configs[:200] == [f"seed{s}" for s in range(1000, 1200)]
    assert configs[200:] == ["corner-7-17", "corner-7+3", "corner+7-17", "corner+7+3"]


def test_final_qpos_is_bitwise_identical_to_m1_2() -> None:
    ref = np.load(REFERENCE_QPOS)
    versions = (
        str(ref["mujoco_version"]),
        str(ref["robosuite_version"]),
        str(ref["numpy_version"]),
    )
    if versions != (mujoco.__version__, robosuite.__version__, np.__version__):
        pytest.skip(f"references are for MuJoCo/robosuite/NumPy {versions}")
    env = make_tabletop_env(seed=0)
    try:
        for seed, expected in zip(ref["seeds"], ref["qpos"], strict=True):
            env.reset(scene_seed=int(seed))
            settle(env)
            result = pick_and_place(env, "target")
            assert np.array_equal(env.sim.data.qpos, expected), f"seed {seed}"
            assert result.label.y == 1 and result.success
    finally:
        env.close()


@pytest.mark.slow
def test_run_pick_place_reproduces_the_204_reference_rows(tmp_path) -> None:
    out = tmp_path / "pick_place.csv"
    command = [sys.executable, str(ROOT / "scripts" / "run_pick_place.py")]
    args = ["--first-seed", "1000", "--seeds", "200", "--corners", "--out", str(out)]
    done = subprocess.run(command + args, capture_output=True, text=True, timeout=1200)
    assert done.returncode == 0, done.stderr
    reference, rows = read_rows(REFERENCE_CSV), read_rows(out)
    # Only wall_s is skipped; the new label columns come after the original ones.
    assert diff_rows(reference, rows, allow_extra_columns=True) == []
    assert list(rows[0])[: len(reference[0])] == list(reference[0])
    # The new outcome reproduces the M1.2 outcome on every reference row.
    for row in rows:
        success = row["success"] == "True"
        assert (row["y"] == "1") == success
        assert (row["failure_stage"] == "") == success
        assert row["flags"] == ""
