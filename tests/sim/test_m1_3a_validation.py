import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("robosuite")

from robot_uncertainty.utils.datasets import TIMING_COLUMNS, diff_rows, read_rows  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "validation" / "m1_3a"
FILES = ("axes", "far", "beside", "carry", "throughput", "sampling_comparison")


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory) -> Path:
    """All five groups and the sampling comparison, rerun into a temporary directory
    with the repository's own command (about 5 minutes)."""
    out = tmp_path_factory.mktemp("m1_3a")
    command = [sys.executable, str(ROOT / "scripts" / "run_distractor_trials.py")]
    done = subprocess.run(
        command + ["--out-dir", str(out)], capture_output=True, text=True, timeout=1800
    )
    assert done.returncode == 0, done.stderr
    return out


@pytest.mark.slow
@pytest.mark.parametrize("name", FILES)
def test_validation_data_regenerates_exactly(regenerated, name) -> None:
    assert TIMING_COLUMNS == ("wall_s",)  # the only columns diff_rows skips
    committed, rows = read_rows(DATA / f"{name}.csv"), read_rows(regenerated / f"{name}.csv")
    assert diff_rows(committed, rows) == []


@pytest.mark.slow
def test_regeneration_records_its_provenance(regenerated) -> None:
    meta = json.loads((regenerated / "provenance.json").read_text())
    committed = json.loads((DATA / "provenance.json").read_text())
    assert meta["physics"] == committed["physics"]
    assert meta["versions"] == committed["versions"]
    assert meta["command"][-2:] == ["--out-dir", str(regenerated)]
