"""Checks on the committed Stage 1 validation datasets; none of these need MuJoCo."""

import itertools
import subprocess
import sys
from pathlib import Path

import pytest

from robot_uncertainty.utils.datasets import diff_rows, read_rows, write_rows

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "validation"

YAW_SWITCH_COLUMNS = [
    "seed", "delta_deg", "eps_deg", "true_rel_deg", "observed_rel_deg", "wrist_deg",
    "face_axis", "face_width_mm", "success", "yaw_err_close_deg", "yaw_err_final_deg",
    "gap_mm", "lift_cm", "steps", "wall_s", "timeouts",
]  # fmt: skip
ALIGNMENT_COLUMNS = [
    "seed", "controller", "cube_width_x_mm", "cube_width_y_mm", "cube_yaw_deg",
    "target_yaw_deg", "wrist_deg", "face_axis", "success", "gap_mm", "lift_cm", "steps",
    "wall_s",
]  # fmt: skip
SCENE_COLUMNS = ["cube_width_x_mm", "cube_width_y_mm", "cube_yaw_deg"]


def test_yaw_switch_dataset_covers_the_full_design() -> None:
    rows = read_rows(DATA / "yaw_switch.csv")
    assert list(rows[0]) == YAW_SWITCH_COLUMNS
    assert len(rows) == 165
    deltas = [40.0, 44.0, 45.0, 46.0, 50.0]
    eps = [-2.0, -1.0, -0.5, -0.1, -0.01, 0.0, 0.01, 0.1, 0.5, 1.0, 2.0]
    conditions = {(int(r["seed"]), float(r["delta_deg"]), float(r["eps_deg"])) for r in rows}
    assert conditions == set(itertools.product(range(3), deltas, eps))
    for r in rows:  # the true yaw is the design value; only the observation is perturbed
        assert abs(float(r["true_rel_deg"]) - float(r["delta_deg"])) < 1e-6


def test_grasp_alignment_dataset_is_paired_by_seed() -> None:
    rows = read_rows(DATA / "grasp_alignment.csv")
    assert list(rows[0]) == ALIGNMENT_COLUMNS
    assert len(rows) == 100
    by_key = {(int(r["seed"]), r["controller"]): r for r in rows}
    assert set(by_key) == set(itertools.product(range(50), ("fixed", "aligned")))
    for seed in range(50):  # both controllers saw the same scene
        fixed, aligned = by_key[seed, "fixed"], by_key[seed, "aligned"]
        assert [fixed[k] for k in SCENE_COLUMNS] == [aligned[k] for k in SCENE_COLUMNS]


def test_write_rows_never_silently_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    write_rows(path, [{"a": 1}])
    with pytest.raises(FileExistsError):
        write_rows(path, [{"a": 2}])
    write_rows(path, [{"a": 2}], overwrite=True)
    assert read_rows(path) == [{"a": "2"}]


def test_diff_rows_ignores_only_timing() -> None:
    base = [{"steps": "80", "wall_s": "0.5"}]
    assert diff_rows(base, [{"steps": "80", "wall_s": "0.7"}]) == []
    assert diff_rows(base, [{"steps": "81", "wall_s": "0.5"}]) == ["row 0, steps: 80 vs 81"]
    assert diff_rows(base, base + base) == ["row count differs: 1 vs 2"]


def test_figures_render_from_committed_data_without_mujoco(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    script = ROOT / "scripts" / "plot_grasp_figures.py"
    code = (
        "import runpy, sys\n"
        f"sys.argv = ['plot', '--figures-dir', {str(tmp_path)!r}]\n"
        f"runpy.run_path({str(script)!r}, run_name='__main__')\n"
        "assert 'mujoco' not in sys.modules and 'robosuite' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "finger_gap_fixed_vs_aligned.png",
        "yaw_switch_45deg_selection_vs_lift.png",
    ]
