import itertools
import json
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
PALM_COLUMNS = [
    "config", "scene_seed", "x_cm", "y_cm", "yaw_deg", "strategy", "grasp_height_mm",
    "success", "step", "phase", "clearance_mm", "palm_contact", "palm_force_n",
    "pad1_force_n", "pad2_force_n",
]  # fmt: skip
PHASES = ["open", "pregrasp", "descend", "close", "lift", "hold"]


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


def test_palm_clearance_dataset_covers_the_design() -> None:
    rows = read_rows(DATA / "palm_clearance.csv")
    assert list(rows[0]) == PALM_COLUMNS
    grasps = {}
    for r in rows:
        grasps.setdefault((r["config"], r["strategy"]), []).append(r)
    configs = [f"seed{s:02d}" for s in range(12)] + [
        f"corner{x}{y}" for x in ("-7", "+7") for y in ("-17", "+3")
    ]
    assert set(grasps) == set(itertools.product(configs, ("centre", "raised")))
    for (config, strategy), group in grasps.items():
        assert [int(r["step"]) for r in group] == list(range(1, len(group) + 1))
        phases = list(dict.fromkeys(r["phase"] for r in group))
        assert phases == PHASES, (config, strategy)  # each phase one contiguous block
        assert {r["grasp_height_mm"] for r in group} == {"0.0" if strategy == "centre" else "15.0"}
        assert {r["success"] for r in group} == {"True"}
        # Both grasps of a configuration start from the same block pose.
        other = grasps[config, "raised" if strategy == "centre" else "centre"]
        assert [group[0][k] for k in ("x_cm", "y_cm", "yaw_deg")] == [
            other[0][k] for k in ("x_cm", "y_cm", "yaw_deg")
        ]


M1_3A = DATA / "m1_3a"
M1_3A_COLUMNS = [
    "config", "scene_seed", "x_cm", "y_cm", "yaw_deg", "choice", "closing_deg", "success",
    "failure_reason", "y", "failure_stage", "flags", "wrist_deg", "commanded_face_axis",
    "face_axis", "timeouts", "unintended_contacts", "centre_error_mm", "steps", "sim_s",
    "wall_s",
]  # fmt: skip
M1_3A_DISTRACTOR_COLUMNS = [
    "d_x_cm", "d_y_cm", "d_yaw_deg", "d_max_mm", "d_final_mm", "d_tilt_max_deg",
    "d_tilt_final_deg", "d_face_start", "d_face_final", "d_toppled", "d_leaning",
    "d_unsettled", "d_com_outside_base", "d_contacts",
]  # fmt: skip
# Appended by M1.3a.1 after every M1.3a column.
SUBSTEP_COLUMNS = [
    "historical_pairs", "historical_unintended", "endpoint_y", "endpoint_failure_stage",
    "endpoint_flags",
]  # fmt: skip
SUBSTEP_DISTRACTOR_COLUMNS = ["d_max_endpoint_mm", "d_tilt_max_endpoint_deg"]
SAMPLING_COLUMNS = [
    "config", "scene_seed", "choice", "known_miss", "endpoint_pairs", "historical_pairs",
    "missed_pairs", "endpoint_collision_object", "historical_collision_object",
    "endpoint_target_distractor", "historical_target_distractor", "d_max_endpoint_mm",
    "d_max_mm", "d_tilt_max_endpoint_deg", "d_tilt_max_deg", "endpoint_y", "y",
    "endpoint_failure_stage", "failure_stage",
]  # fmt: skip
KNOWN_MISSES = {3, 8, 30, 49}
CORNERS = [f"corner{x}{y}" for x in ("-7", "+7") for y in ("-17", "+3")]


def m1_3a(name: str) -> list[dict]:
    return read_rows(M1_3A / f"{name}.csv")


def test_m1_3a_datasets_keep_their_columns_and_append_the_substep_ones() -> None:
    with_distractor = M1_3A_COLUMNS + M1_3A_DISTRACTOR_COLUMNS
    added = SUBSTEP_COLUMNS + SUBSTEP_DISTRACTOR_COLUMNS
    expected = {
        "axes": M1_3A_COLUMNS + SUBSTEP_COLUMNS,
        "far": with_distractor + ["hand_clearance_mm"] + added,
        "beside": with_distractor + added,
        "carry": with_distractor + added,
        "throughput": with_distractor + added,
        "sampling_comparison": SAMPLING_COLUMNS,
    }
    for name, columns in expected.items():
        assert list(m1_3a(name)[0]) == columns, name


def test_m1_3a_datasets_cover_their_designs() -> None:
    axes = m1_3a("axes")
    configs = [f"seed{s}" for s in range(50)] + CORNERS
    assert [(r["config"], r["choice"]) for r in axes] == [
        (c, f"candidate{i}") for c in configs for i in (0, 1)
    ]
    far = m1_3a("far")
    assert len(far) == 20 and {r["choice"] for r in far} == {"legacy"}
    assert all(float(r["hand_clearance_mm"]) >= 150 for r in far)
    beside = m1_3a("beside")
    assert [(r["config"], r["choice"]) for r in beside] == [
        (f"beside{s}", f"{size}-{side}")
        for size in ("tall", "short")
        for s in range(10)
        for side in ("A", "B")
    ]
    carry = m1_3a("carry")
    assert len(carry) == 5 and {r["choice"] for r in carry} == {"legacy"}
    throughput = m1_3a("throughput")
    assert [(r["config"], r["scene_seed"]) for r in throughput] == [
        (f"sampled{s}", str(s)) for s in range(50)
    ]


def pair_phases(column: str) -> set[str]:
    """The pair@phase of each event in an unintended_contacts-style column."""
    return {event.split("x")[0] for event in column.split(";") if event}


def test_substep_columns_cover_the_endpoint_evidence() -> None:
    for name in ("axes", "far", "beside", "carry", "throughput"):
        for r in m1_3a(name):
            # Every endpoint (legacy) unintended event is also a historical one.
            assert pair_phases(r["unintended_contacts"]) <= pair_phases(
                r["historical_unintended"]
            ), (name, r["config"])
            history = set(r["historical_pairs"].split(";"))
            assert {p.split("@")[0] for p in pair_phases(r["historical_unintended"])} <= history
            if "d_max_endpoint_mm" in r:
                assert float(r["d_max_mm"]) >= float(r["d_max_endpoint_mm"])
                assert float(r["d_tilt_max_deg"]) >= float(r["d_tilt_max_endpoint_deg"])


def test_sampling_comparison_matches_the_throughput_attempts() -> None:
    rows, throughput = m1_3a("sampling_comparison"), m1_3a("throughput")
    assert [r["config"] for r in rows] == [r["config"] for r in throughput]
    for r, t in zip(rows, throughput, strict=True):
        assert (r["y"], r["failure_stage"]) == (t["y"], t["failure_stage"])
        assert (r["endpoint_y"], r["endpoint_failure_stage"]) == (
            t["endpoint_y"],
            t["endpoint_failure_stage"],
        )
        assert int(r["historical_pairs"]) >= int(r["endpoint_pairs"])
        for kind in ("collision_object", "target_distractor"):
            if r[f"endpoint_{kind}"] == "True":
                assert r[f"historical_{kind}"] == "True"
        assert float(r["d_max_mm"]) >= float(r["d_max_endpoint_mm"])
        miss = int(r["scene_seed"]) in KNOWN_MISSES
        assert r["known_miss"] == str(miss)
        if miss:  # the impacts M1.3a's control-step contacts missed
            assert r["endpoint_target_distractor"] == "False"
            assert r["historical_target_distractor"] == "True"
            assert r["historical_collision_object"] == "True"
            assert "distractor-target" in r["missed_pairs"].split(";")


def test_m1_3a_provenance_is_recorded() -> None:
    meta = json.loads((M1_3A / "provenance.json").read_text())
    assert meta["base_commit"] == "9f7ce6b291a23e9fc03901219166c3fce5549a20"
    assert meta["physics"]["substeps_per_control_step"] == 25
    assert meta["physics"]["integrator"] == "mjINT_EULER" and meta["physics"]["lite_physics"]
    assert "src/robot_uncertainty/sim/pick_place.py" in meta["source_sha256"]
    assert set(meta["versions"]) == {"python", "numpy", "mujoco", "robosuite"}
    assert meta["command"][:2] == ["python", "scripts/run_distractor_trials.py"]


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


def test_palm_clearance_figure_renders_from_committed_data_without_mujoco(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    script = ROOT / "scripts" / "plot_palm_clearance.py"
    code = (
        "import runpy, sys\n"
        f"sys.argv = ['plot', '--figures-dir', {str(tmp_path)!r}]\n"
        f"runpy.run_path({str(script)!r}, run_name='__main__')\n"
        "assert 'mujoco' not in sys.modules and 'robosuite' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["palm_clearance_centre_vs_raised_grasp.png"]
