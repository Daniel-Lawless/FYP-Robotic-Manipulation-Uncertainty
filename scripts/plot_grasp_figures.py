"""Figures for the Stage 1 top-grasp findings, from the committed experiment CSVs.

Figure 1 (yaw switch): the 45-degree condition of data/validation/yaw_switch.csv, one
point per seed and observation perturbation.
Figure 2 (finger gap): the paired 50-seed data/validation/grasp_alignment.csv, with the
finger span a rigid, unrotated cube would produce, from its recorded size.

Plotting needs only the CSVs and Matplotlib, not MuJoCo. --regenerate first reruns both
experiments (about 5 minutes) into results/regenerated/, which git ignores, and reports
any difference from the committed data other than wall-clock time; it never replaces
the committed data. See data/validation/README.md.

Usage:
    python scripts/plot_grasp_figures.py [--data-dir DIR] [--figures-dir DIR] [--pdf]
    python scripts/plot_grasp_figures.py --regenerate
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from robot_uncertainty.utils.datasets import diff_rows, read_rows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "validation"
REGENERATED_DIR = ROOT / "results" / "regenerated"
FIGURES_DIR = ROOT / "figures"
# Dataset file -> the experiment command that produces it (output path appended).
DATASETS = {
    "yaw_switch.csv": ["scripts/diagnose_yaw_switch.py", "--seeds", "3"],
    "grasp_alignment.csv": ["scripts/compare_grasp_alignment.py", "--seeds", "50"],
}

# Validated categorical slots (light surface), text inks and a recessive grid gray.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK_2, GRID, REFERENCE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#8a8984"
SEED_MARKERS = [("o", 9), ("s", 6.5), ("^", 4.5)]  # shrinking, so coincident points all show

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK_2,
        "axes.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def load(data_dir: Path, name: str) -> list[dict]:
    path = data_dir / name
    if not path.exists():
        command = " ".join(["python", *DATASETS[name], "--out", str(path)])
        sys.exit(f"Dataset {path} not found. Regenerate it (needs MuJoCo) with:\n  {command}")
    rows = read_rows(path)
    print(f"Read {len(rows)} rows from {path}")
    return rows


def regenerate(data_dir: Path) -> None:
    """Rerun both experiments into REGENERATED_DIR and compare with data_dir."""
    for name, command in DATASETS.items():
        out = REGENERATED_DIR / name
        print(f"Regenerating {out.relative_to(ROOT)} ...", flush=True)
        subprocess.run(
            [sys.executable, *command, "--out", str(out), "--overwrite"], cwd=ROOT, check=True
        )
        if not (data_dir / name).exists():
            print(f"{name}: no committed copy in {data_dir} to compare against")
            continue
        diffs = diff_rows(read_rows(data_dir / name), read_rows(out))
        verdict = "identical apart from wall_s" if not diffs else f"{len(diffs)} differences"
        print(f"{name}: regenerated vs {data_dir / name}: {verdict}")
        for line in diffs[:10]:
            print(f"  {line}")
    print(f"Committed data unchanged; regenerated copies are in {REGENERATED_DIR}.")


def save(fig, name: str, figures_dir: Path, pdf: bool) -> None:
    for suffix in ("png", "pdf") if pdf else ("png",):
        out = figures_dir / f"{name}.{suffix}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


def plot_yaw_switch(rows: list[dict]) -> plt.Figure:
    rows = [r for r in rows if float(r["delta_deg"]) == 45.0]
    true_rel = [float(r["true_rel_deg"]) for r in rows]
    assert np.allclose(true_rel, 45.0), "45-degree condition must have true yaw 45"
    by_seed = defaultdict(list)
    for r in rows:
        by_seed[int(r["seed"])].append(r)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    for i, (seed, group) in enumerate(sorted(by_seed.items())):
        group.sort(key=lambda r: float(r["eps_deg"]))
        eps = np.array([float(r["eps_deg"]) for r in group])
        wrist = np.array([float(r["wrist_deg"]) for r in group])
        lift = np.array([float(r["lift_cm"]) for r in group])
        marker, size = SEED_MARKERS[i]
        style = dict(
            color=SERIES[i],
            marker=marker,
            markersize=size,
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
            linewidth=1.5,
            label=f"seed {seed}",
        )
        # Draw each side of the switch as its own line, so no line bridges the jump.
        for side in (wrist > 0, wrist < 0):
            ax_a.plot(eps[side], wrist[side], **style)
            ax_b.plot(eps[side], lift[side], **style)
            style.pop("label", None)

    for ax in (ax_a, ax_b):
        ax.set_xscale("symlog", linthresh=0.01, linscale=0.6)
        ax.axvline(0.0, color=REFERENCE, linewidth=1.0, linestyle=":", zorder=0)
        ticks = [-2, -0.5, -0.1, -0.01, 0, 0.01, 0.1, 0.5, 2]
        ax.set_xticks(ticks, [f"{t:+g}\n{45 + t:g}" if t else "0\n45" for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel(
            "Observation error ε = observed − true yaw (deg, symlog)\n"
            "Observed cube yaw relative to gripper start yaw (deg)"
        )

    ax_a.set_title("A  Selected wrist rotation", loc="left", fontweight="bold")
    ax_a.set_ylabel("Commanded wrist yaw rotation (deg)")
    ax_a.set_yticks([-45, -30, -15, 0, 15, 30, 45])
    ax_a.set_ylim(-52, 52)
    # Which cube body axis the fingers closed along, on each side of the switch.
    for sign, y in ((1, 30), (-1, -30)):
        faces = {r["face_axis"] for r in rows if np.sign(float(r["wrist_deg"])) == sign}
        assert len(faces) == 1, "expected one face pair per side of the switch"
        ax_a.text(-0.1 * sign, y, f"fingers on face pair {faces.pop()}", color=INK_2, ha="center")
    ax_a.text(
        0.012,
        0,
        "ε = 0: observed = true\n= 45° switching boundary",
        color=INK_2,
        ha="left",
        va="center",
        fontsize=8,
    )
    ax_a.legend(loc="lower left")

    lifts = [float(r["lift_cm"]) for r in rows]
    low, high = min(lifts) - 0.05, max(lifts) + 0.05
    verified = sum(r["success"] == "True" for r in rows)
    ax_b.set_title("B  Cube lift height after the grasp", loc="left", fontweight="bold")
    ax_b.set_ylabel("Lift height (cm)")
    ax_b.set_ylim(low, high)
    ax_b.text(
        0.02,
        0.04,
        f"Axis spans {high - low:.2f} cm. Commanded lift 10 cm; success needs "
        f"≥ 5 cm.\n{verified} of {len(rows)} grasps verified.",
        transform=ax_b.transAxes,
        color=INK_2,
        fontsize=8,
    )

    fig.suptitle(
        "The 45° yaw ambiguity switches the selected grasp orientation, not the lift outcome",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.905,
        "Stock Lift cube held at a true yaw of 45° from the gripper's start yaw; "
        "only the observed yaw is perturbed. Identical start state for every ε within a seed.",
        color=INK_2,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def plot_finger_gap(rows: list[dict]) -> plt.Figure:
    seeds = sorted({int(r["seed"]) for r in rows})
    paired = {(int(r["seed"]), r["controller"]): r for r in rows}
    misalign, fixed, aligned, span = [], [], [], []
    for seed in seeds:
        f, a = paired[seed, "fixed"], paired[seed, "aligned"]
        assert f["cube_yaw_deg"] == a["cube_yaw_deg"], f"seed {seed} scenes differ"
        # The fixed controller's fingers close along world x (yaw 0): cube body x sits
        # at the cube's yaw from them, so a rigid cube spans wx|cos| + wy|sin| there.
        psi = np.radians(float(f["cube_yaw_deg"]))
        wx, wy = float(f["cube_width_x_mm"]), float(f["cube_width_y_mm"])
        misalign.append(abs(np.degrees((psi + np.pi / 4) % (np.pi / 2) - np.pi / 4)))
        span.append(wx * abs(np.cos(psi)) + wy * abs(np.sin(psi)))
        fixed.append(float(f["gap_mm"]))
        aligned.append(float(a["gap_mm"]))
    widths = [float(r[k]) for r in rows for k in ("cube_width_x_mm", "cube_width_y_mm")]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.axhspan(
        min(widths),
        max(widths),
        color=GRID,
        zorder=0,
        label=f"face-to-face width, recorded cubes ({min(widths):.1f}–{max(widths):.1f} mm)",
    )
    ax.vlines(misalign, aligned, fixed, color=GRID, linewidth=1.0, zorder=1)
    ax.plot(
        misalign,
        span,
        linestyle="none",
        marker="_",
        markersize=9,
        markeredgewidth=1.5,
        color=REFERENCE,
        label="rigid-cube span if not rotated: corner contact",
    )
    ax.plot(
        misalign,
        fixed,
        linestyle="none",
        marker="o",
        markersize=7,
        color=SERIES[0],
        markeredgecolor=SURFACE,
        markeredgewidth=1.0,
        label="fixed orientation (reconstructed baseline)",
    )
    ax.plot(
        misalign,
        aligned,
        linestyle="none",
        marker="s",
        markersize=6,
        color=SERIES[1],
        markeredgecolor=SURFACE,
        markeredgewidth=1.0,
        label="yaw-aligned",
    )

    ax.set_xlim(-1, 46)
    ax.set_xticks(range(0, 46, 5))
    ax.set_xlabel("Cube face misalignment from the fixed finger axis (deg)")
    ax.set_ylabel("Finger gap after lifting (mm)")
    ax.set_title(
        f"Finger gap after the grasp: fixed vs yaw-aligned controller ({len(seeds)} paired seeds)",
        loc="left",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=8)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the Stage 1 top-grasp figures.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="where the CSVs are")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--pdf", action="store_true", help="also write PDF copies")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help=f"rerun both experiments into {REGENERATED_DIR.relative_to(ROOT)} and compare",
    )
    args = parser.parse_args()

    if args.regenerate:
        regenerate(args.data_dir)
    yaw_rows = load(args.data_dir, "yaw_switch.csv")
    gap_rows = load(args.data_dir, "grasp_alignment.csv")
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    save(
        plot_yaw_switch(yaw_rows), "yaw_switch_45deg_selection_vs_lift", args.figures_dir, args.pdf
    )
    save(plot_finger_gap(gap_rows), "finger_gap_fixed_vs_aligned", args.figures_dir, args.pdf)


if __name__ == "__main__":
    main()
