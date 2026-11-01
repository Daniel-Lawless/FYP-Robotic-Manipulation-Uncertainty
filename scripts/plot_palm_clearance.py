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
from matplotlib.lines import Line2D  # noqa: E402

from robot_uncertainty.utils.datasets import diff_rows, read_rows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "validation"
REGENERATED_DIR = ROOT / "results" / "regenerated"
FIGURES_DIR = ROOT / "figures"
DATASET = "palm_clearance.csv"
COMMAND = ["scripts/audit_palm_clearance.py"]
FIGURE = "palm_clearance_centre_vs_raised_grasp"
THRESHOLD_MM = 5.0  # the palm-clearance regression test's limit

# Same palette and style as scripts/plot_grasp_figures.py.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK_2, GRID, REFERENCE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#8a8984"
PHASE_BAND = "#f1f0ec"
STRATEGIES = {
    "centre": ("centre grasp (grasp_height 0)", SERIES[0], "o"),
    "raised": ("raised grasp (+15 mm)", SERIES[1], "s"),
}

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


def load(data_dir: Path) -> list[dict]:
    path = data_dir / DATASET
    if not path.exists():
        command = " ".join(["python", *COMMAND, "--out", str(path)])
        sys.exit(f"Dataset {path} not found. Regenerate it (needs MuJoCo) with:\n  {command}")
    rows = read_rows(path)
    print(f"Read {len(rows)} rows from {path}")
    return rows


# Plotting needs only the CSV and Matplotlib, not MuJoCo. --regenerate first reruns the
# audit (scripts/audit_palm_clearance.py, about a minute) into results/regenerated/ and
# reports any difference from the committed data; it never replaces the committed data.
# See data/validation/README.md.
def regenerate(data_dir: Path) -> None:
    out = REGENERATED_DIR / DATASET
    print(f"Regenerating {out.relative_to(ROOT)} ...", flush=True)
    subprocess.run(
        [sys.executable, *COMMAND, "--out", str(out), "--overwrite"], cwd=ROOT, check=True
    )
    diffs = diff_rows(read_rows(data_dir / DATASET), read_rows(out))
    print(
        f"{DATASET}: regenerated vs {data_dir / DATASET}: "
        + (f"{len(diffs)} differences" if diffs else "identical")
    )
    for line in diffs[:10]:
        print(f"  {line}")
    print(f"Committed data unchanged; the regenerated copy is in {REGENERATED_DIR}.")


def grasps(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Per (config, strategy): step-ordered clearance (mm), phases and palm contacts."""
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["config"], r["strategy"]].append(r)
    out = {}
    for key, group in grouped.items():
        group.sort(key=lambda r: int(r["step"]))
        assert [int(r["step"]) for r in group] == list(range(1, len(group) + 1)), key
        out[key] = dict(
            step=np.arange(1, len(group) + 1),
            clearance=np.array([float(r["clearance_mm"]) for r in group]),
            phase=[r["phase"] for r in group],
            contact=np.array([r["palm_contact"] == "True" for r in group]),
            success=group[0]["success"] == "True",
        )
    return out


def config_label(config: str) -> str:
    if config.startswith("seed"):
        return f"seed {int(config[4:])}"
    x, y = config[len("corner") :].replace("+", " +").replace("-", " -").split()
    return f"corner ({x}, {y}) cm, 45°"


def plot_minima(ax, data, configs, worst) -> None:
    y = np.arange(len(configs))[::-1]
    minima = {s: np.array([data[c, s]["clearance"].min() for c in configs]) for s in STRATEGIES}
    ax.hlines(y, minima["centre"], minima["raised"], color=GRID, linewidth=1.5, zorder=1)
    for strategy, (label, color, marker) in STRATEGIES.items():
        ax.plot(
            minima[strategy],
            y,
            linestyle="none",
            marker=marker,
            markersize=7 if marker == "o" else 6,
            color=color,
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
            label=label,
            zorder=3,
        )
    ax.axvline(0.0, color=INK_2, linewidth=1.0, zorder=0)
    ax.axvline(THRESHOLD_MM, color=REFERENCE, linewidth=1.0, linestyle="--", zorder=0)
    # Reference labels above the top row, each beside its line.
    top = len(configs) - 0.4
    ax.text(-0.15, top, "contact boundary", color=INK_2, fontsize=7.5, ha="right")
    ax.text(THRESHOLD_MM + 0.15, top, "regression-test threshold", color=INK_2, fontsize=7.5)
    labels = [config_label(c) for c in configs]
    ax.set_yticks(y, labels)
    ax.get_yticklabels()[configs.index(worst)].set_fontweight("bold")
    ax.set_ylim(-0.7, len(configs) + 0.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Minimum signed palm–block clearance over the grasp (mm)\n< 0: penetration")
    ax.set_title("A  Closest palm approach, all 16 configurations", loc="left", fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.17), fontsize=8, ncol=1)


def plot_sequence(ax, grasp, strategy, x_max, show_xlabel) -> None:
    label, color, _ = STRATEGIES[strategy]
    steps, clearance, contact = grasp["step"], grasp["clearance"], grasp["contact"]
    # Phase bands from this grasp's own step counts; phase names along the top.
    edges = [0] + [
        i + 1 for i in range(len(steps) - 1) if grasp["phase"][i + 1] != grasp["phase"][i]
    ]
    edges.append(len(steps))
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        if k % 2 == 0:
            ax.axvspan(lo + 0.5, hi + 0.5, color=PHASE_BAND, zorder=0, linewidth=0)
        ax.text(
            (lo + hi) / 2 + 0.5,
            1.0,
            grasp["phase"][lo],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=INK_2,
        )
    ax.axhline(0.0, color=INK_2, linewidth=1.0, zorder=1)
    ax.axhline(THRESHOLD_MM, color=REFERENCE, linewidth=1.0, linestyle="--", zorder=1)
    ax.plot(steps, clearance, color=color, linewidth=2.0, zorder=3, label=label)
    if contact.any():
        ax.plot(
            steps[contact],
            clearance[contact],
            linestyle="none",
            marker="x",
            markersize=4.5,
            markeredgewidth=1.0,
            color=INK,
            zorder=4,
        )
        ax.text(
            steps[contact][-1],
            -6.5,
            f"palm contact on {contact.sum()} steps",
            fontsize=7.5,
            color=INK,
            ha="right",
            va="center",
        )
    ax.axvline(steps[-1] + 0.5, color=INK_2, linewidth=1.0, linestyle=":", zorder=2)
    ax.text(
        steps[-1] + 1.0,
        -3.0,
        f"grasp ends\nafter {steps[-1]} steps",
        fontsize=7.5,
        color=INK_2,
        va="center",
    )
    i = int(np.argmin(clearance))
    ax.annotate(
        f"min {clearance[i]:+.1f} mm\n({grasp['phase'][i]})",
        (steps[i], clearance[i]),
        xytext=(-46, 22) if clearance[i] > 0 else (-80, -2),
        textcoords="offset points",
        fontsize=7.5,
        color=INK,
        arrowprops=dict(arrowstyle="-", color=INK_2, linewidth=0.8),
    )
    if not contact.any():
        ax.text(2, -3.5, "no palm contact at any step", fontsize=7.5, color=INK_2, va="center")
    ax.set_yscale("symlog", linthresh=10, linscale=1.5)
    ax.set_ylim(-8, 250)
    ticks = [-5, 0, 5, 10, 30, 100]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.5, x_max)
    ax.set_ylabel("Palm–block clearance\n(mm; log scale above 10)")
    ax.text(
        0.995,
        0.97,
        label,
        transform=ax.transAxes,
        color=color,
        fontsize=8.5,
        fontweight="bold",
        ha="right",
        va="top",
    )
    if show_xlabel:
        ax.set_xlabel("Control step since the grasp started (20 Hz)")


# Panel A: minimum signed palm-block clearance over each grasp, centre vs raised grasp
# point, for each of the 16 configurations. Panel B: clearance through the grasp for the
# configuration where the raised grasp came closest, one row per grasp point, each with
# its own phase boundaries and contact events on a shared control-step axis.
def plot(rows: list[dict]) -> plt.Figure:
    data = grasps(rows)
    configs = list(dict.fromkeys(r["config"] for r in rows))
    assert set(data) == {(c, s) for c in configs for s in STRATEGIES}, "unpaired grasps"
    assert all(g["success"] for g in data.values())
    worst = min(configs, key=lambda c: data[c, "raised"]["clearance"].min())

    fig = plt.figure(figsize=(12.5, 5.6))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.55), hspace=0.42, wspace=0.42)
    plot_minima(fig.add_subplot(grid[:, 0]), data, configs, worst)
    x_max = max(len(data[worst, s]["step"]) for s in STRATEGIES) + 13
    top = fig.add_subplot(grid[0, 1])
    bottom = fig.add_subplot(grid[1, 1], sharex=top)
    plot_sequence(top, data[worst, "centre"], "centre", x_max, show_xlabel=False)
    plot_sequence(bottom, data[worst, "raised"], "raised", x_max, show_xlabel=True)
    # One legend for both rows, outside the plotting area.
    handles = [
        Line2D([], [], color=STRATEGIES["centre"][1], linewidth=2.0, label=STRATEGIES["centre"][0]),
        Line2D([], [], color=STRATEGIES["raised"][1], linewidth=2.0, label=STRATEGIES["raised"][0]),
        Line2D(
            [], [], color=INK, marker="x", linestyle="none", label="palm in MuJoCo's contact list"
        ),
        Line2D([], [], color=INK_2, linewidth=1.0, label="0 mm: contact boundary"),
        Line2D(
            [], [], color=REFERENCE, linestyle="--", linewidth=1.0, label="5 mm: test threshold"
        ),
    ]
    bottom.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=7.5
    )
    top.set_title(
        f"B  Through the grasp: {config_label(worst)} (least raised-grasp clearance)\n",
        loc="left",
        fontweight="bold",
    )

    fig.suptitle(
        "Raising the grasp point 15 mm keeps the palm off the 8 cm block; "
        "gripping at its centre presses the palm into it",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.925,
        "Signed distance from MuJoCo's collider (palm collision mesh vs block box); contacts from "
        "the live contact list. All 32 grasps lifted the block. Panel B rows have separate phase "
        "boundaries; the step axis is shared, not aligned by phase.",
        color=INK_2,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.13, right=0.97, top=0.82, bottom=0.2)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the palm-clearance figure.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="where the CSV is")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help=f"rerun the audit into {REGENERATED_DIR.relative_to(ROOT)} and compare",
    )
    args = parser.parse_args()

    if args.regenerate:
        regenerate(args.data_dir)
    rows = load(args.data_dir)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    out = args.figures_dir / f"{FIGURE}.png"
    fig = plot(rows)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
