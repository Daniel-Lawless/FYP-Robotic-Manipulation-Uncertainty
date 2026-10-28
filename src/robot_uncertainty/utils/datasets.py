"""Read, write and compare small experiment datasets stored as CSV.

Writing never replaces an existing file unless asked to, so a rerun cannot silently
change a version-controlled dataset. Comparison ignores columns that measure wall-clock
time, which varies between runs and machines; every other value must match exactly.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

TIMING_COLUMNS = ("wall_s",)


def write_rows(path: Path | str, rows: list[dict], overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path | str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def diff_rows(
    expected: list[dict], actual: list[dict], ignore: Iterable[str] = TIMING_COLUMNS
) -> list[str]:
    """Human-readable differences between two datasets, ignoring the given columns."""
    ignore = set(ignore)
    if expected and actual and list(expected[0]) != list(actual[0]):
        return [f"columns differ: {list(expected[0])} vs {list(actual[0])}"]
    diffs = []
    if len(expected) != len(actual):
        diffs.append(f"row count differs: {len(expected)} vs {len(actual)}")
    for i, (a, b) in enumerate(zip(expected, actual, strict=False)):  # lengths checked above
        diffs += [f"row {i}, {k}: {a[k]} vs {b[k]}" for k in a if k not in ignore and a[k] != b[k]]
    return diffs
