"""Every script must start and print its usage (needs MuJoCo for the imports)."""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("robosuite")

SCRIPTS = sorted((Path(__file__).resolve().parents[2] / "scripts").glob("*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_help_runs(script: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("usage:")
