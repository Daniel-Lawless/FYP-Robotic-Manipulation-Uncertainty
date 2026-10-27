import importlib

import pytest

SUBPACKAGES = [
    "sim",
    "policies",
    "dataset",
    "ensemble",
    "rl",
    "analysis",
    "intervention",
    "utils",
]


def test_package_imports() -> None:
    importlib.import_module("robot_uncertainty")


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    importlib.import_module(f"robot_uncertainty.{name}")
