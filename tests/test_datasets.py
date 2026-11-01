from robot_uncertainty.utils.datasets import TIMING_COLUMNS, diff_rows


def test_timing_columns_are_only_wall_clock() -> None:
    assert TIMING_COLUMNS == ("wall_s",)


def test_extra_columns_are_allowed_only_when_asked_and_never_relax_the_rest() -> None:
    reference = [{"steps": "80", "sim_s": "4.0", "wall_s": "0.7"}]
    regenerated = [{"steps": "80", "sim_s": "4.0", "wall_s": "0.9", "y": "1"}]
    assert diff_rows(reference, regenerated)[0].startswith("columns differ")
    assert diff_rows(reference, regenerated, allow_extra_columns=True) == []
    changed = [{**regenerated[0], "sim_s": "4.05"}]
    assert diff_rows(reference, changed, allow_extra_columns=True) == ["row 0, sim_s: 4.0 vs 4.05"]
    missing = [{"steps": "80", "wall_s": "0.9", "y": "1"}]
    assert diff_rows(reference, missing, allow_extra_columns=True) == ["columns missing: ['sim_s']"]
