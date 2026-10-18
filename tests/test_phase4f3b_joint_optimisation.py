import pandas as pd
import pytest

from workflows.stage4f_4g_full_junction import (
    PROTECTED_MOVEMENTS,
    evaluate_fairness,
    make_candidate_pairs,
    make_candidate_plan,
)
from traffic_signals.full_junction import create_two_phase_plan, is_effective_green


def make_test_plans():
    return {
        "A": create_two_phase_plan(150, 87, 51, 0),
        "B": create_two_phase_plan(150, 88, 50, 36),
        "C": create_two_phase_plan(150, 95, 43, 70),
    }


def test_candidate_grid_contains_35_unique_local_pairs():
    candidates = make_candidate_pairs(70)
    assert len(candidates) == 35
    assert len(set(candidates)) == 35
    assert (0, 70) in candidates
    assert {offset for _, offset in candidates} == {58, 62, 66, 70, 74, 78, 82}
    with pytest.raises(ValueError):
        make_candidate_pairs(70, cycle=0)


def test_candidate_changes_only_c_and_preserves_effective_green():
    base = make_test_plans()
    candidate = make_candidate_plan(base, 5, 74)
    assert candidate["A"] == base["A"]
    assert candidate["B"] == base["B"]
    assert candidate["C"]["ew_green_s"] == 90
    assert candidate["C"]["ns_green_s"] == 48
    assert candidate["C"]["ew_offset_s"] == 74
    assert candidate["C"]["ns_offset_s"] == 20
    assert base["C"]["ew_offset_s"] == 70
    with pytest.raises(ValueError):
        make_candidate_plan(base, 4, 70)


def test_candidate_effective_green_windows_do_not_overlap():
    candidate = make_candidate_plan(make_test_plans(), 10, 82)["C"]
    for time_s in range(150):
        assert not (
            is_effective_green(time_s, candidate, "EW")
            and is_effective_green(time_s, candidate, "NS")
        )


def test_fairness_uses_bootstrap_upper_bound():
    rows = []
    for movement in PROTECTED_MOVEMENTS:
        changes = [1.0, 1.5, 1.8, 1.9]
        if movement == "WB":
            changes = [2.2, 2.3, 2.4, 2.5]
        rows.extend(
            {"movement": movement, "delay_change_s": change}
            for change in changes
        )
    result = evaluate_fairness(pd.DataFrame(rows)).set_index("movement")
    assert bool(result.loc["EB", "fairness_pass"])
    assert not bool(result.loc["WB", "fairness_pass"])
    assert result.loc["WB", "ci_upper_s"] > 2.0
    assert result.loc["C_N", "ci_upper_s"] <= 2.0
