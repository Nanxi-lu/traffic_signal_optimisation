import pandas as pd

from workflows.stage1_2_timing import evaluate_fixed_plans, select_fixed_plans
from traffic_signals.demo import calculate_cycle_plan
from traffic_signals.synthetic_data import generate_synthetic_data


def _phase1_results() -> pd.DataFrame:
    return calculate_cycle_plan(generate_synthetic_data())


def test_selected_fixed_cycles_match_p85_rounding_strategy() -> None:
    plans = select_fixed_plans(_phase1_results())
    actual = plans.pivot(
        index="intersection_id", columns="period", values="fixed_cycle_s"
    )
    expected = {
        "A": {"Overnight": 40, "AM peak": 70, "Interpeak": 40, "PM peak": 80},
        "B": {"Overnight": 40, "AM peak": 100, "Interpeak": 40, "PM peak": 100},
        "C": {"Overnight": 40, "AM peak": 145, "Interpeak": 40, "PM peak": 150},
    }

    for intersection, periods in expected.items():
        for period, cycle in periods.items():
            assert actual.loc[intersection, period] == cycle


def test_fixed_plan_evaluation_covers_every_interval() -> None:
    results = _phase1_results()
    plans = select_fixed_plans(results)
    comparison, summary = evaluate_fixed_plans(results, plans)

    assert len(plans) == 12
    assert len(comparison) == len(results) == 288
    assert len(summary) == 12
    assert comparison["fixed_cycle_s"].notna().all()
    assert summary["intervals"].sum() == 288
    assert summary["original_warnings"].sum() == 3
    assert summary["fixed_warnings"].sum() == 3

    warning_rows = summary.loc[summary["fixed_warnings"] > 0]
    assert list(warning_rows["intersection_id"]) == ["C", "C"]
    assert list(warning_rows["fixed_warnings"]) == [1, 2]
