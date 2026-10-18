import pandas as pd

from workflows.stage1_2_timing import EXPECTED_INTERVALS, build_period_summary
from traffic_signals.demo import calculate_cycle_plan
from traffic_signals.segmentation import assign_period
from traffic_signals.synthetic_data import generate_synthetic_data


def test_assign_period_boundaries() -> None:
    expected = {
        "00:00": "Overnight",
        "05:45": "Overnight",
        "06:00": "AM peak",
        "09:45": "AM peak",
        "10:00": "Interpeak",
        "15:45": "Interpeak",
        "16:00": "PM peak",
        "19:45": "PM peak",
        "20:00": "Overnight",
        "23:45": "Overnight",
    }

    for time, period in expected.items():
        assert assign_period(pd.Timestamp(f"2025-01-15 {time}")) == period


def test_period_summary_has_expected_rows_and_interval_counts() -> None:
    results = calculate_cycle_plan(generate_synthetic_data())
    summary = build_period_summary(results)

    assert len(summary) == 12
    for period, expected_count in EXPECTED_INTERVALS.items():
        period_rows = summary.loc[summary["period"] == period]
        assert len(period_rows) == 3
        assert period_rows["intervals"].eq(expected_count).all()
    assert summary["intervals"].sum() == 288
    assert summary["capacity_warnings"].sum() == 3
