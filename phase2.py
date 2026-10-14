"""Summarise Phase 1 Webster results by illustrative time-of-day period."""

from pathlib import Path

import pandas as pd

from traffic_signals.segmentation import PERIOD_ORDER, assign_period


RESULTS_PATH = Path("outputs/data/synthetic_webster_results.csv")
SUMMARY_PATH = Path("outputs/data/phase2_period_summary.csv")
EXPECTED_INTERVALS = {
    "Overnight": 40,
    "AM peak": 16,
    "Interpeak": 24,
    "PM peak": 16,
}


def build_period_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Assign periods and calculate one summary row per intersection/period."""

    data = results.copy()
    data["period"] = data["time"].apply(assign_period)
    data["capacity_warning"] = (
        data["adequately_served"].astype(str).str.lower().eq("false")
    )

    interval_counts = data.groupby(
        ["intersection_id", "period"]
    ).size().unstack(fill_value=0)
    for period, expected_count in EXPECTED_INTERVALS.items():
        if period not in interval_counts or not interval_counts[period].eq(expected_count).all():
            raise ValueError(
                f"Expected {expected_count} intervals per intersection for {period}"
            )

    summary = (
        data.groupby(["intersection_id", "period"])
        .agg(
            intervals=("time", "count"),
            mean_critical_ratio=("total_critical_flow_ratio", "mean"),
            max_critical_ratio=("total_critical_flow_ratio", "max"),
            mean_cycle=("cycle_length_s", "mean"),
            cycle_p85=("cycle_length_s", lambda values: values.quantile(0.85)),
            max_cycle=("cycle_length_s", "max"),
            capacity_warnings=("capacity_warning", "sum"),
        )
        .reset_index()
    )
    summary["period"] = pd.Categorical(
        summary["period"], categories=PERIOD_ORDER, ordered=True
    )
    return summary.sort_values(["intersection_id", "period"]).reset_index(drop=True)


def main() -> None:
    results = pd.read_csv(RESULTS_PATH, parse_dates=["time"])
    summary = build_period_summary(results)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("Phase 2 time-of-day summary (synthetic Phase 1 results):")
    print(summary.to_string(index=False))
    print(f"\nSummary rows: {len(summary)}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
