"""Select and evaluate exploratory fixed time-of-day signal plans."""

from pathlib import Path

import numpy as np
import pandas as pd

from traffic_signals.segmentation import PERIOD_ORDER, assign_period


RESULTS_PATH = Path("outputs/data/synthetic_webster_results.csv")
PLANS_PATH = Path("outputs/data/phase2_fixed_plans.csv")
COMPARISON_PATH = Path("outputs/data/phase2_interval_comparison.csv")
EVALUATION_PATH = Path("outputs/data/phase2_plan_evaluation.csv")
LOST_TIME_S = 12.0
CAPACITY_TOLERANCE = 1e-9


def select_fixed_plans(results: pd.DataFrame) -> pd.DataFrame:
    """Select each period's p85 cycle, rounded upward to five seconds."""

    data = results.copy()
    data["period"] = data["time"].apply(assign_period)
    plans = (
        data.groupby(["intersection_id", "period"])["cycle_length_s"]
        .quantile(0.85)
        .reset_index(name="p85_cycle_s")
    )
    plans["fixed_cycle_s"] = (
        np.ceil(plans["p85_cycle_s"] / 5.0) * 5.0
    ).clip(lower=40, upper=150).astype(int)
    plans["period"] = pd.Categorical(
        plans["period"], categories=PERIOD_ORDER, ordered=True
    )
    return plans.sort_values(["intersection_id", "period"]).reset_index(drop=True)


def evaluate_fixed_plans(
    results: pd.DataFrame,
    plans: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare fixed plans with every original 15-minute calculation."""

    data = results.copy()
    data["period"] = data["time"].apply(assign_period)
    comparison = data.merge(
        plans[["intersection_id", "period", "fixed_cycle_s"]],
        on=["intersection_id", "period"],
        how="left",
        validate="many_to_one",
    )

    if len(comparison) != len(data):
        raise ValueError("The timing-plan join changed the interval count")
    if len(plans) != 12:
        raise ValueError("Expected 12 plans for three intersections and four periods")
    if comparison["fixed_cycle_s"].isna().any():
        raise ValueError("At least one interval did not receive a fixed timing plan")
    if (comparison["cycle_length_s"] <= LOST_TIME_S).any():
        raise ValueError("Original cycles must be greater than lost time")
    if (comparison["fixed_cycle_s"] <= LOST_TIME_S).any():
        raise ValueError("Fixed cycles must be greater than lost time")

    comparison["original_capacity_limit"] = (
        1.0 - LOST_TIME_S / comparison["cycle_length_s"]
    )
    comparison["fixed_capacity_limit"] = (
        1.0 - LOST_TIME_S / comparison["fixed_cycle_s"]
    )
    comparison["original_warning"] = (
        comparison["total_critical_flow_ratio"]
        > comparison["original_capacity_limit"] + CAPACITY_TOLERANCE
    )
    comparison["fixed_warning"] = (
        comparison["total_critical_flow_ratio"]
        > comparison["fixed_capacity_limit"] + CAPACITY_TOLERANCE
    )
    reported_adequacy = (
        comparison["adequately_served"].astype(str).str.lower().eq("true")
    )
    if not comparison["original_warning"].eq(~reported_adequacy).all():
        raise ValueError("Recalculated original warnings do not match Phase 1 flags")

    summary = (
        comparison.groupby(["intersection_id", "period"])
        .agg(
            intervals=("time", "count"),
            original_mean_cycle=("cycle_length_s", "mean"),
            fixed_cycle=("fixed_cycle_s", "first"),
            original_warnings=("original_warning", "sum"),
            fixed_warnings=("fixed_warning", "sum"),
        )
        .reset_index()
    )
    summary["period"] = pd.Categorical(
        summary["period"], categories=PERIOD_ORDER, ordered=True
    )
    summary = summary.sort_values(["intersection_id", "period"]).reset_index(drop=True)
    return comparison, summary


def main() -> None:
    results = pd.read_csv(RESULTS_PATH, parse_dates=["time"])
    plans = select_fixed_plans(results)
    comparison, summary = evaluate_fixed_plans(results, plans)

    PLANS_PATH.parent.mkdir(parents=True, exist_ok=True)
    plans.to_csv(PLANS_PATH, index=False)
    comparison.to_csv(COMPARISON_PATH, index=False)
    summary.to_csv(EVALUATION_PATH, index=False)

    print("Candidate signal timing plans:")
    print(plans.to_string(index=False))
    print("\nFixed timing plan evaluation:")
    print(summary.to_string(index=False))
    print(f"\nWrote {PLANS_PATH}")
    print(f"Wrote {COMPARISON_PATH}")
    print(f"Wrote {EVALUATION_PATH}")


if __name__ == "__main__":
    main()
