"""Optimise fixed cycle lengths and EW/NS green allocations by grid search."""

from pathlib import Path

import pandas as pd

from traffic_signals.optimisation import (
    CAPACITY_TOLERANCE,
    LOST_TIME_S,
    evaluate_plan,
    optimise_plan,
)
from traffic_signals.segmentation import PERIOD_ORDER


OUTPUT_DIR = Path("outputs/data")
EVALUATION_PATH = OUTPUT_DIR / "phase2_green_split_evaluation.csv"
BASELINE_PLANS_PATH = OUTPUT_DIR / "phase2_green_split_plans.csv"
OPTIMISED_PLANS_PATH = OUTPUT_DIR / "phase2_optimised_plans.csv"
INTERVAL_EVALUATION_PATH = OUTPUT_DIR / "phase2_optimised_interval_evaluation.csv"
MAX_CYCLE_S = 150.0


def _interval_evaluation(
    group: pd.DataFrame,
    baseline: dict[str, float | int],
    optimal: dict[str, float | int],
) -> pd.DataFrame:
    """Add baseline and optimal phase-capacity results to a period's rows."""

    intervals = group.copy()
    for label, result in (("baseline", baseline), ("optimal", optimal)):
        ew_capacity = float(result["ew_capacity_ratio"])
        ns_capacity = float(result["ns_capacity_ratio"])
        intervals[f"{label}_cycle_s"] = float(result["cycle_s"])
        intervals[f"{label}_ew_green_s"] = float(result["ew_green_s"])
        intervals[f"{label}_ns_green_s"] = float(result["ns_green_s"])
        intervals[f"{label}_ew_capacity_ratio"] = ew_capacity
        intervals[f"{label}_ns_capacity_ratio"] = ns_capacity
        intervals[f"{label}_ew_shortfall"] = (
            intervals["ew_y"] - ew_capacity
        ).clip(lower=0.0)
        intervals[f"{label}_ns_shortfall"] = (
            intervals["ns_y"] - ns_capacity
        ).clip(lower=0.0)
        intervals[f"{label}_warning"] = (
            (intervals[f"{label}_ew_shortfall"] > CAPACITY_TOLERANCE)
            | (intervals[f"{label}_ns_shortfall"] > CAPACITY_TOLERANCE)
        )
    return intervals


def optimise_all_plans(
    data: pd.DataFrame,
    baseline_plans: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Optimise each intersection/period plan independently."""

    if len(data) != 288:
        raise ValueError("Expected 288 interval observations")
    if len(baseline_plans) != 12:
        raise ValueError("Expected 12 baseline plans")

    results: list[dict[str, object]] = []
    interval_results: list[pd.DataFrame] = []
    maximum_total_capacity = (MAX_CYCLE_S - LOST_TIME_S) / MAX_CYCLE_S

    for (junction, period), group in data.groupby(
        ["intersection_id", "period"], sort=True
    ):
        matching = baseline_plans.loc[
            (baseline_plans["intersection_id"] == junction)
            & (baseline_plans["period"] == period)
        ]
        if len(matching) != 1:
            raise ValueError(f"Expected one baseline plan for {junction}, {period}")
        old = matching.iloc[0]
        ew_y = group["ew_y"].to_numpy()
        ns_y = group["ns_y"].to_numpy()

        baseline_evaluation = evaluate_plan(
            ew_y,
            ns_y,
            float(old["fixed_cycle_s"]),
            float(old["ew_green_s"]),
        )
        baseline: dict[str, float | int] = {
            "cycle_s": int(old["fixed_cycle_s"]),
            "ew_green_s": float(old["ew_green_s"]),
            **baseline_evaluation,
        }
        optimal = optimise_plan(
            ew_y,
            ns_y,
            float(old["fixed_cycle_s"]),
            float(old["ew_green_s"]),
        )
        aggregate_lower_bound = int(
            (
                group["ew_y"] + group["ns_y"]
                > maximum_total_capacity + CAPACITY_TOLERANCE
            ).sum()
        )

        results.append(
            {
                "intersection_id": junction,
                "period": period,
                "intervals": len(group),
                "baseline_cycle_s": baseline["cycle_s"],
                "optimal_cycle_s": optimal["cycle_s"],
                "baseline_ew_green_s": baseline["ew_green_s"],
                "baseline_ns_green_s": baseline["ns_green_s"],
                "optimal_ew_green_s": optimal["ew_green_s"],
                "optimal_ns_green_s": optimal["ns_green_s"],
                "baseline_warnings": baseline["warning_intervals"],
                "optimal_warnings": optimal["warning_intervals"],
                "baseline_total_shortfall": baseline["total_shortfall"],
                "optimal_total_shortfall": optimal["total_shortfall"],
                "baseline_min_capacity_margin": baseline["min_capacity_margin"],
                "optimal_min_capacity_margin": optimal["min_capacity_margin"],
                "aggregate_lower_bound": aggregate_lower_bound,
            }
        )
        interval_results.append(_interval_evaluation(group, baseline, optimal))

    summary = pd.DataFrame(results)
    summary["period"] = pd.Categorical(
        summary["period"], categories=PERIOD_ORDER, ordered=True
    )
    summary = summary.sort_values(["intersection_id", "period"]).reset_index(drop=True)
    interval_comparison = pd.concat(interval_results, ignore_index=True)

    if len(summary) != 12 or len(interval_comparison) != 288:
        raise ValueError("Optimisation outputs have unexpected row counts")
    if not (summary["optimal_warnings"] <= summary["baseline_warnings"]).all():
        raise ValueError("An optimised plan has more warnings than its baseline")
    return summary, interval_comparison


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(EVALUATION_PATH, parse_dates=["time"])
    baseline_plans = pd.read_csv(BASELINE_PLANS_PATH)
    print(f"Observations: {len(data)}")
    print(f"Baseline plans: {len(baseline_plans)}")

    summary, interval_comparison = optimise_all_plans(data, baseline_plans)
    summary.to_csv(OPTIMISED_PLANS_PATH, index=False)
    interval_comparison.to_csv(INTERVAL_EVALUATION_PATH, index=False)

    print("\nPhase 2D optimisation results:")
    print(summary.to_string(index=False))
    print(f"\nTotal baseline warnings: {int(summary['baseline_warnings'].sum())}")
    print(f"Total optimised warnings: {int(summary['optimal_warnings'].sum())}")
    print(
        "Aggregate theoretical lower bound: "
        f"{int(summary['aggregate_lower_bound'].sum())}"
    )
    print(f"Wrote {OPTIMISED_PLANS_PATH}")
    print(f"Wrote {INTERVAL_EVALUATION_PATH}")


if __name__ == "__main__":
    main()
