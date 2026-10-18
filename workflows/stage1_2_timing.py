"""Phases 1 and 2: synthetic demand checks, time-of-day periods and fixed timing plans.

Merged from the original phase runner scripts. Each section keeps the
functions of one phase; run a phase with ``python -m workflows <phase>`` from
the repository root. Outputs are written to outputs/data and outputs/figures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from traffic_signals.optimisation import (
    CAPACITY_TOLERANCE as PHASE2_OPTIMISE_CAPACITY_TOLERANCE,
    LOST_TIME_S as PHASE2_OPTIMISE_LOST_TIME_S,
    evaluate_plan,
    optimise_plan,
)
from traffic_signals.segmentation import PERIOD_ORDER, assign_period
from traffic_signals.synthetic_data import SyntheticDataConfig


PHASE_DESCRIPTIONS = {
    "phase1_check": "Phase 1 check",
    "phase2_periods": "Phase 2 period summary",
    "phase2_plans": "Phase 2 p85 fixed plans",
    "phase2_splits": "Phase 2 green splits",
    "phase2_optimise": "Phase 2 grid-search optimisation",
}

# ============================================================================
# Phase 1 check (python -m workflows phase1_check)
# Formerly check_phase1.py. Validate the generated Phase 1 demand and Webster results.
# ============================================================================

TRAFFIC_PATH = Path("outputs/data/synthetic_traffic_15min.csv")
CYCLES_PATH = Path("outputs/data/synthetic_webster_results.csv")
LOST_TIME_S = 12.0


def run_phase1_check() -> None:
    traffic = pd.read_csv(TRAFFIC_PATH)
    cycles = pd.read_csv(CYCLES_PATH)

    print("Configured synthetic demand multipliers:")
    print(pd.Series(SyntheticDataConfig().intersection_factors, name="multiplier"))

    print("\nAverage 15-minute traffic counts:")
    print(traffic.groupby("intersection_id")["volume_15min"].mean())

    print("\nSaturation-flow assumptions (vehicles/hour):")
    print(
        traffic.groupby("intersection_id")["saturation_flow_vph"].agg(
            ["min", "mean", "max"]
        )
    )

    print("\nCycle length statistics (seconds):")
    print(
        cycles.groupby("intersection_id")["cycle_length_s"].agg(
            ["min", "mean", "max"]
        )
    )

    print("\nCalculation status by intersection:")
    statuses = pd.crosstab(cycles["intersection_id"], cycles["status"])
    print(statuses.to_string())

    finite_webster = cycles["total_critical_flow_ratio"].lt(1.0)
    expected_uncapped = (1.5 * LOST_TIME_S + 5.0) / (
        1.0 - cycles.loc[finite_webster, "total_critical_flow_ratio"]
    )
    maximum_formula_error = (
        expected_uncapped
        - cycles.loc[finite_webster, "uncapped_cycle_length_s"]
    ).abs().max()
    print("\nWebster formula validation:")
    print(f"Maximum absolute uncapped-cycle difference: {maximum_formula_error:.3e} seconds")

    capped = cycles.loc[
        cycles["cycle_length_s"].eq(150.0),
        [
            "time",
            "intersection_id",
            "total_critical_flow_ratio",
            "uncapped_cycle_length_s",
            "cycle_length_s",
            "adequately_served",
            "status",
        ],
    ]
    print("\nIntervals at the 150-second maximum:")
    if capped.empty:
        print("None")
    else:
        print(capped.to_string(index=False))

# ============================================================================
# Phase 2 period summary (python -m workflows phase2_periods)
# Formerly phase2.py. Summarise Phase 1 Webster results by illustrative time-of-day period.
# ============================================================================

RESULTS_PATH = Path("outputs/data/synthetic_webster_results.csv")
PHASE2_PERIODS_SUMMARY_PATH = Path("outputs/data/phase2_period_summary.csv")
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


def run_phase2_periods() -> None:
    results = pd.read_csv(RESULTS_PATH, parse_dates=["time"])
    summary = build_period_summary(results)

    PHASE2_PERIODS_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(PHASE2_PERIODS_SUMMARY_PATH, index=False)

    print("Phase 2 time-of-day summary (synthetic Phase 1 results):")
    print(summary.to_string(index=False))
    print(f"\nSummary rows: {len(summary)}")
    print(f"Wrote {PHASE2_PERIODS_SUMMARY_PATH}")

# ============================================================================
# Phase 2 p85 fixed plans (python -m workflows phase2_plans)
# Formerly phase2_plans.py. Select and evaluate exploratory fixed time-of-day signal plans.
# ============================================================================

PLANS_PATH = Path("outputs/data/phase2_fixed_plans.csv")
COMPARISON_PATH = Path("outputs/data/phase2_interval_comparison.csv")
PHASE2_PLANS_EVALUATION_PATH = Path("outputs/data/phase2_plan_evaluation.csv")
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


def run_phase2_plans() -> None:
    results = pd.read_csv(RESULTS_PATH, parse_dates=["time"])
    plans = select_fixed_plans(results)
    comparison, summary = evaluate_fixed_plans(results, plans)

    PLANS_PATH.parent.mkdir(parents=True, exist_ok=True)
    plans.to_csv(PLANS_PATH, index=False)
    comparison.to_csv(COMPARISON_PATH, index=False)
    summary.to_csv(PHASE2_PLANS_EVALUATION_PATH, index=False)

    print("Candidate signal timing plans:")
    print(plans.to_string(index=False))
    print("\nFixed timing plan evaluation:")
    print(summary.to_string(index=False))
    print(f"\nWrote {PLANS_PATH}")
    print(f"Wrote {COMPARISON_PATH}")
    print(f"Wrote {PHASE2_PLANS_EVALUATION_PATH}")

# ============================================================================
# Phase 2 green splits (python -m workflows phase2_splits)
# Formerly phase2_green_splits.py. Design and evaluate fixed EW/NS effective-green allocations.
# ============================================================================

FIXED_PLANS_PATH = Path("outputs/data/phase2_fixed_plans.csv")
GREEN_PLANS_PATH = Path("outputs/data/phase2_green_split_plans.csv")
PHASE2_SPLITS_EVALUATION_PATH = Path("outputs/data/phase2_green_split_evaluation.csv")
PHASE2_SPLITS_SUMMARY_PATH = Path("outputs/data/phase2_green_split_summary.csv")
MIN_GREEN_S = 10.0
APPROACH_MAP = {
    "E": "E",
    "EASTBOUND": "E",
    "W": "W",
    "WESTBOUND": "W",
    "N": "N",
    "NORTHBOUND": "N",
    "S": "S",
    "SOUTHBOUND": "S",
}


def build_phase_observations(traffic: pd.DataFrame) -> pd.DataFrame:
    """Convert approach counts into EW and NS critical flow ratios."""

    required = {
        "time",
        "intersection_id",
        "approach",
        "volume_15min",
        "saturation_flow_vph",
    }
    missing = required.difference(traffic.columns)
    if missing:
        raise ValueError(f"Missing traffic columns: {sorted(missing)}")

    data = traffic.copy()
    data["approach"] = (
        data["approach"].str.strip().str.upper().map(APPROACH_MAP)
    )
    if data["approach"].isna().any():
        raise ValueError("Traffic contains an unrecognised approach label")
    if data.duplicated(["time", "intersection_id", "approach"]).any():
        raise ValueError("Each interval must contain only one row per approach")
    if (data["volume_15min"] < 0).any():
        raise ValueError("volume_15min cannot be negative")
    if (data["saturation_flow_vph"] <= 0).any():
        raise ValueError("saturation_flow_vph must be positive")

    data["flow_ratio"] = (
        data["volume_15min"] * 4.0 / data["saturation_flow_vph"]
    )
    phase = data.pivot(
        index=["time", "intersection_id"],
        columns="approach",
        values="flow_ratio",
    ).reset_index()
    required_approaches = ["E", "W", "N", "S"]
    if any(approach not in phase.columns for approach in required_approaches):
        raise ValueError("Every interval must include E, W, N and S approaches")
    if phase[required_approaches].isna().any().any():
        raise ValueError("Every interval must include all four approaches")

    phase["ew_y"] = phase[["E", "W"]].max(axis=1)
    phase["ns_y"] = phase[["N", "S"]].max(axis=1)
    phase["total_y"] = phase["ew_y"] + phase["ns_y"]
    phase["period"] = phase["time"].apply(assign_period)
    return phase.sort_values(["time", "intersection_id"]).reset_index(drop=True)


def select_green_split_plans(
    phase: pd.DataFrame,
    fixed_plans: pd.DataFrame,
    lost_time_s: float = LOST_TIME_S,
    min_green_s: float = MIN_GREEN_S,
) -> pd.DataFrame:
    """Allocate each fixed cycle using period-level p85 phase demand."""

    design = (
        phase.groupby(["intersection_id", "period"])[["ew_y", "ns_y"]]
        .quantile(0.85)
        .reset_index()
        .rename(columns={"ew_y": "design_ew_y", "ns_y": "design_ns_y"})
    )
    plans = fixed_plans.merge(
        design,
        on=["intersection_id", "period"],
        validate="one_to_one",
    )
    if len(plans) != 12:
        raise ValueError("Expected 12 fixed plans for three intersections and four periods")

    plans["available_green_s"] = plans["fixed_cycle_s"] - lost_time_s
    if (plans["available_green_s"] < 2.0 * min_green_s).any():
        raise ValueError("The fixed cycle cannot provide both minimum green times")

    total_design_y = plans["design_ew_y"] + plans["design_ns_y"]
    ew_share = pd.Series(0.5, index=plans.index, dtype=float)
    positive_demand = total_design_y > 0
    ew_share.loc[positive_demand] = (
        plans.loc[positive_demand, "design_ew_y"]
        / total_design_y.loc[positive_demand]
    )
    proportional_ew_green = plans["available_green_s"] * ew_share
    plans["ew_green_s"] = proportional_ew_green.clip(
        lower=min_green_s,
        upper=plans["available_green_s"] - min_green_s,
    )
    plans["ns_green_s"] = plans["available_green_s"] - plans["ew_green_s"]

    allocated = plans["ew_green_s"] + plans["ns_green_s"] + lost_time_s
    if not np.allclose(allocated, plans["fixed_cycle_s"]):
        raise ValueError("Green allocations and lost time do not sum to the cycle")
    if (plans[["ew_green_s", "ns_green_s"]] < min_green_s).any().any():
        raise ValueError("A phase green allocation is below the illustrative minimum")

    plans["period"] = pd.Categorical(
        plans["period"], categories=PERIOD_ORDER, ordered=True
    )
    return plans.sort_values(["intersection_id", "period"]).reset_index(drop=True)


def evaluate_green_split_plans(
    phase: pd.DataFrame,
    plans: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Screen every interval against its fixed phase green allocations."""

    evaluation = phase.merge(
        plans[
            [
                "intersection_id",
                "period",
                "fixed_cycle_s",
                "ew_green_s",
                "ns_green_s",
            ]
        ],
        on=["intersection_id", "period"],
        how="left",
        validate="many_to_one",
    )
    if len(evaluation) != len(phase):
        raise ValueError("The green-plan join changed the interval count")
    if evaluation["fixed_cycle_s"].isna().any():
        raise ValueError("At least one interval did not receive a green plan")

    evaluation["total_capacity_ratio"] = (
        1.0 - LOST_TIME_S / evaluation["fixed_cycle_s"]
    )
    evaluation["ew_capacity_ratio"] = (
        evaluation["ew_green_s"] / evaluation["fixed_cycle_s"]
    )
    evaluation["ns_capacity_ratio"] = (
        evaluation["ns_green_s"] / evaluation["fixed_cycle_s"]
    )
    evaluation["ew_shortfall"] = (
        evaluation["ew_y"] - evaluation["ew_capacity_ratio"]
    ).clip(lower=0.0)
    evaluation["ns_shortfall"] = (
        evaluation["ns_y"] - evaluation["ns_capacity_ratio"]
    ).clip(lower=0.0)
    evaluation["total_shortfall"] = (
        evaluation["ew_shortfall"] + evaluation["ns_shortfall"]
    )
    evaluation["total_ratio_warning"] = (
        evaluation["total_y"]
        > evaluation["total_capacity_ratio"] + CAPACITY_TOLERANCE
    )
    evaluation["ew_split_warning"] = (
        evaluation["ew_y"]
        > evaluation["ew_capacity_ratio"] + CAPACITY_TOLERANCE
    )
    evaluation["ns_split_warning"] = (
        evaluation["ns_y"]
        > evaluation["ns_capacity_ratio"] + CAPACITY_TOLERANCE
    )
    evaluation["fixed_split_warning"] = (
        evaluation["ew_split_warning"] | evaluation["ns_split_warning"]
    )
    evaluation["allocation_only_warning"] = (
        evaluation["fixed_split_warning"] & ~evaluation["total_ratio_warning"]
    )

    summary = (
        evaluation.groupby(["intersection_id", "period"])
        .agg(
            intervals=("time", "count"),
            fixed_cycle_s=("fixed_cycle_s", "first"),
            ew_green_s=("ew_green_s", "first"),
            ns_green_s=("ns_green_s", "first"),
            total_ratio_warnings=("total_ratio_warning", "sum"),
            ew_split_warnings=("ew_split_warning", "sum"),
            ns_split_warnings=("ns_split_warning", "sum"),
            fixed_split_warnings=("fixed_split_warning", "sum"),
            allocation_only_warnings=("allocation_only_warning", "sum"),
            max_ew_shortfall=("ew_shortfall", "max"),
            max_ns_shortfall=("ns_shortfall", "max"),
        )
        .reset_index()
    )
    summary["period"] = pd.Categorical(
        summary["period"], categories=PERIOD_ORDER, ordered=True
    )
    summary = summary.sort_values(["intersection_id", "period"]).reset_index(drop=True)
    return evaluation, summary


def run_phase2_splits() -> None:
    traffic = pd.read_csv(TRAFFIC_PATH, parse_dates=["time"])
    fixed_plans = pd.read_csv(FIXED_PLANS_PATH)
    phase = build_phase_observations(traffic)
    plans = select_green_split_plans(phase, fixed_plans)
    evaluation, summary = evaluate_green_split_plans(phase, plans)

    GREEN_PLANS_PATH.parent.mkdir(parents=True, exist_ok=True)
    plans.to_csv(GREEN_PLANS_PATH, index=False)
    evaluation.to_csv(PHASE2_SPLITS_EVALUATION_PATH, index=False)
    summary.to_csv(PHASE2_SPLITS_SUMMARY_PATH, index=False)

    print("Fixed cycle and effective-green plans:")
    print(plans.to_string(index=False))
    print("\nPhase-specific capacity evaluation:")
    print(summary.to_string(index=False))
    print(f"\nWrote {GREEN_PLANS_PATH}")
    print(f"Wrote {PHASE2_SPLITS_EVALUATION_PATH}")
    print(f"Wrote {PHASE2_SPLITS_SUMMARY_PATH}")

# ============================================================================
# Phase 2 grid-search optimisation (python -m workflows phase2_optimise)
# Formerly phase2_optimise.py. Optimise fixed cycle lengths and EW/NS green allocations by grid search.
# ============================================================================

OUTPUT_DIR = Path("outputs/data")
PHASE2_OPTIMISE_EVALUATION_PATH = OUTPUT_DIR / "phase2_green_split_evaluation.csv"
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
            (intervals[f"{label}_ew_shortfall"] > PHASE2_OPTIMISE_CAPACITY_TOLERANCE)
            | (intervals[f"{label}_ns_shortfall"] > PHASE2_OPTIMISE_CAPACITY_TOLERANCE)
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
    maximum_total_capacity = (MAX_CYCLE_S - PHASE2_OPTIMISE_LOST_TIME_S) / MAX_CYCLE_S

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
                > maximum_total_capacity + PHASE2_OPTIMISE_CAPACITY_TOLERANCE
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


def run_phase2_optimise() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(PHASE2_OPTIMISE_EVALUATION_PATH, parse_dates=["time"])
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


PHASES = {
    "phase1_check": run_phase1_check,
    "phase2_periods": run_phase2_periods,
    "phase2_plans": run_phase2_plans,
    "phase2_splits": run_phase2_splits,
    "phase2_optimise": run_phase2_optimise,
}
