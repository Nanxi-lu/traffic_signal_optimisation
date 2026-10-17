"""Design and evaluate fixed EW/NS effective-green allocations."""

from pathlib import Path

import numpy as np
import pandas as pd

from traffic_signals.segmentation import PERIOD_ORDER, assign_period


TRAFFIC_PATH = Path("outputs/data/synthetic_traffic_15min.csv")
FIXED_PLANS_PATH = Path("outputs/data/phase2_fixed_plans.csv")
GREEN_PLANS_PATH = Path("outputs/data/phase2_green_split_plans.csv")
EVALUATION_PATH = Path("outputs/data/phase2_green_split_evaluation.csv")
SUMMARY_PATH = Path("outputs/data/phase2_green_split_summary.csv")

LOST_TIME_S = 12.0
MIN_GREEN_S = 10.0
CAPACITY_TOLERANCE = 1e-9
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


def main() -> None:
    traffic = pd.read_csv(TRAFFIC_PATH, parse_dates=["time"])
    fixed_plans = pd.read_csv(FIXED_PLANS_PATH)
    phase = build_phase_observations(traffic)
    plans = select_green_split_plans(phase, fixed_plans)
    evaluation, summary = evaluate_green_split_plans(phase, plans)

    GREEN_PLANS_PATH.parent.mkdir(parents=True, exist_ok=True)
    plans.to_csv(GREEN_PLANS_PATH, index=False)
    evaluation.to_csv(EVALUATION_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("Fixed cycle and effective-green plans:")
    print(plans.to_string(index=False))
    print("\nPhase-specific capacity evaluation:")
    print(summary.to_string(index=False))
    print(f"\nWrote {GREEN_PLANS_PATH}")
    print(f"Wrote {EVALUATION_PATH}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
