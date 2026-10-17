"""Create initial common-cycle plans and ideal eastbound offsets."""

from pathlib import Path

import numpy as np
import pandas as pd

from traffic_signals.coordination import (
    calculate_offsets,
    is_in_green,
    optimise_fixed_cycle_split,
)
from traffic_signals.optimisation import LOST_TIME_S
from traffic_signals.segmentation import PERIOD_ORDER


OUTPUT_DIR = Path("outputs/data")
DEMAND_PATH = OUTPUT_DIR / "phase2_green_split_evaluation.csv"
OLD_PLANS_PATH = OUTPUT_DIR / "phase2_optimised_plans.csv"
COMMON_CYCLES_PATH = OUTPUT_DIR / "phase3a_common_cycles.csv"
SIGNAL_PLANS_PATH = OUTPUT_DIR / "phase3a_signal_plans.csv"
BANDWIDTH_PATH = OUTPUT_DIR / "phase3a_bandwidth.csv"


def build_coordinated_plans(
    traffic: pd.DataFrame,
    old_plans: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build common cycles, fixed splits, offsets and ideal EB bandwidths."""

    common_cycles = (
        old_plans.groupby("period", as_index=False)["optimal_cycle_s"]
        .max()
        .rename(columns={"optimal_cycle_s": "common_cycle_s"})
    )
    if len(common_cycles) != 4:
        raise ValueError("Expected four common time-of-day cycles")
    common_cycles["period"] = pd.Categorical(
        common_cycles["period"], categories=PERIOD_ORDER, ordered=True
    )
    common_cycles = common_cycles.sort_values("period").reset_index(drop=True)

    observations = traffic.merge(
        common_cycles,
        on="period",
        how="left",
        validate="many_to_one",
    )
    if observations["common_cycle_s"].isna().any():
        raise ValueError("At least one observation did not receive a common cycle")

    records: list[dict[str, object]] = []
    for (junction, period), group in observations.groupby(
        ["intersection_id", "period"], sort=True
    ):
        cycle_s = int(group["common_cycle_s"].iloc[0])
        result = optimise_fixed_cycle_split(
            group["ew_y"].to_numpy(),
            group["ns_y"].to_numpy(),
            cycle_s,
        )
        travel_times, offsets = calculate_offsets(cycle_s)
        records.append(
            {
                "intersection_id": junction,
                "period": period,
                "common_cycle_s": cycle_s,
                "ew_green_s": result["ew_green_s"],
                "ns_green_s": result["ns_green_s"],
                "lost_time_s": LOST_TIME_S,
                "travel_time_from_a_s": travel_times[junction],
                "offset_s": offsets[junction],
                "warning_intervals": result["warning_intervals"],
                "total_shortfall": result["total_shortfall"],
                "min_capacity_margin": result["min_capacity_margin"],
            }
        )

    signal_plans = pd.DataFrame(records)
    signal_plans["period"] = pd.Categorical(
        signal_plans["period"], categories=PERIOD_ORDER, ordered=True
    )
    signal_plans = signal_plans.sort_values(
        ["intersection_id", "period"]
    ).reset_index(drop=True)

    bandwidth_records: list[dict[str, object]] = []
    for period, group in signal_plans.groupby("period", observed=True):
        cycle_s = int(group["common_cycle_s"].iloc[0])
        bandwidth_s = float(group["ew_green_s"].min())
        bandwidth_records.append(
            {
                "period": period,
                "common_cycle_s": cycle_s,
                "eastbound_bandwidth_s": bandwidth_s,
                "bandwidth_ratio": bandwidth_s / cycle_s,
            }
        )

        plan_lookup = group.set_index("intersection_id")
        travel_times, offsets = calculate_offsets(cycle_s)
        for departure_s in (0.0, bandwidth_s / 2.0, bandwidth_s - 0.01):
            for junction in ("A", "B", "C"):
                arrival_s = departure_s + travel_times[junction]
                green_s = float(plan_lookup.loc[junction, "ew_green_s"])
                if not is_in_green(
                    arrival_s,
                    offsets[junction],
                    green_s,
                    cycle_s,
                ):
                    raise ValueError("Ideal bandwidth verification failed")

    bandwidth_summary = pd.DataFrame(bandwidth_records)
    bandwidth_summary["period"] = pd.Categorical(
        bandwidth_summary["period"], categories=PERIOD_ORDER, ordered=True
    )
    bandwidth_summary = bandwidth_summary.sort_values("period").reset_index(drop=True)

    if len(signal_plans) != 12 or len(bandwidth_summary) != 4:
        raise ValueError("Coordination outputs have unexpected row counts")
    allocated_cycle = (
        signal_plans["ew_green_s"]
        + signal_plans["ns_green_s"]
        + signal_plans["lost_time_s"]
    )
    if not np.allclose(allocated_cycle, signal_plans["common_cycle_s"]):
        raise ValueError("Green allocations and lost time do not equal the cycle")
    return common_cycles, signal_plans, bandwidth_summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    traffic = pd.read_csv(DEMAND_PATH)
    old_plans = pd.read_csv(OLD_PLANS_PATH)
    common_cycles, signal_plans, bandwidth_summary = build_coordinated_plans(
        traffic,
        old_plans,
    )

    common_cycles.to_csv(COMMON_CYCLES_PATH, index=False)
    signal_plans.to_csv(SIGNAL_PLANS_PATH, index=False)
    bandwidth_summary.to_csv(BANDWIDTH_PATH, index=False)

    print("Common corridor cycles:")
    print(common_cycles.to_string(index=False))
    print("\nCoordinated signal plans:")
    print(signal_plans.to_string(index=False))
    print("\nIdeal eastbound bandwidth:")
    print(bandwidth_summary.to_string(index=False))
    print(f"\nTotal capacity warnings: {int(signal_plans['warning_intervals'].sum())}")
    print(f"Wrote {COMMON_CYCLES_PATH}")
    print(f"Wrote {SIGNAL_PLANS_PATH}")
    print(f"Wrote {BANDWIDTH_PATH}")


if __name__ == "__main__":
    main()
