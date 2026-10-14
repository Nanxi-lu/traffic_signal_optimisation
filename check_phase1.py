"""Validate the generated Phase 1 demand and Webster results."""

from pathlib import Path

import pandas as pd

from traffic_signals.synthetic_data import SyntheticDataConfig


TRAFFIC_PATH = Path("outputs/data/synthetic_traffic_15min.csv")
CYCLES_PATH = Path("outputs/data/synthetic_webster_results.csv")
LOST_TIME_S = 12.0


def main() -> None:
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


if __name__ == "__main__":
    main()
