"""Run the Phase 4A matched AM-peak vehicle and queue experiment."""

from pathlib import Path

import numpy as np
import pandas as pd

from traffic_signals.segmentation import assign_period
from traffic_signals.vehicle_simulation import simulate_direction


DATA_DIR = Path("outputs/data")
PERIOD = "AM peak"
DURATION_S = 3600.0
SPEED_KMH = 40.0
HEADWAY_S = 2.0
RANDOM_SEED = 42
DIRECTIONS = ("EB", "WB")
TRAFFIC_PATH = DATA_DIR / "synthetic_traffic_15min.csv"
PHASE3A_PATH = DATA_DIR / "phase3a_signal_plans.csv"
PHASE3F_PATH = DATA_DIR / "phase3f_optimisation_summary.csv"
SUMMARY_PATH = DATA_DIR / "phase4a_summary.csv"
VEHICLES_PATH = DATA_DIR / "phase4a_vehicle_journeys.csv"
QUEUES_PATH = DATA_DIR / "phase4a_queue_summary.csv"
COMBINED_PATH = DATA_DIR / "phase4a_combined_comparison.csv"


def estimate_am_entry_flows(traffic: pd.DataFrame) -> dict[str, float]:
    """Estimate hourly entry flows from the 16 AM 15-minute observations."""

    df = traffic.copy()
    required = {"time", "intersection_id", "approach", "volume_15min"}
    if not required.issubset(df.columns):
        raise ValueError("Missing required traffic columns.")
    df["time"] = pd.to_datetime(df["time"], errors="raise")
    df["period"] = df["time"].apply(assign_period)
    df["intersection_id"] = df["intersection_id"].astype(str).str.upper()
    df["approach"] = df["approach"].astype(str).str.lower()
    am = df.loc[df["period"] == PERIOD]
    selections = {
        "EB": am.loc[
            (am["intersection_id"] == "A") & (am["approach"] == "eastbound"),
            "volume_15min",
        ],
        "WB": am.loc[
            (am["intersection_id"] == "C") & (am["approach"] == "westbound"),
            "volume_15min",
        ],
    }
    flows: dict[str, float] = {}
    for direction, counts in selections.items():
        numeric = pd.to_numeric(counts, errors="raise")
        if len(numeric) != 16:
            raise ValueError(f"Expected 16 AM observations for {direction}.")
        if not np.isfinite(numeric).all() or (numeric < 0).any():
            raise ValueError("Traffic counts must be finite and non-negative.")
        flows[direction] = float(numeric.mean() * 4.0)
    return flows


def generate_entry_times(
    hourly_flows: dict[str, float],
    duration_s: float = DURATION_S,
    random_seed: int = RANDOM_SEED,
) -> dict[str, np.ndarray]:
    """Generate matched Poisson vehicle counts and uniform entry times."""

    if set(hourly_flows) != set(DIRECTIONS):
        raise ValueError("Hourly flows must define EB and WB.")
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Duration must be positive and finite.")
    if any(not np.isfinite(flow) or flow < 0 for flow in hourly_flows.values()):
        raise ValueError("Hourly flows must be finite and non-negative.")

    rng = np.random.default_rng(random_seed)
    entry_times: dict[str, np.ndarray] = {}
    for direction in DIRECTIONS:
        expected_count = hourly_flows[direction] * duration_s / 3600.0
        vehicle_count = int(rng.poisson(expected_count))
        entry_times[direction] = np.sort(rng.uniform(0.0, duration_s, vehicle_count))
    return entry_times


def build_signal_strategies(
    phase3a: pd.DataFrame,
    phase3f: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    """Load the original and protected AM signal strategies."""

    am_plans = phase3a.loc[phase3a["period"] == PERIOD].set_index(
        "intersection_id"
    )
    if set(am_plans.index) != {"A", "B", "C"} or len(am_plans) != 3:
        raise ValueError("Expected one AM plan for intersections A, B and C.")
    cycle_values = am_plans["common_cycle_s"].unique()
    if len(cycle_values) != 1 or int(cycle_values[0]) != 150:
        raise ValueError("Expected a 150-second AM common cycle.")
    cycle = int(cycle_values[0])
    greens = am_plans["ew_green_s"].to_dict()
    original_offsets = am_plans["offset_s"].to_dict()

    matching = phase3f.loc[
        (phase3f["period"] == PERIOD)
        & (phase3f["strategy"] == "Phase3F_bidirectional")
    ]
    if len(matching) != 1:
        raise ValueError("Expected one Phase 3F AM strategy row.")
    new_plan = matching.iloc[0]
    optimised_offsets = {
        "A": 0.0,
        "B": float(new_plan["offset_b_s"]),
        "C": float(new_plan["offset_c_s"]),
    }
    if optimised_offsets != {"A": 0.0, "B": 28.0, "C": 72.0}:
        raise ValueError("Phase 3F AM offsets do not match the verified plan.")
    return {
        "Phase3A": {
            "cycle_s": cycle,
            "greens": greens,
            "offsets": original_offsets,
        },
        "Phase3F": {
            "cycle_s": cycle,
            "greens": greens,
            "offsets": optimised_offsets,
        },
    }


def run_experiment(
    entry_times: dict[str, np.ndarray],
    strategies: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate both strategies with identical generated vehicles."""

    all_vehicles: list[pd.DataFrame] = []
    all_queues: list[pd.DataFrame] = []
    summary_records: list[dict[str, object]] = []
    for strategy_name, signal_plan in strategies.items():
        for direction in DIRECTIONS:
            vehicles, queues = simulate_direction(
                entry_times=entry_times[direction],
                direction=direction,
                signal_plan=signal_plan,
                speed_kmh=SPEED_KMH,
                headway_s=HEADWAY_S,
            )
            if vehicles.empty:
                raise ValueError("No vehicles generated.")
            vehicles["strategy"] = strategy_name
            vehicles["period"] = PERIOD
            vehicles["exits_after_horizon"] = vehicles["exit_time_s"] > DURATION_S
            queues["strategy"] = strategy_name
            queues["period"] = PERIOD
            summary_records.append(
                {
                    "strategy": strategy_name,
                    "direction": direction,
                    "vehicles": len(vehicles),
                    "mean_delay_s": vehicles["total_delay_s"].mean(),
                    "mean_journey_s": vehicles["journey_time_s"].mean(),
                    "mean_waiting_events": vehicles["waiting_events"].mean(),
                    "no_wait_fraction": (vehicles["waiting_events"] == 0).mean(),
                    "max_queue": int(queues["max_queue"].max()),
                    "exits_after_horizon": int(vehicles["exits_after_horizon"].sum()),
                    "last_exit_s": vehicles["exit_time_s"].max(),
                }
            )
            all_vehicles.append(vehicles)
            all_queues.append(queues)

    vehicle_results = pd.concat(all_vehicles, ignore_index=True)
    queue_results = pd.concat(all_queues, ignore_index=True)
    summary = pd.DataFrame(summary_records)
    combined = (
        vehicle_results.groupby("strategy", as_index=False, sort=False)
        .agg(
            vehicles=("vehicle_id", "count"),
            mean_delay_s=("total_delay_s", "mean"),
            mean_journey_s=("journey_time_s", "mean"),
            mean_waiting_events=("waiting_events", "mean"),
            no_wait_fraction=("waiting_events", lambda values: (values == 0).mean()),
            exits_after_horizon=("exits_after_horizon", "sum"),
            last_exit_s=("exit_time_s", "max"),
        )
    )
    maximum_queues = (
        queue_results.groupby("strategy", as_index=False, sort=False)["max_queue"]
        .max()
        .rename(columns={"max_queue": "max_queue"})
    )
    combined = combined.merge(maximum_queues, on="strategy", validate="one_to_one")

    counts = summary.pivot(index="direction", columns="strategy", values="vehicles")
    if not (counts["Phase3A"] == counts["Phase3F"]).all():
        raise ValueError("Strategies did not receive identical vehicle counts.")
    for direction in DIRECTIONS:
        original_entries = vehicle_results.loc[
            (vehicle_results["strategy"] == "Phase3A")
            & (vehicle_results["direction"] == direction),
            "entry_time_s",
        ].to_numpy()
        optimised_entries = vehicle_results.loc[
            (vehicle_results["strategy"] == "Phase3F")
            & (vehicle_results["direction"] == direction),
            "entry_time_s",
        ].to_numpy()
        if not np.array_equal(original_entries, optimised_entries):
            raise ValueError(f"Strategies did not receive identical {direction} entries.")
    free_flow_s = 900.0 / (SPEED_KMH / 3.6)
    if not np.allclose(
        vehicle_results["journey_time_s"],
        free_flow_s + vehicle_results["total_delay_s"],
    ):
        raise ValueError("Journey-time identity failed.")
    for strategy_name, signal_plan in strategies.items():
        cycle_s = float(signal_plan["cycle_s"])
        greens = signal_plan["greens"]
        offsets = signal_plan["offsets"]
        for direction in DIRECTIONS:
            selected = vehicle_results.loc[
                (vehicle_results["strategy"] == strategy_name)
                & (vehicle_results["direction"] == direction)
            ]
            for junction in ("A", "B", "C"):
                departures = selected[f"discharge_{junction.lower()}_s"].to_numpy()
                if np.any(np.diff(departures) < HEADWAY_S - 1e-9):
                    raise ValueError("FIFO discharge headway check failed.")
                phase = (departures - float(offsets[junction])) % cycle_s
                if np.any(phase >= float(greens[junction]) + 1e-9):
                    raise ValueError("A recorded discharge occurred outside green.")
    if not np.isfinite(vehicle_results.select_dtypes(include=["number"])).all().all():
        raise ValueError("Vehicle results contain non-finite values.")
    return summary, vehicle_results, queue_results, combined


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    traffic = pd.read_csv(TRAFFIC_PATH, parse_dates=["time"])
    phase3a = pd.read_csv(PHASE3A_PATH)
    phase3f = pd.read_csv(PHASE3F_PATH)
    hourly_flows = estimate_am_entry_flows(traffic)
    entry_times = generate_entry_times(hourly_flows)
    strategies = build_signal_strategies(phase3a, phase3f)
    summary, vehicles, queues, combined = run_experiment(entry_times, strategies)

    summary.to_csv(SUMMARY_PATH, index=False)
    vehicles.to_csv(VEHICLES_PATH, index=False)
    queues.to_csv(QUEUES_PATH, index=False)
    combined.to_csv(COMBINED_PATH, index=False)

    print("AM hourly entry-flow estimates:", hourly_flows)
    print("Generated vehicles:", {key: len(value) for key, value in entry_times.items()})
    print("\nPhase 4A directional results:")
    print(summary.to_string(index=False))
    print("\nPhase 4A combined comparison:")
    print(combined.to_string(index=False))
    print(f"\nWrote {SUMMARY_PATH}")
    print(f"Wrote {VEHICLES_PATH}")
    print(f"Wrote {QUEUES_PATH}")
    print(f"Wrote {COMBINED_PATH}")


if __name__ == "__main__":
    main()
