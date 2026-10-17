"""Compare Phase 3A and Phase 4E with local north-south traffic."""

from pathlib import Path
from collections.abc import Mapping

import numpy as np
import pandas as pd

from traffic_signals.full_junction import (
    create_two_phase_plan,
    simulate_ns_approach,
)
from traffic_signals.hourly_arrivals import generate_hourly_arrivals
from traffic_signals.vehicle_simulation import simulate_direction


DATA_DIR = Path("outputs/data")
TRAFFIC_PATH = DATA_DIR / "synthetic_traffic_15min.csv"
PHASE3A_PATH = DATA_DIR / "phase3a_signal_plans.csv"
PHASE4E_PATH = DATA_DIR / "phase4e_selected_plans.csv"
CORRIDOR_PATH = DATA_DIR / "phase4f1_corridor_results.csv"
NS_PATH = DATA_DIR / "phase4f1_ns_approaches.csv"
NETWORK_PATH = DATA_DIR / "phase4f1_network_summary.csv"
SIGNAL_PATH = DATA_DIR / "phase4f1_signal_windows.csv"
CAPACITY_PATH = DATA_DIR / "phase4f1_c_ns_capacity.csv"

PERIOD = "AM peak"
SEED = 42
HORIZON_S = 3600.0
HEADWAY_S = 2.0
SPEED_KMH = 40.0
JUNCTIONS = ("A", "B", "C")
STREAMS = (
    ("A", "E"),
    ("C", "W"),
    ("A", "N"),
    ("A", "S"),
    ("B", "N"),
    ("B", "S"),
    ("C", "N"),
    ("C", "S"),
)


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Timestamp,
]:
    """Load synthetic traffic and fixed AM Phase 3A/Phase 4E plans."""

    traffic = pd.read_csv(TRAFFIC_PATH, parse_dates=["time"])
    traffic["intersection_id"] = traffic["intersection_id"].astype(str).str.upper()
    traffic["approach"] = (
        traffic["approach"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace(
            {
                "EASTBOUND": "E",
                "WESTBOUND": "W",
                "NORTHBOUND": "N",
                "SOUTHBOUND": "S",
            }
        )
    )
    plans_3a = pd.read_csv(PHASE3A_PATH)
    period_plans = plans_3a.loc[plans_3a["period"] == PERIOD].set_index(
        "intersection_id"
    )
    if set(period_plans.index) != set(JUNCTIONS) or len(period_plans) != 3:
        raise ValueError("Expected AM Phase 3A plans for A, B and C.")

    plans_4e = pd.read_csv(PHASE4E_PATH)
    selected = plans_4e.loc[plans_4e["period"] == PERIOD]
    if len(selected) != 1:
        raise ValueError("Expected one selected Phase 4E AM plan.")
    selected_row = selected.iloc[0]
    selected_offsets = {
        "A": int(selected_row["A_offset_s"]),
        "B": int(selected_row["B_offset_s"]),
        "C": int(selected_row["C_offset_s"]),
    }
    if selected_offsets != {"A": 0, "B": 36, "C": 70}:
        raise ValueError("Phase 4E AM offsets must be the frozen 0/36/70 plan.")
    study_day = traffic["time"].dt.normalize().min()
    start_time = study_day + pd.Timedelta(hours=8)
    return traffic, period_plans, selected_row, start_time


def build_strategy(
    period_plans: pd.DataFrame,
    offsets: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    """Build the three fixed two-phase junction schedules."""

    if set(offsets) != set(JUNCTIONS):
        raise ValueError("Offsets must define A, B and C.")
    junction_plans: dict[str, dict[str, float]] = {}
    for junction in JUNCTIONS:
        row = period_plans.loc[junction]
        junction_plans[junction] = create_two_phase_plan(
            cycle_s=float(row["common_cycle_s"]),
            ew_green_s=float(row["ew_green_s"]),
            ns_green_s=float(row["ns_green_s"]),
            ew_offset_s=float(offsets[junction]),
            lost_time_s=float(row["lost_time_s"]),
            interphase_loss_s=float(row["lost_time_s"]) / 2.0,
        )
    cycles = {plan["cycle_s"] for plan in junction_plans.values()}
    if len(cycles) != 1:
        raise ValueError("Junction plans do not share a common cycle.")
    return junction_plans


def capacity_diagnostic(
    traffic: pd.DataFrame,
    period_plans: pd.DataFrame,
    start_time: pd.Timestamp,
) -> pd.DataFrame:
    """Compare Junction C's critical N/S demand with its green capacity."""

    c_plan = period_plans.loc["C"]
    capacity_ratio = float(c_plan["ns_green_s"] / c_plan["common_cycle_s"])
    records: list[dict[str, object]] = []
    for time in pd.date_range(start_time, periods=4, freq="15min"):
        subset = traffic.loc[
            (traffic["time"] == time)
            & (traffic["intersection_id"] == "C")
            & (traffic["approach"].isin(["N", "S"]))
        ]
        if len(subset) != 2 or set(subset["approach"]) != {"N", "S"}:
            raise ValueError("Expected N and S observations at Junction C.")
        subset = subset.set_index("approach")
        counts = pd.to_numeric(subset["volume_15min"], errors="raise")
        saturation = pd.to_numeric(
            subset["saturation_flow_vph"], errors="raise"
        )
        if (
            not np.isfinite(counts).all()
            or (counts < 0).any()
            or not np.isfinite(saturation).all()
            or (saturation <= 0).any()
        ):
            raise ValueError("Invalid Junction C demand or saturation flow.")
        critical_approach = str(counts.idxmax())
        critical_count = float(counts.loc[critical_approach])
        saturation_flow = float(saturation.loc[critical_approach])
        critical_flow = critical_count * 4.0
        critical_ratio = critical_flow / saturation_flow
        records.append(
            {
                "time": time,
                "intersection_id": "C",
                "north_count_15min": int(counts.loc["N"]),
                "south_count_15min": int(counts.loc["S"]),
                "critical_approach": critical_approach,
                "critical_count_15min": int(critical_count),
                "critical_flow_vph": critical_flow,
                "saturation_flow_vph": saturation_flow,
                "critical_ratio": critical_ratio,
                "ns_capacity_ratio": capacity_ratio,
                "ns_capacity_vph": capacity_ratio * saturation_flow,
                "exceeds_green_capacity": bool(
                    critical_ratio > capacity_ratio + 1e-12
                ),
            }
        )
    result = pd.DataFrame(records)
    if len(result) != 4:
        raise ValueError("Expected four Junction C capacity rows.")
    return result


def _corridor_metrics(
    vehicles: pd.DataFrame,
    queues: pd.DataFrame,
) -> dict[str, float | int]:
    """Summarise one east-west corridor direction."""

    vehicle_count = len(vehicles)
    return {
        "vehicles": vehicle_count,
        "junction_passages": vehicle_count * 3,
        "total_delay_s": float(vehicles["total_delay_s"].sum()),
        "mean_delay_s": (
            float(vehicles["total_delay_s"].mean()) if vehicle_count else 0.0
        ),
        "mean_journey_s": (
            float(vehicles["journey_time_s"].mean()) if vehicle_count else 0.0
        ),
        "waiting_events": int(vehicles["waiting_events"].sum()),
        "mean_waiting_events": (
            float(vehicles["waiting_events"].mean()) if vehicle_count else 0.0
        ),
        "max_queue_veh": int(queues["max_queue"].max()),
        "queue_at_horizon_veh": int(queues["queue_at_horizon_veh"].sum()),
        "exits_after_horizon": int(
            (vehicles["exit_time_s"] > HORIZON_S).sum()
        ),
    }


def run_experiment() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run the matched 08:00--09:00 Phase 3A/Phase 4E comparison."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    traffic, period_plans, selected, start_time = load_inputs()
    arrivals = generate_hourly_arrivals(
        traffic=traffic,
        start_time=start_time,
        streams=STREAMS,
        seed=SEED,
    )
    print("Generated streams:")
    for stream, times in arrivals.items():
        print(stream, len(times))

    original_offsets = period_plans["offset_s"].to_dict()
    optimised_offsets = {
        "A": int(selected["A_offset_s"]),
        "B": int(selected["B_offset_s"]),
        "C": int(selected["C_offset_s"]),
    }
    strategies = {
        "Phase3A": build_strategy(period_plans, original_offsets),
        "Phase4E": build_strategy(period_plans, optimised_offsets),
    }

    corridor_records: list[dict[str, object]] = []
    ns_records: list[dict[str, object]] = []
    network_records: list[dict[str, object]] = []
    signal_records: list[dict[str, object]] = []

    for strategy_name, junction_plans in strategies.items():
        corridor_plan = {
            "cycle_s": junction_plans["A"]["cycle_s"],
            "greens": {
                junction: junction_plans[junction]["ew_green_s"]
                for junction in JUNCTIONS
            },
            "offsets": {
                junction: junction_plans[junction]["ew_offset_s"]
                for junction in JUNCTIONS
            },
        }
        for junction, plan in junction_plans.items():
            signal_records.append(
                {"strategy": strategy_name, "junction": junction, **plan}
            )

        corridor_delay_total = 0.0
        corridor_vehicle_total = 0
        corridor_passage_total = 0
        corridor_queue_total = 0
        corridor_waiting_events = 0
        for direction, stream_key in (
            ("EB", ("A", "E")),
            ("WB", ("C", "W")),
        ):
            vehicles, queues = simulate_direction(
                entry_times=arrivals[stream_key],
                direction=direction,
                signal_plan=corridor_plan,
                speed_kmh=SPEED_KMH,
                headway_s=HEADWAY_S,
                horizon_s=HORIZON_S,
            )
            metrics = _corridor_metrics(vehicles, queues)
            corridor_records.append(
                {
                    "strategy": strategy_name,
                    "direction": direction,
                    **metrics,
                }
            )
            corridor_delay_total += float(metrics["total_delay_s"])
            corridor_vehicle_total += int(metrics["vehicles"])
            corridor_passage_total += int(metrics["junction_passages"])
            corridor_queue_total += int(metrics["queue_at_horizon_veh"])
            corridor_waiting_events += int(metrics["waiting_events"])

        ns_delay_total = 0.0
        ns_vehicle_total = 0
        ns_queue_total = 0
        ns_waiting_events = 0
        for junction in JUNCTIONS:
            for approach in ("N", "S"):
                metrics = simulate_ns_approach(
                    arrivals=arrivals[(junction, approach)],
                    plan=junction_plans[junction],
                    headway_s=HEADWAY_S,
                    horizon_s=HORIZON_S,
                )
                ns_records.append(
                    {
                        "strategy": strategy_name,
                        "junction": junction,
                        "approach": approach,
                        **metrics,
                    }
                )
                ns_delay_total += float(metrics["total_delay_s"])
                ns_vehicle_total += int(metrics["vehicles"])
                ns_queue_total += int(metrics["queue_at_horizon_veh"])
                ns_waiting_events += int(metrics["waiting_events"])

        total_delay = corridor_delay_total + ns_delay_total
        total_passages = corridor_passage_total + ns_vehicle_total
        network_records.append(
            {
                "strategy": strategy_name,
                "corridor_vehicles": corridor_vehicle_total,
                "corridor_junction_passages": corridor_passage_total,
                "corridor_total_delay_s": corridor_delay_total,
                "corridor_mean_delay_per_vehicle_s": (
                    corridor_delay_total / corridor_vehicle_total
                ),
                "ns_vehicles": ns_vehicle_total,
                "ns_junction_passages": ns_vehicle_total,
                "ns_total_delay_s": ns_delay_total,
                "ns_mean_delay_per_vehicle_s": ns_delay_total / ns_vehicle_total,
                "total_junction_passages": total_passages,
                "total_delay_s": total_delay,
                "mean_delay_per_junction_passage_s": total_delay
                / total_passages,
                "corridor_queue_at_horizon_veh": corridor_queue_total,
                "ns_queue_at_horizon_veh": ns_queue_total,
                "total_queue_at_horizon_veh": corridor_queue_total
                + ns_queue_total,
                "corridor_waiting_events": corridor_waiting_events,
                "ns_waiting_events": ns_waiting_events,
            }
        )

    corridor = pd.DataFrame(corridor_records)
    ns_results = pd.DataFrame(ns_records)
    network = pd.DataFrame(network_records)
    signal_windows = pd.DataFrame(signal_records)
    capacity = capacity_diagnostic(traffic, period_plans, start_time)
    if len(corridor) != 4:
        raise ValueError("Expected four corridor rows.")
    if len(ns_results) != 12:
        raise ValueError("Expected twelve north-south approach rows.")
    if len(network) != 2:
        raise ValueError("Expected two network rows.")
    if len(signal_windows) != 6:
        raise ValueError("Expected six signal-window rows.")
    if corridor.groupby("strategy")["vehicles"].sum().nunique() != 1:
        raise ValueError("Corridor strategies received different demand.")
    if ns_results.groupby("strategy")["vehicles"].sum().nunique() != 1:
        raise ValueError("North-south strategies received different demand.")

    corridor.to_csv(CORRIDOR_PATH, index=False)
    ns_results.to_csv(NS_PATH, index=False)
    network.to_csv(NETWORK_PATH, index=False)
    signal_windows.to_csv(SIGNAL_PATH, index=False)
    capacity.to_csv(CAPACITY_PATH, index=False)

    print("\nCorridor results:")
    print(corridor.to_string(index=False))
    print("\nNorth-south approach results:")
    print(ns_results.to_string(index=False))
    print("\nNetwork summary:")
    print(network.to_string(index=False))
    print("\nJunction C north-south capacity diagnostic:")
    print(capacity.to_string(index=False))
    return corridor, ns_results, network, signal_windows, capacity


if __name__ == "__main__":
    run_experiment()
