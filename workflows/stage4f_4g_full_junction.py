"""Phases 4F to 4J: full-junction simulation, green-split optimisation and audits.

Merged from the original phase runner scripts. Each section keeps the
functions of one phase; run a phase with ``python -m workflows <phase>`` from
the repository root. Outputs are written to outputs/data and outputs/figures.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "traffic-signals-matplotlib")
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from traffic_signals.full_junction import create_two_phase_plan, simulate_ns_approach
from traffic_signals.green_split_sensitivity import (
    modify_c_green,
    theoretical_capacity_vph,
)
from traffic_signals.hourly_arrivals import generate_hourly_arrivals
from traffic_signals.network_evaluation import build_period_plans, evaluate_network
from traffic_signals.vehicle_simulation import simulate_direction
from workflows.stage4a_4e_corridor import bootstrap_mean_ci


PHASE_DESCRIPTIONS = {
    "phase4f1": "Phase 4F-1 full-junction AM experiment",
    "phase4f2": "Phase 4F-2 multi-seed full junction",
    "phase4f3a": "Phase 4F-3A Junction C green-split sensitivity",
    "phase4f3b": "Phase 4F-3B joint split and offset optimisation",
    "phase4g": "Phase 4G integrated audit",
    "phase4h": "Phase 4H PM plan with fairness against Phase 3A",
    "phase4i": "Phase 4I AM plan with fairness against Phase 3A",
    "phase4j": "Phase 4J demand stress test of the Phase 4H plan",
}

# ============================================================================
# Phase 4F-1 full-junction AM experiment (python -m workflows phase4f1)
# Formerly phase4f1_full_junction.py. Compare Phase 3A and Phase 4E with local north-south traffic.
# ============================================================================

DATA_DIR = Path("outputs/data")
TRAFFIC_PATH = DATA_DIR / "synthetic_traffic_15min.csv"
PHASE3A_PATH = DATA_DIR / "phase3a_signal_plans.csv"
PHASE4E_PATH = DATA_DIR / "phase4e_selected_plans.csv"
CORRIDOR_PATH = DATA_DIR / "phase4f1_corridor_results.csv"
NS_PATH = DATA_DIR / "phase4f1_ns_approaches.csv"
NETWORK_PATH = DATA_DIR / "phase4f1_network_summary.csv"
SIGNAL_PATH = DATA_DIR / "phase4f1_signal_windows.csv"
PHASE4F1_CAPACITY_PATH = DATA_DIR / "phase4f1_c_ns_capacity.csv"
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


def load_inputs_phase4f1() -> tuple[
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


def run_experiment_phase4f1() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run the matched 08:00--09:00 Phase 3A/Phase 4E comparison."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    traffic, period_plans, selected, start_time = load_inputs_phase4f1()
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
    capacity.to_csv(PHASE4F1_CAPACITY_PATH, index=False)

    print("\nCorridor results:")
    print(corridor.to_string(index=False))
    print("\nNorth-south approach results:")
    print(ns_results.to_string(index=False))
    print("\nNetwork summary:")
    print(network.to_string(index=False))
    print("\nJunction C north-south capacity diagnostic:")
    print(capacity.to_string(index=False))
    return corridor, ns_results, network, signal_windows, capacity


def run_phase4f1() -> None:
    """Run this phase (the former script entry point)."""
    run_experiment_phase4f1()

# ============================================================================
# Phase 4F-2 multi-seed full junction (python -m workflows phase4f2)
# Formerly phase4f2_multi_seed_full_junction.py. Evaluate fixed Phase 3A and Phase 4E full-junction plans over fresh seeds.
# ============================================================================

FIGURE_DIR = Path("outputs/figures")
PHASE4F1_NETWORK_PATH = DATA_DIR / "phase4f1_network_summary.csv"
PHASE4F2_MOVEMENT_METRICS_PATH = DATA_DIR / "phase4f2_movement_metrics.csv"
PHASE4F2_NETWORK_METRICS_PATH = DATA_DIR / "phase4f2_network_metrics.csv"
PHASE4F2_PAIRED_MOVEMENTS_PATH = DATA_DIR / "phase4f2_paired_movements.csv"
PHASE4F2_MOVEMENT_SUMMARY_PATH = DATA_DIR / "phase4f2_movement_summary.csv"
PHASE4F2_PAIRED_NETWORK_PATH = DATA_DIR / "phase4f2_paired_network.csv"
PHASE4F2_NETWORK_SUMMARY_PATH = DATA_DIR / "phase4f2_network_summary.csv"
PHASE4F2_CAPACITY_PATH = DATA_DIR / "phase4f2_c_capacity.csv"
PHASE4F2_FIGURE_PATH = FIGURE_DIR / "phase4f2_network_delay.png"
PERIOD_HOURS = {
    "Overnight": 2,
    "AM peak": 8,
    "Interpeak": 12,
    "PM peak": 17,
}
PHASE4F2_SEEDS = range(200, 230)


def load_inputs_phase4f2() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and normalise synthetic demand and fixed timing plans."""

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
    plans_4e = pd.read_csv(PHASE4E_PATH)
    return traffic, plans_3a, plans_4e


def make_arrivals(
    traffic: pd.DataFrame,
    period: str,
    seed: int,
) -> dict[tuple[str, str], np.ndarray]:
    """Generate the eight matched streams for one representative hour."""

    if period not in PERIOD_HOURS:
        raise ValueError(f"Unknown period: {period}.")
    study_date = traffic["time"].dt.normalize().min()
    start_time = study_date + pd.Timedelta(hours=PERIOD_HOURS[period])
    return generate_hourly_arrivals(
        traffic=traffic,
        start_time=start_time,
        streams=STREAMS,
        seed=seed,
    )


def verify_phase4f1_regression(
    traffic: pd.DataFrame,
    plans_3a: pd.DataFrame,
    plans_4e: pd.DataFrame,
) -> None:
    """Reproduce the Phase 4F-1 AM seed-42 network results after refactoring."""

    previous = pd.read_csv(PHASE4F1_NETWORK_PATH).set_index("strategy")
    strategies = build_period_plans(plans_3a, plans_4e, "AM peak")
    arrivals = make_arrivals(traffic, "AM peak", seed=42)
    field_mapping = {
        "corridor_mean_delay_s": "corridor_mean_delay_per_vehicle_s",
        "ns_mean_delay_s": "ns_mean_delay_per_vehicle_s",
        "mean_delay_per_passage_s": "mean_delay_per_junction_passage_s",
        "total_delay_vehicle_seconds": "total_delay_s",
        "total_queue_at_hour": "total_queue_at_horizon_veh",
    }
    for strategy, junction_plans in strategies.items():
        _, current = evaluate_network(
            arrivals=arrivals,
            junction_plans=junction_plans,
            strategy=strategy,
            period="AM peak",
            seed=42,
            horizon_s=HORIZON_S,
        )
        for current_field, previous_field in field_mapping.items():
            if not np.isclose(
                float(current[current_field]),
                float(previous.loc[strategy, previous_field]),
                atol=1e-6,
            ):
                raise AssertionError(
                    f"Regression mismatch: {strategy}/{current_field}"
                )
    print("Phase 4F-1 regression passed.")


def run_experiment_phase4f2(
    traffic: pd.DataFrame,
    plans_3a: pd.DataFrame,
    plans_4e: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run four periods, 30 seeds, two strategies and eight movements."""

    movement_frames: list[pd.DataFrame] = []
    network_records: list[dict[str, object]] = []
    for period in PERIOD_HOURS:
        strategies = build_period_plans(plans_3a, plans_4e, period)
        for seed in PHASE4F2_SEEDS:
            arrivals = make_arrivals(traffic, period, seed)
            for strategy, junction_plans in strategies.items():
                movements, network = evaluate_network(
                    arrivals=arrivals,
                    junction_plans=junction_plans,
                    strategy=strategy,
                    period=period,
                    seed=seed,
                    horizon_s=HORIZON_S,
                )
                movement_frames.append(movements)
                network_records.append(network)
        print(f"Completed {period}")

    movement_results = pd.concat(movement_frames, ignore_index=True)
    network_results = pd.DataFrame(network_records)
    if len(movement_results) != 1920:
        raise ValueError("Expected 1,920 movement metric rows.")
    if len(network_results) != 240:
        raise ValueError("Expected 240 network metric rows.")
    movement_counts = movement_results.pivot(
        index=["period", "seed", "movement"],
        columns="strategy",
        values="vehicles",
    )
    if not (movement_counts["Phase3A"] == movement_counts["Phase4E"]).all():
        raise ValueError("Strategies received different movement demand.")
    movement_results.to_csv(PHASE4F2_MOVEMENT_METRICS_PATH, index=False)
    network_results.to_csv(PHASE4F2_NETWORK_METRICS_PATH, index=False)
    return movement_results, network_results


def summarise_movements_phase4f2(
    movements: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create paired movement comparisons and 32 bootstrap summaries."""

    keys = ["period", "seed", "movement"]
    baseline = movements.loc[movements["strategy"] == "Phase3A"].drop(
        columns="strategy"
    )
    optimised = movements.loc[movements["strategy"] == "Phase4E"].drop(
        columns="strategy"
    )
    paired = baseline.merge(
        optimised,
        on=keys,
        suffixes=("_3a", "_4e"),
        validate="one_to_one",
    )
    if len(paired) != 960:
        raise ValueError("Expected 960 paired movement rows.")
    if not (paired["vehicles_3a"] == paired["vehicles_4e"]).all():
        raise ValueError("Paired movement vehicle counts differ.")
    paired["delay_change_s"] = (
        paired["mean_delay_per_vehicle_s_4e"]
        - paired["mean_delay_per_vehicle_s_3a"]
    )
    paired["queue_change_veh"] = (
        paired["queue_at_hour_veh_4e"]
        - paired["queue_at_hour_veh_3a"]
    )

    records: list[dict[str, object]] = []
    for period in PERIOD_HOURS:
        for movement in ("EB", "WB", "A_N", "A_S", "B_N", "B_S", "C_N", "C_S"):
            group = paired.loc[
                (paired["period"] == period)
                & (paired["movement"] == movement)
            ]
            changes = group["delay_change_s"].to_numpy()
            low, high = bootstrap_mean_ci(changes)
            records.append(
                {
                    "period": period,
                    "movement": movement,
                    "category": group["category_3a"].iloc[0],
                    "replications": len(group),
                    "phase3a_mean_delay_s": group[
                        "mean_delay_per_vehicle_s_3a"
                    ].mean(),
                    "phase4e_mean_delay_s": group[
                        "mean_delay_per_vehicle_s_4e"
                    ].mean(),
                    "mean_delay_change_s": changes.mean(),
                    "ci_lower_s": low,
                    "ci_upper_s": high,
                    "seeds_better": int((changes < 0).sum()),
                    "phase3a_queue_at_hour": group[
                        "queue_at_hour_veh_3a"
                    ].mean(),
                    "phase4e_queue_at_hour": group[
                        "queue_at_hour_veh_4e"
                    ].mean(),
                    "mean_queue_change_veh": group[
                        "queue_change_veh"
                    ].mean(),
                }
            )
    summary = pd.DataFrame(records)
    if len(summary) != 32 or not summary["replications"].eq(30).all():
        raise ValueError("Expected 32 movement summaries with 30 replications.")
    paired.to_csv(PHASE4F2_PAIRED_MOVEMENTS_PATH, index=False)
    summary.to_csv(PHASE4F2_MOVEMENT_SUMMARY_PATH, index=False)
    return paired, summary


def summarise_network_phase4f2(
    network: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create paired network differences and four period summaries."""

    keys = ["period", "seed"]
    baseline = network.loc[network["strategy"] == "Phase3A"].drop(
        columns="strategy"
    )
    optimised = network.loc[network["strategy"] == "Phase4E"].drop(
        columns="strategy"
    )
    paired = baseline.merge(
        optimised,
        on=keys,
        suffixes=("_3a", "_4e"),
        validate="one_to_one",
    )
    if len(paired) != 120:
        raise ValueError("Expected 120 paired network rows.")
    paired["delay_change_s"] = (
        paired["mean_delay_per_passage_s_4e"]
        - paired["mean_delay_per_passage_s_3a"]
    )
    paired["queue_change_veh"] = (
        paired["total_queue_at_hour_4e"]
        - paired["total_queue_at_hour_3a"]
    )

    records: list[dict[str, object]] = []
    for period in PERIOD_HOURS:
        group = paired.loc[paired["period"] == period]
        changes = group["delay_change_s"].to_numpy()
        low, high = bootstrap_mean_ci(changes)
        records.append(
            {
                "period": period,
                "replications": len(group),
                "phase3a_delay_per_passage_s": group[
                    "mean_delay_per_passage_s_3a"
                ].mean(),
                "phase4e_delay_per_passage_s": group[
                    "mean_delay_per_passage_s_4e"
                ].mean(),
                "mean_delay_change_s": changes.mean(),
                "ci_lower_s": low,
                "ci_upper_s": high,
                "seeds_better": int((changes < 0).sum()),
                "corridor_delay_change_s": (
                    group["corridor_mean_delay_s_4e"]
                    - group["corridor_mean_delay_s_3a"]
                ).mean(),
                "ns_delay_change_s": (
                    group["ns_mean_delay_s_4e"]
                    - group["ns_mean_delay_s_3a"]
                ).mean(),
                "phase3a_queue_at_hour": group[
                    "total_queue_at_hour_3a"
                ].mean(),
                "phase4e_queue_at_hour": group[
                    "total_queue_at_hour_4e"
                ].mean(),
                "mean_queue_change_veh": group["queue_change_veh"].mean(),
            }
        )
    summary = pd.DataFrame(records)
    if len(summary) != 4 or not summary["replications"].eq(30).all():
        raise ValueError("Expected four network summaries with 30 replications.")
    paired.to_csv(PHASE4F2_PAIRED_NETWORK_PATH, index=False)
    summary.to_csv(PHASE4F2_NETWORK_SUMMARY_PATH, index=False)
    return paired, summary


def diagnose_c_capacity(
    traffic: pd.DataFrame,
    plans_3a: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate Junction C critical N/S ratios in each representative hour."""

    records: list[dict[str, object]] = []
    study_date = traffic["time"].dt.normalize().min()
    for period, hour in PERIOD_HOURS.items():
        rows = plans_3a.loc[
            (plans_3a["period"] == period)
            & (plans_3a["intersection_id"] == "C")
        ]
        if len(rows) != 1:
            raise ValueError(f"Missing Junction C timing plan for {period}.")
        plan = rows.iloc[0]
        capacity_ratio = float(
            plan["ns_green_s"] / plan["common_cycle_s"]
        )
        start = study_date + pd.Timedelta(hours=hour)
        for timestamp in pd.date_range(start, periods=4, freq="15min"):
            counts = traffic.loc[
                (traffic["time"] == timestamp)
                & (traffic["intersection_id"] == "C")
                & (traffic["approach"].isin(["N", "S"]))
            ]
            if len(counts) != 2 or set(counts["approach"]) != {"N", "S"}:
                raise ValueError("Expected N and S demand at Junction C.")
            counts = counts.set_index("approach")
            volumes = pd.to_numeric(counts["volume_15min"], errors="raise")
            saturation = pd.to_numeric(
                counts["saturation_flow_vph"], errors="raise"
            )
            critical_approach = str(volumes.idxmax())
            critical_count = float(volumes.loc[critical_approach])
            saturation_flow = float(saturation.loc[critical_approach])
            critical_flow = critical_count * 4.0
            critical_ratio = critical_flow / saturation_flow
            records.append(
                {
                    "period": period,
                    "time": timestamp,
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
    if len(result) != 16:
        raise ValueError("Expected 16 Junction C capacity rows.")
    result.to_csv(PHASE4F2_CAPACITY_PATH, index=False)
    return result


def plot_network_results(summary: pd.DataFrame) -> None:
    """Plot paired synthetic network-delay changes and bootstrap intervals."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ordered = summary.set_index("period").loc[list(PERIOD_HOURS)]
    x = np.arange(len(ordered))
    y = ordered["mean_delay_change_s"].to_numpy()
    errors = np.vstack(
        [
            y - ordered["ci_lower_s"].to_numpy(),
            ordered["ci_upper_s"].to_numpy() - y,
        ]
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(x, y, yerr=errors, marker="o", capsize=5, linestyle="none")
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered.index)
    ax.set_title("Phase 4F-2: Synthetic Network Delay Change Across 30 Seeds")
    ax.set_ylabel("Phase 4E minus Phase 3A delay per passage (s)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PHASE4F2_FIGURE_PATH, dpi=180)
    plt.close(fig)


def run_phase4f2() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run regression, repeated evaluation, summaries, diagnostics and plot."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    traffic, plans_3a, plans_4e = load_inputs_phase4f2()
    verify_phase4f1_regression(traffic, plans_3a, plans_4e)
    movements, network = run_experiment_phase4f2(traffic, plans_3a, plans_4e)
    _, movement_summary = summarise_movements_phase4f2(movements)
    _, network_summary = summarise_network_phase4f2(network)
    capacity = diagnose_c_capacity(traffic, plans_3a)
    plot_network_results(network_summary)
    print("\nNetwork comparison:")
    print(network_summary.to_string(index=False))
    print("\nJunction C movement comparison:")
    print(
        movement_summary.loc[
            movement_summary["movement"].isin(["C_N", "C_S"])
        ].to_string(index=False)
    )
    print("\nCapacity warnings:")
    print(capacity.loc[capacity["exceeds_green_capacity"]].to_string(index=False))
    return network_summary, movement_summary, capacity

# ============================================================================
# Phase 4F-3A Junction C green-split sensitivity (python -m workflows phase4f3a)
# Formerly phase4f3a_green_split_sensitivity.py. Run the Phase 4F-3A Junction C green-split sensitivity experiment.
# ============================================================================

PHASE4F3A_PERIODS = ("AM peak", "PM peak")
EXTRA_GREEN = (0, 3, 5, 7, 10)
PHASE4F3A_SEEDS = range(300, 312)
PHASE4F3A_MOVEMENT_METRICS_PATH = DATA_DIR / "phase4f3a_movement_metrics.csv"
PHASE4F3A_NETWORK_METRICS_PATH = DATA_DIR / "phase4f3a_network_metrics.csv"
PHASE4F3A_CAPACITY_PATH = DATA_DIR / "phase4f3a_capacity_scenarios.csv"
PHASE4F3A_PAIRED_NETWORK_PATH = DATA_DIR / "phase4f3a_paired_network.csv"
PHASE4F3A_NETWORK_SUMMARY_PATH = DATA_DIR / "phase4f3a_network_summary.csv"
PHASE4F3A_PAIRED_MOVEMENTS_PATH = DATA_DIR / "phase4f3a_paired_movements.csv"
PHASE4F3A_MOVEMENT_SUMMARY_PATH = DATA_DIR / "phase4f3a_movement_summary.csv"


def run_experiment_phase4f3a() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run 120 matched full-junction simulations and save raw metrics."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    traffic, plans_3a, plans_4e = load_inputs_phase4f2()
    movement_frames: list[pd.DataFrame] = []
    network_records: list[dict[str, object]] = []
    capacity_records: list[dict[str, object]] = []

    for period in PHASE4F3A_PERIODS:
        base_plans = build_period_plans(plans_3a, plans_4e, period)["Phase4E"]
        for extra in EXTRA_GREEN:
            plans = modify_c_green(base_plans, extra)
            c_plan = plans["C"]
            capacity_records.append(
                {
                    "period": period,
                    "extra_ns_green_s": extra,
                    "cycle_s": c_plan["cycle_s"],
                    "lost_time_s": c_plan["lost_time_s"],
                    "C_ew_green_s": c_plan["ew_green_s"],
                    "C_ns_green_s": c_plan["ns_green_s"],
                    "C_ew_offset_s": c_plan["ew_offset_s"],
                    "C_ns_offset_s": c_plan["ns_offset_s"],
                    "C_ew_capacity_vph": theoretical_capacity_vph(
                        c_plan["ew_green_s"], c_plan["cycle_s"]
                    ),
                    "C_ns_capacity_vph": theoretical_capacity_vph(
                        c_plan["ns_green_s"], c_plan["cycle_s"]
                    ),
                }
            )

        for seed in PHASE4F3A_SEEDS:
            arrivals = make_arrivals(traffic, period, seed)
            for extra in EXTRA_GREEN:
                plans = modify_c_green(base_plans, extra)
                strategy = f"Phase4E_C_NS_plus_{extra}s"
                movement, network = evaluate_network(
                    arrivals=arrivals,
                    junction_plans=plans,
                    strategy=strategy,
                    period=period,
                    seed=seed,
                    horizon_s=HORIZON_S,
                )
                movement.insert(3, "extra_ns_green_s", extra)
                network["extra_ns_green_s"] = extra
                movement_frames.append(movement)
                network_records.append(network)
        print(f"Completed {period}")

    movements = pd.concat(movement_frames, ignore_index=True)
    network = pd.DataFrame(network_records)
    capacities = pd.DataFrame(capacity_records)
    if len(movements) != 960:
        raise ValueError("Expected 960 movement metric rows.")
    if len(network) != 120:
        raise ValueError("Expected 120 network metric rows.")
    if len(capacities) != 10:
        raise ValueError("Expected 10 capacity scenarios.")

    matched_counts = movements.groupby(
        ["period", "seed", "movement"]
    )["vehicles"].nunique()
    if not matched_counts.eq(1).all():
        raise ValueError("Green scenarios received different vehicle counts.")

    c_queues = (
        movements.loc[movements["movement"].isin(["C_N", "C_S"])]
        .groupby(["period", "seed", "extra_ns_green_s"], as_index=False)
        .agg(C_ns_queue_at_hour=("queue_at_hour_veh", "sum"))
    )
    network = network.merge(
        c_queues,
        on=["period", "seed", "extra_ns_green_s"],
        validate="one_to_one",
    )

    movements.to_csv(PHASE4F3A_MOVEMENT_METRICS_PATH, index=False)
    network.to_csv(PHASE4F3A_NETWORK_METRICS_PATH, index=False)
    capacities.to_csv(PHASE4F3A_CAPACITY_PATH, index=False)
    return movements, network, capacities


def summarise_network_phase4f3a(
    network: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair every green scenario with the unchanged Phase 4E baseline."""

    baseline = network.loc[network["extra_ns_green_s"] == 0].drop(
        columns=["strategy", "extra_ns_green_s"]
    )
    candidates = network.drop(columns="strategy")
    paired = candidates.merge(
        baseline,
        on=["period", "seed"],
        suffixes=("_candidate", "_baseline"),
        validate="many_to_one",
    )
    if len(paired) != 120:
        raise ValueError("Expected 120 paired network rows.")
    for count_field in ("corridor_vehicles", "ns_vehicles", "total_passages"):
        if not paired[f"{count_field}_candidate"].eq(
            paired[f"{count_field}_baseline"]
        ).all():
            raise ValueError(f"Paired {count_field} values differ.")

    paired["delay_change_s"] = (
        paired["mean_delay_per_passage_s_candidate"]
        - paired["mean_delay_per_passage_s_baseline"]
    )
    paired["queue_change_veh"] = (
        paired["total_queue_at_hour_candidate"]
        - paired["total_queue_at_hour_baseline"]
    )
    paired["C_ns_queue_change_veh"] = (
        paired["C_ns_queue_at_hour_candidate"]
        - paired["C_ns_queue_at_hour_baseline"]
    )
    paired["corridor_delay_change_s"] = (
        paired["corridor_mean_delay_s_candidate"]
        - paired["corridor_mean_delay_s_baseline"]
    )
    paired["ns_delay_change_s"] = (
        paired["ns_mean_delay_s_candidate"]
        - paired["ns_mean_delay_s_baseline"]
    )

    records: list[dict[str, object]] = []
    for period in PHASE4F3A_PERIODS:
        for extra in EXTRA_GREEN:
            group = paired.loc[
                (paired["period"] == period)
                & (paired["extra_ns_green_s"] == extra)
            ]
            delay_changes = group["delay_change_s"].to_numpy()
            queue_changes = group["queue_change_veh"].to_numpy()
            c_queue_changes = group["C_ns_queue_change_veh"].to_numpy()
            delay_low, delay_high = bootstrap_mean_ci(delay_changes)
            queue_low, queue_high = bootstrap_mean_ci(queue_changes)
            c_queue_low, c_queue_high = bootstrap_mean_ci(c_queue_changes)
            records.append(
                {
                    "period": period,
                    "extra_ns_green_s": extra,
                    "replications": len(group),
                    "baseline_delay_per_passage_s": group[
                        "mean_delay_per_passage_s_baseline"
                    ].mean(),
                    "candidate_delay_per_passage_s": group[
                        "mean_delay_per_passage_s_candidate"
                    ].mean(),
                    "mean_delay_change_s": delay_changes.mean(),
                    "ci_lower_s": delay_low,
                    "ci_upper_s": delay_high,
                    "seeds_better": int((delay_changes < 0).sum()),
                    "corridor_delay_change_s": group[
                        "corridor_delay_change_s"
                    ].mean(),
                    "ns_delay_change_s": group["ns_delay_change_s"].mean(),
                    "baseline_queue_at_hour": group[
                        "total_queue_at_hour_baseline"
                    ].mean(),
                    "candidate_queue_at_hour": group[
                        "total_queue_at_hour_candidate"
                    ].mean(),
                    "mean_queue_change_veh": queue_changes.mean(),
                    "queue_ci_lower_veh": queue_low,
                    "queue_ci_upper_veh": queue_high,
                    "baseline_C_ns_queue_at_hour": group[
                        "C_ns_queue_at_hour_baseline"
                    ].mean(),
                    "candidate_C_ns_queue_at_hour": group[
                        "C_ns_queue_at_hour_candidate"
                    ].mean(),
                    "mean_C_ns_queue_change_veh": c_queue_changes.mean(),
                    "C_ns_queue_ci_lower_veh": c_queue_low,
                    "C_ns_queue_ci_upper_veh": c_queue_high,
                }
            )

    summary = pd.DataFrame(records)
    if len(summary) != 10 or not summary["replications"].eq(12).all():
        raise ValueError("Expected 10 network summaries with 12 replications.")
    paired.to_csv(PHASE4F3A_PAIRED_NETWORK_PATH, index=False)
    summary.to_csv(PHASE4F3A_NETWORK_SUMMARY_PATH, index=False)
    return paired, summary


def summarise_movements_phase4f3a(
    movements: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair movement outcomes with the zero-transfer Phase 4E baseline."""

    baseline = movements.loc[movements["extra_ns_green_s"] == 0].drop(
        columns=["strategy", "extra_ns_green_s"]
    )
    candidates = movements.drop(columns="strategy")
    paired = candidates.merge(
        baseline,
        on=["period", "seed", "movement"],
        suffixes=("_candidate", "_baseline"),
        validate="many_to_one",
    )
    if len(paired) != 960:
        raise ValueError("Expected 960 paired movement rows.")
    if not paired["vehicles_candidate"].eq(paired["vehicles_baseline"]).all():
        raise ValueError("Paired movement vehicle counts differ.")
    paired["delay_change_s"] = (
        paired["mean_delay_per_vehicle_s_candidate"]
        - paired["mean_delay_per_vehicle_s_baseline"]
    )
    paired["queue_change_veh"] = (
        paired["queue_at_hour_veh_candidate"]
        - paired["queue_at_hour_veh_baseline"]
    )

    records: list[dict[str, object]] = []
    movements_order = ("EB", "WB", "A_N", "A_S", "B_N", "B_S", "C_N", "C_S")
    for period in PHASE4F3A_PERIODS:
        for extra in EXTRA_GREEN:
            for movement in movements_order:
                group = paired.loc[
                    (paired["period"] == period)
                    & (paired["extra_ns_green_s"] == extra)
                    & (paired["movement"] == movement)
                ]
                delay_changes = group["delay_change_s"].to_numpy()
                queue_changes = group["queue_change_veh"].to_numpy()
                delay_low, delay_high = bootstrap_mean_ci(delay_changes)
                queue_low, queue_high = bootstrap_mean_ci(queue_changes)
                records.append(
                    {
                        "period": period,
                        "extra_ns_green_s": extra,
                        "movement": movement,
                        "category": group["category_candidate"].iloc[0],
                        "replications": len(group),
                        "baseline_mean_delay_s": group[
                            "mean_delay_per_vehicle_s_baseline"
                        ].mean(),
                        "candidate_mean_delay_s": group[
                            "mean_delay_per_vehicle_s_candidate"
                        ].mean(),
                        "mean_delay_change_s": delay_changes.mean(),
                        "ci_lower_s": delay_low,
                        "ci_upper_s": delay_high,
                        "seeds_better": int((delay_changes < 0).sum()),
                        "baseline_queue_at_hour": group[
                            "queue_at_hour_veh_baseline"
                        ].mean(),
                        "candidate_queue_at_hour": group[
                            "queue_at_hour_veh_candidate"
                        ].mean(),
                        "mean_queue_change_veh": queue_changes.mean(),
                        "queue_ci_lower_veh": queue_low,
                        "queue_ci_upper_veh": queue_high,
                    }
                )

    summary = pd.DataFrame(records)
    if len(summary) != 80 or not summary["replications"].eq(12).all():
        raise ValueError("Expected 80 movement summaries with 12 replications.")
    paired.to_csv(PHASE4F3A_PAIRED_MOVEMENTS_PATH, index=False)
    summary.to_csv(PHASE4F3A_MOVEMENT_SUMMARY_PATH, index=False)
    return paired, summary


def _period_slug(period: str) -> str:
    return period.lower().replace(" ", "_")


def plot_results(network_summary: pd.DataFrame) -> None:
    """Save delay and Junction C residual-queue sensitivity figures."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for period in PHASE4F3A_PERIODS:
        selected = network_summary.loc[
            network_summary["period"] == period
        ].sort_values("extra_ns_green_s")
        x = selected["extra_ns_green_s"].to_numpy(dtype=float)

        delay = selected["mean_delay_change_s"].to_numpy(dtype=float)
        delay_low = selected["ci_lower_s"].to_numpy(dtype=float)
        delay_high = selected["ci_upper_s"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, delay, marker="o")
        ax.fill_between(x, delay_low, delay_high, alpha=0.2)
        ax.axhline(0, linestyle="--", linewidth=1, color="black")
        ax.set_title(f"Synthetic Junction C Green-Split Sensitivity: {period}")
        ax.set_xlabel("Additional NS effective green at C (s)")
        ax.set_ylabel("Change in network delay per passage (s)")
        ax.set_xticks(EXTRA_GREEN)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / f"phase4f3a_network_delay_{_period_slug(period)}.png",
            dpi=180,
        )
        plt.close(fig)

        queue = selected["mean_C_ns_queue_change_veh"].to_numpy(dtype=float)
        queue_low = selected["C_ns_queue_ci_lower_veh"].to_numpy(dtype=float)
        queue_high = selected["C_ns_queue_ci_upper_veh"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, queue, marker="o", color="tab:red")
        ax.fill_between(x, queue_low, queue_high, alpha=0.2, color="tab:red")
        ax.axhline(0, linestyle="--", linewidth=1, color="black")
        ax.set_title(f"Synthetic Junction C NS Residual Queue: {period}")
        ax.set_xlabel("Additional NS effective green at C (s)")
        ax.set_ylabel("Change in C north-south queue at hour end (vehicles)")
        ax.set_xticks(EXTRA_GREEN)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / f"phase4f3a_c_ns_queue_{_period_slug(period)}.png",
            dpi=180,
        )
        plt.close(fig)


def run_phase4f3a() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the sensitivity experiment, save summaries and print key results."""

    movements, network, capacities = run_experiment_phase4f3a()
    _, network_summary = summarise_network_phase4f3a(network)
    _, movement_summary = summarise_movements_phase4f3a(movements)
    plot_results(network_summary)
    print("\nTheoretical approach capacities:")
    print(capacities.to_string(index=False))
    print("\nNetwork sensitivity:")
    print(network_summary.to_string(index=False))
    print("\nKey movement-level results:")
    print(
        movement_summary.loc[
            movement_summary["movement"].isin(["EB", "WB", "C_N", "C_S"])
        ].to_string(index=False)
    )
    print("\nPhase 4F-3A completed.")
    return network_summary, movement_summary, capacities

# ============================================================================
# Phase 4F-3B joint split and offset optimisation (python -m workflows phase4f3b)
# Formerly phase4f3b_joint_optimisation.py. Joint Junction C green-split and offset optimisation for Phase 4F-3B.
# ============================================================================

PHASE4F3B_PERIODS = ("AM peak", "PM peak")
GREEN_TRANSFERS = (0, 3, 5, 7, 10)
OFFSET_SHIFTS = (-12, -8, -4, 0, 4, 8, 12)
SCREEN_SEEDS = range(400, 404)
TRAIN_SEEDS = range(400, 412)
HOLDOUT_SEEDS = range(420, 450)
PROTECTED_MOVEMENTS = ("EB", "WB", "C_N", "C_S")
UNCHANGED_MOVEMENTS = ("A_N", "A_S", "B_N", "B_S")
ALL_MOVEMENTS = ("EB", "WB", "A_N", "A_S", "B_N", "B_S", "C_N", "C_S")
FAIRNESS_LIMIT_S = 2.0
SCREENING_PATH = DATA_DIR / "phase4f3b_screening.csv"
TRAINING_CANDIDATES_PATH = DATA_DIR / "phase4f3b_training_candidates.csv"
TRAINING_FAIRNESS_PATH = DATA_DIR / "phase4f3b_training_fairness.csv"
SELECTED_PLANS_PATH = DATA_DIR / "phase4f3b_selected_plans.csv"
HOLDOUT_MOVEMENTS_PATH = DATA_DIR / "phase4f3b_holdout_movements.csv"
HOLDOUT_NETWORK_PATH = DATA_DIR / "phase4f3b_holdout_network.csv"
HOLDOUT_NETWORK_SUMMARY_PATH = DATA_DIR / "phase4f3b_holdout_network_summary.csv"
HOLDOUT_MOVEMENT_SUMMARY_PATH = DATA_DIR / "phase4f3b_holdout_movement_summary.csv"


def make_candidate_plan(
    base_plans: Mapping[str, Mapping[str, float]],
    extra_ns: int,
    c_offset: float,
) -> dict[str, dict[str, float]]:
    """Change only Junction C's green split and EW offset."""

    if set(base_plans) != {"A", "B", "C"}:
        raise ValueError("Expected junctions A, B and C.")
    if extra_ns not in GREEN_TRANSFERS:
        raise ValueError("Unsupported NS green transfer.")
    if not np.isfinite(c_offset):
        raise ValueError("Junction C offset must be finite.")

    plans = {junction: dict(plan) for junction, plan in base_plans.items()}
    original = plans["C"]
    ew_green = float(original["ew_green_s"]) - extra_ns
    ns_green = float(original["ns_green_s"]) + extra_ns
    if ew_green < 10 or ns_green < 10:
        raise ValueError("Minimum effective green violated.")
    plans["C"] = create_two_phase_plan(
        cycle_s=float(original["cycle_s"]),
        ew_green_s=ew_green,
        ns_green_s=ns_green,
        ew_offset_s=float(c_offset),
        lost_time_s=float(original["lost_time_s"]),
        interphase_loss_s=float(original["interphase_loss_s"]),
    )
    return plans


def make_candidate_pairs(
    base_c_offset: int,
    cycle: int = 150,
) -> list[tuple[int, int]]:
    """Return the 35 green-transfer and local offset combinations."""

    if not isinstance(base_c_offset, (int, np.integer)):
        raise ValueError("Base offset must be an integer.")
    if not isinstance(cycle, (int, np.integer)) or cycle <= 0:
        raise ValueError("Cycle must be a positive integer.")
    pairs = [
        (extra, int((base_c_offset + shift) % cycle))
        for extra in GREEN_TRANSFERS
        for shift in OFFSET_SHIFTS
    ]
    if len(pairs) != 35 or len(set(pairs)) != 35:
        raise ValueError("Expected 35 unique candidate pairs.")
    return pairs


def prepare_arrivals(
    traffic: pd.DataFrame,
    period: str,
    seeds: Iterable[int],
) -> dict[int, dict[tuple[str, str], np.ndarray]]:
    """Generate each matched eight-stream arrival set once."""

    return {int(seed): make_arrivals(traffic, period, int(seed)) for seed in seeds}


def evaluate_plan(
    arrival_cache: Mapping[int, Mapping[tuple[str, str], np.ndarray]],
    base_plans: Mapping[str, Mapping[str, float]],
    period: str,
    pair: tuple[int, int],
    name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one candidate against every cached arrival replication."""

    extra, c_offset = pair
    plans = make_candidate_plan(base_plans, extra, c_offset)
    movement_frames: list[pd.DataFrame] = []
    network_records: list[dict[str, object]] = []
    for seed, arrivals in arrival_cache.items():
        movements, network = evaluate_network(
            arrivals=arrivals,
            junction_plans=plans,
            strategy=name,
            period=period,
            seed=seed,
            horizon_s=HORIZON_S,
        )
        movements["extra_ns_green_s"] = extra
        movements["C_offset_s"] = c_offset
        network["extra_ns_green_s"] = extra
        network["C_offset_s"] = c_offset
        network["C_ns_queue_at_hour"] = int(
            movements.loc[
                movements["movement"].isin(["C_N", "C_S"]),
                "queue_at_hour_veh",
            ].sum()
        )
        movement_frames.append(movements)
        network_records.append(network)
    return pd.concat(movement_frames, ignore_index=True), pd.DataFrame(network_records)


def compare_plans(
    base_mov: pd.DataFrame,
    base_net: pd.DataFrame,
    new_mov: pd.DataFrame,
    new_net: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair candidate metrics with a benchmark receiving identical demand."""

    paired_mov = base_mov.merge(
        new_mov,
        on=["period", "seed", "movement"],
        suffixes=("_base", "_new"),
        validate="one_to_one",
    )
    if not paired_mov["vehicles_base"].eq(paired_mov["vehicles_new"]).all():
        raise ValueError("Movement vehicle counts differ between plans.")
    paired_mov["delay_change_s"] = (
        paired_mov["mean_delay_per_vehicle_s_new"]
        - paired_mov["mean_delay_per_vehicle_s_base"]
    )
    paired_mov["queue_change_veh"] = (
        paired_mov["queue_at_hour_veh_new"]
        - paired_mov["queue_at_hour_veh_base"]
    )

    paired_net = base_net.merge(
        new_net,
        on=["period", "seed"],
        suffixes=("_base", "_new"),
        validate="one_to_one",
    )
    for field in ("corridor_vehicles", "ns_vehicles", "total_passages"):
        if not paired_net[f"{field}_base"].eq(paired_net[f"{field}_new"]).all():
            raise ValueError(f"Network {field} differs between plans.")
    paired_net["delay_change_s"] = (
        paired_net["mean_delay_per_passage_s_new"]
        - paired_net["mean_delay_per_passage_s_base"]
    )
    paired_net["queue_change_veh"] = (
        paired_net["total_queue_at_hour_new"]
        - paired_net["total_queue_at_hour_base"]
    )
    paired_net["C_ns_queue_change_veh"] = (
        paired_net["C_ns_queue_at_hour_new"]
        - paired_net["C_ns_queue_at_hour_base"]
    )
    return paired_mov, paired_net


def evaluate_fairness(paired_mov: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the four movement-level training fairness constraints."""

    records: list[dict[str, object]] = []
    for movement in PROTECTED_MOVEMENTS:
        selected = paired_mov.loc[paired_mov["movement"] == movement]
        changes = selected["delay_change_s"].to_numpy(dtype=float)
        if len(changes) < 2:
            raise ValueError(f"Insufficient fairness samples for {movement}.")
        low, high = bootstrap_mean_ci(
            changes,
            n_bootstrap=4000,
            seed=2026,
        )
        records.append(
            {
                "movement": movement,
                "replications": len(changes),
                "mean_delay_change_s": changes.mean(),
                "ci_lower_s": low,
                "ci_upper_s": high,
                "fairness_pass": bool(high <= FAIRNESS_LIMIT_S),
            }
        )
    return pd.DataFrame(records)


def _baseline_pair(base_plans: Mapping[str, Mapping[str, float]]) -> tuple[int, int]:
    return 0, int(base_plans["C"]["ew_offset_s"])


def screen_candidates(
    cache: Mapping[int, Mapping[tuple[str, str], np.ndarray]],
    base_plans: Mapping[str, Mapping[str, float]],
    period: str,
) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """Rank all 35 pairs on four seeds and retain a diverse shortlist."""

    baseline_pair = _baseline_pair(base_plans)
    baseline_mov, baseline_net = evaluate_plan(
        cache, base_plans, period, baseline_pair, "Phase4E"
    )
    records: list[dict[str, object]] = []
    for pair in make_candidate_pairs(
        baseline_pair[1], int(base_plans["C"]["cycle_s"])
    ):
        candidate_mov, candidate_net = evaluate_plan(
            cache, base_plans, period, pair, "Candidate"
        )
        _, paired_net = compare_plans(
            baseline_mov, baseline_net, candidate_mov, candidate_net
        )
        records.append(
            {
                "period": period,
                "extra_ns_green_s": pair[0],
                "C_offset_s": pair[1],
                "replications": len(paired_net),
                "screen_delay_change_s": paired_net["delay_change_s"].mean(),
                "screen_C_ns_queue_change_veh": paired_net[
                    "C_ns_queue_change_veh"
                ].mean(),
            }
        )

    ranking = pd.DataFrame(records).sort_values(
        ["screen_delay_change_s", "screen_C_ns_queue_change_veh", "C_offset_s"],
        kind="stable",
    )
    if len(ranking) != 35:
        raise ValueError("Expected 35 screening rows per period.")
    chosen = {
        (int(row.extra_ns_green_s), int(row.C_offset_s))
        for row in ranking.head(12).itertuples()
    }
    for _, group in ranking.groupby("extra_ns_green_s", sort=False):
        chosen.update(
            (int(row.extra_ns_green_s), int(row.C_offset_s))
            for row in group.head(2).itertuples()
        )
    chosen.add(baseline_pair)
    shortlist = sorted(chosen)
    return ranking.reset_index(drop=True), shortlist


def _assert_unchanged_approaches(paired_mov: pd.DataFrame) -> None:
    unchanged = paired_mov.loc[
        paired_mov["movement"].isin(UNCHANGED_MOVEMENTS)
    ]
    if not np.allclose(unchanged["delay_change_s"], 0.0, atol=1e-12):
        raise AssertionError("A/B local approach delay changed unexpectedly.")
    if not unchanged["queue_change_veh"].eq(0).all():
        raise AssertionError("A/B local approach queues changed unexpectedly.")


def optimise_training(
    cache: Mapping[int, Mapping[tuple[str, str], np.ndarray]],
    base_plans: Mapping[str, Mapping[str, float]],
    period: str,
    shortlist: Iterable[tuple[int, int]],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Apply network and movement fairness criteria on all training seeds."""

    baseline_pair = _baseline_pair(base_plans)
    baseline_mov, baseline_net = evaluate_plan(
        cache, base_plans, period, baseline_pair, "Phase4E"
    )
    original_offset = baseline_pair[1]
    cycle = int(base_plans["C"]["cycle_s"])
    candidate_records: list[dict[str, object]] = []
    fairness_frames: list[pd.DataFrame] = []

    for extra, offset in shortlist:
        candidate_mov, candidate_net = evaluate_plan(
            cache, base_plans, period, (extra, offset), "Candidate"
        )
        paired_mov, paired_net = compare_plans(
            baseline_mov, baseline_net, candidate_mov, candidate_net
        )
        _assert_unchanged_approaches(paired_mov)
        fairness = evaluate_fairness(paired_mov)
        fairness.insert(0, "period", period)
        fairness.insert(1, "extra_ns_green_s", extra)
        fairness.insert(2, "C_offset_s", offset)
        fairness_frames.append(fairness)

        delay_changes = paired_net["delay_change_s"].to_numpy(dtype=float)
        delay_low, delay_high = bootstrap_mean_ci(delay_changes)
        mean_delay_change = float(delay_changes.mean())
        fairness_pass = bool(fairness["fairness_pass"].all())
        network_pass = bool(mean_delay_change <= 1e-12)
        candidate_records.append(
            {
                "period": period,
                "extra_ns_green_s": extra,
                "C_offset_s": offset,
                "replications": len(paired_net),
                "mean_network_delay_change_s": mean_delay_change,
                "network_delay_ci_lower_s": delay_low,
                "network_delay_ci_upper_s": delay_high,
                "mean_C_ns_queue_change_veh": paired_net[
                    "C_ns_queue_change_veh"
                ].mean(),
                "network_performance_pass": network_pass,
                "movement_fairness_pass": fairness_pass,
                "feasible": bool(network_pass and fairness_pass),
                "offset_distance_s": min(
                    (offset - original_offset) % cycle,
                    (original_offset - offset) % cycle,
                ),
            }
        )

    candidates = pd.DataFrame(candidate_records)
    fairness_checks = pd.concat(fairness_frames, ignore_index=True)
    feasible = candidates.loc[candidates["feasible"]].sort_values(
        [
            "mean_network_delay_change_s",
            "mean_C_ns_queue_change_veh",
            "offset_distance_s",
        ],
        kind="stable",
    )
    if feasible.empty:
        raise AssertionError("Phase 4E fallback should always be feasible.")
    winner = feasible.iloc[0]
    winning_plan = make_candidate_plan(
        base_plans,
        int(winner["extra_ns_green_s"]),
        float(winner["C_offset_s"]),
    )["C"]
    selected: dict[str, object] = {
        "period": period,
        "extra_ns_green_s": int(winner["extra_ns_green_s"]),
        "C_ew_offset_s": int(winner["C_offset_s"]),
        "C_ns_offset_s": winning_plan["ns_offset_s"],
        "C_ew_green_s": winning_plan["ew_green_s"],
        "C_ns_green_s": winning_plan["ns_green_s"],
        "training_mean_network_delay_change_s": winner[
            "mean_network_delay_change_s"
        ],
        "training_mean_C_ns_queue_change_veh": winner[
            "mean_C_ns_queue_change_veh"
        ],
        "training_network_performance_pass": bool(
            winner["network_performance_pass"]
        ),
        "training_movement_fairness_pass": bool(winner["movement_fairness_pass"]),
        "training_candidate_count": len(candidates),
    }
    return selected, candidates, fairness_checks


def _assert_regression_row(
    current: pd.Series,
    previous: pd.Series,
    label: str,
) -> None:
    fields = (
        "corridor_vehicles",
        "ns_vehicles",
        "total_passages",
        "corridor_mean_delay_s",
        "ns_mean_delay_s",
        "mean_delay_per_passage_s",
        "total_delay_vehicle_seconds",
        "corridor_queue_at_hour",
        "ns_queue_at_hour",
        "total_queue_at_hour",
    )
    for field in fields:
        if not np.isclose(float(current[field]), float(previous[field]), atol=1e-9):
            raise AssertionError(f"Regression mismatch for {label}/{field}.")


def verify_previous_results(
    traffic: pd.DataFrame,
    plans_3a: pd.DataFrame,
    plans_4e: pd.DataFrame,
) -> None:
    """Reproduce one Phase 4F-2 and one Phase 4F-3A row per period."""

    previous_f2 = pd.read_csv(DATA_DIR / "phase4f2_network_metrics.csv")
    previous_f3a = pd.read_csv(DATA_DIR / "phase4f3a_network_metrics.csv")
    for period in PHASE4F3B_PERIODS:
        base_plans = build_period_plans(plans_3a, plans_4e, period)["Phase4E"]
        offset = int(base_plans["C"]["ew_offset_s"])
        checks = (
            (200, (0, offset), previous_f2, None, "Phase4E"),
            (300, (5, offset), previous_f3a, 5, None),
        )
        for seed, pair, previous, extra_filter, strategy_filter in checks:
            cache = prepare_arrivals(traffic, period, [seed])
            _, current_frame = evaluate_plan(
                cache, base_plans, period, pair, "Regression"
            )
            selected = previous.loc[
                (previous["period"] == period) & (previous["seed"] == seed)
            ]
            if extra_filter is not None:
                selected = selected.loc[
                    selected["extra_ns_green_s"] == extra_filter
                ]
            if strategy_filter is not None:
                selected = selected.loc[selected["strategy"] == strategy_filter]
            if len(selected) != 1:
                raise ValueError(f"Missing prior regression row for {period}/{seed}.")
            _assert_regression_row(
                current_frame.iloc[0],
                selected.iloc[0],
                f"{period}/{seed}/{pair}",
            )
    print("Phase 4F-2 and Phase 4F-3A regressions passed.")


def run_training() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Screen candidates and select one fairness-feasible plan per period."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    traffic, plans_3a, plans_4e = load_inputs_phase4f2()
    verify_previous_results(traffic, plans_3a, plans_4e)
    selected_records: list[dict[str, object]] = []
    screening_frames: list[pd.DataFrame] = []
    training_frames: list[pd.DataFrame] = []
    fairness_frames: list[pd.DataFrame] = []

    for period in PHASE4F3B_PERIODS:
        base_plans = build_period_plans(plans_3a, plans_4e, period)["Phase4E"]
        training_cache = prepare_arrivals(traffic, period, TRAIN_SEEDS)
        screening_cache = {seed: training_cache[seed] for seed in SCREEN_SEEDS}
        ranking, shortlist = screen_candidates(screening_cache, base_plans, period)
        selected, candidates, checks = optimise_training(
            training_cache, base_plans, period, shortlist
        )
        selected_records.append(selected)
        screening_frames.append(ranking)
        training_frames.append(candidates)
        fairness_frames.append(checks)
        print(
            f"Completed {period} training: {len(shortlist)} shortlisted candidates"
        )

    selected_plans = pd.DataFrame(selected_records)
    screening = pd.concat(screening_frames, ignore_index=True)
    training = pd.concat(training_frames, ignore_index=True)
    fairness = pd.concat(fairness_frames, ignore_index=True)
    if len(screening) != 70:
        raise ValueError("Expected 70 screening rows.")
    if len(selected_plans) != 2:
        raise ValueError("Expected two selected plans.")
    if len(fairness) != 4 * len(training):
        raise ValueError("Expected four fairness checks per training candidate.")

    screening.to_csv(SCREENING_PATH, index=False)
    training.to_csv(TRAINING_CANDIDATES_PATH, index=False)
    fairness.to_csv(TRAINING_FAIRNESS_PATH, index=False)
    selected_plans.to_csv(SELECTED_PLANS_PATH, index=False)
    return traffic, plans_3a, plans_4e, selected_plans


def run_holdout(
    traffic: pd.DataFrame,
    plans_3a: pd.DataFrame,
    plans_4e: pd.DataFrame,
    selected_plans: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate frozen Phase 4E, split-only and joint plans on holdout seeds."""

    movement_frames: list[pd.DataFrame] = []
    network_frames: list[pd.DataFrame] = []
    for period in PHASE4F3B_PERIODS:
        base_plans = build_period_plans(plans_3a, plans_4e, period)["Phase4E"]
        original_offset = int(base_plans["C"]["ew_offset_s"])
        selected = selected_plans.loc[selected_plans["period"] == period]
        if len(selected) != 1:
            raise ValueError(f"Expected one selected plan for {period}.")
        row = selected.iloc[0]
        extra = int(row["extra_ns_green_s"])
        new_offset = int(row["C_ew_offset_s"])
        holdout_cache = prepare_arrivals(traffic, period, HOLDOUT_SEEDS)
        strategies = {
            "Phase4E": (0, original_offset),
            "SplitOnly": (extra, original_offset),
            "Joint": (extra, new_offset),
        }
        for name, pair in strategies.items():
            movements, network = evaluate_plan(
                holdout_cache, base_plans, period, pair, name
            )
            movement_frames.append(movements)
            network_frames.append(network)
        print(f"Completed {period} holdout")

    movements = pd.concat(movement_frames, ignore_index=True)
    network = pd.concat(network_frames, ignore_index=True)
    if len(movements) != 1440:
        raise ValueError("Expected 1,440 holdout movement rows.")
    if len(network) != 180:
        raise ValueError("Expected 180 holdout network rows.")
    movements.to_csv(HOLDOUT_MOVEMENTS_PATH, index=False)
    network.to_csv(HOLDOUT_NETWORK_PATH, index=False)
    return movements, network


def summarise_holdout(
    movements: pd.DataFrame,
    network: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the joint plan with Phase 4E and split-only benchmarks."""

    network_records: list[dict[str, object]] = []
    movement_records: list[dict[str, object]] = []
    for period in PHASE4F3B_PERIODS:
        period_mov = movements.loc[movements["period"] == period]
        period_net = network.loc[network["period"] == period]
        joint_mov = period_mov.loc[period_mov["strategy"] == "Joint"]
        joint_net = period_net.loc[period_net["strategy"] == "Joint"]
        for benchmark in ("Phase4E", "SplitOnly"):
            base_mov = period_mov.loc[period_mov["strategy"] == benchmark]
            base_net = period_net.loc[period_net["strategy"] == benchmark]
            paired_mov, paired_net = compare_plans(
                base_mov, base_net, joint_mov, joint_net
            )
            _assert_unchanged_approaches(paired_mov)

            delay_changes = paired_net["delay_change_s"].to_numpy(dtype=float)
            delay_low, delay_high = bootstrap_mean_ci(delay_changes)
            c_queue_changes = paired_net["C_ns_queue_change_veh"].to_numpy(
                dtype=float
            )
            queue_low, queue_high = bootstrap_mean_ci(c_queue_changes)
            network_records.append(
                {
                    "period": period,
                    "benchmark": benchmark,
                    "replications": len(paired_net),
                    "benchmark_mean_delay_per_passage_s": paired_net[
                        "mean_delay_per_passage_s_base"
                    ].mean(),
                    "joint_mean_delay_per_passage_s": paired_net[
                        "mean_delay_per_passage_s_new"
                    ].mean(),
                    "mean_delay_change_s": delay_changes.mean(),
                    "ci_lower_s": delay_low,
                    "ci_upper_s": delay_high,
                    "seeds_better": int((delay_changes < 0).sum()),
                    "benchmark_C_ns_queue_at_hour": paired_net[
                        "C_ns_queue_at_hour_base"
                    ].mean(),
                    "joint_C_ns_queue_at_hour": paired_net[
                        "C_ns_queue_at_hour_new"
                    ].mean(),
                    "mean_C_ns_queue_change_veh": c_queue_changes.mean(),
                    "C_ns_queue_ci_lower_veh": queue_low,
                    "C_ns_queue_ci_upper_veh": queue_high,
                }
            )

            for movement in ALL_MOVEMENTS:
                group = paired_mov.loc[paired_mov["movement"] == movement]
                changes = group["delay_change_s"].to_numpy(dtype=float)
                low, high = bootstrap_mean_ci(changes)
                queue_changes = group["queue_change_veh"].to_numpy(dtype=float)
                movement_records.append(
                    {
                        "period": period,
                        "benchmark": benchmark,
                        "movement": movement,
                        "protected_movement": movement in PROTECTED_MOVEMENTS,
                        "replications": len(group),
                        "benchmark_mean_delay_s": group[
                            "mean_delay_per_vehicle_s_base"
                        ].mean(),
                        "joint_mean_delay_s": group[
                            "mean_delay_per_vehicle_s_new"
                        ].mean(),
                        "mean_delay_change_s": changes.mean(),
                        "ci_lower_s": low,
                        "ci_upper_s": high,
                        "passes_2s_upper": bool(high <= FAIRNESS_LIMIT_S),
                        "benchmark_queue_at_hour": group[
                            "queue_at_hour_veh_base"
                        ].mean(),
                        "joint_queue_at_hour": group[
                            "queue_at_hour_veh_new"
                        ].mean(),
                        "mean_queue_change_veh": queue_changes.mean(),
                    }
                )

    network_summary = pd.DataFrame(network_records)
    movement_summary = pd.DataFrame(movement_records)
    if len(network_summary) != 4:
        raise ValueError("Expected four holdout network summaries.")
    if len(movement_summary) != 32:
        raise ValueError("Expected 32 holdout movement summaries.")
    network_summary.to_csv(HOLDOUT_NETWORK_SUMMARY_PATH, index=False)
    movement_summary.to_csv(HOLDOUT_MOVEMENT_SUMMARY_PATH, index=False)
    return network_summary, movement_summary


def plot_holdout(network: pd.DataFrame) -> None:
    """Plot synthetic holdout network delay under all three strategies."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    order = ["Phase4E", "SplitOnly", "Joint"]
    for period in PHASE4F3B_PERIODS:
        selected = network.loc[network["period"] == period]
        means = (
            selected.groupby("strategy")["mean_delay_per_passage_s"]
            .mean()
            .reindex(order)
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(means.index, means.values)
        ax.set_title(f"Synthetic Junction C Joint Optimisation: {period}")
        ax.set_ylabel("Mean network delay per passage (s)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        slug = period.lower().replace(" ", "_")
        fig.savefig(
            FIGURE_DIR / f"phase4f3b_holdout_{slug}.png",
            dpi=180,
        )
        plt.close(fig)


def run_phase4f3b() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run training, frozen holdout validation, summaries and figures."""

    traffic, plans_3a, plans_4e, selected_plans = run_training()
    movements, network = run_holdout(
        traffic, plans_3a, plans_4e, selected_plans
    )
    network_summary, movement_summary = summarise_holdout(movements, network)
    plot_holdout(network)
    print("\nSelected Junction C plans:")
    print(selected_plans.to_string(index=False))
    print("\nHoldout network comparison:")
    print(network_summary.to_string(index=False))
    print("\nProtected movement comparison versus Phase 4E:")
    print(
        movement_summary.loc[
            (movement_summary["benchmark"] == "Phase4E")
            & movement_summary["movement"].isin(PROTECTED_MOVEMENTS)
        ].to_string(index=False)
    )
    print("\nPhase 4F-3B completed.")
    return selected_plans, network_summary, movement_summary

# ============================================================================
# Phase 4G integrated audit (python -m workflows phase4g)
# Formerly phase4g_integrated_audit.py. Audit the frozen integrated signal strategy against Phase 3A and Phase 4E.
# ============================================================================

PHASE4G_PERIODS = ("Overnight", "AM peak", "Interpeak", "PM peak")
STRATEGIES = ("Phase3A", "Phase4E", "Integrated")
PHASE4G_SEEDS = range(500, 530)
SIGNAL_PLANS_PATH = DATA_DIR / "phase4g_signal_plans.csv"
PHASE4G_NETWORK_METRICS_PATH = DATA_DIR / "phase4g_network_metrics.csv"
PHASE4G_MOVEMENT_METRICS_PATH = DATA_DIR / "phase4g_movement_metrics.csv"
PHASE4G_NETWORK_SUMMARY_PATH = DATA_DIR / "phase4g_network_summary.csv"
PHASE4G_MOVEMENT_SUMMARY_PATH = DATA_DIR / "phase4g_movement_summary.csv"
FAIRNESS_AUDIT_PATH = DATA_DIR / "phase4g_fairness_audit.csv"
PHASE4G_FIGURE_PATH = FIGURE_DIR / "phase4g_integrated_network.png"


def build_integrated_plans(
    plans_3a: pd.DataFrame,
    plans_4e: pd.DataFrame,
    period: str,
) -> dict[str, dict[str, dict[str, float]]]:
    """Create all three frozen strategies without modifying saved plans."""

    if period not in PHASE4G_PERIODS:
        raise ValueError(f"Unknown period: {period}.")
    strategies = build_period_plans(plans_3a, plans_4e, period)
    integrated = {
        junction: dict(plan)
        for junction, plan in strategies["Phase4E"].items()
    }
    if period == "PM peak":
        integrated = make_candidate_plan(
            base_plans=integrated,
            extra_ns=3,
            c_offset=83,
        )
        c_plan = integrated["C"]
        expected = {
            "ew_green_s": 89.0,
            "ns_green_s": 49.0,
            "ew_offset_s": 83.0,
            "ns_offset_s": 28.0,
        }
        for field, value in expected.items():
            if not np.isclose(c_plan[field], value):
                raise AssertionError(f"Unexpected integrated PM C {field}.")
    elif integrated != strategies["Phase4E"]:
        raise AssertionError("Non-PM integrated plan must equal Phase 4E.")
    strategies["Integrated"] = integrated
    return strategies


def run_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run 360 matched full-junction simulations using frozen signal plans."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    traffic, plans_3a, plans_4e = load_inputs_phase4f2()
    movement_frames: list[pd.DataFrame] = []
    network_records: list[dict[str, object]] = []
    signal_records: list[dict[str, object]] = []

    for period in PHASE4G_PERIODS:
        strategies = build_integrated_plans(plans_3a, plans_4e, period)
        if tuple(strategies) != STRATEGIES:
            raise AssertionError("Unexpected strategy order or names.")
        for name, junction_plans in strategies.items():
            for junction, plan in junction_plans.items():
                signal_records.append(
                    {
                        "period": period,
                        "strategy": name,
                        "junction": junction,
                        **plan,
                    }
                )
        for seed in PHASE4G_SEEDS:
            arrivals = make_arrivals(traffic, period, seed)
            for strategy, junction_plans in strategies.items():
                movements, network = evaluate_network(
                    arrivals=arrivals,
                    junction_plans=junction_plans,
                    strategy=strategy,
                    period=period,
                    seed=seed,
                    horizon_s=HORIZON_S,
                )
                movement_frames.append(movements)
                network_records.append(network)
        print(f"Completed {period}")

    movements = pd.concat(movement_frames, ignore_index=True)
    network = pd.DataFrame(network_records)
    signal_plans = pd.DataFrame(signal_records)
    if len(movements) != 2880:
        raise ValueError("Expected 2,880 movement metric rows.")
    if len(network) != 360:
        raise ValueError("Expected 360 network metric rows.")
    if len(signal_plans) != 36:
        raise ValueError("Expected 36 signal-plan rows.")

    movement_counts = movements.pivot(
        index=["period", "seed", "movement"],
        columns="strategy",
        values="vehicles",
    )
    if not movement_counts.nunique(axis=1).eq(1).all():
        raise ValueError("Strategies received different movement demand.")
    passage_counts = network.pivot(
        index=["period", "seed"],
        columns="strategy",
        values="total_passages",
    )
    if not passage_counts.nunique(axis=1).eq(1).all():
        raise ValueError("Strategies used different passage counts.")

    non_pm_movements = movements.loc[movements["period"] != "PM peak"]
    non_pm_network = network.loc[network["period"] != "PM peak"]
    for frame, keys, fields in (
        (
            non_pm_movements,
            ["period", "seed", "movement"],
            ["total_delay_s", "queue_at_hour_veh", "max_queue_veh"],
        ),
        (
            non_pm_network,
            ["period", "seed"],
            ["mean_delay_per_passage_s", "total_queue_at_hour"],
        ),
    ):
        phase4e = frame.loc[frame["strategy"] == "Phase4E"].drop(
            columns="strategy"
        )
        integrated = frame.loc[frame["strategy"] == "Integrated"].drop(
            columns="strategy"
        )
        paired = phase4e.merge(
            integrated,
            on=keys,
            suffixes=("_4e", "_integrated"),
            validate="one_to_one",
        )
        for field in fields:
            if not np.allclose(
                paired[f"{field}_4e"],
                paired[f"{field}_integrated"],
                atol=1e-12,
            ):
                raise AssertionError(
                    f"Non-PM Integrated and Phase 4E differ for {field}."
                )

    signal_plans.to_csv(SIGNAL_PLANS_PATH, index=False)
    network.to_csv(PHASE4G_NETWORK_METRICS_PATH, index=False)
    movements.to_csv(PHASE4G_MOVEMENT_METRICS_PATH, index=False)
    return movements, network


def summarise_audit(
    movements: pd.DataFrame,
    network: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create direct paired comparisons against Phase 3A and Phase 4E."""

    network_records: list[dict[str, object]] = []
    movement_records: list[dict[str, object]] = []
    final_network = network.loc[network["strategy"] == "Integrated"].drop(
        columns="strategy"
    )
    final_movements = movements.loc[
        movements["strategy"] == "Integrated"
    ].drop(columns="strategy")

    for benchmark in ("Phase3A", "Phase4E"):
        base_network = network.loc[network["strategy"] == benchmark].drop(
            columns="strategy"
        )
        base_movements = movements.loc[
            movements["strategy"] == benchmark
        ].drop(columns="strategy")
        paired_network = base_network.merge(
            final_network,
            on=["period", "seed"],
            suffixes=("_base", "_final"),
            validate="one_to_one",
        )
        paired_movements = base_movements.merge(
            final_movements,
            on=["period", "seed", "movement"],
            suffixes=("_base", "_final"),
            validate="one_to_one",
        )
        if not paired_network["total_passages_base"].eq(
            paired_network["total_passages_final"]
        ).all():
            raise ValueError("Paired network passage counts differ.")
        if not paired_movements["vehicles_base"].eq(
            paired_movements["vehicles_final"]
        ).all():
            raise ValueError("Paired movement vehicle counts differ.")

        paired_network["delay_change_s"] = (
            paired_network["mean_delay_per_passage_s_final"]
            - paired_network["mean_delay_per_passage_s_base"]
        )
        paired_network["queue_change_veh"] = (
            paired_network["total_queue_at_hour_final"]
            - paired_network["total_queue_at_hour_base"]
        )
        paired_movements["delay_change_s"] = (
            paired_movements["mean_delay_per_vehicle_s_final"]
            - paired_movements["mean_delay_per_vehicle_s_base"]
        )
        paired_movements["queue_change_veh"] = (
            paired_movements["queue_at_hour_veh_final"]
            - paired_movements["queue_at_hour_veh_base"]
        )

        for period in PHASE4G_PERIODS:
            net_group = paired_network.loc[paired_network["period"] == period]
            delay_changes = net_group["delay_change_s"].to_numpy(dtype=float)
            delay_low, delay_high = bootstrap_mean_ci(delay_changes)
            queue_changes = net_group["queue_change_veh"].to_numpy(dtype=float)
            queue_low, queue_high = bootstrap_mean_ci(queue_changes)
            network_records.append(
                {
                    "period": period,
                    "benchmark": benchmark,
                    "replications": len(net_group),
                    "benchmark_mean_delay_per_passage_s": net_group[
                        "mean_delay_per_passage_s_base"
                    ].mean(),
                    "integrated_mean_delay_per_passage_s": net_group[
                        "mean_delay_per_passage_s_final"
                    ].mean(),
                    "mean_delay_change_s": delay_changes.mean(),
                    "ci_lower_s": delay_low,
                    "ci_upper_s": delay_high,
                    "seeds_better": int((delay_changes < 0).sum()),
                    "corridor_delay_change_s": (
                        net_group["corridor_mean_delay_s_final"]
                        - net_group["corridor_mean_delay_s_base"]
                    ).mean(),
                    "ns_delay_change_s": (
                        net_group["ns_mean_delay_s_final"]
                        - net_group["ns_mean_delay_s_base"]
                    ).mean(),
                    "benchmark_queue_at_hour": net_group[
                        "total_queue_at_hour_base"
                    ].mean(),
                    "integrated_queue_at_hour": net_group[
                        "total_queue_at_hour_final"
                    ].mean(),
                    "mean_queue_change_veh": queue_changes.mean(),
                    "queue_ci_lower_veh": queue_low,
                    "queue_ci_upper_veh": queue_high,
                }
            )

            for movement in ALL_MOVEMENTS:
                group = paired_movements.loc[
                    (paired_movements["period"] == period)
                    & (paired_movements["movement"] == movement)
                ]
                changes = group["delay_change_s"].to_numpy(dtype=float)
                low, high = bootstrap_mean_ci(changes)
                queue_changes = group["queue_change_veh"].to_numpy(dtype=float)
                movement_records.append(
                    {
                        "period": period,
                        "benchmark": benchmark,
                        "movement": movement,
                        "protected_movement": movement in PROTECTED_MOVEMENTS,
                        "replications": len(group),
                        "benchmark_mean_delay_s": group[
                            "mean_delay_per_vehicle_s_base"
                        ].mean(),
                        "integrated_mean_delay_s": group[
                            "mean_delay_per_vehicle_s_final"
                        ].mean(),
                        "mean_delay_change_s": changes.mean(),
                        "ci_lower_s": low,
                        "ci_upper_s": high,
                        "passes_2s_upper": bool(high <= FAIRNESS_LIMIT_S),
                        "benchmark_queue_at_hour": group[
                            "queue_at_hour_veh_base"
                        ].mean(),
                        "integrated_queue_at_hour": group[
                            "queue_at_hour_veh_final"
                        ].mean(),
                        "mean_queue_change_veh": queue_changes.mean(),
                    }
                )

    network_summary = pd.DataFrame(network_records)
    movement_summary = pd.DataFrame(movement_records)
    fairness_audit = movement_summary.loc[
        movement_summary["protected_movement"]
    ].copy()
    if len(network_summary) != 8:
        raise ValueError("Expected eight network summary rows.")
    if len(movement_summary) != 64:
        raise ValueError("Expected 64 movement summary rows.")
    if len(fairness_audit) != 32:
        raise ValueError("Expected 32 fairness audit rows.")

    network_summary.to_csv(PHASE4G_NETWORK_SUMMARY_PATH, index=False)
    movement_summary.to_csv(PHASE4G_MOVEMENT_SUMMARY_PATH, index=False)
    fairness_audit.to_csv(FAIRNESS_AUDIT_PATH, index=False)
    return network_summary, movement_summary


def plot_network_comparison(network_summary: pd.DataFrame) -> None:
    """Plot integrated-minus-Phase-3A synthetic network delay changes."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    selected = (
        network_summary.loc[network_summary["benchmark"] == "Phase3A"]
        .set_index("period")
        .reindex(PHASE4G_PERIODS)
    )
    x = np.arange(len(PHASE4G_PERIODS))
    y = selected["mean_delay_change_s"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, y)
    ax.axhline(0, linestyle="--", linewidth=1, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(PHASE4G_PERIODS)
    ax.set_ylabel("Integrated minus Phase 3A delay (s/passage)")
    ax.set_title("Synthetic Integrated Signal Strategy: Network Delay Comparison")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PHASE4G_FIGURE_PATH, dpi=180)
    plt.close(fig)


def run_phase4g() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the frozen integrated audit, summaries and comparison figure."""

    movements, network = run_audit()
    network_summary, movement_summary = summarise_audit(movements, network)
    plot_network_comparison(network_summary)
    print("\nIntegrated network comparison:")
    print(network_summary.to_string(index=False))
    print("\nPM protected movements versus Phase 3A:")
    print(
        movement_summary.loc[
            (movement_summary["period"] == "PM peak")
            & (movement_summary["benchmark"] == "Phase3A")
            & movement_summary["movement"].isin(PROTECTED_MOVEMENTS)
        ].to_string(index=False)
    )
    print("\nPM protected movements versus Phase 4E:")
    print(
        movement_summary.loc[
            (movement_summary["period"] == "PM peak")
            & (movement_summary["benchmark"] == "Phase4E")
            & movement_summary["movement"].isin(PROTECTED_MOVEMENTS)
        ].to_string(index=False)
    )
    print("\nPhase 4G completed.")
    return network_summary, movement_summary

# ============================================================================
# Phases 4H and 4I: fairness against the original baseline
# (python -m workflows phase4h, python -m workflows phase4i)
# Follow-up to Phase 4G. The Phase 4F-3B search checked movement fairness
# against Phase 4E, so delay could accumulate against Phase 3A. These phases
# repeat the same 35-candidate Junction C search and selection rule, but
# measure network delay and all four protected movements against Phase 3A.
# Each protocol was fixed before its first run, with fresh seeds no earlier
# phase used. Phase 4H covers the PM peak; Phase 4I covers the AM peak.
# ============================================================================

PHASE4H_PERIOD = "PM peak"
PHASE4H_SCREEN_SEEDS = range(600, 604)
PHASE4H_TRAIN_SEEDS = range(600, 612)
PHASE4H_HOLDOUT_SEEDS = range(620, 650)
PHASE4H_INTEGRATED_PAIR = (3, 83)
PHASE4I_PERIOD = "AM peak"
PHASE4I_SCREEN_SEEDS = range(700, 704)
PHASE4I_TRAIN_SEEDS = range(700, 712)
PHASE4I_HOLDOUT_SEEDS = range(720, 750)
PHASE4I_INTEGRATED_PAIR = (0, 70)
EARLIER_SEEDS = (
    set(range(200, 230)) | set(range(300, 312)) | set(range(400, 450)) | set(range(500, 530))
)


def baseline_search_paths(tag: str) -> dict[str, Path]:
    """Output files for one fairness-against-baseline phase."""

    names = (
        "screening",
        "training_candidates",
        "training_fairness",
        "selected_plan",
        "network_metrics",
        "movement_metrics",
        "network_summary",
        "movement_summary",
    )
    return {name: DATA_DIR / f"{tag}_{name}.csv" for name in names}


PHASE4H_SCREENING_PATH = baseline_search_paths("phase4h")["screening"]
PHASE4H_TRAINING_PATH = baseline_search_paths("phase4h")["training_candidates"]
PHASE4H_TRAINING_FAIRNESS_PATH = baseline_search_paths("phase4h")["training_fairness"]
PHASE4H_SELECTED_PATH = baseline_search_paths("phase4h")["selected_plan"]
PHASE4H_NETWORK_METRICS_PATH = baseline_search_paths("phase4h")["network_metrics"]
PHASE4H_MOVEMENT_METRICS_PATH = baseline_search_paths("phase4h")["movement_metrics"]
PHASE4H_NETWORK_SUMMARY_PATH = baseline_search_paths("phase4h")["network_summary"]
PHASE4H_MOVEMENT_SUMMARY_PATH = baseline_search_paths("phase4h")["movement_summary"]


def evaluate_fixed_plans(
    arrival_cache: Mapping[int, Mapping[tuple[str, str], np.ndarray]],
    junction_plans: Mapping[str, Mapping[str, float]],
    period: str,
    name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one complete A/B/C plan on every cached arrival replication."""

    movement_frames: list[pd.DataFrame] = []
    network_records: list[dict[str, object]] = []
    for seed, arrivals in arrival_cache.items():
        movements, network = evaluate_network(
            arrivals=arrivals,
            junction_plans=junction_plans,
            strategy=name,
            period=period,
            seed=seed,
            horizon_s=HORIZON_S,
        )
        network["C_ns_queue_at_hour"] = int(
            movements.loc[
                movements["movement"].isin(["C_N", "C_S"]), "queue_at_hour_veh"
            ].sum()
        )
        movement_frames.append(movements)
        network_records.append(network)
    return pd.concat(movement_frames, ignore_index=True), pd.DataFrame(network_records)


def _assert_a_unchanged(paired_mov: pd.DataFrame) -> None:
    """Junction A timing is identical in Phase 3A, 4E and every candidate."""

    unchanged = paired_mov.loc[paired_mov["movement"].isin(["A_N", "A_S"])]
    if not np.allclose(unchanged["delay_change_s"], 0.0, atol=1e-12):
        raise AssertionError("Junction A approach delay changed unexpectedly.")


def screen_against_baseline(
    cache: Mapping[int, Mapping[tuple[str, str], np.ndarray]],
    plans_3a: Mapping[str, Mapping[str, float]],
    plans_4e: Mapping[str, Mapping[str, float]],
    period: str,
) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """Rank the 35 Phase 4F-3B candidates against Phase 3A on four seeds."""

    base_mov, base_net = evaluate_fixed_plans(cache, plans_3a, period, "Phase3A")
    baseline_pair = _baseline_pair(plans_4e)
    records: list[dict[str, object]] = []
    for pair in make_candidate_pairs(baseline_pair[1], int(plans_4e["C"]["cycle_s"])):
        cand_mov, cand_net = evaluate_plan(cache, plans_4e, period, pair, "Candidate")
        _, paired_net = compare_plans(base_mov, base_net, cand_mov, cand_net)
        records.append(
            {
                "period": period,
                "extra_ns_green_s": pair[0],
                "C_offset_s": pair[1],
                "replications": len(paired_net),
                "screen_delay_change_vs_3a_s": paired_net["delay_change_s"].mean(),
                "screen_C_ns_queue_change_veh": paired_net["C_ns_queue_change_veh"].mean(),
            }
        )
    ranking = pd.DataFrame(records).sort_values(
        ["screen_delay_change_vs_3a_s", "screen_C_ns_queue_change_veh", "C_offset_s"],
        kind="stable",
    )
    if len(ranking) != 35:
        raise ValueError("Expected 35 screening rows.")
    chosen = {
        (int(r.extra_ns_green_s), int(r.C_offset_s)) for r in ranking.head(12).itertuples()
    }
    for _, group in ranking.groupby("extra_ns_green_s", sort=False):
        chosen.update((int(r.extra_ns_green_s), int(r.C_offset_s)) for r in group.head(2).itertuples())
    chosen.add(baseline_pair)
    return ranking.reset_index(drop=True), sorted(chosen)


def train_against_baseline(
    cache: Mapping[int, Mapping[tuple[str, str], np.ndarray]],
    plans_3a: Mapping[str, Mapping[str, float]],
    plans_4e: Mapping[str, Mapping[str, float]],
    shortlist: Iterable[tuple[int, int]],
    period: str,
) -> tuple[dict[str, object] | None, pd.DataFrame, pd.DataFrame]:
    """Apply the Phase 4F-3B criteria, measured against Phase 3A."""

    base_mov, base_net = evaluate_fixed_plans(cache, plans_3a, period, "Phase3A")
    original_offset = _baseline_pair(plans_4e)[1]
    cycle = int(plans_4e["C"]["cycle_s"])
    candidate_records: list[dict[str, object]] = []
    fairness_frames: list[pd.DataFrame] = []
    for extra, offset in shortlist:
        cand_mov, cand_net = evaluate_plan(cache, plans_4e, period, (extra, offset), "Candidate")
        paired_mov, paired_net = compare_plans(base_mov, base_net, cand_mov, cand_net)
        _assert_a_unchanged(paired_mov)
        fairness = evaluate_fairness(paired_mov)
        fairness.insert(0, "period", period)
        fairness.insert(1, "extra_ns_green_s", extra)
        fairness.insert(2, "C_offset_s", offset)
        fairness_frames.append(fairness)
        changes = paired_net["delay_change_s"].to_numpy(dtype=float)
        low, high = bootstrap_mean_ci(changes)
        network_pass = bool(changes.mean() <= 1e-12)
        fairness_pass = bool(fairness["fairness_pass"].all())
        candidate_records.append(
            {
                "period": period,
                "extra_ns_green_s": extra,
                "C_offset_s": offset,
                "replications": len(paired_net),
                "mean_network_delay_change_vs_3a_s": float(changes.mean()),
                "network_delay_ci_lower_s": low,
                "network_delay_ci_upper_s": high,
                "mean_C_ns_queue_change_veh": paired_net["C_ns_queue_change_veh"].mean(),
                "network_performance_pass": network_pass,
                "movement_fairness_pass": fairness_pass,
                "feasible": bool(network_pass and fairness_pass),
                "offset_distance_s": min(
                    (offset - original_offset) % cycle, (original_offset - offset) % cycle
                ),
            }
        )
    candidates = pd.DataFrame(candidate_records)
    fairness_checks = pd.concat(fairness_frames, ignore_index=True)
    feasible = candidates.loc[candidates["feasible"]].sort_values(
        ["mean_network_delay_change_vs_3a_s", "mean_C_ns_queue_change_veh", "offset_distance_s"],
        kind="stable",
    )
    if feasible.empty:
        return None, candidates, fairness_checks
    winner = feasible.iloc[0]
    plan_c = make_candidate_plan(plans_4e, int(winner["extra_ns_green_s"]), float(winner["C_offset_s"]))["C"]
    selected = {
        "period": period,
        "extra_ns_green_s": int(winner["extra_ns_green_s"]),
        "C_ew_offset_s": int(winner["C_offset_s"]),
        "C_ns_offset_s": plan_c["ns_offset_s"],
        "C_ew_green_s": plan_c["ew_green_s"],
        "C_ns_green_s": plan_c["ns_green_s"],
        "training_mean_network_delay_change_vs_3a_s": winner["mean_network_delay_change_vs_3a_s"],
        "training_feasible_candidates": int(len(feasible)),
        "training_candidate_count": int(len(candidates)),
    }
    return selected, candidates, fairness_checks


def summarise_against_baseline(
    movements: pd.DataFrame,
    network: pd.DataFrame,
    comparisons: Iterable[tuple[str, str]],
    period: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired holdout comparisons for each (benchmark, candidate) pair."""

    network_records: list[dict[str, object]] = []
    movement_records: list[dict[str, object]] = []
    for benchmark, candidate in comparisons:
        paired_mov, paired_net = compare_plans(
            movements.loc[movements["strategy"] == benchmark],
            network.loc[network["strategy"] == benchmark],
            movements.loc[movements["strategy"] == candidate],
            network.loc[network["strategy"] == candidate],
        )
        changes = paired_net["delay_change_s"].to_numpy(dtype=float)
        low, high = bootstrap_mean_ci(changes)
        queue = paired_net["queue_change_veh"].to_numpy(dtype=float)
        q_low, q_high = bootstrap_mean_ci(queue)
        network_records.append(
            {
                "period": period,
                "benchmark": benchmark,
                "candidate": candidate,
                "replications": len(paired_net),
                "benchmark_mean_delay_per_passage_s": paired_net["mean_delay_per_passage_s_base"].mean(),
                "candidate_mean_delay_per_passage_s": paired_net["mean_delay_per_passage_s_new"].mean(),
                "mean_delay_change_s": changes.mean(),
                "ci_lower_s": low,
                "ci_upper_s": high,
                "seeds_better": int((changes < 0).sum()),
                "mean_queue_change_veh": queue.mean(),
                "queue_ci_lower_veh": q_low,
                "queue_ci_upper_veh": q_high,
            }
        )
        for movement in ALL_MOVEMENTS:
            group = paired_mov.loc[paired_mov["movement"] == movement]
            m_changes = group["delay_change_s"].to_numpy(dtype=float)
            m_low, m_high = bootstrap_mean_ci(m_changes)
            movement_records.append(
                {
                    "period": period,
                    "benchmark": benchmark,
                    "candidate": candidate,
                    "movement": movement,
                    "protected_movement": movement in PROTECTED_MOVEMENTS,
                    "replications": len(group),
                    "benchmark_mean_delay_s": group["mean_delay_per_vehicle_s_base"].mean(),
                    "candidate_mean_delay_s": group["mean_delay_per_vehicle_s_new"].mean(),
                    "mean_delay_change_s": m_changes.mean(),
                    "ci_lower_s": m_low,
                    "ci_upper_s": m_high,
                    "passes_2s_upper": bool(m_high <= FAIRNESS_LIMIT_S),
                }
            )
    return pd.DataFrame(network_records), pd.DataFrame(movement_records)


def run_search_against_baseline(
    tag: str,
    period: str,
    screen_seeds: range,
    train_seeds: range,
    holdout_seeds: range,
    integrated_pair: tuple[int, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Screen, train and holdout-test a Junction C plan fair against Phase 3A."""

    if EARLIER_SEEDS & (set(train_seeds) | set(holdout_seeds)):
        raise ValueError(f"{tag} seeds must not overlap earlier phases.")
    if set(train_seeds) & set(holdout_seeds):
        raise ValueError(f"{tag} training and holdout seeds must differ.")
    if not set(screen_seeds) <= set(train_seeds):
        raise ValueError(f"{tag} screening seeds must be training seeds.")
    paths = baseline_search_paths(tag)
    label = "Phase" + tag.removeprefix("phase").upper()

    traffic, plans_3a_table, plans_4e_table = load_inputs_phase4f2()
    built = build_period_plans(plans_3a_table, plans_4e_table, period)
    plans_3a, plans_4e = built["Phase3A"], built["Phase4E"]

    training_cache = prepare_arrivals(traffic, period, train_seeds)
    screening_cache = {seed: training_cache[seed] for seed in screen_seeds}
    ranking, shortlist = screen_against_baseline(screening_cache, plans_3a, plans_4e, period)
    selected, candidates, fairness = train_against_baseline(
        training_cache, plans_3a, plans_4e, shortlist, period
    )
    ranking.to_csv(paths["screening"], index=False)
    candidates.to_csv(paths["training_candidates"], index=False)
    fairness.to_csv(paths["training_fairness"], index=False)
    if selected is None:
        pd.DataFrame([{"period": period, "selected": False}]).to_csv(paths["selected_plan"], index=False)
        print(f"{label}: no candidate met the criteria against Phase 3A.")
        return pd.DataFrame(), candidates, fairness
    pd.DataFrame([selected]).to_csv(paths["selected_plan"], index=False)

    holdout_cache = prepare_arrivals(traffic, period, holdout_seeds)
    strategies = {
        "Phase3A": plans_3a,
        "Phase4E": plans_4e,
        "Integrated": make_candidate_plan(plans_4e, *integrated_pair),
        label: make_candidate_plan(plans_4e, selected["extra_ns_green_s"], selected["C_ew_offset_s"]),
    }
    movement_frames, network_frames = [], []
    for name, plans in strategies.items():
        movements, network = evaluate_fixed_plans(holdout_cache, plans, period, name)
        movement_frames.append(movements)
        network_frames.append(network)
    movements = pd.concat(movement_frames, ignore_index=True)
    network = pd.concat(network_frames, ignore_index=True)
    movements.to_csv(paths["movement_metrics"], index=False)
    network.to_csv(paths["network_metrics"], index=False)
    network_summary, movement_summary = summarise_against_baseline(
        movements,
        network,
        [("Phase3A", label), ("Phase4E", label), ("Phase3A", "Integrated"), ("Integrated", label)],
        period,
    )
    network_summary.to_csv(paths["network_summary"], index=False)
    movement_summary.to_csv(paths["movement_summary"], index=False)
    print(f"\n{label} selected plan:")
    print(pd.DataFrame([selected]).to_string(index=False))
    print(f"\n{label} holdout network comparison:")
    print(network_summary.to_string(index=False))
    print("\nProtected movements:")
    print(movement_summary.loc[movement_summary["protected_movement"]].to_string(index=False))
    return network_summary, movement_summary, candidates


def run_phase4h() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """PM peak: Junction C search with fairness against Phase 3A."""

    return run_search_against_baseline(
        "phase4h",
        PHASE4H_PERIOD,
        PHASE4H_SCREEN_SEEDS,
        PHASE4H_TRAIN_SEEDS,
        PHASE4H_HOLDOUT_SEEDS,
        PHASE4H_INTEGRATED_PAIR,
    )


def run_phase4i() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """AM peak: Junction C search with fairness against Phase 3A."""

    return run_search_against_baseline(
        "phase4i",
        PHASE4I_PERIOD,
        PHASE4I_SCREEN_SEEDS,
        PHASE4I_TRAIN_SEEDS,
        PHASE4I_HOLDOUT_SEEDS,
        PHASE4I_INTEGRATED_PAIR,
    )

# ============================================================================
# Phase 4J demand stress test of the Phase 4H plan (python -m workflows phase4j)
# Checks whether the PM Phase 4H plan still keeps every protected movement
# within the +2 s limit against Phase 3A when demand differs from the level
# it was chosen for. Every approach's 15-minute counts are scaled by the same
# factor (80% to 120%); within a factor all plans receive identical arrivals.
# No plan is re-tuned. Fixed before the first run: fresh seeds 800-829.
# ============================================================================

PHASE4J_PERIOD = "PM peak"
PHASE4J_FACTORS = (0.8, 0.9, 1.0, 1.1, 1.2)
PHASE4J_SEEDS = range(800, 830)
PHASE4J_NETWORK_METRICS_PATH = DATA_DIR / "phase4j_network_metrics.csv"
PHASE4J_MOVEMENT_METRICS_PATH = DATA_DIR / "phase4j_movement_metrics.csv"
PHASE4J_NETWORK_SUMMARY_PATH = DATA_DIR / "phase4j_network_summary.csv"
PHASE4J_MOVEMENT_SUMMARY_PATH = DATA_DIR / "phase4j_movement_summary.csv"


def scale_demand(traffic: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Scale every 15-minute count, and therefore every Poisson rate, by factor."""

    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Demand factor must be positive.")
    scaled = traffic.copy()
    scaled["volume_15min"] = scaled["volume_15min"].astype(float) * float(factor)
    return scaled


def run_phase4j() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stress-test the frozen Phase 4H plan at five demand levels."""

    used = (
        EARLIER_SEEDS
        | set(PHASE4H_TRAIN_SEEDS) | set(PHASE4H_HOLDOUT_SEEDS)
        | set(PHASE4I_TRAIN_SEEDS) | set(PHASE4I_HOLDOUT_SEEDS)
    )
    if used & set(PHASE4J_SEEDS):
        raise ValueError("Phase 4J seeds must not overlap earlier phases.")
    selected = pd.read_csv(PHASE4H_SELECTED_PATH).iloc[0]
    phase4h_pair = (int(selected["extra_ns_green_s"]), int(selected["C_ew_offset_s"]))

    traffic, plans_3a_table, plans_4e_table = load_inputs_phase4f2()
    built = build_period_plans(plans_3a_table, plans_4e_table, PHASE4J_PERIOD)
    plans_4e = built["Phase4E"]
    strategies = {
        "Phase3A": built["Phase3A"],
        "Integrated": make_candidate_plan(plans_4e, *PHASE4H_INTEGRATED_PAIR),
        "Phase4H": make_candidate_plan(plans_4e, *phase4h_pair),
    }
    movement_frames, network_frames, net_summaries, mov_summaries = [], [], [], []
    for factor in PHASE4J_FACTORS:
        cache = prepare_arrivals(scale_demand(traffic, factor), PHASE4J_PERIOD, PHASE4J_SEEDS)
        factor_mov, factor_net = [], []
        for name, plans in strategies.items():
            movements, network = evaluate_fixed_plans(cache, plans, PHASE4J_PERIOD, name)
            movements.insert(0, "demand_factor", factor)
            network.insert(0, "demand_factor", factor)
            factor_mov.append(movements)
            factor_net.append(network)
        movements = pd.concat(factor_mov, ignore_index=True)
        network = pd.concat(factor_net, ignore_index=True)
        net_summary, mov_summary = summarise_against_baseline(
            movements.drop(columns="demand_factor"),
            network.drop(columns="demand_factor"),
            [("Phase3A", "Phase4H"), ("Phase3A", "Integrated")],
            PHASE4J_PERIOD,
        )
        net_summary.insert(0, "demand_factor", factor)
        mov_summary.insert(0, "demand_factor", factor)
        movement_frames.append(movements)
        network_frames.append(network)
        net_summaries.append(net_summary)
        mov_summaries.append(mov_summary)
        print(f"Completed demand factor {factor:.0%}")

    pd.concat(movement_frames, ignore_index=True).to_csv(PHASE4J_MOVEMENT_METRICS_PATH, index=False)
    pd.concat(network_frames, ignore_index=True).to_csv(PHASE4J_NETWORK_METRICS_PATH, index=False)
    network_summary = pd.concat(net_summaries, ignore_index=True)
    movement_summary = pd.concat(mov_summaries, ignore_index=True)
    network_summary.to_csv(PHASE4J_NETWORK_SUMMARY_PATH, index=False)
    movement_summary.to_csv(PHASE4J_MOVEMENT_SUMMARY_PATH, index=False)
    print("\nPhase 4J network delay change against Phase 3A:")
    print(network_summary[["demand_factor", "candidate", "mean_delay_change_s", "ci_lower_s", "ci_upper_s", "seeds_better"]].to_string(index=False))
    print("\nProtected movements against Phase 3A:")
    protected = movement_summary.loc[movement_summary["protected_movement"]]
    print(protected[["demand_factor", "candidate", "movement", "mean_delay_change_s", "ci_upper_s", "passes_2s_upper"]].to_string(index=False))
    return network_summary, movement_summary



PHASES = {
    "phase4f1": run_phase4f1,
    "phase4f2": run_phase4f2,
    "phase4f3a": run_phase4f3a,
    "phase4f3b": run_phase4f3b,
    "phase4g": run_phase4g,
    "phase4h": run_phase4h,
    "phase4i": run_phase4i,
    "phase4j": run_phase4j,
}
