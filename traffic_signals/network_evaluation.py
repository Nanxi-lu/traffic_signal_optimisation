"""Reusable movement-level and network-level full-junction evaluation."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from traffic_signals.full_junction import (
    create_two_phase_plan,
    simulate_ns_approach,
)
from traffic_signals.vehicle_simulation import simulate_direction


JUNCTIONS = ("A", "B", "C")


def build_period_plans(
    plans_3a: pd.DataFrame,
    plans_4e: pd.DataFrame,
    period: str,
) -> dict[str, dict[str, dict[str, float]]]:
    """Build fixed Phase 3A and Phase 4E two-phase plans for one period."""

    original = plans_3a.loc[plans_3a["period"] == period].set_index(
        "intersection_id"
    )
    if set(original.index) != set(JUNCTIONS) or len(original) != 3:
        raise ValueError(f"Expected A, B and C Phase 3A plans for {period}.")
    selected = plans_4e.loc[plans_4e["period"] == period]
    if len(selected) != 1:
        raise ValueError(f"Expected one Phase 4E plan for {period}.")
    selected_row = selected.iloc[0]
    offsets = {
        "Phase3A": original["offset_s"].to_dict(),
        "Phase4E": {
            "A": float(selected_row["A_offset_s"]),
            "B": float(selected_row["B_offset_s"]),
            "C": float(selected_row["C_offset_s"]),
        },
    }

    strategies: dict[str, dict[str, dict[str, float]]] = {}
    for strategy, strategy_offsets in offsets.items():
        junction_plans: dict[str, dict[str, float]] = {}
        for junction in JUNCTIONS:
            row = original.loc[junction]
            lost_time = float(row.get("lost_time_s", 12.0))
            junction_plans[junction] = create_two_phase_plan(
                cycle_s=float(row["common_cycle_s"]),
                ew_green_s=float(row["ew_green_s"]),
                ns_green_s=float(row["ns_green_s"]),
                ew_offset_s=float(strategy_offsets[junction]),
                lost_time_s=lost_time,
                interphase_loss_s=lost_time / 2.0,
            )
        cycles = {plan["cycle_s"] for plan in junction_plans.values()}
        if len(cycles) != 1:
            raise ValueError(f"Inconsistent common cycle for {period}.")
        strategies[strategy] = junction_plans
    return strategies


def evaluate_network(
    arrivals: Mapping[tuple[str, str], np.ndarray],
    junction_plans: Mapping[str, Mapping[str, float]],
    strategy: str,
    period: str,
    seed: int,
    horizon_s: float = 3600.0,
    speed_kmh: float = 40.0,
    headway_s: float = 2.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Simulate eight traffic streams and return movements plus network summary."""

    required_streams = {
        ("A", "E"),
        ("C", "W"),
        ("A", "N"),
        ("A", "S"),
        ("B", "N"),
        ("B", "S"),
        ("C", "N"),
        ("C", "S"),
    }
    if set(arrivals) != required_streams:
        raise ValueError("Arrivals must define exactly eight traffic streams.")
    if set(junction_plans) != set(JUNCTIONS):
        raise ValueError("Junction plans must define A, B and C.")

    cycle = float(junction_plans["A"]["cycle_s"])
    corridor_plan = {
        "cycle_s": cycle,
        "greens": {
            junction: float(junction_plans[junction]["ew_green_s"])
            for junction in JUNCTIONS
        },
        "offsets": {
            junction: float(junction_plans[junction]["ew_offset_s"])
            for junction in JUNCTIONS
        },
    }
    records: list[dict[str, object]] = []

    def add_record(
        movement: str,
        category: str,
        vehicles: int,
        total_delay_s: float,
        passages: int,
        queue_at_hour_veh: int,
        max_queue_veh: int,
        waiting_events: int,
        exits_after_hour: int,
    ) -> None:
        records.append(
            {
                "period": period,
                "seed": seed,
                "strategy": strategy,
                "movement": movement,
                "category": category,
                "vehicles": vehicles,
                "passages": passages,
                "total_delay_s": total_delay_s,
                "mean_delay_per_vehicle_s": (
                    total_delay_s / vehicles if vehicles else 0.0
                ),
                "delay_per_passage_s": (
                    total_delay_s / passages if passages else 0.0
                ),
                "queue_at_hour_veh": queue_at_hour_veh,
                "max_queue_veh": max_queue_veh,
                "waiting_events": waiting_events,
                "exits_after_hour": exits_after_hour,
            }
        )

    for direction, stream_key in (
        ("EB", ("A", "E")),
        ("WB", ("C", "W")),
    ):
        vehicles, queues = simulate_direction(
            entry_times=arrivals[stream_key],
            direction=direction,
            signal_plan=corridor_plan,
            speed_kmh=speed_kmh,
            headway_s=headway_s,
            horizon_s=horizon_s,
        )
        vehicle_count = len(vehicles)
        add_record(
            movement=direction,
            category="Corridor",
            vehicles=vehicle_count,
            total_delay_s=float(vehicles["total_delay_s"].sum()),
            passages=vehicle_count * 3,
            queue_at_hour_veh=int(queues["queue_at_horizon_veh"].sum()),
            max_queue_veh=int(queues["max_queue"].max()),
            waiting_events=int(vehicles["waiting_events"].sum()),
            exits_after_hour=int((vehicles["exit_time_s"] > horizon_s).sum()),
        )

    for junction in JUNCTIONS:
        for approach in ("N", "S"):
            result = simulate_ns_approach(
                arrivals=arrivals[(junction, approach)],
                plan=junction_plans[junction],
                headway_s=headway_s,
                horizon_s=horizon_s,
            )
            add_record(
                movement=f"{junction}_{approach}",
                category="SideRoad",
                vehicles=int(result["vehicles"]),
                total_delay_s=float(result["total_delay_s"]),
                passages=int(result["vehicles"]),
                queue_at_hour_veh=int(result["queue_at_horizon_veh"]),
                max_queue_veh=int(result["max_queue_veh"]),
                waiting_events=int(result["waiting_events"]),
                exits_after_hour=int(result["exits_after_horizon"]),
            )

    movements = pd.DataFrame(records)
    if len(movements) != 8:
        raise ValueError("Expected eight movement records.")
    corridor = movements.loc[movements["category"] == "Corridor"]
    side_roads = movements.loc[movements["category"] == "SideRoad"]
    corridor_delay = float(corridor["total_delay_s"].sum())
    side_delay = float(side_roads["total_delay_s"].sum())
    corridor_n = int(corridor["vehicles"].sum())
    side_n = int(side_roads["vehicles"].sum())
    total_delay = corridor_delay + side_delay
    total_passages = int(movements["passages"].sum())
    network: dict[str, object] = {
        "period": period,
        "seed": seed,
        "strategy": strategy,
        "corridor_vehicles": corridor_n,
        "ns_vehicles": side_n,
        "total_passages": total_passages,
        "corridor_mean_delay_s": corridor_delay / corridor_n,
        "ns_mean_delay_s": side_delay / side_n,
        "mean_delay_per_passage_s": total_delay / total_passages,
        "total_delay_vehicle_seconds": total_delay,
        "corridor_queue_at_hour": int(corridor["queue_at_hour_veh"].sum()),
        "ns_queue_at_hour": int(side_roads["queue_at_hour_veh"].sum()),
        "total_queue_at_hour": int(movements["queue_at_hour_veh"].sum()),
    }
    return movements, network
