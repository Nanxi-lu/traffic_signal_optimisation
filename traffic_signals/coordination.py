"""Mathematical helpers for initial three-junction signal coordination."""

import numpy as np

from traffic_signals.optimisation import (
    LOST_TIME_S,
    MIN_GREEN_S,
    evaluate_plan,
)


def calculate_offsets(
    cycle_s: float,
    distance_ab_m: float = 400.0,
    distance_bc_m: float = 500.0,
    speed_kmh: float = 40.0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Calculate ideal eastbound travel times and offsets for A, B and C."""

    values = (cycle_s, distance_ab_m, distance_bc_m, speed_kmh)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Cycle, distances and speed must be finite")
    if cycle_s <= 0 or speed_kmh <= 0:
        raise ValueError("Cycle and speed must be positive")
    if distance_ab_m <= 0 or distance_bc_m <= 0:
        raise ValueError("Distances must be positive")

    speed_mps = speed_kmh / 3.6
    travel_ab = distance_ab_m / speed_mps
    travel_ac = (distance_ab_m + distance_bc_m) / speed_mps
    travel_times = {"A": 0.0, "B": travel_ab, "C": travel_ac}
    offsets = {
        intersection: travel_time % cycle_s
        for intersection, travel_time in travel_times.items()
    }
    return travel_times, offsets


def optimise_fixed_cycle_split(
    ew_y: np.ndarray | list[float],
    ns_y: np.ndarray | list[float],
    cycle_s: int,
) -> dict[str, float | int]:
    """Find the best integer-second EW/NS split for a specified cycle."""

    ew_demand = np.asarray(ew_y, dtype=float)
    ns_demand = np.asarray(ns_y, dtype=float)
    if ew_demand.shape != ns_demand.shape or ew_demand.size == 0:
        raise ValueError("Invalid demand arrays")
    if not np.isfinite(ew_demand).all() or not np.isfinite(ns_demand).all():
        raise ValueError("Demand values must be finite")
    if np.any(ew_demand < 0) or np.any(ns_demand < 0):
        raise ValueError("Demand ratios cannot be negative")
    if not isinstance(cycle_s, (int, np.integer)):
        raise ValueError("cycle_s must be an integer number of seconds")

    available_green_s = cycle_s - LOST_TIME_S
    if available_green_s < 2.0 * MIN_GREEN_S:
        raise ValueError("Not enough effective green time")

    total_mean = float(ew_demand.mean() + ns_demand.mean())
    if total_mean > 0:
        target_ew_green_s = float(
            available_green_s * ew_demand.mean() / total_mean
        )
    else:
        target_ew_green_s = float(available_green_s / 2.0)

    best: dict[str, float | int] | None = None
    best_score: tuple[float, ...] | None = None
    maximum_ew_green = int(available_green_s - MIN_GREEN_S)
    for ew_green_s in range(int(MIN_GREEN_S), maximum_ew_green + 1):
        evaluation = evaluate_plan(
            ew_demand,
            ns_demand,
            cycle_s,
            ew_green_s,
        )
        candidate: dict[str, float | int] = {
            "cycle_s": cycle_s,
            "ew_green_s": float(ew_green_s),
            "target_ew_green_s": target_ew_green_s,
            **evaluation,
        }
        score = (
            float(candidate["warning_intervals"]),
            round(float(candidate["total_shortfall"]), 10),
            -round(float(candidate["min_capacity_margin"]), 10),
            round(abs(float(candidate["ew_green_s"]) - target_ew_green_s), 10),
            float(candidate["ew_green_s"]),
        )
        if best_score is None or score < best_score:
            best = candidate
            best_score = score

    if best is None:
        raise ValueError("No feasible green split found")
    return best


def is_in_green(
    arrival_s: float,
    offset_s: float,
    green_s: float,
    cycle_s: float,
) -> bool:
    """Check an arrival against a repeating effective-green window [start, end)."""

    values = (arrival_s, offset_s, green_s, cycle_s)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Arrival, offset, green and cycle values must be finite")
    if cycle_s <= 0:
        raise ValueError("cycle_s must be positive")
    if green_s < 0 or green_s > cycle_s:
        raise ValueError("green_s must be between zero and the cycle length")
    phase_time = (arrival_s - offset_s) % cycle_s
    return bool(phase_time < green_s)
