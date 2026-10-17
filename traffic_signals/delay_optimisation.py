"""Delay-aware coarse-to-fine signal-offset optimisation."""

from collections.abc import Mapping
from itertools import product

import numpy as np

from traffic_signals.offset_optimisation import circular_distance
from traffic_signals.vehicle_simulation import simulate_direction


DIRECTIONS = ("EB", "WB")
INTERSECTIONS = ("A", "B", "C")
DURATION_S = 3600.0
SPEED_KMH = 40.0
HEADWAY_S = 2.0


def _validate_plan(
    cycle: int,
    greens: Mapping[str, float],
    offsets: Mapping[str, float],
) -> None:
    """Validate a fixed three-intersection signal plan."""

    if not isinstance(cycle, (int, np.integer)) or cycle <= 0:
        raise ValueError("Cycle must be a positive integer.")
    if set(greens) != set(INTERSECTIONS):
        raise ValueError("Greens must define intersections A, B and C.")
    if set(offsets) != set(INTERSECTIONS):
        raise ValueError("Offsets must define intersections A, B and C.")
    if any(
        not np.isfinite(greens[junction])
        or not 0 < greens[junction] <= cycle
        for junction in INTERSECTIONS
    ):
        raise ValueError("Invalid green duration.")
    if any(not np.isfinite(value) for value in offsets.values()):
        raise ValueError("Offsets must be finite.")


def _validate_arrivals(
    arrivals_by_seed: Mapping[int, Mapping[str, np.ndarray]],
) -> None:
    """Validate the paired arrival samples used by the search."""

    if not arrivals_by_seed:
        raise ValueError("At least one arrival seed is required.")
    for arrivals in arrivals_by_seed.values():
        if set(arrivals) != set(DIRECTIONS):
            raise ValueError("Each seed must define EB and WB arrivals.")
        for direction in DIRECTIONS:
            values = np.asarray(arrivals[direction], dtype=float)
            if (
                values.ndim != 1
                or values.size == 0
                or not np.isfinite(values).all()
                or np.any(values < 0)
                or np.any(values >= DURATION_S)
            ):
                raise ValueError("Invalid arrival sample.")


def evaluate_offsets(
    arrivals_by_seed: Mapping[int, Mapping[str, np.ndarray]],
    cycle: int,
    greens: Mapping[str, float],
    offsets: Mapping[str, float],
) -> dict[str, float]:
    """Evaluate vehicle-weighted delay and mean residual queue for offsets."""

    _validate_plan(cycle, greens, offsets)
    _validate_arrivals(arrivals_by_seed)
    delay_totals = {direction: 0.0 for direction in DIRECTIONS}
    vehicle_totals = {direction: 0 for direction in DIRECTIONS}
    residual_queues: list[int] = []
    signal_plan = {
        "cycle_s": cycle,
        "greens": dict(greens),
        "offsets": dict(offsets),
    }

    for arrivals in arrivals_by_seed.values():
        seed_residual_queue = 0
        for direction in DIRECTIONS:
            vehicles, queues = simulate_direction(
                entry_times=arrivals[direction],
                direction=direction,
                signal_plan=signal_plan,
                speed_kmh=SPEED_KMH,
                headway_s=HEADWAY_S,
                horizon_s=DURATION_S,
            )
            delay_totals[direction] += float(vehicles["total_delay_s"].sum())
            vehicle_totals[direction] += len(vehicles)
            seed_residual_queue += int(queues["queue_at_horizon_veh"].sum())
        residual_queues.append(seed_residual_queue)

    total_vehicles = sum(vehicle_totals.values())
    if total_vehicles == 0 or any(count == 0 for count in vehicle_totals.values()):
        raise ValueError("Each direction must contain at least one vehicle.")
    return {
        "EB_delay_s": delay_totals["EB"] / vehicle_totals["EB"],
        "WB_delay_s": delay_totals["WB"] / vehicle_totals["WB"],
        "combined_delay_s": sum(delay_totals.values()) / total_vehicles,
        "residual_queue_mean": float(np.mean(residual_queues)),
    }


def search_delay_offsets(
    arrivals_by_seed: Mapping[int, Mapping[str, np.ndarray]],
    cycle: int,
    greens: Mapping[str, float],
    original_offsets: Mapping[str, float],
    previous_offsets: Mapping[str, float],
    fairness_limit_s: float = 2.0,
) -> dict[str, object]:
    """Find the best feasible offsets in the specified coarse-to-fine search."""

    _validate_plan(cycle, greens, original_offsets)
    _validate_plan(cycle, greens, previous_offsets)
    _validate_arrivals(arrivals_by_seed)
    if not np.isfinite(fairness_limit_s) or fairness_limit_s < 0:
        raise ValueError("Fairness limit must be non-negative and finite.")

    def pair_from_offsets(offsets: Mapping[str, float]) -> tuple[int, int]:
        values: list[int] = []
        for junction in ("B", "C"):
            value = float(offsets[junction]) % cycle
            if not np.isclose(value, round(value)):
                raise ValueError("Search offsets must be whole seconds.")
            values.append(int(round(value)) % cycle)
        return values[0], values[1]

    original_pair = pair_from_offsets(original_offsets)
    previous_pair = pair_from_offsets(previous_offsets)
    cache: dict[tuple[int, int], dict[str, float]] = {}

    def evaluate(pair: tuple[int, int]) -> dict[str, float]:
        if pair not in cache:
            cache[pair] = evaluate_offsets(
                arrivals_by_seed,
                cycle,
                greens,
                {"A": 0, "B": pair[0], "C": pair[1]},
            )
        return cache[pair]

    baseline = evaluate(original_pair)

    def score(pair: tuple[int, int]) -> tuple[float, ...] | None:
        metrics = evaluate(pair)
        if any(
            metrics[f"{direction}_delay_s"]
            > baseline[f"{direction}_delay_s"] + fairness_limit_s + 1e-9
            for direction in DIRECTIONS
        ):
            return None
        offset_change = circular_distance(
            pair[0], original_pair[0], cycle
        ) + circular_distance(pair[1], original_pair[1], cycle)
        return (
            round(metrics["combined_delay_s"], 8),
            round(metrics["residual_queue_mean"], 8),
            offset_change,
            pair[0],
            pair[1],
        )

    coarse = set(product(range(0, cycle, 10), range(0, cycle, 10)))
    coarse.update([original_pair, previous_pair])
    ranked: list[tuple[tuple[float, ...], tuple[int, int]]] = []
    for pair in sorted(coarse):
        candidate_score = score(pair)
        if candidate_score is not None:
            ranked.append((candidate_score, pair))
    ranked.sort()

    centers = [pair for _, pair in ranked[:3]]
    centers.extend([original_pair, previous_pair])
    for center_b, center_c in centers:
        for db in range(-5, 6):
            for dc in range(-5, 6):
                pair = ((center_b + db) % cycle, (center_c + dc) % cycle)
                score(pair)

    feasible: list[tuple[tuple[float, ...], tuple[int, int]]] = []
    for pair in cache:
        candidate_score = score(pair)
        if candidate_score is not None:
            feasible.append((candidate_score, pair))
    feasible.sort()
    if not feasible:
        raise RuntimeError("No feasible plan found.")
    best_pair = feasible[0][1]
    return {
        "offsets": {"A": 0, "B": best_pair[0], "C": best_pair[1]},
        "baseline": baseline,
        "optimised": evaluate(best_pair),
        "candidates_tested": len(cache),
        "feasible_candidates": len(feasible),
    }
