"""Simplified deterministic vehicle and FIFO queue simulation mathematics."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


JUNCTIONS = {
    "EB": (("A", 0.0), ("B", 400.0), ("C", 900.0)),
    "WB": (("C", 0.0), ("B", 500.0), ("A", 900.0)),
}


def next_green_time(
    arrival_s: float,
    offset_s: float,
    green_s: float,
    cycle_s: float,
) -> float:
    """Find the earliest effective-green time at or after arrival."""

    values = np.asarray([arrival_s, offset_s, green_s, cycle_s], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Signal parameters must be finite.")
    if cycle_s <= 0 or not 0 < green_s <= cycle_s:
        raise ValueError("Invalid cycle or green duration.")
    phase = (arrival_s - offset_s) % cycle_s
    if phase < green_s:
        return float(arrival_s)
    cycle_index = np.floor((arrival_s - offset_s) / cycle_s)
    return float(offset_s + (cycle_index + 1.0) * cycle_s)


def _maximum_queue_after_events(
    arrivals: np.ndarray,
    departures: np.ndarray,
) -> int:
    """Return the largest waiting queue after simultaneous events are applied."""

    if arrivals.size == 0:
        return 0
    event_times = np.unique(np.concatenate([arrivals, departures]))
    arrived = np.searchsorted(arrivals, event_times, side="right")
    discharged = np.searchsorted(departures, event_times, side="right")
    return int(np.max(arrived - discharged))


def serve_approach(
    arrivals: Sequence[float] | np.ndarray,
    cycle_s: float,
    offset_s: float,
    green_s: float,
    headway_s: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Discharge sorted arrivals FIFO through one repeating green window."""

    arrival_array = np.asarray(arrivals, dtype=float)
    if arrival_array.ndim != 1 or not np.isfinite(arrival_array).all():
        raise ValueError("Arrivals must be a finite one-dimensional sequence.")
    if np.any(arrival_array < 0):
        raise ValueError("Arrival times cannot be negative.")
    if not np.isfinite(headway_s) or headway_s <= 0:
        raise ValueError("Discharge headway must be positive and finite.")
    if arrival_array.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float), 0
    if np.any(np.diff(arrival_array) < 0):
        raise ValueError("Arrivals must be sorted in non-decreasing order.")

    departures = np.empty(arrival_array.size, dtype=float)
    previous_departure = -np.inf
    for index, arrival_s in enumerate(arrival_array):
        earliest_s = max(float(arrival_s), previous_departure + headway_s)
        departure_s = next_green_time(earliest_s, offset_s, green_s, cycle_s)
        departures[index] = departure_s
        previous_departure = departure_s

    waiting = departures - arrival_array
    max_queue = _maximum_queue_after_events(arrival_array, departures)
    return departures, waiting, max_queue


def simulate_direction(
    entry_times: Sequence[float] | np.ndarray,
    direction: str,
    signal_plan: Mapping[str, object],
    speed_kmh: float = 40.0,
    headway_s: float = 2.0,
    horizon_s: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate one direction through three independent FIFO approaches."""

    if direction not in JUNCTIONS:
        raise ValueError("Direction must be EB or WB.")
    if not np.isfinite(speed_kmh) or speed_kmh <= 0:
        raise ValueError("Speed must be positive and finite.")
    if horizon_s is not None:
        if not np.isfinite(horizon_s) or horizon_s < 0:
            raise ValueError("Invalid observation horizon.")
    required = {"cycle_s", "greens", "offsets"}
    if not required.issubset(signal_plan):
        raise ValueError("Signal plan must define cycle_s, greens and offsets.")

    cycle_s = float(signal_plan["cycle_s"])
    greens = signal_plan["greens"]
    offsets = signal_plan["offsets"]
    intersections = {name for name, _ in JUNCTIONS[direction]}
    if not isinstance(greens, Mapping) or set(greens) != intersections:
        raise ValueError("Green times must define intersections A, B and C.")
    if not isinstance(offsets, Mapping) or set(offsets) != intersections:
        raise ValueError("Offsets must define intersections A, B and C.")

    entries = np.asarray(entry_times, dtype=float)
    if entries.ndim != 1 or not np.isfinite(entries).all() or np.any(entries < 0):
        raise ValueError("Entry times must be a finite non-negative sequence.")
    entries = np.sort(entries)
    vehicle_count = entries.size
    vehicle_ids = [f"{direction}_{index:05d}" for index in range(vehicle_count)]
    vehicle_data: dict[str, object] = {
        "vehicle_id": vehicle_ids,
        "direction": direction,
        "entry_time_s": entries,
    }
    queue_records: list[dict[str, object]] = []

    speed_mps = speed_kmh / 3.6
    arrivals = entries.copy()
    previous_distance_m = 0.0
    total_delay = np.zeros(vehicle_count, dtype=float)
    waiting_events = np.zeros(vehicle_count, dtype=int)
    final_departures = entries.copy()

    for junction, distance_m in JUNCTIONS[direction]:
        if distance_m > previous_distance_m:
            arrivals = final_departures + (distance_m - previous_distance_m) / speed_mps
        departures, waiting, max_queue = serve_approach(
            arrivals,
            cycle_s=cycle_s,
            offset_s=float(offsets[junction]),
            green_s=float(greens[junction]),
            headway_s=headway_s,
        )
        key = junction.lower()
        vehicle_data[f"arrival_{key}_s"] = arrivals
        vehicle_data[f"discharge_{key}_s"] = departures
        vehicle_data[f"wait_{key}_s"] = waiting
        total_delay += waiting
        waiting_events += waiting > 1e-12
        queue_record = {
            "direction": direction,
            "intersection_id": junction,
            "vehicles": vehicle_count,
            "mean_wait_s": float(waiting.mean()) if vehicle_count else 0.0,
            "max_queue": max_queue,
            "last_discharge_s": (
                float(departures[-1]) if vehicle_count else np.nan
            ),
        }
        if horizon_s is not None:
            n_arrived = np.searchsorted(arrivals, horizon_s, side="right")
            n_departed = np.searchsorted(departures, horizon_s, side="right")
            queue_record["queue_at_horizon_veh"] = int(n_arrived - n_departed)
        queue_records.append(queue_record)
        final_departures = departures
        previous_distance_m = distance_m

    vehicle_data["exit_time_s"] = final_departures
    vehicle_data["total_delay_s"] = total_delay
    vehicle_data["journey_time_s"] = final_departures - entries
    vehicle_data["waiting_events"] = waiting_events
    return pd.DataFrame(vehicle_data), pd.DataFrame(queue_records)
