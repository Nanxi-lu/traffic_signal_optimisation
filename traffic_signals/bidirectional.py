"""Idealised all-green progression checks in both corridor directions."""

from collections.abc import Mapping

import numpy as np


DISTANCES = {
    "EB": {"A": 0.0, "B": 400.0, "C": 900.0},
    "WB": {"C": 0.0, "B": 500.0, "A": 900.0},
}
INTERSECTIONS = {"A", "B", "C"}


def count_directional_green(
    cycle_s: int,
    greens: Mapping[str, float],
    offsets: Mapping[str, float],
    speed_kmh: float,
    direction: str,
) -> int:
    """Count whole-second departures encountering EW green at all signals."""

    if direction not in DISTANCES:
        raise ValueError("Direction must be EB or WB.")
    if not isinstance(cycle_s, (int, np.integer)):
        raise ValueError("Cycle must be an integer.")
    if cycle_s <= 0:
        raise ValueError("Cycle must be positive.")
    if not np.isfinite(speed_kmh) or speed_kmh <= 0:
        raise ValueError("Speed must be positive and finite.")
    if set(greens) != INTERSECTIONS:
        raise ValueError("Greens must define intersections A, B and C.")
    if set(offsets) != INTERSECTIONS:
        raise ValueError("Offsets must define intersections A, B and C.")
    if any(not np.isfinite(value) for value in greens.values()):
        raise ValueError("Green durations must be finite.")
    if any(value < 0 or value > cycle_s for value in greens.values()):
        raise ValueError("Green durations must be between zero and the cycle.")
    if any(not np.isfinite(value) for value in offsets.values()):
        raise ValueError("Offsets must be finite.")

    departures = np.arange(cycle_s, dtype=float)
    all_green = np.ones(cycle_s, dtype=bool)
    for intersection, distance_m in DISTANCES[direction].items():
        travel_time_s = distance_m * 3.6 / speed_kmh
        arrival_s = departures + travel_time_s
        signal_time_s = (arrival_s - offsets[intersection]) % cycle_s
        all_green &= signal_time_s < greens[intersection]
    return int(all_green.sum())
