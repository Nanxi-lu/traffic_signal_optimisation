"""Integer-second offset search for robust idealised eastbound progression."""

from collections.abc import Mapping, Sequence

import numpy as np


SPEEDS = [30.0, 35.0, 40.0, 45.0, 50.0]
DISTANCES = {"A": 0.0, "B": 400.0, "C": 900.0}
INTERSECTIONS = ("A", "B", "C")


def _validate_signal_inputs(
    cycle: int,
    greens: Mapping[str, float],
    offsets: Mapping[str, float],
) -> None:
    """Validate one three-intersection signal plan."""

    if not isinstance(cycle, (int, np.integer)) or cycle <= 0:
        raise ValueError("Cycle must be a positive integer number of seconds.")
    if set(greens) != set(INTERSECTIONS):
        raise ValueError("Greens must define intersections A, B and C.")
    if set(offsets) != set(INTERSECTIONS):
        raise ValueError("Offsets must define intersections A, B and C.")
    if any(not np.isfinite(value) for value in greens.values()):
        raise ValueError("Green durations must be finite.")
    if any(value < 0 or value > cycle for value in greens.values()):
        raise ValueError("Green durations must be between zero and the cycle.")
    if any(not np.isfinite(value) for value in offsets.values()):
        raise ValueError("Offsets must be finite.")


def count_all_green(
    cycle: int,
    greens: Mapping[str, float],
    offsets: Mapping[str, float],
    speed: float,
) -> int:
    """Count whole-second departures passing every signal on effective green."""

    _validate_signal_inputs(cycle, greens, offsets)
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("Speed must be positive and finite.")

    departures = np.arange(cycle, dtype=float)
    all_green = np.ones(cycle, dtype=bool)
    for junction in INTERSECTIONS:
        travel_time = DISTANCES[junction] * 3.6 / speed
        arrival = departures + travel_time
        signal_time = (arrival - offsets[junction]) % cycle
        all_green &= signal_time < greens[junction]
    return int(all_green.sum())


def circular_distance(a: float, b: float, cycle: int) -> float:
    """Return the smallest separation between two offsets in a cycle."""

    if not isinstance(cycle, (int, np.integer)) or cycle <= 0:
        raise ValueError("Cycle must be a positive integer number of seconds.")
    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("Offsets must be finite.")
    difference = abs(a - b) % cycle
    return float(min(difference, cycle - difference))


def optimise_offsets(
    cycle: int,
    greens: Mapping[str, float],
    baseline_offsets: Mapping[str, float],
    speeds: Sequence[float] = SPEEDS,
) -> dict[str, object]:
    """Optimise integer B/C offsets using the stated robustness priorities."""

    _validate_signal_inputs(cycle, greens, baseline_offsets)
    if not speeds or any(not np.isfinite(speed) or speed <= 0 for speed in speeds):
        raise ValueError("Speeds must be a non-empty sequence of positive values.")
    try:
        design_index = list(speeds).index(40.0)
    except ValueError as error:
        raise ValueError("Speeds must include the 40 km/h design speed.") from error

    best: dict[str, object] | None = None
    best_score: tuple[float, ...] | None = None
    for offset_b in range(cycle):
        for offset_c in range(cycle):
            candidate_offsets = {"A": 0.0, "B": float(offset_b), "C": float(offset_c)}
            counts = [
                count_all_green(cycle, greens, candidate_offsets, speed)
                for speed in speeds
            ]
            offset_change = circular_distance(
                offset_b, baseline_offsets["B"], cycle
            ) + circular_distance(offset_c, baseline_offsets["C"], cycle)
            score = (
                float(sum(counts)),
                float(min(counts)),
                float(counts[design_index]),
                -offset_change,
                -float(offset_b),
                -float(offset_c),
            )
            if best_score is None or score > best_score:
                best_score = score
                best = {
                    "offsets": candidate_offsets,
                    "counts": counts,
                    "mean_fraction": float(np.mean(counts) / cycle),
                    "worst_fraction": float(min(counts) / cycle),
                    "design_fraction": float(counts[design_index] / cycle),
                    "offset_change_s": float(offset_change),
                }

    if best is None:
        raise ValueError("No offset combination was evaluated.")
    return best
