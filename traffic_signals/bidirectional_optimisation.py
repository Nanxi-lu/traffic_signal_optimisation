"""Vectorised exhaustive search for protected bidirectional signal offsets."""

from collections.abc import Mapping

import numpy as np

from traffic_signals.bidirectional import DISTANCES, count_directional_green
from traffic_signals.offset_optimisation import circular_distance


SPEEDS = (30.0, 35.0, 40.0, 45.0, 50.0)
DIRECTIONS = ("EB", "WB")
JUNCTIONS = ("A", "B", "C")


def _validate_plan(
    cycle: int,
    greens: Mapping[str, float],
    baseline_offsets: Mapping[str, float],
) -> None:
    if not isinstance(cycle, (int, np.integer)) or cycle <= 0:
        raise ValueError("Cycle must be a positive integer.")
    if set(greens) != set(JUNCTIONS):
        raise ValueError("Missing green times.")
    if set(baseline_offsets) != set(JUNCTIONS):
        raise ValueError("Missing offsets.")
    if any(
        not np.isfinite(greens[junction])
        or not 0 < greens[junction] <= cycle
        for junction in JUNCTIONS
    ):
        raise ValueError("Invalid green duration.")
    if any(not np.isfinite(value) for value in baseline_offsets.values()):
        raise ValueError("Offsets must be finite.")


def build_green_masks(
    cycle: int,
    greens: Mapping[str, float],
    direction: str,
) -> dict[str, np.ndarray]:
    """Precompute green checks by speed, candidate offset and departure."""

    _validate_plan(cycle, greens, {junction: 0.0 for junction in JUNCTIONS})
    if direction not in DIRECTIONS:
        raise ValueError("Direction must be EB or WB.")

    departures = np.arange(cycle, dtype=float)
    trial_offsets = np.arange(cycle, dtype=float)[:, None]
    masks: dict[str, list[np.ndarray]] = {junction: [] for junction in JUNCTIONS}
    for speed in SPEEDS:
        for junction in JUNCTIONS:
            distance_m = DISTANCES[direction][junction]
            arrival_s = departures + distance_m * 3.6 / speed
            if junction == "A":
                mask = (arrival_s % cycle) < greens[junction]
            else:
                signal_time_s = (arrival_s[None, :] - trial_offsets) % cycle
                mask = signal_time_s < greens[junction]
            masks[junction].append(mask)
    return {
        junction: np.stack(speed_masks, axis=0)
        for junction, speed_masks in masks.items()
    }


def optimise_bidirectional(
    cycle: int,
    greens: Mapping[str, float],
    baseline_offsets: Mapping[str, float],
    weights: Mapping[str, float],
    protection_pp: float = 5.0,
) -> dict[str, object]:
    """Maximise weighted progression subject to directional 40 km/h protection."""

    _validate_plan(cycle, greens, baseline_offsets)
    if set(weights) != set(DIRECTIONS):
        raise ValueError("Both directional weights are required.")
    if any(not np.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("Invalid directional weights.")
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("Invalid directional weights.")
    if not np.isfinite(protection_pp) or protection_pp < 0:
        raise ValueError("Protection must be a non-negative finite value.")

    masks = {
        direction: build_green_masks(cycle, greens, direction)
        for direction in DIRECTIONS
    }
    design_index = SPEEDS.index(40.0)
    baseline_design = {
        direction: count_directional_green(
            cycle,
            greens,
            baseline_offsets,
            40.0,
            direction,
        )
        / cycle
        for direction in DIRECTIONS
    }
    minimum_design = {
        direction: baseline_design[direction] - protection_pp / 100.0
        for direction in DIRECTIONS
    }

    best: dict[str, object] | None = None
    best_score: tuple[float, ...] | None = None
    for offset_b in range(cycle):
        counts_by_direction: dict[str, np.ndarray] = {}
        for direction in DIRECTIONS:
            masks_for_direction = masks[direction]
            upstream = (
                masks_for_direction["A"]
                & masks_for_direction["B"][:, offset_b, :]
            )
            counts_by_direction[direction] = (
                upstream[:, None, :] & masks_for_direction["C"]
            ).sum(axis=2)

        for offset_c in range(cycle):
            fractions = {
                direction: counts_by_direction[direction][:, offset_c] / cycle
                for direction in DIRECTIONS
            }
            if any(
                fractions[direction][design_index]
                < minimum_design[direction] - 1e-12
                for direction in DIRECTIONS
            ):
                continue

            direction_means = {
                direction: float(fractions[direction].mean())
                for direction in DIRECTIONS
            }
            weighted_mean = sum(
                weights[direction] * direction_means[direction]
                for direction in DIRECTIONS
            )
            weighted_design = sum(
                weights[direction] * fractions[direction][design_index]
                for direction in DIRECTIONS
            )
            offset_change = circular_distance(
                offset_b, baseline_offsets["B"], cycle
            ) + circular_distance(offset_c, baseline_offsets["C"], cycle)
            score = (
                round(weighted_mean, 12),
                round(min(direction_means.values()), 12),
                round(float(weighted_design), 12),
                -float(offset_change),
                -float(offset_b),
                -float(offset_c),
            )
            if best_score is None or score > best_score:
                best_score = score
                best = {
                    "offsets": {"A": 0.0, "B": float(offset_b), "C": float(offset_c)},
                    "counts": {
                        direction: counts_by_direction[direction][:, offset_c]
                        .astype(int)
                        .tolist()
                        for direction in DIRECTIONS
                    },
                    "weighted_mean": float(weighted_mean),
                    "direction_means": direction_means,
                    "weighted_design_40": float(weighted_design),
                    "direction_design_40": {
                        direction: float(fractions[direction][design_index])
                        for direction in DIRECTIONS
                    },
                    "baseline_design_40": baseline_design,
                    "offset_change_s": float(offset_change),
                }

    if best is None:
        raise ValueError("No offset pair satisfies the design-speed protection.")
    return best
