"""Transparent grid-search optimisation for fixed two-phase signal plans."""

import numpy as np


LOST_TIME_S = 12.0
MIN_GREEN_S = 10.0
CAPACITY_TOLERANCE = 1e-9


def evaluate_plan(
    ew_y: np.ndarray | list[float],
    ns_y: np.ndarray | list[float],
    cycle_s: float,
    ew_green_s: float,
) -> dict[str, float | int]:
    """Evaluate one fixed plan against multiple 15-minute observations."""

    ew_demand = np.asarray(ew_y, dtype=float)
    ns_demand = np.asarray(ns_y, dtype=float)
    if ew_demand.shape != ns_demand.shape or ew_demand.size == 0:
        raise ValueError("Invalid demand arrays")
    if not np.isfinite(ew_demand).all() or not np.isfinite(ns_demand).all():
        raise ValueError("Demand values must be finite")
    if np.any(ew_demand < 0) or np.any(ns_demand < 0):
        raise ValueError("Demand ratios cannot be negative")
    if not np.isfinite(cycle_s) or not np.isfinite(ew_green_s):
        raise ValueError("Cycle and green times must be finite")
    if cycle_s <= LOST_TIME_S:
        raise ValueError("Cycle must exceed lost time")

    ns_green_s = float(cycle_s - LOST_TIME_S - ew_green_s)
    if min(ew_green_s, ns_green_s) < MIN_GREEN_S - CAPACITY_TOLERANCE:
        raise ValueError("Minimum green constraint violated")

    ew_capacity_ratio = float(ew_green_s / cycle_s)
    ns_capacity_ratio = float(ns_green_s / cycle_s)
    ew_shortfall = np.clip(ew_demand - ew_capacity_ratio, 0.0, None)
    ns_shortfall = np.clip(ns_demand - ns_capacity_ratio, 0.0, None)
    warning_mask = (
        (ew_shortfall > CAPACITY_TOLERANCE)
        | (ns_shortfall > CAPACITY_TOLERANCE)
    )
    margins = np.concatenate(
        [ew_capacity_ratio - ew_demand, ns_capacity_ratio - ns_demand]
    )

    return {
        "ns_green_s": ns_green_s,
        "ew_capacity_ratio": ew_capacity_ratio,
        "ns_capacity_ratio": ns_capacity_ratio,
        "warning_intervals": int(warning_mask.sum()),
        "total_shortfall": float(ew_shortfall.sum() + ns_shortfall.sum()),
        "max_shortfall": float(max(ew_shortfall.max(), ns_shortfall.max())),
        "min_capacity_margin": float(margins.min()),
    }


def optimise_plan(
    ew_y: np.ndarray | list[float],
    ns_y: np.ndarray | list[float],
    baseline_cycle_s: float,
    baseline_ew_green_s: float,
) -> dict[str, float | int]:
    """Search for the best fixed-time plan under explicit priorities.

    Priority order: fewer warning intervals, smaller total shortfall, shorter
    cycle, then greater minimum phase-capacity margin.
    """

    baseline = evaluate_plan(
        ew_y,
        ns_y,
        baseline_cycle_s,
        baseline_ew_green_s,
    )
    best: dict[str, float | int] = {
        "cycle_s": int(baseline_cycle_s),
        "ew_green_s": float(baseline_ew_green_s),
        **baseline,
    }

    def score(result: dict[str, float | int]) -> tuple[float, ...]:
        return (
            float(result["warning_intervals"]),
            round(float(result["total_shortfall"]), 10),
            float(result["cycle_s"]),
            -round(float(result["min_capacity_margin"]), 10),
            float(result["ew_green_s"]),
        )

    for cycle_s in range(40, 151, 5):
        available_green_s = cycle_s - LOST_TIME_S
        maximum_ew_green = int(available_green_s - MIN_GREEN_S)
        for ew_green_s in range(int(MIN_GREEN_S), maximum_ew_green + 1):
            candidate_evaluation = evaluate_plan(
                ew_y,
                ns_y,
                cycle_s,
                ew_green_s,
            )
            candidate: dict[str, float | int] = {
                "cycle_s": cycle_s,
                "ew_green_s": float(ew_green_s),
                **candidate_evaluation,
            }
            if score(candidate) < score(best):
                best = candidate

    return best
