"""Nested synthetic arrival streams for demand-sensitivity experiments."""

from collections.abc import Iterable

import numpy as np


def create_demand_scenarios(
    baseline_arrivals: np.ndarray,
    baseline_rate: float,
    duration_s: float,
    multipliers: Iterable[float],
    rng: np.random.Generator,
) -> dict[float, np.ndarray]:
    """Create nested demand scenarios while preserving the baseline stream.

    Scenarios below 1.0 retain a nested random subset of baseline arrivals.
    Scenarios above 1.0 add successive independent Poisson arrival streams, so
    each higher multiplier contains every vehicle in the preceding scenario.
    """

    baseline = np.asarray(baseline_arrivals, dtype=float)
    factors = sorted(set(multipliers))
    if 1.0 not in factors:
        raise ValueError("A baseline multiplier of 1.0 is required.")
    if not all(np.isfinite(factor) and factor > 0 for factor in factors):
        raise ValueError("Demand multipliers must be positive.")
    if (
        not np.isfinite(baseline_rate)
        or baseline_rate < 0
        or not np.isfinite(duration_s)
        or duration_s <= 0
    ):
        raise ValueError("Invalid demand rate or duration.")
    if baseline.ndim != 1 or not np.isfinite(baseline).all():
        raise ValueError("Invalid arrival times.")
    if np.any(baseline < 0) or np.any(baseline >= duration_s):
        raise ValueError("Arrival times must fall within the generation period.")
    if np.any(np.diff(baseline) < 0):
        raise ValueError("Arrival times must be sorted.")

    scenarios: dict[float, np.ndarray] = {1.0: baseline.copy()}

    retention_draws = rng.random(len(baseline))
    for factor in (value for value in factors if value < 1.0):
        scenarios[factor] = baseline[retention_draws < factor].copy()

    previous_factor = 1.0
    previous_arrivals = baseline.copy()
    for factor in (value for value in factors if value > 1.0):
        additional_count = int(
            rng.poisson((factor - previous_factor) * baseline_rate * duration_s)
        )
        additional = rng.uniform(0.0, duration_s, additional_count)
        previous_arrivals = np.sort(
            np.concatenate([previous_arrivals, additional])
        )
        scenarios[factor] = previous_arrivals.copy()
        previous_factor = factor

    return {factor: scenarios[factor] for factor in factors}
