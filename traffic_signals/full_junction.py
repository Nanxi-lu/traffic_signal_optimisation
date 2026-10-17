"""Two-phase signal schedules and local north-south approach simulation."""

from collections.abc import Mapping, Sequence

import numpy as np

from traffic_signals.vehicle_simulation import serve_approach


def create_two_phase_plan(
    cycle_s: float,
    ew_green_s: float,
    ns_green_s: float,
    ew_offset_s: float,
    lost_time_s: float = 12.0,
    interphase_loss_s: float = 6.0,
) -> dict[str, float]:
    """Create two mutually exclusive effective-green windows.

    The schedule is EW green, one interphase loss, NS green, then the second
    interphase loss before the next EW green begins.
    """

    values = np.asarray(
        [
            cycle_s,
            ew_green_s,
            ns_green_s,
            ew_offset_s,
            lost_time_s,
            interphase_loss_s,
        ],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise ValueError("Signal schedule values must be finite.")
    if cycle_s <= 0 or ew_green_s <= 0 or ns_green_s <= 0:
        raise ValueError("Cycle and effective greens must be positive.")
    if lost_time_s < 0 or interphase_loss_s < 0:
        raise ValueError("Lost times cannot be negative.")
    if not np.isclose(2.0 * interphase_loss_s, lost_time_s):
        raise ValueError("The two interphase losses must equal total lost time.")
    if not np.isclose(ew_green_s + ns_green_s + lost_time_s, cycle_s):
        raise ValueError("Effective greens and lost time must equal the cycle.")

    normalised_ew_offset = float(ew_offset_s % cycle_s)
    ns_offset = float(
        (normalised_ew_offset + ew_green_s + interphase_loss_s) % cycle_s
    )
    return {
        "cycle_s": float(cycle_s),
        "ew_green_s": float(ew_green_s),
        "ns_green_s": float(ns_green_s),
        "lost_time_s": float(lost_time_s),
        "interphase_loss_s": float(interphase_loss_s),
        "ew_offset_s": normalised_ew_offset,
        "ns_offset_s": ns_offset,
    }


def is_effective_green(
    time_s: float,
    plan: Mapping[str, float],
    phase: str,
) -> bool:
    """Return whether a time falls within the requested effective green."""

    if phase not in {"EW", "NS"}:
        raise ValueError("Phase must be EW or NS.")
    if not np.isfinite(time_s):
        raise ValueError("Time must be finite.")
    required = {
        "cycle_s",
        "ew_green_s",
        "ns_green_s",
        "ew_offset_s",
        "ns_offset_s",
    }
    if not required.issubset(plan):
        raise ValueError("Incomplete two-phase plan.")
    cycle_s = float(plan["cycle_s"])
    offset_s = float(plan[f"{phase.lower()}_offset_s"])
    green_s = float(plan[f"{phase.lower()}_green_s"])
    position = (float(time_s) - offset_s) % cycle_s
    return bool(position < green_s)


def simulate_ns_approach(
    arrivals: Sequence[float] | np.ndarray,
    plan: Mapping[str, float],
    headway_s: float = 2.0,
    horizon_s: float = 3600.0,
) -> dict[str, float | int]:
    """Simulate one local north or south approach using the NS phase."""

    required = {"cycle_s", "ns_offset_s", "ns_green_s"}
    if not required.issubset(plan):
        raise ValueError("Plan must define the NS effective-green window.")
    if not np.isfinite(horizon_s) or horizon_s < 0:
        raise ValueError("Observation horizon must be finite and non-negative.")
    arrival_array = np.asarray(arrivals, dtype=float)
    departures, waiting, max_queue = serve_approach(
        arrivals=arrival_array,
        cycle_s=float(plan["cycle_s"]),
        offset_s=float(plan["ns_offset_s"]),
        green_s=float(plan["ns_green_s"]),
        headway_s=headway_s,
    )
    n_arrived = np.searchsorted(arrival_array, horizon_s, side="right")
    n_departed = np.searchsorted(departures, horizon_s, side="right")
    return {
        "vehicles": len(arrival_array),
        "total_delay_s": float(waiting.sum()),
        "mean_delay_s": float(waiting.mean()) if len(arrival_array) else 0.0,
        "waiting_events": int((waiting > 1e-9).sum()),
        "max_queue_veh": max_queue,
        "queue_at_horizon_veh": int(n_arrived - n_departed),
        "exits_after_horizon": int((departures > horizon_s).sum()),
    }
