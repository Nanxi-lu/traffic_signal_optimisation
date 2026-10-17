"""Helpers for Junction C effective-green sensitivity experiments."""

from collections.abc import Mapping

import numpy as np

from traffic_signals.full_junction import create_two_phase_plan


JUNCTIONS = {"A", "B", "C"}
MIN_GREEN_S = 10.0


def modify_c_green(
    base_plans: Mapping[str, Mapping[str, float]],
    extra_ns_s: float,
) -> dict[str, dict[str, float]]:
    """Transfer effective green from EW to NS at C without mutating the input.

    Plans at A and B are copied unchanged. Junction C's EW offset is fixed, so
    its consecutive NS phase begins earlier by the amount transferred.
    """

    if not np.isscalar(extra_ns_s) or not np.isfinite(extra_ns_s):
        raise ValueError("Green transfer must be finite.")
    transfer = float(extra_ns_s)
    if transfer < 0:
        raise ValueError("Green transfer must be non-negative.")
    if set(base_plans) != JUNCTIONS:
        raise ValueError("Expected three junction plans: A, B and C.")

    modified = {junction: dict(plan) for junction, plan in base_plans.items()}
    original = modified["C"]
    required = {
        "cycle_s",
        "ew_green_s",
        "ns_green_s",
        "lost_time_s",
        "interphase_loss_s",
        "ew_offset_s",
    }
    if not required.issubset(original):
        raise ValueError("Junction C plan is incomplete.")

    ew_green = float(original["ew_green_s"]) - transfer
    ns_green = float(original["ns_green_s"]) + transfer
    if ew_green < MIN_GREEN_S or ns_green < MIN_GREEN_S:
        raise ValueError("Each phase needs at least 10 seconds of green.")

    modified["C"] = create_two_phase_plan(
        cycle_s=float(original["cycle_s"]),
        ew_green_s=ew_green,
        ns_green_s=ns_green,
        ew_offset_s=float(original["ew_offset_s"]),
        lost_time_s=float(original["lost_time_s"]),
        interphase_loss_s=float(original["interphase_loss_s"]),
    )
    return modified


def theoretical_capacity_vph(
    green_s: float,
    cycle_s: float,
    saturation_flow_vph: float = 1800.0,
) -> float:
    """Return the illustrative approach capacity ``green / cycle * saturation``."""

    values = np.asarray([green_s, cycle_s, saturation_flow_vph], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Capacity inputs must be finite.")
    if cycle_s <= 0 or saturation_flow_vph <= 0:
        raise ValueError("Cycle and saturation flow must be positive.")
    if green_s < 0 or green_s > cycle_s:
        raise ValueError("Green time must be between zero and the cycle length.")
    return float(green_s / cycle_s * saturation_flow_vph)
