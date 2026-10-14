"""A transparent Webster-cycle calculation for a two-phase teaching model."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WebsterResult:
    """Cycle calculation plus the safeguards needed to interpret it."""

    cycle_length_s: float
    uncapped_cycle_length_s: float
    total_critical_flow_ratio: float
    effective_green_s: float
    capped: bool
    adequately_served: bool
    status: str


def counts_15min_to_hourly_flow(volume_15min: float) -> float:
    """Convert a 15-minute count to an equivalent hourly flow rate."""

    if not np.isfinite(volume_15min) or volume_15min < 0:
        raise ValueError("volume_15min must be a non-negative finite value")
    return float(volume_15min) * 4.0


def calculate_webster_cycle(
    total_critical_flow_ratio: float,
    lost_time_s: float = 12.0,
    min_cycle_s: float = 40.0,
    max_cycle_s: float = 150.0,
) -> WebsterResult:
    """Calculate ``C = (1.5L + 5) / (1 - Y)`` with explicit safeguards.

    ``Y`` is the sum of the EW and NS critical flow ratios. If ``Y >= 1``,
    the Webster expression has no finite feasible result and the maximum cycle
    is returned only as an operational flag, not as a claim of adequate service.
    For a finite result, capacity is conservatively flagged as inadequate when
    ``Y`` exceeds the fraction of the selected cycle available as effective
    green, ``(C - L) / C``.
    """

    values = (total_critical_flow_ratio, lost_time_s, min_cycle_s, max_cycle_s)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("All Webster inputs must be finite")
    if total_critical_flow_ratio < 0:
        raise ValueError("total_critical_flow_ratio cannot be negative")
    if lost_time_s < 0:
        raise ValueError("lost_time_s cannot be negative")
    if min_cycle_s <= lost_time_s:
        raise ValueError("min_cycle_s must be greater than lost_time_s")
    if max_cycle_s < min_cycle_s:
        raise ValueError("max_cycle_s must be at least min_cycle_s")

    if total_critical_flow_ratio >= 1.0:
        return WebsterResult(
            cycle_length_s=float(max_cycle_s),
            uncapped_cycle_length_s=float("inf"),
            total_critical_flow_ratio=float(total_critical_flow_ratio),
            effective_green_s=float(max_cycle_s - lost_time_s),
            capped=True,
            adequately_served=False,
            status="oversaturated_y_ge_1",
        )

    uncapped = (1.5 * lost_time_s + 5.0) / (1.0 - total_critical_flow_ratio)
    selected = float(np.clip(uncapped, min_cycle_s, max_cycle_s))
    capped = not np.isclose(selected, uncapped)
    effective_green = selected - lost_time_s
    serviceable_ratio = effective_green / selected
    adequately_served = total_critical_flow_ratio <= serviceable_ratio + 1e-12

    if total_critical_flow_ratio == 0:
        status = "no_demand_min_cycle"
    elif selected == min_cycle_s and uncapped < min_cycle_s:
        status = "minimum_cycle_applied"
    elif selected == max_cycle_s and uncapped > max_cycle_s:
        status = "maximum_cycle_insufficient" if not adequately_served else "maximum_cycle_applied"
    else:
        status = "webster_cycle"

    return WebsterResult(
        cycle_length_s=selected,
        uncapped_cycle_length_s=float(uncapped),
        total_critical_flow_ratio=float(total_critical_flow_ratio),
        effective_green_s=float(effective_green),
        capped=capped,
        adequately_served=bool(adequately_served),
        status=status,
    )


def allocate_effective_green(
    critical_flow_ratios: Mapping[str, float],
    cycle_length_s: float,
    lost_time_s: float = 12.0,
) -> dict[str, float]:
    """Allocate effective green in proportion to phase critical-flow ratios.

    A zero-demand interval receives zero effective green allocations; the
    unassigned time is deliberate because no demand-based split is meaningful.
    """

    if not critical_flow_ratios:
        raise ValueError("At least one phase is required")
    if not np.isfinite(cycle_length_s) or not np.isfinite(lost_time_s):
        raise ValueError("Cycle and lost time must be finite")
    if lost_time_s < 0 or cycle_length_s <= lost_time_s:
        raise ValueError("cycle_length_s must be greater than non-negative lost_time_s")

    ratios = {phase: float(value) for phase, value in critical_flow_ratios.items()}
    if any(not np.isfinite(value) or value < 0 for value in ratios.values()):
        raise ValueError("Critical flow ratios must be non-negative finite values")

    total_ratio = sum(ratios.values())
    if total_ratio == 0:
        return {phase: 0.0 for phase in ratios}

    effective_green = cycle_length_s - lost_time_s
    return {
        phase: effective_green * ratio / total_ratio
        for phase, ratio in ratios.items()
    }


def evaluate_two_phase_interval(
    interval_data: pd.DataFrame,
    lost_time_s: float = 12.0,
    min_cycle_s: float = 40.0,
    max_cycle_s: float = 150.0,
) -> dict[str, object]:
    """Evaluate one intersection and interval using concurrent EW/NS phases.

    Opposing approaches run concurrently in this simplified model. Therefore,
    each phase's critical flow is the larger volume-to-saturation ratio of its
    two approaches: max(eastbound, westbound) for EW and max(northbound,
    southbound) for NS. Turning movements, lane groups and intergreen detail
    are intentionally outside Phase 1.
    """

    required = {"approach", "volume_15min", "saturation_flow_vph"}
    missing = required.difference(interval_data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    expected_approaches = {"eastbound", "westbound", "northbound", "southbound"}
    if set(interval_data["approach"]) != expected_approaches or len(interval_data) != 4:
        raise ValueError("Each interval must contain exactly one row for each of four approaches")
    if interval_data["approach"].duplicated().any():
        raise ValueError("Approaches must not be duplicated within an interval")

    data = interval_data.set_index("approach")
    flow_ratios: dict[str, float] = {}
    for approach, row in data.iterrows():
        saturation = float(row["saturation_flow_vph"])
        if not np.isfinite(saturation) or saturation <= 0:
            raise ValueError("saturation_flow_vph must be positive and finite")
        hourly_flow = counts_15min_to_hourly_flow(float(row["volume_15min"]))
        flow_ratios[approach] = hourly_flow / saturation

    ew_approach = max(("eastbound", "westbound"), key=flow_ratios.get)
    ns_approach = max(("northbound", "southbound"), key=flow_ratios.get)
    phase_ratios = {"EW": flow_ratios[ew_approach], "NS": flow_ratios[ns_approach]}
    result = calculate_webster_cycle(
        sum(phase_ratios.values()),
        lost_time_s=lost_time_s,
        min_cycle_s=min_cycle_s,
        max_cycle_s=max_cycle_s,
    )
    greens = allocate_effective_green(
        phase_ratios,
        cycle_length_s=result.cycle_length_s,
        lost_time_s=lost_time_s,
    )

    return {
        "ew_critical_approach": ew_approach,
        "ns_critical_approach": ns_approach,
        "ew_critical_flow_ratio": phase_ratios["EW"],
        "ns_critical_flow_ratio": phase_ratios["NS"],
        "total_critical_flow_ratio": result.total_critical_flow_ratio,
        "uncapped_cycle_length_s": result.uncapped_cycle_length_s,
        "cycle_length_s": result.cycle_length_s,
        "ew_effective_green_s": greens["EW"],
        "ns_effective_green_s": greens["NS"],
        "capped": result.capped,
        "adequately_served": result.adequately_served,
        "status": result.status,
    }
