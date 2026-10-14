"""Educational tools for synthetic traffic-signal analysis."""

from traffic_signals.synthetic_data import SyntheticDataConfig, generate_synthetic_data
from traffic_signals.webster import (
    WebsterResult,
    allocate_effective_green,
    calculate_webster_cycle,
    counts_15min_to_hourly_flow,
    evaluate_two_phase_interval,
)

__all__ = [
    "SyntheticDataConfig",
    "WebsterResult",
    "allocate_effective_green",
    "calculate_webster_cycle",
    "counts_15min_to_hourly_flow",
    "evaluate_two_phase_interval",
    "generate_synthetic_data",
]
