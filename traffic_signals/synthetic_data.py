"""Reproducible, explicitly hypothetical traffic-demand data."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SyntheticDataConfig:
    """Configurable assumptions used to create one synthetic day.

    Demand parameters are hourly rates used only to parameterise a Poisson
    count generator. The stored observations remain 15-minute vehicle counts.
    Peak values are additions above the base flow, not total peak demand.
    """

    seed: int = 42
    start_time: str = "2025-01-15 00:00:00"
    interval_minutes: int = 15
    intersections: tuple[str, ...] = ("A", "B", "C")
    approaches: tuple[str, ...] = ("eastbound", "westbound", "northbound", "southbound")
    saturation_flow_vph: float = 1800.0
    intersection_factors: dict[str, float] = field(
        default_factory=lambda: {"A": 0.92, "B": 1.00, "C": 1.12}
    )
    base_flow_vph: dict[str, float] = field(
        default_factory=lambda: {
            "eastbound": 200.0,
            "westbound": 200.0,
            "northbound": 120.0,
            "southbound": 130.0,
        }
    )
    am_peak_addition_vph: dict[str, float] = field(
        default_factory=lambda: {
            "eastbound": 720.0,
            "westbound": 280.0,
            "northbound": 330.0,
            "southbound": 300.0,
        }
    )
    pm_peak_addition_vph: dict[str, float] = field(
        default_factory=lambda: {
            "eastbound": 300.0,
            "westbound": 760.0,
            "northbound": 300.0,
            "southbound": 350.0,
        }
    )
    am_peak_hour: float = 8.0
    pm_peak_hour: float = 17.5
    peak_width_hours: float = 1.35


def _validate_config(config: SyntheticDataConfig) -> None:
    if config.interval_minutes != 15:
        raise ValueError("Phase 1 data must use 15-minute intervals")
    if not config.intersections:
        raise ValueError("At least one intersection is required")
    if set(config.approaches) != {"eastbound", "westbound", "northbound", "southbound"}:
        raise ValueError("Approaches must be eastbound, westbound, northbound and southbound")
    if not np.isfinite(config.saturation_flow_vph) or config.saturation_flow_vph <= 0:
        raise ValueError("saturation_flow_vph must be a positive finite value")
    if not np.isfinite(config.peak_width_hours) or config.peak_width_hours <= 0:
        raise ValueError("peak_width_hours must be a positive finite value")

    for intersection in config.intersections:
        factor = config.intersection_factors.get(intersection)
        if factor is None or not np.isfinite(factor) or factor <= 0:
            raise ValueError(f"Missing positive intersection factor for {intersection}")

    demand_mappings = (
        config.base_flow_vph,
        config.am_peak_addition_vph,
        config.pm_peak_addition_vph,
    )
    for approach in config.approaches:
        for mapping in demand_mappings:
            value = mapping.get(approach)
            if value is None or not np.isfinite(value) or value < 0:
                raise ValueError(f"Missing non-negative demand assumption for {approach}")


def _gaussian_peak(hour: float, centre: float, width: float) -> float:
    return float(np.exp(-0.5 * ((hour - centre) / width) ** 2))


def generate_synthetic_data(config: SyntheticDataConfig | None = None) -> pd.DataFrame:
    """Create 24 hours of reproducible, hypothetical 15-minute counts.

    A smooth AM and PM commuter profile defines the expected hourly demand.
    Poisson sampling then produces integer counts for each 15-minute interval.
    The expected hourly rate is divided by four before sampling, so the output
    never labels an hourly flow rate as a 15-minute count.
    """

    config = config or SyntheticDataConfig()
    _validate_config(config)
    rng = np.random.default_rng(config.seed)
    periods = 24 * 60 // config.interval_minutes
    times = pd.date_range(config.start_time, periods=periods, freq=f"{config.interval_minutes}min")

    records: list[dict[str, object]] = []
    for timestamp in times:
        hour = timestamp.hour + timestamp.minute / 60.0
        am_factor = _gaussian_peak(hour, config.am_peak_hour, config.peak_width_hours)
        pm_factor = _gaussian_peak(hour, config.pm_peak_hour, config.peak_width_hours)

        for intersection in config.intersections:
            location_factor = config.intersection_factors[intersection]
            for approach in config.approaches:
                expected_vph = location_factor * (
                    config.base_flow_vph[approach]
                    + config.am_peak_addition_vph[approach] * am_factor
                    + config.pm_peak_addition_vph[approach] * pm_factor
                )
                expected_count = expected_vph * config.interval_minutes / 60.0
                records.append(
                    {
                        "time": timestamp,
                        "intersection_id": intersection,
                        "approach": approach,
                        "volume_15min": int(rng.poisson(expected_count)),
                        "saturation_flow_vph": float(config.saturation_flow_vph),
                    }
                )

    return pd.DataFrame.from_records(records)
