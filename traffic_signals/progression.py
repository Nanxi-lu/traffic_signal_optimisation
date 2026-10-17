"""Idealised vehicle progression calculations for the three-junction corridor."""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from traffic_signals.coordination import is_in_green


CORRIDOR_DISTANCES_M = {"A": 0.0, "B": 400.0, "C": 900.0}
DESIGN_SPEED_KMH = 40.0


def calculate_travel_times(
    speed_kmh: float,
    distances_m: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return travel time from A to every corridor intersection."""

    if not np.isfinite(speed_kmh) or speed_kmh <= 0:
        raise ValueError("speed_kmh must be a positive finite value")
    distances = distances_m or CORRIDOR_DISTANCES_M
    if set(distances) != {"A", "B", "C"}:
        raise ValueError("Distances must define intersections A, B and C")
    if any(not np.isfinite(value) or value < 0 for value in distances.values()):
        raise ValueError("Distances must be non-negative finite values")

    speed_mps = speed_kmh / 3.6
    return {
        intersection: float(distance_m / speed_mps)
        for intersection, distance_m in distances.items()
    }


def evaluate_departures(
    period_plans: pd.DataFrame,
    speed_kmh: float,
    departure_times_s: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Check idealised departures against all three repeating EW green windows."""

    if set(period_plans["intersection_id"]) != {"A", "B", "C"} or len(period_plans) != 3:
        raise ValueError("period_plans must contain one row for A, B and C")
    if period_plans["intersection_id"].duplicated().any():
        raise ValueError("Each intersection must have exactly one plan")
    if period_plans["common_cycle_s"].nunique() != 1:
        raise ValueError("All intersections must use one common cycle")

    cycle_s = float(period_plans["common_cycle_s"].iloc[0])
    if departure_times_s is None:
        departures = np.arange(0.0, cycle_s, 1.0)
    else:
        departures = np.asarray(list(departure_times_s), dtype=float)
    if departures.size == 0 or not np.isfinite(departures).all():
        raise ValueError("Departure times must be a non-empty finite sequence")

    plans = period_plans.set_index("intersection_id")
    travel_times = calculate_travel_times(speed_kmh)
    records: list[dict[str, object]] = []
    for departure_s in departures:
        record: dict[str, object] = {
            "departure_s": float(departure_s),
            "speed_kmh": float(speed_kmh),
        }
        all_green = True
        for intersection in ("A", "B", "C"):
            arrival_s = float(departure_s + travel_times[intersection])
            on_green = is_in_green(
                arrival_s=arrival_s,
                offset_s=float(plans.loc[intersection, "offset_s"]),
                green_s=float(plans.loc[intersection, "ew_green_s"]),
                cycle_s=cycle_s,
            )
            record[f"arrival_{intersection.lower()}_s"] = arrival_s
            record[f"green_{intersection.lower()}"] = on_green
            all_green = all_green and on_green
        record["all_green"] = all_green
        records.append(record)
    return pd.DataFrame.from_records(records)


def summarise_progression(
    period_plans: pd.DataFrame,
    period: str,
    speed_kmh: float,
) -> dict[str, object]:
    """Summarise whole-second departures over one common signal cycle."""

    departures = evaluate_departures(period_plans, speed_kmh)
    travel_times = calculate_travel_times(speed_kmh)
    total_departures = len(departures)
    all_green_departures = int(departures["all_green"].sum())
    return {
        "period": period,
        "speed_kmh": float(speed_kmh),
        "common_cycle_s": int(period_plans["common_cycle_s"].iloc[0]),
        "departures_tested": total_departures,
        "all_green_departures": all_green_departures,
        "progression_fraction": all_green_departures / total_departures,
        "travel_time_ab_s": travel_times["B"],
        "travel_time_ac_s": travel_times["C"],
    }
