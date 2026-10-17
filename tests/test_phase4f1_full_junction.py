import numpy as np
import pandas as pd
import pytest

from traffic_signals.full_junction import (
    create_two_phase_plan,
    is_effective_green,
    simulate_ns_approach,
)
from traffic_signals.hourly_arrivals import generate_hourly_arrivals


def test_am_phase_schedule():
    plan = create_two_phase_plan(
        cycle_s=150,
        ew_green_s=87,
        ns_green_s=51,
        ew_offset_s=0,
    )
    assert plan["ns_offset_s"] == pytest.approx(93)
    assert is_effective_green(86, plan, "EW")
    assert not is_effective_green(87, plan, "EW")
    assert not is_effective_green(92, plan, "NS")
    assert is_effective_green(93, plan, "NS")
    assert not is_effective_green(144, plan, "NS")


def test_no_simultaneous_effective_green():
    plan = create_two_phase_plan(150, 95, 43, 81)
    for time_s in np.linspace(0, 300, 1201):
        assert not (
            is_effective_green(time_s, plan, "EW")
            and is_effective_green(time_s, plan, "NS")
        )


def test_changing_ew_offset_shifts_ns_offset():
    original = create_two_phase_plan(150, 95, 43, 81)
    coordinated = create_two_phase_plan(150, 95, 43, 70)
    assert original["ns_offset_s"] == pytest.approx(32)
    assert coordinated["ns_offset_s"] == pytest.approx(21)


def test_ns_fifo_discharge():
    plan = create_two_phase_plan(150, 87, 51, 0)
    result = simulate_ns_approach(
        arrivals=np.array([90.0, 90.5, 91.0]),
        plan=plan,
        headway_s=2.0,
        horizon_s=94.0,
    )
    assert result["vehicles"] == 3
    assert result["total_delay_s"] == pytest.approx(3 + 4.5 + 6)
    assert result["waiting_events"] == 3
    assert result["max_queue_veh"] == 3
    assert result["queue_at_horizon_veh"] == 2


def test_hourly_arrivals_are_reproducible_and_slot_specific():
    slots = pd.date_range("2025-01-15 08:00", periods=4, freq="15min")
    traffic = pd.DataFrame(
        {
            "time": list(slots) + list(slots),
            "intersection_id": ["A"] * 4 + ["C"] * 4,
            "approach": ["E"] * 4 + ["W"] * 4,
            "volume_15min": [0, 0, 40, 0, 0, 30, 0, 0],
        }
    )
    streams = [("A", "E"), ("C", "W")]
    first = generate_hourly_arrivals(traffic, slots[0], streams, seed=42)
    second = generate_hourly_arrivals(traffic, slots[0], streams, seed=42)
    for stream in streams:
        np.testing.assert_array_equal(first[stream], second[stream])
    assert np.all((first[("A", "E")] >= 1800) & (first[("A", "E")] < 2700))
    assert np.all((first[("C", "W")] >= 900) & (first[("C", "W")] < 1800))
