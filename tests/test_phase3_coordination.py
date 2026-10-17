import pytest

from traffic_signals.coordination import (
    calculate_offsets,
    is_in_green,
    optimise_fixed_cycle_split,
)


def test_travel_times() -> None:
    times, offsets = calculate_offsets(150)
    assert times["B"] == pytest.approx(36)
    assert times["C"] == pytest.approx(81)
    assert offsets["A"] == pytest.approx(0)
    assert offsets["B"] == pytest.approx(36)
    assert offsets["C"] == pytest.approx(81)


def test_offset_wraparound() -> None:
    _, offsets = calculate_offsets(40)
    assert offsets["C"] == pytest.approx(1)


def test_green_window() -> None:
    assert is_in_green(42, 36, 25, 150)
    assert not is_in_green(70, 36, 25, 150)
    assert not is_in_green(61, 36, 25, 150)


def test_fixed_cycle_optimisation() -> None:
    result = optimise_fixed_cycle_split(
        ew_y=[0.30, 0.45],
        ns_y=[0.20, 0.25],
        cycle_s=80,
    )
    assert result["warning_intervals"] == 0
    assert result["ew_green_s"] >= 10
    assert result["ns_green_s"] >= 10
    assert result["ew_green_s"] + result["ns_green_s"] + 12 == 80


def test_invalid_coordination_inputs_raise() -> None:
    with pytest.raises(ValueError):
        calculate_offsets(0)
    with pytest.raises(ValueError):
        optimise_fixed_cycle_split([0.3], [0.2], cycle_s=30)
