import pytest

from traffic_signals.offset_optimisation import (
    SPEEDS,
    circular_distance,
    count_all_green,
    optimise_offsets,
)


def test_original_am_bandwidth() -> None:
    greens = {"A": 87, "B": 88, "C": 95}
    offsets = {"A": 0, "B": 36, "C": 81}
    assert count_all_green(150, greens, offsets, 40) == 87


def test_original_overnight_at_30() -> None:
    greens = {"A": 16, "B": 16, "C": 15}
    offsets = {"A": 0, "B": 36, "C": 1}
    assert count_all_green(40, greens, offsets, 30) == 0


def test_circular_offset_distance() -> None:
    assert circular_distance(1, 39, 40) == pytest.approx(2)


def test_optimisation_preserves_primary_objective_and_fixed_a() -> None:
    greens = {"A": 16, "B": 16, "C": 15}
    baseline_offsets = {"A": 0, "B": 36, "C": 1}
    baseline_mean = sum(
        count_all_green(40, greens, baseline_offsets, speed) for speed in SPEEDS
    ) / (40 * len(SPEEDS))
    optimum = optimise_offsets(40, greens, baseline_offsets)
    assert optimum["offsets"]["A"] == 0
    assert optimum["mean_fraction"] >= baseline_mean
