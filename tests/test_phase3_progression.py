import pandas as pd
import pytest

from workflows.stage3_coordination import TEST_SPEEDS_KMH, build_speed_sensitivity
from traffic_signals.progression import (
    calculate_travel_times,
    evaluate_departures,
    summarise_progression,
)


def _signal_plans() -> pd.DataFrame:
    return pd.read_csv("outputs/data/phase3a_signal_plans.csv")


@pytest.mark.parametrize(
    "speed_kmh,expected_ab,expected_ac",
    [
        (30.0, 48.0, 108.0),
        (35.0, 41.142857, 92.571429),
        (40.0, 36.0, 81.0),
        (45.0, 32.0, 72.0),
        (50.0, 28.8, 64.8),
    ],
)
def test_travel_times_at_sensitivity_speeds(
    speed_kmh: float,
    expected_ab: float,
    expected_ac: float,
) -> None:
    times = calculate_travel_times(speed_kmh)
    assert times["B"] == pytest.approx(expected_ab)
    assert times["C"] == pytest.approx(expected_ac)


def test_design_speed_reproduces_phase3a_bandwidths() -> None:
    plans = _signal_plans()
    expected = {
        "Overnight": 0.375,
        "AM peak": 0.580,
        "Interpeak": 0.425,
        "PM peak": 0.580,
    }
    for period, expected_fraction in expected.items():
        summary = summarise_progression(
            plans.loc[plans["period"] == period],
            period,
            40.0,
        )
        assert summary["progression_fraction"] == pytest.approx(expected_fraction)


def test_speed_sensitivity_contains_twenty_rows() -> None:
    results = build_speed_sensitivity(_signal_plans())
    assert len(results) == 20
    assert set(results["speed_kmh"]) == set(TEST_SPEEDS_KMH)
    assert results.groupby("period").size().eq(5).all()


def test_whole_second_departures_cover_one_cycle() -> None:
    plans = _signal_plans()
    overnight = plans.loc[plans["period"] == "Overnight"]
    departures = evaluate_departures(overnight, 40.0)
    assert len(departures) == 40
    assert departures["departure_s"].iloc[0] == 0
    assert departures["departure_s"].iloc[-1] == 39
