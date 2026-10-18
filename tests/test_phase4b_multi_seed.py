import numpy as np
import pandas as pd
import pytest

from workflows.stage4a_4e_corridor import (
    bootstrap_mean_ci,
    generate_arrivals,
    get_signal_strategies,
)


def _traffic() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period": ["AM peak", "AM peak"],
            "intersection_id": ["A", "C"],
            "approach": ["E", "W"],
            "volume_15min": [100, 80],
        }
    )


def test_arrival_generation_reproducible() -> None:
    first = generate_arrivals(_traffic(), "AM peak", 42)
    second = generate_arrivals(_traffic(), "AM peak", 42)
    assert np.array_equal(first["EB"], second["EB"])
    assert np.array_equal(first["WB"], second["WB"])


def test_arrivals_within_hour() -> None:
    arrivals = generate_arrivals(_traffic(), "AM peak", 43)
    for values in arrivals.values():
        assert np.all(values >= 0)
        assert np.all(values < 3600)
        assert np.all(np.diff(values) >= 0)


def test_bootstrap_mean_interval_is_reproducible() -> None:
    values = np.array([-4.0, -3.0, -2.0, -1.0])
    first = bootstrap_mean_ci(values, n_bootstrap=500, seed=2026)
    second = bootstrap_mean_ci(values, n_bootstrap=500, seed=2026)
    assert np.array_equal(first, second)
    assert first[0] <= values.mean() <= first[1]


def test_signal_strategy_loader() -> None:
    phase3a = pd.DataFrame(
        {
            "period": ["AM peak"] * 3,
            "intersection_id": ["A", "B", "C"],
            "common_cycle_s": [150] * 3,
            "ew_green_s": [87, 88, 95],
            "offset_s": [0, 36, 81],
        }
    )
    phase3f = pd.DataFrame(
        {
            "period": ["AM peak"],
            "strategy": ["Phase3F_bidirectional"],
            "offset_b_s": [28],
            "offset_c_s": [72],
        }
    )
    strategies = get_signal_strategies(phase3a, phase3f, "AM peak")
    assert strategies["Phase3A"]["offsets"] == {"A": 0, "B": 36, "C": 81}
    assert strategies["Phase3F"]["offsets"] == {
        "A": 0.0,
        "B": 28.0,
        "C": 72.0,
    }
