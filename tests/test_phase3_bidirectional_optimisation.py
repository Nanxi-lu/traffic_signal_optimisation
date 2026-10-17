import pandas as pd
import pytest

from traffic_signals.bidirectional import count_directional_green
from traffic_signals.bidirectional_optimisation import SPEEDS, optimise_bidirectional
from traffic_signals.demand_weights import calculate_direction_weights


def test_directional_weights() -> None:
    traffic = pd.DataFrame(
        {
            "time": ["2025-01-15 08:00"] * 2,
            "intersection_id": ["A", "C"],
            "approach": ["E", "W"],
            "volume_15min": [60, 40],
        }
    )
    result = calculate_direction_weights(traffic)
    assert result["EB_weight"].iloc[0] == pytest.approx(0.6)
    assert result["WB_weight"].iloc[0] == pytest.approx(0.4)


def test_am_baseline_regression() -> None:
    greens = {"A": 87, "B": 88, "C": 95}
    offsets = {"A": 0, "B": 36, "C": 81}
    assert count_directional_green(150, greens, offsets, 40, "EB") == 87
    assert count_directional_green(150, greens, offsets, 40, "WB") == 15


def test_bidirectional_optimisation_respects_design_protection() -> None:
    greens = {"A": 16, "B": 16, "C": 15}
    baseline = {"A": 0, "B": 36, "C": 1}
    result = optimise_bidirectional(
        40, greens, baseline, {"EB": 0.6, "WB": 0.4}, protection_pp=5.0
    )
    for direction in ("EB", "WB"):
        assert result["direction_design_40"][direction] >= (
            result["baseline_design_40"][direction] - 0.05
        )


def test_invalid_directional_weights_raise() -> None:
    with pytest.raises(ValueError, match="Invalid directional weights"):
        optimise_bidirectional(
            40,
            {"A": 16, "B": 16, "C": 15},
            {"A": 0, "B": 36, "C": 1},
            {"EB": 0.8, "WB": 0.3},
        )
