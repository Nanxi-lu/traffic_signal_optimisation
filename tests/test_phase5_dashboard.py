from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traffic_signals.dashboard_data import (
    bootstrap_change,
    green_segments,
    movement_comparison,
    network_comparison,
    pair_results,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "outputs" / "data"


def example_network() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period": ["AM peak"] * 4,
            "seed": [1, 2, 1, 2],
            "strategy": ["Phase3A", "Phase3A", "Integrated", "Integrated"],
            "total_passages": [30, 30, 30, 30],
            "mean_delay_per_passage_s": [10.0, 12.0, 8.0, 10.0],
        }
    )


def test_pair_results_matches_exact_seeds_and_calculates_change() -> None:
    paired = pair_results(
        example_network(),
        "AM peak",
        "Phase3A",
        "Integrated",
        "mean_delay_per_passage_s",
        ["seed"],
        "total_passages",
    )
    assert paired["seed"].tolist() == [1, 2]
    assert paired["change"].tolist() == [-2.0, -2.0]


def test_pair_results_rejects_different_observation_counts() -> None:
    frame = example_network()
    frame.loc[frame.index[-1], "total_passages"] = 29
    with pytest.raises(ValueError, match="counts differ"):
        pair_results(
            frame,
            "AM peak",
            "Phase3A",
            "Integrated",
            "mean_delay_per_passage_s",
            ["seed"],
            "total_passages",
        )


def test_bootstrap_change_is_reproducible_and_validates_input() -> None:
    first = bootstrap_change([-3.0, -2.0, -1.0], n_bootstrap=500, seed=2026)
    second = bootstrap_change([-3.0, -2.0, -1.0], n_bootstrap=500, seed=2026)
    assert first == second
    assert first[0] <= -2.0 <= first[1]
    with pytest.raises(ValueError, match="at least two"):
        bootstrap_change([1.0])


def test_network_comparison_summarises_paired_delay() -> None:
    result = network_comparison(
        example_network(), "AM peak", "Phase3A", "Integrated"
    )
    assert result["mean_delay_change_s"] == pytest.approx(-2.0)
    assert result["candidate_mean_delay_s"] == pytest.approx(9.0)
    assert result["replications"] == 2
    assert result["seeds_better"] == 2


def test_green_segments_handles_wrapping_and_invalid_timings() -> None:
    assert green_segments(10.0, 20.0, 60.0) == [(10.0, 30.0)]
    assert green_segments(50.0, 20.0, 60.0) == [
        (50.0, 60.0),
        (0.0, 10.0),
    ]
    assert green_segments(70.0, 20.0, 60.0) == [(10.0, 30.0)]
    with pytest.raises(ValueError, match="Invalid"):
        green_segments(0.0, 60.0, 60.0)


def test_saved_pm_results_reproduce_network_and_fairness_findings() -> None:
    network = pd.read_csv(DATA_DIR / "phase4g_network_metrics.csv")
    movements = pd.read_csv(DATA_DIR / "phase4g_movement_metrics.csv")

    versus_3a = network_comparison(
        network, "PM peak", "Phase3A", "Integrated"
    )
    assert versus_3a["mean_delay_change_s"] == pytest.approx(-8.610980, abs=1e-6)

    fairness_3a = movement_comparison(
        movements, "PM peak", "Phase3A", "Integrated"
    ).set_index("movement")
    assert fairness_3a.loc["EB", "mean_delay_change_s"] == pytest.approx(
        2.906, abs=1e-3
    )
    assert not bool(fairness_3a.loc["EB", "passes_2s_upper"])

    fairness_4e = movement_comparison(
        movements, "PM peak", "Phase4E", "Integrated"
    ).set_index("movement")
    assert fairness_4e.loc["EB", "mean_delay_change_s"] == pytest.approx(
        1.620, abs=1e-3
    )
    assert bool(fairness_4e.loc["EB", "passes_2s_upper"])
