import numpy as np
import pandas as pd

from phase4b_multi_seed import generate_arrivals
from phase4d2_stress_validation import (
    FACTORS,
    SEEDS,
    generate_stress_arrivals,
    summarise_run,
)


def make_test_traffic() -> pd.DataFrame:
    times = pd.date_range("2025-01-15 06:00", periods=16, freq="15min")
    rows = []
    for time in times:
        for junction, approach in [("A", "E"), ("C", "W")]:
            rows.append(
                {
                    "time": time,
                    "period": "AM peak",
                    "intersection_id": junction,
                    "approach": approach,
                    "volume_15min": 60,
                }
            )
    return pd.DataFrame(rows)


def test_fresh_seed_range() -> None:
    assert list(SEEDS) == list(range(72, 102))
    assert len(SEEDS) == 30
    assert FACTORS == (0.75, 1.0, 1.25, 1.5)


def test_stress_arrivals_preserve_baseline_and_are_reproducible() -> None:
    traffic = make_test_traffic()
    first = generate_stress_arrivals(traffic, 72)
    second = generate_stress_arrivals(traffic, 72)
    baseline = generate_arrivals(traffic, "AM peak", 72)
    for direction in ("EB", "WB"):
        assert np.array_equal(first[1.0][direction], baseline[direction])
        for factor in FACTORS:
            assert np.array_equal(
                first[factor][direction], second[factor][direction]
            )


def test_stress_arrivals_are_nested() -> None:
    scenarios = generate_stress_arrivals(make_test_traffic(), 73)
    for direction in ("EB", "WB"):
        assert np.isin(scenarios[0.75][direction], scenarios[1.0][direction]).all()
        assert np.isin(scenarios[1.0][direction], scenarios[1.25][direction]).all()
        assert np.isin(scenarios[1.25][direction], scenarios[1.5][direction]).all()


def test_summarise_run_uses_residual_queue_and_vehicle_weights() -> None:
    vehicles = pd.DataFrame(
        {
            "total_delay_s": [1.0, 3.0],
            "journey_time_s": [82.0, 84.0],
            "waiting_events": [0, 2],
            "exit_time_s": [3590.0, 3610.0],
        }
    )
    queues = pd.DataFrame(
        {
            "max_queue": [4, 6],
            "queue_at_horizon_veh": [2, 3],
        }
    )
    summary = summarise_run(vehicles, queues)
    assert summary["vehicles"] == 2
    assert summary["mean_delay_s"] == 2.0
    assert summary["no_wait_fraction"] == 0.5
    assert summary["max_queue_veh"] == 6
    assert summary["queue_at_hour_veh"] == 5
    assert summary["unfinished_journeys"] == 1
