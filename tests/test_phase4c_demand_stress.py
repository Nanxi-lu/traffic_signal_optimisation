import numpy as np
import pytest

from traffic_signals.demand_stress import create_demand_scenarios
from traffic_signals.vehicle_simulation import simulate_direction


def _scenarios(baseline: np.ndarray) -> dict[float, np.ndarray]:
    return create_demand_scenarios(
        baseline_arrivals=baseline,
        baseline_rate=0.01,
        duration_s=3600,
        multipliers=[0.75, 1.0, 1.25, 1.5],
        rng=np.random.default_rng(42),
    )


def test_baseline_preserved() -> None:
    baseline = np.array([10.0, 20.0, 30.0])
    scenarios = _scenarios(baseline)
    assert np.array_equal(scenarios[1.0], baseline)


def test_nested_arrival_scenarios() -> None:
    baseline = np.array([10.0, 20.0, 30.0, 40.0])
    scenarios = _scenarios(baseline)
    assert np.isin(scenarios[0.75], scenarios[1.0]).all()
    assert np.isin(scenarios[1.0], scenarios[1.25]).all()
    assert np.isin(scenarios[1.25], scenarios[1.5]).all()
    assert all(np.all(np.diff(values) >= 0) for values in scenarios.values())


def test_queue_at_observation_horizon() -> None:
    _, queues = simulate_direction(
        entry_times=np.array([0.0, 0.5, 1.0]),
        direction="EB",
        signal_plan={
            "cycle_s": 40,
            "greens": {"A": 10, "B": 10, "C": 10},
            "offsets": {"A": 0, "B": 0, "C": 0},
        },
        speed_kmh=40,
        headway_s=2,
        horizon_s=3,
    )
    queue_a = queues.loc[
        queues["intersection_id"] == "A", "queue_at_horizon_veh"
    ].item()
    assert queue_a == 1


def test_invalid_stress_inputs_and_horizon_raise() -> None:
    with pytest.raises(ValueError, match="baseline multiplier"):
        create_demand_scenarios(
            np.array([10.0]),
            0.01,
            3600,
            [0.75, 1.25],
            np.random.default_rng(42),
        )
    with pytest.raises(ValueError, match="Invalid observation horizon"):
        simulate_direction(
            [0.0],
            "EB",
            {
                "cycle_s": 40,
                "greens": {"A": 10, "B": 10, "C": 10},
                "offsets": {"A": 0, "B": 0, "C": 0},
            },
            horizon_s=-1,
        )
