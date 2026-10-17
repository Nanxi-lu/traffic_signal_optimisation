import numpy as np
import pytest

from traffic_signals.vehicle_simulation import (
    next_green_time,
    serve_approach,
    simulate_direction,
)


def test_next_green_time() -> None:
    assert next_green_time(12, 0, 10, 40) == pytest.approx(40)
    assert next_green_time(44, 36, 16, 40) == pytest.approx(44)


def test_fifo_discharge() -> None:
    departures, waiting, max_queue = serve_approach(
        [0, 0.5, 1], cycle_s=40, offset_s=0, green_s=10, headway_s=2
    )
    assert departures == pytest.approx([0, 2, 4])
    assert waiting == pytest.approx([0, 1.5, 3])
    assert max_queue == 2


def test_red_arrivals_wait_for_next_green() -> None:
    departures, waiting, _ = serve_approach(
        [12, 13], cycle_s=40, offset_s=0, green_s=10, headway_s=2
    )
    assert departures == pytest.approx([40, 42])
    assert waiting == pytest.approx([28, 29])


def test_simulated_journey_time_equals_free_flow_plus_delay() -> None:
    vehicles, queues = simulate_direction(
        entry_times=np.array([0.0, 5.0, 10.0]),
        direction="EB",
        signal_plan={
            "cycle_s": 150,
            "greens": {"A": 87, "B": 88, "C": 95},
            "offsets": {"A": 0, "B": 36, "C": 81},
        },
        speed_kmh=40,
        headway_s=2,
    )
    assert np.allclose(
        vehicles["journey_time_s"], 81.0 + vehicles["total_delay_s"]
    )
    assert len(queues) == 3


def test_invalid_simulation_inputs_raise() -> None:
    with pytest.raises(ValueError, match="Direction must be EB or WB"):
        simulate_direction(
            [0],
            "NS",
            {
                "cycle_s": 40,
                "greens": {"A": 10, "B": 10, "C": 10},
                "offsets": {"A": 0, "B": 0, "C": 0},
            },
        )
