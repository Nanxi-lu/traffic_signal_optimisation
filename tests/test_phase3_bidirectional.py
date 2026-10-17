import pytest

from traffic_signals.bidirectional import count_directional_green


def test_original_eastbound_bandwidth() -> None:
    result = count_directional_green(
        cycle_s=150,
        greens={"A": 87, "B": 88, "C": 95},
        offsets={"A": 0, "B": 36, "C": 81},
        speed_kmh=40,
        direction="EB",
    )
    assert result == 87


def test_ideal_westbound_alignment() -> None:
    result = count_directional_green(
        cycle_s=150,
        greens={"A": 60, "B": 60, "C": 60},
        offsets={"A": 81, "B": 45, "C": 0},
        speed_kmh=40,
        direction="WB",
    )
    assert result == 60


def test_short_cycle_wraparound() -> None:
    result = count_directional_green(
        cycle_s=40,
        greens={"A": 10, "B": 10, "C": 10},
        offsets={"A": 0, "B": 36, "C": 1},
        speed_kmh=40,
        direction="EB",
    )
    assert result == 10


def test_invalid_direction_raises() -> None:
    with pytest.raises(ValueError, match="Direction must be EB or WB"):
        count_directional_green(
            cycle_s=40,
            greens={"A": 10, "B": 10, "C": 10},
            offsets={"A": 0, "B": 0, "C": 0},
            speed_kmh=40,
            direction="northbound",
        )
