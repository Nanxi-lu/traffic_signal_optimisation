import pandas as pd
import pytest

from traffic_signals.webster import (
    allocate_effective_green,
    calculate_webster_cycle,
    evaluate_two_phase_interval,
)


def test_webster_formula_uncapped_example() -> None:
    result = calculate_webster_cycle(0.65, lost_time_s=12.0)

    assert result.uncapped_cycle_length_s == pytest.approx(65.7142857)
    assert result.cycle_length_s == pytest.approx(65.7142857)
    assert result.status == "webster_cycle"
    assert result.adequately_served


def test_effective_green_is_proportional_to_critical_ratios() -> None:
    greens = allocate_effective_green(
        {"EW": 0.45, "NS": 0.15}, cycle_length_s=72.0, lost_time_s=12.0
    )

    assert greens["EW"] == pytest.approx(45.0)
    assert greens["NS"] == pytest.approx(15.0)
    assert sum(greens.values()) == pytest.approx(60.0)


def test_zero_demand_is_safe_and_allocates_no_green() -> None:
    result = calculate_webster_cycle(0.0)
    greens = allocate_effective_green(
        {"EW": 0.0, "NS": 0.0}, result.cycle_length_s
    )

    assert result.cycle_length_s == 40.0
    assert result.status == "no_demand_min_cycle"
    assert result.adequately_served
    assert greens == {"EW": 0.0, "NS": 0.0}


def test_y_at_or_above_one_is_flagged_as_oversaturated() -> None:
    result = calculate_webster_cycle(1.0)

    assert result.cycle_length_s == 150.0
    assert result.uncapped_cycle_length_s == float("inf")
    assert result.capped
    assert not result.adequately_served
    assert result.status == "oversaturated_y_ge_1"


def test_maximum_cycle_can_be_flagged_as_insufficient() -> None:
    result = calculate_webster_cycle(0.95)

    assert result.cycle_length_s == 150.0
    assert result.capped
    assert not result.adequately_served
    assert result.status == "maximum_cycle_insufficient"


@pytest.mark.parametrize(
    "ratio,lost_time,min_cycle,max_cycle",
    [
        (-0.1, 12.0, 40.0, 150.0),
        (float("nan"), 12.0, 40.0, 150.0),
        (0.5, -1.0, 40.0, 150.0),
        (0.5, 40.0, 40.0, 150.0),
        (0.5, 12.0, 160.0, 150.0),
    ],
)
def test_invalid_cycle_inputs_raise(
    ratio: float, lost_time: float, min_cycle: float, max_cycle: float
) -> None:
    with pytest.raises(ValueError):
        calculate_webster_cycle(ratio, lost_time, min_cycle, max_cycle)


def test_critical_flow_selection_uses_larger_opposing_ratio() -> None:
    interval = pd.DataFrame(
        {
            "approach": ["eastbound", "westbound", "northbound", "southbound"],
            "volume_15min": [100, 150, 80, 60],
            "saturation_flow_vph": [2000.0, 2000.0, 1600.0, 1600.0],
        }
    )

    result = evaluate_two_phase_interval(interval)

    assert result["ew_critical_approach"] == "westbound"
    assert result["ns_critical_approach"] == "northbound"
    assert result["ew_critical_flow_ratio"] == pytest.approx(0.30)
    assert result["ns_critical_flow_ratio"] == pytest.approx(0.20)
