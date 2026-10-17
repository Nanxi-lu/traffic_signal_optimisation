import pytest

from traffic_signals.optimisation import evaluate_plan, optimise_plan


def test_capacity_warning() -> None:
    result = evaluate_plan(
        ew_y=[0.30],
        ns_y=[0.40],
        cycle_s=60,
        ew_green_s=30,
    )
    assert result["warning_intervals"] == 1


def test_optimizer_fixes_allocation() -> None:
    result = optimise_plan(
        ew_y=[0.30],
        ns_y=[0.40],
        baseline_cycle_s=60,
        baseline_ew_green_s=30,
    )
    assert result["warning_intervals"] == 0


def test_unavoidable_capacity_warning() -> None:
    result = optimise_plan(
        ew_y=[0.655556],
        ns_y=[0.284444],
        baseline_cycle_s=145,
        baseline_ew_green_s=90.392470,
    )
    assert result["warning_intervals"] == 1


def test_invalid_demand_arrays_raise() -> None:
    with pytest.raises(ValueError):
        evaluate_plan(
            ew_y=[0.30, 0.40],
            ns_y=[0.20],
            cycle_s=60,
            ew_green_s=25,
        )
