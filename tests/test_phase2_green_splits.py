import numpy as np
import pandas as pd

from workflows.stage1_2_timing import (
    LOST_TIME_S,
    MIN_GREEN_S,
    build_phase_observations,
    evaluate_green_split_plans,
    select_green_split_plans,
)
from workflows.stage1_2_timing import select_fixed_plans
from traffic_signals.demo import calculate_cycle_plan
from traffic_signals.synthetic_data import generate_synthetic_data


def _phase2c_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    traffic = generate_synthetic_data()
    phase1 = calculate_cycle_plan(traffic)
    fixed_plans = select_fixed_plans(phase1)
    return traffic, phase1, fixed_plans


def test_phase_ratios_match_phase1_critical_ratios() -> None:
    traffic, phase1, _ = _phase2c_inputs()
    phase = build_phase_observations(traffic)
    joined = phase.merge(
        phase1,
        on=["time", "intersection_id"],
        validate="one_to_one",
    )

    np.testing.assert_allclose(joined["ew_y"], joined["ew_critical_flow_ratio"])
    np.testing.assert_allclose(joined["ns_y"], joined["ns_critical_flow_ratio"])
    np.testing.assert_allclose(joined["total_y"], joined["total_critical_flow_ratio"])


def test_green_plans_respect_cycle_and_minimum_green_constraints() -> None:
    traffic, _, fixed_plans = _phase2c_inputs()
    phase = build_phase_observations(traffic)
    plans = select_green_split_plans(phase, fixed_plans)

    assert len(plans) == 12
    np.testing.assert_allclose(
        plans["ew_green_s"] + plans["ns_green_s"] + LOST_TIME_S,
        plans["fixed_cycle_s"],
    )
    assert (plans["ew_green_s"] >= MIN_GREEN_S).all()
    assert (plans["ns_green_s"] >= MIN_GREEN_S).all()


def test_green_plan_evaluation_covers_all_intervals() -> None:
    traffic, _, fixed_plans = _phase2c_inputs()
    phase = build_phase_observations(traffic)
    plans = select_green_split_plans(phase, fixed_plans)
    evaluation, summary = evaluate_green_split_plans(phase, plans)

    assert len(phase) == len(evaluation) == 288
    assert len(summary) == 12
    assert summary["intervals"].sum() == 288
    assert evaluation["fixed_cycle_s"].notna().all()
    assert (evaluation[["ew_shortfall", "ns_shortfall"]] >= 0).all().all()
    assert summary["total_ratio_warnings"].sum() == 3
    assert summary["fixed_split_warnings"].sum() == 4
    assert summary["allocation_only_warnings"].sum() == 1

    warning_rows = summary.loc[summary["fixed_split_warnings"] > 0]
    assert list(warning_rows["intersection_id"]) == ["C", "C"]
    assert list(warning_rows["fixed_split_warnings"]) == [2, 2]
