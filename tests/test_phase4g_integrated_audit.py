import pandas as pd

from workflows.stage4f_4g_full_junction import (
    PHASE4G_PERIODS as PERIODS,
    PHASE4G_SEEDS as SEEDS,
    build_integrated_plans,
)
from traffic_signals.full_junction import is_effective_green


def make_test_inputs(period):
    if period == "PM peak":
        ew = [87, 90, 92]
        ns = [51, 48, 46]
        original_offsets = [0, 36, 81]
        selected_offsets = [0, 30, 79]
    else:
        ew = [87, 88, 95]
        ns = [51, 50, 43]
        original_offsets = [0, 36, 81]
        selected_offsets = [0, 36, 70]
    original = pd.DataFrame(
        {
            "period": [period] * 3,
            "intersection_id": ["A", "B", "C"],
            "common_cycle_s": [150] * 3,
            "ew_green_s": ew,
            "ns_green_s": ns,
            "lost_time_s": [12] * 3,
            "offset_s": original_offsets,
        }
    )
    selected = pd.DataFrame(
        {
            "period": [period],
            "A_offset_s": [selected_offsets[0]],
            "B_offset_s": [selected_offsets[1]],
            "C_offset_s": [selected_offsets[2]],
        }
    )
    return original, selected


def test_integrated_audit_configuration():
    assert PERIODS == ("Overnight", "AM peak", "Interpeak", "PM peak")
    assert list(SEEDS) == list(range(500, 530))


def test_pm_integrated_plan_changes_only_junction_c():
    original, selected = make_test_inputs("PM peak")
    plans = build_integrated_plans(original, selected, "PM peak")
    assert plans["Integrated"]["A"] == plans["Phase4E"]["A"]
    assert plans["Integrated"]["B"] == plans["Phase4E"]["B"]
    c_plan = plans["Integrated"]["C"]
    assert c_plan["ew_green_s"] == 89
    assert c_plan["ns_green_s"] == 49
    assert c_plan["ew_offset_s"] == 83
    assert c_plan["ns_offset_s"] == 28
    assert plans["Phase4E"]["C"]["ew_green_s"] == 92
    assert plans["Phase4E"]["C"]["ew_offset_s"] == 79


def test_non_pm_integrated_plans_equal_phase4e_without_aliasing():
    for period in ("Overnight", "AM peak", "Interpeak"):
        original, selected = make_test_inputs(period)
        plans = build_integrated_plans(original, selected, period)
        assert plans["Integrated"] == plans["Phase4E"]
        for junction in ("A", "B", "C"):
            assert plans["Integrated"][junction] is not plans["Phase4E"][junction]


def test_integrated_effective_green_windows_never_overlap():
    for period in PERIODS:
        original, selected = make_test_inputs(period)
        plans = build_integrated_plans(original, selected, period)
        for plan in plans["Integrated"].values():
            for time_s in range(int(plan["cycle_s"])):
                assert not (
                    is_effective_green(time_s, plan, "EW")
                    and is_effective_green(time_s, plan, "NS")
                )
