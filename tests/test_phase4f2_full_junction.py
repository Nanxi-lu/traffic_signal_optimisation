import numpy as np
import pandas as pd

from phase4f2_multi_seed_full_junction import PERIOD_HOURS, SEEDS
from traffic_signals.network_evaluation import (
    build_period_plans,
    evaluate_network,
)


def make_test_plans():
    original = pd.DataFrame(
        {
            "period": ["Overnight"] * 3,
            "intersection_id": ["A", "B", "C"],
            "common_cycle_s": [40, 40, 40],
            "ew_green_s": [16, 16, 15],
            "ns_green_s": [12, 12, 13],
            "offset_s": [0, 36, 1],
            "lost_time_s": [12, 12, 12],
        }
    )
    selected = pd.DataFrame(
        {
            "period": ["Overnight"],
            "A_offset_s": [0],
            "B_offset_s": [39],
            "C_offset_s": [2],
        }
    )
    return original, selected


def make_arrivals():
    return {
        ("A", "E"): np.array([0.0, 8.0]),
        ("C", "W"): np.array([1.0, 9.0]),
        ("A", "N"): np.array([2.0]),
        ("A", "S"): np.array([3.0]),
        ("B", "N"): np.array([4.0]),
        ("B", "S"): np.array([5.0]),
        ("C", "N"): np.array([6.0]),
        ("C", "S"): np.array([7.0]),
    }


def test_seed_and_period_configuration():
    assert len(SEEDS) == 30
    assert list(SEEDS) == list(range(200, 230))
    assert PERIOD_HOURS == {
        "Overnight": 2,
        "AM peak": 8,
        "Interpeak": 12,
        "PM peak": 17,
    }


def test_build_period_plans_preserves_greens_and_changes_offsets():
    original, selected = make_test_plans()
    strategies = build_period_plans(original, selected, "Overnight")
    assert strategies["Phase3A"]["B"]["ew_offset_s"] == 36
    assert strategies["Phase4E"]["B"]["ew_offset_s"] == 39
    assert strategies["Phase3A"]["C"]["ew_green_s"] == 15
    assert strategies["Phase4E"]["C"]["ns_green_s"] == 13


def test_evaluate_network_returns_eight_movements_and_one_summary():
    original, selected = make_test_plans()
    plan = build_period_plans(original, selected, "Overnight")["Phase3A"]
    movements, network = evaluate_network(
        arrivals=make_arrivals(),
        junction_plans=plan,
        strategy="Phase3A",
        period="Overnight",
        seed=200,
    )
    assert len(movements) == 8
    assert set(movements["movement"]) == {
        "EB",
        "WB",
        "A_N",
        "A_S",
        "B_N",
        "B_S",
        "C_N",
        "C_S",
    }
    assert network["total_passages"] == movements["passages"].sum()


def test_fixed_strategies_receive_identical_vehicle_counts():
    original, selected = make_test_plans()
    strategies = build_period_plans(original, selected, "Overnight")
    evaluated = {}
    for strategy, plan in strategies.items():
        movements, network = evaluate_network(
            arrivals=make_arrivals(),
            junction_plans=plan,
            strategy=strategy,
            period="Overnight",
            seed=200,
        )
        evaluated[strategy] = (movements.set_index("movement"), network)
    assert evaluated["Phase3A"][0]["vehicles"].equals(
        evaluated["Phase4E"][0]["vehicles"]
    )
    assert (
        evaluated["Phase3A"][1]["total_passages"]
        == evaluated["Phase4E"][1]["total_passages"]
    )
