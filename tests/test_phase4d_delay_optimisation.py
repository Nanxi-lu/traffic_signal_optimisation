import numpy as np

from traffic_signals.delay_optimisation import (
    evaluate_offsets,
    search_delay_offsets,
)


def make_test_arrivals() -> dict[int, dict[str, np.ndarray]]:
    return {
        42: {
            "EB": np.array([0, 5, 10, 25, 45]),
            "WB": np.array([0, 8, 17, 32, 50]),
        }
    }


def test_evaluation_is_deterministic() -> None:
    offsets = {"A": 0, "B": 36, "C": 1}
    greens = {"A": 16, "B": 16, "C": 15}
    first = evaluate_offsets(make_test_arrivals(), 40, greens, offsets)
    second = evaluate_offsets(make_test_arrivals(), 40, greens, offsets)
    assert first == second


def test_optimisation_respects_fairness() -> None:
    greens = {"A": 16, "B": 16, "C": 15}
    original = {"A": 0, "B": 36, "C": 1}
    previous = {"A": 0, "B": 33, "C": 33}
    result = search_delay_offsets(
        make_test_arrivals(),
        40,
        greens,
        original,
        previous,
        fairness_limit_s=2.0,
    )
    for direction in ["EB", "WB"]:
        assert (
            result["optimised"][f"{direction}_delay_s"]
            <= result["baseline"][f"{direction}_delay_s"] + 2.0 + 1e-9
        )
    assert (
        result["optimised"]["combined_delay_s"]
        <= result["baseline"]["combined_delay_s"] + 1e-9
    )
