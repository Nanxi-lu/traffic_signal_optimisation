import numpy as np
import pandas as pd

from phase4b_multi_seed import generate_arrivals
from phase4e_four_period_optimisation import (
    FACTORS,
    HOLDOUT_SEEDS,
    OPTIMISE_PERIODS,
    PERIODS,
    TRAIN_SEEDS,
    candidate_space,
    prepare_period_arrivals,
)


def test_train_holdout_separation():
    assert len(TRAIN_SEEDS) == 12
    assert len(HOLDOUT_SEEDS) == 30
    assert set(TRAIN_SEEDS).isdisjoint(HOLDOUT_SEEDS)


def test_am_plan_not_reoptimised():
    assert "AM peak" in PERIODS
    assert "AM peak" not in OPTIMISE_PERIODS


def test_candidate_wraparound_and_prior_plans_included():
    original = {"A": 0, "B": 36, "C": 1}
    previous = {"A": 0, "B": 39, "C": 2}
    candidates = candidate_space(40, original, previous)
    assert (36, 1) in candidates
    assert (39, 2) in candidates
    assert (1, 2) in candidates
    assert all(0 <= b < 40 and 0 <= c < 40 for b, c in candidates)


def test_period_arrivals_preserve_generated_baseline_and_are_nested():
    traffic = pd.DataFrame(
        {
            "period": ["Overnight"] * 4,
            "intersection_id": ["A", "A", "C", "C"],
            "approach": ["E", "E", "W", "W"],
            "volume_15min": [20, 24, 18, 22],
        }
    )
    seed = 144
    scenarios = prepare_period_arrivals(traffic, "Overnight", [seed])[seed]
    baseline = generate_arrivals(traffic, "Overnight", seed)
    assert tuple(scenarios) == FACTORS
    for direction in ("EB", "WB"):
        np.testing.assert_array_equal(scenarios[1.0][direction], baseline[direction])
        sets = {
            factor: set(scenarios[factor][direction])
            for factor in FACTORS
        }
        assert sets[0.75].issubset(sets[1.0])
        assert sets[1.0].issubset(sets[1.25])
        assert sets[1.25].issubset(sets[1.5])
