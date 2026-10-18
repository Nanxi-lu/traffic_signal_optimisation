import pandas as pd

from workflows.stage4a_4e_corridor import (
    FACTORS,
    check_robust_fairness,
    compare_candidates,
    generate_candidates,
)


def make_paired_changes(change: float) -> pd.DataFrame:
    rows = []
    for factor in FACTORS:
        for direction in ["EB", "WB"]:
            for seed in range(12):
                rows.append(
                    {
                        "factor": factor,
                        "direction": direction,
                        "seed": seed,
                        "delay_change_s": change,
                    }
                )
    return pd.DataFrame(rows)


def test_prior_candidates_included() -> None:
    candidates = generate_candidates()
    assert (36, 81) in candidates
    assert (28, 72) in candidates
    assert (32, 68) in candidates


def test_candidate_comparison_uses_matched_vehicle_counts() -> None:
    baseline = pd.DataFrame(
        {
            "seed": [1],
            "factor": [1.0],
            "direction": ["EB"],
            "vehicles": [10],
            "mean_delay_s": [5.0],
            "queue_at_hour_veh": [3],
        }
    )
    candidate = baseline.assign(mean_delay_s=6.5, queue_at_hour_veh=2)
    paired = compare_candidates(baseline, candidate)
    assert paired["delay_change_s"].item() == 1.5
    assert paired["queue_change_veh"].item() == -1


def test_robust_fairness_passes_when_upper_bound_is_below_limit() -> None:
    passed, checks = check_robust_fairness(make_paired_changes(1.5))
    assert passed
    assert len(checks) == 8
    assert checks["passes"].all()


def test_robust_fairness_fails_when_upper_bound_exceeds_limit() -> None:
    passed, checks = check_robust_fairness(make_paired_changes(2.1))
    assert not passed
    assert not checks["passes"].any()
