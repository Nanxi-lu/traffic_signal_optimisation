"""Select and holdout-test an AM offset plan with robust fairness checks."""

from pathlib import Path
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from phase4b_multi_seed import bootstrap_mean_ci
from phase4d2_stress_validation import (
    generate_stress_arrivals,
    load_inputs,
)
from traffic_signals.offset_optimisation import circular_distance
from traffic_signals.vehicle_simulation import simulate_direction


DATA_DIR = Path("outputs/data")
FACTORS = (0.75, 1.0, 1.25, 1.5)
SCREEN_FACTORS = (1.0, 1.5)
SCREEN_SEEDS = range(102, 106)
TRAIN_SEEDS = range(102, 114)
HOLDOUT_SEEDS = range(114, 144)
FAIRNESS_LIMIT_S = 2.0
SCREEN_LIMIT_S = 3.0
DURATION_S = 3600.0
BASELINE_PAIR = (36, 81)
PHASE3F_PAIR = (28, 72)
PHASE4D_PAIR = (32, 68)
SCREENING_PATH = DATA_DIR / "phase4d3_screening.csv"
TRAINING_PATH = DATA_DIR / "phase4d3_training_candidates.csv"
PLAN_PATH = DATA_DIR / "phase4d3_selected_plan.csv"
HOLDOUT_PATH = DATA_DIR / "phase4d3_holdout_metrics.csv"
HOLDOUT_SUMMARY_PATH = DATA_DIR / "phase4d3_holdout_summary.csv"


def evaluate_candidate(
    arrival_scenarios: Mapping[int, Mapping[float, Mapping[str, np.ndarray]]],
    base_plan: Mapping[str, object],
    pair: tuple[int, int],
    factors: Iterable[float] = FACTORS,
) -> pd.DataFrame:
    """Evaluate an offset pair over specified seeds and demand factors."""

    cycle = int(base_plan["cycle_s"])
    factor_values = tuple(factors)
    if not arrival_scenarios or not factor_values:
        raise ValueError("Arrival scenarios and factors are required.")
    plan = {
        "cycle_s": cycle,
        "greens": dict(base_plan["greens"]),
        "offsets": {
            "A": 0,
            "B": int(pair[0]) % cycle,
            "C": int(pair[1]) % cycle,
        },
    }
    records: list[dict[str, object]] = []
    for seed, scenarios in arrival_scenarios.items():
        for factor in factor_values:
            if factor not in scenarios:
                raise ValueError("Missing demand scenario.")
            total_delay = 0.0
            total_vehicles = 0
            total_queue = 0
            for direction in ("EB", "WB"):
                vehicles, queues = simulate_direction(
                    entry_times=scenarios[factor][direction],
                    direction=direction,
                    signal_plan=plan,
                    speed_kmh=40.0,
                    headway_s=2.0,
                    horizon_s=DURATION_S,
                )
                vehicle_count = len(vehicles)
                delay_sum = float(vehicles["total_delay_s"].sum())
                queue_at_hour = int(queues["queue_at_horizon_veh"].sum())
                records.append(
                    {
                        "seed": seed,
                        "factor": factor,
                        "direction": direction,
                        "vehicles": vehicle_count,
                        "mean_delay_s": delay_sum / vehicle_count,
                        "queue_at_hour_veh": queue_at_hour,
                    }
                )
                total_delay += delay_sum
                total_vehicles += vehicle_count
                total_queue += queue_at_hour
            records.append(
                {
                    "seed": seed,
                    "factor": factor,
                    "direction": "Combined",
                    "vehicles": total_vehicles,
                    "mean_delay_s": total_delay / total_vehicles,
                    "queue_at_hour_veh": total_queue,
                }
            )
    result = pd.DataFrame(records)
    expected_rows = len(arrival_scenarios) * len(factor_values) * 3
    if len(result) != expected_rows:
        raise ValueError("Candidate evaluation row count is incorrect.")
    return result


def compare_candidates(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate paired differences for identical arrival scenarios."""

    keys = ["seed", "factor", "direction"]
    paired = baseline.merge(
        candidate,
        on=keys,
        suffixes=("_3a", "_candidate"),
        validate="one_to_one",
    )
    if not (paired["vehicles_3a"] == paired["vehicles_candidate"]).all():
        raise ValueError("Candidate vehicle counts do not match Phase 3A.")
    paired["delay_change_s"] = (
        paired["mean_delay_s_candidate"] - paired["mean_delay_s_3a"]
    )
    paired["queue_change_veh"] = (
        paired["queue_at_hour_veh_candidate"]
        - paired["queue_at_hour_veh_3a"]
    )
    return paired


def check_robust_fairness(
    paired: pd.DataFrame,
    limit_s: float = FAIRNESS_LIMIT_S,
) -> tuple[bool, pd.DataFrame]:
    """Check directional bootstrap upper endpoints at every demand factor."""

    if not np.isfinite(limit_s):
        raise ValueError("Fairness limit must be finite.")
    checks: list[dict[str, object]] = []
    directional = paired.loc[paired["direction"].isin(["EB", "WB"])]
    for (factor, direction), group in directional.groupby(
        ["factor", "direction"], sort=True
    ):
        differences = group["delay_change_s"].to_numpy()
        low, high = bootstrap_mean_ci(
            differences,
            n_bootstrap=2000,
            seed=2026,
        )
        checks.append(
            {
                "factor": factor,
                "direction": direction,
                "replicates": len(group),
                "mean_change_s": differences.mean(),
                "ci_lower_s": low,
                "ci_upper_s": high,
                "passes": bool(high <= limit_s + 1e-9),
            }
        )
    check_frame = pd.DataFrame(checks)
    expected_checks = directional["factor"].nunique() * 2
    if len(check_frame) != expected_checks:
        raise ValueError("Fairness checks are incomplete.")
    return bool(check_frame["passes"].all()), check_frame


def generate_candidates() -> list[tuple[int, int]]:
    """Generate the specified local candidate-offset set."""

    candidates = {
        (b, c)
        for b in range(26, 41, 2)
        for c in range(62, 85, 2)
    }
    candidates.update({BASELINE_PAIR, PHASE3F_PAIR, PHASE4D_PAIR})
    return sorted(candidates)


def prepare_arrivals(
    traffic: pd.DataFrame,
    seeds: Iterable[int],
) -> dict[int, dict[float, dict[str, np.ndarray]]]:
    """Generate each nested arrival set once for reuse across candidates."""

    return {seed: generate_stress_arrivals(traffic, seed) for seed in seeds}


def _candidate_score_components(
    paired: pd.DataFrame,
    pair: tuple[int, int],
    cycle: int,
) -> tuple[float, float, float]:
    """Return the stated delay, queue and offset-change priorities."""

    combined = paired.loc[paired["direction"] == "Combined"]
    mean_delay_change = float(combined["delay_change_s"].mean())
    mean_queue_change = float(combined["queue_change_veh"].mean())
    offset_change = circular_distance(
        pair[0], BASELINE_PAIR[0], cycle
    ) + circular_distance(pair[1], BASELINE_PAIR[1], cycle)
    return mean_delay_change, mean_queue_change, offset_change


def screen_candidates(
    traffic: pd.DataFrame,
    base_plan: Mapping[str, object],
) -> tuple[list[tuple[int, int]], dict[int, dict[float, dict[str, np.ndarray]]], pd.DataFrame]:
    """Apply the relaxed four-seed screen at 100% and 150% demand."""

    all_training = prepare_arrivals(traffic, TRAIN_SEEDS)
    screening = {seed: all_training[seed] for seed in SCREEN_SEEDS}
    original = evaluate_candidate(
        screening,
        base_plan,
        BASELINE_PAIR,
        factors=SCREEN_FACTORS,
    )
    ranking: list[dict[str, object]] = []
    cycle = int(base_plan["cycle_s"])
    for pair in generate_candidates():
        candidate = evaluate_candidate(
            screening,
            base_plan,
            pair,
            factors=SCREEN_FACTORS,
        )
        paired = compare_candidates(original, candidate)
        directional_means = (
            paired.loc[paired["direction"].isin(["EB", "WB"])]
            .groupby(["factor", "direction"])["delay_change_s"]
            .mean()
        )
        combined_factor_changes = (
            paired.loc[paired["direction"] == "Combined"]
            .groupby("factor")["delay_change_s"]
            .mean()
        )
        relaxed_fairness_pass = bool(
            (directional_means <= SCREEN_LIMIT_S + 1e-9).all()
        )
        performance_pass = bool(
            (combined_factor_changes <= 1e-9).all()
        )
        mean_delay, mean_queue, offset_change = _candidate_score_components(
            paired, pair, cycle
        )
        ranking.append(
            {
                "B_offset_s": pair[0],
                "C_offset_s": pair[1],
                "max_directional_mean_change_s": directional_means.max(),
                "mean_combined_delay_change_s": mean_delay,
                "mean_residual_queue_change_veh": mean_queue,
                "offset_change_s": offset_change,
                "relaxed_fairness_pass": relaxed_fairness_pass,
                "performance_pass": performance_pass,
                "shortlisted": relaxed_fairness_pass and performance_pass,
            }
        )
    screening_results = pd.DataFrame(ranking).sort_values(
        [
            "shortlisted",
            "mean_combined_delay_change_s",
            "mean_residual_queue_change_veh",
            "offset_change_s",
            "B_offset_s",
            "C_offset_s",
        ],
        ascending=[False, True, True, True, True, True],
        ignore_index=True,
    )
    screening_results["screen_rank"] = np.arange(1, len(screening_results) + 1)
    shortlist = [
        (int(row.B_offset_s), int(row.C_offset_s))
        for row in screening_results.itertuples()
        if row.shortlisted
    ]
    if BASELINE_PAIR not in shortlist:
        raise RuntimeError("Phase 3A fallback did not pass screening.")
    return shortlist, all_training, screening_results


def select_robust_plan(
    shortlist: Iterable[tuple[int, int]],
    training_arrivals: Mapping[int, Mapping[float, Mapping[str, np.ndarray]]],
    base_plan: Mapping[str, object],
) -> tuple[tuple[int, int], pd.DataFrame]:
    """Select the best training candidate satisfying all stated constraints."""

    baseline = evaluate_candidate(
        training_arrivals,
        base_plan,
        BASELINE_PAIR,
    )
    training_records: list[dict[str, object]] = []
    feasible_candidates: list[tuple[tuple[float, ...], tuple[int, int]]] = []
    cycle = int(base_plan["cycle_s"])
    for pair in shortlist:
        candidate = evaluate_candidate(training_arrivals, base_plan, pair)
        paired = compare_candidates(baseline, candidate)
        fairness_pass, checks = check_robust_fairness(paired)
        combined = paired.loc[paired["direction"] == "Combined"]
        factor_changes = combined.groupby("factor")["delay_change_s"].mean()
        performance_pass = bool((factor_changes <= 1e-9).all())
        mean_delay, mean_queue, offset_change = _candidate_score_components(
            paired, pair, cycle
        )
        feasible = fairness_pass and performance_pass
        record = {
            "B_offset_s": pair[0],
            "C_offset_s": pair[1],
            "fairness_pass": fairness_pass,
            "performance_pass": performance_pass,
            "feasible": feasible,
            "mean_combined_delay_change_s": mean_delay,
            "mean_residual_queue_change_veh": mean_queue,
            "offset_change_s": offset_change,
            "max_fairness_ci_upper_s": checks["ci_upper_s"].max(),
            "failed_fairness_checks": int((~checks["passes"]).sum()),
        }
        for check in checks.itertuples():
            label = f"{check.direction}_{int(round(check.factor * 100))}"
            record[f"{label}_mean_change_s"] = check.mean_change_s
            record[f"{label}_ci_upper_s"] = check.ci_upper_s
            record[f"{label}_passes"] = check.passes
        training_records.append(record)
        if feasible:
            score = (
                round(mean_delay, 8),
                round(mean_queue, 8),
                offset_change,
                pair[0],
                pair[1],
            )
            feasible_candidates.append((score, pair))

    if not feasible_candidates:
        raise RuntimeError("No feasible plan found, including Phase 3A fallback.")
    feasible_candidates.sort()
    selected_pair = feasible_candidates[0][1]
    training_summary = pd.DataFrame(training_records).sort_values(
        [
            "feasible",
            "mean_combined_delay_change_s",
            "mean_residual_queue_change_veh",
            "offset_change_s",
        ],
        ascending=[False, True, True, True],
        ignore_index=True,
    )
    training_summary["selected"] = (
        (training_summary["B_offset_s"] == selected_pair[0])
        & (training_summary["C_offset_s"] == selected_pair[1])
    )
    return selected_pair, training_summary


def _summarise_holdout_comparison(
    paired: pd.DataFrame,
    benchmark: str,
) -> list[dict[str, object]]:
    """Summarise one benchmark comparison over factors and directions."""

    records: list[dict[str, object]] = []
    for factor in FACTORS:
        for direction in ("EB", "WB", "Combined"):
            group = paired.loc[
                (paired["factor"] == factor)
                & (paired["direction"] == direction)
            ]
            low, high = bootstrap_mean_ci(
                group["delay_change_s"].to_numpy(),
                n_bootstrap=2000,
                seed=2026,
            )
            fairness_applicable = (
                benchmark == "Phase3A" and direction in ("EB", "WB")
            )
            records.append(
                {
                    "factor": factor,
                    "direction": direction,
                    "benchmark": benchmark,
                    "replicates": len(group),
                    "baseline_delay_s": group["mean_delay_s_3a"].mean(),
                    "phase4d3_delay_s": group[
                        "mean_delay_s_candidate"
                    ].mean(),
                    "mean_delay_change_s": group["delay_change_s"].mean(),
                    "ci_lower_s": low,
                    "ci_upper_s": high,
                    "seeds_better": int((group["delay_change_s"] < 0).sum()),
                    "baseline_queue_at_hour_veh": group[
                        "queue_at_hour_veh_3a"
                    ].mean(),
                    "phase4d3_queue_at_hour_veh": group[
                        "queue_at_hour_veh_candidate"
                    ].mean(),
                    "mean_queue_change_veh": group[
                        "queue_change_veh"
                    ].mean(),
                    "fairness_applicable": fairness_applicable,
                    "fairness_pass": (
                        bool(high <= FAIRNESS_LIMIT_S + 1e-9)
                        if fairness_applicable
                        else None
                    ),
                }
            )
    return records


def validate_holdout(
    traffic: pd.DataFrame,
    strategies: Mapping[str, Mapping[str, object]],
    selected_pair: tuple[int, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the frozen plan and compare it with all earlier strategies."""

    holdout_arrivals = prepare_arrivals(traffic, HOLDOUT_SEEDS)
    base_plan = strategies["Phase3A"]
    candidate_pairs = {
        "Phase3A": BASELINE_PAIR,
        "Phase3F": PHASE3F_PAIR,
        "Phase4D": PHASE4D_PAIR,
        "Phase4D3": selected_pair,
    }
    frames: list[pd.DataFrame] = []
    for name, pair in candidate_pairs.items():
        result = evaluate_candidate(holdout_arrivals, base_plan, pair)
        result["strategy"] = name
        frames.append(result)
    holdout = pd.concat(frames, ignore_index=True)
    if len(holdout) != 1440:
        raise ValueError("Expected 1,440 holdout metric rows.")

    candidate = holdout.loc[holdout["strategy"] == "Phase4D3"].drop(
        columns="strategy"
    )
    summary_records: list[dict[str, object]] = []
    for benchmark in ("Phase3A", "Phase3F", "Phase4D"):
        baseline = holdout.loc[holdout["strategy"] == benchmark].drop(
            columns="strategy"
        )
        paired = compare_candidates(baseline, candidate)
        summary_records.extend(_summarise_holdout_comparison(paired, benchmark))
    summary = pd.DataFrame(summary_records)
    if len(summary) != 36 or not summary["replicates"].eq(30).all():
        raise ValueError("Expected 36 holdout summary rows.")
    return holdout, summary


def main() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Screen, train, freeze and holdout-test the Phase 4D-3 plan."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    traffic, strategies = load_inputs()
    base_plan = strategies["Phase3A"]
    shortlist, training_arrivals, screening = screen_candidates(
        traffic, base_plan
    )
    print("Shortlisted candidates:", len(shortlist))
    selected, training_summary = select_robust_plan(
        shortlist,
        training_arrivals,
        base_plan,
    )
    selected_row = training_summary.loc[training_summary["selected"]].iloc[0]
    selected_plan = pd.DataFrame(
        [
            {
                "period": "AM peak",
                "cycle_s": base_plan["cycle_s"],
                "A_offset_s": 0,
                "B_offset_s": selected[0],
                "C_offset_s": selected[1],
                "training_seeds": len(TRAIN_SEEDS),
                "fairness_limit_s": FAIRNESS_LIMIT_S,
                "training_fairness_pass": selected_row["fairness_pass"],
                "training_performance_pass": selected_row["performance_pass"],
                "training_mean_combined_delay_change_s": selected_row[
                    "mean_combined_delay_change_s"
                ],
                "training_mean_residual_queue_change_veh": selected_row[
                    "mean_residual_queue_change_veh"
                ],
            }
        ]
    )
    holdout, holdout_summary = validate_holdout(
        traffic,
        strategies,
        selected,
    )

    screening.to_csv(SCREENING_PATH, index=False)
    training_summary.to_csv(TRAINING_PATH, index=False)
    selected_plan.to_csv(PLAN_PATH, index=False)
    holdout.to_csv(HOLDOUT_PATH, index=False)
    holdout_summary.to_csv(HOLDOUT_SUMMARY_PATH, index=False)

    print("Selected Phase 4D-3 offsets:", selected)
    print("\nSelected training feasibility:")
    print(selected_row.to_string())
    print("\nHoldout comparison with Phase 3A:")
    print(
        holdout_summary.loc[
            holdout_summary["benchmark"] == "Phase3A"
        ].to_string(index=False)
    )
    print(f"\nWrote {SCREENING_PATH}")
    print(f"Wrote {TRAINING_PATH}")
    print(f"Wrote {PLAN_PATH}")
    print(f"Wrote {HOLDOUT_PATH}")
    print(f"Wrote {HOLDOUT_SUMMARY_PATH}")
    return screening, training_summary, selected_plan, holdout, holdout_summary


if __name__ == "__main__":
    main()
