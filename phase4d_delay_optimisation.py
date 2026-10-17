"""Run Phase 4D-1 AM delay-aware offset optimisation and validation."""

from pathlib import Path

import numpy as np
import pandas as pd

from phase4b_multi_seed import (
    bootstrap_mean_ci,
    generate_arrivals,
    get_signal_strategies,
)
from traffic_signals.delay_optimisation import (
    evaluate_offsets,
    search_delay_offsets,
)
from traffic_signals.segmentation import assign_period


DATA_DIR = Path("outputs/data")
PERIOD = "AM peak"
TRAIN_SEEDS = range(42, 45)
VALIDATION_SEEDS = range(45, 72)
FAIRNESS_LIMIT_S = 2.0
TRAFFIC_PATH = DATA_DIR / "synthetic_traffic_15min.csv"
PHASE3A_PATH = DATA_DIR / "phase3a_signal_plans.csv"
PHASE3F_PATH = DATA_DIR / "phase3f_optimisation_summary.csv"
PHASE4B_METRICS_PATH = DATA_DIR / "phase4b_replication_metrics.csv"
PLAN_PATH = DATA_DIR / "phase4d_selected_plan.csv"
TRAINING_PATH = DATA_DIR / "phase4d_training_metrics.csv"
VALIDATION_PATH = DATA_DIR / "phase4d_validation_metrics.csv"
SUMMARY_PATH = DATA_DIR / "phase4d_validation_summary.csv"


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and normalise the traffic and fixed signal plans."""

    traffic = pd.read_csv(TRAFFIC_PATH, parse_dates=["time"])
    phase3a = pd.read_csv(PHASE3A_PATH)
    phase3f = pd.read_csv(PHASE3F_PATH)
    traffic["period"] = traffic["time"].apply(assign_period)
    traffic["intersection_id"] = traffic["intersection_id"].astype(str).str.upper()
    traffic["approach"] = (
        traffic["approach"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"EASTBOUND": "E", "WESTBOUND": "W"})
    )
    return traffic, phase3a, phase3f


def optimise_training_offsets(
    traffic: pd.DataFrame,
    phase3a: pd.DataFrame,
    phase3f: pd.DataFrame,
) -> tuple[dict[str, object], dict[str, dict[str, object]], pd.DataFrame]:
    """Select one AM candidate using only training seeds 42--44."""

    strategies = get_signal_strategies(phase3a, phase3f, PERIOD)
    original = strategies["Phase3A"]
    previous = strategies["Phase3F"]
    training = {
        seed: generate_arrivals(traffic, PERIOD, seed) for seed in TRAIN_SEEDS
    }
    result = search_delay_offsets(
        arrivals_by_seed=training,
        cycle=int(original["cycle_s"]),
        greens=original["greens"],
        original_offsets=original["offsets"],
        previous_offsets=previous["offsets"],
        fairness_limit_s=FAIRNESS_LIMIT_S,
    )
    for direction in ("EB", "WB"):
        if (
            result["optimised"][f"{direction}_delay_s"]
            > result["baseline"][f"{direction}_delay_s"]
            + FAIRNESS_LIMIT_S
            + 1e-9
        ):
            raise ValueError("Selected plan violates the training fairness limit.")

    selected_plan = {
        "cycle_s": original["cycle_s"],
        "greens": original["greens"],
        "offsets": result["offsets"],
    }
    training_rows = []
    for strategy, plan in (
        ("Phase3A", original),
        ("Phase3F", previous),
        ("Phase4D", selected_plan),
    ):
        training_rows.append(
            {
                "strategy": strategy,
                **evaluate_offsets(
                    training,
                    int(plan["cycle_s"]),
                    plan["greens"],
                    plan["offsets"],
                ),
            }
        )
    return result, strategies, pd.DataFrame(training_rows)


def build_selected_plan(
    result: dict[str, object],
    original: dict[str, object],
) -> pd.DataFrame:
    """Create the one-row provisional Phase 4D timing-plan output."""

    offsets = result["offsets"]
    baseline = result["baseline"]
    optimised = result["optimised"]
    return pd.DataFrame(
        [
            {
                "period": PERIOD,
                "strategy": "Phase4D_delay_aware_candidate",
                "cycle_s": original["cycle_s"],
                "A_offset_s": offsets["A"],
                "B_offset_s": offsets["B"],
                "C_offset_s": offsets["C"],
                "fairness_limit_s": FAIRNESS_LIMIT_S,
                "training_seeds": "42-44",
                "candidates_tested": result["candidates_tested"],
                "feasible_candidates": result["feasible_candidates"],
                "baseline_EB_delay_s": baseline["EB_delay_s"],
                "baseline_WB_delay_s": baseline["WB_delay_s"],
                "baseline_combined_delay_s": baseline["combined_delay_s"],
                "optimised_EB_delay_s": optimised["EB_delay_s"],
                "optimised_WB_delay_s": optimised["WB_delay_s"],
                "optimised_combined_delay_s": optimised["combined_delay_s"],
                "optimised_residual_queue_mean": optimised[
                    "residual_queue_mean"
                ],
            }
        ]
    )


def validate_unseen_seeds(
    traffic: pd.DataFrame,
    strategies: dict[str, dict[str, object]],
    selected_offsets: dict[str, float],
) -> pd.DataFrame:
    """Evaluate Phase 3A, Phase 3F and Phase 4D on seeds 45--71."""

    original = strategies["Phase3A"]
    candidate = {
        "cycle_s": original["cycle_s"],
        "greens": original["greens"],
        "offsets": selected_offsets,
    }
    plans = {**strategies, "Phase4D": candidate}
    validation_records: list[dict[str, object]] = []
    for seed in VALIDATION_SEEDS:
        arrivals = generate_arrivals(traffic, PERIOD, seed)
        for strategy, plan in plans.items():
            validation_records.append(
                {
                    "period": PERIOD,
                    "seed": seed,
                    "strategy": strategy,
                    **evaluate_offsets(
                        {seed: arrivals},
                        int(plan["cycle_s"]),
                        plan["greens"],
                        plan["offsets"],
                    ),
                }
            )
    validation = pd.DataFrame(validation_records)
    if len(validation) != 81:
        raise ValueError("Expected 81 unseen-seed validation rows.")
    return validation


def verify_phase4b_regression(validation: pd.DataFrame) -> None:
    """Verify Phase 3A/3F unseen-seed delays against saved Phase 4B metrics."""

    phase4b = pd.read_csv(PHASE4B_METRICS_PATH)
    reference = phase4b.loc[
        (phase4b["period"] == PERIOD)
        & (phase4b["seed"].isin(VALIDATION_SEEDS))
        & (phase4b["strategy"].isin(["Phase3A", "Phase3F"]))
    ]
    current = validation.loc[
        validation["strategy"].isin(["Phase3A", "Phase3F"])
    ]
    comparison = current.merge(
        reference,
        on=["period", "seed", "strategy"],
        validate="one_to_many",
    )
    if len(comparison) != 162:
        raise ValueError("Expected 162 Phase 4B delay regression rows.")
    metric_by_direction = {
        "EB": "EB_delay_s",
        "WB": "WB_delay_s",
        "Combined": "combined_delay_s",
    }
    for direction, column in metric_by_direction.items():
        selected = comparison.loc[comparison["direction"] == direction]
        if len(selected) != 54 or not np.allclose(
            selected[column], selected["mean_delay_s"], atol=1e-9
        ):
            raise ValueError(f"Phase 4B regression failed for {direction} delay.")
    print("Phase 4B unseen-seed delay regression passed.")


def summarise_validation(validation: pd.DataFrame) -> pd.DataFrame:
    """Build the specified eight paired unseen-seed comparisons."""

    metrics_to_compare = [
        "EB_delay_s",
        "WB_delay_s",
        "combined_delay_s",
        "residual_queue_mean",
    ]
    summary_records: list[dict[str, object]] = []
    for metric in metrics_to_compare:
        pivot = validation.pivot(
            index="seed",
            columns="strategy",
            values=metric,
        )
        for benchmark in ["Phase3A", "Phase3F"]:
            change = pivot["Phase4D"] - pivot[benchmark]
            low, high = bootstrap_mean_ci(change.to_numpy())
            summary_records.append(
                {
                    "metric": metric,
                    "benchmark": benchmark,
                    "benchmark_mean": pivot[benchmark].mean(),
                    "phase4d_mean": pivot["Phase4D"].mean(),
                    "mean_change": change.mean(),
                    "ci_lower": low,
                    "ci_upper": high,
                    "seeds_better": int((change < 0).sum()),
                    "seeds_tested": len(change),
                }
            )
    summary = pd.DataFrame(summary_records)
    if len(summary) != 8 or not summary["seeds_tested"].eq(27).all():
        raise ValueError("Expected eight comparisons across 27 validation seeds.")
    return summary


def main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select an AM candidate, validate it once and save all outputs."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    traffic, phase3a, phase3f = _load_inputs()
    result, strategies, training = optimise_training_offsets(
        traffic, phase3a, phase3f
    )
    selected = build_selected_plan(result, strategies["Phase3A"])
    validation = validate_unseen_seeds(
        traffic, strategies, result["offsets"]
    )
    verify_phase4b_regression(validation)
    summary = summarise_validation(validation)

    selected.to_csv(PLAN_PATH, index=False)
    training.to_csv(TRAINING_PATH, index=False)
    validation.to_csv(VALIDATION_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\nPhase 4D selected offsets:")
    print(result["offsets"])
    print("\nTraining baseline:")
    print(result["baseline"])
    print("\nTraining optimised:")
    print(result["optimised"])
    print("\nTraining strategy comparison:")
    print(training.to_string(index=False))
    print("\nCandidates tested:", result["candidates_tested"])
    print("Feasible candidates:", result["feasible_candidates"])
    print("\nUnseen-seed validation:")
    print(summary.to_string(index=False))
    print(f"\nWrote {PLAN_PATH}")
    print(f"Wrote {TRAINING_PATH}")
    print(f"Wrote {VALIDATION_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    return selected, training, validation, summary


if __name__ == "__main__":
    main()
