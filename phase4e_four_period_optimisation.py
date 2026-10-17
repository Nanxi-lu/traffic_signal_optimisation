"""Select and validate fairness-protected offsets for four operating periods."""

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "traffic-signals-matplotlib")
)
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase4b_multi_seed import (
    bootstrap_mean_ci,
    generate_arrivals,
    get_signal_strategies,
)
from phase4d3_robust_fairness import (
    check_robust_fairness,
    compare_candidates,
    evaluate_candidate,
)
from traffic_signals.demand_stress import create_demand_scenarios
from traffic_signals.offset_optimisation import circular_distance
from traffic_signals.segmentation import assign_period


DATA_DIR = Path("outputs/data")
FIGURE_DIR = Path("outputs/figures")
PERIODS = ("Overnight", "AM peak", "Interpeak", "PM peak")
OPTIMISE_PERIODS = ("Overnight", "Interpeak", "PM peak")
FACTORS = (0.75, 1.0, 1.25, 1.5)
SCREEN_FACTORS = (1.0, 1.5)
TRAIN_SEEDS = range(144, 156)
HOLDOUT_SEEDS = range(156, 186)
FAIRNESS_LIMIT_S = 2.0
SCREEN_LIMIT_S = 3.0
DURATION_S = 3600.0
ENTRY_APPROACHES = {"EB": ("A", "E"), "WB": ("C", "W")}

TRAFFIC_PATH = DATA_DIR / "synthetic_traffic_15min.csv"
PHASE3A_PATH = DATA_DIR / "phase3a_signal_plans.csv"
PHASE3F_PATH = DATA_DIR / "phase3f_optimisation_summary.csv"
FROZEN_AM_PATH = DATA_DIR / "phase4d3_selected_plan.csv"
SCREENING_PATH = DATA_DIR / "phase4e_screening.csv"
TRAINING_PATH = DATA_DIR / "phase4e_training_candidates.csv"
SELECTED_PATH = DATA_DIR / "phase4e_selected_plans.csv"
HOLDOUT_PATH = DATA_DIR / "phase4e_holdout_metrics.csv"
SUMMARY_PATH = DATA_DIR / "phase4e_holdout_summary.csv"


def prepare_period_arrivals(
    traffic: pd.DataFrame,
    period: str,
    seeds: Iterable[int],
) -> dict[int, dict[float, dict[str, np.ndarray]]]:
    """Create matched nested demand scenarios for one operating period."""

    if period not in PERIODS:
        raise ValueError(f"Unknown operating period: {period}.")
    required = {"period", "intersection_id", "approach", "volume_15min"}
    if not required.issubset(traffic.columns):
        raise ValueError("Missing required traffic columns.")

    rates: dict[str, float] = {}
    for direction, (junction, approach) in ENTRY_APPROACHES.items():
        selected = traffic.loc[
            (traffic["period"] == period)
            & (traffic["intersection_id"] == junction)
            & (traffic["approach"] == approach)
        ]
        if selected.empty:
            raise ValueError(f"Missing traffic data for {period}, {direction}.")
        counts = pd.to_numeric(
            selected["volume_15min"], errors="raise"
        ).to_numpy()
        if not np.isfinite(counts).all() or np.any(counts < 0):
            raise ValueError("Invalid traffic counts.")
        rates[direction] = float(counts.sum() / (len(counts) * 900.0))

    period_index = PERIODS.index(period)
    all_scenarios: dict[int, dict[float, dict[str, np.ndarray]]] = {}
    for seed in seeds:
        baseline = generate_arrivals(traffic, period, seed)
        by_direction: dict[str, dict[float, np.ndarray]] = {}
        for direction_index, direction in enumerate(("EB", "WB")):
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [int(seed), period_index, direction_index, 405]
                )
            )
            by_direction[direction] = create_demand_scenarios(
                baseline_arrivals=baseline[direction],
                baseline_rate=rates[direction],
                duration_s=DURATION_S,
                multipliers=FACTORS,
                rng=rng,
            )
        all_scenarios[int(seed)] = {
            factor: {
                direction: by_direction[direction][factor]
                for direction in ("EB", "WB")
            }
            for factor in FACTORS
        }
    return all_scenarios


def candidate_space(
    cycle: int,
    original: Mapping[str, float],
    phase3f: Mapping[str, float],
    radius: int = 6,
) -> list[tuple[int, int]]:
    """Generate local whole-second candidates around Phase 3A and Phase 3F."""

    if not isinstance(cycle, (int, np.integer)) or cycle <= 0:
        raise ValueError("Cycle must be a positive integer.")
    if not isinstance(radius, (int, np.integer)) or radius < 0:
        raise ValueError("Radius must be a non-negative integer.")
    candidates: set[tuple[int, int]] = set()
    for reference in (original, phase3f):
        if not {"B", "C"}.issubset(reference):
            raise ValueError("Reference offsets must include B and C.")
        b_offset = int(reference["B"])
        c_offset = int(reference["C"])
        for b_change in range(-radius, radius + 1, 2):
            for c_change in range(-radius, radius + 1, 2):
                candidates.add(
                    (
                        (b_offset + b_change) % cycle,
                        (c_offset + c_change) % cycle,
                    )
                )
    candidates.add((int(original["B"]) % cycle, int(original["C"]) % cycle))
    candidates.add((int(phase3f["B"]) % cycle, int(phase3f["C"]) % cycle))
    return sorted(candidates)


def _score_components(
    paired: pd.DataFrame,
    pair: tuple[int, int],
    original_pair: tuple[int, int],
    cycle: int,
) -> tuple[float, float, float]:
    """Return equal-scenario delay, queue and circular offset-change scores."""

    combined = paired.loc[paired["direction"] == "Combined"]
    mean_delay = float(combined["delay_change_s"].mean())
    mean_queue = float(combined["queue_change_veh"].mean())
    offset_change = circular_distance(
        pair[0], original_pair[0], cycle
    ) + circular_distance(pair[1], original_pair[1], cycle)
    return mean_delay, mean_queue, offset_change


def screen_period(
    period: str,
    arrivals: Mapping[int, Mapping[float, Mapping[str, np.ndarray]]],
    strategies: Mapping[str, Mapping[str, object]],
) -> tuple[list[tuple[int, int]], pd.DataFrame]:
    """Apply the documented Phase 4D-3-style relaxed four-seed screen."""

    base_plan = strategies["Phase3A"]
    cycle = int(base_plan["cycle_s"])
    original = base_plan["offsets"]
    previous = strategies["Phase3F"]["offsets"]
    original_pair = (int(original["B"]), int(original["C"]))
    candidates = candidate_space(cycle, original, previous)
    screen_seed_values = tuple(TRAIN_SEEDS)[:4]
    screen_arrivals = {seed: arrivals[seed] for seed in screen_seed_values}
    baseline = evaluate_candidate(
        screen_arrivals,
        base_plan,
        original_pair,
        factors=SCREEN_FACTORS,
    )

    records: list[dict[str, object]] = []
    for pair in candidates:
        tested = evaluate_candidate(
            screen_arrivals,
            base_plan,
            pair,
            factors=SCREEN_FACTORS,
        )
        paired = compare_candidates(baseline, tested)
        directional_means = (
            paired.loc[paired["direction"].isin(["EB", "WB"])]
            .groupby(["factor", "direction"])["delay_change_s"]
            .mean()
        )
        combined_changes = (
            paired.loc[paired["direction"] == "Combined"]
            .groupby("factor")["delay_change_s"]
            .mean()
        )
        relaxed_fairness = bool(
            (directional_means <= SCREEN_LIMIT_S + 1e-9).all()
        )
        performance = bool((combined_changes <= 1e-9).all())
        mean_delay, mean_queue, offset_change = _score_components(
            paired, pair, original_pair, cycle
        )
        records.append(
            {
                "period": period,
                "B_offset_s": pair[0],
                "C_offset_s": pair[1],
                "max_directional_mean_change_s": directional_means.max(),
                "mean_combined_delay_change_s": mean_delay,
                "mean_residual_queue_change_veh": mean_queue,
                "offset_change_s": offset_change,
                "relaxed_fairness_pass": relaxed_fairness,
                "performance_pass": performance,
                "shortlisted": relaxed_fairness and performance,
            }
        )

    screening = pd.DataFrame(records).sort_values(
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
    screening["screen_rank"] = np.arange(1, len(screening) + 1)
    shortlist = [
        (int(row.B_offset_s), int(row.C_offset_s))
        for row in screening.itertuples()
        if row.shortlisted
    ]
    if original_pair not in shortlist:
        raise RuntimeError(f"Phase 3A fallback failed screening for {period}.")
    return shortlist, screening


def optimise_period(
    period: str,
    arrivals: Mapping[int, Mapping[float, Mapping[str, np.ndarray]]],
    strategies: Mapping[str, Mapping[str, object]],
    shortlist: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], pd.DataFrame]:
    """Select the best fully feasible training candidate for one period."""

    base_plan = strategies["Phase3A"]
    cycle = int(base_plan["cycle_s"])
    original = base_plan["offsets"]
    original_pair = (int(original["B"]), int(original["C"]))
    baseline = evaluate_candidate(arrivals, base_plan, original_pair)
    records: list[dict[str, object]] = []
    feasible: list[tuple[tuple[float, ...], tuple[int, int]]] = []

    for pair in shortlist:
        tested = evaluate_candidate(arrivals, base_plan, pair)
        paired = compare_candidates(baseline, tested)
        fairness_pass, checks = check_robust_fairness(
            paired, limit_s=FAIRNESS_LIMIT_S
        )
        combined = paired.loc[paired["direction"] == "Combined"]
        factor_changes = combined.groupby("factor")["delay_change_s"].mean()
        performance_pass = bool((factor_changes <= 1e-9).all())
        mean_delay, mean_queue, offset_change = _score_components(
            paired, pair, original_pair, cycle
        )
        is_feasible = fairness_pass and performance_pass
        record: dict[str, object] = {
            "period": period,
            "B_offset_s": pair[0],
            "C_offset_s": pair[1],
            "fairness_pass": fairness_pass,
            "performance_pass": performance_pass,
            "feasible": is_feasible,
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
        records.append(record)
        if is_feasible:
            score = (
                round(mean_delay, 8),
                round(mean_queue, 8),
                offset_change,
                pair[0],
                pair[1],
            )
            feasible.append((score, pair))

    if not feasible:
        raise RuntimeError(f"No feasible plan found for {period}.")
    feasible.sort()
    selected = feasible[0][1]
    training = pd.DataFrame(records).sort_values(
        [
            "feasible",
            "mean_combined_delay_change_s",
            "mean_residual_queue_change_veh",
            "offset_change_s",
        ],
        ascending=[False, True, True, True],
        ignore_index=True,
    )
    training["selected"] = (
        (training["B_offset_s"] == selected[0])
        & (training["C_offset_s"] == selected[1])
    )
    return selected, training


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate the existing Phase 3 and frozen AM inputs."""

    traffic = pd.read_csv(TRAFFIC_PATH, parse_dates=["time"])
    traffic["period"] = traffic["time"].apply(assign_period)
    traffic["intersection_id"] = traffic["intersection_id"].astype(str).str.upper()
    traffic["approach"] = (
        traffic["approach"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"EASTBOUND": "E", "WESTBOUND": "W"})
    )
    phase3a = pd.read_csv(PHASE3A_PATH)
    phase3f = pd.read_csv(PHASE3F_PATH)
    frozen_am = pd.read_csv(FROZEN_AM_PATH)
    if len(frozen_am) != 1:
        raise ValueError("Expected exactly one frozen Phase 4D-3 AM plan.")
    row = frozen_am.iloc[0]
    frozen_pair = (int(row["B_offset_s"]), int(row["C_offset_s"]))
    if frozen_pair != (36, 70) or int(row["A_offset_s"]) != 0:
        raise ValueError("Frozen AM offsets must be 0/36/70 seconds.")
    if int(row["cycle_s"]) != 150:
        raise ValueError("Frozen AM cycle must be 150 seconds.")
    return traffic, phase3a, phase3f, frozen_am


def build_four_period_strategy() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, tuple[int, int]],
]:
    """Optimise three periods, retain the frozen AM plan and save selections."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    traffic, phase3a, phase3f, frozen_am = _load_inputs()
    selected_pairs: dict[str, tuple[int, int]] = {
        "AM peak": (
            int(frozen_am.iloc[0]["B_offset_s"]),
            int(frozen_am.iloc[0]["C_offset_s"]),
        )
    }
    screening_frames: list[pd.DataFrame] = []
    training_frames: list[pd.DataFrame] = []
    selection_records: list[dict[str, object]] = []

    for period in PERIODS:
        strategies = get_signal_strategies(phase3a, phase3f, period)
        base_plan = strategies["Phase3A"]
        original = base_plan["offsets"]
        if period == "AM peak":
            row = frozen_am.iloc[0]
            selection_records.append(
                {
                    "period": period,
                    "cycle_s": int(base_plan["cycle_s"]),
                    "A_offset_s": 0,
                    "B_offset_s": selected_pairs[period][0],
                    "C_offset_s": selected_pairs[period][1],
                    "source": "Phase4D3_frozen",
                    "optimised_in_phase4e": False,
                    "candidate_count": 0,
                    "shortlist_count": 0,
                    "training_fairness_pass": bool(row["training_fairness_pass"]),
                    "training_performance_pass": bool(
                        row["training_performance_pass"]
                    ),
                    "training_mean_combined_delay_change_s": row[
                        "training_mean_combined_delay_change_s"
                    ],
                    "training_mean_residual_queue_change_veh": row[
                        "training_mean_residual_queue_change_veh"
                    ],
                }
            )
            continue

        arrivals = prepare_period_arrivals(traffic, period, TRAIN_SEEDS)
        shortlist, screening = screen_period(period, arrivals, strategies)
        selected, training = optimise_period(
            period, arrivals, strategies, shortlist
        )
        selected_pairs[period] = selected
        screening_frames.append(screening)
        training_frames.append(training)
        selected_row = training.loc[training["selected"]].iloc[0]
        selection_records.append(
            {
                "period": period,
                "cycle_s": int(base_plan["cycle_s"]),
                "A_offset_s": 0,
                "B_offset_s": selected[0],
                "C_offset_s": selected[1],
                "source": "Phase4E_optimised",
                "optimised_in_phase4e": True,
                "candidate_count": len(screening),
                "shortlist_count": len(shortlist),
                "training_fairness_pass": bool(selected_row["fairness_pass"]),
                "training_performance_pass": bool(
                    selected_row["performance_pass"]
                ),
                "training_mean_combined_delay_change_s": selected_row[
                    "mean_combined_delay_change_s"
                ],
                "training_mean_residual_queue_change_veh": selected_row[
                    "mean_residual_queue_change_veh"
                ],
            }
        )
        print(
            f"Selected {period}: 0/{selected[0]}/{selected[1]} "
            f"from {len(shortlist)} shortlisted candidates"
        )

    screening_results = pd.concat(screening_frames, ignore_index=True)
    training_results = pd.concat(training_frames, ignore_index=True)
    selected_plans = pd.DataFrame(selection_records)
    selected_plans["period"] = pd.Categorical(
        selected_plans["period"], categories=PERIODS, ordered=True
    )
    selected_plans = selected_plans.sort_values("period").reset_index(drop=True)
    selected_plans["period"] = selected_plans["period"].astype(str)

    screening_results.to_csv(SCREENING_PATH, index=False)
    training_results.to_csv(TRAINING_PATH, index=False)
    selected_plans.to_csv(SELECTED_PATH, index=False)
    return traffic, phase3a, phase3f, selected_pairs


def _summarise_comparison(
    paired: pd.DataFrame,
    period: str,
    benchmark: str,
) -> list[dict[str, object]]:
    """Summarise holdout changes for one period and benchmark."""

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
            baseline_delay = float(group["mean_delay_s_3a"].mean())
            mean_change = float(group["delay_change_s"].mean())
            fairness_applicable = (
                benchmark == "Phase3A" and direction in ("EB", "WB")
            )
            records.append(
                {
                    "period": period,
                    "factor": factor,
                    "direction": direction,
                    "benchmark": benchmark,
                    "replicates": len(group),
                    "baseline_delay_s": baseline_delay,
                    "phase4e_delay_s": group["mean_delay_s_candidate"].mean(),
                    "mean_delay_change_s": mean_change,
                    "mean_delay_change_pct": 100.0 * mean_change / baseline_delay,
                    "ci_lower_s": low,
                    "ci_upper_s": high,
                    "seeds_better": int((group["delay_change_s"] < 0).sum()),
                    "baseline_queue_at_hour_veh": group[
                        "queue_at_hour_veh_3a"
                    ].mean(),
                    "phase4e_queue_at_hour_veh": group[
                        "queue_at_hour_veh_candidate"
                    ].mean(),
                    "mean_queue_change_veh": group["queue_change_veh"].mean(),
                    "fairness_applicable": fairness_applicable,
                    "fairness_pass": (
                        bool(high <= FAIRNESS_LIMIT_S + 1e-9)
                        if fairness_applicable
                        else None
                    ),
                }
            )
    return records


def validate_four_period_strategy(
    traffic: pd.DataFrame,
    phase3a: pd.DataFrame,
    phase3f: pd.DataFrame,
    selected_pairs: Mapping[str, tuple[int, int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the frozen four-period strategy on untouched holdout seeds."""

    metric_frames: list[pd.DataFrame] = []
    summary_records: list[dict[str, object]] = []
    for period in PERIODS:
        strategies = get_signal_strategies(phase3a, phase3f, period)
        base_plan = strategies["Phase3A"]
        original = strategies["Phase3A"]["offsets"]
        previous = strategies["Phase3F"]["offsets"]
        pairs = {
            "Phase3A": (int(original["B"]), int(original["C"])),
            "Phase3F": (int(previous["B"]), int(previous["C"])),
            "Phase4E": selected_pairs[period],
        }
        arrivals = prepare_period_arrivals(traffic, period, HOLDOUT_SEEDS)
        period_frames: dict[str, pd.DataFrame] = {}
        for strategy, pair in pairs.items():
            metrics = evaluate_candidate(arrivals, base_plan, pair)
            metrics["period"] = period
            metrics["strategy"] = strategy
            period_frames[strategy] = metrics
            metric_frames.append(metrics)

        candidate = period_frames["Phase4E"].drop(columns=["period", "strategy"])
        for benchmark in ("Phase3A", "Phase3F"):
            baseline = period_frames[benchmark].drop(
                columns=["period", "strategy"]
            )
            paired = compare_candidates(baseline, candidate)
            summary_records.extend(
                _summarise_comparison(paired, period, benchmark)
            )
        print(f"Completed holdout validation for {period}")

    metrics = pd.concat(metric_frames, ignore_index=True)
    summary = pd.DataFrame(summary_records)
    if len(metrics) != 4320:
        raise ValueError("Expected 4,320 holdout metric rows.")
    if len(summary) != 96 or not summary["replicates"].eq(30).all():
        raise ValueError("Expected 96 holdout summary rows.")
    metrics.to_csv(HOLDOUT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    return metrics, summary


def plot_holdout_results(metrics: pd.DataFrame) -> None:
    """Save one synthetic combined-delay comparison figure per period."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for period in PERIODS:
        selected = metrics.loc[
            (metrics["period"] == period)
            & (metrics["direction"] == "Combined")
        ]
        means = (
            selected.groupby(["strategy", "factor"])["mean_delay_s"]
            .mean()
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        for strategy in ("Phase3A", "Phase3F", "Phase4E"):
            subset = means.loc[means["strategy"] == strategy].sort_values(
                "factor"
            )
            ax.plot(
                subset["factor"] * 100,
                subset["mean_delay_s"],
                marker="o",
                label=strategy,
            )
        ax.set_title(f"Phase 4E: Synthetic simulated delay in {period}")
        ax.set_xlabel("Demand relative to synthetic baseline (%)")
        ax.set_ylabel("Combined mean simulated delay (seconds)")
        ax.set_xticks([75, 100, 125, 150])
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        filename = period.lower().replace(" ", "_")
        fig.savefig(FIGURE_DIR / f"phase4e_delay_{filename}.png", dpi=180)
        plt.close(fig)


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build, freeze, validate and plot the Phase 4E strategy."""

    traffic, phase3a, phase3f, selected = build_four_period_strategy()
    metrics, summary = validate_four_period_strategy(
        traffic, phase3a, phase3f, selected
    )
    plot_holdout_results(metrics)
    print("\nSelected four-period offsets:")
    print(pd.read_csv(SELECTED_PATH).to_string(index=False))
    print("\nHoldout comparison against Phase 3A:")
    print(summary.loc[summary["benchmark"] == "Phase3A"].to_string(index=False))
    print("\nPhase 4E completed.")
    return metrics, summary


if __name__ == "__main__":
    main()
