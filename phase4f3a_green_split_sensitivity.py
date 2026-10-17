"""Run the Phase 4F-3A Junction C green-split sensitivity experiment."""

import os
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

from phase4b_multi_seed import bootstrap_mean_ci
from phase4f2_multi_seed_full_junction import load_inputs, make_arrivals
from traffic_signals.green_split_sensitivity import (
    modify_c_green,
    theoretical_capacity_vph,
)
from traffic_signals.network_evaluation import (
    build_period_plans,
    evaluate_network,
)


DATA_DIR = Path("outputs/data")
FIGURE_DIR = Path("outputs/figures")
PERIODS = ("AM peak", "PM peak")
EXTRA_GREEN = (0, 3, 5, 7, 10)
SEEDS = range(300, 312)
HORIZON_S = 3600.0

MOVEMENT_METRICS_PATH = DATA_DIR / "phase4f3a_movement_metrics.csv"
NETWORK_METRICS_PATH = DATA_DIR / "phase4f3a_network_metrics.csv"
CAPACITY_PATH = DATA_DIR / "phase4f3a_capacity_scenarios.csv"
PAIRED_NETWORK_PATH = DATA_DIR / "phase4f3a_paired_network.csv"
NETWORK_SUMMARY_PATH = DATA_DIR / "phase4f3a_network_summary.csv"
PAIRED_MOVEMENTS_PATH = DATA_DIR / "phase4f3a_paired_movements.csv"
MOVEMENT_SUMMARY_PATH = DATA_DIR / "phase4f3a_movement_summary.csv"


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run 120 matched full-junction simulations and save raw metrics."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    traffic, plans_3a, plans_4e = load_inputs()
    movement_frames: list[pd.DataFrame] = []
    network_records: list[dict[str, object]] = []
    capacity_records: list[dict[str, object]] = []

    for period in PERIODS:
        base_plans = build_period_plans(plans_3a, plans_4e, period)["Phase4E"]
        for extra in EXTRA_GREEN:
            plans = modify_c_green(base_plans, extra)
            c_plan = plans["C"]
            capacity_records.append(
                {
                    "period": period,
                    "extra_ns_green_s": extra,
                    "cycle_s": c_plan["cycle_s"],
                    "lost_time_s": c_plan["lost_time_s"],
                    "C_ew_green_s": c_plan["ew_green_s"],
                    "C_ns_green_s": c_plan["ns_green_s"],
                    "C_ew_offset_s": c_plan["ew_offset_s"],
                    "C_ns_offset_s": c_plan["ns_offset_s"],
                    "C_ew_capacity_vph": theoretical_capacity_vph(
                        c_plan["ew_green_s"], c_plan["cycle_s"]
                    ),
                    "C_ns_capacity_vph": theoretical_capacity_vph(
                        c_plan["ns_green_s"], c_plan["cycle_s"]
                    ),
                }
            )

        for seed in SEEDS:
            arrivals = make_arrivals(traffic, period, seed)
            for extra in EXTRA_GREEN:
                plans = modify_c_green(base_plans, extra)
                strategy = f"Phase4E_C_NS_plus_{extra}s"
                movement, network = evaluate_network(
                    arrivals=arrivals,
                    junction_plans=plans,
                    strategy=strategy,
                    period=period,
                    seed=seed,
                    horizon_s=HORIZON_S,
                )
                movement.insert(3, "extra_ns_green_s", extra)
                network["extra_ns_green_s"] = extra
                movement_frames.append(movement)
                network_records.append(network)
        print(f"Completed {period}")

    movements = pd.concat(movement_frames, ignore_index=True)
    network = pd.DataFrame(network_records)
    capacities = pd.DataFrame(capacity_records)
    if len(movements) != 960:
        raise ValueError("Expected 960 movement metric rows.")
    if len(network) != 120:
        raise ValueError("Expected 120 network metric rows.")
    if len(capacities) != 10:
        raise ValueError("Expected 10 capacity scenarios.")

    matched_counts = movements.groupby(
        ["period", "seed", "movement"]
    )["vehicles"].nunique()
    if not matched_counts.eq(1).all():
        raise ValueError("Green scenarios received different vehicle counts.")

    c_queues = (
        movements.loc[movements["movement"].isin(["C_N", "C_S"])]
        .groupby(["period", "seed", "extra_ns_green_s"], as_index=False)
        .agg(C_ns_queue_at_hour=("queue_at_hour_veh", "sum"))
    )
    network = network.merge(
        c_queues,
        on=["period", "seed", "extra_ns_green_s"],
        validate="one_to_one",
    )

    movements.to_csv(MOVEMENT_METRICS_PATH, index=False)
    network.to_csv(NETWORK_METRICS_PATH, index=False)
    capacities.to_csv(CAPACITY_PATH, index=False)
    return movements, network, capacities


def summarise_network(
    network: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair every green scenario with the unchanged Phase 4E baseline."""

    baseline = network.loc[network["extra_ns_green_s"] == 0].drop(
        columns=["strategy", "extra_ns_green_s"]
    )
    candidates = network.drop(columns="strategy")
    paired = candidates.merge(
        baseline,
        on=["period", "seed"],
        suffixes=("_candidate", "_baseline"),
        validate="many_to_one",
    )
    if len(paired) != 120:
        raise ValueError("Expected 120 paired network rows.")
    for count_field in ("corridor_vehicles", "ns_vehicles", "total_passages"):
        if not paired[f"{count_field}_candidate"].eq(
            paired[f"{count_field}_baseline"]
        ).all():
            raise ValueError(f"Paired {count_field} values differ.")

    paired["delay_change_s"] = (
        paired["mean_delay_per_passage_s_candidate"]
        - paired["mean_delay_per_passage_s_baseline"]
    )
    paired["queue_change_veh"] = (
        paired["total_queue_at_hour_candidate"]
        - paired["total_queue_at_hour_baseline"]
    )
    paired["C_ns_queue_change_veh"] = (
        paired["C_ns_queue_at_hour_candidate"]
        - paired["C_ns_queue_at_hour_baseline"]
    )
    paired["corridor_delay_change_s"] = (
        paired["corridor_mean_delay_s_candidate"]
        - paired["corridor_mean_delay_s_baseline"]
    )
    paired["ns_delay_change_s"] = (
        paired["ns_mean_delay_s_candidate"]
        - paired["ns_mean_delay_s_baseline"]
    )

    records: list[dict[str, object]] = []
    for period in PERIODS:
        for extra in EXTRA_GREEN:
            group = paired.loc[
                (paired["period"] == period)
                & (paired["extra_ns_green_s"] == extra)
            ]
            delay_changes = group["delay_change_s"].to_numpy()
            queue_changes = group["queue_change_veh"].to_numpy()
            c_queue_changes = group["C_ns_queue_change_veh"].to_numpy()
            delay_low, delay_high = bootstrap_mean_ci(delay_changes)
            queue_low, queue_high = bootstrap_mean_ci(queue_changes)
            c_queue_low, c_queue_high = bootstrap_mean_ci(c_queue_changes)
            records.append(
                {
                    "period": period,
                    "extra_ns_green_s": extra,
                    "replications": len(group),
                    "baseline_delay_per_passage_s": group[
                        "mean_delay_per_passage_s_baseline"
                    ].mean(),
                    "candidate_delay_per_passage_s": group[
                        "mean_delay_per_passage_s_candidate"
                    ].mean(),
                    "mean_delay_change_s": delay_changes.mean(),
                    "ci_lower_s": delay_low,
                    "ci_upper_s": delay_high,
                    "seeds_better": int((delay_changes < 0).sum()),
                    "corridor_delay_change_s": group[
                        "corridor_delay_change_s"
                    ].mean(),
                    "ns_delay_change_s": group["ns_delay_change_s"].mean(),
                    "baseline_queue_at_hour": group[
                        "total_queue_at_hour_baseline"
                    ].mean(),
                    "candidate_queue_at_hour": group[
                        "total_queue_at_hour_candidate"
                    ].mean(),
                    "mean_queue_change_veh": queue_changes.mean(),
                    "queue_ci_lower_veh": queue_low,
                    "queue_ci_upper_veh": queue_high,
                    "baseline_C_ns_queue_at_hour": group[
                        "C_ns_queue_at_hour_baseline"
                    ].mean(),
                    "candidate_C_ns_queue_at_hour": group[
                        "C_ns_queue_at_hour_candidate"
                    ].mean(),
                    "mean_C_ns_queue_change_veh": c_queue_changes.mean(),
                    "C_ns_queue_ci_lower_veh": c_queue_low,
                    "C_ns_queue_ci_upper_veh": c_queue_high,
                }
            )

    summary = pd.DataFrame(records)
    if len(summary) != 10 or not summary["replications"].eq(12).all():
        raise ValueError("Expected 10 network summaries with 12 replications.")
    paired.to_csv(PAIRED_NETWORK_PATH, index=False)
    summary.to_csv(NETWORK_SUMMARY_PATH, index=False)
    return paired, summary


def summarise_movements(
    movements: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair movement outcomes with the zero-transfer Phase 4E baseline."""

    baseline = movements.loc[movements["extra_ns_green_s"] == 0].drop(
        columns=["strategy", "extra_ns_green_s"]
    )
    candidates = movements.drop(columns="strategy")
    paired = candidates.merge(
        baseline,
        on=["period", "seed", "movement"],
        suffixes=("_candidate", "_baseline"),
        validate="many_to_one",
    )
    if len(paired) != 960:
        raise ValueError("Expected 960 paired movement rows.")
    if not paired["vehicles_candidate"].eq(paired["vehicles_baseline"]).all():
        raise ValueError("Paired movement vehicle counts differ.")
    paired["delay_change_s"] = (
        paired["mean_delay_per_vehicle_s_candidate"]
        - paired["mean_delay_per_vehicle_s_baseline"]
    )
    paired["queue_change_veh"] = (
        paired["queue_at_hour_veh_candidate"]
        - paired["queue_at_hour_veh_baseline"]
    )

    records: list[dict[str, object]] = []
    movements_order = ("EB", "WB", "A_N", "A_S", "B_N", "B_S", "C_N", "C_S")
    for period in PERIODS:
        for extra in EXTRA_GREEN:
            for movement in movements_order:
                group = paired.loc[
                    (paired["period"] == period)
                    & (paired["extra_ns_green_s"] == extra)
                    & (paired["movement"] == movement)
                ]
                delay_changes = group["delay_change_s"].to_numpy()
                queue_changes = group["queue_change_veh"].to_numpy()
                delay_low, delay_high = bootstrap_mean_ci(delay_changes)
                queue_low, queue_high = bootstrap_mean_ci(queue_changes)
                records.append(
                    {
                        "period": period,
                        "extra_ns_green_s": extra,
                        "movement": movement,
                        "category": group["category_candidate"].iloc[0],
                        "replications": len(group),
                        "baseline_mean_delay_s": group[
                            "mean_delay_per_vehicle_s_baseline"
                        ].mean(),
                        "candidate_mean_delay_s": group[
                            "mean_delay_per_vehicle_s_candidate"
                        ].mean(),
                        "mean_delay_change_s": delay_changes.mean(),
                        "ci_lower_s": delay_low,
                        "ci_upper_s": delay_high,
                        "seeds_better": int((delay_changes < 0).sum()),
                        "baseline_queue_at_hour": group[
                            "queue_at_hour_veh_baseline"
                        ].mean(),
                        "candidate_queue_at_hour": group[
                            "queue_at_hour_veh_candidate"
                        ].mean(),
                        "mean_queue_change_veh": queue_changes.mean(),
                        "queue_ci_lower_veh": queue_low,
                        "queue_ci_upper_veh": queue_high,
                    }
                )

    summary = pd.DataFrame(records)
    if len(summary) != 80 or not summary["replications"].eq(12).all():
        raise ValueError("Expected 80 movement summaries with 12 replications.")
    paired.to_csv(PAIRED_MOVEMENTS_PATH, index=False)
    summary.to_csv(MOVEMENT_SUMMARY_PATH, index=False)
    return paired, summary


def _period_slug(period: str) -> str:
    return period.lower().replace(" ", "_")


def plot_results(network_summary: pd.DataFrame) -> None:
    """Save delay and Junction C residual-queue sensitivity figures."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for period in PERIODS:
        selected = network_summary.loc[
            network_summary["period"] == period
        ].sort_values("extra_ns_green_s")
        x = selected["extra_ns_green_s"].to_numpy(dtype=float)

        delay = selected["mean_delay_change_s"].to_numpy(dtype=float)
        delay_low = selected["ci_lower_s"].to_numpy(dtype=float)
        delay_high = selected["ci_upper_s"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, delay, marker="o")
        ax.fill_between(x, delay_low, delay_high, alpha=0.2)
        ax.axhline(0, linestyle="--", linewidth=1, color="black")
        ax.set_title(f"Synthetic Junction C Green-Split Sensitivity: {period}")
        ax.set_xlabel("Additional NS effective green at C (s)")
        ax.set_ylabel("Change in network delay per passage (s)")
        ax.set_xticks(EXTRA_GREEN)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / f"phase4f3a_network_delay_{_period_slug(period)}.png",
            dpi=180,
        )
        plt.close(fig)

        queue = selected["mean_C_ns_queue_change_veh"].to_numpy(dtype=float)
        queue_low = selected["C_ns_queue_ci_lower_veh"].to_numpy(dtype=float)
        queue_high = selected["C_ns_queue_ci_upper_veh"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, queue, marker="o", color="tab:red")
        ax.fill_between(x, queue_low, queue_high, alpha=0.2, color="tab:red")
        ax.axhline(0, linestyle="--", linewidth=1, color="black")
        ax.set_title(f"Synthetic Junction C NS Residual Queue: {period}")
        ax.set_xlabel("Additional NS effective green at C (s)")
        ax.set_ylabel("Change in C north-south queue at hour end (vehicles)")
        ax.set_xticks(EXTRA_GREEN)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / f"phase4f3a_c_ns_queue_{_period_slug(period)}.png",
            dpi=180,
        )
        plt.close(fig)


def main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the sensitivity experiment, save summaries and print key results."""

    movements, network, capacities = run_experiment()
    _, network_summary = summarise_network(network)
    _, movement_summary = summarise_movements(movements)
    plot_results(network_summary)
    print("\nTheoretical approach capacities:")
    print(capacities.to_string(index=False))
    print("\nNetwork sensitivity:")
    print(network_summary.to_string(index=False))
    print("\nKey movement-level results:")
    print(
        movement_summary.loc[
            movement_summary["movement"].isin(["EB", "WB", "C_N", "C_S"])
        ].to_string(index=False)
    )
    print("\nPhase 4F-3A completed.")
    return network_summary, movement_summary, capacities


if __name__ == "__main__":
    main()
