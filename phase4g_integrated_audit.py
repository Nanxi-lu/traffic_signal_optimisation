"""Audit the frozen integrated signal strategy against Phase 3A and Phase 4E."""

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
from phase4f3b_joint_optimisation import (
    FAIRNESS_LIMIT_S,
    PROTECTED_MOVEMENTS,
    make_candidate_plan,
)
from traffic_signals.network_evaluation import build_period_plans, evaluate_network


DATA_DIR = Path("outputs/data")
FIGURE_DIR = Path("outputs/figures")
PERIODS = ("Overnight", "AM peak", "Interpeak", "PM peak")
STRATEGIES = ("Phase3A", "Phase4E", "Integrated")
SEEDS = range(500, 530)
HORIZON_S = 3600.0
ALL_MOVEMENTS = ("EB", "WB", "A_N", "A_S", "B_N", "B_S", "C_N", "C_S")

SIGNAL_PLANS_PATH = DATA_DIR / "phase4g_signal_plans.csv"
NETWORK_METRICS_PATH = DATA_DIR / "phase4g_network_metrics.csv"
MOVEMENT_METRICS_PATH = DATA_DIR / "phase4g_movement_metrics.csv"
NETWORK_SUMMARY_PATH = DATA_DIR / "phase4g_network_summary.csv"
MOVEMENT_SUMMARY_PATH = DATA_DIR / "phase4g_movement_summary.csv"
FAIRNESS_AUDIT_PATH = DATA_DIR / "phase4g_fairness_audit.csv"
FIGURE_PATH = FIGURE_DIR / "phase4g_integrated_network.png"


def build_integrated_plans(
    plans_3a: pd.DataFrame,
    plans_4e: pd.DataFrame,
    period: str,
) -> dict[str, dict[str, dict[str, float]]]:
    """Create all three frozen strategies without modifying saved plans."""

    if period not in PERIODS:
        raise ValueError(f"Unknown period: {period}.")
    strategies = build_period_plans(plans_3a, plans_4e, period)
    integrated = {
        junction: dict(plan)
        for junction, plan in strategies["Phase4E"].items()
    }
    if period == "PM peak":
        integrated = make_candidate_plan(
            base_plans=integrated,
            extra_ns=3,
            c_offset=83,
        )
        c_plan = integrated["C"]
        expected = {
            "ew_green_s": 89.0,
            "ns_green_s": 49.0,
            "ew_offset_s": 83.0,
            "ns_offset_s": 28.0,
        }
        for field, value in expected.items():
            if not np.isclose(c_plan[field], value):
                raise AssertionError(f"Unexpected integrated PM C {field}.")
    elif integrated != strategies["Phase4E"]:
        raise AssertionError("Non-PM integrated plan must equal Phase 4E.")
    strategies["Integrated"] = integrated
    return strategies


def run_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run 360 matched full-junction simulations using frozen signal plans."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    traffic, plans_3a, plans_4e = load_inputs()
    movement_frames: list[pd.DataFrame] = []
    network_records: list[dict[str, object]] = []
    signal_records: list[dict[str, object]] = []

    for period in PERIODS:
        strategies = build_integrated_plans(plans_3a, plans_4e, period)
        if tuple(strategies) != STRATEGIES:
            raise AssertionError("Unexpected strategy order or names.")
        for name, junction_plans in strategies.items():
            for junction, plan in junction_plans.items():
                signal_records.append(
                    {
                        "period": period,
                        "strategy": name,
                        "junction": junction,
                        **plan,
                    }
                )
        for seed in SEEDS:
            arrivals = make_arrivals(traffic, period, seed)
            for strategy, junction_plans in strategies.items():
                movements, network = evaluate_network(
                    arrivals=arrivals,
                    junction_plans=junction_plans,
                    strategy=strategy,
                    period=period,
                    seed=seed,
                    horizon_s=HORIZON_S,
                )
                movement_frames.append(movements)
                network_records.append(network)
        print(f"Completed {period}")

    movements = pd.concat(movement_frames, ignore_index=True)
    network = pd.DataFrame(network_records)
    signal_plans = pd.DataFrame(signal_records)
    if len(movements) != 2880:
        raise ValueError("Expected 2,880 movement metric rows.")
    if len(network) != 360:
        raise ValueError("Expected 360 network metric rows.")
    if len(signal_plans) != 36:
        raise ValueError("Expected 36 signal-plan rows.")

    movement_counts = movements.pivot(
        index=["period", "seed", "movement"],
        columns="strategy",
        values="vehicles",
    )
    if not movement_counts.nunique(axis=1).eq(1).all():
        raise ValueError("Strategies received different movement demand.")
    passage_counts = network.pivot(
        index=["period", "seed"],
        columns="strategy",
        values="total_passages",
    )
    if not passage_counts.nunique(axis=1).eq(1).all():
        raise ValueError("Strategies used different passage counts.")

    non_pm_movements = movements.loc[movements["period"] != "PM peak"]
    non_pm_network = network.loc[network["period"] != "PM peak"]
    for frame, keys, fields in (
        (
            non_pm_movements,
            ["period", "seed", "movement"],
            ["total_delay_s", "queue_at_hour_veh", "max_queue_veh"],
        ),
        (
            non_pm_network,
            ["period", "seed"],
            ["mean_delay_per_passage_s", "total_queue_at_hour"],
        ),
    ):
        phase4e = frame.loc[frame["strategy"] == "Phase4E"].drop(
            columns="strategy"
        )
        integrated = frame.loc[frame["strategy"] == "Integrated"].drop(
            columns="strategy"
        )
        paired = phase4e.merge(
            integrated,
            on=keys,
            suffixes=("_4e", "_integrated"),
            validate="one_to_one",
        )
        for field in fields:
            if not np.allclose(
                paired[f"{field}_4e"],
                paired[f"{field}_integrated"],
                atol=1e-12,
            ):
                raise AssertionError(
                    f"Non-PM Integrated and Phase 4E differ for {field}."
                )

    signal_plans.to_csv(SIGNAL_PLANS_PATH, index=False)
    network.to_csv(NETWORK_METRICS_PATH, index=False)
    movements.to_csv(MOVEMENT_METRICS_PATH, index=False)
    return movements, network


def summarise_audit(
    movements: pd.DataFrame,
    network: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create direct paired comparisons against Phase 3A and Phase 4E."""

    network_records: list[dict[str, object]] = []
    movement_records: list[dict[str, object]] = []
    final_network = network.loc[network["strategy"] == "Integrated"].drop(
        columns="strategy"
    )
    final_movements = movements.loc[
        movements["strategy"] == "Integrated"
    ].drop(columns="strategy")

    for benchmark in ("Phase3A", "Phase4E"):
        base_network = network.loc[network["strategy"] == benchmark].drop(
            columns="strategy"
        )
        base_movements = movements.loc[
            movements["strategy"] == benchmark
        ].drop(columns="strategy")
        paired_network = base_network.merge(
            final_network,
            on=["period", "seed"],
            suffixes=("_base", "_final"),
            validate="one_to_one",
        )
        paired_movements = base_movements.merge(
            final_movements,
            on=["period", "seed", "movement"],
            suffixes=("_base", "_final"),
            validate="one_to_one",
        )
        if not paired_network["total_passages_base"].eq(
            paired_network["total_passages_final"]
        ).all():
            raise ValueError("Paired network passage counts differ.")
        if not paired_movements["vehicles_base"].eq(
            paired_movements["vehicles_final"]
        ).all():
            raise ValueError("Paired movement vehicle counts differ.")

        paired_network["delay_change_s"] = (
            paired_network["mean_delay_per_passage_s_final"]
            - paired_network["mean_delay_per_passage_s_base"]
        )
        paired_network["queue_change_veh"] = (
            paired_network["total_queue_at_hour_final"]
            - paired_network["total_queue_at_hour_base"]
        )
        paired_movements["delay_change_s"] = (
            paired_movements["mean_delay_per_vehicle_s_final"]
            - paired_movements["mean_delay_per_vehicle_s_base"]
        )
        paired_movements["queue_change_veh"] = (
            paired_movements["queue_at_hour_veh_final"]
            - paired_movements["queue_at_hour_veh_base"]
        )

        for period in PERIODS:
            net_group = paired_network.loc[paired_network["period"] == period]
            delay_changes = net_group["delay_change_s"].to_numpy(dtype=float)
            delay_low, delay_high = bootstrap_mean_ci(delay_changes)
            queue_changes = net_group["queue_change_veh"].to_numpy(dtype=float)
            queue_low, queue_high = bootstrap_mean_ci(queue_changes)
            network_records.append(
                {
                    "period": period,
                    "benchmark": benchmark,
                    "replications": len(net_group),
                    "benchmark_mean_delay_per_passage_s": net_group[
                        "mean_delay_per_passage_s_base"
                    ].mean(),
                    "integrated_mean_delay_per_passage_s": net_group[
                        "mean_delay_per_passage_s_final"
                    ].mean(),
                    "mean_delay_change_s": delay_changes.mean(),
                    "ci_lower_s": delay_low,
                    "ci_upper_s": delay_high,
                    "seeds_better": int((delay_changes < 0).sum()),
                    "corridor_delay_change_s": (
                        net_group["corridor_mean_delay_s_final"]
                        - net_group["corridor_mean_delay_s_base"]
                    ).mean(),
                    "ns_delay_change_s": (
                        net_group["ns_mean_delay_s_final"]
                        - net_group["ns_mean_delay_s_base"]
                    ).mean(),
                    "benchmark_queue_at_hour": net_group[
                        "total_queue_at_hour_base"
                    ].mean(),
                    "integrated_queue_at_hour": net_group[
                        "total_queue_at_hour_final"
                    ].mean(),
                    "mean_queue_change_veh": queue_changes.mean(),
                    "queue_ci_lower_veh": queue_low,
                    "queue_ci_upper_veh": queue_high,
                }
            )

            for movement in ALL_MOVEMENTS:
                group = paired_movements.loc[
                    (paired_movements["period"] == period)
                    & (paired_movements["movement"] == movement)
                ]
                changes = group["delay_change_s"].to_numpy(dtype=float)
                low, high = bootstrap_mean_ci(changes)
                queue_changes = group["queue_change_veh"].to_numpy(dtype=float)
                movement_records.append(
                    {
                        "period": period,
                        "benchmark": benchmark,
                        "movement": movement,
                        "protected_movement": movement in PROTECTED_MOVEMENTS,
                        "replications": len(group),
                        "benchmark_mean_delay_s": group[
                            "mean_delay_per_vehicle_s_base"
                        ].mean(),
                        "integrated_mean_delay_s": group[
                            "mean_delay_per_vehicle_s_final"
                        ].mean(),
                        "mean_delay_change_s": changes.mean(),
                        "ci_lower_s": low,
                        "ci_upper_s": high,
                        "passes_2s_upper": bool(high <= FAIRNESS_LIMIT_S),
                        "benchmark_queue_at_hour": group[
                            "queue_at_hour_veh_base"
                        ].mean(),
                        "integrated_queue_at_hour": group[
                            "queue_at_hour_veh_final"
                        ].mean(),
                        "mean_queue_change_veh": queue_changes.mean(),
                    }
                )

    network_summary = pd.DataFrame(network_records)
    movement_summary = pd.DataFrame(movement_records)
    fairness_audit = movement_summary.loc[
        movement_summary["protected_movement"]
    ].copy()
    if len(network_summary) != 8:
        raise ValueError("Expected eight network summary rows.")
    if len(movement_summary) != 64:
        raise ValueError("Expected 64 movement summary rows.")
    if len(fairness_audit) != 32:
        raise ValueError("Expected 32 fairness audit rows.")

    network_summary.to_csv(NETWORK_SUMMARY_PATH, index=False)
    movement_summary.to_csv(MOVEMENT_SUMMARY_PATH, index=False)
    fairness_audit.to_csv(FAIRNESS_AUDIT_PATH, index=False)
    return network_summary, movement_summary


def plot_network_comparison(network_summary: pd.DataFrame) -> None:
    """Plot integrated-minus-Phase-3A synthetic network delay changes."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    selected = (
        network_summary.loc[network_summary["benchmark"] == "Phase3A"]
        .set_index("period")
        .reindex(PERIODS)
    )
    x = np.arange(len(PERIODS))
    y = selected["mean_delay_change_s"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, y)
    ax.axhline(0, linestyle="--", linewidth=1, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(PERIODS)
    ax.set_ylabel("Integrated minus Phase 3A delay (s/passage)")
    ax.set_title("Synthetic Integrated Signal Strategy: Network Delay Comparison")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=180)
    plt.close(fig)


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the frozen integrated audit, summaries and comparison figure."""

    movements, network = run_audit()
    network_summary, movement_summary = summarise_audit(movements, network)
    plot_network_comparison(network_summary)
    print("\nIntegrated network comparison:")
    print(network_summary.to_string(index=False))
    print("\nPM protected movements versus Phase 3A:")
    print(
        movement_summary.loc[
            (movement_summary["period"] == "PM peak")
            & (movement_summary["benchmark"] == "Phase3A")
            & movement_summary["movement"].isin(PROTECTED_MOVEMENTS)
        ].to_string(index=False)
    )
    print("\nPM protected movements versus Phase 4E:")
    print(
        movement_summary.loc[
            (movement_summary["period"] == "PM peak")
            & (movement_summary["benchmark"] == "Phase4E")
            & movement_summary["movement"].isin(PROTECTED_MOVEMENTS)
        ].to_string(index=False)
    )
    print("\nPhase 4G completed.")
    return network_summary, movement_summary


if __name__ == "__main__":
    main()
