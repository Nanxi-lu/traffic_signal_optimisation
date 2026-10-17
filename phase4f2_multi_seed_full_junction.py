"""Evaluate fixed Phase 3A and Phase 4E full-junction plans over fresh seeds."""

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
from traffic_signals.hourly_arrivals import generate_hourly_arrivals
from traffic_signals.network_evaluation import (
    build_period_plans,
    evaluate_network,
)


DATA_DIR = Path("outputs/data")
FIGURE_DIR = Path("outputs/figures")
TRAFFIC_PATH = DATA_DIR / "synthetic_traffic_15min.csv"
PHASE3A_PATH = DATA_DIR / "phase3a_signal_plans.csv"
PHASE4E_PATH = DATA_DIR / "phase4e_selected_plans.csv"
PHASE4F1_NETWORK_PATH = DATA_DIR / "phase4f1_network_summary.csv"
MOVEMENT_METRICS_PATH = DATA_DIR / "phase4f2_movement_metrics.csv"
NETWORK_METRICS_PATH = DATA_DIR / "phase4f2_network_metrics.csv"
PAIRED_MOVEMENTS_PATH = DATA_DIR / "phase4f2_paired_movements.csv"
MOVEMENT_SUMMARY_PATH = DATA_DIR / "phase4f2_movement_summary.csv"
PAIRED_NETWORK_PATH = DATA_DIR / "phase4f2_paired_network.csv"
NETWORK_SUMMARY_PATH = DATA_DIR / "phase4f2_network_summary.csv"
CAPACITY_PATH = DATA_DIR / "phase4f2_c_capacity.csv"
FIGURE_PATH = FIGURE_DIR / "phase4f2_network_delay.png"

PERIOD_HOURS = {
    "Overnight": 2,
    "AM peak": 8,
    "Interpeak": 12,
    "PM peak": 17,
}
SEEDS = range(200, 230)
HORIZON_S = 3600.0
STREAMS = (
    ("A", "E"),
    ("C", "W"),
    ("A", "N"),
    ("A", "S"),
    ("B", "N"),
    ("B", "S"),
    ("C", "N"),
    ("C", "S"),
)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and normalise synthetic demand and fixed timing plans."""

    traffic = pd.read_csv(TRAFFIC_PATH, parse_dates=["time"])
    traffic["intersection_id"] = traffic["intersection_id"].astype(str).str.upper()
    traffic["approach"] = (
        traffic["approach"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace(
            {
                "EASTBOUND": "E",
                "WESTBOUND": "W",
                "NORTHBOUND": "N",
                "SOUTHBOUND": "S",
            }
        )
    )
    plans_3a = pd.read_csv(PHASE3A_PATH)
    plans_4e = pd.read_csv(PHASE4E_PATH)
    return traffic, plans_3a, plans_4e


def make_arrivals(
    traffic: pd.DataFrame,
    period: str,
    seed: int,
) -> dict[tuple[str, str], np.ndarray]:
    """Generate the eight matched streams for one representative hour."""

    if period not in PERIOD_HOURS:
        raise ValueError(f"Unknown period: {period}.")
    study_date = traffic["time"].dt.normalize().min()
    start_time = study_date + pd.Timedelta(hours=PERIOD_HOURS[period])
    return generate_hourly_arrivals(
        traffic=traffic,
        start_time=start_time,
        streams=STREAMS,
        seed=seed,
    )


def verify_phase4f1_regression(
    traffic: pd.DataFrame,
    plans_3a: pd.DataFrame,
    plans_4e: pd.DataFrame,
) -> None:
    """Reproduce the Phase 4F-1 AM seed-42 network results after refactoring."""

    previous = pd.read_csv(PHASE4F1_NETWORK_PATH).set_index("strategy")
    strategies = build_period_plans(plans_3a, plans_4e, "AM peak")
    arrivals = make_arrivals(traffic, "AM peak", seed=42)
    field_mapping = {
        "corridor_mean_delay_s": "corridor_mean_delay_per_vehicle_s",
        "ns_mean_delay_s": "ns_mean_delay_per_vehicle_s",
        "mean_delay_per_passage_s": "mean_delay_per_junction_passage_s",
        "total_delay_vehicle_seconds": "total_delay_s",
        "total_queue_at_hour": "total_queue_at_horizon_veh",
    }
    for strategy, junction_plans in strategies.items():
        _, current = evaluate_network(
            arrivals=arrivals,
            junction_plans=junction_plans,
            strategy=strategy,
            period="AM peak",
            seed=42,
            horizon_s=HORIZON_S,
        )
        for current_field, previous_field in field_mapping.items():
            if not np.isclose(
                float(current[current_field]),
                float(previous.loc[strategy, previous_field]),
                atol=1e-6,
            ):
                raise AssertionError(
                    f"Regression mismatch: {strategy}/{current_field}"
                )
    print("Phase 4F-1 regression passed.")


def run_experiment(
    traffic: pd.DataFrame,
    plans_3a: pd.DataFrame,
    plans_4e: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run four periods, 30 seeds, two strategies and eight movements."""

    movement_frames: list[pd.DataFrame] = []
    network_records: list[dict[str, object]] = []
    for period in PERIOD_HOURS:
        strategies = build_period_plans(plans_3a, plans_4e, period)
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

    movement_results = pd.concat(movement_frames, ignore_index=True)
    network_results = pd.DataFrame(network_records)
    if len(movement_results) != 1920:
        raise ValueError("Expected 1,920 movement metric rows.")
    if len(network_results) != 240:
        raise ValueError("Expected 240 network metric rows.")
    movement_counts = movement_results.pivot(
        index=["period", "seed", "movement"],
        columns="strategy",
        values="vehicles",
    )
    if not (movement_counts["Phase3A"] == movement_counts["Phase4E"]).all():
        raise ValueError("Strategies received different movement demand.")
    movement_results.to_csv(MOVEMENT_METRICS_PATH, index=False)
    network_results.to_csv(NETWORK_METRICS_PATH, index=False)
    return movement_results, network_results


def summarise_movements(
    movements: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create paired movement comparisons and 32 bootstrap summaries."""

    keys = ["period", "seed", "movement"]
    baseline = movements.loc[movements["strategy"] == "Phase3A"].drop(
        columns="strategy"
    )
    optimised = movements.loc[movements["strategy"] == "Phase4E"].drop(
        columns="strategy"
    )
    paired = baseline.merge(
        optimised,
        on=keys,
        suffixes=("_3a", "_4e"),
        validate="one_to_one",
    )
    if len(paired) != 960:
        raise ValueError("Expected 960 paired movement rows.")
    if not (paired["vehicles_3a"] == paired["vehicles_4e"]).all():
        raise ValueError("Paired movement vehicle counts differ.")
    paired["delay_change_s"] = (
        paired["mean_delay_per_vehicle_s_4e"]
        - paired["mean_delay_per_vehicle_s_3a"]
    )
    paired["queue_change_veh"] = (
        paired["queue_at_hour_veh_4e"]
        - paired["queue_at_hour_veh_3a"]
    )

    records: list[dict[str, object]] = []
    for period in PERIOD_HOURS:
        for movement in ("EB", "WB", "A_N", "A_S", "B_N", "B_S", "C_N", "C_S"):
            group = paired.loc[
                (paired["period"] == period)
                & (paired["movement"] == movement)
            ]
            changes = group["delay_change_s"].to_numpy()
            low, high = bootstrap_mean_ci(changes)
            records.append(
                {
                    "period": period,
                    "movement": movement,
                    "category": group["category_3a"].iloc[0],
                    "replications": len(group),
                    "phase3a_mean_delay_s": group[
                        "mean_delay_per_vehicle_s_3a"
                    ].mean(),
                    "phase4e_mean_delay_s": group[
                        "mean_delay_per_vehicle_s_4e"
                    ].mean(),
                    "mean_delay_change_s": changes.mean(),
                    "ci_lower_s": low,
                    "ci_upper_s": high,
                    "seeds_better": int((changes < 0).sum()),
                    "phase3a_queue_at_hour": group[
                        "queue_at_hour_veh_3a"
                    ].mean(),
                    "phase4e_queue_at_hour": group[
                        "queue_at_hour_veh_4e"
                    ].mean(),
                    "mean_queue_change_veh": group[
                        "queue_change_veh"
                    ].mean(),
                }
            )
    summary = pd.DataFrame(records)
    if len(summary) != 32 or not summary["replications"].eq(30).all():
        raise ValueError("Expected 32 movement summaries with 30 replications.")
    paired.to_csv(PAIRED_MOVEMENTS_PATH, index=False)
    summary.to_csv(MOVEMENT_SUMMARY_PATH, index=False)
    return paired, summary


def summarise_network(
    network: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create paired network differences and four period summaries."""

    keys = ["period", "seed"]
    baseline = network.loc[network["strategy"] == "Phase3A"].drop(
        columns="strategy"
    )
    optimised = network.loc[network["strategy"] == "Phase4E"].drop(
        columns="strategy"
    )
    paired = baseline.merge(
        optimised,
        on=keys,
        suffixes=("_3a", "_4e"),
        validate="one_to_one",
    )
    if len(paired) != 120:
        raise ValueError("Expected 120 paired network rows.")
    paired["delay_change_s"] = (
        paired["mean_delay_per_passage_s_4e"]
        - paired["mean_delay_per_passage_s_3a"]
    )
    paired["queue_change_veh"] = (
        paired["total_queue_at_hour_4e"]
        - paired["total_queue_at_hour_3a"]
    )

    records: list[dict[str, object]] = []
    for period in PERIOD_HOURS:
        group = paired.loc[paired["period"] == period]
        changes = group["delay_change_s"].to_numpy()
        low, high = bootstrap_mean_ci(changes)
        records.append(
            {
                "period": period,
                "replications": len(group),
                "phase3a_delay_per_passage_s": group[
                    "mean_delay_per_passage_s_3a"
                ].mean(),
                "phase4e_delay_per_passage_s": group[
                    "mean_delay_per_passage_s_4e"
                ].mean(),
                "mean_delay_change_s": changes.mean(),
                "ci_lower_s": low,
                "ci_upper_s": high,
                "seeds_better": int((changes < 0).sum()),
                "corridor_delay_change_s": (
                    group["corridor_mean_delay_s_4e"]
                    - group["corridor_mean_delay_s_3a"]
                ).mean(),
                "ns_delay_change_s": (
                    group["ns_mean_delay_s_4e"]
                    - group["ns_mean_delay_s_3a"]
                ).mean(),
                "phase3a_queue_at_hour": group[
                    "total_queue_at_hour_3a"
                ].mean(),
                "phase4e_queue_at_hour": group[
                    "total_queue_at_hour_4e"
                ].mean(),
                "mean_queue_change_veh": group["queue_change_veh"].mean(),
            }
        )
    summary = pd.DataFrame(records)
    if len(summary) != 4 or not summary["replications"].eq(30).all():
        raise ValueError("Expected four network summaries with 30 replications.")
    paired.to_csv(PAIRED_NETWORK_PATH, index=False)
    summary.to_csv(NETWORK_SUMMARY_PATH, index=False)
    return paired, summary


def diagnose_c_capacity(
    traffic: pd.DataFrame,
    plans_3a: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate Junction C critical N/S ratios in each representative hour."""

    records: list[dict[str, object]] = []
    study_date = traffic["time"].dt.normalize().min()
    for period, hour in PERIOD_HOURS.items():
        rows = plans_3a.loc[
            (plans_3a["period"] == period)
            & (plans_3a["intersection_id"] == "C")
        ]
        if len(rows) != 1:
            raise ValueError(f"Missing Junction C timing plan for {period}.")
        plan = rows.iloc[0]
        capacity_ratio = float(
            plan["ns_green_s"] / plan["common_cycle_s"]
        )
        start = study_date + pd.Timedelta(hours=hour)
        for timestamp in pd.date_range(start, periods=4, freq="15min"):
            counts = traffic.loc[
                (traffic["time"] == timestamp)
                & (traffic["intersection_id"] == "C")
                & (traffic["approach"].isin(["N", "S"]))
            ]
            if len(counts) != 2 or set(counts["approach"]) != {"N", "S"}:
                raise ValueError("Expected N and S demand at Junction C.")
            counts = counts.set_index("approach")
            volumes = pd.to_numeric(counts["volume_15min"], errors="raise")
            saturation = pd.to_numeric(
                counts["saturation_flow_vph"], errors="raise"
            )
            critical_approach = str(volumes.idxmax())
            critical_count = float(volumes.loc[critical_approach])
            saturation_flow = float(saturation.loc[critical_approach])
            critical_flow = critical_count * 4.0
            critical_ratio = critical_flow / saturation_flow
            records.append(
                {
                    "period": period,
                    "time": timestamp,
                    "critical_approach": critical_approach,
                    "critical_count_15min": int(critical_count),
                    "critical_flow_vph": critical_flow,
                    "saturation_flow_vph": saturation_flow,
                    "critical_ratio": critical_ratio,
                    "ns_capacity_ratio": capacity_ratio,
                    "ns_capacity_vph": capacity_ratio * saturation_flow,
                    "exceeds_green_capacity": bool(
                        critical_ratio > capacity_ratio + 1e-12
                    ),
                }
            )
    result = pd.DataFrame(records)
    if len(result) != 16:
        raise ValueError("Expected 16 Junction C capacity rows.")
    result.to_csv(CAPACITY_PATH, index=False)
    return result


def plot_network_results(summary: pd.DataFrame) -> None:
    """Plot paired synthetic network-delay changes and bootstrap intervals."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ordered = summary.set_index("period").loc[list(PERIOD_HOURS)]
    x = np.arange(len(ordered))
    y = ordered["mean_delay_change_s"].to_numpy()
    errors = np.vstack(
        [
            y - ordered["ci_lower_s"].to_numpy(),
            ordered["ci_upper_s"].to_numpy() - y,
        ]
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(x, y, yerr=errors, marker="o", capsize=5, linestyle="none")
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered.index)
    ax.set_title("Phase 4F-2: Synthetic Network Delay Change Across 30 Seeds")
    ax.set_ylabel("Phase 4E minus Phase 3A delay per passage (s)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=180)
    plt.close(fig)


def main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run regression, repeated evaluation, summaries, diagnostics and plot."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    traffic, plans_3a, plans_4e = load_inputs()
    verify_phase4f1_regression(traffic, plans_3a, plans_4e)
    movements, network = run_experiment(traffic, plans_3a, plans_4e)
    _, movement_summary = summarise_movements(movements)
    _, network_summary = summarise_network(network)
    capacity = diagnose_c_capacity(traffic, plans_3a)
    plot_network_results(network_summary)
    print("\nNetwork comparison:")
    print(network_summary.to_string(index=False))
    print("\nJunction C movement comparison:")
    print(
        movement_summary.loc[
            movement_summary["movement"].isin(["C_N", "C_S"])
        ].to_string(index=False)
    )
    print("\nCapacity warnings:")
    print(capacity.loc[capacity["exceeds_green_capacity"]].to_string(index=False))
    return network_summary, movement_summary, capacity


if __name__ == "__main__":
    main()
