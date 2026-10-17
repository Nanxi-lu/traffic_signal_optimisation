"""Run Phase 4B across four periods and 30 matched arrival seeds."""

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

from traffic_signals.segmentation import PERIOD_ORDER, assign_period
from traffic_signals.vehicle_simulation import simulate_direction


DATA_DIR = Path("outputs/data")
FIGURE_DIR = Path("outputs/figures")
PERIODS = list(PERIOD_ORDER)
SEEDS = range(42, 72)
DURATION_S = 3600.0
SPEED_KMH = 40.0
HEADWAY_S = 2.0
ENTRY_APPROACHES = {"EB": ("A", "E"), "WB": ("C", "W")}
STRATEGIES = ("Phase3A", "Phase3F")
TRAFFIC_PATH = DATA_DIR / "synthetic_traffic_15min.csv"
PHASE3A_PATH = DATA_DIR / "phase3a_signal_plans.csv"
PHASE3F_PATH = DATA_DIR / "phase3f_optimisation_summary.csv"
PHASE4A_SUMMARY_PATH = DATA_DIR / "phase4a_summary.csv"
PHASE4A_COMBINED_PATH = DATA_DIR / "phase4a_combined_comparison.csv"
METRICS_PATH = DATA_DIR / "phase4b_replication_metrics.csv"
PAIRED_PATH = DATA_DIR / "phase4b_paired_differences.csv"
SUMMARY_PATH = DATA_DIR / "phase4b_uncertainty_summary.csv"
FIGURE_PATH = FIGURE_DIR / "phase4b_delay_comparison.png"


def generate_arrivals(
    traffic: pd.DataFrame,
    period: str,
    seed: int,
) -> dict[str, np.ndarray]:
    """Generate paired synthetic EB and WB arrival streams."""

    required = {"period", "intersection_id", "approach", "volume_15min"}
    if not required.issubset(traffic.columns):
        raise ValueError("Missing required traffic columns.")
    rng = np.random.default_rng(seed)
    result: dict[str, np.ndarray] = {}
    for direction, (junction, approach) in ENTRY_APPROACHES.items():
        selected = traffic.loc[
            (traffic["period"] == period)
            & (traffic["intersection_id"] == junction)
            & (traffic["approach"] == approach)
        ]
        if selected.empty:
            raise ValueError(f"Missing {direction} observations for {period}.")
        counts = pd.to_numeric(selected["volume_15min"], errors="raise").to_numpy()
        if not np.isfinite(counts).all() or np.any(counts < 0):
            raise ValueError("Invalid vehicle counts.")
        arrival_rate = counts.sum() / (len(counts) * 900.0)
        vehicle_count = int(rng.poisson(arrival_rate * DURATION_S))
        result[direction] = np.sort(
            rng.uniform(0.0, DURATION_S, vehicle_count)
        )
    return result


def get_signal_strategies(
    phase3a: pd.DataFrame,
    phase3f: pd.DataFrame,
    period: str,
) -> dict[str, dict[str, object]]:
    """Get original and protected signal plans for one period."""

    group = phase3a.loc[phase3a["period"] == period].set_index("intersection_id")
    if set(group.index) != {"A", "B", "C"} or len(group) != 3:
        raise ValueError("Expected A, B and C signal plans.")
    cycles = group["common_cycle_s"].unique()
    if len(cycles) != 1:
        raise ValueError("Common cycle is inconsistent.")
    cycle = int(cycles[0])
    greens = group["ew_green_s"].to_dict()
    original_offsets = group["offset_s"].to_dict()
    matching = phase3f.loc[
        (phase3f["period"] == period)
        & (phase3f["strategy"] == "Phase3F_bidirectional")
    ]
    if len(matching) != 1:
        raise ValueError("Missing Phase 3F plan.")
    optimised = matching.iloc[0]
    new_offsets = {
        "A": 0.0,
        "B": float(optimised["offset_b_s"]),
        "C": float(optimised["offset_c_s"]),
    }
    return {
        "Phase3A": {
            "cycle_s": cycle,
            "greens": greens,
            "offsets": original_offsets,
        },
        "Phase3F": {
            "cycle_s": cycle,
            "greens": greens,
            "offsets": new_offsets,
        },
    }


def calculate_metrics(
    vehicles: pd.DataFrame,
    queues: pd.DataFrame,
) -> dict[str, float | int]:
    """Summarise one simulated stream or combined corridor."""

    if vehicles.empty or queues.empty:
        raise ValueError("Cannot summarise an empty simulation.")
    return {
        "vehicles": len(vehicles),
        "mean_delay_s": float(vehicles["total_delay_s"].mean()),
        "mean_journey_s": float(vehicles["journey_time_s"].mean()),
        "mean_waiting_events": float(vehicles["waiting_events"].mean()),
        "no_wait_fraction": float((vehicles["waiting_events"] == 0).mean()),
        "max_queue": int(queues["max_queue"].max()),
        "exits_after_horizon": int((vehicles["exit_time_s"] > DURATION_S).sum()),
        "last_exit_s": float(vehicles["exit_time_s"].max()),
    }


def run_experiment(
    traffic: pd.DataFrame,
    phase3a: pd.DataFrame,
    phase3f: pd.DataFrame,
) -> pd.DataFrame:
    """Run all period, seed, strategy and direction combinations."""

    records: list[dict[str, object]] = []
    for period in PERIODS:
        strategies = get_signal_strategies(phase3a, phase3f, period)
        for seed in SEEDS:
            arrivals = generate_arrivals(traffic, period, seed)
            for strategy_name, signal_plan in strategies.items():
                direction_vehicles: list[pd.DataFrame] = []
                direction_queues: list[pd.DataFrame] = []
                for direction in ("EB", "WB"):
                    vehicles, queues = simulate_direction(
                        entry_times=arrivals[direction],
                        direction=direction,
                        signal_plan=signal_plan,
                        speed_kmh=SPEED_KMH,
                        headway_s=HEADWAY_S,
                    )
                    records.append(
                        {
                            "period": period,
                            "seed": seed,
                            "strategy": strategy_name,
                            "direction": direction,
                            **calculate_metrics(vehicles, queues),
                        }
                    )
                    direction_vehicles.append(vehicles)
                    direction_queues.append(queues)
                combined_vehicles = pd.concat(direction_vehicles, ignore_index=True)
                combined_queues = pd.concat(direction_queues, ignore_index=True)
                records.append(
                    {
                        "period": period,
                        "seed": seed,
                        "strategy": strategy_name,
                        "direction": "Combined",
                        **calculate_metrics(combined_vehicles, combined_queues),
                    }
                )

    results = pd.DataFrame(records)
    if len(results) != 720:
        raise ValueError("Expected 720 replication metric rows.")
    counts = results.pivot(
        index=["period", "seed", "direction"],
        columns="strategy",
        values="vehicles",
    )
    if not (counts["Phase3A"] == counts["Phase3F"]).all():
        raise ValueError("Matched strategies received different vehicle counts.")
    return results


def bootstrap_mean_ci(
    values: np.ndarray | pd.Series,
    n_bootstrap: int = 4000,
    seed: int = 2026,
) -> np.ndarray:
    """Bootstrap a 95% interval for the mean paired change."""

    value_array = np.asarray(values, dtype=float)
    if len(value_array) < 2 or not np.isfinite(value_array).all():
        raise ValueError("Need at least two finite paired changes.")
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(value_array),
        size=(n_bootstrap, len(value_array)),
    )
    bootstrap_means = value_array[indices].mean(axis=1)
    return np.quantile(bootstrap_means, [0.025, 0.975])


def evaluate_replicates(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create paired strategy differences and bootstrap summaries."""

    baseline = results.loc[results["strategy"] == "Phase3A"].drop(
        columns="strategy"
    )
    optimised = results.loc[results["strategy"] == "Phase3F"].drop(
        columns="strategy"
    )
    paired = baseline.merge(
        optimised,
        on=["period", "seed", "direction"],
        suffixes=("_3a", "_3f"),
        validate="one_to_one",
    )
    if len(paired) != 360:
        raise ValueError("Expected 360 paired comparisons.")
    if not (paired["vehicles_3a"] == paired["vehicles_3f"]).all():
        raise ValueError("Paired vehicle counts differ.")
    paired["delay_change_s"] = paired["mean_delay_s_3f"] - paired["mean_delay_s_3a"]
    paired["journey_change_s"] = (
        paired["mean_journey_s_3f"] - paired["mean_journey_s_3a"]
    )
    paired["waiting_event_change"] = (
        paired["mean_waiting_events_3f"]
        - paired["mean_waiting_events_3a"]
    )
    paired["no_wait_change"] = (
        paired["no_wait_fraction_3f"] - paired["no_wait_fraction_3a"]
    )
    paired["max_queue_change"] = paired["max_queue_3f"] - paired["max_queue_3a"]

    summary_records: list[dict[str, object]] = []
    for period in PERIODS:
        for direction in ("EB", "WB", "Combined"):
            group = paired.loc[
                (paired["period"] == period) & (paired["direction"] == direction)
            ]
            ci_lower, ci_upper = bootstrap_mean_ci(group["delay_change_s"])
            summary_records.append(
                {
                    "period": period,
                    "direction": direction,
                    "replicates": len(group),
                    "mean_delay_3a_s": group["mean_delay_s_3a"].mean(),
                    "mean_delay_3f_s": group["mean_delay_s_3f"].mean(),
                    "mean_delay_change_s": group["delay_change_s"].mean(),
                    "ci_lower_s": ci_lower,
                    "ci_upper_s": ci_upper,
                    "fraction_seeds_better": (group["delay_change_s"] < 0).mean(),
                    "mean_waiting_event_change": group[
                        "waiting_event_change"
                    ].mean(),
                    "mean_no_wait_change": group["no_wait_change"].mean(),
                    "mean_max_queue_change": group["max_queue_change"].mean(),
                }
            )
    summary = pd.DataFrame(summary_records)
    if len(summary) != 12 or not summary["replicates"].eq(30).all():
        raise ValueError("Expected 12 uncertainty rows with 30 replicates each.")
    return paired, summary


def verify_phase4a_regression(results: pd.DataFrame) -> None:
    """Check that AM seed 42 reproduces the saved Phase 4A metrics."""

    phase4a_directional = pd.read_csv(PHASE4A_SUMMARY_PATH)
    phase4a_combined = pd.read_csv(PHASE4A_COMBINED_PATH)
    reference = pd.concat(
        [
            phase4a_directional,
            phase4a_combined.assign(direction="Combined"),
        ],
        ignore_index=True,
    )
    current = results.loc[
        (results["period"] == "AM peak") & (results["seed"] == 42)
    ]
    comparison = current.merge(
        reference,
        on=["strategy", "direction"],
        suffixes=("_4b", "_4a"),
        validate="one_to_one",
    )
    if len(comparison) != 6:
        raise ValueError("AM seed 42 regression has unexpected row count.")
    exact_columns = ("vehicles", "max_queue", "exits_after_horizon")
    float_columns = (
        "mean_delay_s",
        "mean_journey_s",
        "mean_waiting_events",
        "no_wait_fraction",
        "last_exit_s",
    )
    for column in exact_columns:
        if not (
            comparison[f"{column}_4b"] == comparison[f"{column}_4a"]
        ).all():
            raise ValueError(f"Phase 4A regression failed for {column}.")
    for column in float_columns:
        if not np.allclose(
            comparison[f"{column}_4b"], comparison[f"{column}_4a"]
        ):
            raise ValueError(f"Phase 4A regression failed for {column}.")


def plot_delay_changes(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot combined paired delay changes with bootstrap intervals."""

    combined = (
        summary.loc[summary["direction"] == "Combined"]
        .set_index("period")
        .reindex(PERIODS)
    )
    x = np.arange(len(PERIODS))
    y = combined["mean_delay_change_s"].to_numpy()
    lower = y - combined["ci_lower_s"].to_numpy()
    upper = combined["ci_upper_s"].to_numpy() - y
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(x, y, yerr=np.vstack([lower, upper]), fmt="o", capsize=5)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x, PERIODS)
    ax.set_ylabel("Phase 3F minus Phase 3A delay (seconds)")
    ax.set_title("Synthetic Paired Delay Differences Across 30 Seeds")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
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

    results = run_experiment(traffic, phase3a, phase3f)
    verify_phase4a_regression(results)
    paired, summary = evaluate_replicates(results)
    results.to_csv(METRICS_PATH, index=False)
    paired.to_csv(PAIRED_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    plot_delay_changes(summary, FIGURE_PATH)

    print("Phase 4B paired comparison:")
    print(summary.to_string(index=False))
    print("\nAM peak seed 42 regression reference:")
    print(
        results.loc[
            (results["period"] == "AM peak")
            & (results["seed"] == 42)
            & (results["direction"] != "Combined")
        ].to_string(index=False)
    )
    print("\nAM peak seed 42 regression checks passed.")
    print(f"Wrote {METRICS_PATH}")
    print(f"Wrote {PAIRED_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {FIGURE_PATH}")
    return results, paired, summary


if __name__ == "__main__":
    main()
