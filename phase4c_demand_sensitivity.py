"""Run Phase 4C demand and congestion sensitivity experiments."""

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

from phase4b_multi_seed import (
    bootstrap_mean_ci,
    generate_arrivals,
    get_signal_strategies,
)
from traffic_signals.demand_stress import create_demand_scenarios
from traffic_signals.segmentation import PERIOD_ORDER, assign_period
from traffic_signals.vehicle_simulation import simulate_direction


DATA_DIR = Path("outputs/data")
FIGURE_DIR = Path("outputs/figures")
PERIODS = list(PERIOD_ORDER)
FACTORS = (0.75, 1.0, 1.25, 1.5)
SEEDS = range(42, 72)
DURATION_S = 3600.0
SPEED_KMH = 40.0
HEADWAY_S = 2.0
ENTRY_APPROACHES = {"EB": ("A", "E"), "WB": ("C", "W")}
TRAFFIC_PATH = DATA_DIR / "synthetic_traffic_15min.csv"
PHASE3A_PATH = DATA_DIR / "phase3a_signal_plans.csv"
PHASE3F_PATH = DATA_DIR / "phase3f_optimisation_summary.csv"
PHASE4B_METRICS_PATH = DATA_DIR / "phase4b_replication_metrics.csv"
METRICS_PATH = DATA_DIR / "phase4c_replication_metrics.csv"
QUEUES_PATH = DATA_DIR / "phase4c_approach_queues.csv"
PAIRED_PATH = DATA_DIR / "phase4c_paired_differences.csv"
SUMMARY_PATH = DATA_DIR / "phase4c_sensitivity_summary.csv"


def _arrival_rates(traffic: pd.DataFrame, period: str) -> dict[str, float]:
    """Calculate mean baseline entry rates in vehicles per second."""

    rates: dict[str, float] = {}
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
        rates[direction] = float(counts.sum() / (len(counts) * 900.0))
    return rates


def summarise_simulation(
    vehicles: pd.DataFrame,
    queues: pd.DataFrame,
) -> dict[str, float | int]:
    """Summarise delay, residual queues and post-hour clearance."""

    if vehicles.empty or queues.empty:
        raise ValueError("No vehicles to evaluate.")
    if "queue_at_horizon_veh" not in queues:
        raise ValueError("Queue data do not contain an observation horizon.")
    return {
        "vehicles": len(vehicles),
        "mean_delay_s": float(vehicles["total_delay_s"].mean()),
        "mean_journey_s": float(vehicles["journey_time_s"].mean()),
        "mean_waiting_events": float(vehicles["waiting_events"].mean()),
        "max_queue_veh": int(queues["max_queue"].max()),
        "queue_at_horizon_veh": int(queues["queue_at_horizon_veh"].sum()),
        "unfinished_journeys": int(
            (vehicles["exit_time_s"] > DURATION_S).sum()
        ),
        "clearance_tail_s": float(
            max(0.0, vehicles["exit_time_s"].max() - DURATION_S)
        ),
    }


def run_experiment(
    traffic: pd.DataFrame,
    phase3a: pd.DataFrame,
    phase3f: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all period, seed, demand, strategy and direction combinations."""

    records: list[dict[str, object]] = []
    queue_records: list[pd.DataFrame] = []
    for period_index, period in enumerate(PERIODS):
        strategies = get_signal_strategies(phase3a, phase3f, period)
        rates = _arrival_rates(traffic, period)
        for seed in SEEDS:
            baseline = generate_arrivals(traffic, period, seed)
            scenarios: dict[str, dict[float, np.ndarray]] = {}
            for direction_index, direction in enumerate(("EB", "WB")):
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [seed, period_index, direction_index, 404]
                    )
                )
                scenarios[direction] = create_demand_scenarios(
                    baseline_arrivals=baseline[direction],
                    baseline_rate=rates[direction],
                    duration_s=DURATION_S,
                    multipliers=FACTORS,
                    rng=rng,
                )

            for factor in FACTORS:
                for strategy_name, signal_plan in strategies.items():
                    direction_vehicles: list[pd.DataFrame] = []
                    direction_queues: list[pd.DataFrame] = []
                    for direction in ("EB", "WB"):
                        vehicles, queues = simulate_direction(
                            entry_times=scenarios[direction][factor],
                            direction=direction,
                            signal_plan=signal_plan,
                            speed_kmh=SPEED_KMH,
                            headway_s=HEADWAY_S,
                            horizon_s=DURATION_S,
                        )
                        records.append(
                            {
                                "period": period,
                                "seed": seed,
                                "factor": factor,
                                "strategy": strategy_name,
                                "direction": direction,
                                **summarise_simulation(vehicles, queues),
                            }
                        )
                        queue_records.append(
                            queues.assign(
                                period=period,
                                seed=seed,
                                factor=factor,
                                strategy=strategy_name,
                            )
                        )
                        direction_vehicles.append(vehicles)
                        direction_queues.append(queues)

                    combined_vehicles = pd.concat(
                        direction_vehicles, ignore_index=True
                    )
                    combined_queues = pd.concat(
                        direction_queues, ignore_index=True
                    )
                    records.append(
                        {
                            "period": period,
                            "seed": seed,
                            "factor": factor,
                            "strategy": strategy_name,
                            "direction": "Combined",
                            **summarise_simulation(
                                combined_vehicles, combined_queues
                            ),
                        }
                    )

    results = pd.DataFrame(records)
    approaches = pd.concat(queue_records, ignore_index=True)
    if len(results) != 2880:
        raise ValueError("Expected 2,880 replication metric rows.")
    if len(approaches) != 5760:
        raise ValueError("Expected 5,760 approach queue rows.")
    counts = results.pivot(
        index=["period", "seed", "factor", "direction"],
        columns="strategy",
        values="vehicles",
    )
    if not (counts["Phase3A"] == counts["Phase3F"]).all():
        raise ValueError("Matched strategies received different vehicle counts.")
    return results, approaches


def evaluate_demand_sensitivity(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create paired strategy differences and 30-seed summaries."""

    baseline = results.loc[results["strategy"] == "Phase3A"].drop(
        columns="strategy"
    )
    optimised = results.loc[results["strategy"] == "Phase3F"].drop(
        columns="strategy"
    )
    keys = ["period", "seed", "factor", "direction"]
    paired = baseline.merge(
        optimised,
        on=keys,
        suffixes=("_3a", "_3f"),
        validate="one_to_one",
    )
    if len(paired) != 1440:
        raise ValueError("Expected 1,440 paired comparisons.")
    if not (paired["vehicles_3a"] == paired["vehicles_3f"]).all():
        raise ValueError("Paired vehicle counts differ.")
    paired["delay_change_s"] = (
        paired["mean_delay_s_3f"] - paired["mean_delay_s_3a"]
    )
    paired["queue_change"] = (
        paired["queue_at_horizon_veh_3f"]
        - paired["queue_at_horizon_veh_3a"]
    )

    summary_records: list[dict[str, object]] = []
    for period in PERIODS:
        for factor in FACTORS:
            for direction in ("EB", "WB", "Combined"):
                group = paired.loc[
                    (paired["period"] == period)
                    & (paired["factor"] == factor)
                    & (paired["direction"] == direction)
                ]
                ci_lower, ci_upper = bootstrap_mean_ci(group["delay_change_s"])
                summary_records.append(
                    {
                        "period": period,
                        "factor": factor,
                        "direction": direction,
                        "replicates": len(group),
                        "phase3a_delay_s": group["mean_delay_s_3a"].mean(),
                        "phase3f_delay_s": group["mean_delay_s_3f"].mean(),
                        "mean_delay_change_s": group["delay_change_s"].mean(),
                        "ci_lower_s": ci_lower,
                        "ci_upper_s": ci_upper,
                        "fraction_seeds_better": (
                            group["delay_change_s"] < 0
                        ).mean(),
                        "phase3a_queue_at_horizon_veh": group[
                            "queue_at_horizon_veh_3a"
                        ].mean(),
                        "phase3f_queue_at_horizon_veh": group[
                            "queue_at_horizon_veh_3f"
                        ].mean(),
                        "mean_queue_change_veh": group["queue_change"].mean(),
                        "phase3a_unfinished_journeys": group[
                            "unfinished_journeys_3a"
                        ].mean(),
                        "phase3f_unfinished_journeys": group[
                            "unfinished_journeys_3f"
                        ].mean(),
                        "phase3a_clearance_tail_s": group[
                            "clearance_tail_s_3a"
                        ].mean(),
                        "phase3f_clearance_tail_s": group[
                            "clearance_tail_s_3f"
                        ].mean(),
                    }
                )
    summary = pd.DataFrame(summary_records)
    if len(summary) != 48 or not summary["replicates"].eq(30).all():
        raise ValueError("Expected 48 sensitivity rows with 30 replicates each.")
    return paired, summary


def verify_phase4b_regression(results: pd.DataFrame) -> None:
    """Check that 100% demand reproduces every comparable Phase 4B metric."""

    previous = pd.read_csv(PHASE4B_METRICS_PATH)
    current = results.loc[results["factor"] == 1.0]
    keys = ["period", "seed", "strategy", "direction"]
    comparison = current.merge(
        previous,
        on=keys,
        suffixes=("_4c", "_4b"),
        validate="one_to_one",
    )
    if len(comparison) != 720:
        raise ValueError("Expected 720 Phase 4B regression rows.")
    exact_pairs = (
        ("vehicles_4c", "vehicles_4b"),
        ("max_queue_veh", "max_queue"),
        ("unfinished_journeys", "exits_after_horizon"),
    )
    for current_column, previous_column in exact_pairs:
        if not (
            comparison[current_column] == comparison[previous_column]
        ).all():
            raise ValueError(f"Phase 4B regression failed for {current_column}.")
    for column in ("mean_delay_s", "mean_journey_s", "mean_waiting_events"):
        if not np.allclose(
            comparison[f"{column}_4c"],
            comparison[f"{column}_4b"],
            atol=1e-9,
        ):
            raise ValueError(f"Phase 4B regression failed for {column}.")
    expected_tail = np.maximum(0.0, comparison["last_exit_s"] - DURATION_S)
    if not np.allclose(comparison["clearance_tail_s"], expected_tail, atol=1e-9):
        raise ValueError("Phase 4B regression failed for clearance tail.")
    print("Phase 4B baseline regression passed.")


def plot_sensitivity(summary: pd.DataFrame) -> list[Path]:
    """Plot combined synthetic delay against demand for every period."""

    output_paths: list[Path] = []
    for period in PERIODS:
        selected = summary.loc[
            (summary["period"] == period)
            & (summary["direction"] == "Combined")
        ].sort_values("factor")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(
            selected["factor"] * 100,
            selected["phase3a_delay_s"],
            marker="o",
            label="Phase 3A",
        )
        ax.plot(
            selected["factor"] * 100,
            selected["phase3f_delay_s"],
            marker="s",
            label="Phase 3F",
        )
        ax.set_xlabel("Demand relative to baseline (%)")
        ax.set_ylabel("Mean simulated delay (seconds)")
        ax.set_title(f"Synthetic Congestion Sensitivity: {period}")
        ax.set_xticks([75, 100, 125, 150])
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        filename = period.lower().replace(" ", "_")
        output_path = FIGURE_DIR / f"phase4c_demand_sensitivity_{filename}.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        output_paths.append(output_path)
    return output_paths


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and normalise Phase 4C input data."""

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


def main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run Phase 4C, save outputs and print the 48-row summary."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    traffic, phase3a, phase3f = _load_inputs()
    results, approaches = run_experiment(traffic, phase3a, phase3f)
    verify_phase4b_regression(results)
    paired, summary = evaluate_demand_sensitivity(results)
    results.to_csv(METRICS_PATH, index=False)
    approaches.to_csv(QUEUES_PATH, index=False)
    paired.to_csv(PAIRED_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    figure_paths = plot_sensitivity(summary)

    print("\nPhase 4C sensitivity summary:")
    print(summary.to_string(index=False))
    print("\nPM peak westbound focus:")
    print(
        summary.loc[
            (summary["period"] == "PM peak")
            & (summary["direction"] == "WB")
            & (summary["factor"].isin([1.0, 1.25, 1.5]))
        ].to_string(index=False)
    )
    print(f"\nWrote {METRICS_PATH}")
    print(f"Wrote {QUEUES_PATH}")
    print(f"Wrote {PAIRED_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    for path in figure_paths:
        print(f"Wrote {path}")
    print("\nPhase 4C completed.")
    return results, approaches, paired, summary


if __name__ == "__main__":
    main()
