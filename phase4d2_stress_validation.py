"""Stress-test the frozen Phase 4D AM offsets on fresh arrival seeds."""

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
from traffic_signals.segmentation import assign_period
from traffic_signals.vehicle_simulation import simulate_direction


DATA_DIR = Path("outputs/data")
FIGURE_DIR = Path("outputs/figures")
PERIOD = "AM peak"
FACTORS = (0.75, 1.0, 1.25, 1.5)
SEEDS = range(72, 102)
DURATION_S = 3600.0
SPEED_KMH = 40.0
HEADWAY_S = 2.0
FAIRNESS_LIMIT_S = 2.0
ENTRY_APPROACHES = {"EB": ("A", "E"), "WB": ("C", "W")}
TRAFFIC_PATH = DATA_DIR / "synthetic_traffic_15min.csv"
PHASE3A_PATH = DATA_DIR / "phase3a_signal_plans.csv"
PHASE3F_PATH = DATA_DIR / "phase3f_optimisation_summary.csv"
PHASE4D_PATH = DATA_DIR / "phase4d_selected_plan.csv"
METRICS_PATH = DATA_DIR / "phase4d2_replication_metrics.csv"
PAIRED_PATH = DATA_DIR / "phase4d2_paired_differences.csv"
SUMMARY_PATH = DATA_DIR / "phase4d2_stress_summary.csv"
CONGESTION_FIGURE_PATH = FIGURE_DIR / "phase4d2_congestion_comparison.png"
FAIRNESS_FIGURE_PATH = FIGURE_DIR / "phase4d2_eastbound_fairness.png"


def load_inputs() -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Load traffic and the three fixed AM signal strategies."""

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
    phase4d = pd.read_csv(PHASE4D_PATH)
    strategies = get_signal_strategies(phase3a, phase3f, PERIOD)

    selected = phase4d.loc[phase4d["period"] == PERIOD]
    if len(selected) != 1:
        raise ValueError("Expected one selected Phase 4D AM plan.")
    row = selected.iloc[0]
    offsets = {
        "A": float(row["A_offset_s"]),
        "B": float(row["B_offset_s"]),
        "C": float(row["C_offset_s"]),
    }
    if offsets != {"A": 0.0, "B": 32.0, "C": 68.0}:
        raise ValueError("Phase 4D offsets do not match the frozen 0/32/68 plan.")
    original = strategies["Phase3A"]
    if not np.isclose(float(row["cycle_s"]), float(original["cycle_s"])):
        raise ValueError("Phase 4D cycle differs from Phase 3A.")
    strategies["Phase4D"] = {
        "cycle_s": original["cycle_s"],
        "greens": original["greens"],
        "offsets": offsets,
    }
    return traffic, strategies


def generate_stress_arrivals(
    traffic: pd.DataFrame,
    seed: int,
) -> dict[float, dict[str, np.ndarray]]:
    """Create nested fresh-seed AM arrivals shared by all strategies."""

    baseline = generate_arrivals(traffic, PERIOD, seed)
    by_direction: dict[str, dict[float, np.ndarray]] = {}
    for direction_index, direction in enumerate(("EB", "WB")):
        junction, approach = ENTRY_APPROACHES[direction]
        selected = traffic.loc[
            (traffic["period"] == PERIOD)
            & (traffic["intersection_id"] == junction)
            & (traffic["approach"] == approach)
        ]
        if len(selected) != 16:
            raise ValueError(f"Expected 16 AM counts for {direction}.")
        counts = pd.to_numeric(selected["volume_15min"], errors="raise").to_numpy()
        if not np.isfinite(counts).all() or np.any(counts < 0):
            raise ValueError("Invalid AM traffic counts.")
        rate = float(counts.sum() / (len(counts) * 900.0))
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, 1, direction_index, 404])
        )
        by_direction[direction] = create_demand_scenarios(
            baseline_arrivals=baseline[direction],
            baseline_rate=rate,
            duration_s=DURATION_S,
            multipliers=FACTORS,
            rng=rng,
        )

    return {
        factor: {
            direction: by_direction[direction][factor]
            for direction in ("EB", "WB")
        }
        for factor in FACTORS
    }


def summarise_run(
    vehicles: pd.DataFrame,
    queues: pd.DataFrame,
) -> dict[str, float | int]:
    """Summarise one directional or combined simulation output."""

    if vehicles.empty or queues.empty:
        raise ValueError("Empty simulation output.")
    required_vehicle = {
        "total_delay_s",
        "journey_time_s",
        "waiting_events",
        "exit_time_s",
    }
    required_queue = {"max_queue", "queue_at_horizon_veh"}
    if not required_vehicle.issubset(vehicles) or not required_queue.issubset(queues):
        raise ValueError("Missing simulation output columns.")
    return {
        "vehicles": len(vehicles),
        "mean_delay_s": float(vehicles["total_delay_s"].mean()),
        "mean_journey_s": float(vehicles["journey_time_s"].mean()),
        "mean_waiting_events": float(vehicles["waiting_events"].mean()),
        "no_wait_fraction": float((vehicles["waiting_events"] == 0).mean()),
        "max_queue_veh": int(queues["max_queue"].max()),
        "queue_at_hour_veh": int(queues["queue_at_horizon_veh"].sum()),
        "unfinished_journeys": int(
            (vehicles["exit_time_s"] > DURATION_S).sum()
        ),
    }


def run_experiment(
    traffic: pd.DataFrame,
    strategies: dict[str, dict[str, object]],
) -> pd.DataFrame:
    """Run 30 fresh seeds, four demand factors and three fixed strategies."""

    records: list[dict[str, object]] = []
    for seed in SEEDS:
        scenarios = generate_stress_arrivals(traffic, seed)
        for factor in FACTORS:
            for strategy_name, signal_plan in strategies.items():
                vehicle_frames: list[pd.DataFrame] = []
                queue_frames: list[pd.DataFrame] = []
                for direction in ("EB", "WB"):
                    vehicles, queues = simulate_direction(
                        entry_times=scenarios[factor][direction],
                        direction=direction,
                        signal_plan=signal_plan,
                        speed_kmh=SPEED_KMH,
                        headway_s=HEADWAY_S,
                        horizon_s=DURATION_S,
                    )
                    records.append(
                        {
                            "period": PERIOD,
                            "seed": seed,
                            "factor": factor,
                            "strategy": strategy_name,
                            "direction": direction,
                            **summarise_run(vehicles, queues),
                        }
                    )
                    vehicle_frames.append(vehicles)
                    queue_frames.append(queues)

                combined_vehicles = pd.concat(vehicle_frames, ignore_index=True)
                combined_queues = pd.concat(queue_frames, ignore_index=True)
                records.append(
                    {
                        "period": PERIOD,
                        "seed": seed,
                        "factor": factor,
                        "strategy": strategy_name,
                        "direction": "Combined",
                        **summarise_run(combined_vehicles, combined_queues),
                    }
                )
        print(f"Completed seed {seed}")

    results = pd.DataFrame(records)
    if len(results) != 1080:
        raise ValueError("Expected 1,080 replication metric rows.")
    counts = results.pivot(
        index=["seed", "factor", "direction"],
        columns="strategy",
        values="vehicles",
    )
    if not (counts["Phase3A"] == counts["Phase3F"]).all():
        raise ValueError("Phase 3A and Phase 3F vehicle counts differ.")
    if not (counts["Phase3A"] == counts["Phase4D"]).all():
        raise ValueError("Phase 3A and Phase 4D vehicle counts differ.")
    if not np.allclose(
        results["mean_journey_s"],
        81.0 + results["mean_delay_s"],
        atol=1e-9,
    ):
        raise ValueError("Journey-time identity failed.")
    return results


def evaluate_stress_results(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create 720 paired rows and 24 stress-summary rows."""

    records: list[dict[str, object]] = []
    paired_frames: list[pd.DataFrame] = []
    keys = ["period", "seed", "factor", "direction"]
    candidate = results.loc[results["strategy"] == "Phase4D"].drop(
        columns="strategy"
    )
    for benchmark in ("Phase3A", "Phase3F"):
        baseline = results.loc[results["strategy"] == benchmark].drop(
            columns="strategy"
        )
        paired = baseline.merge(
            candidate,
            on=keys,
            suffixes=("_baseline", "_4d"),
            validate="one_to_one",
        )
        paired["benchmark"] = benchmark
        paired["delay_change_s"] = (
            paired["mean_delay_s_4d"] - paired["mean_delay_s_baseline"]
        )
        paired["queue_change_veh"] = (
            paired["queue_at_hour_veh_4d"]
            - paired["queue_at_hour_veh_baseline"]
        )
        paired_frames.append(paired)

        for factor in FACTORS:
            for direction in ("EB", "WB", "Combined"):
                group = paired.loc[
                    (paired["factor"] == factor)
                    & (paired["direction"] == direction)
                ]
                low, high = bootstrap_mean_ci(group["delay_change_s"].to_numpy())
                within_limit = group["delay_change_s"] <= FAIRNESS_LIMIT_S
                fairness_applicable = (
                    benchmark == "Phase3A" and direction in ("EB", "WB")
                )
                records.append(
                    {
                        "factor": factor,
                        "direction": direction,
                        "benchmark": benchmark,
                        "replicates": len(group),
                        "baseline_delay_s": group[
                            "mean_delay_s_baseline"
                        ].mean(),
                        "phase4d_delay_s": group["mean_delay_s_4d"].mean(),
                        "mean_delay_change_s": group["delay_change_s"].mean(),
                        "ci_lower_s": low,
                        "ci_upper_s": high,
                        "seeds_better": int((group["delay_change_s"] < 0).sum()),
                        "baseline_queue_at_hour_veh": group[
                            "queue_at_hour_veh_baseline"
                        ].mean(),
                        "phase4d_queue_at_hour_veh": group[
                            "queue_at_hour_veh_4d"
                        ].mean(),
                        "mean_queue_change_veh": group[
                            "queue_change_veh"
                        ].mean(),
                        "fairness_applicable": fairness_applicable,
                        "mean_within_fairness": (
                            bool(
                                group["delay_change_s"].mean()
                                <= FAIRNESS_LIMIT_S
                            )
                            if fairness_applicable
                            else None
                        ),
                        "ci_upper_within_fairness": (
                            bool(high <= FAIRNESS_LIMIT_S)
                            if fairness_applicable
                            else None
                        ),
                        "seeds_within_fairness": (
                            int(within_limit.sum())
                            if fairness_applicable
                            else None
                        ),
                        "all_seeds_within_fairness": (
                            bool(within_limit.all())
                            if fairness_applicable
                            else None
                        ),
                    }
                )

    paired_results = pd.concat(paired_frames, ignore_index=True)
    summary = pd.DataFrame(records)
    if len(paired_results) != 720:
        raise ValueError("Expected 720 paired differences.")
    if len(summary) != 24 or not summary["replicates"].eq(30).all():
        raise ValueError("Expected 24 summary rows with 30 replicates each.")
    return paired_results, summary


def plot_results(summary: pd.DataFrame) -> None:
    """Create synthetic congestion and eastbound-fairness figures."""

    factors = np.asarray(FACTORS) * 100
    combined_3a = summary.loc[
        (summary["direction"] == "Combined")
        & (summary["benchmark"] == "Phase3A")
    ].sort_values("factor")
    combined_3f = summary.loc[
        (summary["direction"] == "Combined")
        & (summary["benchmark"] == "Phase3F")
    ].sort_values("factor")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        factors,
        combined_3a["baseline_delay_s"],
        marker="o",
        label="Phase 3A",
    )
    ax.plot(
        factors,
        combined_3f["baseline_delay_s"],
        marker="^",
        label="Phase 3F",
    )
    ax.plot(
        factors,
        combined_3a["phase4d_delay_s"],
        marker="s",
        label="Phase 4D",
    )
    ax.set_xlabel("Demand relative to baseline (%)")
    ax.set_ylabel("Mean simulated delay (seconds)")
    ax.set_title("Synthetic AM Congestion Stress Test")
    ax.set_xticks(factors)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CONGESTION_FIGURE_PATH, dpi=180)
    plt.close(fig)

    eastbound = summary.loc[
        (summary["direction"] == "EB")
        & (summary["benchmark"] == "Phase3A")
    ].sort_values("factor")
    changes = eastbound["mean_delay_change_s"].to_numpy()
    lower_error = changes - eastbound["ci_lower_s"].to_numpy()
    upper_error = eastbound["ci_upper_s"].to_numpy() - changes
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        factors,
        changes,
        yerr=np.vstack([lower_error, upper_error]),
        marker="o",
        capsize=5,
        label="Phase 4D minus Phase 3A",
    )
    ax.axhline(
        FAIRNESS_LIMIT_S,
        color="red",
        linestyle="--",
        label="2-second fairness threshold",
    )
    ax.set_xlabel("Demand relative to baseline (%)")
    ax.set_ylabel("Eastbound mean delay change (seconds)")
    ax.set_title("Synthetic AM Eastbound Fairness Stress Test")
    ax.set_xticks(factors)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FAIRNESS_FIGURE_PATH, dpi=180)
    plt.close(fig)


def main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run Phase 4D-2 and save its metrics, comparisons and figures."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    traffic, strategies = load_inputs()
    results = run_experiment(traffic, strategies)
    paired, summary = evaluate_stress_results(results)
    results.to_csv(METRICS_PATH, index=False)
    paired.to_csv(PAIRED_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    plot_results(summary)

    print("\nPhase 4D-2 results versus Phase 3A:")
    print(
        summary.loc[summary["benchmark"] == "Phase3A"].to_string(index=False)
    )
    print(f"\nWrote {METRICS_PATH}")
    print(f"Wrote {PAIRED_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {CONGESTION_FIGURE_PATH}")
    print(f"Wrote {FAIRNESS_FIGURE_PATH}")
    print("\nPhase 4D-2 completed.")
    return results, paired, summary


if __name__ == "__main__":
    main()
