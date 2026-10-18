"""Phases 4A to 4E: corridor vehicle simulation, multi-seed evaluation and delay-aware offsets.

Merged from the original phase runner scripts. Each section keeps the
functions of one phase; run a phase with ``python -m workflows <phase>`` from
the repository root. Outputs are written to outputs/data and outputs/figures.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "traffic-signals-matplotlib")
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from traffic_signals.delay_optimisation import evaluate_offsets, search_delay_offsets
from traffic_signals.demand_stress import create_demand_scenarios
from traffic_signals.offset_optimisation import circular_distance
from traffic_signals.segmentation import PERIOD_ORDER, assign_period
from traffic_signals.vehicle_simulation import simulate_direction


PHASE_DESCRIPTIONS = {
    "phase4a": "Phase 4A vehicle simulation",
    "phase4b": "Phase 4B multi-seed evaluation",
    "phase4c": "Phase 4C demand sensitivity",
    "phase4d1": "Phase 4D-1 delay-aware AM offsets",
    "phase4d2": "Phase 4D-2 stress validation",
    "phase4d3": "Phase 4D-3 robust fairness",
    "phase4e": "Phase 4E four-period offsets",
}

# ============================================================================
# Phase 4A vehicle simulation (python -m workflows phase4a)
# Formerly phase4_vehicle_simulation.py. Run the Phase 4A matched AM-peak vehicle and queue experiment.
# ============================================================================

DATA_DIR = Path("outputs/data")
PERIOD = "AM peak"
DURATION_S = 3600.0
SPEED_KMH = 40.0
HEADWAY_S = 2.0
RANDOM_SEED = 42
DIRECTIONS = ("EB", "WB")
TRAFFIC_PATH = DATA_DIR / "synthetic_traffic_15min.csv"
PHASE3A_PATH = DATA_DIR / "phase3a_signal_plans.csv"
PHASE3F_PATH = DATA_DIR / "phase3f_optimisation_summary.csv"
PHASE4A_SUMMARY_PATH = DATA_DIR / "phase4a_summary.csv"
VEHICLES_PATH = DATA_DIR / "phase4a_vehicle_journeys.csv"
PHASE4A_QUEUES_PATH = DATA_DIR / "phase4a_queue_summary.csv"
COMBINED_PATH = DATA_DIR / "phase4a_combined_comparison.csv"


def estimate_am_entry_flows(traffic: pd.DataFrame) -> dict[str, float]:
    """Estimate hourly entry flows from the 16 AM 15-minute observations."""

    df = traffic.copy()
    required = {"time", "intersection_id", "approach", "volume_15min"}
    if not required.issubset(df.columns):
        raise ValueError("Missing required traffic columns.")
    df["time"] = pd.to_datetime(df["time"], errors="raise")
    df["period"] = df["time"].apply(assign_period)
    df["intersection_id"] = df["intersection_id"].astype(str).str.upper()
    df["approach"] = df["approach"].astype(str).str.lower()
    am = df.loc[df["period"] == PERIOD]
    selections = {
        "EB": am.loc[
            (am["intersection_id"] == "A") & (am["approach"] == "eastbound"),
            "volume_15min",
        ],
        "WB": am.loc[
            (am["intersection_id"] == "C") & (am["approach"] == "westbound"),
            "volume_15min",
        ],
    }
    flows: dict[str, float] = {}
    for direction, counts in selections.items():
        numeric = pd.to_numeric(counts, errors="raise")
        if len(numeric) != 16:
            raise ValueError(f"Expected 16 AM observations for {direction}.")
        if not np.isfinite(numeric).all() or (numeric < 0).any():
            raise ValueError("Traffic counts must be finite and non-negative.")
        flows[direction] = float(numeric.mean() * 4.0)
    return flows


def generate_entry_times(
    hourly_flows: dict[str, float],
    duration_s: float = DURATION_S,
    random_seed: int = RANDOM_SEED,
) -> dict[str, np.ndarray]:
    """Generate matched Poisson vehicle counts and uniform entry times."""

    if set(hourly_flows) != set(DIRECTIONS):
        raise ValueError("Hourly flows must define EB and WB.")
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Duration must be positive and finite.")
    if any(not np.isfinite(flow) or flow < 0 for flow in hourly_flows.values()):
        raise ValueError("Hourly flows must be finite and non-negative.")

    rng = np.random.default_rng(random_seed)
    entry_times: dict[str, np.ndarray] = {}
    for direction in DIRECTIONS:
        expected_count = hourly_flows[direction] * duration_s / 3600.0
        vehicle_count = int(rng.poisson(expected_count))
        entry_times[direction] = np.sort(rng.uniform(0.0, duration_s, vehicle_count))
    return entry_times


def build_signal_strategies(
    phase3a: pd.DataFrame,
    phase3f: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    """Load the original and protected AM signal strategies."""

    am_plans = phase3a.loc[phase3a["period"] == PERIOD].set_index(
        "intersection_id"
    )
    if set(am_plans.index) != {"A", "B", "C"} or len(am_plans) != 3:
        raise ValueError("Expected one AM plan for intersections A, B and C.")
    cycle_values = am_plans["common_cycle_s"].unique()
    if len(cycle_values) != 1 or int(cycle_values[0]) != 150:
        raise ValueError("Expected a 150-second AM common cycle.")
    cycle = int(cycle_values[0])
    greens = am_plans["ew_green_s"].to_dict()
    original_offsets = am_plans["offset_s"].to_dict()

    matching = phase3f.loc[
        (phase3f["period"] == PERIOD)
        & (phase3f["strategy"] == "Phase3F_bidirectional")
    ]
    if len(matching) != 1:
        raise ValueError("Expected one Phase 3F AM strategy row.")
    new_plan = matching.iloc[0]
    optimised_offsets = {
        "A": 0.0,
        "B": float(new_plan["offset_b_s"]),
        "C": float(new_plan["offset_c_s"]),
    }
    if optimised_offsets != {"A": 0.0, "B": 28.0, "C": 72.0}:
        raise ValueError("Phase 3F AM offsets do not match the verified plan.")
    return {
        "Phase3A": {
            "cycle_s": cycle,
            "greens": greens,
            "offsets": original_offsets,
        },
        "Phase3F": {
            "cycle_s": cycle,
            "greens": greens,
            "offsets": optimised_offsets,
        },
    }


def run_experiment_phase4a(
    entry_times: dict[str, np.ndarray],
    strategies: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate both strategies with identical generated vehicles."""

    all_vehicles: list[pd.DataFrame] = []
    all_queues: list[pd.DataFrame] = []
    summary_records: list[dict[str, object]] = []
    for strategy_name, signal_plan in strategies.items():
        for direction in DIRECTIONS:
            vehicles, queues = simulate_direction(
                entry_times=entry_times[direction],
                direction=direction,
                signal_plan=signal_plan,
                speed_kmh=SPEED_KMH,
                headway_s=HEADWAY_S,
            )
            if vehicles.empty:
                raise ValueError("No vehicles generated.")
            vehicles["strategy"] = strategy_name
            vehicles["period"] = PERIOD
            vehicles["exits_after_horizon"] = vehicles["exit_time_s"] > DURATION_S
            queues["strategy"] = strategy_name
            queues["period"] = PERIOD
            summary_records.append(
                {
                    "strategy": strategy_name,
                    "direction": direction,
                    "vehicles": len(vehicles),
                    "mean_delay_s": vehicles["total_delay_s"].mean(),
                    "mean_journey_s": vehicles["journey_time_s"].mean(),
                    "mean_waiting_events": vehicles["waiting_events"].mean(),
                    "no_wait_fraction": (vehicles["waiting_events"] == 0).mean(),
                    "max_queue": int(queues["max_queue"].max()),
                    "exits_after_horizon": int(vehicles["exits_after_horizon"].sum()),
                    "last_exit_s": vehicles["exit_time_s"].max(),
                }
            )
            all_vehicles.append(vehicles)
            all_queues.append(queues)

    vehicle_results = pd.concat(all_vehicles, ignore_index=True)
    queue_results = pd.concat(all_queues, ignore_index=True)
    summary = pd.DataFrame(summary_records)
    combined = (
        vehicle_results.groupby("strategy", as_index=False, sort=False)
        .agg(
            vehicles=("vehicle_id", "count"),
            mean_delay_s=("total_delay_s", "mean"),
            mean_journey_s=("journey_time_s", "mean"),
            mean_waiting_events=("waiting_events", "mean"),
            no_wait_fraction=("waiting_events", lambda values: (values == 0).mean()),
            exits_after_horizon=("exits_after_horizon", "sum"),
            last_exit_s=("exit_time_s", "max"),
        )
    )
    maximum_queues = (
        queue_results.groupby("strategy", as_index=False, sort=False)["max_queue"]
        .max()
        .rename(columns={"max_queue": "max_queue"})
    )
    combined = combined.merge(maximum_queues, on="strategy", validate="one_to_one")

    counts = summary.pivot(index="direction", columns="strategy", values="vehicles")
    if not (counts["Phase3A"] == counts["Phase3F"]).all():
        raise ValueError("Strategies did not receive identical vehicle counts.")
    for direction in DIRECTIONS:
        original_entries = vehicle_results.loc[
            (vehicle_results["strategy"] == "Phase3A")
            & (vehicle_results["direction"] == direction),
            "entry_time_s",
        ].to_numpy()
        optimised_entries = vehicle_results.loc[
            (vehicle_results["strategy"] == "Phase3F")
            & (vehicle_results["direction"] == direction),
            "entry_time_s",
        ].to_numpy()
        if not np.array_equal(original_entries, optimised_entries):
            raise ValueError(f"Strategies did not receive identical {direction} entries.")
    free_flow_s = 900.0 / (SPEED_KMH / 3.6)
    if not np.allclose(
        vehicle_results["journey_time_s"],
        free_flow_s + vehicle_results["total_delay_s"],
    ):
        raise ValueError("Journey-time identity failed.")
    for strategy_name, signal_plan in strategies.items():
        cycle_s = float(signal_plan["cycle_s"])
        greens = signal_plan["greens"]
        offsets = signal_plan["offsets"]
        for direction in DIRECTIONS:
            selected = vehicle_results.loc[
                (vehicle_results["strategy"] == strategy_name)
                & (vehicle_results["direction"] == direction)
            ]
            for junction in ("A", "B", "C"):
                departures = selected[f"discharge_{junction.lower()}_s"].to_numpy()
                if np.any(np.diff(departures) < HEADWAY_S - 1e-9):
                    raise ValueError("FIFO discharge headway check failed.")
                phase = (departures - float(offsets[junction])) % cycle_s
                if np.any(phase >= float(greens[junction]) + 1e-9):
                    raise ValueError("A recorded discharge occurred outside green.")
    if not np.isfinite(vehicle_results.select_dtypes(include=["number"])).all().all():
        raise ValueError("Vehicle results contain non-finite values.")
    return summary, vehicle_results, queue_results, combined


def run_phase4a() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    traffic = pd.read_csv(TRAFFIC_PATH, parse_dates=["time"])
    phase3a = pd.read_csv(PHASE3A_PATH)
    phase3f = pd.read_csv(PHASE3F_PATH)
    hourly_flows = estimate_am_entry_flows(traffic)
    entry_times = generate_entry_times(hourly_flows)
    strategies = build_signal_strategies(phase3a, phase3f)
    summary, vehicles, queues, combined = run_experiment_phase4a(entry_times, strategies)

    summary.to_csv(PHASE4A_SUMMARY_PATH, index=False)
    vehicles.to_csv(VEHICLES_PATH, index=False)
    queues.to_csv(PHASE4A_QUEUES_PATH, index=False)
    combined.to_csv(COMBINED_PATH, index=False)

    print("AM hourly entry-flow estimates:", hourly_flows)
    print("Generated vehicles:", {key: len(value) for key, value in entry_times.items()})
    print("\nPhase 4A directional results:")
    print(summary.to_string(index=False))
    print("\nPhase 4A combined comparison:")
    print(combined.to_string(index=False))
    print(f"\nWrote {PHASE4A_SUMMARY_PATH}")
    print(f"Wrote {VEHICLES_PATH}")
    print(f"Wrote {PHASE4A_QUEUES_PATH}")
    print(f"Wrote {COMBINED_PATH}")

# ============================================================================
# Phase 4B multi-seed evaluation (python -m workflows phase4b)
# Formerly phase4b_multi_seed.py. Run Phase 4B across four periods and 30 matched arrival seeds.
# ============================================================================

FIGURE_DIR = Path("outputs/figures")
PHASE4B_PERIODS = list(PERIOD_ORDER)
PHASE4B_SEEDS = range(42, 72)
ENTRY_APPROACHES = {"EB": ("A", "E"), "WB": ("C", "W")}
STRATEGIES = ("Phase3A", "Phase3F")
PHASE4A_COMBINED_PATH = DATA_DIR / "phase4a_combined_comparison.csv"
PHASE4B_METRICS_PATH = DATA_DIR / "phase4b_replication_metrics.csv"
PHASE4B_PAIRED_PATH = DATA_DIR / "phase4b_paired_differences.csv"
PHASE4B_SUMMARY_PATH = DATA_DIR / "phase4b_uncertainty_summary.csv"
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


def run_experiment_phase4b(
    traffic: pd.DataFrame,
    phase3a: pd.DataFrame,
    phase3f: pd.DataFrame,
) -> pd.DataFrame:
    """Run all period, seed, strategy and direction combinations."""

    records: list[dict[str, object]] = []
    for period in PHASE4B_PERIODS:
        strategies = get_signal_strategies(phase3a, phase3f, period)
        for seed in PHASE4B_SEEDS:
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
    for period in PHASE4B_PERIODS:
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
        .reindex(PHASE4B_PERIODS)
    )
    x = np.arange(len(PHASE4B_PERIODS))
    y = combined["mean_delay_change_s"].to_numpy()
    lower = y - combined["ci_lower_s"].to_numpy()
    upper = combined["ci_upper_s"].to_numpy() - y
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(x, y, yerr=np.vstack([lower, upper]), fmt="o", capsize=5)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x, PHASE4B_PERIODS)
    ax.set_ylabel("Phase 3F minus Phase 3A delay (seconds)")
    ax.set_title("Synthetic Paired Delay Differences Across 30 Seeds")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_phase4b() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    results = run_experiment_phase4b(traffic, phase3a, phase3f)
    verify_phase4a_regression(results)
    paired, summary = evaluate_replicates(results)
    results.to_csv(PHASE4B_METRICS_PATH, index=False)
    paired.to_csv(PHASE4B_PAIRED_PATH, index=False)
    summary.to_csv(PHASE4B_SUMMARY_PATH, index=False)
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
    print(f"Wrote {PHASE4B_METRICS_PATH}")
    print(f"Wrote {PHASE4B_PAIRED_PATH}")
    print(f"Wrote {PHASE4B_SUMMARY_PATH}")
    print(f"Wrote {FIGURE_PATH}")
    return results, paired, summary

# ============================================================================
# Phase 4C demand sensitivity (python -m workflows phase4c)
# Formerly phase4c_demand_sensitivity.py. Run Phase 4C demand and congestion sensitivity experiments.
# ============================================================================

PHASE4C_PERIODS = list(PERIOD_ORDER)
FACTORS = (0.75, 1.0, 1.25, 1.5)
PHASE4C_SEEDS = range(42, 72)
PHASE4C_METRICS_PATH = DATA_DIR / "phase4c_replication_metrics.csv"
PHASE4C_QUEUES_PATH = DATA_DIR / "phase4c_approach_queues.csv"
PHASE4C_PAIRED_PATH = DATA_DIR / "phase4c_paired_differences.csv"
PHASE4C_SUMMARY_PATH = DATA_DIR / "phase4c_sensitivity_summary.csv"


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


def run_experiment_phase4c(
    traffic: pd.DataFrame,
    phase3a: pd.DataFrame,
    phase3f: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all period, seed, demand, strategy and direction combinations."""

    records: list[dict[str, object]] = []
    queue_records: list[pd.DataFrame] = []
    for period_index, period in enumerate(PHASE4C_PERIODS):
        strategies = get_signal_strategies(phase3a, phase3f, period)
        rates = _arrival_rates(traffic, period)
        for seed in PHASE4C_SEEDS:
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
    for period in PHASE4C_PERIODS:
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


def verify_phase4b_regression_phase4c(results: pd.DataFrame) -> None:
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
    for period in PHASE4C_PERIODS:
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


def _load_inputs_phase4c() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def run_phase4c() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run Phase 4C, save outputs and print the 48-row summary."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    traffic, phase3a, phase3f = _load_inputs_phase4c()
    results, approaches = run_experiment_phase4c(traffic, phase3a, phase3f)
    verify_phase4b_regression_phase4c(results)
    paired, summary = evaluate_demand_sensitivity(results)
    results.to_csv(PHASE4C_METRICS_PATH, index=False)
    approaches.to_csv(PHASE4C_QUEUES_PATH, index=False)
    paired.to_csv(PHASE4C_PAIRED_PATH, index=False)
    summary.to_csv(PHASE4C_SUMMARY_PATH, index=False)
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
    print(f"\nWrote {PHASE4C_METRICS_PATH}")
    print(f"Wrote {PHASE4C_QUEUES_PATH}")
    print(f"Wrote {PHASE4C_PAIRED_PATH}")
    print(f"Wrote {PHASE4C_SUMMARY_PATH}")
    for path in figure_paths:
        print(f"Wrote {path}")
    print("\nPhase 4C completed.")
    return results, approaches, paired, summary

# ============================================================================
# Phase 4D-1 delay-aware AM offsets (python -m workflows phase4d1)
# Formerly phase4d_delay_optimisation.py. Run Phase 4D-1 AM delay-aware offset optimisation and validation.
# ============================================================================

PHASE4D1_TRAIN_SEEDS = range(42, 45)
VALIDATION_SEEDS = range(45, 72)
FAIRNESS_LIMIT_S = 2.0
PHASE4D1_PLAN_PATH = DATA_DIR / "phase4d_selected_plan.csv"
PHASE4D1_TRAINING_PATH = DATA_DIR / "phase4d_training_metrics.csv"
VALIDATION_PATH = DATA_DIR / "phase4d_validation_metrics.csv"
PHASE4D1_SUMMARY_PATH = DATA_DIR / "phase4d_validation_summary.csv"


def _load_inputs_phase4d1() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        seed: generate_arrivals(traffic, PERIOD, seed) for seed in PHASE4D1_TRAIN_SEEDS
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


def verify_phase4b_regression_phase4d1(validation: pd.DataFrame) -> None:
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


def run_phase4d1() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select an AM candidate, validate it once and save all outputs."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    traffic, phase3a, phase3f = _load_inputs_phase4d1()
    result, strategies, training = optimise_training_offsets(
        traffic, phase3a, phase3f
    )
    selected = build_selected_plan(result, strategies["Phase3A"])
    validation = validate_unseen_seeds(
        traffic, strategies, result["offsets"]
    )
    verify_phase4b_regression_phase4d1(validation)
    summary = summarise_validation(validation)

    selected.to_csv(PHASE4D1_PLAN_PATH, index=False)
    training.to_csv(PHASE4D1_TRAINING_PATH, index=False)
    validation.to_csv(VALIDATION_PATH, index=False)
    summary.to_csv(PHASE4D1_SUMMARY_PATH, index=False)

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
    print(f"\nWrote {PHASE4D1_PLAN_PATH}")
    print(f"Wrote {PHASE4D1_TRAINING_PATH}")
    print(f"Wrote {VALIDATION_PATH}")
    print(f"Wrote {PHASE4D1_SUMMARY_PATH}")
    return selected, training, validation, summary

# ============================================================================
# Phase 4D-2 stress validation (python -m workflows phase4d2)
# Formerly phase4d2_stress_validation.py. Stress-test the frozen Phase 4D AM offsets on fresh arrival seeds.
# ============================================================================

PHASE4D2_SEEDS = range(72, 102)
PHASE4D_PATH = DATA_DIR / "phase4d_selected_plan.csv"
PHASE4D2_METRICS_PATH = DATA_DIR / "phase4d2_replication_metrics.csv"
PHASE4D2_PAIRED_PATH = DATA_DIR / "phase4d2_paired_differences.csv"
PHASE4D2_SUMMARY_PATH = DATA_DIR / "phase4d2_stress_summary.csv"
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


def run_experiment_phase4d2(
    traffic: pd.DataFrame,
    strategies: dict[str, dict[str, object]],
) -> pd.DataFrame:
    """Run 30 fresh seeds, four demand factors and three fixed strategies."""

    records: list[dict[str, object]] = []
    for seed in PHASE4D2_SEEDS:
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


def run_phase4d2() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run Phase 4D-2 and save its metrics, comparisons and figures."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    traffic, strategies = load_inputs()
    results = run_experiment_phase4d2(traffic, strategies)
    paired, summary = evaluate_stress_results(results)
    results.to_csv(PHASE4D2_METRICS_PATH, index=False)
    paired.to_csv(PHASE4D2_PAIRED_PATH, index=False)
    summary.to_csv(PHASE4D2_SUMMARY_PATH, index=False)
    plot_results(summary)

    print("\nPhase 4D-2 results versus Phase 3A:")
    print(
        summary.loc[summary["benchmark"] == "Phase3A"].to_string(index=False)
    )
    print(f"\nWrote {PHASE4D2_METRICS_PATH}")
    print(f"Wrote {PHASE4D2_PAIRED_PATH}")
    print(f"Wrote {PHASE4D2_SUMMARY_PATH}")
    print(f"Wrote {CONGESTION_FIGURE_PATH}")
    print(f"Wrote {FAIRNESS_FIGURE_PATH}")
    print("\nPhase 4D-2 completed.")
    return results, paired, summary

# ============================================================================
# Phase 4D-3 robust fairness (python -m workflows phase4d3)
# Formerly phase4d3_robust_fairness.py. Select and holdout-test an AM offset plan with robust fairness checks.
# ============================================================================

SCREEN_FACTORS = (1.0, 1.5)
SCREEN_SEEDS = range(102, 106)
PHASE4D3_TRAIN_SEEDS = range(102, 114)
PHASE4D3_HOLDOUT_SEEDS = range(114, 144)
SCREEN_LIMIT_S = 3.0
BASELINE_PAIR = (36, 81)
PHASE3F_PAIR = (28, 72)
PHASE4D_PAIR = (32, 68)
PHASE4D3_SCREENING_PATH = DATA_DIR / "phase4d3_screening.csv"
PHASE4D3_TRAINING_PATH = DATA_DIR / "phase4d3_training_candidates.csv"
PHASE4D3_PLAN_PATH = DATA_DIR / "phase4d3_selected_plan.csv"
PHASE4D3_HOLDOUT_PATH = DATA_DIR / "phase4d3_holdout_metrics.csv"
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

    all_training = prepare_arrivals(traffic, PHASE4D3_TRAIN_SEEDS)
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

    holdout_arrivals = prepare_arrivals(traffic, PHASE4D3_HOLDOUT_SEEDS)
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


def run_phase4d3() -> tuple[
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
                "training_seeds": len(PHASE4D3_TRAIN_SEEDS),
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

    screening.to_csv(PHASE4D3_SCREENING_PATH, index=False)
    training_summary.to_csv(PHASE4D3_TRAINING_PATH, index=False)
    selected_plan.to_csv(PHASE4D3_PLAN_PATH, index=False)
    holdout.to_csv(PHASE4D3_HOLDOUT_PATH, index=False)
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
    print(f"\nWrote {PHASE4D3_SCREENING_PATH}")
    print(f"Wrote {PHASE4D3_TRAINING_PATH}")
    print(f"Wrote {PHASE4D3_PLAN_PATH}")
    print(f"Wrote {PHASE4D3_HOLDOUT_PATH}")
    print(f"Wrote {HOLDOUT_SUMMARY_PATH}")
    return screening, training_summary, selected_plan, holdout, holdout_summary

# ============================================================================
# Phase 4E four-period offsets (python -m workflows phase4e)
# Formerly phase4e_four_period_optimisation.py. Select and validate fairness-protected offsets for four operating periods.
# ============================================================================

PHASE4E_PERIODS = ("Overnight", "AM peak", "Interpeak", "PM peak")
OPTIMISE_PERIODS = ("Overnight", "Interpeak", "PM peak")
PHASE4E_TRAIN_SEEDS = range(144, 156)
PHASE4E_HOLDOUT_SEEDS = range(156, 186)
FROZEN_AM_PATH = DATA_DIR / "phase4d3_selected_plan.csv"
PHASE4E_SCREENING_PATH = DATA_DIR / "phase4e_screening.csv"
PHASE4E_TRAINING_PATH = DATA_DIR / "phase4e_training_candidates.csv"
SELECTED_PATH = DATA_DIR / "phase4e_selected_plans.csv"
PHASE4E_HOLDOUT_PATH = DATA_DIR / "phase4e_holdout_metrics.csv"
PHASE4E_SUMMARY_PATH = DATA_DIR / "phase4e_holdout_summary.csv"


def prepare_period_arrivals(
    traffic: pd.DataFrame,
    period: str,
    seeds: Iterable[int],
) -> dict[int, dict[float, dict[str, np.ndarray]]]:
    """Create matched nested demand scenarios for one operating period."""

    if period not in PHASE4E_PERIODS:
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

    period_index = PHASE4E_PERIODS.index(period)
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
    screen_seed_values = tuple(PHASE4E_TRAIN_SEEDS)[:4]
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


def _load_inputs_phase4e() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    traffic, phase3a, phase3f, frozen_am = _load_inputs_phase4e()
    selected_pairs: dict[str, tuple[int, int]] = {
        "AM peak": (
            int(frozen_am.iloc[0]["B_offset_s"]),
            int(frozen_am.iloc[0]["C_offset_s"]),
        )
    }
    screening_frames: list[pd.DataFrame] = []
    training_frames: list[pd.DataFrame] = []
    selection_records: list[dict[str, object]] = []

    for period in PHASE4E_PERIODS:
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

        arrivals = prepare_period_arrivals(traffic, period, PHASE4E_TRAIN_SEEDS)
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
        selected_plans["period"], categories=PHASE4E_PERIODS, ordered=True
    )
    selected_plans = selected_plans.sort_values("period").reset_index(drop=True)
    selected_plans["period"] = selected_plans["period"].astype(str)

    screening_results.to_csv(PHASE4E_SCREENING_PATH, index=False)
    training_results.to_csv(PHASE4E_TRAINING_PATH, index=False)
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
    for period in PHASE4E_PERIODS:
        strategies = get_signal_strategies(phase3a, phase3f, period)
        base_plan = strategies["Phase3A"]
        original = strategies["Phase3A"]["offsets"]
        previous = strategies["Phase3F"]["offsets"]
        pairs = {
            "Phase3A": (int(original["B"]), int(original["C"])),
            "Phase3F": (int(previous["B"]), int(previous["C"])),
            "Phase4E": selected_pairs[period],
        }
        arrivals = prepare_period_arrivals(traffic, period, PHASE4E_HOLDOUT_SEEDS)
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
    metrics.to_csv(PHASE4E_HOLDOUT_PATH, index=False)
    summary.to_csv(PHASE4E_SUMMARY_PATH, index=False)
    return metrics, summary


def plot_holdout_results(metrics: pd.DataFrame) -> None:
    """Save one synthetic combined-delay comparison figure per period."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for period in PHASE4E_PERIODS:
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


def run_phase4e() -> tuple[pd.DataFrame, pd.DataFrame]:
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


PHASES = {
    "phase4a": run_phase4a,
    "phase4b": run_phase4b,
    "phase4c": run_phase4c,
    "phase4d1": run_phase4d1,
    "phase4d2": run_phase4d2,
    "phase4d3": run_phase4d3,
    "phase4e": run_phase4e,
}
