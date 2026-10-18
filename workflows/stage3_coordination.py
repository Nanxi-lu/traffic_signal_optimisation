"""Phase 3: common-cycle coordination, progression and offset optimisation.

Merged from the original phase runner scripts. Each section keeps the
functions of one phase; run a phase with ``python -m workflows <phase>`` from
the repository root. Outputs are written to outputs/data and outputs/figures.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "traffic-signals-matplotlib")
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from traffic_signals.bidirectional import count_directional_green
from traffic_signals.bidirectional_optimisation import (
    DIRECTIONS as PHASE3F_DIRECTIONS,
    SPEEDS as PHASE3F_SPEEDS,
    optimise_bidirectional,
)
from traffic_signals.coordination import (
    calculate_offsets,
    is_in_green,
    optimise_fixed_cycle_split,
)
from traffic_signals.demand_weights import calculate_direction_weights
from traffic_signals.offset_optimisation import (
    SPEEDS,
    circular_distance,
    count_all_green,
    optimise_offsets,
)
from traffic_signals.optimisation import LOST_TIME_S
from traffic_signals.progression import (
    CORRIDOR_DISTANCES_M,
    DESIGN_SPEED_KMH,
    calculate_travel_times,
    summarise_progression,
)
from traffic_signals.segmentation import PERIOD_ORDER


PHASE_DESCRIPTIONS = {
    "phase3a": "Phase 3A coordination",
    "phase3b": "Phase 3B time-space diagrams",
    "phase3c": "Phase 3C speed sensitivity",
    "phase3d": "Phase 3D robust offsets",
    "phase3e": "Phase 3E bidirectional evaluation",
    "phase3f": "Phase 3F demand-weighted offsets",
}

# ============================================================================
# Phase 3A coordination (python -m workflows phase3a)
# Formerly phase3_coordination.py. Create initial common-cycle plans and ideal eastbound offsets.
# ============================================================================

OUTPUT_DIR = Path("outputs/data")
DEMAND_PATH = OUTPUT_DIR / "phase2_green_split_evaluation.csv"
OLD_PLANS_PATH = OUTPUT_DIR / "phase2_optimised_plans.csv"
COMMON_CYCLES_PATH = OUTPUT_DIR / "phase3a_common_cycles.csv"
SIGNAL_PLANS_PATH = OUTPUT_DIR / "phase3a_signal_plans.csv"
PHASE3A_BANDWIDTH_PATH = OUTPUT_DIR / "phase3a_bandwidth.csv"


def build_coordinated_plans(
    traffic: pd.DataFrame,
    old_plans: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build common cycles, fixed splits, offsets and ideal EB bandwidths."""

    common_cycles = (
        old_plans.groupby("period", as_index=False)["optimal_cycle_s"]
        .max()
        .rename(columns={"optimal_cycle_s": "common_cycle_s"})
    )
    if len(common_cycles) != 4:
        raise ValueError("Expected four common time-of-day cycles")
    common_cycles["period"] = pd.Categorical(
        common_cycles["period"], categories=PERIOD_ORDER, ordered=True
    )
    common_cycles = common_cycles.sort_values("period").reset_index(drop=True)

    observations = traffic.merge(
        common_cycles,
        on="period",
        how="left",
        validate="many_to_one",
    )
    if observations["common_cycle_s"].isna().any():
        raise ValueError("At least one observation did not receive a common cycle")

    records: list[dict[str, object]] = []
    for (junction, period), group in observations.groupby(
        ["intersection_id", "period"], sort=True
    ):
        cycle_s = int(group["common_cycle_s"].iloc[0])
        result = optimise_fixed_cycle_split(
            group["ew_y"].to_numpy(),
            group["ns_y"].to_numpy(),
            cycle_s,
        )
        travel_times, offsets = calculate_offsets(cycle_s)
        records.append(
            {
                "intersection_id": junction,
                "period": period,
                "common_cycle_s": cycle_s,
                "ew_green_s": result["ew_green_s"],
                "ns_green_s": result["ns_green_s"],
                "lost_time_s": LOST_TIME_S,
                "travel_time_from_a_s": travel_times[junction],
                "offset_s": offsets[junction],
                "warning_intervals": result["warning_intervals"],
                "total_shortfall": result["total_shortfall"],
                "min_capacity_margin": result["min_capacity_margin"],
            }
        )

    signal_plans = pd.DataFrame(records)
    signal_plans["period"] = pd.Categorical(
        signal_plans["period"], categories=PERIOD_ORDER, ordered=True
    )
    signal_plans = signal_plans.sort_values(
        ["intersection_id", "period"]
    ).reset_index(drop=True)

    bandwidth_records: list[dict[str, object]] = []
    for period, group in signal_plans.groupby("period", observed=True):
        cycle_s = int(group["common_cycle_s"].iloc[0])
        bandwidth_s = float(group["ew_green_s"].min())
        bandwidth_records.append(
            {
                "period": period,
                "common_cycle_s": cycle_s,
                "eastbound_bandwidth_s": bandwidth_s,
                "bandwidth_ratio": bandwidth_s / cycle_s,
            }
        )

        plan_lookup = group.set_index("intersection_id")
        travel_times, offsets = calculate_offsets(cycle_s)
        for departure_s in (0.0, bandwidth_s / 2.0, bandwidth_s - 0.01):
            for junction in ("A", "B", "C"):
                arrival_s = departure_s + travel_times[junction]
                green_s = float(plan_lookup.loc[junction, "ew_green_s"])
                if not is_in_green(
                    arrival_s,
                    offsets[junction],
                    green_s,
                    cycle_s,
                ):
                    raise ValueError("Ideal bandwidth verification failed")

    bandwidth_summary = pd.DataFrame(bandwidth_records)
    bandwidth_summary["period"] = pd.Categorical(
        bandwidth_summary["period"], categories=PERIOD_ORDER, ordered=True
    )
    bandwidth_summary = bandwidth_summary.sort_values("period").reset_index(drop=True)

    if len(signal_plans) != 12 or len(bandwidth_summary) != 4:
        raise ValueError("Coordination outputs have unexpected row counts")
    allocated_cycle = (
        signal_plans["ew_green_s"]
        + signal_plans["ns_green_s"]
        + signal_plans["lost_time_s"]
    )
    if not np.allclose(allocated_cycle, signal_plans["common_cycle_s"]):
        raise ValueError("Green allocations and lost time do not equal the cycle")
    return common_cycles, signal_plans, bandwidth_summary


def run_phase3a() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    traffic = pd.read_csv(DEMAND_PATH)
    old_plans = pd.read_csv(OLD_PLANS_PATH)
    common_cycles, signal_plans, bandwidth_summary = build_coordinated_plans(
        traffic,
        old_plans,
    )

    common_cycles.to_csv(COMMON_CYCLES_PATH, index=False)
    signal_plans.to_csv(SIGNAL_PLANS_PATH, index=False)
    bandwidth_summary.to_csv(PHASE3A_BANDWIDTH_PATH, index=False)

    print("Common corridor cycles:")
    print(common_cycles.to_string(index=False))
    print("\nCoordinated signal plans:")
    print(signal_plans.to_string(index=False))
    print("\nIdeal eastbound bandwidth:")
    print(bandwidth_summary.to_string(index=False))
    print(f"\nTotal capacity warnings: {int(signal_plans['warning_intervals'].sum())}")
    print(f"Wrote {COMMON_CYCLES_PATH}")
    print(f"Wrote {SIGNAL_PLANS_PATH}")
    print(f"Wrote {PHASE3A_BANDWIDTH_PATH}")

# ============================================================================
# Phase 3B time-space diagrams (python -m workflows phase3b)
# Formerly phase3_time_space.py. Generate and numerically verify Phase 3B eastbound time-space diagrams.
# ============================================================================

PHASE3B_PLANS_PATH = Path("outputs/data/phase3a_signal_plans.csv")
PHASE3B_BANDWIDTH_PATH = Path("outputs/data/phase3a_bandwidth.csv")
VERIFICATION_PATH = Path("outputs/data/phase3b_bandwidth_verification.csv")
FIGURES_DIR = Path("outputs/figures")


def _slug(period: str) -> str:
    return period.lower().replace(" ", "_")


def plot_time_space_diagram(
    period_plans: pd.DataFrame,
    period: str,
    bandwidth_s: float,
    output_path: Path,
) -> None:
    """Plot repeating green windows and the ideal progression band."""

    cycle_s = float(period_plans["common_cycle_s"].iloc[0])
    travel_times = calculate_travel_times(DESIGN_SPEED_KMH)
    plans = period_plans.set_index("intersection_id")
    distances = np.array([CORRIDOR_DISTANCES_M[name] for name in ("A", "B", "C")])
    trajectory_start = np.array([travel_times[name] for name in ("A", "B", "C")])
    trajectory_end = trajectory_start + bandwidth_s
    horizon_s = max(cycle_s + travel_times["C"], trajectory_end[-1] + 5.0)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for intersection in ("A", "B", "C"):
        distance_m = CORRIDOR_DISTANCES_M[intersection]
        offset_s = float(plans.loc[intersection, "offset_s"])
        green_s = float(plans.loc[intersection, "ew_green_s"])
        for cycle_index in range(-1, int(np.ceil(horizon_s / cycle_s)) + 1):
            start_s = offset_s + cycle_index * cycle_s
            end_s = start_s + green_s
            visible_start = max(0.0, start_s)
            visible_end = min(horizon_s, end_s)
            if visible_start < visible_end:
                ax.plot(
                    [visible_start, visible_end],
                    [distance_m, distance_m],
                    color="seagreen",
                    linewidth=12,
                    solid_capstyle="butt",
                )

    ax.fill_betweenx(
        distances,
        trajectory_start,
        trajectory_end,
        color="cornflowerblue",
        alpha=0.25,
        label=f"Ideal departure band ({bandwidth_s:.0f} s)",
    )
    for departure_s, label in (
        (0.0, "Band start"),
        (bandwidth_s / 2.0, "Sample vehicle"),
        (bandwidth_s - 0.01, "Band end"),
    ):
        ax.plot(
            trajectory_start + departure_s,
            distances,
            linewidth=1.8,
            label=label,
        )

    ax.scatter(
        [0.0, 0.0, 0.0],
        distances,
        color="black",
        s=24,
        label="Intersections",
        zorder=3,
    )
    ax.set_xlim(0.0, horizon_s)
    ax.set_ylim(-40.0, 940.0)
    ax.set_yticks(distances, labels=["A · 0 m", "B · 400 m", "C · 900 m"])
    ax.set_xlabel("Time from reference cycle start (seconds)")
    ax.set_ylabel("Distance eastbound from Intersection A")
    ax.set_title(
        f"Synthetic {period} eastbound time-space diagram\n"
        f"Design speed {DESIGN_SPEED_KMH:.0f} km/h; idealised effective-green windows"
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_phase3b() -> None:
    plans = pd.read_csv(PHASE3B_PLANS_PATH)
    phase3a_bandwidth = pd.read_csv(PHASE3B_BANDWIDTH_PATH)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for period in PERIOD_ORDER:
        period_plans = plans.loc[plans["period"] == period]
        expected = phase3a_bandwidth.loc[
            phase3a_bandwidth["period"] == period
        ].iloc[0]
        summary = summarise_progression(period_plans, period, DESIGN_SPEED_KMH)
        summary["phase3a_bandwidth_s"] = float(expected["eastbound_bandwidth_s"])
        summary["phase3a_bandwidth_ratio"] = float(expected["bandwidth_ratio"])
        summary["fraction_matches_phase3a"] = bool(
            np.isclose(
                summary["progression_fraction"],
                summary["phase3a_bandwidth_ratio"],
            )
        )
        records.append(summary)
        plot_time_space_diagram(
            period_plans,
            period,
            float(expected["eastbound_bandwidth_s"]),
            FIGURES_DIR / f"phase3b_time_space_{_slug(period)}.png",
        )

    verification = pd.DataFrame.from_records(records)
    if len(verification) != 4 or not verification["fraction_matches_phase3a"].all():
        raise ValueError("Phase 3B numerical bandwidth verification failed")
    verification.to_csv(VERIFICATION_PATH, index=False)
    print("Phase 3B bandwidth verification:")
    print(verification.to_string(index=False))
    print(f"\nWrote {VERIFICATION_PATH}")
    print(f"Wrote four figures under {FIGURES_DIR}")

# ============================================================================
# Phase 3C speed sensitivity (python -m workflows phase3c)
# Formerly phase3_speed_sensitivity.py. Evaluate idealised eastbound progression at five vehicle speeds.
# ============================================================================

PHASE3C_PLANS_PATH = Path("outputs/data/phase3a_signal_plans.csv")
RESULTS_PATH = Path("outputs/data/phase3c_speed_sensitivity.csv")
PHASE3C_FIGURE_PATH = Path("outputs/figures/phase3c_speed_sensitivity.png")
TEST_SPEEDS_KMH = (30.0, 35.0, 40.0, 45.0, 50.0)


def build_speed_sensitivity(
    plans: pd.DataFrame,
    speeds_kmh: tuple[float, ...] = TEST_SPEEDS_KMH,
) -> pd.DataFrame:
    """Evaluate whole-second departures for every period and test speed."""

    records: list[dict[str, object]] = []
    for period in PERIOD_ORDER:
        period_plans = plans.loc[plans["period"] == period]
        for speed_kmh in speeds_kmh:
            record = summarise_progression(period_plans, period, speed_kmh)
            record["design_speed"] = speed_kmh == DESIGN_SPEED_KMH
            records.append(record)
    results = pd.DataFrame.from_records(records)
    if len(results) != len(PERIOD_ORDER) * len(speeds_kmh):
        raise ValueError("Speed-sensitivity output has an unexpected row count")
    return results


def plot_speed_sensitivity(results: pd.DataFrame, output_path: Path) -> None:
    """Plot idealised all-green progression fraction against vehicle speed."""

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for period in PERIOD_ORDER:
        group = results.loc[results["period"] == period].sort_values("speed_kmh")
        ax.plot(
            group["speed_kmh"],
            group["progression_fraction"] * 100.0,
            marker="o",
            linewidth=2,
            label=period,
        )
    ax.axvline(
        DESIGN_SPEED_KMH,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Design speed",
    )
    ax.set_xlabel("Vehicle speed (km/h)")
    ax.set_ylabel("Idealised all-green departures (%)")
    ax.set_title(
        "Synthetic eastbound green-wave speed sensitivity\n"
        "Fixed Phase 3A cycles, offsets and effective-green allocations"
    )
    ax.set_xticks(list(TEST_SPEEDS_KMH))
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_phase3c() -> None:
    plans = pd.read_csv(PHASE3C_PLANS_PATH)
    results = build_speed_sensitivity(plans)

    design_results = results.loc[results["design_speed"]].set_index("period")
    expected = {
        "Overnight": 0.375,
        "AM peak": 0.580,
        "Interpeak": 0.425,
        "PM peak": 0.580,
    }
    for period, expected_fraction in expected.items():
        actual = float(design_results.loc[period, "progression_fraction"])
        if abs(actual - expected_fraction) > 1e-12:
            raise ValueError(f"Design-speed verification failed for {period}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    plot_speed_sensitivity(results, PHASE3C_FIGURE_PATH)
    print("Phase 3C speed sensitivity:")
    print(results.to_string(index=False))
    print(f"\nWrote {RESULTS_PATH}")
    print(f"Wrote {PHASE3C_FIGURE_PATH}")

# ============================================================================
# Phase 3D robust offsets (python -m workflows phase3d)
# Formerly phase3_offset_optimisation.py. Run Phase 3D robust green-wave offset optimisation.
# ============================================================================

DATA_DIR = Path("outputs/data")
FIGURE_DIR = Path("outputs/figures")
PHASE3D_PLANS_PATH = DATA_DIR / "phase3a_signal_plans.csv"
PHASE3D_SUMMARY_PATH = DATA_DIR / "phase3d_optimised_offsets.csv"
COMPARISON_PATH = DATA_DIR / "phase3d_robustness_comparison.csv"


def build_offset_optimisation(
    plans: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Optimise each period and return summary and speed-level results."""

    if len(plans) != 12:
        raise ValueError("Expected 12 Phase 3A signal-plan rows.")

    results: list[dict[str, object]] = []
    speed_results: list[dict[str, object]] = []
    for period in PERIOD_ORDER:
        group = plans.loc[plans["period"] == period].set_index("intersection_id")
        if set(group.index) != {"A", "B", "C"} or len(group) != 3:
            raise ValueError(f"{period} must contain one plan for A, B and C.")
        cycles = group["common_cycle_s"].unique()
        if len(cycles) != 1:
            raise ValueError(f"{period} must use one common cycle.")

        cycle = int(cycles[0])
        greens = group["ew_green_s"].to_dict()
        baseline_offsets = group["offset_s"].to_dict()
        baseline_counts = [
            count_all_green(cycle, greens, baseline_offsets, speed)
            for speed in SPEEDS
        ]
        optimum = optimise_offsets(cycle, greens, baseline_offsets)
        optimal_offsets = optimum["offsets"]
        optimal_counts = optimum["counts"]

        baseline_fractions = [count / cycle for count in baseline_counts]
        optimal_fractions = [count / cycle for count in optimal_counts]
        design_index = SPEEDS.index(40.0)
        results.append(
            {
                "period": period,
                "common_cycle_s": cycle,
                "baseline_offset_b_s": baseline_offsets["B"],
                "baseline_offset_c_s": baseline_offsets["C"],
                "optimal_offset_b_s": optimal_offsets["B"],
                "optimal_offset_c_s": optimal_offsets["C"],
                "offset_change_b_s": circular_distance(
                    optimal_offsets["B"], baseline_offsets["B"], cycle
                ),
                "offset_change_c_s": circular_distance(
                    optimal_offsets["C"], baseline_offsets["C"], cycle
                ),
                "baseline_mean": sum(baseline_fractions) / len(SPEEDS),
                "optimal_mean": sum(optimal_fractions) / len(SPEEDS),
                "mean_change": (
                    sum(optimal_fractions) - sum(baseline_fractions)
                )
                / len(SPEEDS),
                "baseline_worst": min(baseline_fractions),
                "optimal_worst": min(optimal_fractions),
                "worst_change": min(optimal_fractions) - min(baseline_fractions),
                "baseline_design_40": baseline_fractions[design_index],
                "optimal_design_40": optimal_fractions[design_index],
                "design_40_change": (
                    optimal_fractions[design_index]
                    - baseline_fractions[design_index]
                ),
            }
        )
        for speed, baseline_count, optimal_count in zip(
            SPEEDS, baseline_counts, optimal_counts, strict=True
        ):
            speed_results.append(
                {
                    "period": period,
                    "speed_kmh": speed,
                    "common_cycle_s": cycle,
                    "baseline_all_green": baseline_count,
                    "optimal_all_green": optimal_count,
                    "baseline_fraction": baseline_count / cycle,
                    "optimal_fraction": optimal_count / cycle,
                    "fraction_change": (optimal_count - baseline_count) / cycle,
                }
            )

    summary = pd.DataFrame(results)
    comparison = pd.DataFrame(speed_results)
    if len(summary) != 4 or len(comparison) != 20:
        raise ValueError("Phase 3D outputs have unexpected row counts.")
    if not (summary["optimal_mean"] >= summary["baseline_mean"] - 1e-12).all():
        raise ValueError("Optimisation reduced the primary mean objective.")
    return summary, comparison


def plot_robustness(comparison: pd.DataFrame, figure_dir: Path) -> None:
    """Plot original and optimised progression for each operating period."""

    figure_dir.mkdir(parents=True, exist_ok=True)
    for period in PERIOD_ORDER:
        group = comparison.loc[comparison["period"] == period].sort_values(
            "speed_kmh"
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            group["speed_kmh"],
            group["baseline_fraction"] * 100.0,
            marker="o",
            label="Original offsets",
        )
        ax.plot(
            group["speed_kmh"],
            group["optimal_fraction"] * 100.0,
            marker="s",
            label="Robust offsets",
        )
        ax.set_xlabel("Vehicle speed (km/h)")
        ax.set_ylabel("Idealised all-green departures (%)")
        ax.set_title(f"Synthetic offset robustness: {period}")
        ax.set_xticks(SPEEDS)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        filename = period.lower().replace(" ", "_")
        fig.savefig(figure_dir / f"phase3d_robustness_{filename}.png", dpi=180)
        plt.close(fig)


def run_phase3d() -> None:
    plans = pd.read_csv(PHASE3D_PLANS_PATH)
    summary, comparison = build_offset_optimisation(plans)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(PHASE3D_SUMMARY_PATH, index=False)
    comparison.to_csv(COMPARISON_PATH, index=False)
    plot_robustness(comparison, FIGURE_DIR)

    print("Phase 3D offset optimisation:")
    print(summary.to_string(index=False))
    print("\nSpeed sensitivity comparison:")
    print(comparison.to_string(index=False))
    print(f"\nWrote {PHASE3D_SUMMARY_PATH}")
    print(f"Wrote {COMPARISON_PATH}")
    print(f"Wrote four figures under {FIGURE_DIR}")

# ============================================================================
# Phase 3E bidirectional evaluation (python -m workflows phase3e)
# Formerly phase3_bidirectional.py. Run Phase 3E bidirectional progression evaluation.
# ============================================================================

DIRECTIONS = ("EB", "WB")
PHASE3E_STRATEGIES = ("Original", "Optimised")
ORIGINAL_PLANS_PATH = DATA_DIR / "phase3a_signal_plans.csv"
ROBUST_OFFSETS_PATH = DATA_DIR / "phase3d_optimised_offsets.csv"
PREVIOUS_COMPARISON_PATH = DATA_DIR / "phase3d_robustness_comparison.csv"
EVALUATION_PATH = DATA_DIR / "phase3e_bidirectional_evaluation.csv"
DIRECTION_SUMMARY_PATH = DATA_DIR / "phase3e_direction_summary.csv"
COMBINED_SUMMARY_PATH = DATA_DIR / "phase3e_combined_summary.csv"


def build_bidirectional_evaluation(
    original_plans: pd.DataFrame,
    robust_offsets: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate two offset strategies in both directions and at five speeds."""

    if len(original_plans) != 12:
        raise ValueError("Expected 12 Phase 3A signal-plan rows.")
    if len(robust_offsets) != 4:
        raise ValueError("Expected four Phase 3D offset rows.")

    records: list[dict[str, object]] = []
    for period in PERIOD_ORDER:
        group = original_plans.loc[
            original_plans["period"] == period
        ].set_index("intersection_id")
        if set(group.index) != {"A", "B", "C"} or len(group) != 3:
            raise ValueError(f"{period} must contain one plan for A, B and C.")
        cycle_values = group["common_cycle_s"].unique()
        if len(cycle_values) != 1:
            raise ValueError(f"{period} must use one common cycle.")
        cycle_s = int(cycle_values[0])
        greens = group["ew_green_s"].to_dict()
        baseline = group["offset_s"].to_dict()

        matching = robust_offsets.loc[robust_offsets["period"] == period]
        if len(matching) != 1:
            raise ValueError(f"Expected one Phase 3D offset row for {period}.")
        candidate = matching.iloc[0]
        optimised = {
            "A": 0.0,
            "B": float(candidate["optimal_offset_b_s"]),
            "C": float(candidate["optimal_offset_c_s"]),
        }

        for strategy, offsets in {
            "Original": baseline,
            "Optimised": optimised,
        }.items():
            for direction in DIRECTIONS:
                for speed_kmh in SPEEDS:
                    count = count_directional_green(
                        cycle_s,
                        greens,
                        offsets,
                        speed_kmh,
                        direction,
                    )
                    records.append(
                        {
                            "period": period,
                            "strategy": strategy,
                            "direction": direction,
                            "speed_kmh": speed_kmh,
                            "common_cycle_s": cycle_s,
                            "all_green_departures": count,
                            "progression_fraction": count / cycle_s,
                        }
                    )

    evaluation = pd.DataFrame(records)
    if len(evaluation) != 80:
        raise ValueError("Expected 80 bidirectional evaluation rows.")
    return evaluation


def build_summaries(
    evaluation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build direction-level and equally weighted bidirectional summaries."""

    direction_summary = (
        evaluation.groupby(
            ["period", "strategy", "direction"],
            as_index=False,
            sort=False,
        )
        .agg(
            mean_progression=("progression_fraction", "mean"),
            worst_progression=("progression_fraction", "min"),
        )
    )
    design_speed = evaluation.loc[
        evaluation["speed_kmh"] == 40.0,
        ["period", "strategy", "direction", "progression_fraction"],
    ].rename(columns={"progression_fraction": "design_speed_progression"})
    direction_summary = direction_summary.merge(
        design_speed,
        on=["period", "strategy", "direction"],
        validate="one_to_one",
    )

    combined_summary = (
        evaluation.groupby(
            ["period", "strategy"],
            as_index=False,
            sort=False,
        )
        .agg(
            bidirectional_mean=("progression_fraction", "mean"),
            bidirectional_worst=("progression_fraction", "min"),
        )
    )
    combined_design = (
        evaluation.loc[evaluation["speed_kmh"] == 40.0]
        .groupby(["period", "strategy"], as_index=False, sort=False)
        .agg(bidirectional_design_40=("progression_fraction", "mean"))
    )
    combined_summary = combined_summary.merge(
        combined_design,
        on=["period", "strategy"],
        validate="one_to_one",
    )
    if len(direction_summary) != 16 or len(combined_summary) != 8:
        raise ValueError("Bidirectional summaries have unexpected row counts.")
    return direction_summary, combined_summary


def verify_eastbound_regression(
    evaluation: pd.DataFrame,
    previous: pd.DataFrame,
) -> None:
    """Verify Phase 3E eastbound results against the Phase 3D output."""

    eastbound = evaluation.loc[evaluation["direction"] == "EB"]
    for strategy, old_column in (
        ("Original", "baseline_fraction"),
        ("Optimised", "optimal_fraction"),
    ):
        current = eastbound.loc[eastbound["strategy"] == strategy]
        comparison = current.merge(
            previous[["period", "speed_kmh", old_column]],
            on=["period", "speed_kmh"],
            validate="one_to_one",
        )
        if len(comparison) != 20 or not np.allclose(
            comparison["progression_fraction"], comparison[old_column]
        ):
            raise ValueError(f"Eastbound regression failed for {strategy} offsets.")


def plot_bidirectional(evaluation: pd.DataFrame, figure_dir: Path) -> None:
    """Plot original and optimised progression in both directions."""

    figure_dir.mkdir(parents=True, exist_ok=True)
    colours = {"EB": "tab:blue", "WB": "tab:orange"}
    for period in PERIOD_ORDER:
        group = evaluation.loc[evaluation["period"] == period]
        fig, ax = plt.subplots(figsize=(9, 5))
        for direction in DIRECTIONS:
            for strategy in PHASE3E_STRATEGIES:
                selected = group.loc[
                    (group["direction"] == direction)
                    & (group["strategy"] == strategy)
                ].sort_values("speed_kmh")
                ax.plot(
                    selected["speed_kmh"],
                    selected["progression_fraction"] * 100.0,
                    color=colours[direction],
                    marker="o" if strategy == "Original" else "s",
                    linestyle="-" if strategy == "Original" else "--",
                    label=f"{direction}: {strategy}",
                )
        ax.set_title(f"Synthetic bidirectional progression: {period}")
        ax.set_xlabel("Travel speed (km/h)")
        ax.set_ylabel("Idealised all-green departures (%)")
        ax.set_xticks(SPEEDS)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend(ncol=2)
        fig.tight_layout()
        filename = period.lower().replace(" ", "_")
        fig.savefig(figure_dir / f"phase3e_bidirectional_{filename}.png", dpi=180)
        plt.close(fig)


def run_phase3e() -> None:
    original_plans = pd.read_csv(ORIGINAL_PLANS_PATH)
    robust_offsets = pd.read_csv(ROBUST_OFFSETS_PATH)
    previous = pd.read_csv(PREVIOUS_COMPARISON_PATH)
    evaluation = build_bidirectional_evaluation(original_plans, robust_offsets)
    verify_eastbound_regression(evaluation, previous)
    direction_summary, combined_summary = build_summaries(evaluation)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(EVALUATION_PATH, index=False)
    direction_summary.to_csv(DIRECTION_SUMMARY_PATH, index=False)
    combined_summary.to_csv(COMBINED_SUMMARY_PATH, index=False)
    plot_bidirectional(evaluation, FIGURE_DIR)

    print("Eastbound regression checks passed.")
    print("\nPhase 3E direction summary:")
    print(direction_summary.to_string(index=False))
    print("\nPhase 3E combined bidirectional summary:")
    print(combined_summary.to_string(index=False))
    print(f"\nWrote {EVALUATION_PATH}")
    print(f"Wrote {DIRECTION_SUMMARY_PATH}")
    print(f"Wrote {COMBINED_SUMMARY_PATH}")
    print(f"Wrote four figures under {FIGURE_DIR}")

# ============================================================================
# Phase 3F demand-weighted offsets (python -m workflows phase3f)
# Formerly phase3_bidirectional_optimisation.py. Run Phase 3F demand-weighted bidirectional offset optimisation.
# ============================================================================

TRAFFIC_PATH = DATA_DIR / "synthetic_traffic_15min.csv"
PHASE3F_PLANS_PATH = DATA_DIR / "phase3a_signal_plans.csv"
PHASE3D_PATH = DATA_DIR / "phase3d_optimised_offsets.csv"
WEIGHTS_PATH = DATA_DIR / "phase3f_direction_weights.csv"
PHASE3F_SUMMARY_PATH = DATA_DIR / "phase3f_optimisation_summary.csv"
DETAIL_PATH = DATA_DIR / "phase3f_speed_direction_comparison.csv"
PHASE3F_FIGURE_PATH = FIGURE_DIR / "phase3f_strategy_comparison.png"
PROTECTION_PP = 5.0
PHASE3F_STRATEGIES = (
    "Phase3A_original",
    "Phase3D_eastbound",
    "Phase3F_bidirectional",
)


def _evaluate_strategy(
    period: str,
    strategy: str,
    cycle: int,
    greens: dict[str, float],
    offsets: dict[str, float],
    weights: dict[str, float],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    counts: dict[str, list[int]] = {direction: [] for direction in PHASE3F_DIRECTIONS}
    details: list[dict[str, object]] = []
    for direction in PHASE3F_DIRECTIONS:
        for speed_kmh in PHASE3F_SPEEDS:
            count = count_directional_green(
                cycle, greens, offsets, speed_kmh, direction
            )
            counts[direction].append(count)
            fraction = count / cycle
            details.append(
                {
                    "period": period,
                    "strategy": strategy,
                    "direction": direction,
                    "speed_kmh": speed_kmh,
                    "common_cycle_s": cycle,
                    "all_green_departures": count,
                    "progression_fraction": fraction,
                    "direction_weight": weights[direction],
                    "weighted_contribution": weights[direction] * fraction,
                }
            )

    fractions = {
        direction: [count / cycle for count in counts[direction]]
        for direction in PHASE3F_DIRECTIONS
    }
    means = {
        direction: sum(fractions[direction]) / len(PHASE3F_SPEEDS)
        for direction in PHASE3F_DIRECTIONS
    }
    design_index = PHASE3F_SPEEDS.index(40.0)
    design = {
        direction: fractions[direction][design_index]
        for direction in PHASE3F_DIRECTIONS
    }
    summary = {
        "period": period,
        "strategy": strategy,
        "common_cycle_s": cycle,
        "offset_b_s": offsets["B"],
        "offset_c_s": offsets["C"],
        "EB_weight": weights["EB"],
        "WB_weight": weights["WB"],
        "EB_mean": means["EB"],
        "WB_mean": means["WB"],
        "weaker_direction_mean": min(means.values()),
        "weighted_mean": sum(
            weights[direction] * means[direction] for direction in PHASE3F_DIRECTIONS
        ),
        "EB_40": design["EB"],
        "WB_40": design["WB"],
        "weighted_40": sum(
            weights[direction] * design[direction] for direction in PHASE3F_DIRECTIONS
        ),
    }
    return summary, details


def build_phase3f_results(
    plans: pd.DataFrame,
    phase3d: pd.DataFrame,
    weights_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Optimise and compare three offset strategies for each period."""

    if len(plans) != 12 or len(phase3d) != 4 or len(weights_df) != 4:
        raise ValueError("Unexpected Phase 3F input row counts.")
    summary_records: list[dict[str, object]] = []
    detail_records: list[dict[str, object]] = []

    for period in PERIOD_ORDER:
        group = plans.loc[plans["period"] == period].set_index("intersection_id")
        if set(group.index) != {"A", "B", "C"} or len(group) != 3:
            raise ValueError(f"{period} must contain one plan for A, B and C.")
        cycle_values = group["common_cycle_s"].unique()
        if len(cycle_values) != 1:
            raise ValueError(f"{period} must use one common cycle.")
        cycle = int(cycle_values[0])
        greens = group["ew_green_s"].to_dict()
        original = group["offset_s"].to_dict()

        weight_rows = weights_df.loc[weights_df["period"] == period]
        old_rows = phase3d.loc[phase3d["period"] == period]
        if len(weight_rows) != 1 or len(old_rows) != 1:
            raise ValueError(f"Missing Phase 3D offsets or demand weights for {period}.")
        weight_row = weight_rows.iloc[0]
        old = old_rows.iloc[0]
        weights = {
            "EB": float(weight_row["EB_weight"]),
            "WB": float(weight_row["WB_weight"]),
        }
        previous = {
            "A": 0.0,
            "B": float(old["optimal_offset_b_s"]),
            "C": float(old["optimal_offset_c_s"]),
        }
        optimum = optimise_bidirectional(
            cycle,
            greens,
            original,
            weights,
            protection_pp=PROTECTION_PP,
        )
        strategies = {
            "Phase3A_original": original,
            "Phase3D_eastbound": previous,
            "Phase3F_bidirectional": optimum["offsets"],
        }
        for strategy, offsets in strategies.items():
            summary, details = _evaluate_strategy(
                period, strategy, cycle, greens, offsets, weights
            )
            summary_records.append(summary)
            detail_records.extend(details)

    summary = pd.DataFrame(summary_records)
    details = pd.DataFrame(detail_records)
    if len(summary) != 12 or len(details) != 120:
        raise ValueError("Phase 3F outputs have unexpected row counts.")
    for period in PERIOD_ORDER:
        group = summary.loc[summary["period"] == period].set_index("strategy")
        baseline = group.loc["Phase3A_original"]
        optimised = group.loc["Phase3F_bidirectional"]
        if optimised["weighted_mean"] < baseline["weighted_mean"] - 1e-9:
            raise ValueError(f"Weighted mean fell below baseline for {period}.")
        for direction in PHASE3F_DIRECTIONS:
            if optimised[f"{direction}_40"] < baseline[f"{direction}_40"] - 0.05 - 1e-9:
                raise ValueError(f"{direction} design-speed protection failed for {period}.")
    return summary, details


def plot_strategy_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot demand-weighted progression for the three strategies."""

    chart = summary.pivot(
        index="period", columns="strategy", values="weighted_mean"
    ).reindex(PERIOD_ORDER)
    ax = (chart * 100.0).plot(kind="bar", figsize=(10, 5), width=0.75)
    ax.set_title("Synthetic Demand-Weighted Bidirectional Progression")
    ax.set_xlabel("Operating period")
    ax.set_ylabel("Weighted all-green departure fraction (%)")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Strategy", fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def run_phase3f() -> None:
    traffic = pd.read_csv(TRAFFIC_PATH)
    plans = pd.read_csv(PHASE3F_PLANS_PATH)
    phase3d = pd.read_csv(PHASE3D_PATH)
    weights_df = calculate_direction_weights(traffic)
    summary, details = build_phase3f_results(plans, phase3d, weights_df)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    weights_df.to_csv(WEIGHTS_PATH, index=False)
    summary.to_csv(PHASE3F_SUMMARY_PATH, index=False)
    details.to_csv(DETAIL_PATH, index=False)
    plot_strategy_comparison(summary, PHASE3F_FIGURE_PATH)

    print("Directional demand weights:")
    print(weights_df.to_string(index=False))
    print("\nPhase 3F strategy comparison:")
    print(summary.to_string(index=False))
    print("\nPhase 3F completed.")
    print(f"Wrote {WEIGHTS_PATH}")
    print(f"Wrote {PHASE3F_SUMMARY_PATH}")
    print(f"Wrote {DETAIL_PATH}")
    print(f"Wrote {PHASE3F_FIGURE_PATH}")


PHASES = {
    "phase3a": run_phase3a,
    "phase3b": run_phase3b,
    "phase3c": run_phase3c,
    "phase3d": run_phase3d,
    "phase3e": run_phase3e,
    "phase3f": run_phase3f,
}
