"""Run Phase 3D robust green-wave offset optimisation."""

import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "traffic-signals-matplotlib")
)
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from traffic_signals.offset_optimisation import (
    SPEEDS,
    circular_distance,
    count_all_green,
    optimise_offsets,
)
from traffic_signals.segmentation import PERIOD_ORDER


DATA_DIR = Path("outputs/data")
FIGURE_DIR = Path("outputs/figures")
PLANS_PATH = DATA_DIR / "phase3a_signal_plans.csv"
SUMMARY_PATH = DATA_DIR / "phase3d_optimised_offsets.csv"
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


def main() -> None:
    plans = pd.read_csv(PLANS_PATH)
    summary, comparison = build_offset_optimisation(plans)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    comparison.to_csv(COMPARISON_PATH, index=False)
    plot_robustness(comparison, FIGURE_DIR)

    print("Phase 3D offset optimisation:")
    print(summary.to_string(index=False))
    print("\nSpeed sensitivity comparison:")
    print(comparison.to_string(index=False))
    print(f"\nWrote {SUMMARY_PATH}")
    print(f"Wrote {COMPARISON_PATH}")
    print(f"Wrote four figures under {FIGURE_DIR}")


if __name__ == "__main__":
    main()
