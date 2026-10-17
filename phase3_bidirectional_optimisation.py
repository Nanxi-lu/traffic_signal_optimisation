"""Run Phase 3F demand-weighted bidirectional offset optimisation."""

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

from traffic_signals.bidirectional import count_directional_green
from traffic_signals.bidirectional_optimisation import (
    DIRECTIONS,
    SPEEDS,
    optimise_bidirectional,
)
from traffic_signals.demand_weights import calculate_direction_weights
from traffic_signals.segmentation import PERIOD_ORDER


DATA_DIR = Path("outputs/data")
FIGURE_DIR = Path("outputs/figures")
TRAFFIC_PATH = DATA_DIR / "synthetic_traffic_15min.csv"
PLANS_PATH = DATA_DIR / "phase3a_signal_plans.csv"
PHASE3D_PATH = DATA_DIR / "phase3d_optimised_offsets.csv"
WEIGHTS_PATH = DATA_DIR / "phase3f_direction_weights.csv"
SUMMARY_PATH = DATA_DIR / "phase3f_optimisation_summary.csv"
DETAIL_PATH = DATA_DIR / "phase3f_speed_direction_comparison.csv"
FIGURE_PATH = FIGURE_DIR / "phase3f_strategy_comparison.png"
PROTECTION_PP = 5.0
STRATEGIES = (
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
    counts: dict[str, list[int]] = {direction: [] for direction in DIRECTIONS}
    details: list[dict[str, object]] = []
    for direction in DIRECTIONS:
        for speed_kmh in SPEEDS:
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
        for direction in DIRECTIONS
    }
    means = {
        direction: sum(fractions[direction]) / len(SPEEDS)
        for direction in DIRECTIONS
    }
    design_index = SPEEDS.index(40.0)
    design = {
        direction: fractions[direction][design_index]
        for direction in DIRECTIONS
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
            weights[direction] * means[direction] for direction in DIRECTIONS
        ),
        "EB_40": design["EB"],
        "WB_40": design["WB"],
        "weighted_40": sum(
            weights[direction] * design[direction] for direction in DIRECTIONS
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
        for direction in DIRECTIONS:
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


def main() -> None:
    traffic = pd.read_csv(TRAFFIC_PATH)
    plans = pd.read_csv(PLANS_PATH)
    phase3d = pd.read_csv(PHASE3D_PATH)
    weights_df = calculate_direction_weights(traffic)
    summary, details = build_phase3f_results(plans, phase3d, weights_df)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    weights_df.to_csv(WEIGHTS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    details.to_csv(DETAIL_PATH, index=False)
    plot_strategy_comparison(summary, FIGURE_PATH)

    print("Directional demand weights:")
    print(weights_df.to_string(index=False))
    print("\nPhase 3F strategy comparison:")
    print(summary.to_string(index=False))
    print("\nPhase 3F completed.")
    print(f"Wrote {WEIGHTS_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {DETAIL_PATH}")
    print(f"Wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
