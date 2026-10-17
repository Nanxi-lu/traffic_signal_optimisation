"""Run Phase 3E bidirectional progression evaluation."""

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

from traffic_signals.bidirectional import count_directional_green
from traffic_signals.offset_optimisation import SPEEDS
from traffic_signals.segmentation import PERIOD_ORDER


DIRECTIONS = ("EB", "WB")
STRATEGIES = ("Original", "Optimised")
DATA_DIR = Path("outputs/data")
FIGURE_DIR = Path("outputs/figures")
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
            for strategy in STRATEGIES:
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


def main() -> None:
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


if __name__ == "__main__":
    main()
