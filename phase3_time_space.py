"""Generate and numerically verify Phase 3B eastbound time-space diagrams."""

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

from traffic_signals.progression import (
    CORRIDOR_DISTANCES_M,
    DESIGN_SPEED_KMH,
    calculate_travel_times,
    summarise_progression,
)
from traffic_signals.segmentation import PERIOD_ORDER


PLANS_PATH = Path("outputs/data/phase3a_signal_plans.csv")
BANDWIDTH_PATH = Path("outputs/data/phase3a_bandwidth.csv")
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


def main() -> None:
    plans = pd.read_csv(PLANS_PATH)
    phase3a_bandwidth = pd.read_csv(BANDWIDTH_PATH)
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


if __name__ == "__main__":
    main()
