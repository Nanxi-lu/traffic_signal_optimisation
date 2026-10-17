"""Evaluate idealised eastbound progression at five vehicle speeds."""

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

from traffic_signals.progression import DESIGN_SPEED_KMH, summarise_progression
from traffic_signals.segmentation import PERIOD_ORDER


PLANS_PATH = Path("outputs/data/phase3a_signal_plans.csv")
RESULTS_PATH = Path("outputs/data/phase3c_speed_sensitivity.csv")
FIGURE_PATH = Path("outputs/figures/phase3c_speed_sensitivity.png")
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


def main() -> None:
    plans = pd.read_csv(PLANS_PATH)
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
    plot_speed_sensitivity(results, FIGURE_PATH)
    print("Phase 3C speed sensitivity:")
    print(results.to_string(index=False))
    print(f"\nWrote {RESULTS_PATH}")
    print(f"Wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
