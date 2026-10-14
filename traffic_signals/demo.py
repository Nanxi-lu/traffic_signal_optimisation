"""Run the Phase 1 synthetic-data and Webster-cycle demonstration."""

import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "traffic-signals-matplotlib")
)
import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from traffic_signals.synthetic_data import SyntheticDataConfig, generate_synthetic_data
from traffic_signals.webster import evaluate_two_phase_interval


APPROACH_LABELS = {
    "eastbound": "Eastbound",
    "westbound": "Westbound",
    "northbound": "Northbound",
    "southbound": "Southbound",
}


def calculate_cycle_plan(
    traffic_data: pd.DataFrame,
    lost_time_s: float = 12.0,
    min_cycle_s: float = 40.0,
    max_cycle_s: float = 150.0,
) -> pd.DataFrame:
    """Calculate one simplified two-phase plan per time and intersection."""

    records: list[dict[str, object]] = []
    for (timestamp, intersection), group in traffic_data.groupby(
        ["time", "intersection_id"], sort=True
    ):
        analysis = evaluate_two_phase_interval(
            group,
            lost_time_s=lost_time_s,
            min_cycle_s=min_cycle_s,
            max_cycle_s=max_cycle_s,
        )
        records.append(
            {"time": timestamp, "intersection_id": intersection, **analysis}
        )
    return pd.DataFrame.from_records(records)


def plot_demand_profile(traffic_data: pd.DataFrame, output_path: Path) -> None:
    """Plot mean 15-minute approach counts across the three intersections."""

    profile = (
        traffic_data.groupby(["time", "approach"], as_index=False)["volume_15min"]
        .mean()
        .sort_values("time")
    )
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for approach, group in profile.groupby("approach"):
        ax.plot(
            group["time"],
            group["volume_15min"],
            label=APPROACH_LABELS[approach],
            linewidth=1.8,
        )
    ax.set_title("Synthetic traffic demand profile (hypothetical data)")
    ax.set_xlabel("Time of day")
    ax.set_ylabel("Mean demand (vehicles per 15 minutes)")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.autofmt_xdate(rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_cycle_lengths(cycle_plan: pd.DataFrame, output_path: Path) -> None:
    """Plot bounded Webster cycle calculations for each intersection."""

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for intersection, group in cycle_plan.groupby("intersection_id"):
        ax.plot(
            group["time"],
            group["cycle_length_s"],
            label=f"Intersection {intersection}",
            linewidth=1.7,
        )
    inadequate = cycle_plan.loc[~cycle_plan["adequately_served"]]
    if not inadequate.empty:
        ax.scatter(
            inadequate["time"],
            inadequate["cycle_length_s"],
            color="crimson",
            marker="x",
            label="Capacity warning",
            zorder=3,
        )
    ax.set_title("Synthetic Webster cycle lengths (simplified two-phase model)")
    ax.set_xlabel("Time of day")
    ax.set_ylabel("Selected cycle length (seconds)")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.autofmt_xdate(rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lost-time", type=float, default=12.0)
    parser.add_argument("--min-cycle", type=float, default=40.0)
    parser.add_argument("--max-cycle", type=float, default=150.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures_dir = args.output_dir / "figures"
    data_dir = args.output_dir / "data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    traffic_data = generate_synthetic_data(SyntheticDataConfig(seed=args.seed))
    cycle_plan = calculate_cycle_plan(
        traffic_data,
        lost_time_s=args.lost_time,
        min_cycle_s=args.min_cycle,
        max_cycle_s=args.max_cycle,
    )

    traffic_path = data_dir / "synthetic_traffic_15min.csv"
    cycle_path = data_dir / "synthetic_webster_results.csv"
    demand_figure = figures_dir / "synthetic_demand_profile.png"
    cycle_figure = figures_dir / "synthetic_webster_cycles.png"

    traffic_data.to_csv(traffic_path, index=False)
    cycle_plan.to_csv(cycle_path, index=False)
    plot_demand_profile(traffic_data, demand_figure)
    plot_cycle_lengths(cycle_plan, cycle_figure)

    warning_count = int((~cycle_plan["adequately_served"]).sum())
    print("Phase 1 complete: all data and results are synthetic.")
    print(f"Traffic rows: {len(traffic_data)}")
    print(f"Cycle-plan rows: {len(cycle_plan)}")
    print(f"Capacity-warning intervals: {warning_count}")
    print(f"Wrote {traffic_path}")
    print(f"Wrote {cycle_path}")
    print(f"Wrote {demand_figure}")
    print(f"Wrote {cycle_figure}")


if __name__ == "__main__":
    main()
