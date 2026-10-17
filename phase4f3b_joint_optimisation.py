"""Joint Junction C green-split and offset optimisation for Phase 4F-3B."""

import os
from collections.abc import Iterable, Mapping
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

from phase4b_multi_seed import bootstrap_mean_ci
from phase4f2_multi_seed_full_junction import load_inputs, make_arrivals
from traffic_signals.full_junction import create_two_phase_plan
from traffic_signals.network_evaluation import build_period_plans, evaluate_network


DATA_DIR = Path("outputs/data")
FIGURE_DIR = Path("outputs/figures")
PERIODS = ("AM peak", "PM peak")
GREEN_TRANSFERS = (0, 3, 5, 7, 10)
OFFSET_SHIFTS = (-12, -8, -4, 0, 4, 8, 12)
SCREEN_SEEDS = range(400, 404)
TRAIN_SEEDS = range(400, 412)
HOLDOUT_SEEDS = range(420, 450)
PROTECTED_MOVEMENTS = ("EB", "WB", "C_N", "C_S")
UNCHANGED_MOVEMENTS = ("A_N", "A_S", "B_N", "B_S")
ALL_MOVEMENTS = ("EB", "WB", "A_N", "A_S", "B_N", "B_S", "C_N", "C_S")
FAIRNESS_LIMIT_S = 2.0
HORIZON_S = 3600.0

SCREENING_PATH = DATA_DIR / "phase4f3b_screening.csv"
TRAINING_CANDIDATES_PATH = DATA_DIR / "phase4f3b_training_candidates.csv"
TRAINING_FAIRNESS_PATH = DATA_DIR / "phase4f3b_training_fairness.csv"
SELECTED_PLANS_PATH = DATA_DIR / "phase4f3b_selected_plans.csv"
HOLDOUT_MOVEMENTS_PATH = DATA_DIR / "phase4f3b_holdout_movements.csv"
HOLDOUT_NETWORK_PATH = DATA_DIR / "phase4f3b_holdout_network.csv"
HOLDOUT_NETWORK_SUMMARY_PATH = DATA_DIR / "phase4f3b_holdout_network_summary.csv"
HOLDOUT_MOVEMENT_SUMMARY_PATH = DATA_DIR / "phase4f3b_holdout_movement_summary.csv"


def make_candidate_plan(
    base_plans: Mapping[str, Mapping[str, float]],
    extra_ns: int,
    c_offset: float,
) -> dict[str, dict[str, float]]:
    """Change only Junction C's green split and EW offset."""

    if set(base_plans) != {"A", "B", "C"}:
        raise ValueError("Expected junctions A, B and C.")
    if extra_ns not in GREEN_TRANSFERS:
        raise ValueError("Unsupported NS green transfer.")
    if not np.isfinite(c_offset):
        raise ValueError("Junction C offset must be finite.")

    plans = {junction: dict(plan) for junction, plan in base_plans.items()}
    original = plans["C"]
    ew_green = float(original["ew_green_s"]) - extra_ns
    ns_green = float(original["ns_green_s"]) + extra_ns
    if ew_green < 10 or ns_green < 10:
        raise ValueError("Minimum effective green violated.")
    plans["C"] = create_two_phase_plan(
        cycle_s=float(original["cycle_s"]),
        ew_green_s=ew_green,
        ns_green_s=ns_green,
        ew_offset_s=float(c_offset),
        lost_time_s=float(original["lost_time_s"]),
        interphase_loss_s=float(original["interphase_loss_s"]),
    )
    return plans


def make_candidate_pairs(
    base_c_offset: int,
    cycle: int = 150,
) -> list[tuple[int, int]]:
    """Return the 35 green-transfer and local offset combinations."""

    if not isinstance(base_c_offset, (int, np.integer)):
        raise ValueError("Base offset must be an integer.")
    if not isinstance(cycle, (int, np.integer)) or cycle <= 0:
        raise ValueError("Cycle must be a positive integer.")
    pairs = [
        (extra, int((base_c_offset + shift) % cycle))
        for extra in GREEN_TRANSFERS
        for shift in OFFSET_SHIFTS
    ]
    if len(pairs) != 35 or len(set(pairs)) != 35:
        raise ValueError("Expected 35 unique candidate pairs.")
    return pairs


def prepare_arrivals(
    traffic: pd.DataFrame,
    period: str,
    seeds: Iterable[int],
) -> dict[int, dict[tuple[str, str], np.ndarray]]:
    """Generate each matched eight-stream arrival set once."""

    return {int(seed): make_arrivals(traffic, period, int(seed)) for seed in seeds}


def evaluate_plan(
    arrival_cache: Mapping[int, Mapping[tuple[str, str], np.ndarray]],
    base_plans: Mapping[str, Mapping[str, float]],
    period: str,
    pair: tuple[int, int],
    name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one candidate against every cached arrival replication."""

    extra, c_offset = pair
    plans = make_candidate_plan(base_plans, extra, c_offset)
    movement_frames: list[pd.DataFrame] = []
    network_records: list[dict[str, object]] = []
    for seed, arrivals in arrival_cache.items():
        movements, network = evaluate_network(
            arrivals=arrivals,
            junction_plans=plans,
            strategy=name,
            period=period,
            seed=seed,
            horizon_s=HORIZON_S,
        )
        movements["extra_ns_green_s"] = extra
        movements["C_offset_s"] = c_offset
        network["extra_ns_green_s"] = extra
        network["C_offset_s"] = c_offset
        network["C_ns_queue_at_hour"] = int(
            movements.loc[
                movements["movement"].isin(["C_N", "C_S"]),
                "queue_at_hour_veh",
            ].sum()
        )
        movement_frames.append(movements)
        network_records.append(network)
    return pd.concat(movement_frames, ignore_index=True), pd.DataFrame(network_records)


def compare_plans(
    base_mov: pd.DataFrame,
    base_net: pd.DataFrame,
    new_mov: pd.DataFrame,
    new_net: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair candidate metrics with a benchmark receiving identical demand."""

    paired_mov = base_mov.merge(
        new_mov,
        on=["period", "seed", "movement"],
        suffixes=("_base", "_new"),
        validate="one_to_one",
    )
    if not paired_mov["vehicles_base"].eq(paired_mov["vehicles_new"]).all():
        raise ValueError("Movement vehicle counts differ between plans.")
    paired_mov["delay_change_s"] = (
        paired_mov["mean_delay_per_vehicle_s_new"]
        - paired_mov["mean_delay_per_vehicle_s_base"]
    )
    paired_mov["queue_change_veh"] = (
        paired_mov["queue_at_hour_veh_new"]
        - paired_mov["queue_at_hour_veh_base"]
    )

    paired_net = base_net.merge(
        new_net,
        on=["period", "seed"],
        suffixes=("_base", "_new"),
        validate="one_to_one",
    )
    for field in ("corridor_vehicles", "ns_vehicles", "total_passages"):
        if not paired_net[f"{field}_base"].eq(paired_net[f"{field}_new"]).all():
            raise ValueError(f"Network {field} differs between plans.")
    paired_net["delay_change_s"] = (
        paired_net["mean_delay_per_passage_s_new"]
        - paired_net["mean_delay_per_passage_s_base"]
    )
    paired_net["queue_change_veh"] = (
        paired_net["total_queue_at_hour_new"]
        - paired_net["total_queue_at_hour_base"]
    )
    paired_net["C_ns_queue_change_veh"] = (
        paired_net["C_ns_queue_at_hour_new"]
        - paired_net["C_ns_queue_at_hour_base"]
    )
    return paired_mov, paired_net


def evaluate_fairness(paired_mov: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the four movement-level training fairness constraints."""

    records: list[dict[str, object]] = []
    for movement in PROTECTED_MOVEMENTS:
        selected = paired_mov.loc[paired_mov["movement"] == movement]
        changes = selected["delay_change_s"].to_numpy(dtype=float)
        if len(changes) < 2:
            raise ValueError(f"Insufficient fairness samples for {movement}.")
        low, high = bootstrap_mean_ci(
            changes,
            n_bootstrap=4000,
            seed=2026,
        )
        records.append(
            {
                "movement": movement,
                "replications": len(changes),
                "mean_delay_change_s": changes.mean(),
                "ci_lower_s": low,
                "ci_upper_s": high,
                "fairness_pass": bool(high <= FAIRNESS_LIMIT_S),
            }
        )
    return pd.DataFrame(records)


def _baseline_pair(base_plans: Mapping[str, Mapping[str, float]]) -> tuple[int, int]:
    return 0, int(base_plans["C"]["ew_offset_s"])


def screen_candidates(
    cache: Mapping[int, Mapping[tuple[str, str], np.ndarray]],
    base_plans: Mapping[str, Mapping[str, float]],
    period: str,
) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """Rank all 35 pairs on four seeds and retain a diverse shortlist."""

    baseline_pair = _baseline_pair(base_plans)
    baseline_mov, baseline_net = evaluate_plan(
        cache, base_plans, period, baseline_pair, "Phase4E"
    )
    records: list[dict[str, object]] = []
    for pair in make_candidate_pairs(
        baseline_pair[1], int(base_plans["C"]["cycle_s"])
    ):
        candidate_mov, candidate_net = evaluate_plan(
            cache, base_plans, period, pair, "Candidate"
        )
        _, paired_net = compare_plans(
            baseline_mov, baseline_net, candidate_mov, candidate_net
        )
        records.append(
            {
                "period": period,
                "extra_ns_green_s": pair[0],
                "C_offset_s": pair[1],
                "replications": len(paired_net),
                "screen_delay_change_s": paired_net["delay_change_s"].mean(),
                "screen_C_ns_queue_change_veh": paired_net[
                    "C_ns_queue_change_veh"
                ].mean(),
            }
        )

    ranking = pd.DataFrame(records).sort_values(
        ["screen_delay_change_s", "screen_C_ns_queue_change_veh", "C_offset_s"],
        kind="stable",
    )
    if len(ranking) != 35:
        raise ValueError("Expected 35 screening rows per period.")
    chosen = {
        (int(row.extra_ns_green_s), int(row.C_offset_s))
        for row in ranking.head(12).itertuples()
    }
    for _, group in ranking.groupby("extra_ns_green_s", sort=False):
        chosen.update(
            (int(row.extra_ns_green_s), int(row.C_offset_s))
            for row in group.head(2).itertuples()
        )
    chosen.add(baseline_pair)
    shortlist = sorted(chosen)
    return ranking.reset_index(drop=True), shortlist


def _assert_unchanged_approaches(paired_mov: pd.DataFrame) -> None:
    unchanged = paired_mov.loc[
        paired_mov["movement"].isin(UNCHANGED_MOVEMENTS)
    ]
    if not np.allclose(unchanged["delay_change_s"], 0.0, atol=1e-12):
        raise AssertionError("A/B local approach delay changed unexpectedly.")
    if not unchanged["queue_change_veh"].eq(0).all():
        raise AssertionError("A/B local approach queues changed unexpectedly.")


def optimise_training(
    cache: Mapping[int, Mapping[tuple[str, str], np.ndarray]],
    base_plans: Mapping[str, Mapping[str, float]],
    period: str,
    shortlist: Iterable[tuple[int, int]],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Apply network and movement fairness criteria on all training seeds."""

    baseline_pair = _baseline_pair(base_plans)
    baseline_mov, baseline_net = evaluate_plan(
        cache, base_plans, period, baseline_pair, "Phase4E"
    )
    original_offset = baseline_pair[1]
    cycle = int(base_plans["C"]["cycle_s"])
    candidate_records: list[dict[str, object]] = []
    fairness_frames: list[pd.DataFrame] = []

    for extra, offset in shortlist:
        candidate_mov, candidate_net = evaluate_plan(
            cache, base_plans, period, (extra, offset), "Candidate"
        )
        paired_mov, paired_net = compare_plans(
            baseline_mov, baseline_net, candidate_mov, candidate_net
        )
        _assert_unchanged_approaches(paired_mov)
        fairness = evaluate_fairness(paired_mov)
        fairness.insert(0, "period", period)
        fairness.insert(1, "extra_ns_green_s", extra)
        fairness.insert(2, "C_offset_s", offset)
        fairness_frames.append(fairness)

        delay_changes = paired_net["delay_change_s"].to_numpy(dtype=float)
        delay_low, delay_high = bootstrap_mean_ci(delay_changes)
        mean_delay_change = float(delay_changes.mean())
        fairness_pass = bool(fairness["fairness_pass"].all())
        network_pass = bool(mean_delay_change <= 1e-12)
        candidate_records.append(
            {
                "period": period,
                "extra_ns_green_s": extra,
                "C_offset_s": offset,
                "replications": len(paired_net),
                "mean_network_delay_change_s": mean_delay_change,
                "network_delay_ci_lower_s": delay_low,
                "network_delay_ci_upper_s": delay_high,
                "mean_C_ns_queue_change_veh": paired_net[
                    "C_ns_queue_change_veh"
                ].mean(),
                "network_performance_pass": network_pass,
                "movement_fairness_pass": fairness_pass,
                "feasible": bool(network_pass and fairness_pass),
                "offset_distance_s": min(
                    (offset - original_offset) % cycle,
                    (original_offset - offset) % cycle,
                ),
            }
        )

    candidates = pd.DataFrame(candidate_records)
    fairness_checks = pd.concat(fairness_frames, ignore_index=True)
    feasible = candidates.loc[candidates["feasible"]].sort_values(
        [
            "mean_network_delay_change_s",
            "mean_C_ns_queue_change_veh",
            "offset_distance_s",
        ],
        kind="stable",
    )
    if feasible.empty:
        raise AssertionError("Phase 4E fallback should always be feasible.")
    winner = feasible.iloc[0]
    winning_plan = make_candidate_plan(
        base_plans,
        int(winner["extra_ns_green_s"]),
        float(winner["C_offset_s"]),
    )["C"]
    selected: dict[str, object] = {
        "period": period,
        "extra_ns_green_s": int(winner["extra_ns_green_s"]),
        "C_ew_offset_s": int(winner["C_offset_s"]),
        "C_ns_offset_s": winning_plan["ns_offset_s"],
        "C_ew_green_s": winning_plan["ew_green_s"],
        "C_ns_green_s": winning_plan["ns_green_s"],
        "training_mean_network_delay_change_s": winner[
            "mean_network_delay_change_s"
        ],
        "training_mean_C_ns_queue_change_veh": winner[
            "mean_C_ns_queue_change_veh"
        ],
        "training_network_performance_pass": bool(
            winner["network_performance_pass"]
        ),
        "training_movement_fairness_pass": bool(winner["movement_fairness_pass"]),
        "training_candidate_count": len(candidates),
    }
    return selected, candidates, fairness_checks


def _assert_regression_row(
    current: pd.Series,
    previous: pd.Series,
    label: str,
) -> None:
    fields = (
        "corridor_vehicles",
        "ns_vehicles",
        "total_passages",
        "corridor_mean_delay_s",
        "ns_mean_delay_s",
        "mean_delay_per_passage_s",
        "total_delay_vehicle_seconds",
        "corridor_queue_at_hour",
        "ns_queue_at_hour",
        "total_queue_at_hour",
    )
    for field in fields:
        if not np.isclose(float(current[field]), float(previous[field]), atol=1e-9):
            raise AssertionError(f"Regression mismatch for {label}/{field}.")


def verify_previous_results(
    traffic: pd.DataFrame,
    plans_3a: pd.DataFrame,
    plans_4e: pd.DataFrame,
) -> None:
    """Reproduce one Phase 4F-2 and one Phase 4F-3A row per period."""

    previous_f2 = pd.read_csv(DATA_DIR / "phase4f2_network_metrics.csv")
    previous_f3a = pd.read_csv(DATA_DIR / "phase4f3a_network_metrics.csv")
    for period in PERIODS:
        base_plans = build_period_plans(plans_3a, plans_4e, period)["Phase4E"]
        offset = int(base_plans["C"]["ew_offset_s"])
        checks = (
            (200, (0, offset), previous_f2, None, "Phase4E"),
            (300, (5, offset), previous_f3a, 5, None),
        )
        for seed, pair, previous, extra_filter, strategy_filter in checks:
            cache = prepare_arrivals(traffic, period, [seed])
            _, current_frame = evaluate_plan(
                cache, base_plans, period, pair, "Regression"
            )
            selected = previous.loc[
                (previous["period"] == period) & (previous["seed"] == seed)
            ]
            if extra_filter is not None:
                selected = selected.loc[
                    selected["extra_ns_green_s"] == extra_filter
                ]
            if strategy_filter is not None:
                selected = selected.loc[selected["strategy"] == strategy_filter]
            if len(selected) != 1:
                raise ValueError(f"Missing prior regression row for {period}/{seed}.")
            _assert_regression_row(
                current_frame.iloc[0],
                selected.iloc[0],
                f"{period}/{seed}/{pair}",
            )
    print("Phase 4F-2 and Phase 4F-3A regressions passed.")


def run_training() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Screen candidates and select one fairness-feasible plan per period."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    traffic, plans_3a, plans_4e = load_inputs()
    verify_previous_results(traffic, plans_3a, plans_4e)
    selected_records: list[dict[str, object]] = []
    screening_frames: list[pd.DataFrame] = []
    training_frames: list[pd.DataFrame] = []
    fairness_frames: list[pd.DataFrame] = []

    for period in PERIODS:
        base_plans = build_period_plans(plans_3a, plans_4e, period)["Phase4E"]
        training_cache = prepare_arrivals(traffic, period, TRAIN_SEEDS)
        screening_cache = {seed: training_cache[seed] for seed in SCREEN_SEEDS}
        ranking, shortlist = screen_candidates(screening_cache, base_plans, period)
        selected, candidates, checks = optimise_training(
            training_cache, base_plans, period, shortlist
        )
        selected_records.append(selected)
        screening_frames.append(ranking)
        training_frames.append(candidates)
        fairness_frames.append(checks)
        print(
            f"Completed {period} training: {len(shortlist)} shortlisted candidates"
        )

    selected_plans = pd.DataFrame(selected_records)
    screening = pd.concat(screening_frames, ignore_index=True)
    training = pd.concat(training_frames, ignore_index=True)
    fairness = pd.concat(fairness_frames, ignore_index=True)
    if len(screening) != 70:
        raise ValueError("Expected 70 screening rows.")
    if len(selected_plans) != 2:
        raise ValueError("Expected two selected plans.")
    if len(fairness) != 4 * len(training):
        raise ValueError("Expected four fairness checks per training candidate.")

    screening.to_csv(SCREENING_PATH, index=False)
    training.to_csv(TRAINING_CANDIDATES_PATH, index=False)
    fairness.to_csv(TRAINING_FAIRNESS_PATH, index=False)
    selected_plans.to_csv(SELECTED_PLANS_PATH, index=False)
    return traffic, plans_3a, plans_4e, selected_plans


def run_holdout(
    traffic: pd.DataFrame,
    plans_3a: pd.DataFrame,
    plans_4e: pd.DataFrame,
    selected_plans: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate frozen Phase 4E, split-only and joint plans on holdout seeds."""

    movement_frames: list[pd.DataFrame] = []
    network_frames: list[pd.DataFrame] = []
    for period in PERIODS:
        base_plans = build_period_plans(plans_3a, plans_4e, period)["Phase4E"]
        original_offset = int(base_plans["C"]["ew_offset_s"])
        selected = selected_plans.loc[selected_plans["period"] == period]
        if len(selected) != 1:
            raise ValueError(f"Expected one selected plan for {period}.")
        row = selected.iloc[0]
        extra = int(row["extra_ns_green_s"])
        new_offset = int(row["C_ew_offset_s"])
        holdout_cache = prepare_arrivals(traffic, period, HOLDOUT_SEEDS)
        strategies = {
            "Phase4E": (0, original_offset),
            "SplitOnly": (extra, original_offset),
            "Joint": (extra, new_offset),
        }
        for name, pair in strategies.items():
            movements, network = evaluate_plan(
                holdout_cache, base_plans, period, pair, name
            )
            movement_frames.append(movements)
            network_frames.append(network)
        print(f"Completed {period} holdout")

    movements = pd.concat(movement_frames, ignore_index=True)
    network = pd.concat(network_frames, ignore_index=True)
    if len(movements) != 1440:
        raise ValueError("Expected 1,440 holdout movement rows.")
    if len(network) != 180:
        raise ValueError("Expected 180 holdout network rows.")
    movements.to_csv(HOLDOUT_MOVEMENTS_PATH, index=False)
    network.to_csv(HOLDOUT_NETWORK_PATH, index=False)
    return movements, network


def summarise_holdout(
    movements: pd.DataFrame,
    network: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the joint plan with Phase 4E and split-only benchmarks."""

    network_records: list[dict[str, object]] = []
    movement_records: list[dict[str, object]] = []
    for period in PERIODS:
        period_mov = movements.loc[movements["period"] == period]
        period_net = network.loc[network["period"] == period]
        joint_mov = period_mov.loc[period_mov["strategy"] == "Joint"]
        joint_net = period_net.loc[period_net["strategy"] == "Joint"]
        for benchmark in ("Phase4E", "SplitOnly"):
            base_mov = period_mov.loc[period_mov["strategy"] == benchmark]
            base_net = period_net.loc[period_net["strategy"] == benchmark]
            paired_mov, paired_net = compare_plans(
                base_mov, base_net, joint_mov, joint_net
            )
            _assert_unchanged_approaches(paired_mov)

            delay_changes = paired_net["delay_change_s"].to_numpy(dtype=float)
            delay_low, delay_high = bootstrap_mean_ci(delay_changes)
            c_queue_changes = paired_net["C_ns_queue_change_veh"].to_numpy(
                dtype=float
            )
            queue_low, queue_high = bootstrap_mean_ci(c_queue_changes)
            network_records.append(
                {
                    "period": period,
                    "benchmark": benchmark,
                    "replications": len(paired_net),
                    "benchmark_mean_delay_per_passage_s": paired_net[
                        "mean_delay_per_passage_s_base"
                    ].mean(),
                    "joint_mean_delay_per_passage_s": paired_net[
                        "mean_delay_per_passage_s_new"
                    ].mean(),
                    "mean_delay_change_s": delay_changes.mean(),
                    "ci_lower_s": delay_low,
                    "ci_upper_s": delay_high,
                    "seeds_better": int((delay_changes < 0).sum()),
                    "benchmark_C_ns_queue_at_hour": paired_net[
                        "C_ns_queue_at_hour_base"
                    ].mean(),
                    "joint_C_ns_queue_at_hour": paired_net[
                        "C_ns_queue_at_hour_new"
                    ].mean(),
                    "mean_C_ns_queue_change_veh": c_queue_changes.mean(),
                    "C_ns_queue_ci_lower_veh": queue_low,
                    "C_ns_queue_ci_upper_veh": queue_high,
                }
            )

            for movement in ALL_MOVEMENTS:
                group = paired_mov.loc[paired_mov["movement"] == movement]
                changes = group["delay_change_s"].to_numpy(dtype=float)
                low, high = bootstrap_mean_ci(changes)
                queue_changes = group["queue_change_veh"].to_numpy(dtype=float)
                movement_records.append(
                    {
                        "period": period,
                        "benchmark": benchmark,
                        "movement": movement,
                        "protected_movement": movement in PROTECTED_MOVEMENTS,
                        "replications": len(group),
                        "benchmark_mean_delay_s": group[
                            "mean_delay_per_vehicle_s_base"
                        ].mean(),
                        "joint_mean_delay_s": group[
                            "mean_delay_per_vehicle_s_new"
                        ].mean(),
                        "mean_delay_change_s": changes.mean(),
                        "ci_lower_s": low,
                        "ci_upper_s": high,
                        "passes_2s_upper": bool(high <= FAIRNESS_LIMIT_S),
                        "benchmark_queue_at_hour": group[
                            "queue_at_hour_veh_base"
                        ].mean(),
                        "joint_queue_at_hour": group[
                            "queue_at_hour_veh_new"
                        ].mean(),
                        "mean_queue_change_veh": queue_changes.mean(),
                    }
                )

    network_summary = pd.DataFrame(network_records)
    movement_summary = pd.DataFrame(movement_records)
    if len(network_summary) != 4:
        raise ValueError("Expected four holdout network summaries.")
    if len(movement_summary) != 32:
        raise ValueError("Expected 32 holdout movement summaries.")
    network_summary.to_csv(HOLDOUT_NETWORK_SUMMARY_PATH, index=False)
    movement_summary.to_csv(HOLDOUT_MOVEMENT_SUMMARY_PATH, index=False)
    return network_summary, movement_summary


def plot_holdout(network: pd.DataFrame) -> None:
    """Plot synthetic holdout network delay under all three strategies."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    order = ["Phase4E", "SplitOnly", "Joint"]
    for period in PERIODS:
        selected = network.loc[network["period"] == period]
        means = (
            selected.groupby("strategy")["mean_delay_per_passage_s"]
            .mean()
            .reindex(order)
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(means.index, means.values)
        ax.set_title(f"Synthetic Junction C Joint Optimisation: {period}")
        ax.set_ylabel("Mean network delay per passage (s)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        slug = period.lower().replace(" ", "_")
        fig.savefig(
            FIGURE_DIR / f"phase4f3b_holdout_{slug}.png",
            dpi=180,
        )
        plt.close(fig)


def main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run training, frozen holdout validation, summaries and figures."""

    traffic, plans_3a, plans_4e, selected_plans = run_training()
    movements, network = run_holdout(
        traffic, plans_3a, plans_4e, selected_plans
    )
    network_summary, movement_summary = summarise_holdout(movements, network)
    plot_holdout(network)
    print("\nSelected Junction C plans:")
    print(selected_plans.to_string(index=False))
    print("\nHoldout network comparison:")
    print(network_summary.to_string(index=False))
    print("\nProtected movement comparison versus Phase 4E:")
    print(
        movement_summary.loc[
            (movement_summary["benchmark"] == "Phase4E")
            & movement_summary["movement"].isin(PROTECTED_MOVEMENTS)
        ].to_string(index=False)
    )
    print("\nPhase 4F-3B completed.")
    return selected_plans, network_summary, movement_summary


if __name__ == "__main__":
    main()
