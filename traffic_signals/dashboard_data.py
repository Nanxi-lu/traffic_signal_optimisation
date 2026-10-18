"""Testable data preparation for the read-only Phase 4G results dashboard.

Every function in this module reads or summarises saved Phase 4G records. None
of them runs the queue model, generates arrivals or searches for signal plans.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


PERIODS = ("Overnight", "AM peak", "Interpeak", "PM peak")
STRATEGIES = ("Phase3A", "Phase4E", "Integrated")
STRATEGY_LABELS = {
    "Phase3A": "Phase 3A",
    "Phase4E": "Phase 4E",
    "Integrated": "Integrated",
}
STRATEGY_DESCRIPTIONS = {
    "Phase3A": (
        "Original coordinated baseline: common Webster-based cycle, fixed "
        "green splits and design-speed eastbound offsets."
    ),
    "Phase4E": (
        "Fairness-protected offsets selected separately for each period; "
        "cycles and green splits unchanged from Phase 3A."
    ),
    "Integrated": (
        "Phase 4E plans, with only the PM peak Junction C green split and "
        "offsets replaced by the Phase 4F-3B joint selection."
    ),
}
BENCHMARKS = ("Phase3A", "Phase4E")
CANDIDATES = ("Phase4E", "Integrated")
MOVEMENTS = ("EB", "WB", "A_N", "A_S", "B_N", "B_S", "C_N", "C_S")
MOVEMENT_LABELS = {
    "EB": "EB",
    "WB": "WB",
    "A_N": "A-N",
    "A_S": "A-S",
    "B_N": "B-N",
    "B_S": "B-S",
    "C_N": "C-N",
    "C_S": "C-S",
}
PROTECTED_MOVEMENTS = ("EB", "WB", "C_N", "C_S")
FAIRNESS_LIMIT_S = 2.0
JUNCTIONS = ("A", "B", "C")
JUNCTION_POSITIONS_M = {"A": 0, "B": 400, "C": 900}
SEEDS = tuple(range(500, 530))

FILES = {
    "network": "phase4g_network_metrics.csv",
    "movements": "phase4g_movement_metrics.csv",
    "plans": "phase4g_signal_plans.csv",
    "network_summary": "phase4g_network_summary.csv",
    "movement_summary": "phase4g_movement_summary.csv",
}

REQUIRED_COLUMNS = {
    "network": (
        "period",
        "seed",
        "strategy",
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
    ),
    "movements": (
        "period",
        "seed",
        "strategy",
        "movement",
        "category",
        "vehicles",
        "passages",
        "total_delay_s",
        "mean_delay_per_vehicle_s",
        "queue_at_hour_veh",
    ),
    "plans": (
        "period",
        "strategy",
        "junction",
        "cycle_s",
        "ew_green_s",
        "ns_green_s",
        "lost_time_s",
        "interphase_loss_s",
        "ew_offset_s",
        "ns_offset_s",
    ),
    "network_summary": (
        "period",
        "benchmark",
        "replications",
        "mean_delay_change_s",
        "ci_lower_s",
        "ci_upper_s",
        "seeds_better",
        "mean_queue_change_veh",
    ),
    "movement_summary": (
        "period",
        "benchmark",
        "movement",
        "protected_movement",
        "mean_delay_change_s",
        "ci_lower_s",
        "ci_upper_s",
        "passes_2s_upper",
    ),
}

NETWORK_METRICS = {
    "mean_delay_per_passage_s": ("Network delay", "s per junction passage"),
    "corridor_mean_delay_s": ("Corridor (EB + WB) delay", "s per vehicle"),
    "ns_mean_delay_s": ("North-south delay", "s per vehicle"),
    "total_queue_at_hour": ("Residual network queue", "vehicles"),
}

_ABS_TOL = 1e-6


class DashboardDataError(ValueError):
    """Raised when saved dashboard evidence is missing or inconsistent."""


def _check_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}.")


def _check_unique(frame: pd.DataFrame, keys: Sequence[str], name: str) -> None:
    duplicated = frame.duplicated(list(keys), keep=False)
    if duplicated.any():
        example = frame.loc[duplicated, list(keys)].iloc[0].to_dict()
        raise DashboardDataError(
            f"Duplicate rows in {name} for key {example}."
        )


def _check_complete(
    frame: pd.DataFrame,
    keys: Sequence[str],
    levels: Sequence[Sequence[object]],
    name: str,
) -> None:
    expected = pd.MultiIndex.from_product(levels, names=list(keys))
    actual = pd.MultiIndex.from_frame(frame[list(keys)])
    missing = expected.difference(actual)
    extra = actual.difference(expected)
    if len(missing):
        raise DashboardDataError(
            f"{name} is missing {len(missing)} expected rows, "
            f"for example {dict(zip(keys, missing[0]))}."
        )
    if len(extra):
        raise DashboardDataError(
            f"{name} contains {len(extra)} unexpected rows, "
            f"for example {dict(zip(keys, extra[0]))}."
        )


def validate_signal_plans(plans: pd.DataFrame) -> None:
    """Check saved plans against the two-phase timing convention.

    The model runs EW effective green from ``ew_offset_s``, then one
    interphase loss, then NS effective green, then the second interphase loss.
    Every window repeats with the common cycle.
    """

    _check_columns(plans, REQUIRED_COLUMNS["plans"])
    _check_unique(plans, ["period", "strategy", "junction"], "signal plans")
    _check_complete(
        plans,
        ["period", "strategy", "junction"],
        [PERIODS, STRATEGIES, JUNCTIONS],
        "Signal plans",
    )
    numeric = plans[list(REQUIRED_COLUMNS["plans"][3:])].astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise DashboardDataError("Signal plans contain non-finite timings.")
    total = numeric["ew_green_s"] + numeric["ns_green_s"] + numeric["lost_time_s"]
    if not np.allclose(total, numeric["cycle_s"]):
        raise DashboardDataError(
            "Effective greens plus lost time do not equal the cycle."
        )
    if not np.allclose(2 * numeric["interphase_loss_s"], numeric["lost_time_s"]):
        raise DashboardDataError("Interphase losses do not sum to lost time.")
    expected_ns = (
        numeric["ew_offset_s"] + numeric["ew_green_s"] + numeric["interphase_loss_s"]
    ) % numeric["cycle_s"]
    if not np.allclose(expected_ns, numeric["ns_offset_s"] % numeric["cycle_s"]):
        raise DashboardDataError(
            "NS offsets do not follow EW green plus one interphase loss."
        )
    cycles = plans.groupby(["period", "strategy"])["cycle_s"].nunique()
    if (cycles != 1).any():
        raise DashboardDataError("A saved plan does not use one common cycle.")


def reconcile_counts(network: pd.DataFrame, movements: pd.DataFrame) -> None:
    """Check that movement records add up to every network record."""

    keys = ["period", "seed", "strategy"]
    corridor = movements["category"].eq("Corridor")
    side = movements["category"].eq("SideRoad")
    if not (corridor | side).all():
        raise DashboardDataError("Unknown movement category in saved records.")
    if not movements.loc[corridor, "movement"].isin(["EB", "WB"]).all():
        raise DashboardDataError("Only EB and WB may be corridor movements.")
    if not movements.loc[corridor, "passages"].eq(
        3 * movements.loc[corridor, "vehicles"]
    ).all():
        raise DashboardDataError("Corridor vehicles must pass three junctions.")
    if not movements.loc[side, "passages"].eq(movements.loc[side, "vehicles"]).all():
        raise DashboardDataError("Side-road vehicles must pass one junction.")

    totals = movements.groupby(keys).agg(
        passages=("passages", "sum"),
        delay=("total_delay_s", "sum"),
        queue=("queue_at_hour_veh", "sum"),
    )
    corridor_vehicles = movements.loc[corridor].groupby(keys)["vehicles"].sum()
    side_vehicles = movements.loc[side].groupby(keys)["vehicles"].sum()
    joined = network.set_index(keys).join(totals, how="outer")
    joined = joined.join(corridor_vehicles.rename("movement_corridor_vehicles"))
    joined = joined.join(side_vehicles.rename("movement_ns_vehicles"))
    if joined[["total_passages", "passages"]].isna().any().any():
        raise DashboardDataError("Network and movement records do not align.")
    checks = {
        "passages": joined["passages"].eq(joined["total_passages"]),
        "corridor vehicles": joined["movement_corridor_vehicles"].eq(
            joined["corridor_vehicles"]
        ),
        "north-south vehicles": joined["movement_ns_vehicles"].eq(
            joined["ns_vehicles"]
        ),
        "residual queues": joined["queue"].eq(joined["total_queue_at_hour"]),
        "total delay": np.isclose(
            joined["delay"], joined["total_delay_vehicle_seconds"], atol=1e-6
        ),
        "delay per passage": np.isclose(
            joined["total_delay_vehicle_seconds"] / joined["total_passages"],
            joined["mean_delay_per_passage_s"],
            atol=1e-9,
        ),
    }
    for label, passed in checks.items():
        if not np.asarray(passed).all():
            raise DashboardDataError(
                f"Movement and network {label} do not reconcile."
            )


def load_phase4g_data(data_dir: Path | str) -> dict[str, pd.DataFrame]:
    """Load and validate the committed Phase 4G evidence tables."""

    data_dir = Path(data_dir)
    missing = [name for name in FILES.values() if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing saved Phase 4G result files in "
            f"{data_dir}: {', '.join(missing)}. These files are produced by "
            "`python -m workflows phase4g` and must be committed to the repository."
        )
    data = {key: pd.read_csv(data_dir / name) for key, name in FILES.items()}
    for key, frame in data.items():
        try:
            _check_columns(frame, REQUIRED_COLUMNS[key])
        except ValueError as error:
            raise DashboardDataError(f"{FILES[key]}: {error}") from error

    network = data["network"]
    movements = data["movements"]
    _check_unique(network, ["period", "seed", "strategy"], "network metrics")
    _check_unique(
        movements, ["period", "seed", "strategy", "movement"], "movement metrics"
    )
    _check_complete(
        network,
        ["period", "strategy", "seed"],
        [PERIODS, STRATEGIES, SEEDS],
        "Network metrics",
    )
    _check_complete(
        movements,
        ["period", "strategy", "seed", "movement"],
        [PERIODS, STRATEGIES, SEEDS, MOVEMENTS],
        "Movement metrics",
    )
    reconcile_counts(network, movements)
    validate_signal_plans(data["plans"])
    return data


def pair_results(
    frame: pd.DataFrame,
    period: str,
    benchmark: str,
    candidate: str,
    metric: str,
    keys: Sequence[str],
    count_field: str,
) -> pd.DataFrame:
    """Pair exact simulation records and calculate candidate-minus-benchmark.

    A negative change means the candidate value is lower. The observation count
    must match within every pair so that unlike simulations cannot be compared.
    """

    if period not in PERIODS:
        raise ValueError(f"Unknown period: {period}.")
    if benchmark not in STRATEGIES or candidate not in STRATEGIES:
        raise ValueError("Unknown dashboard strategy.")
    if benchmark == candidate:
        raise ValueError("Benchmark and candidate must be different strategies.")
    if not keys:
        raise ValueError("At least one pairing key is required.")
    required = list(dict.fromkeys(["period", "strategy", *keys, metric, count_field]))
    _check_columns(frame, required)

    selected = frame.loc[frame["period"] == period]
    columns = list(dict.fromkeys([*keys, metric, count_field]))
    base = selected.loc[selected["strategy"] == benchmark, columns].copy()
    new = selected.loc[selected["strategy"] == candidate, columns].copy()
    if base.empty or new.empty:
        raise ValueError("Both selected strategies must have saved results.")
    if base.duplicated(list(keys)).any() or new.duplicated(list(keys)).any():
        raise ValueError("Pairing keys must uniquely identify each saved result.")

    paired = base.merge(
        new,
        on=list(keys),
        how="outer",
        suffixes=("_benchmark", "_candidate"),
        indicator=True,
        validate="one_to_one",
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("Strategies do not contain exactly matching records.")
    paired = paired.drop(columns="_merge")
    if not paired[f"{count_field}_benchmark"].eq(
        paired[f"{count_field}_candidate"]
    ).all():
        raise ValueError("Paired observation counts differ between strategies.")

    numeric = paired[
        [
            f"{metric}_benchmark",
            f"{metric}_candidate",
            f"{count_field}_benchmark",
            f"{count_field}_candidate",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Paired values must be finite numbers.")
    paired["change"] = (
        paired[f"{metric}_candidate"] - paired[f"{metric}_benchmark"]
    )
    return paired.sort_values(list(keys)).reset_index(drop=True)


def bootstrap_change(
    values: Sequence[float] | np.ndarray | pd.Series,
    n_bootstrap: int = 4000,
    seed: int = 2026,
) -> tuple[float, float]:
    """Return a reproducible percentile 95% interval for a paired mean change.

    The resampling scheme, draw count and generator seed match
    ``workflows.stage4a_4e_corridor.bootstrap_mean_ci``, which produced the saved Phase 4G
    summaries, so recalculated intervals reproduce the committed values.
    """

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("Need at least two finite paired changes.")
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(n_bootstrap, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def network_metric_comparison(
    frame: pd.DataFrame,
    period: str,
    benchmark: str,
    candidate: str,
    metric: str,
) -> dict[str, object]:
    """Pair one saved network metric by seed and summarise the change."""

    if metric not in NETWORK_METRICS:
        raise ValueError(f"Unsupported network metric: {metric}.")
    paired = pair_results(
        frame,
        period,
        benchmark,
        candidate,
        metric=metric,
        keys=["seed"],
        count_field="total_passages",
    )
    low, high = bootstrap_change(paired["change"])
    return {
        "metric": metric,
        "benchmark_mean": float(paired[f"{metric}_benchmark"].mean()),
        "candidate_mean": float(paired[f"{metric}_candidate"].mean()),
        "mean_change": float(paired["change"].mean()),
        "ci_lower": low,
        "ci_upper": high,
        "replications": len(paired),
        "seeds_lower": int((paired["change"] < 0).sum()),
        "seeds_equal": int((paired["change"] == 0).sum()),
        "paired": paired,
    }


def network_comparison(
    frame: pd.DataFrame,
    period: str,
    benchmark: str,
    candidate: str,
) -> dict[str, float | int]:
    """Summarise paired network delay for one saved Phase 4G comparison."""

    result = network_metric_comparison(
        frame, period, benchmark, candidate, "mean_delay_per_passage_s"
    )
    return {
        "benchmark_mean_delay_s": result["benchmark_mean"],
        "candidate_mean_delay_s": result["candidate_mean"],
        "mean_delay_change_s": result["mean_change"],
        "ci_lower_s": result["ci_lower"],
        "ci_upper_s": result["ci_upper"],
        "replications": result["replications"],
        "seeds_better": result["seeds_lower"],
    }


def movement_comparison(
    frame: pd.DataFrame,
    period: str,
    benchmark: str,
    candidate: str,
) -> pd.DataFrame:
    """Summarise paired delay and fairness for every saved movement.

    Movements are paired on both movement identifier and seed. The +2-second
    rule is only assessed for the four protected movements; other movements
    are reported with ``fairness_check`` set to "Not assessed".
    """

    _check_columns(frame, ["movement", "mean_delay_per_vehicle_s", "vehicles"])
    records: list[dict[str, object]] = []
    available = set(frame.loc[frame["period"] == period, "movement"])
    for movement in MOVEMENTS:
        if movement not in available:
            continue
        movement_rows = frame.loc[frame["movement"] == movement]
        paired = pair_results(
            movement_rows,
            period,
            benchmark,
            candidate,
            metric="mean_delay_per_vehicle_s",
            keys=["seed"],
            count_field="vehicles",
        )
        low, high = bootstrap_change(paired["change"])
        protected = movement in PROTECTED_MOVEMENTS
        passes = bool(high <= FAIRNESS_LIMIT_S)
        records.append(
            {
                "movement": movement,
                "movement_label": MOVEMENT_LABELS[movement],
                "protected_movement": protected,
                "benchmark_mean_delay_s": paired[
                    "mean_delay_per_vehicle_s_benchmark"
                ].mean(),
                "candidate_mean_delay_s": paired[
                    "mean_delay_per_vehicle_s_candidate"
                ].mean(),
                "mean_delay_change_s": paired["change"].mean(),
                "ci_lower_s": low,
                "ci_upper_s": high,
                "passes_2s_upper": passes,
                "fairness_check": (
                    ("Pass" if passes else "Fail") if protected else "Not assessed"
                ),
                "replications": len(paired),
            }
        )
    if not records:
        raise ValueError("No movement results are available for the selected period.")
    return pd.DataFrame(records)


def fairness_failures(
    frame: pd.DataFrame,
    period: str,
    benchmark: str,
    candidate: str,
) -> list[str]:
    """Return protected movements whose upper bound exceeds +2 seconds."""

    comparison = movement_comparison(frame, period, benchmark, candidate)
    failing = comparison.loc[comparison["fairness_check"] == "Fail", "movement"]
    return failing.tolist()


def plan_rows(plans: pd.DataFrame, period: str, strategy: str) -> pd.DataFrame:
    """Return one validated A/B/C plan in junction order."""

    rows = plans.loc[(plans["period"] == period) & (plans["strategy"] == strategy)]
    if sorted(rows["junction"]) != list(JUNCTIONS) or len(rows) != 3:
        raise DashboardDataError(
            f"Plan for {strategy} in {period} must contain A, B and C once."
        )
    return rows.set_index("junction").loc[list(JUNCTIONS)].reset_index()


def plans_identical(
    plans: pd.DataFrame, period: str, first: str, second: str
) -> bool:
    """Return whether two saved strategies use identical timings."""

    columns = list(REQUIRED_COLUMNS["plans"][3:])
    a = plan_rows(plans, period, first)[columns].to_numpy(dtype=float)
    b = plan_rows(plans, period, second)[columns].to_numpy(dtype=float)
    return bool(np.allclose(a, b))


def green_segments(
    offset_s: float,
    green_s: float,
    cycle_s: float,
) -> list[tuple[float, float]]:
    """Split a repeating effective-green window at the cycle boundary.

    Windows are half-open, ``[start, end)``, matching
    ``traffic_signals.full_junction.is_effective_green``.
    """

    values = np.asarray([offset_s, green_s, cycle_s], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite signal timing.")
    if cycle_s <= 0 or not 0 < green_s < cycle_s:
        raise ValueError("Invalid effective-green duration.")
    start = float(offset_s % cycle_s)
    end = start + float(green_s)
    if end <= cycle_s:
        return [(start, end)]
    return [(start, float(cycle_s)), (0.0, end - float(cycle_s))]


def phase_windows(plan: Mapping[str, float]) -> list[dict[str, object]]:
    """Return every EW, NS and interphase window in one cycle.

    Each window is split at the cycle boundary when it wraps, so the returned
    segments always lie within ``[0, cycle_s]``.
    """

    cycle = float(plan["cycle_s"])
    ew_green = float(plan["ew_green_s"])
    ns_green = float(plan["ns_green_s"])
    loss = float(plan["interphase_loss_s"])
    ew_start = float(plan["ew_offset_s"]) % cycle
    ns_start = float(plan["ns_offset_s"]) % cycle
    if not np.isclose((ew_start + ew_green + loss) % cycle, ns_start):
        raise ValueError("NS offset does not follow the EW phase convention.")
    windows = [
        ("EW green", ew_start, ew_green),
        ("Interphase", (ew_start + ew_green) % cycle, loss),
        ("NS green", ns_start, ns_green),
        ("Interphase", (ns_start + ns_green) % cycle, loss),
    ]
    segments: list[dict[str, object]] = []
    for kind, start, duration in windows:
        if duration <= 0:
            continue
        for seg_start, seg_end in green_segments(start, duration, cycle):
            segments.append(
                {
                    "kind": kind,
                    "start_s": seg_start,
                    "end_s": seg_end,
                    "window_start_s": start,
                    "duration_s": duration,
                    "wraps": start + duration > cycle,
                }
            )
    covered = sum(s["end_s"] - s["start_s"] for s in segments)
    if not np.isclose(covered, cycle):
        raise ValueError("Phase windows do not cover the full cycle.")
    return segments
