"""Testable data preparation for the read-only Phase 5A dashboard."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


PERIODS = ("Overnight", "AM peak", "Interpeak", "PM peak")
STRATEGIES = ("Phase3A", "Phase4E", "Integrated")
MOVEMENTS = ("EB", "WB", "A_N", "A_S", "B_N", "B_S", "C_N", "C_S")
PROTECTED_MOVEMENTS = ("EB", "WB", "C_N", "C_S")
FAIRNESS_LIMIT_S = 2.0


def _check_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}.")


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
    """Return a reproducible percentile 95% interval for a paired mean change."""

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


def network_comparison(
    frame: pd.DataFrame,
    period: str,
    benchmark: str,
    candidate: str,
) -> dict[str, float | int]:
    """Summarise paired network delay for one saved Phase 4G comparison."""

    metric = "mean_delay_per_passage_s"
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
        "benchmark_mean_delay_s": float(paired[f"{metric}_benchmark"].mean()),
        "candidate_mean_delay_s": float(paired[f"{metric}_candidate"].mean()),
        "mean_delay_change_s": float(paired["change"].mean()),
        "ci_lower_s": low,
        "ci_upper_s": high,
        "replications": len(paired),
        "seeds_better": int((paired["change"] < 0).sum()),
    }


def movement_comparison(
    frame: pd.DataFrame,
    period: str,
    benchmark: str,
    candidate: str,
) -> pd.DataFrame:
    """Summarise paired delay and fairness for every saved movement."""

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
        records.append(
            {
                "movement": movement,
                "protected_movement": movement in PROTECTED_MOVEMENTS,
                "benchmark_mean_delay_s": paired[
                    "mean_delay_per_vehicle_s_benchmark"
                ].mean(),
                "candidate_mean_delay_s": paired[
                    "mean_delay_per_vehicle_s_candidate"
                ].mean(),
                "mean_delay_change_s": paired["change"].mean(),
                "ci_lower_s": low,
                "ci_upper_s": high,
                "passes_2s_upper": bool(high <= FAIRNESS_LIMIT_S),
                "replications": len(paired),
            }
        )
    if not records:
        raise ValueError("No movement results are available for the selected period.")
    return pd.DataFrame(records)


def green_segments(
    offset_s: float,
    green_s: float,
    cycle_s: float,
) -> list[tuple[float, float]]:
    """Split a repeating effective-green window at the cycle boundary."""

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
