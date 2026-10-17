"""Synthetic arrivals that preserve four consecutive 15-minute profiles."""

from collections.abc import Iterable

import numpy as np
import pandas as pd


def generate_hourly_arrivals(
    traffic: pd.DataFrame,
    start_time: str | pd.Timestamp,
    streams: Iterable[tuple[str, str]],
    seed: int,
) -> dict[tuple[str, str], np.ndarray]:
    """Generate Poisson arrivals from four stored 15-minute counts."""

    required = {
        "time",
        "intersection_id",
        "approach",
        "volume_15min",
    }
    if not required.issubset(traffic.columns):
        raise ValueError("Missing required traffic columns.")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("Seed must be an integer.")
    start = pd.Timestamp(start_time)
    if pd.isna(start):
        raise ValueError("Invalid start time.")
    slots = pd.date_range(start, periods=4, freq="15min")
    stream_list = list(streams)
    if not stream_list or len(set(stream_list)) != len(stream_list):
        raise ValueError("Streams must be non-empty and unique.")

    times = pd.to_datetime(traffic["time"], errors="raise")
    results: dict[tuple[str, str], np.ndarray] = {}
    for stream_index, (junction, approach) in enumerate(stream_list):
        selected = traffic.loc[
            (traffic["intersection_id"] == junction)
            & (traffic["approach"] == approach)
            & (times.isin(slots))
        ].copy()
        selected["time"] = times.loc[selected.index]
        selected = selected.sort_values("time")
        if len(selected) != 4 or not selected["time"].reset_index(
            drop=True
        ).equals(pd.Series(slots)):
            raise ValueError(
                f"Expected four 15-minute observations for {junction}/{approach}."
            )
        counts = pd.to_numeric(
            selected["volume_15min"], errors="raise"
        ).to_numpy(dtype=float)
        if not np.isfinite(counts).all() or np.any(counts < 0):
            raise ValueError("Invalid 15-minute traffic counts.")

        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), stream_index, 406])
        )
        generated: list[np.ndarray] = []
        for slot_index, expected_count in enumerate(counts):
            vehicle_count = int(rng.poisson(expected_count))
            generated.append(
                slot_index * 900.0
                + rng.uniform(0.0, 900.0, vehicle_count)
            )
        arrivals = (
            np.sort(np.concatenate(generated))
            if generated
            else np.array([], dtype=float)
        )
        results[(junction, approach)] = arrivals
    return results
