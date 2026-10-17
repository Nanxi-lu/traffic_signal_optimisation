"""Synthetic directional demand proxies for corridor coordination."""

import numpy as np
import pandas as pd

from traffic_signals.segmentation import PERIOD_ORDER, assign_period


def calculate_direction_weights(traffic: pd.DataFrame) -> pd.DataFrame:
    """Estimate period EB/WB weights from the two corridor-entry approaches."""

    df = traffic.copy()
    required = {"time", "intersection_id", "approach", "volume_15min"}
    if not required.issubset(df.columns):
        raise ValueError("Missing required traffic columns.")
    df["time"] = pd.to_datetime(df["time"], errors="raise")
    df["period"] = df["time"].apply(assign_period)
    df["intersection_id"] = df["intersection_id"].astype(str).str.strip().str.upper()
    df["approach"] = (
        df["approach"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"EASTBOUND": "E", "WESTBOUND": "W"})
    )
    counts = pd.to_numeric(df["volume_15min"], errors="raise")
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Traffic counts must be finite and non-negative.")
    df["volume_15min"] = counts

    entries = df.loc[
        ((df["intersection_id"] == "A") & (df["approach"] == "E"))
        | ((df["intersection_id"] == "C") & (df["approach"] == "W"))
    ].copy()
    entries["direction"] = np.where(entries["intersection_id"] == "A", "EB", "WB")
    totals = (
        entries.groupby(["period", "direction"], as_index=False)["volume_15min"]
        .sum()
        .pivot(index="period", columns="direction", values="volume_15min")
        .fillna(0.0)
    )
    for direction in ("EB", "WB"):
        if direction not in totals:
            totals[direction] = 0.0
    total_demand = totals["EB"] + totals["WB"]
    if (total_demand <= 0).any() or totals.empty:
        raise ValueError("Each represented period needs positive corridor-entry demand.")

    result = totals.reset_index().rename(
        columns={"EB": "EB_volume_15min", "WB": "WB_volume_15min"}
    )
    result["EB_weight"] = result["EB_volume_15min"] / total_demand.to_numpy()
    result["WB_weight"] = result["WB_volume_15min"] / total_demand.to_numpy()
    order = {period: index for index, period in enumerate(PERIOD_ORDER)}
    result["_order"] = result["period"].map(order)
    return result.sort_values("_order").drop(columns="_order").reset_index(drop=True)
