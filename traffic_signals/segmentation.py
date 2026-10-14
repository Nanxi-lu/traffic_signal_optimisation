"""Rule-based time-of-day segmentation for Phase 2."""

import pandas as pd


PERIOD_ORDER = ("Overnight", "AM peak", "Interpeak", "PM peak")


def assign_period(timestamp: pd.Timestamp) -> str:
    """Assign a timestamp to one of four illustrative operating periods."""

    hour = timestamp.hour + timestamp.minute / 60.0
    if 6 <= hour < 10:
        return "AM peak"
    if 10 <= hour < 16:
        return "Interpeak"
    if 16 <= hour < 20:
        return "PM peak"
    return "Overnight"
