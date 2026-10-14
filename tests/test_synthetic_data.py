import pandas as pd
import pytest

from traffic_signals.synthetic_data import SyntheticDataConfig, generate_synthetic_data
from traffic_signals.webster import counts_15min_to_hourly_flow


def test_generated_data_has_expected_shape_columns_and_units() -> None:
    data = generate_synthetic_data()

    assert len(data) == 96 * 3 * 4
    assert list(data.columns) == [
        "time",
        "intersection_id",
        "approach",
        "volume_15min",
        "saturation_flow_vph",
    ]
    assert data["time"].nunique() == 96
    assert set(data["intersection_id"]) == {"A", "B", "C"}
    assert set(data["approach"]) == {
        "eastbound",
        "westbound",
        "northbound",
        "southbound",
    }
    assert pd.api.types.is_integer_dtype(data["volume_15min"])
    assert (data["volume_15min"] >= 0).all()
    assert (data["saturation_flow_vph"] == 1800.0).all()


def test_generation_is_reproducible_with_seed_42() -> None:
    first = generate_synthetic_data(SyntheticDataConfig(seed=42))
    second = generate_synthetic_data(SyntheticDataConfig(seed=42))
    pd.testing.assert_frame_equal(first, second)


def test_15_minute_count_to_hourly_flow_conversion() -> None:
    assert counts_15min_to_hourly_flow(125) == pytest.approx(500.0)


@pytest.mark.parametrize("invalid_count", [-1, float("nan"), float("inf")])
def test_hourly_conversion_rejects_invalid_counts(invalid_count: float) -> None:
    with pytest.raises(ValueError):
        counts_15min_to_hourly_flow(invalid_count)
