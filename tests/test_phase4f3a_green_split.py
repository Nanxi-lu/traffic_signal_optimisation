import numpy as np
import pytest

from traffic_signals.full_junction import (
    create_two_phase_plan,
    is_effective_green,
)
from traffic_signals.green_split_sensitivity import (
    modify_c_green,
    theoretical_capacity_vph,
)


def make_plans():
    return {
        junction: create_two_phase_plan(
            cycle_s=150,
            ew_green_s=95,
            ns_green_s=43,
            ew_offset_s=70,
        )
        for junction in ("A", "B", "C")
    }


def test_green_time_conservation():
    modified = modify_c_green(make_plans(), 5)
    c = modified["C"]
    assert c["ew_green_s"] == 90
    assert c["ns_green_s"] == 48
    assert c["ew_green_s"] + c["ns_green_s"] + c["lost_time_s"] == 150


def test_a_and_b_remain_unchanged():
    original = make_plans()
    modified = modify_c_green(original, 7)
    assert modified["A"] == original["A"]
    assert modified["B"] == original["B"]
    assert modified["A"] is not original["A"]
    assert modified["B"] is not original["B"]


def test_baseline_preserved_without_mutation():
    original = make_plans()
    snapshot = {junction: dict(plan) for junction, plan in original.items()}
    modified = modify_c_green(original, 0)
    assert modified == original
    assert original == snapshot
    assert modified["C"] is not original["C"]


def test_ns_phase_boundary_moves_earlier_without_overlap():
    original = make_plans()
    modified = modify_c_green(original, 5)
    assert modified["C"]["ew_offset_s"] == original["C"]["ew_offset_s"]
    assert modified["C"]["ns_offset_s"] == pytest.approx(
        (original["C"]["ns_offset_s"] - 5) % 150
    )
    times = np.arange(0.0, 150.0, 0.25)
    overlaps = [
        is_effective_green(time_s, modified["C"], "EW")
        and is_effective_green(time_s, modified["C"], "NS")
        for time_s in times
    ]
    assert not any(overlaps)


def test_theoretical_capacity_and_invalid_inputs():
    assert theoretical_capacity_vph(46, 150) == pytest.approx(552)
    with pytest.raises(ValueError):
        modify_c_green(make_plans(), -1)
    with pytest.raises(ValueError):
        theoretical_capacity_vph(151, 150)
