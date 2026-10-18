"""Regression tests that pin the dashboard to the saved Phase 4G evidence."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traffic_signals.dashboard_data import (
    BENCHMARKS,
    FAIRNESS_LIMIT_S,
    FILES,
    MOVEMENTS,
    PERIODS,
    PROTECTED_MOVEMENTS,
    DashboardDataError,
    fairness_failures,
    green_segments,
    load_phase4g_data,
    movement_comparison,
    network_comparison,
    network_metric_comparison,
    pair_results,
    phase_windows,
    plan_rows,
    plans_identical,
    validate_signal_plans,
)
from traffic_signals.full_junction import is_effective_green


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "outputs" / "data"


@pytest.fixture(scope="module")
def data() -> dict[str, pd.DataFrame]:
    return load_phase4g_data(DATA_DIR)


# 1. PM Integrated vs Phase 3A network delay.
def test_pm_integrated_vs_phase3a_network_delay(data) -> None:
    result = network_comparison(data["network"], "PM peak", "Phase3A", "Integrated")
    assert result["mean_delay_change_s"] == pytest.approx(-8.611, abs=5e-4)
    assert result["ci_lower_s"] == pytest.approx(-9.976, abs=5e-4)
    assert result["ci_upper_s"] == pytest.approx(-7.284, abs=5e-4)
    assert result["seeds_better"] == 30
    assert result["replications"] == 30
    queue = network_metric_comparison(
        data["network"], "PM peak", "Phase3A", "Integrated", "total_queue_at_hour"
    )
    assert queue["mean_change"] == pytest.approx(-27.433, abs=5e-4)


def test_headline_network_delay_for_all_periods(data) -> None:
    expected = {
        "Overnight": -0.186,
        "AM peak": -1.434,
        "Interpeak": -0.166,
        "PM peak": -8.611,
    }
    for period, value in expected.items():
        result = network_comparison(data["network"], period, "Phase3A", "Integrated")
        assert result["mean_delay_change_s"] == pytest.approx(value, abs=5e-4)


# 2. PM Integrated vs Phase 3A eastbound delay.
def test_pm_integrated_vs_phase3a_eastbound_delay(data) -> None:
    table = movement_comparison(
        data["movements"], "PM peak", "Phase3A", "Integrated"
    ).set_index("movement")
    assert table.loc["EB", "mean_delay_change_s"] == pytest.approx(2.906, abs=5e-4)
    assert table.loc["EB", "ci_lower_s"] == pytest.approx(2.732, abs=5e-4)
    assert table.loc["EB", "ci_upper_s"] == pytest.approx(3.084, abs=5e-4)
    assert table.loc["WB", "mean_delay_change_s"] == pytest.approx(-9.376, abs=5e-4)
    assert table.loc["C_N", "mean_delay_change_s"] == pytest.approx(-11.540, abs=5e-4)
    assert table.loc["C_S", "mean_delay_change_s"] == pytest.approx(-82.519, abs=5e-4)


# 3. PM Integrated vs Phase 4E eastbound delay.
def test_pm_integrated_vs_phase4e_eastbound_delay(data) -> None:
    table = movement_comparison(
        data["movements"], "PM peak", "Phase4E", "Integrated"
    ).set_index("movement")
    assert table.loc["EB", "mean_delay_change_s"] == pytest.approx(1.620, abs=5e-4)


# 4. PM EB fairness fails versus Phase 3A.
def test_pm_eastbound_fairness_fails_against_phase3a(data) -> None:
    failures = fairness_failures(data["movements"], "PM peak", "Phase3A", "Integrated")
    assert failures == ["EB"]
    table = movement_comparison(
        data["movements"], "PM peak", "Phase3A", "Integrated"
    ).set_index("movement")
    assert table.loc["EB", "ci_upper_s"] > FAIRNESS_LIMIT_S
    assert table.loc["EB", "fairness_check"] == "Fail"


# 5. PM EB fairness passes versus Phase 4E.
def test_pm_eastbound_fairness_passes_against_phase4e(data) -> None:
    assert fairness_failures(data["movements"], "PM peak", "Phase4E", "Integrated") == []


def test_only_protected_movements_receive_a_fairness_verdict(data) -> None:
    table = movement_comparison(data["movements"], "PM peak", "Phase3A", "Integrated")
    unprotected = table.loc[~table["movement"].isin(PROTECTED_MOVEMENTS)]
    assert set(unprotected["fairness_check"]) == {"Not assessed"}
    protected = table.loc[table["movement"].isin(PROTECTED_MOVEMENTS)]
    assert set(protected["fairness_check"]) <= {"Pass", "Fail"}
    assert table["movement"].tolist() == list(MOVEMENTS)


def test_complete_set_of_saved_integrated_fairness_failures(data) -> None:
    # The saved Phase 4G summary also records a marginal AM C-N exceedance
    # against Phase 3A (upper bound about +2.197 s). The AM Integrated plan is
    # identical to Phase 4E, whose AM offsets were selected before C-N was
    # modelled. Pin the complete set so neither failure can disappear silently.
    expected = {
        ("AM peak", "Phase3A"): ["C_N"],
        ("PM peak", "Phase3A"): ["EB"],
    }
    for period in PERIODS:
        for benchmark in BENCHMARKS:
            failures = fairness_failures(
                data["movements"], period, benchmark, "Integrated"
            )
            assert failures == expected.get((period, benchmark), [])
    am = movement_comparison(
        data["movements"], "AM peak", "Phase3A", "Integrated"
    ).set_index("movement")
    assert am.loc["C_N", "mean_delay_change_s"] == pytest.approx(0.919, abs=5e-4)
    assert am.loc["C_N", "ci_upper_s"] == pytest.approx(2.197, abs=5e-4)


# 6. Non-PM Integrated plans equal Phase 4E.
def test_non_pm_integrated_plans_equal_phase4e(data) -> None:
    for period in ("Overnight", "AM peak", "Interpeak"):
        assert plans_identical(data["plans"], period, "Phase4E", "Integrated")
        result = network_comparison(data["network"], period, "Phase4E", "Integrated")
        assert result["mean_delay_change_s"] == 0.0
    assert not plans_identical(data["plans"], "PM peak", "Phase4E", "Integrated")


def test_pm_integrated_changes_only_junction_c(data) -> None:
    e = plan_rows(data["plans"], "PM peak", "Phase4E").set_index("junction")
    i = plan_rows(data["plans"], "PM peak", "Integrated").set_index("junction")
    pd.testing.assert_frame_equal(
        e.loc[["A", "B"]].drop(columns="strategy"),
        i.loc[["A", "B"]].drop(columns="strategy"),
    )


# 7. Junction C Integrated PM timing matches the Phase 4G and 4F-3B selection.
def test_pm_junction_c_integrated_timing(data) -> None:
    c = plan_rows(data["plans"], "PM peak", "Integrated").set_index("junction").loc["C"]
    assert c["cycle_s"] == 150
    assert c["ew_green_s"] == 89
    assert c["ns_green_s"] == 49
    assert c["ew_offset_s"] == 83
    assert c["ns_offset_s"] == 28
    selected = pd.read_csv(DATA_DIR / "phase4f3b_selected_plans.csv").set_index("period")
    pm = selected.loc["PM peak"]
    assert (pm["C_ew_green_s"], pm["C_ns_green_s"]) == (89, 49)
    assert (pm["C_ew_offset_s"], pm["C_ns_offset_s"]) == (83, 28)


# 8. Cyclic diagrams wrap correctly and match the simulation convention.
def test_phase_windows_wrap_and_match_simulation_convention(data) -> None:
    for (period, strategy), _ in data["plans"].groupby(["period", "strategy"]):
        for _, row in plan_rows(data["plans"], period, strategy).iterrows():
            plan = row.to_dict()
            segments = phase_windows(plan)
            cycle = int(plan["cycle_s"])
            for second in range(cycle):
                t = second + 0.5
                for kind, phase in (("EW green", "EW"), ("NS green", "NS")):
                    in_segment = any(
                        s["kind"] == kind and s["start_s"] <= t < s["end_s"]
                        for s in segments
                    )
                    assert in_segment == is_effective_green(t, plan, phase)
            for segment in segments:
                assert 0.0 <= segment["start_s"] < segment["end_s"] <= cycle


def test_pm_junction_c_ew_green_wraps_at_cycle_boundary(data) -> None:
    c = plan_rows(data["plans"], "PM peak", "Integrated").set_index("junction").loc["C"]
    ew = [s for s in phase_windows(c.to_dict()) if s["kind"] == "EW green"]
    assert [(s["start_s"], s["end_s"]) for s in ew] == [(83.0, 150.0), (0.0, 22.0)]
    assert green_segments(83, 89, 150) == [(83.0, 150.0), (0.0, 22.0)]


# 9. All dashboard datasets load from repository-relative paths.
def test_dashboard_loads_all_datasets_from_repository() -> None:
    for filename in FILES.values():
        assert (DATA_DIR / filename).is_file()
    data = load_phase4g_data(DATA_DIR)
    assert len(data["network"]) == 4 * 3 * 30
    assert len(data["movements"]) == 4 * 3 * 30 * 8
    assert len(data["plans"]) == 4 * 3 * 3


def test_dashboard_source_has_no_absolute_paths_or_simulation_calls() -> None:
    source = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    assert "/Users/" not in source and "C:\\" not in source
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    forbidden = {"evaluate_network", "simulate_direction", "search_delay_offsets"}
    assert not forbidden & imported
    assert not any(m.startswith("phase4") for m in modules)


def test_missing_file_gives_helpful_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="python -m workflows phase4g"):
        load_phase4g_data(tmp_path)


# Data-processing safeguards.
def test_duplicate_and_missing_rows_are_rejected(tmp_path, data) -> None:
    for name, frame in data.items():
        frame.to_csv(tmp_path / FILES[name], index=False)
    network = data["network"]
    pd.concat([network, network.iloc[[0]]]).to_csv(
        tmp_path / FILES["network"], index=False
    )
    with pytest.raises(DashboardDataError, match="Duplicate"):
        load_phase4g_data(tmp_path)
    network.iloc[1:].to_csv(tmp_path / FILES["network"], index=False)
    with pytest.raises(DashboardDataError, match="missing"):
        load_phase4g_data(tmp_path)


def test_count_mismatch_is_rejected(tmp_path, data) -> None:
    for name, frame in data.items():
        frame.to_csv(tmp_path / FILES[name], index=False)
    network = data["network"].copy()
    network.loc[0, "total_passages"] += 1
    network.to_csv(tmp_path / FILES["network"], index=False)
    with pytest.raises(DashboardDataError, match="reconcile"):
        load_phase4g_data(tmp_path)


def test_movement_pairing_requires_matching_seeds(data) -> None:
    movements = data["movements"]
    drop = movements.index[
        (movements["period"] == "PM peak")
        & (movements["strategy"] == "Integrated")
        & (movements["movement"] == "EB")
        & (movements["seed"] == 500)
    ]
    with pytest.raises(ValueError, match="exactly matching"):
        movement_comparison(movements.drop(drop), "PM peak", "Phase3A", "Integrated")


def test_same_benchmark_and_candidate_is_rejected(data) -> None:
    with pytest.raises(ValueError, match="different strategies"):
        network_comparison(data["network"], "PM peak", "Phase4E", "Phase4E")


def test_negative_change_means_lower_candidate_delay(data) -> None:
    paired = pair_results(
        data["network"],
        "PM peak",
        "Phase3A",
        "Integrated",
        "mean_delay_per_passage_s",
        ["seed"],
        "total_passages",
    )
    lower = (
        paired["mean_delay_per_passage_s_candidate"]
        < paired["mean_delay_per_passage_s_benchmark"]
    )
    assert lower.equals(paired["change"] < 0)


def test_recalculated_summaries_reproduce_saved_phase4g_tables(data) -> None:
    saved = data["network_summary"]
    for _, row in saved.iterrows():
        result = network_comparison(
            data["network"], row["period"], row["benchmark"], "Integrated"
        )
        assert result["mean_delay_change_s"] == pytest.approx(row["mean_delay_change_s"], abs=1e-9)
        assert result["ci_lower_s"] == pytest.approx(row["ci_lower_s"], abs=1e-9)
        assert result["ci_upper_s"] == pytest.approx(row["ci_upper_s"], abs=1e-9)
        assert result["seeds_better"] == row["seeds_better"]
    movement_saved = data["movement_summary"]
    for (period, benchmark), group in movement_saved.groupby(["period", "benchmark"]):
        table = movement_comparison(
            data["movements"], period, benchmark, "Integrated"
        ).set_index("movement")
        for _, row in group.iterrows():
            calc = table.loc[row["movement"]]
            assert calc["mean_delay_change_s"] == pytest.approx(row["mean_delay_change_s"], abs=1e-9)
            assert calc["ci_upper_s"] == pytest.approx(row["ci_upper_s"], abs=1e-9)
            assert bool(calc["passes_2s_upper"]) == bool(row["passes_2s_upper"])


def test_invalid_plan_convention_is_rejected(data) -> None:
    plans = data["plans"].copy()
    plans.loc[0, "ns_offset_s"] = (plans.loc[0, "ns_offset_s"] + 1) % plans.loc[0, "cycle_s"]
    with pytest.raises(DashboardDataError, match="NS offsets"):
        validate_signal_plans(plans)
    assert np.isfinite(data["plans"]["cycle_s"]).all()


def test_streamlit_app_renders_default_pm_fairness_failure() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60).run()
    assert not app.exception
    assert app.error, "The PM Integrated vs Phase 3A fairness failure must be visible."
    assert "EB" in app.error[0].value and "+2.906" in app.error[0].value
    metrics = {m.label: m for m in app.metric}
    assert metrics["Integrated network delay"].delta.startswith("-8.611")
    assert metrics["Seeds with lower delay"].value == "30 / 30"


def test_streamlit_app_reports_pass_against_phase4e() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60).run()
    app.selectbox[1].set_value("Phase4E").run()
    assert not app.exception
    assert not app.error
    assert app.success and "pass" in app.success[0].value


# GitHub Pages site.
def test_site_data_is_current_and_matches_validated_layer() -> None:
    import json
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from export_site_data import OUTPUT_PATH, build_site_data

    saved = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    assert saved == json.loads(json.dumps(build_site_data())), (
        "docs/assets/phase4g_results.json is stale: run scripts/export_site_data.py"
    )
    pm = saved["comparisons"]["PM peak|Phase3A|Integrated"]
    assert pm["metrics"]["mean_delay_per_passage_s"]["mean_change"] == pytest.approx(-8.611, abs=5e-4)
    eb = next(m for m in pm["movements"] if m["movement"] == "EB")
    assert eb["check"] == "Fail" and eb["mean_change"] == pytest.approx(2.906, abs=5e-4)
    assert "PM peak|Phase4E|Phase4E" not in saved["comparisons"]


def test_site_page_is_self_contained() -> None:
    page = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    assert "assets/phase4g_results.json" in page
    assert "<script src=" not in page and "<link rel=\"stylesheet\"" not in page
    assert chr(0x2014) not in page
    assert (ROOT / "docs" / ".nojekyll").exists()


# Phase 4H follow-up: fairness measured against the original Phase 3A plan.
def test_phase4h_seeds_are_fresh_and_separate() -> None:
    from workflows.stage4f_4g_full_junction import (
        PHASE4G_SEEDS,
        PHASE4H_HOLDOUT_SEEDS,
        PHASE4H_SCREEN_SEEDS,
        PHASE4H_TRAIN_SEEDS,
        HOLDOUT_SEEDS,
        TRAIN_SEEDS,
    )

    earlier = set(PHASE4G_SEEDS) | set(HOLDOUT_SEEDS) | set(TRAIN_SEEDS) | set(range(200, 230))
    assert not earlier & (set(PHASE4H_TRAIN_SEEDS) | set(PHASE4H_HOLDOUT_SEEDS))
    assert not set(PHASE4H_TRAIN_SEEDS) & set(PHASE4H_HOLDOUT_SEEDS)
    assert set(PHASE4H_SCREEN_SEEDS) <= set(PHASE4H_TRAIN_SEEDS)


def test_phase4h_selected_plan_and_holdout_results() -> None:
    selected = pd.read_csv(DATA_DIR / "phase4h_selected_plan.csv").iloc[0]
    assert (selected["C_ew_green_s"], selected["C_ns_green_s"]) == (87, 51)
    assert (selected["C_ew_offset_s"], selected["C_ns_offset_s"]) == (79, 22)

    network = pd.read_csv(DATA_DIR / "phase4h_network_summary.csv").set_index(["benchmark", "candidate"])
    vs3a = network.loc[("Phase3A", "Phase4H")]
    assert vs3a["mean_delay_change_s"] == pytest.approx(-10.520, abs=5e-4)
    assert vs3a["ci_upper_s"] < 0 and vs3a["seeds_better"] == 30

    movements = pd.read_csv(DATA_DIR / "phase4h_movement_summary.csv")
    protected = movements.loc[movements["protected_movement"]].set_index(["benchmark", "candidate", "movement"])
    for movement in PROTECTED_MOVEMENTS:
        assert bool(protected.loc[("Phase3A", "Phase4H", movement), "passes_2s_upper"])
    assert protected.loc[("Phase3A", "Phase4H", "EB"), "mean_delay_change_s"] == pytest.approx(1.298, abs=5e-4)
    # The original finding replicates on the fresh seeds.
    assert not bool(protected.loc[("Phase3A", "Integrated", "EB"), "passes_2s_upper"])
    # The trade-off moves to westbound against Phase 4E.
    assert not bool(protected.loc[("Phase4E", "Phase4H", "WB"), "passes_2s_upper"])


def test_phase4h_integrated_plan_matches_phase4g() -> None:
    from workflows.stage4f_4g_full_junction import (
        PHASE4H_INTEGRATED_PAIR,
        build_period_plans,
        load_inputs_phase4f2,
        make_candidate_plan,
    )

    _, plans_3a, plans_4e = load_inputs_phase4f2()
    base = build_period_plans(plans_3a, plans_4e, "PM peak")["Phase4E"]
    c = make_candidate_plan(base, *PHASE4H_INTEGRATED_PAIR)["C"]
    saved = load_phase4g_data(DATA_DIR)["plans"]
    row = saved.loc[(saved["period"] == "PM peak") & (saved["strategy"] == "Integrated") & (saved["junction"] == "C")].iloc[0]
    for key in ("ew_green_s", "ns_green_s", "ew_offset_s", "ns_offset_s"):
        assert c[key] == row[key]


# Phase 4I: AM peak under the Phase 4H rules.
def test_phase4i_protocol_and_outcome() -> None:
    from workflows.stage4f_4g_full_junction import (
        EARLIER_SEEDS,
        PHASE4H_HOLDOUT_SEEDS,
        PHASE4H_TRAIN_SEEDS,
        PHASE4I_HOLDOUT_SEEDS,
        PHASE4I_TRAIN_SEEDS,
    )

    used = EARLIER_SEEDS | set(PHASE4H_TRAIN_SEEDS) | set(PHASE4H_HOLDOUT_SEEDS)
    assert not used & (set(PHASE4I_TRAIN_SEEDS) | set(PHASE4I_HOLDOUT_SEEDS))
    selected = pd.read_csv(DATA_DIR / "phase4i_selected_plan.csv").iloc[0]
    assert not bool(selected["selected"])
    candidates = pd.read_csv(DATA_DIR / "phase4i_training_candidates.csv")
    assert len(candidates) == 16
    assert candidates["network_performance_pass"].all()
    assert not candidates["movement_fairness_pass"].any()
    fairness = pd.read_csv(DATA_DIR / "phase4i_training_fairness.csv")
    eb_added = fairness.loc[(fairness["movement"] == "EB") & (fairness["extra_ns_green_s"] > 0)]
    assert (eb_added["ci_upper_s"] > FAIRNESS_LIMIT_S).all()
