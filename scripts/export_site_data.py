"""Export saved Phase 4G comparisons for the static GitHub Pages site.

Run from the repository root:

    python scripts/export_site_data.py

The site cannot run Python, so every comparison it can display is calculated
here with the same validated functions the Streamlit dashboard uses
(``traffic_signals.dashboard_data``) and written to
``docs/assets/phase4g_results.json``. No simulation or optimisation is run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traffic_signals.dashboard_data import (  # noqa: E402
    BENCHMARKS,
    CANDIDATES,
    FAIRNESS_LIMIT_S,
    FILES,
    JUNCTION_POSITIONS_M,
    MOVEMENT_LABELS,
    MOVEMENTS,
    NETWORK_METRICS,
    PERIODS,
    PROTECTED_MOVEMENTS,
    SEEDS,
    STRATEGIES,
    STRATEGY_DESCRIPTIONS,
    STRATEGY_LABELS,
    load_phase4g_data,
    movement_comparison,
    network_metric_comparison,
    phase_windows,
    plan_rows,
    plans_identical,
)


DATA_DIR = ROOT / "outputs" / "data"
SITE_ASSETS = ROOT / "docs" / "assets"
OUTPUT_PATH = SITE_ASSETS / "phase4g_results.json"


def _r(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def build_phase4h() -> dict:
    """Saved Phase 4H follow-up summaries (fresh holdout seeds 620 to 649)."""

    import pandas as pd

    selected = pd.read_csv(DATA_DIR / "phase4h_selected_plan.csv").iloc[0]
    network = pd.read_csv(DATA_DIR / "phase4h_network_summary.csv")
    movements = pd.read_csv(DATA_DIR / "phase4h_movement_summary.csv")
    return {
        "seeds": [620, 649],
        "selected": {
            "extra_ns_green_s": int(selected["extra_ns_green_s"]),
            "C_ew_green_s": _r(selected["C_ew_green_s"], 3),
            "C_ns_green_s": _r(selected["C_ns_green_s"], 3),
            "C_ew_offset_s": _r(selected["C_ew_offset_s"], 3),
            "C_ns_offset_s": _r(selected["C_ns_offset_s"], 3),
        },
        "network": [
            {
                "benchmark": r.benchmark,
                "candidate": r.candidate,
                "benchmark_mean": _r(r.benchmark_mean_delay_per_passage_s),
                "candidate_mean": _r(r.candidate_mean_delay_per_passage_s),
                "mean_change": _r(r.mean_delay_change_s),
                "ci_lower": _r(r.ci_lower_s),
                "ci_upper": _r(r.ci_upper_s),
                "seeds_lower": int(r.seeds_better),
                "queue_change": _r(r.mean_queue_change_veh),
            }
            for r in network.itertuples()
        ],
        "movements": [
            {
                "benchmark": r.benchmark,
                "candidate": r.candidate,
                "movement": r.movement,
                "label": MOVEMENT_LABELS[r.movement],
                "mean_change": _r(r.mean_delay_change_s),
                "ci_lower": _r(r.ci_lower_s),
                "ci_upper": _r(r.ci_upper_s),
                "check": "Pass" if r.passes_2s_upper else "Fail",
            }
            for r in movements.itertuples()
            if r.protected_movement
        ],
    }


def build_phase4i() -> dict:
    """Saved Phase 4I outcome: the AM peak search under the Phase 4H rules."""

    import pandas as pd

    candidates = pd.read_csv(DATA_DIR / "phase4i_training_candidates.csv")
    fairness = pd.read_csv(DATA_DIR / "phase4i_training_fairness.csv")
    added_green = candidates["extra_ns_green_s"] > 0
    eb = fairness.loc[fairness["movement"] == "EB"].merge(
        candidates[["extra_ns_green_s", "C_offset_s"]], on=["extra_ns_green_s", "C_offset_s"]
    )
    return {
        "candidates": int(len(candidates)),
        "feasible": int(candidates["feasible"].sum()),
        "network_gain_range_s": [
            _r(-candidates["mean_network_delay_change_vs_3a_s"].max(), 1),
            _r(-candidates["mean_network_delay_change_vs_3a_s"].min(), 1),
        ],
        "min_eb_mean_with_added_green_s": _r(
            eb.loc[eb["extra_ns_green_s"] > 0, "mean_delay_change_s"].min(), 1
        ),
        "added_green_candidates": int(added_green.sum()),
    }


def build_phase4j() -> dict:
    """Saved Phase 4J stress test: Phase 4H and Integrated at 80% to 120% demand."""

    import pandas as pd

    network = pd.read_csv(DATA_DIR / "phase4j_network_summary.csv")
    movements = pd.read_csv(DATA_DIR / "phase4j_movement_summary.csv")
    protected = movements.loc[movements["protected_movement"]]
    rows = []
    for (factor, candidate), group in protected.groupby(["demand_factor", "candidate"]):
        net = network.loc[(network["demand_factor"] == factor) & (network["candidate"] == candidate)].iloc[0]
        eb = group.set_index("movement").loc["EB"]
        rows.append(
            {
                "factor": float(factor),
                "candidate": candidate,
                "network_change": _r(net["mean_delay_change_s"]),
                "network_ci": [_r(net["ci_lower_s"]), _r(net["ci_upper_s"])],
                "eb_change": _r(eb["mean_delay_change_s"]),
                "eb_ci": [_r(eb["ci_lower_s"]), _r(eb["ci_upper_s"])],
                "all_within_limit": bool(group["passes_2s_upper"].all()),
            }
        )
    return {"seeds": [800, 829], "rows": rows}


def build_site_data() -> dict:
    data = load_phase4g_data(DATA_DIR)
    network, movements, plans = data["network"], data["movements"], data["plans"]

    plan_export: dict[str, dict[str, list[dict]]] = {}
    for period in PERIODS:
        plan_export[period] = {}
        for strategy in STRATEGIES:
            rows = []
            for _, row in plan_rows(plans, period, strategy).iterrows():
                plan = row.to_dict()
                rows.append(
                    {
                        "junction": plan["junction"],
                        "cycle_s": _r(plan["cycle_s"], 3),
                        "ew_green_s": _r(plan["ew_green_s"], 3),
                        "ns_green_s": _r(plan["ns_green_s"], 3),
                        "ew_offset_s": _r(plan["ew_offset_s"], 3),
                        "ns_offset_s": _r(plan["ns_offset_s"], 3),
                        "windows": [
                            {
                                "kind": w["kind"],
                                "start_s": _r(w["start_s"], 3),
                                "end_s": _r(w["end_s"], 3),
                                "window_start_s": _r(w["window_start_s"], 3),
                                "duration_s": _r(w["duration_s"], 3),
                                "wraps": bool(w["wraps"]),
                            }
                            for w in phase_windows(plan)
                        ],
                    }
                )
            plan_export[period][strategy] = rows

    comparisons: dict[str, dict] = {}
    for period in PERIODS:
        for benchmark in BENCHMARKS:
            for candidate in CANDIDATES:
                if benchmark == candidate:
                    continue
                metrics = {}
                for metric in NETWORK_METRICS:
                    result = network_metric_comparison(
                        network, period, benchmark, candidate, metric
                    )
                    paired = result["paired"]
                    metrics[metric] = {
                        "benchmark_mean": _r(result["benchmark_mean"]),
                        "candidate_mean": _r(result["candidate_mean"]),
                        "mean_change": _r(result["mean_change"]),
                        "ci_lower": _r(result["ci_lower"]),
                        "ci_upper": _r(result["ci_upper"]),
                        "replications": int(result["replications"]),
                        "seeds_lower": int(result["seeds_lower"]),
                        "seed_changes": [
                            [int(s), _r(c)] for s, c in zip(paired["seed"], paired["change"])
                        ],
                    }
                table = movement_comparison(movements, period, benchmark, candidate)
                comparisons[f"{period}|{benchmark}|{candidate}"] = {
                    "period": period,
                    "benchmark": benchmark,
                    "candidate": candidate,
                    "identical_plans": plans_identical(plans, period, benchmark, candidate),
                    "metrics": metrics,
                    "movements": [
                        {
                            "movement": row.movement,
                            "label": row.movement_label,
                            "protected": bool(row.protected_movement),
                            "benchmark_mean": _r(row.benchmark_mean_delay_s),
                            "candidate_mean": _r(row.candidate_mean_delay_s),
                            "mean_change": _r(row.mean_delay_change_s),
                            "ci_lower": _r(row.ci_lower_s),
                            "ci_upper": _r(row.ci_upper_s),
                            "check": row.fairness_check,
                        }
                        for row in table.itertuples()
                    ],
                }

    followup = build_phase4h()
    return {
        "phase4h": followup,
        "phase4i": build_phase4i(),
        "phase4j": build_phase4j(),
        "meta": {
            "description": (
                "Saved Phase 4G audit of a synthetic three-junction corridor. "
                "Values are recalculated from committed CSV records; no simulation is run."
            ),
            "source_files": [f"outputs/data/{name}" for name in FILES.values()],
            "seeds": list(SEEDS),
            "bootstrap": "4,000 percentile resamples of paired seed differences, generator seed 2026",
            "fairness_limit_s": FAIRNESS_LIMIT_S,
            "periods": list(PERIODS),
            "benchmarks": list(BENCHMARKS),
            "candidates": list(CANDIDATES),
            "strategy_labels": STRATEGY_LABELS,
            "strategy_descriptions": STRATEGY_DESCRIPTIONS,
            "movements": list(MOVEMENTS),
            "movement_labels": MOVEMENT_LABELS,
            "protected_movements": list(PROTECTED_MOVEMENTS),
            "junction_positions_m": JUNCTION_POSITIONS_M,
            "metrics": {k: {"label": v[0], "unit": v[1]} for k, v in NETWORK_METRICS.items()},
        },
        "plans": plan_export,
        "comparisons": comparisons,
    }


def main() -> Path:
    SITE_ASSETS.mkdir(parents=True, exist_ok=True)
    site_data = build_site_data()
    OUTPUT_PATH.write_text(
        json.dumps(site_data, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)} ({OUTPUT_PATH.stat().st_size / 1024:.0f} KB)")
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
