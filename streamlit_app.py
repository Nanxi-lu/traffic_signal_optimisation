"""Read-only Streamlit explorer for the saved synthetic Phase 4G experiment.

Every number shown here is recalculated from committed CSV records in
``outputs/data``. Changing a selection changes which saved comparison is
displayed. The app never runs the queue model, generates arrivals or searches
for signal plans.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from traffic_signals.dashboard_data import (
    BENCHMARKS,
    CANDIDATES,
    FAIRNESS_LIMIT_S,
    MOVEMENT_LABELS,
    NETWORK_METRICS,
    PERIODS,
    PROTECTED_MOVEMENTS,
    STRATEGY_DESCRIPTIONS,
    STRATEGY_LABELS,
    DashboardDataError,
    load_phase4g_data,
    movement_comparison,
    network_metric_comparison,
    phase_windows,
    plan_rows,
    plans_identical,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "outputs" / "data"
CORRIDOR_IMAGE = ROOT / "figures" / "corridor_network.png"
REPO_URL = "https://github.com/Nanxi-lu/traffic_signal_optimisation"
RELATED_URL = "https://github.com/Nanxi-lu/liverpool_bus_realtime_panel"

INK = "#1f1f1e"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e8e7e1"
LOWER = "#2a78d6"
HIGHER = "#eb6834"
NOT_ASSESSED = "#b8b6ae"
FAIL = "#d03b3b"
EW_COLOUR = "#2a78d6"
NS_COLOUR = "#eb6834"
LOSS_COLOUR = "#dcdad3"
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


st.set_page_config(
    page_title="Traffic Signal Optimisation",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1200px;}
    [data-testid="stMetric"] {
        background: #f7f7f4;
        border: 1px solid #e6e5df;
        border-radius: 0.6rem;
        padding: 0.75rem 0.9rem;
    }
    [data-testid="stMetricValue"] {font-size: 1.55rem;}
    .scope-note {
        border-left: 4px solid #2a78d6;
        background: #f3f7fd;
        padding: 0.7rem 0.95rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0 1rem;
        font-size: 0.95rem;
    }
    @media (max-width: 640px) {
        .block-container {padding-left: 1rem; padding-right: 1rem;}
        h1 {font-size: 1.7rem !important;}
        [data-testid="stMetricValue"] {font-size: 1.3rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> dict[str, pd.DataFrame]:
    """Load and validate the committed Phase 4G evidence tables."""

    return load_phase4g_data(DATA_DIR)


def label(strategy: str) -> str:
    return STRATEGY_LABELS[strategy]


def base_layout(figure: go.Figure, height: int, title: str | None = None) -> go.Figure:
    figure.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=INK)) if title else None,
        height=height,
        margin=dict(l=90, r=20, t=50 if title else 20, b=70),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=INK_2, size=13),
        hoverlabel=dict(bgcolor="white", font_size=13),
    )
    figure.update_xaxes(
        gridcolor=GRID, zerolinecolor=MUTED, linecolor="#c3c2b7", automargin=True,
        title_standoff=12,
    )
    figure.update_yaxes(
        gridcolor=GRID, zerolinecolor=MUTED, linecolor="#c3c2b7", automargin=True,
        title_standoff=8,
    )
    return figure


def fmt_interval(low: float, high: float, digits: int = 3) -> str:
    return f"{low:+.{digits}f} to {high:+.{digits}f}"


# ------------------------------------------------------------------- data
try:
    data = load_dashboard_data()
except (FileNotFoundError, DashboardDataError, pd.errors.ParserError) as error:
    st.error(f"The saved results could not be loaded. {error}")
    st.stop()

network = data["network"]


@st.cache_data(show_spinner=False)
def load_phase4h() -> dict[str, float] | None:
    """Headline Phase 4H follow-up values from the saved summaries, if present."""

    try:
        net = pd.read_csv(DATA_DIR / "phase4h_network_summary.csv")
        mov = pd.read_csv(DATA_DIR / "phase4h_movement_summary.csv")
        plan = pd.read_csv(DATA_DIR / "phase4h_selected_plan.csv").iloc[0]
    except (FileNotFoundError, pd.errors.ParserError, IndexError):
        return None
    pick = mov.loc[(mov["benchmark"] == "Phase3A") & (mov["candidate"] == "Phase4H")].set_index("movement")
    row = net.loc[(net["benchmark"] == "Phase3A") & (net["candidate"] == "Phase4H")].iloc[0]
    return {
        "net": float(row["mean_delay_change_s"]),
        "eb": float(pick.loc["EB", "mean_delay_change_s"]),
        "wb": float(pick.loc["WB", "mean_delay_change_s"]),
        "ns": float(plan["C_ns_green_s"]),
    }


PHASE4H = load_phase4h()
movements = data["movements"]
plans = data["plans"]

# ----------------------------------------------------------------- header
st.title("Traffic Signal Optimisation")
st.markdown(
    "Python research explorer for fixed-time signal coordination on a **synthetic** "
    "three-junction corridor. It shows the saved Phase 4G audit: three signal strategies, "
    "four representative hours and 30 matched arrival seeds each."
)
st.markdown(
    '<div class="scope-note"><strong>Read-only results explorer.</strong> Pick a saved '
    "comparison below; nothing here reruns the simulation. Traffic is simulated.</div>",
    unsafe_allow_html=True,
)

control_columns = st.columns(3)
period = control_columns[0].selectbox("Operating period", PERIODS, index=3)
benchmark = control_columns[1].selectbox(
    "Benchmark", BENCHMARKS, index=0, format_func=label,
    help="The reference strategy. Differences are candidate minus benchmark.",
)
candidate = control_columns[2].selectbox(
    "Candidate", CANDIDATES, index=1, format_func=label,
    help="The strategy being evaluated against the benchmark.",
)

if benchmark == candidate:
    st.info(
        f"{label(benchmark)} cannot be compared with itself. Choose Phase 3A as the "
        "benchmark, or Integrated as the candidate."
    )
    st.stop()

identical = plans_identical(plans, period, benchmark, candidate)
fairness = movement_comparison(movements, period, benchmark, candidate)
failures = fairness.loc[fairness["fairness_check"] == "Fail"]

# The fairness verdict is shown above the tabs so it stays visible everywhere.
if not failures.empty:
    details = "; ".join(
        f"**{row.movement_label}** mean {row.mean_delay_change_s:+.3f} s/vehicle, 95% interval "
        f"{fmt_interval(row.ci_lower_s, row.ci_upper_s)} s"
        for row in failures.itertuples()
    )
    headline = (
        f"Over the fairness limit: {label(candidate)} vs {label(benchmark)}, {period}."
    )
    extra = ""
    if (period, benchmark, candidate) == ("PM peak", "Phase3A", "Integrated"):
        extra = (
            " Network delay falls, but eastbound drivers wait longer than the 2-second limit "
            "allows. The same plan passes against Phase 4E (+1.620 s). Phase 4H fixes this: "
            "see the Overview tab."
        )
    elif period == "AM peak" and benchmark == "Phase3A":
        extra = (
            " A marginal miss: the average change is under 1 s, but the top of the interval "
            "just passes +2 s."
        )
    st.error(f"{headline} {details}.{extra}", icon="⚠️")
elif identical:
    st.info(
        f"In the {period}, {label(candidate)} uses exactly the same signal timings as "
        f"{label(benchmark)}, so every paired difference is zero by construction.",
        icon="ℹ️",
    )
else:
    st.success(
        f"All four protected movements pass the +2-second upper-bound check for "
        f"{label(candidate)} vs {label(benchmark)} in the {period}.",
        icon="✅",
    )

overview_tab, performance_tab, fairness_tab, timing_tab, methods_tab = st.tabs(
    ("Overview", "Performance", "Movement fairness", "Signal timing", "Method and limitations")
)

delay = network_metric_comparison(
    network, period, benchmark, candidate, "mean_delay_per_passage_s"
)
queue = network_metric_comparison(
    network, period, benchmark, candidate, "total_queue_at_hour"
)

# --------------------------------------------------------------- overview
with overview_tab:
    st.subheader(f"{period}: {label(candidate)} vs {label(benchmark)}")
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        f"{label(benchmark)} network delay",
        f"{delay['benchmark_mean']:.2f} s",
        help="Mean delay per junction passage across 30 seeds. Corridor vehicles pass "
        "three junctions; north-south vehicles pass one.",
    )
    metric_columns[1].metric(
        f"{label(candidate)} network delay",
        f"{delay['candidate_mean']:.2f} s",
        f"{delay['mean_change']:+.3f} s per passage",
        delta_color="inverse",
        help=f"95% paired bootstrap interval for the change: "
        f"{fmt_interval(delay['ci_lower'], delay['ci_upper'])} s.",
    )
    metric_columns[2].metric(
        "Residual queue change",
        f"{queue['mean_change']:+.1f} veh",
        help=f"Vehicles still queued at 3,600 s, candidate minus benchmark. 95% interval "
        f"{fmt_interval(queue['ci_lower'], queue['ci_upper'], 1)} vehicles.",
    )
    metric_columns[3].metric(
        "Seeds with lower delay",
        f"{delay['seeds_lower']} / {delay['replications']}",
        help="Matched arrival seeds where candidate network delay was strictly lower.",
    )
    st.caption(
        f"Network delay change {delay['mean_change']:+.3f} s per junction passage, 95% paired "
        f"bootstrap interval {fmt_interval(delay['ci_lower'], delay['ci_upper'])} s. "
        "Negative values mean lower candidate delay."
    )

    if CORRIDOR_IMAGE.exists():
        st.image(
            str(CORRIDOR_IMAGE), width="stretch",
            caption="Synthetic corridor: A at 0 m, B at 400 m, C at 900 m. Not a real location.",
        )
    else:
        st.warning("Corridor diagram not found. Run scripts/make_portfolio_figures.py.")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Strategies**")
        for strategy in ("Phase3A", "Phase4E", "Integrated"):
            st.markdown(f"- **{label(strategy)}**: {STRATEGY_DESCRIPTIONS[strategy]}")
    with right:
        st.markdown(
            "**Main finding.** In the evening peak the Integrated plan cut network delay in all 30 "
            "runs, but eastbound drivers waited 2.9 s longer than under the original plan, over "
            "the 2-second limit. A better average does not mean everyone is better off."
        )
        if PHASE4H is not None:
            h = PHASE4H
            st.info(
                f"**The fix: Phase 4H.** The same Junction C search, repeated with fairness "
                f"measured against the original Phase 3A plan and tested on 30 new runs "
                f"(seeds 620 to 649). Network delay {h['net']:+.2f} s per passage, eastbound "
                f"{h['eb']:+.2f} s and westbound {h['wb']:+.2f} s per car: every protected "
                f"direction within the limit. Junction C side roads get {h['ns']:.0f} s of green "
                f"instead of 46 s.",
                icon="✅",
            )

# ------------------------------------------------------------ performance
with performance_tab:
    st.subheader("Network and corridor performance")
    rows = []
    for metric, (name, unit) in NETWORK_METRICS.items():
        result = network_metric_comparison(network, period, benchmark, candidate, metric)
        rows.append(
            {
                "Measure": name,
                "Unit": unit,
                label(benchmark): result["benchmark_mean"],
                label(candidate): result["candidate_mean"],
                "Mean change": result["mean_change"],
                "95% lower": result["ci_lower"],
                "95% upper": result["ci_upper"],
                "Seeds lower": f"{result['seeds_lower']} / {result['replications']}",
            }
        )
    summary = pd.DataFrame(rows)
    st.dataframe(
        summary.round(3), hide_index=True, width="stretch",
        column_config={"Unit": st.column_config.TextColumn(width="medium")},
    )
    st.caption(
        f"{period}, candidate minus benchmark, paired by seed (500 to 529). "
        "Residual queue is measured at the end of the one-hour arrival window."
    )

    metric = st.selectbox(
        "Measure to plot",
        list(NETWORK_METRICS),
        format_func=lambda key: f"{NETWORK_METRICS[key][0]} ({NETWORK_METRICS[key][1]})",
    )
    name, unit = NETWORK_METRICS[metric]

    period_rows = []
    for p in PERIODS:
        r = network_metric_comparison(network, p, benchmark, candidate, metric)
        period_rows.append((p, r))
    across = go.Figure()
    across.add_trace(
        go.Scatter(
            x=[r["mean_change"] for _, r in period_rows],
            y=[p for p, _ in period_rows],
            mode="markers",
            marker=dict(size=12, color=LOWER, line=dict(color="white", width=1.5)),
            error_x=dict(
                type="data", symmetric=False, thickness=2.5, width=0, color=LOWER,
                array=[r["ci_upper"] - r["mean_change"] for _, r in period_rows],
                arrayminus=[r["mean_change"] - r["ci_lower"] for _, r in period_rows],
            ),
            customdata=[
                [r["ci_lower"], r["ci_upper"], r["seeds_lower"]] for _, r in period_rows
            ],
            hovertemplate=(
                "%{y}<br>Mean change: %{x:.3f}<br>95% interval: %{customdata[0]:.3f} to "
                "%{customdata[1]:.3f}<br>Seeds lower: %{customdata[2]}/30<extra></extra>"
            ),
        )
    )
    across.add_vline(x=0, line_color=MUTED, line_width=1)
    base_layout(across, 320, f"{name}: change in every period")
    across.update_xaxes(title=f"{label(candidate)} minus {label(benchmark)} ({unit})")
    across.update_yaxes(categoryorder="array", categoryarray=list(reversed(PERIODS)))
    st.plotly_chart(across, width="stretch", config=PLOTLY_CONFIG, theme=None)

    paired = network_metric_comparison(network, period, benchmark, candidate, metric)["paired"]
    seeds = go.Figure()
    seeds.add_trace(
        go.Bar(
            x=paired["seed"].astype(str),
            y=paired["change"],
            marker_color=[LOWER if c < 0 else (HIGHER if c > 0 else NOT_ASSESSED) for c in paired["change"]],
            customdata=paired[[f"{metric}_benchmark", f"{metric}_candidate"]].to_numpy(),
            hovertemplate=(
                "Seed %{x}<br>Change: %{y:.3f}<br>Benchmark: %{customdata[0]:.3f}"
                "<br>Candidate: %{customdata[1]:.3f}<extra></extra>"
            ),
        )
    )
    seeds.add_hline(y=0, line_color=MUTED, line_width=1)
    seeds.add_hline(
        y=paired["change"].mean(), line_color=INK, line_dash="dot", line_width=1.2,
        annotation_text=f"mean {paired['change'].mean():+.3f}", annotation_position="bottom left",
        annotation_bgcolor="white",
    )
    base_layout(seeds, 340, f"{period}: {name.lower()} change by matched seed")
    seeds.update_xaxes(title="Arrival seed", type="category", tickangle=-90, tickfont=dict(size=10))
    seeds.update_yaxes(title=f"Change ({unit})")
    st.plotly_chart(seeds, width="stretch", config=PLOTLY_CONFIG, theme=None)
    st.caption("Blue bars: candidate lower than benchmark. Orange bars: candidate higher.")

# --------------------------------------------------------------- fairness
with fairness_tab:
    st.subheader(f"{period}: movement-level delay")
    st.markdown(
        "Protected movements (EB, WB, C-N, C-S) pass when the **upper end** of their 95% "
        "paired bootstrap interval is no more than **+2 s per vehicle**. The other four "
        "side-road movements were never given a fairness criterion, so they are shown as "
        "*not assessed* rather than as passing."
    )
    table = fairness.copy()
    colours = [
        FAIL if s == "Fail" else (LOWER if p else NOT_ASSESSED)
        for s, p in zip(table["fairness_check"], table["protected_movement"])
    ]
    names = [
        f"{m} *" if prot else m
        for m, prot in zip(table["movement_label"], table["protected_movement"])
    ]
    figure = go.Figure(
        go.Bar(
            x=table["mean_delay_change_s"],
            y=names,
            orientation="h",
            marker_color=colours,
            error_x=dict(
                type="data", symmetric=False, color=INK, thickness=1.4, width=4,
                array=table["ci_upper_s"] - table["mean_delay_change_s"],
                arrayminus=table["mean_delay_change_s"] - table["ci_lower_s"],
            ),
            customdata=table[["ci_lower_s", "ci_upper_s", "fairness_check"]].to_numpy(),
            hovertemplate=(
                "%{y}<br>Mean change: %{x:.3f} s/vehicle<br>95% interval: "
                "%{customdata[0]:.3f} to %{customdata[1]:.3f}<br>+2 s check: "
                "%{customdata[2]}<extra></extra>"
            ),
        )
    )
    figure.add_vline(x=0, line_color=MUTED, line_width=1)
    figure.add_vline(
        x=FAIRNESS_LIMIT_S, line_dash="dash", line_color=FAIL, line_width=1.5,
        annotation_text="+2 s limit", annotation_position="bottom right",
        annotation_font_color=FAIL,
    )
    base_layout(figure, 440, f"{label(candidate)} minus {label(benchmark)} mean delay")
    figure.update_xaxes(title="Delay change (s per vehicle)")
    figure.update_yaxes(autorange="reversed")
    st.plotly_chart(figure, width="stretch", config=PLOTLY_CONFIG, theme=None)

    with st.expander("Zoom to protected movements near the threshold"):
        near = table.loc[table["protected_movement"]]
        zoom = go.Figure(
            go.Scatter(
                x=near["mean_delay_change_s"], y=near["movement_label"], mode="markers",
                marker=dict(
                    size=12,
                    color=[FAIL if s == "Fail" else LOWER for s in near["fairness_check"]],
                ),
                error_x=dict(
                    type="data", symmetric=False, thickness=2.5, width=0, color=INK,
                    array=near["ci_upper_s"] - near["mean_delay_change_s"],
                    arrayminus=near["mean_delay_change_s"] - near["ci_lower_s"],
                ),
                hovertemplate="%{y}: %{x:.3f} s/vehicle<extra></extra>",
            )
        )
        zoom.add_vline(x=FAIRNESS_LIMIT_S, line_dash="dash", line_color=FAIL)
        zoom.add_vline(x=0, line_color=MUTED)
        base_layout(zoom, 280)
        zoom.update_xaxes(range=[-15, 5], title="Delay change (s per vehicle), axis -15 to +5")
        zoom.update_yaxes(autorange="reversed")
        st.plotly_chart(zoom, width="stretch", config=PLOTLY_CONFIG, theme=None)

    display = table[
        [
            "movement_label",
            "fairness_check",
            "mean_delay_change_s",
            "ci_lower_s",
            "ci_upper_s",
            "benchmark_mean_delay_s",
            "candidate_mean_delay_s",
        ]
    ].rename(
        columns={
            "movement_label": "Movement",
            "benchmark_mean_delay_s": f"{label(benchmark)} (s/veh)",
            "candidate_mean_delay_s": f"{label(candidate)} (s/veh)",
            "mean_delay_change_s": "Mean change (s)",
            "ci_lower_s": "95% lower (s)",
            "ci_upper_s": "95% upper (s)",
            "fairness_check": "+2 s check",
        }
    )
    st.dataframe(display.round(3), hide_index=True, width="stretch")
    st.caption(
        "\\* Protected movement. EB and WB are corridor movements; C-N and C-S are the "
        "northbound and southbound approaches at Junction C. The +2-second rule is an "
        "experimental threshold, not an engineering or legal standard. Each interval is an "
        "individual percentile bootstrap interval (4,000 resamples of 30 paired seed "
        "differences), not a simultaneous guarantee. Negative changes mean lower candidate delay."
    )

# ------------------------------------------------------------ signal timing
def timing_figure(strategy: str) -> go.Figure:
    rows = plan_rows(plans, period, strategy)
    cycle = float(rows["cycle_s"].iloc[0])
    figure = go.Figure()
    shown: set[str] = set()
    colours = {"EW green": EW_COLOUR, "NS green": NS_COLOUR, "Interphase": LOSS_COLOUR}
    for _, row in rows.iterrows():
        junction = row["junction"]
        for seg in phase_windows(row.to_dict()):
            kind = seg["kind"]
            figure.add_trace(
                go.Bar(
                    x=[seg["end_s"] - seg["start_s"]],
                    base=[seg["start_s"]],
                    y=[f"Junction {junction}"],
                    orientation="h",
                    marker=dict(color=colours[kind], line=dict(color="white", width=1.5)),
                    name={"EW green": "EW effective green", "NS green": "NS effective green",
                          "Interphase": "Interphase lost time"}[kind],
                    legendgroup=kind,
                    showlegend=kind not in shown,
                    hovertemplate=(
                        f"Junction {junction} {kind}<br>Window starts {seg['window_start_s']:.0f} s, "
                        f"lasts {seg['duration_s']:.0f} s"
                        + ("<br>Wraps past the cycle boundary" if seg["wraps"] else "")
                        + f"<br>Segment {seg['start_s']:.0f} to {seg['end_s']:.0f} s<extra></extra>"
                    ),
                )
            )
            shown.add(kind)
    base_layout(figure, 260, f"{label(strategy)}: {cycle:.0f} s common cycle")
    figure.update_layout(
        barmode="overlay",
        showlegend=False,
        margin=dict(l=90, r=20, t=50, b=60),
    )
    figure.update_xaxes(range=[0, cycle], title="Seconds within cycle", dtick=25 if cycle > 60 else 10)
    figure.update_yaxes(categoryorder="array", categoryarray=["Junction C", "Junction B", "Junction A"])
    return figure


with timing_tab:
    st.subheader(f"{period}: signal timings")
    st.caption(
        "Each junction runs EW effective green from its EW offset, a 6 s interphase loss, NS "
        "effective green, then a second 6 s loss. Windows that pass the end of the cycle "
        "continue from 0 s. These are analytical effective-green windows, not controller displays."
    )
    st.markdown(
        f'<span style="color:{EW_COLOUR}">■</span> EW effective green &nbsp; '
        f'<span style="color:{NS_COLOUR}">■</span> NS effective green &nbsp; '
        f'<span style="color:#b8b6ae">■</span> 6 s interphase lost time',
        unsafe_allow_html=True,
    )
    for strategy in (benchmark, candidate):
        st.plotly_chart(timing_figure(strategy), width="stretch", config=PLOTLY_CONFIG, theme=None)

    columns = ["cycle_s", "ew_green_s", "ns_green_s", "ew_offset_s", "ns_offset_s"]
    b = plan_rows(plans, period, benchmark).set_index("junction")[columns]
    c = plan_rows(plans, period, candidate).set_index("junction")[columns]
    comparison = pd.concat({label(benchmark): b, label(candidate): c}, axis=1)
    comparison.columns = [
        f"{strategy}: {col.replace('_s', ' (s)').replace('_', ' ').replace('ew', 'EW').replace('ns', 'NS')}"
        for strategy, col in comparison.columns
    ]
    changed = (b != c).any(axis=1)
    comparison.insert(0, "Changed", changed.map({True: "Yes", False: "No"}))
    st.dataframe(comparison.reset_index().rename(columns={"junction": "Junction"}),
                 hide_index=True, width="stretch")
    if identical:
        st.caption("The two strategies use identical timings in this period.")

# --------------------------------------------------------------- methods
with methods_tab:
    st.subheader("How the results were produced")
    st.markdown(
        f"""
**Synthetic demand.** A reproducible synthetic day of 15-minute counts was generated for
eastbound, westbound, northbound and southbound approaches at three junctions, using
smooth AM and PM commuter peaks and seeded Poisson draws. No observed traffic counts are used.

**Webster fixed-time optimisation.** Each 15-minute count is converted to an hourly flow and
divided by an assumed 1,800 veh/h saturation flow. The larger EW and NS ratios give
`Y`, and Webster's formula `C = (1.5L + 5) / (1 - Y)` with 12 s lost time gives a cycle bounded
to 40 to 150 s. Effective green is shared in proportion to the critical ratios.

**Coordination and offsets.** Junctions share a common cycle in each period. Offsets were
first set for 40 km/h eastbound progression (Phase 3A), then searched locally using
simulated delay with directional fairness constraints (Phase 4E). Phase 4F-3B jointly
searched the PM Junction C green split and offset.

**Vehicle-level FIFO queue simulation.** Vehicles arrive by a Poisson process over one
representative hour, travel between junctions at 40 km/h and join unlimited first-in,
first-out queues that discharge at one vehicle every 2 s during effective green.

**Capacity.** With 1,800 veh/h saturation flow, an approach's nominal capacity is
1,800 × green ÷ cycle vehicles per hour. Junction C's north-south capacity shortfall in the
peaks motivated the Phase 4F green-split work.

**Multi-seed paired comparisons.** Phase 4G froze all three strategies, then simulated each
with the same 30 arrival seeds (500 to 529) in each of four representative hours (02:00,
08:00, 12:00, 17:00). Differences are taken seed by seed, with matching vehicle and passage
counts checked for every pair.

**Bootstrap evaluation.** Each 95% interval resamples the 30 paired differences 4,000 times
(percentile method, fixed generator seed 2026). Intervals describe sampling variation under the
synthetic arrival model only. They are individual intervals, not simultaneous guarantees, and
are not evidence of real-world performance.

**Fairness constraint.** EB, WB, C-N and C-S are protected. A strategy passes for a movement when
the upper end of that movement's interval is no more than +{FAIRNESS_LIMIT_S:.0f} s per vehicle.
This threshold is an experimental modelling choice.
        """
    )
    st.subheader("Limitations")
    st.markdown(
        f"""
- The traffic is simulated, not counted on a real road.
- The model is simplified: no turning traffic, pedestrians or queues backing up into the
  previous junction.
- Each period is one representative hour rather than a continuous day.

The full list is in [docs/methodology.md]({REPO_URL}/blob/main/docs/methodology.md#limitations-and-next-steps).
        """
    )
    with st.expander("Saved evidence files"):
        for filename in (
            "phase4g_network_metrics.csv",
            "phase4g_movement_metrics.csv",
            "phase4g_signal_plans.csv",
            "phase4g_network_summary.csv",
            "phase4g_movement_summary.csv",
        ):
            st.code(f"outputs/data/{filename}", language=None)

st.divider()
st.caption(
    f"Source code and full methodology: [GitHub repository]({REPO_URL}). Related project: "
    f"[Liverpool bus reliability panel]({RELATED_URL}), operational public transport "
    "reliability from GTFS and GTFS-RT data."
)
