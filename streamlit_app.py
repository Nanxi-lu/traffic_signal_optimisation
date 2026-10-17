"""Read-only Streamlit dashboard for the saved synthetic Phase 4G experiment."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from traffic_signals.dashboard_data import (
    FAIRNESS_LIMIT_S,
    PERIODS,
    PROTECTED_MOVEMENTS,
    bootstrap_change,
    green_segments,
    movement_comparison,
    network_comparison,
    pair_results,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "outputs" / "data"
FILES = {
    "network": "phase4g_network_metrics.csv",
    "movements": "phase4g_movement_metrics.csv",
    "plans": "phase4g_signal_plans.csv",
}


st.set_page_config(
    page_title="Traffic Signal Optimisation",
    page_icon="🚦",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {
        background: #f7faf9;
        border: 1px solid #dce7e2;
        border-radius: 0.75rem;
        padding: 0.85rem 1rem;
    }
    [data-testid="stSidebar"] {border-right: 1px solid #e5e7eb;}
    .evidence-note {
        border-left: 4px solid #0f766e;
        background: #f0fdfa;
        padding: 0.8rem 1rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0 1.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_dashboard_data() -> dict[str, pd.DataFrame]:
    """Load the three committed Phase 4G evidence tables."""

    data: dict[str, pd.DataFrame] = {}
    for name, filename in FILES.items():
        path = DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing dashboard input: {path}")
        data[name] = pd.read_csv(path)
    return data


def mean_paired_change(
    frame: pd.DataFrame,
    period: str,
    benchmark: str,
    candidate: str,
    metric: str,
    count_field: str,
) -> tuple[pd.DataFrame, float, float, float]:
    """Pair a network metric and return its mean change and interval."""

    paired = pair_results(
        frame,
        period,
        benchmark,
        candidate,
        metric=metric,
        keys=["seed"],
        count_field=count_field,
    )
    low, high = bootstrap_change(paired["change"])
    return paired, float(paired["change"].mean()), low, high


def plot_signal_plan(plan_rows: pd.DataFrame, title: str) -> go.Figure:
    """Plot effective EW and NS green windows within one common cycle."""

    expected = {"A", "B", "C"}
    if set(plan_rows["junction"]) != expected or len(plan_rows) != 3:
        raise ValueError("A signal plan must contain one row for A, B and C.")
    cycles = plan_rows["cycle_s"].unique()
    if len(cycles) != 1:
        raise ValueError("Signal plan does not have one common cycle.")
    cycle = float(cycles[0])
    figure = go.Figure()
    shown_legend: set[str] = set()
    labels: list[str] = []

    for junction in ("A", "B", "C"):
        row = plan_rows.loc[plan_rows["junction"] == junction].iloc[0]
        phases = (
            ("EW", row["ew_offset_s"], row["ew_green_s"], "#0f766e"),
            ("NS", row["ns_offset_s"], row["ns_green_s"], "#2563eb"),
        )
        for phase, offset, duration, colour in phases:
            label = f"{junction}  ·  {phase}"
            labels.append(label)
            for start, end in green_segments(offset, duration, cycle):
                figure.add_trace(
                    go.Bar(
                        x=[end - start],
                        base=[start],
                        y=[label],
                        orientation="h",
                        name=f"{phase} effective green",
                        legendgroup=phase,
                        showlegend=phase not in shown_legend,
                        marker_color=colour,
                        hovertemplate=(
                            f"{junction} {phase}<br>Start: {start:.1f}s"
                            f"<br>End: {end:.1f}s<extra></extra>"
                        ),
                    )
                )
            shown_legend.add(phase)

    figure.update_layout(
        title=title,
        barmode="overlay",
        height=430,
        margin=dict(l=15, r=15, t=55, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=1.12, x=0),
        xaxis=dict(
            title=f"Seconds in {cycle:.0f}-second cycle",
            range=[0, cycle],
            gridcolor="#e5e7eb",
        ),
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(labels))),
    )
    return figure


try:
    dashboard_data = load_dashboard_data()
except (FileNotFoundError, pd.errors.ParserError) as error:
    st.error(str(error))
    st.stop()

network = dashboard_data["network"]
movements = dashboard_data["movements"]
plans = dashboard_data["plans"]

st.title("Traffic Signal Optimisation")
st.caption(
    "Synthetic three-junction corridor · 30 matched seeds per representative "
    "hour · saved Phase 4G results"
)
st.markdown(
    '<div class="evidence-note"><strong>Read-only evidence viewer.</strong> '
    "Selections recalculate paired summaries from committed CSV records; they "
    "do not run a simulation or optimise a signal plan.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Experiment selection")
    period = st.selectbox("Operating period", PERIODS, index=3)
    benchmark = st.selectbox("Benchmark", ("Phase3A", "Phase4E"), index=0)
    candidate = st.selectbox(
        "Candidate strategy", ("Phase4E", "Integrated"), index=1
    )
    st.divider()
    st.caption(
        "Negative changes mean the candidate produced a lower value in the "
        "saved synthetic experiment."
    )

overview_tab, fairness_tab, timing_tab, methods_tab = st.tabs(
    ("Overview", "Movement Fairness", "Signal Timing", "Methodology")
)

with overview_tab:
    st.subheader(f"{period}: network performance")
    delay = network_comparison(network, period, benchmark, candidate)
    queue_pairs, queue_change, queue_low, queue_high = mean_paired_change(
        network,
        period,
        benchmark,
        candidate,
        "total_queue_at_hour",
        "total_passages",
    )
    corridor_pairs, corridor_change, corridor_low, corridor_high = mean_paired_change(
        network,
        period,
        benchmark,
        candidate,
        "corridor_mean_delay_s",
        "total_passages",
    )
    ns_pairs, ns_change, ns_low, ns_high = mean_paired_change(
        network,
        period,
        benchmark,
        candidate,
        "ns_mean_delay_s",
        "total_passages",
    )
    selected_candidate = network.loc[
        (network["period"] == period) & (network["strategy"] == candidate)
    ]

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Candidate network delay",
        f"{delay['candidate_mean_delay_s']:.2f} s/passage",
        f"{delay['mean_delay_change_s']:+.2f} s",
        delta_color="inverse",
    )
    metric_columns[1].metric(
        "Candidate residual queue",
        f"{selected_candidate['total_queue_at_hour'].mean():.1f} vehicles",
        f"{queue_change:+.1f} vehicles",
        delta_color="inverse",
    )
    metric_columns[2].metric(
        "Corridor delay change",
        f"{corridor_change:+.2f} s/vehicle",
        help=f"95% paired bootstrap interval: {corridor_low:.2f} to {corridor_high:.2f} s.",
        delta_color="off",
    )
    metric_columns[3].metric(
        "North–south delay change",
        f"{ns_change:+.2f} s/vehicle",
        help=f"95% paired bootstrap interval: {ns_low:.2f} to {ns_high:.2f} s.",
        delta_color="off",
    )

    st.caption(
        f"Candidate minus benchmark network delay: "
        f"{delay['mean_delay_change_s']:+.3f} s/passage "
        f"(paired bootstrap 95% interval {delay['ci_lower_s']:+.3f} to "
        f"{delay['ci_upper_s']:+.3f}; {delay['replications']} matched seeds). "
        f"Residual-queue interval: {queue_low:+.2f} to {queue_high:+.2f} vehicles."
    )

    change_pairs = pair_results(
        network,
        period,
        benchmark,
        candidate,
        "mean_delay_per_passage_s",
        ["seed"],
        "total_passages",
    )
    overview_figure = go.Figure()
    overview_figure.add_trace(
        go.Box(
            y=change_pairs["change"],
            name=f"{candidate} − {benchmark}",
            boxpoints="all",
            jitter=0.35,
            pointpos=0,
            marker=dict(color="#0f766e", size=6, opacity=0.65),
            line_color="#0f766e",
            hovertemplate="Seed change: %{y:.3f} s/passage<extra></extra>",
        )
    )
    overview_figure.add_hline(y=0, line_dash="dash", line_color="#64748b")
    overview_figure.update_layout(
        title="Paired network-delay changes across saved seeds",
        yaxis_title="Candidate minus benchmark (seconds per junction passage)",
        xaxis_title="",
        height=430,
        showlegend=False,
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=55, b=20),
    )
    st.plotly_chart(overview_figure, width="stretch")

with fairness_tab:
    st.subheader(f"{period}: movement-level delay comparison")
    comparison = movement_comparison(
        movements, period, benchmark, candidate
    ).copy()
    colours = [
        "#0f766e" if protected else "#94a3b8"
        for protected in comparison["protected_movement"]
    ]
    fairness_figure = go.Figure(
        go.Bar(
            x=comparison["mean_delay_change_s"],
            y=comparison["movement"],
            orientation="h",
            marker_color=colours,
            error_x=dict(
                type="data",
                symmetric=False,
                array=comparison["ci_upper_s"] - comparison["mean_delay_change_s"],
                arrayminus=comparison["mean_delay_change_s"] - comparison["ci_lower_s"],
            ),
            hovertemplate=(
                "%{y}<br>Mean change: %{x:.3f}s/vehicle<extra></extra>"
            ),
        )
    )
    fairness_figure.add_vline(x=0, line_dash="dash", line_color="#64748b")
    fairness_figure.add_vline(
        x=FAIRNESS_LIMIT_S,
        line_dash="dot",
        line_color="#dc2626",
        annotation_text="+2 s upper-bound limit",
        annotation_position="top right",
    )
    fairness_figure.update_layout(
        title="Candidate minus benchmark mean movement delay",
        xaxis_title="Delay change (seconds per vehicle)",
        yaxis_title="Movement",
        height=510,
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=60, b=20),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fairness_figure, width="stretch")

    protected = comparison.loc[
        comparison["movement"].isin(PROTECTED_MOVEMENTS)
    ].copy()
    protected["criterion"] = protected["passes_2s_upper"].map(
        {True: "Pass", False: "Fail"}
    )
    failures = protected.loc[~protected["passes_2s_upper"], "movement"].tolist()
    if failures:
        st.warning(
            "Fairness check failed for: "
            + ", ".join(failures)
            + ". Its paired bootstrap upper bound exceeds +2 seconds per vehicle."
        )
    else:
        st.success(
            "All four protected movement checks pass for this selected comparison."
        )

    display = protected[
        [
            "movement",
            "benchmark_mean_delay_s",
            "candidate_mean_delay_s",
            "mean_delay_change_s",
            "ci_lower_s",
            "ci_upper_s",
            "criterion",
        ]
    ].rename(
        columns={
            "movement": "Movement",
            "benchmark_mean_delay_s": "Benchmark delay (s)",
            "candidate_mean_delay_s": "Candidate delay (s)",
            "mean_delay_change_s": "Mean change (s)",
            "ci_lower_s": "95% lower (s)",
            "ci_upper_s": "95% upper (s)",
            "criterion": "+2 s upper-bound check",
        }
    )
    numeric_columns = display.select_dtypes(include="number").columns
    display[numeric_columns] = display[numeric_columns].round(3)
    st.dataframe(display, hide_index=True, width="stretch")
    st.caption(
        "EB, WB, C_N and C_S are the protected movements. The +2-second rule is "
        "an experimental movement-delay threshold, not an established engineering "
        "or legal fairness standard. Error bars are individual paired-bootstrap "
        "intervals, not a simultaneous guarantee."
    )

with timing_tab:
    st.subheader(f"{period}: effective-green timing")
    st.caption(
        "Offsets locate repeating effective-green windows within the common cycle. "
        "These are analytical phase windows, not a controller display or field plan."
    )
    benchmark_plan = plans.loc[
        (plans["period"] == period) & (plans["strategy"] == benchmark)
    ]
    candidate_plan = plans.loc[
        (plans["period"] == period) & (plans["strategy"] == candidate)
    ]
    timing_columns = st.columns(2)
    with timing_columns[0]:
        st.plotly_chart(
            plot_signal_plan(benchmark_plan, benchmark), width="stretch"
        )
    with timing_columns[1]:
        st.plotly_chart(
            plot_signal_plan(candidate_plan, candidate), width="stretch"
        )

    timing_table = pd.concat(
        [
            benchmark_plan.assign(role="Benchmark"),
            candidate_plan.assign(role="Candidate"),
        ],
        ignore_index=True,
    )[
        [
            "role",
            "strategy",
            "junction",
            "cycle_s",
            "ew_green_s",
            "ns_green_s",
            "ew_offset_s",
            "ns_offset_s",
        ]
    ].rename(
        columns={
            "role": "Role",
            "strategy": "Strategy",
            "junction": "Junction",
            "cycle_s": "Cycle (s)",
            "ew_green_s": "EW green (s)",
            "ns_green_s": "NS green (s)",
            "ew_offset_s": "EW offset (s)",
            "ns_offset_s": "NS offset (s)",
        }
    )
    st.dataframe(timing_table, hide_index=True, width="stretch")

with methods_tab:
    st.subheader("Research methodology")
    st.markdown(
        """
        ### Study design

        This educational prototype models a synthetic corridor with three
        intersections and four representative one-hour operating periods. The saved
        experiment combines Webster-based fixed timing, common-cycle coordination,
        bidirectional offset optimisation, a simplified FIFO queue simulation,
        competing EW/NS effective-green phases and movement-level constraints.

        ### Evaluation boundary

        Phase 4G froze the three strategies before evaluating each period with 30
        matched synthetic arrival seeds (500–529). The dashboard pairs records by
        exact seed and reports candidate-minus-benchmark changes. All generated
        vehicles are followed through the simplified model; residual queues are
        measured at the end of the one-hour arrival-generation window. The interface
        reads those saved records only—it does not run optimisation or simulation.

        ### Fairness check

        EB, WB, C_N and C_S use an experimental rule: the upper bound of each
        movement's individual 95% paired-bootstrap interval must be no more than
        +2 seconds per vehicle. Passing this rule is not a field, legal or
        simultaneous statistical guarantee.

        ### Main limitations

        - All demand and results are synthetic and uncalibrated against observed data.
        - The model uses simplified FIFO discharge, constant travel assumptions and
          unlimited queue storage; it omits spillback, finite lanes, turning, lane
          changing, acceleration, actuated control and timing-plan transitions.
        - Four independent representative hours are not a continuous full-day model.
        - Bootstrap intervals describe variation under the saved synthetic arrival
          process, not uncertainty about real traffic or model validity.
        - Lower simulated delay is not a claim of real-world traffic improvement.
        """
    )
    with st.expander("Saved evidence files"):
        for filename in FILES.values():
            st.code(f"outputs/data/{filename}", language=None)
