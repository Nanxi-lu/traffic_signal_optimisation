"""Build the portfolio figures from the committed Phase 4G result files.

Run from the repository root:

    python scripts/make_portfolio_figures.py

Every plotted value is read from ``outputs/data/phase4g_*.csv`` through the
validated dashboard data layer. The script does not run the queue model or
change any saved result. Figures are written to ``figures/``.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "traffic-signals-matplotlib")
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import logging  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
import pandas as pd  # noqa: E402

from traffic_signals.dashboard_data import (  # noqa: E402
    FAIRNESS_LIMIT_S,
    JUNCTION_POSITIONS_M,
    MOVEMENT_LABELS,
    MOVEMENTS,
    PERIODS,
    PROTECTED_MOVEMENTS,
    STRATEGIES,
    STRATEGY_LABELS,
    load_phase4g_data,
    movement_comparison,
    network_comparison,
    phase_windows,
    plan_rows,
)


DATA_DIR = ROOT / "outputs" / "data"
FIGURE_DIR = ROOT / "figures"

# Restrained palette (validated for colour-vision deficiency separation).
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"
PANEL = "#f6f6f3"
STRATEGY_COLOURS = {
    "Phase3A": "#86b6ef",
    "Phase4E": "#3987e5",
    "Integrated": "#184f95",
}
EW_COLOUR = "#2a78d6"
NS_COLOUR = "#eb6834"
LOSS_COLOUR = "#e1e0d9"
LOWER_COLOUR = "#2a78d6"
NOT_ASSESSED = "#b8b6ae"
FAIL = "#d03b3b"
PASS_TEXT = "#006300"

LIGHT_THEME = {
    "road": "#dcdad3", "box": "#cfcdc5", "marking": "white", "housing": "#2b2b2a",
    "ink": INK, "ink_2": INK_2, "ew": EW_COLOUR, "ns": NS_COLOUR,
}
# Dark theme for the LinkedIn summary. Text and marks clear WCAG contrast on
# the #141413 surface (blue 6.2:1, red 5.7:1, secondary ink 10.3:1).
DARK = {
    "surface": "#141413", "panel": "#22221f", "ink": "#f0efec", "ink_2": "#c3c2b7",
    "muted": "#9c9a93", "grid": "#2e2d2a", "axis": "#4a4944", "lower": "#5598e7",
    "fail": "#e66767", "pass": "#5fc35f",
}
DARK_THEME = {
    "road": "#34332f", "box": "#403f3a", "marking": "#6b6a64", "housing": "#050505",
    "ink": DARK["ink"], "ink_2": DARK["ink_2"], "ew": DARK["lower"], "ns": "#eb6834",
}

SCOPE_NOTE = (
    "Synthetic three-junction corridor. Phase 4G audit, 30 matched arrival seeds "
    "(500 to 529) per representative hour. Simulated results, not field observations."
)

plt.rcParams.update(
    {
        "font.family": ["Inter", "DejaVu Sans"],
        "font.size": 11,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titleweight": "semibold",
        "axes.titlesize": 13,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "svg.fonttype": "path",
    }
)


def _save(fig: plt.Figure, name: str, svg: bool = False, dpi: int = 200) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = [FIGURE_DIR / f"{name}.png"]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight", pad_inches=0.25)
    if svg:
        paths.append(FIGURE_DIR / f"{name}.svg")
        fig.savefig(paths[1], bbox_inches="tight", pad_inches=0.25)
        # Matplotlib leaves trailing spaces in SVG path data; strip them so the
        # file passes ``git diff --check``.
        text = paths[1].read_text(encoding="utf-8")
        paths[1].write_text(
            "\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8"
        )
    plt.close(fig)
    return paths


def _header(fig: plt.Figure, title: str, subtitle: str, top: float = 1.0) -> None:
    """Place a title above a subtitle without overlap, above the plot area."""

    height_pt = fig.get_figheight() * 72
    sub_h = 10.5 * 1.5 / height_pt
    fig.text(0.01, top, subtitle, ha="left", va="bottom", fontsize=10.5, color=INK_2)
    fig.text(
        0.01, top + sub_h + 4 / height_pt, title, ha="left", va="bottom",
        fontsize=17, weight="semibold", color=INK,
    )


def _note(fig: plt.Figure, text: str, y: float = 0.01, size: float = 9.5) -> None:
    fig.text(0.01, y, text, ha="left", va="bottom", color=MUTED, fontsize=size, wrap=True)


# ---------------------------------------------------------------- corridor
def _signal_head(ax: plt.Axes, x: float, y: float, scale: float = 1.0,
                 housing: str = "#2b2b2a") -> None:
    w, h = 16 * scale, 40 * scale
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={4 * scale}",
            fc=housing,
            ec="#6b6a64" if housing == "#050505" else "none",
            lw=0.8,
            zorder=6,
        )
    )
    for i, colour in enumerate(("#e34948", "#eda100", "#0ca30c")):
        ax.add_patch(
            plt.Circle(
                (x, y + h / 2 - (i + 0.5) * h / 3),
                4.2 * scale,
                fc=colour,
                ec="none",
                zorder=7,
            )
        )


def draw_corridor(ax: plt.Axes, compact: bool = False, theme: dict | None = None) -> None:
    """Draw the A-B-C corridor schematic in metres along the x-axis."""

    t = theme or LIGHT_THEME
    road = t["road"]
    road_half = 26
    side_len = 120 if not compact else 95
    xs = JUNCTION_POSITIONS_M
    ax.add_patch(
        Rectangle((-140, -road_half), 1180, 2 * road_half, fc=road, ec="none", zorder=1)
    )
    ax.plot([-140, 1040], [0, 0], color=t["marking"], lw=1.6, ls=(0, (8, 7)), zorder=2)
    for name, x in xs.items():
        ax.add_patch(
            Rectangle(
                (x - road_half, -side_len - road_half),
                2 * road_half,
                2 * (side_len + road_half),
                fc=road,
                ec="none",
                zorder=1,
            )
        )
        ax.add_patch(
            Rectangle(
                (x - road_half, -road_half),
                2 * road_half,
                2 * road_half,
                fc=t["box"],
                ec="none",
                zorder=2,
            )
        )
        ax.plot(
            [x, x], [road_half, side_len + road_half], color=t["marking"], lw=1.4,
            ls=(0, (6, 6)), zorder=2,
        )
        ax.plot(
            [x, x], [-road_half, -side_len - road_half], color=t["marking"], lw=1.4,
            ls=(0, (6, 6)), zorder=2,
        )
        # Northbound arrives from the south on the east side; southbound
        # arrives from the north on the west side.
        nb_x, sb_x = x + 13, x - 13
        ax.add_patch(
            FancyArrowPatch(
                (nb_x, -side_len - 10), (nb_x, -road_half - 6),
                arrowstyle="-|>", mutation_scale=14, color=t["ns"], lw=2.2, zorder=4,
            )
        )
        ax.add_patch(
            FancyArrowPatch(
                (sb_x, side_len + 10), (sb_x, road_half + 6),
                arrowstyle="-|>", mutation_scale=14, color=t["ns"], lw=2.2, zorder=4,
            )
        )
        _signal_head(ax, x + road_half + 22, road_half + 30, 0.8 if compact else 1.0, t["housing"])
        label_size = 13 if compact else 14
        ax.text(
            x, side_len + road_half + 28, f"Junction {name}" if compact else f"Junction {name}  ·  {x} m",
            ha="center", va="bottom", fontsize=label_size, weight="semibold", color=t["ink"],
        )
        if not compact:
            ax.text(
                x, -side_len - road_half - 22, f"{name}-S ↓   {name}-N ↑",
                ha="center", va="top", fontsize=10.5, color=t["ink_2"],
            )
    # Corridor movements: EB on the northern half, WB on the southern half
    # (left-hand traffic convention).
    ax.add_patch(
        FancyArrowPatch(
            (-130, 13), (1030, 13), arrowstyle="-|>", mutation_scale=18,
            color=t["ew"], lw=2.6, zorder=4,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (1030, -13), (-130, -13), arrowstyle="-|>", mutation_scale=18,
            color=t["ew"], lw=2.6, zorder=4,
        )
    )
    label_dy = 24 if compact else 13
    ax.text(-150, label_dy, "EB →", ha="right", va="center", fontsize=12, weight="semibold", color=t["ew"])
    ax.text(-150, -label_dy, "← WB", ha="right", va="center", fontsize=12, weight="semibold", color=t["ew"])
    # Distance dimension line.
    dim_y = -side_len - road_half - (75 if not compact else 30)
    for a, b in (("A", "B"), ("B", "C")):
        xa, xb = xs[a], xs[b]
        ax.add_patch(
            FancyArrowPatch(
                (xa, dim_y), (xb, dim_y), arrowstyle="<|-|>", mutation_scale=10,
                color=t["ink_2"], lw=1.1, zorder=3,
            )
        )
        ax.text(
            (xa + xb) / 2, dim_y - 10, f"{xb - xa} m", ha="center", va="top",
            fontsize=12 if compact else 11.5, color=t["ink_2"],
        )
    for name, x in xs.items():
        ax.plot([x, x], [dim_y - 6, dim_y + 6], color=t["ink_2"], lw=1.1)
    ax.set_xlim(-260, 1080)
    ax.set_ylim(dim_y - 45, side_len + road_half + 70)
    ax.set_aspect("equal")
    ax.axis("off")


def make_corridor_figure() -> list[Path]:
    fig, ax = plt.subplots(figsize=(12, 5.2))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.08)
    draw_corridor(ax)
    _header(
        fig,
        "Synthetic three-junction signalised corridor",
        "Fixed-time, two-phase control at every junction: EW green (EB and WB together), "
        "then NS green (northbound and southbound approaches together).",
        top=0.93,
    )
    _note(
        fig,
        "Hypothetical layout, not a real location. EB and WB vehicles travel A to C or C to A at a "
        "constant 40 km/h; north-south vehicles are one-junction local trips. Unlimited FIFO queue storage, "
        "2 s discharge headway.",
        y=0.0,
    )
    return _save(fig, "corridor_network", svg=True)


# ----------------------------------------------------- network performance
def make_network_figure(data: dict[str, pd.DataFrame]) -> list[Path]:
    network = data["network"]
    means = (
        network.groupby(["period", "strategy"])["mean_delay_per_passage_s"].mean().unstack()
    )
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.3}
    )
    fig.subplots_adjust(top=0.84, bottom=0.2)
    width = 0.26
    for i, strategy in enumerate(STRATEGIES):
        values = [means.loc[p, strategy] for p in PERIODS]
        positions = [k + (i - 1) * (width + 0.02) for k in range(len(PERIODS))]
        ax1.bar(
            positions, values, width=width, color=STRATEGY_COLOURS[strategy],
            label=STRATEGY_LABELS[strategy], zorder=3,
        )
        for x, v in zip(positions, values):
            ax1.text(x, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=INK_2)
    ax1.set_xticks(range(len(PERIODS)), PERIODS)
    ax1.set_ylabel("Mean delay (s per junction passage)")
    ax1.yaxis.grid(True, zorder=0)
    ax1.set_ylim(0, max(means.max()) * 1.18)
    ax1.legend(loc="upper left", ncols=3, fontsize=10)
    ax1.set_title("Mean network delay by strategy")
    ax1.tick_params(axis="x", length=0)

    y_positions = list(range(len(PERIODS)))[::-1]
    for yi, period in zip(y_positions, PERIODS):
        r = network_comparison(network, period, "Phase3A", "Integrated")
        ax2.plot([r["ci_lower_s"], r["ci_upper_s"]], [yi, yi], color=LOWER_COLOUR, lw=2.4,
                 zorder=3, solid_capstyle="round")
        ax2.plot(r["mean_delay_change_s"], yi, "o", ms=8, color=LOWER_COLOUR, zorder=4,
                 mec="white", mew=1.5)
        ax2.text(
            r["mean_delay_change_s"], yi + 0.2,
            f"{r['mean_delay_change_s']:+.3f} s, lower in {r['seeds_better']}/{r['replications']} seeds",
            ha="right" if period != "PM peak" else "center", va="bottom", fontsize=9.5, color=INK,
        )
    ax2.axvline(0, color=INK_2, lw=1)
    ax2.set_yticks(y_positions, PERIODS)
    ax2.set_ylim(-0.5, len(PERIODS) - 0.4)
    ax2.set_xlim(-12.5, 1.0)
    ax2.xaxis.grid(True, zorder=0)
    ax2.set_xlabel("Integrated minus Phase 3A (s per junction passage)\nnegative = lower candidate delay")
    ax2.set_title("Paired change, 95% bootstrap interval")
    ax2.tick_params(axis="y", length=0)
    _header(
        fig,
        "Network delay in the Phase 4G audit",
        "Phase 3A: original coordinated baseline.  Phase 4E: fairness-protected offsets.  "
        "Integrated: Phase 4E plus a revised PM Junction C split and offset.",
        top=0.92,
    )
    _note(fig, SCOPE_NOTE + "\nIntervals are individual percentile bootstrap intervals, "
          "not simultaneous guarantees.", y=0.0)
    return _save(fig, "network_performance")


# --------------------------------------------------------------- fairness
def make_fairness_figure(data: dict[str, pd.DataFrame]) -> list[Path]:
    movements = data["movements"]
    vs3a = movement_comparison(movements, "PM peak", "Phase3A", "Integrated")
    vs4e = movement_comparison(movements, "PM peak", "Phase4E", "Integrated").set_index("movement")
    fig, (ax1, axt, ax2) = plt.subplots(
        1, 3, figsize=(13.5, 6.0),
        gridspec_kw={"width_ratios": [1.3, 0.42, 1], "wspace": 0.12},
    )
    fig.subplots_adjust(top=0.82, bottom=0.2, left=0.06, right=0.98)
    order = list(MOVEMENTS)
    table = vs3a.set_index("movement").loc[order]
    y = list(range(len(order)))[::-1]
    for yi, movement in zip(y, order):
        row = table.loc[movement]
        if row["fairness_check"] == "Fail":
            colour = FAIL
        elif row["protected_movement"]:
            colour = LOWER_COLOUR
        else:
            colour = NOT_ASSESSED
        ax1.barh(yi, row["mean_delay_change_s"], height=0.62, color=colour, zorder=3)
        ax1.plot([row["ci_lower_s"], row["ci_upper_s"]], [yi, yi], color=INK, lw=1.2, zorder=4)
        status = row["fairness_check"]
        tag = {"Pass": "✓ pass", "Fail": "✗ FAIL", "Not assessed": "not assessed"}[status]
        tag_colour = {"Pass": PASS_TEXT, "Fail": FAIL, "Not assessed": MUTED}[status]
        axt.text(0.05, yi, f"{row['mean_delay_change_s']:+.3f}", ha="left", va="center",
                 fontsize=10, color=INK,
                 weight="semibold" if row["protected_movement"] else "normal")
        axt.text(0.5, yi, tag, ha="left", va="center", fontsize=10, color=tag_colour,
                 weight="semibold" if status == "Fail" else "normal")
    axt.text(0.05, len(order) - 0.45, "mean (s)", fontsize=9, color=MUTED, va="bottom")
    axt.text(0.5, len(order) - 0.45, "+2 s check", fontsize=9, color=MUTED, va="bottom")
    for ax in (ax1, axt):
        ax.set_ylim(-0.7, len(order) - 0.1)
    axt.set_xlim(0, 1)
    axt.axis("off")
    ax1.axvline(0, color=INK_2, lw=1, zorder=2)
    ax1.axvline(FAIRNESS_LIMIT_S, color=FAIL, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax1.set_yticks(y, [
        MOVEMENT_LABELS[m] + (" *" if m in PROTECTED_MOVEMENTS else "") for m in order
    ])
    ax1.set_xlim(-100, 8)
    ax1.xaxis.grid(True, zorder=0)
    ax1.tick_params(axis="y", length=0)
    ax1.set_xlabel("Integrated minus Phase 3A mean delay\n(s per vehicle; dashed red line = +2 s limit)")
    ax1.set_title("PM peak, every movement vs Phase 3A")

    rows = [
        ("EB vs Phase 3A (original baseline)", table.loc["EB"]),
        ("EB vs Phase 4E (fairness-protected)", vs4e.loc["EB"]),
    ]
    for yi, (label, row) in zip([1, 0], rows):
        ax2.text(-0.2, yi + 0.5, label, ha="left", va="center", fontsize=10.5,
                 color=INK, weight="semibold")
        failed = row["ci_upper_s"] > FAIRNESS_LIMIT_S
        colour = FAIL if failed else LOWER_COLOUR
        ax2.plot([row["ci_lower_s"], row["ci_upper_s"]], [yi, yi], color=colour, lw=3.5,
                 zorder=3, solid_capstyle="round")
        ax2.plot(row["mean_delay_change_s"], yi, "o", ms=10, color=colour, mec="white",
                 mew=1.5, zorder=4)
        ax2.text(
            row["mean_delay_change_s"], yi + 0.18,
            f"{row['mean_delay_change_s']:+.3f} s  [{row['ci_lower_s']:+.3f} to {row['ci_upper_s']:+.3f}]",
            ha="center", va="bottom", fontsize=10, color=INK,
        )
        ax2.text(
            row["mean_delay_change_s"], yi - 0.18,
            "✗ upper bound above +2 s" if failed else "✓ upper bound within +2 s",
            ha="center", va="top", fontsize=10, color=FAIL if failed else PASS_TEXT,
            weight="semibold",
        )
    ax2.axvline(FAIRNESS_LIMIT_S, color=FAIL, lw=1.2, ls=(0, (4, 3)))
    ax2.text(FAIRNESS_LIMIT_S, -0.5, " +2 s limit", color=FAIL, fontsize=9.5, ha="left", va="bottom")
    ax2.axvline(0, color=INK_2, lw=1)
    ax2.set_xlim(-0.3, 4.3)
    ax2.set_ylim(-0.6, 1.75)
    ax2.set_yticks([])
    ax2.spines["left"].set_visible(False)
    ax2.xaxis.grid(True, zorder=0)
    ax2.tick_params(axis="y", length=0)
    ax2.set_xlabel("Change in eastbound mean delay (s per vehicle)")
    ax2.set_title("Eastbound zoom: the benchmark matters")

    _header(
        fig,
        "Lower network delay did not mean lower delay for every movement",
        "* Protected movement. Pass requires the upper end of its 95% paired bootstrap interval "
        "to be no more than +2 s per vehicle (an experimental threshold, not an engineering standard).",
        top=0.9,
    )
    _note(
        fig,
        SCOPE_NOTE + "\nBlack lines and brackets show 95% percentile bootstrap intervals (4,000 resamples "
        "of the 30 paired seed differences). Non-protected movements were not tested against the threshold.",
        y=0.0,
    )
    return _save(fig, "fairness_tradeoff")


# ----------------------------------------------------------- signal timing
def draw_timing(ax: plt.Axes, plans: pd.DataFrame, period: str, strategy: str,
                highlight_c: bool = False) -> None:
    rows = plan_rows(plans, period, strategy)
    cycle = float(rows["cycle_s"].iloc[0])
    for yi, (_, row) in zip([2, 1, 0], rows.iterrows()):
        plan = row.to_dict()
        for seg in phase_windows(plan):
            colour = {"EW green": EW_COLOUR, "NS green": NS_COLOUR, "Interphase": LOSS_COLOUR}[seg["kind"]]
            ax.add_patch(
                Rectangle(
                    (seg["start_s"], yi - 0.3), seg["end_s"] - seg["start_s"], 0.6,
                    fc=colour, ec=SURFACE, lw=1.5, zorder=3,
                )
            )
            if seg["kind"] != "Interphase" and seg["end_s"] - seg["start_s"] >= 14:
                is_start = abs(seg["start_s"] - seg["window_start_s"]) < 1e-9
                text = (
                    f"{seg['kind'][:2]} {seg['duration_s']:.0f} s from {seg['window_start_s']:.0f} s"
                    if is_start else f"{seg['kind'][:2]} continues"
                )
                ax.text(
                    (seg["start_s"] + seg["end_s"]) / 2, yi, text, ha="center", va="center",
                    fontsize=8.5, color="white", weight="semibold", zorder=4,
                )
        if highlight_c and row["junction"] == "C":
            ax.add_patch(
                Rectangle((0.3, yi - 0.42), cycle - 0.6, 0.84, fc="none", ec=INK, lw=1.4,
                          ls=(0, (3, 2)), zorder=5, clip_on=False)
            )
    ax.set_xlim(0, cycle)
    ax.set_ylim(-0.6, 2.6)
    ax.set_yticks([2, 1, 0], ["A", "B", "C"])
    ax.set_xticks(range(0, int(cycle) + 1, 25 if cycle > 60 else 10))
    ax.xaxis.grid(True, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def make_timing_figure(data: dict[str, pd.DataFrame]) -> list[Path]:
    plans = data["plans"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.8), sharex=True)
    fig.subplots_adjust(top=0.82, bottom=0.2, hspace=0.45)
    for ax, strategy in zip(axes, STRATEGIES):
        draw_timing(ax, plans, "PM peak", strategy, highlight_c=strategy == "Integrated")
        ax.set_title(f"{STRATEGY_LABELS[strategy]}", fontsize=13)
    axes[-1].set_xlabel("Time within the common 150 s cycle (s)")
    c = plan_rows(plans, "PM peak", "Integrated").set_index("junction").loc["C"]
    e = plan_rows(plans, "PM peak", "Phase4E").set_index("junction").loc["C"]
    handles = [
        Rectangle((0, 0), 1, 1, fc=EW_COLOUR), Rectangle((0, 0), 1, 1, fc=NS_COLOUR),
        Rectangle((0, 0), 1, 1, fc=LOSS_COLOUR),
    ]
    fig.legend(
        handles, ["EW effective green", "NS effective green", "Interphase lost time (6 s)"],
        loc="upper left", bbox_to_anchor=(0.01, 0.905), ncols=3, fontsize=10,
    )
    _header(
        fig,
        "PM peak signal timings",
        "Offsets locate repeating effective-green windows in the common cycle. A window that "
        "passes 150 s continues from 0 s.",
        top=0.925,
    )
    fig.text(
        0.01, 0.045,
        f"Integrated Junction C (dashed box): cycle {c['cycle_s']:.0f} s, EW green {c['ew_green_s']:.0f} s, "
        f"NS green {c['ns_green_s']:.0f} s, EW offset {c['ew_offset_s']:.0f} s, NS offset "
        f"{c['ns_offset_s']:.0f} s.\nPhase 4E Junction C: EW green {e['ew_green_s']:.0f} s, NS green "
        f"{e['ns_green_s']:.0f} s, EW offset {e['ew_offset_s']:.0f} s, NS offset {e['ns_offset_s']:.0f} s. "
        "Junctions A and B are identical in Phase 4E and Integrated.",
        ha="left", va="bottom", fontsize=10.5, color=INK, linespacing=1.4,
    )
    _note(fig, "Analytical effective-green windows from outputs/data/phase4g_signal_plans.csv, "
          "not a controller display or field timing plan.", y=0.0)
    return _save(fig, "signal_timing_pm")


# --------------------------------------------------------------- LinkedIn
def make_linkedin_figure(data: dict[str, pd.DataFrame]) -> list[Path]:
    """Dark-background 4:5 summary image for a LinkedIn post."""

    d = DARK
    network = data["network"]
    movements = data["movements"]
    delay = network_comparison(network, "PM peak", "Phase3A", "Integrated")
    vs3a = movement_comparison(movements, "PM peak", "Phase3A", "Integrated").set_index("movement")
    fig = plt.figure(figsize=(8, 10), facecolor=d["surface"])
    fig.text(0.06, 0.962, "Traffic signal optimisation  ·  Python simulation study",
             fontsize=12.5, color=d["ink_2"], weight="medium", va="top")
    fig.text(0.06, 0.928, "Network gains can hide\nmovement-level losses", fontsize=27,
             weight="bold", color=d["ink"], va="top", linespacing=1.12)
    fig.text(
        0.06, 0.822,
        "Retiming traffic lights on a simulated three-junction road.\n"
        "Evening peak, same 30 random rush hours for every plan.",
        fontsize=12.5, color=d["ink_2"], va="top", linespacing=1.4,
    )
    ax_map = fig.add_axes([0.05, 0.58, 0.9, 0.165])
    ax_map.set_facecolor(d["surface"])
    draw_corridor(ax_map, compact=True, theme=DARK_THEME)

    def tile(x: float, title: str, value: str, sub: str, colour: str) -> None:
        fig.patches.append(
            FancyBboxPatch((x, 0.375), 0.42, 0.19, boxstyle="round,pad=0,rounding_size=0.015",
                           transform=fig.transFigure, fc=d["panel"], ec="none", zorder=0)
        )
        fig.text(x + 0.025, 0.548, title, fontsize=12.5, color=d["ink_2"], va="top", weight="medium")
        fig.text(x + 0.025, 0.515, value, fontsize=32, color=colour, va="top", weight="bold")
        fig.text(x + 0.025, 0.457, sub, fontsize=10.5, color=d["ink_2"], va="top", linespacing=1.4)

    tile(
        0.06, "Network delay", f"{delay['mean_delay_change_s']:+.1f} s",
        f"per junction passage vs Phase 3A\n95% CI {delay['ci_lower_s']:.1f} to "
        f"{delay['ci_upper_s']:.1f} s\nlower in {delay['seeds_better']} of 30 seeds",
        d["lower"],
    )
    eb = vs3a.loc["EB"]
    tile(
        0.52, "Eastbound delay", f"{eb['mean_delay_change_s']:+.1f} s",
        f"per vehicle vs Phase 3A\n95% CI +{eb['ci_lower_s']:.1f} to +{eb['ci_upper_s']:.1f} s\n"
        "above the +2 s fairness limit",
        d["fail"],
    )

    fig.text(0.06, 0.338, "Protected movements vs Phase 3A (s per vehicle; dashed = +2 s limit)",
             fontsize=12, weight="semibold", color=d["ink"], va="top")
    ax = fig.add_axes([0.15, 0.155, 0.8, 0.155], facecolor=d["surface"])
    order = list(PROTECTED_MOVEMENTS)
    y = list(range(len(order)))[::-1]
    for yi, movement in zip(y, order):
        row = vs3a.loc[movement]
        failed = row["fairness_check"] == "Fail"
        colour = d["fail"] if failed else d["lower"]
        ax.barh(yi, row["mean_delay_change_s"], height=0.62, color=colour, zorder=3)
        ax.plot([row["ci_lower_s"], row["ci_upper_s"]], [yi, yi], color=d["ink"], lw=1.2, zorder=4)
        ax.text(4.5, yi, f"{row['mean_delay_change_s']:+.1f} s", va="center", ha="left",
                fontsize=12, color=d["fail"] if failed else d["ink"], weight="semibold")
    ax.axvline(0, color=d["ink_2"], lw=1)
    ax.axvline(FAIRNESS_LIMIT_S, color=d["fail"], lw=1.2, ls=(0, (4, 3)))
    ax.set_yticks(y, [MOVEMENT_LABELS[m] for m in order], fontsize=12, color=d["ink_2"])
    ax.set_xlim(-100, 18)
    ax.tick_params(axis="x", labelsize=10.5, colors=d["ink_2"])
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=d["grid"], zorder=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(d["axis"])
    h_net = pd.read_csv(DATA_DIR / "phase4h_network_summary.csv").set_index(["benchmark", "candidate"])
    h_gain = -h_net.loc[("Phase3A", "Phase4H"), "mean_delay_change_s"]
    fig.text(
        0.06, 0.112,
        f"The fix: checking fairness against the original plan found a timing that saves\n"
        f"{h_gain:.1f} s with every direction within the limit, even with traffic \u00b120%.",
        fontsize=11.5, color=d["ink"], va="top", linespacing=1.4,
    )
    fig.text(
        0.06, 0.052,
        "Simulated traffic and a simplified queue model.\n"
        "github.com/Nanxi-lu/traffic_signal_optimisation",
        fontsize=10, color=d["muted"], va="top", linespacing=1.45,
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "linkedin_summary.png"
    fig.savefig(path, dpi=150, facecolor=d["surface"])
    plt.close(fig)
    return [path]


def main() -> list[Path]:
    data = load_phase4g_data(DATA_DIR)
    written: list[Path] = []
    written += make_corridor_figure()
    written += make_network_figure(data)
    written += make_fairness_figure(data)
    written += make_timing_figure(data)
    written += make_linkedin_figure(data)
    for path in written:
        print(f"Wrote {path.relative_to(ROOT)}")
    return written


if __name__ == "__main__":
    main()
