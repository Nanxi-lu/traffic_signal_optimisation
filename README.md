# Traffic Signal Optimisation

## Python-Based Traffic Signal Coordination and Network Performance Evaluation

This is an independent Python project investigating how fixed-time traffic signal
coordination, green allocation and offset optimisation can influence congestion,
vehicle delay and network performance.

It models a synthetic three-junction corridor and evaluates interventions using
vehicle-level queue simulation, matched stochastic experiments and directional
fairness constraints.

> **Synthetic and experimental.** All traffic demand is generated, the model is not
> field calibrated, and no timing plan here has been deployed. Results describe the
> behaviour of a simplified simulation, not observed traffic.

![Synthetic three-junction corridor](figures/corridor_network.png)

**In one sentence:** in the simulated PM peak, the integrated timing strategy cut mean
network delay by 8.6 seconds per junction passage against the original coordinated
baseline, but eastbound delay rose by 2.9 seconds per vehicle, which exceeds the
project's predefined +2-second fairness threshold.

---

## Contents

- [A. Project overview](#a-project-overview)
- [B. Main findings](#b-main-findings)
- [C. Interactive dashboard](#c-interactive-dashboard)
- [D. Network visualisation](#d-network-visualisation)
- [E. Methodology](#e-methodology)
- [F. Results and limitations](#f-results-and-limitations)
- [G. Reproducibility](#g-reproducibility)
- [H. Related project](#h-related-project)
- [Project structure](#project-structure)

---

## A. Project overview

**The transport problem.** On a signalised urban corridor, every junction divides
green time between competing movements. Poorly coordinated junctions make vehicles
stop repeatedly, lengthen queues and waste capacity. Changing a junction's timing
moves delay between movements, so improving one direction can make another worse.

**Why signal coordination matters.** When neighbouring junctions share a common cycle
and their green windows are offset by the travel time between them, a platoon of
vehicles can pass several junctions with fewer stops (a "green wave"). Offsets that
suit one direction can penalise the opposite direction and the side roads, so
coordination is always a trade-off.

**What the model does.**

- Generates a reproducible synthetic day of 15-minute counts for every approach.
- Calculates Webster fixed-time cycles and green splits, then common-cycle plans.
- Optimises offsets for progression and then for simulated delay.
- Simulates individual vehicles through first-in, first-out (FIFO) queues at all three
  junctions, including competing north-south traffic.
- Compares strategies on the same 30 random arrival streams and reports paired
  bootstrap intervals.
- Checks every protected movement against a predefined fairness threshold.

**Decisions it can help investigate.**

- How much network delay can offset and green-split changes remove in each period?
- Which movements gain and which lose when the network total improves?
- Does a fairness verdict depend on which baseline you compare against?
- Where is junction capacity, rather than coordination, the binding constraint?

---

## B. Main findings

All values come from the saved Phase 4G audit (`outputs/data/phase4g_*.csv`). The audit
froze three strategies and then simulated each one with the same 30 arrival seeds
(500 to 529) in four representative hours.

| Strategy | Definition |
|---|---|
| **Phase 3A** | Original coordinated baseline: common Webster-based cycle, fixed green splits and design-speed eastbound offsets. |
| **Phase 4E** | Fairness-protected offsets selected separately for each period. Cycles and green splits unchanged. |
| **Integrated** | Phase 4E, with only the PM peak Junction C green split and offsets replaced by the Phase 4F-3B joint selection. |

### Network delay: Integrated minus Phase 3A

Mean delay per junction passage. Negative values mean lower delay with the Integrated
strategy.

| Period | Phase 3A (s) | Integrated (s) | Change (s) | 95% bootstrap interval (s) | Seeds with lower delay | Residual queue change (veh) |
|---|---:|---:|---:|---|---:|---:|
| Overnight | 6.77 | 6.58 | **-0.186** | -0.229 to -0.142 | 29 / 30 | +0.43 |
| AM peak | 33.56 | 32.13 | **-1.434** | -1.560 to -1.300 | 30 / 30 | -5.00 |
| Interpeak | 6.81 | 6.64 | **-0.166** | -0.186 to -0.144 | 29 / 30 | 0.00 |
| PM peak | 57.99 | 49.38 | **-8.611** | -9.976 to -7.284 | 30 / 30 | -27.43 |

In the Overnight, AM peak and Interpeak hours, the Integrated plans are identical to
Phase 4E. Only the PM peak differs. Against Phase 4E, the PM Integrated strategy
changes network delay by -7.522 s per passage (95% interval -8.897 to -6.165).

![Network delay by strategy and period](figures/network_performance.png)

### The fairness trade-off

> **Network-wide delay reductions do not necessarily imply improvements for every
> traffic movement.** The integrated PM timing strategy reduced simulated network delay
> but exceeded the predefined eastbound fairness threshold against the original baseline.

PM peak, Integrated minus Phase 3A, mean delay per vehicle for the four protected
movements:

| Movement | Change (s/veh) | 95% bootstrap interval (s/veh) | +2 s upper-bound check |
|---|---:|---|---|
| EB (eastbound corridor) | **+2.906** | +2.732 to +3.084 | **Fail** |
| WB (westbound corridor) | -9.376 | -9.788 to -8.908 | Pass |
| C-N (northbound at C) | -11.540 | -14.698 to -8.942 | Pass |
| C-S (southbound at C) | -82.519 | -97.832 to -67.151 | Pass |

The **benchmark matters**. Against Phase 4E, the same PM Integrated candidate changes
eastbound delay by +1.620 s per vehicle (95% interval +1.569 to +1.673) and **passes**.
It passes against the fairness-protected Phase 4E plan, but not against the original
Phase 3A baseline, because the Phase 4E offsets had already added some eastbound delay.

![PM peak movement delay changes against the fairness threshold](figures/fairness_tradeoff.png)

**A second, marginal exceedance.** In the AM peak, the Integrated plan (identical to
Phase 4E) changes Junction C northbound (C-N) delay by +0.919 s per vehicle against
Phase 3A. Its upper bound is +2.197 s, just above the +2 s limit. The Phase 4E AM
offsets were selected before north-south traffic was modelled, so C-N was not part of
that selection. Every other protected movement passes in every period against both
benchmarks.

**How to read the threshold.** A protected movement passes when the upper end of its 95%
paired bootstrap interval is no more than +2 s per vehicle. The +2 s value is an
experimental modelling choice, not an engineering or legal standard. The four
side-road movements at A and B were never given a fairness criterion and are reported as
"not assessed".

---

## C. Interactive dashboard

**Project website:** [nanxi-lu.github.io/traffic_signal_optimisation](https://nanxi-lu.github.io/traffic_signal_optimisation/)
is a static, read-only version of the results explorer, hosted on GitHub Pages. It loads
quickly on a phone and needs no Python. Its data file, `docs/assets/phase4g_results.json`,
is exported from the saved CSVs by `scripts/export_site_data.py` using the same validated
functions as the dashboard.

A read-only Streamlit explorer lets you choose a period, benchmark and candidate and see
the saved paired results: network, corridor and north-south delay, residual queues,
seed-level changes, movement fairness and signal timings. Every value is recalculated
from the committed CSVs. The app never reruns the simulation or optimisation.

**Streamlit app:** not yet deployed. Run it locally:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open the address Streamlit prints (normally `http://localhost:8501`). The default
view is the PM peak, Integrated against Phase 3A, with the eastbound fairness failure
shown at the top of every tab. Deployment notes are in
[docs/deployment.md](docs/deployment.md).

---

## D. Network visualisation

| Item | Value |
|---|---|
| Junctions | A at 0 m, B at 400 m, C at 900 m |
| Corridor movements | EB (A to C) and WB (C to A), constant 40 km/h |
| Side-road movements | Northbound and southbound approaches at each junction, one-junction local trips |
| Control | Fixed-time, two phases per junction: EW green, then NS green |
| Lost time | 12 s per cycle, modelled as two 6 s interphase gaps |
| Discharge | One vehicle every 2 s from each queue during effective green |

The PM peak timings use a common 150 s cycle. The Integrated strategy changes only
Junction C: EW green 89 s, NS green 49 s, EW offset 83 s and NS offset 28 s (Phase 4E
used 92 s, 46 s, 79 s and 27 s). Because 83 + 89 = 172 s, Junction C's EW green window
wraps past the end of the cycle and continues from 0 s to 22 s.

![PM peak signal timings for the three strategies](figures/signal_timing_pm.png)

---

## E. Methodology

The full phase-by-phase record is preserved in
[docs/methodology.md](docs/methodology.md). In summary:

1. **Demand segmentation.** A seeded synthetic day of 15-minute counts (three junctions,
   four approaches) with AM and PM commuter peaks is divided into four operating periods:
   Overnight, AM peak, Interpeak and PM peak.
2. **Webster timing.** Counts are converted to hourly flows and divided by an assumed
   1,800 veh/h saturation flow. The critical EW and NS flow ratios give `Y`, and Webster's
   `C = (1.5L + 5) / (1 - Y)` with `L = 12 s` gives a cycle bounded to 40 to 150 s. Green
   is allocated in proportion to the critical ratios.
3. **Coordination.** In each period all junctions share the longest of their independently
   optimised cycles, and Phase 3A offsets
   align EW green with 40 km/h eastbound travel times.
4. **Offset optimisation.** Offsets were then searched for speed robustness, bidirectional
   progression and finally simulated delay (Phases 3D to 4E). The delay searches used a
   +2 s directional fairness constraint, with separate training and untouched holdout seeds.
5. **Queue simulation.** Vehicles arrive by a Poisson process, travel at 40 km/h and join
   unlimited FIFO queues with 2 s discharge headways. North-south traffic competes for a
   mutually exclusive NS phase (Phase 4F).
6. **Green allocation.** Junction C was capacity-constrained in the peaks, so Phase
   4F-3A tested moving EW green to NS, and Phase 4F-3B jointly searched Junction C's split
   and offset with training and holdout seeds.
7. **Fairness evaluation.** Phase 4G froze all strategies and audited them on 30 fresh
   matched seeds per period, pairing results by seed and movement and checking four
   protected movements against the +2 s upper-bound rule.

---

## F. Results and limitations

**Trade-offs.** The largest network gain is in the PM peak, where the revised Junction C
split gives the heavily delayed north-south approaches more green. That benefit is
partly paid for by eastbound corridor traffic. Lower simulated delay can also coexist
with a higher residual queue: overnight, the mean residual queue rose by 0.43 vehicles
even though delay fell.

**Statistical uncertainty.** Each interval is a 95% percentile bootstrap interval from
4,000 resamples of the 30 paired seed differences. The intervals quantify sampling
variation under the synthetic arrival model only. They are individual intervals, not
simultaneous guarantees across periods and movements, and they are not evidence of
real-world performance.

**What these results do not show.** They do not show that any plan has been deployed, that
the model reflects a real corridor, that every movement benefits, or that any strategy is
globally optimal. All searches were local.

**Main model limitations.**

- Synthetic rather than observed traffic counts.
- Unlimited queue storage, so no queue spillback or junction blocking.
- Simplified two-phase signal control with no actuation or plan transitions.
- Simplified vehicle movement and discharge (constant speed, fixed headway).
- No detailed turning movements, lane groups or pedestrian phases.
- No field calibration or validation.
- Four representative hours rather than a continuous full-day simulation.

The phase-by-phase limitations are listed in
[docs/methodology.md](docs/methodology.md#limitations-and-next-steps).

---

## G. Reproducibility

Requires Python 3.10 or later.

```bash
git clone https://github.com/Nanxi-lu/traffic_signal_optimisation.git
cd traffic_signal_optimisation
python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

| Task | Command |
|---|---|
| Run the test suite | `python -m pytest -v` |
| Launch the dashboard | `streamlit run streamlit_app.py` |
| Rebuild the portfolio figures from saved CSVs | `python scripts/make_portfolio_figures.py` |
| Rebuild the website data from saved CSVs | `python scripts/export_site_data.py` |
| Preview the website locally | `python -m http.server --directory docs` |
| List the research phases | `python -m workflows --list` |
| Regenerate the Phase 1 synthetic data | `python -m traffic_signals.demo` |
| Rerun one phase, for example Phase 3A | `python -m workflows phase3a` |
| Rerun every phase in order (overwrites `outputs/`) | `python -m workflows all` |
| Rerun the Phase 4G audit (overwrites the saved Phase 4G outputs) | `python -m workflows phase4g` |

`requirements.txt` holds only the dashboard runtime packages; `requirements-dev.txt`
adds matplotlib and pytest for the research scripts and tests. All results needed by the
dashboard and figures are committed, so neither requires rerunning any experiment. The
regression tests in `tests/test_phase5b_portfolio.py` pin the headline Phase 4G values
and check that the dashboard reproduces the saved summary tables exactly.

---

## H. Related project

[**Liverpool bus real-time panel**](https://nanxi-lu.github.io/liverpool_bus_realtime_panel/)
([code](https://github.com/Nanxi-lu/liverpool_bus_realtime_panel)) uses GTFS schedules and
real-time vehicle location data to measure how Route 10A journeys compare with their
timetabled runtimes. It shows when journeys ran late, but its data cannot separate junction
delay from other causes such as boarding, roadworks or dispatching.

This project looks at one of those causes from the other direction: it simulates how
fixed-time signal coordination distributes delay across movements, on a synthetic corridor.
The two are complementary: one measures observed performance, the other reasons about a
mechanism in a controlled setting. Both use paired comparisons and report results by time
period. This project uses no real-time or observed data and makes no claim about bus
performance in Liverpool or anywhere else.

---

## Project structure

```text
traffic_signal_optimisation/
├── README.md                     Portfolio overview (this page)
├── streamlit_app.py              Read-only results dashboard
├── requirements.txt              Dashboard runtime dependencies
├── requirements-dev.txt          Research, figure and test dependencies
├── traffic_signals/              Core library: demand, Webster, coordination,
│   │                             queue simulation, network evaluation
│   └── dashboard_data.py         Validated loading, pairing and bootstrap for the dashboard
├── workflows/                    Research workflow, one module per stage
│   ├── stage1_2_timing.py        Phases 1 and 2: demand checks and fixed timing plans
│   ├── stage3_coordination.py    Phase 3: coordination and offset optimisation
│   ├── stage4a_4e_corridor.py    Phases 4A to 4E: corridor vehicle simulation
│   └── stage4f_4g_full_junction.py Phases 4F and 4G: full junction and integrated audit
├── scripts/
│   ├── make_portfolio_figures.py Builds figures/ from the saved Phase 4G CSVs
│   └── export_site_data.py       Exports saved comparisons for the GitHub Pages site
├── outputs/
│   ├── data/                     Saved CSV results for every phase
│   └── figures/                  Research figures produced by the phase scripts
├── figures/                      Portfolio figures used in this README
├── docs/
│   ├── index.html                GitHub Pages site (static results explorer)
│   ├── assets/                   Site data exported from the saved CSVs
│   ├── methodology.md            Full Phase 1 to Phase 5A technical record
│   ├── deployment.md             GitHub Pages and Streamlit Community Cloud guide
│   └── linkedin_post_draft.md    Draft post and recommended image
├── tests/                        pytest suite, including dashboard regression tests
└── .streamlit/config.toml        Dashboard theme and server settings
```
