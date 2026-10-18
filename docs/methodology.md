# Methodology and development record

This document preserves the full Phase 1 to Phase 5A technical record that previously formed
the main README. The [README](../README.md) now gives a short case-study overview; this page
is the reference for how every saved output was produced.

All data and results are **synthetic**. The repository contains matched simulated
comparisons, but no real-world traffic performance improvement is claimed.

## Contents

1. [Scope of the completed work](#scope)
2. [Running each phase](#running-each-phase)
3. [Data and units](#data-and-units)
4. [Webster method](#webster-method)
5. [Phase 4G evidence and the dashboard data layer](#phase-4g-evidence-and-the-dashboard-data-layer)
6. [Saved outputs](#outputs)
7. [Limitations by phase](#limitations-and-next-steps)
8. [Proposed extensions](#proposed-extensions)

## Scope

The completed work includes:

- reproducible 15-minute synthetic counts for eastbound, westbound,
  northbound and southbound approaches;
- transparent two-phase EW/NS Webster cycle and green-split calculations;
- rule-based time-of-day periods and grid-searched fixed timing plans;
- initial common-cycle coordination and ideal eastbound offsets;
- design-speed time-space diagrams and numerical bandwidth verification;
- fixed-offset sensitivity checks at five vehicle speeds;
- exhaustive integer-second offset optimisation for robust progression;
- bidirectional evaluation of original and eastbound-optimised offsets;
- demand-weighted bidirectional optimisation with design-speed protection;
- matched AM-peak vehicle and FIFO queue simulation;
- four-period, 30-seed matched simulation with bootstrap delay intervals;
- congestion stress testing and delay-aware AM offset optimisation;
- robust directional-fairness screening with untouched holdout seeds;
- fairness-protected offset selection across four operating periods;
- an initial full-junction experiment with competing north-south traffic;
- a four-period, 30-seed full-junction robustness evaluation;
- Junction C green-split sensitivity for the AM and PM peaks;
- fairness-constrained joint Junction C split-and-offset optimisation with
  untouched holdout seeds;
- a frozen four-period integrated-strategy audit against both Phase 3A and
  Phase 4E;
- a read-only Streamlit dashboard for paired network, movement-fairness and
  signal-timing comparisons; and
- CSV outputs, figures and automated tests.

It deliberately does **not** yet implement a field-calibrated or microscopic
traffic simulation, stochastic speeds, finite queue storage, machine learning,
reinforcement learning or interactive re-optimisation.

## Running each phase

### Environment

From the repository root, create an isolated environment and install the full
research and test dependency set (`requirements.txt` alone contains only the
dashboard runtime packages):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

### Phase runners

The phase runners live in four stage modules in `workflows/`
(`stage1_2_timing`, `stage3_coordination`, `stage4a_4e_corridor` and
`stage4f_4g_full_junction`). Each section of a module keeps the functions of one phase,
and each former script's `main()` is now `run_<phase>()`. Run any phase from the
repository root with `python -m workflows <phase>`; `python -m workflows --list` shows them
all and `python -m workflows all` runs the whole chain in order. Every phase reads earlier
outputs and overwrites its own files in `outputs/`.

```bash
python -m traffic_signals.demo
```

The default run uses seed 42, 12 seconds of lost time, and cycle bounds of
40-150 seconds. These can be changed without editing source code:

```bash
python -m traffic_signals.demo \
  --seed 42 \
  --lost-time 12 \
  --min-cycle 40 \
  --max-cycle 150 \
  --output-dir outputs
```

Run the tests with:

```bash
python -m pytest
```

Launch the Phase 5A dashboard with:

```bash
streamlit run streamlit_app.py
```

The dashboard opens at the local address printed by Streamlit (normally
`http://localhost:8501`). It reads the committed Phase 4G CSV files and does not
run optimisation or generate new simulation results. Its default view is the
PM peak with Phase 3A as benchmark and Integrated as candidate. Use the sidebar
to compare any saved period, Phase 3A or Phase 4E benchmark, and Phase 4E or
Integrated candidate.

For deployment, connect this GitHub repository to Streamlit Community Cloud,
select the `main` branch and set `streamlit_app.py` as the entry point. The
required `outputs/data/phase4g_*.csv` files and dependencies must remain in the
repository. GitHub Pages can link to the deployed app but cannot run this Python
Streamlit application itself.

After running the Phase 1 demo, generate the Phase 2 time-of-day summary with:

```bash
python -m workflows phase2_periods
```

The initial illustrative periods are Overnight (20:00-06:00), AM peak
(06:00-10:00), Interpeak (10:00-16:00), and PM peak (16:00-20:00).

Select and evaluate the exploratory p85 fixed timing plans with:

```bash
python -m workflows phase2_plans
```

This rounds each period's 85th-percentile Webster cycle upward to the next five
seconds, within the 40-150 second bounds. It is an exploratory heuristic, not
an established optimal timing rule.

Design and evaluate fixed EW/NS effective-green allocations with:

```bash
python -m workflows phase2_splits
```

This uses each phase's period-level 85th-percentile critical flow ratio to
allocate the available effective green proportionally, subject to an
illustrative 10-second minimum for each phase. It then screens every interval
against the fixed phase allocations. The minimum is a prototype assumption,
not a field-deployment safety standard.

Optimise each period's cycle and EW/NS split by transparent grid search with:

```bash
python -m workflows phase2_optimise
```

The search considers 40-150 second cycles in five-second increments and
one-second EW green increments, while preserving 12 seconds of lost time and at
least 10 seconds for each phase. It prioritises fewer phase warnings, smaller
total shortfall, shorter cycles, and finally a greater minimum capacity margin.

Create the initial common-cycle eastbound coordination plans with:

```bash
python -m workflows phase3a
```

This selects the longest independently optimised cycle in each period, assumes
400 m from A to B, 500 m from B to C and a constant 40 km/h progression speed,
then aligns EW green starts to the resulting 0, 36 and 81 second cumulative
travel times. The resulting bandwidth is theoretical and assumes exact travel
times with no queues or disturbances.

Generate and numerically verify the four design-speed time-space diagrams with:

```bash
python -m workflows phase3b
```

Evaluate the unchanged Phase 3A plans at 30, 35, 40, 45 and 50 km/h with:

```bash
python -m workflows phase3c
```

For each period and speed, this samples every whole-second departure within one
signal cycle and reports the fraction that encounters effective EW green at A,
B and C. This is a theoretical progression-window measure, not an estimate of
the percentage of real vehicles completing the corridor without stopping.

Optimise the B and C offsets for equal-weighted robustness across the five
tested speeds with:

```bash
python -m workflows phase3d
```

The search keeps A at zero and checks every integer B/C offset pair. It first
maximises mean progression across 30, 35, 40, 45 and 50 km/h, then worst-speed
progression, then 40 km/h progression, then proximity to the Phase 3A offsets.
These are exploratory priorities rather than a measured vehicle-speed
distribution.

Evaluate the original and Phase 3D offsets for both eastbound and westbound
progression with:

```bash
python -m workflows phase3e
```

This applies the same whole-second arrival-window calculation in the A-B-C
and C-B-A directions at all five test speeds. The combined summary weights
the two directions and five speeds equally; it is not weighted by traffic
demand. The runner also verifies that all eastbound values reproduce the Phase
3D results before writing the Phase 3E outputs.

Optimise offsets using synthetic directional-demand proxies with:

```bash
python -m workflows phase3f
```

The Phase 3F search derives period weights from eastbound counts entering at A
and westbound counts entering at C. It keeps A at zero, preserves every Phase
3A cycle and green allocation, and prevents either direction's 40 km/h
progression from falling more than five percentage points below its Phase 3A
value. The five-point threshold is an experimental modelling choice, not a
traffic engineering standard.

Compare the Phase 3A and Phase 3F AM offsets using the same generated vehicles:

```bash
python -m workflows phase4a
```

Phase 4A estimates hourly entry demand from the 16 synthetic AM observations at
A eastbound and C westbound, draws Poisson vehicle counts with seed 42, and
distributes entry times uniformly over one hour. Vehicles travel at 40 km/h and
join unlimited FIFO queues with a two-second discharge headway. The AM timing
plan continues after the one-hour generation window until every vehicle exits.
`waiting_events` counts junctions with positive waiting time; it is not a count
of confirmed physical stops.

Repeat the matched comparison for all four periods and seeds 42-71 with:

```bash
python -m workflows phase4b
```

Phase 4B simulates a representative hour for each period and calculates paired
Phase 3F-minus-Phase 3A delay differences. Its 95% bootstrap intervals resample
the 30 complete paired-seed differences. They quantify variation under the
synthetic arrival process and fixed model assumptions, not uncertainty about
real traffic conditions or model validity.

Test demand and congestion sensitivity at 75%, 100%, 125% and 150% of the
baseline demand with:

```bash
python -m workflows phase4c
```

Phase 4C preserves each Phase 4B baseline arrival stream at 100% demand. Lower
demand retains a nested subset of those vehicles, while higher demand adds
successive independent Poisson arrivals. Both signal strategies receive the
same vehicles. The simulation records queues still waiting at each approach at
3,600 seconds, then continues until every generated vehicle exits. Its
`clearance_tail_s` includes normal corridor travel time as well as congestion
delay and should not be interpreted entirely as delay.

Select and validate one provisional AM delay-aware offset plan with:

```bash
python -m workflows phase4d1
```

Phase 4D-1 keeps the Phase 3A cycle and green allocations fixed. It uses only
seeds 42-44 to conduct a 10-second coarse offset search followed by one-second
refinement around selected candidates. Candidate directional mean delay may be
no more than two seconds above its Phase 3A training baseline. Seeds 45-71 are
reserved for a single unseen-sample comparison with Phase 3A and Phase 3F. The
search is not exhaustive, and its result is a provisional AM candidate rather
than a final or globally optimal timing plan.

Stress-test the frozen AM Phase 4D candidate with fresh seeds using:

```bash
python -m workflows phase4d2
```

Phase 4D-2 does not search for new offsets. It compares the fixed 0/32/68-second
plan with Phase 3A and Phase 3F using seeds 72-101 at 75%, 100%, 125% and 150%
of baseline AM demand. All strategies receive identical nested arrival streams.
Directional fairness is reported at three distinct levels against Phase 3A:
the mean change, the upper bootstrap bound and the number of individual seeds
whose delay increase is no greater than two seconds.

Select and holdout-test a more robustly fair AM plan with:

```bash
python -m workflows phase4d3
```

Phase 4D-3 keeps the 150-second cycle and Phase 3A green allocations fixed and
searches a local B/C offset grid around earlier strategies. A relaxed four-seed
screen reduces the candidate pool. Final training uses seeds 102-113 at 75%,
100%, 125% and 150% demand. A plan is feasible only when the upper endpoint of
each of eight directional bootstrap intervals is no greater than two seconds
relative to Phase 3A and combined mean delay does not increase at any factor.
The selected offsets are frozen before comparison with Phase 3A, Phase 3F and
Phase 4D on untouched seeds 114-143. The four demand scenarios are weighted
equally during selection; their probabilities in real traffic are unknown.

Build and holdout-test the four-period Phase 4E strategy with:

```bash
python -m workflows phase4e
```

Phase 4E retains the frozen 0/36/70-second AM plan and independently searches
local B/C offset neighbourhoods around Phase 3A and Phase 3F for Overnight,
Interpeak and PM peak. It reuses the Phase 4D-3 relaxed four-seed screen: at
100% and 150% demand, directional mean delay changes must be no greater than
three seconds and combined delay must not increase. Full training then uses
seeds 144-155 and all four demand factors with the two-second bootstrap-upper-
endpoint constraint. The four selected plans are frozen before validation on
seeds 156-185. Signal cycles and effective green allocations remain unchanged.

Compare the AM Phase 3A and Phase 4E offsets with competing north-south traffic:

```bash
python -m workflows phase4f1
```

Phase 4F-1 uses the synthetic 08:00-09:00 counts without averaging them over
the wider AM period. Each stored 15-minute count is the mean of a separate
Poisson arrival process, and both signal strategies receive identical generated
vehicles. The existing east-west corridor simulation is unchanged. Northbound
and southbound vehicles are local one-junction trips served by a mutually
exclusive NS effective-green phase. Both phases use independent FIFO queues,
two-second discharge headways and unlimited storage. The 12 seconds of total
lost time are represented as two illustrative six-second interphase gaps.

Repeat the full-junction comparison across four representative hours and 30
fresh seeds with:

```bash
python -m workflows phase4f2
```

Phase 4F-2 evaluates Overnight at 02:00-03:00, AM peak at 08:00-09:00,
Interpeak at 12:00-13:00 and PM peak at 17:00-18:00. It preserves each hour's
four synthetic 15-minute profiles and compares the fixed Phase 3A and Phase 4E
plans using matched streams for seeds 200-229. Results are reported separately
for eight movements and for network delay per junction passage. The four hours
are representative snapshots and are not combined into a daily average.

Test Junction C green allocation while holding the Phase 4E offsets fixed with:

```bash
python -m workflows phase4f3a
```

Phase 4F-3A transfers 0, 3, 5, 7 or 10 seconds of effective green from EW to
NS at Junction C in the representative AM and PM peak hours. Plans at A and B,
the 150-second common cycle, 12 seconds of lost time and C's EW offset remain
unchanged. Because the phases are consecutive, transferring green also moves
C's NS phase start earlier. Each scenario uses the same eight arrival streams
for seeds 300-311. This is a synthetic sensitivity experiment, not selection
or validation of a new operational timing plan. The earlier two-second
directional fairness threshold is reported only as a reference here, not
enforced as a constraint.

Jointly search Junction C's green split and EW offset with:

```bash
python -m workflows phase4f3b
```

Phase 4F-3B keeps A and B unchanged and searches 35 Junction C candidates: the
five Phase 4F-3A green transfers crossed with offsets from 12 seconds before to
12 seconds after the Phase 4E offset in four-second increments. Seeds 400-403
screen the grid, seeds 400-411 enforce the training criteria, and untouched
seeds 420-449 compare the frozen Phase 4E, split-only and joint plans. A plan
is training-feasible only if network mean delay per passage does not increase
and the upper endpoint of each EB, WB, C-N and C-S delay-change bootstrap
interval is no greater than two seconds. Phase 4E is always retained as the
fallback. This is a local synthetic search, not a global or field-validated
optimum.

Audit the frozen integrated strategy against Phase 3A and Phase 4E with:

```bash
python -m workflows phase4g
```

Phase 4G performs no optimisation. It uses the four Phase 4E plans, replacing
only PM Junction C with the Phase 4F-3B 89/49-second EW/NS split and 83-second
EW offset. The audit compares Phase 3A, Phase 4E and this integrated strategy
over matched seeds 500-529 in the representative 02:00, 08:00, 12:00 and
17:00 hours. Network and movement changes are calculated directly against each
benchmark rather than adding results from experiments with different seeds.
The non-PM integrated plans are required to reproduce Phase 4E exactly.

## Data and units

`SyntheticDataConfig` in `traffic_signals/synthetic_data.py` holds the main
assumptions. The default synthetic day starts at midnight on 15 January 2025
and contains 96 intervals at each of three intersections, with four approaches
per intersection (1,152 rows). Configurable inputs include:

- random seed and start time;
- intersection names and demand scale factors;
- base hourly demand assumptions;
- AM and PM peak additions, centre times and peak width; and
- approach saturation flow (default 1,800 vehicles/hour).

The peak profiles are illustrative commuter patterns: eastbound demand is
higher in the AM, westbound demand is higher in the PM, and north/south demand
has smaller peaks. Smooth Gaussian profiles set an expected hourly rate, then
a seeded Poisson draw creates integer 15-minute counts.

The stored schema is:

| Column | Meaning and unit |
|---|---|
| `time` | Start of the 15-minute interval |
| `intersection_id` | Synthetic intersection A, B or C |
| `approach` | eastbound, westbound, northbound or southbound |
| `volume_15min` | Vehicle count observed in that 15-minute interval |
| `saturation_flow_vph` | Assumed approach saturation flow in vehicles/hour |

The analysis explicitly converts `volume_15min` to an equivalent hourly flow
by multiplying by four before comparing it with `saturation_flow_vph`. Counts
and flow rates are not treated as the same unit.

## Webster method

For this teaching model, eastbound and westbound movements run concurrently in
one phase, while northbound and southbound movements run concurrently in the
other. For each interval:

1. Convert each 15-minute count to an equivalent hourly flow rate.
2. Divide flow by saturation flow to obtain each approach flow ratio.
3. Select the larger of eastbound/westbound as the EW critical ratio, and the
   larger of northbound/southbound as the NS critical ratio. Opposing flows are
   not added because they run concurrently in this simplified model.
4. Sum the two critical ratios to obtain `Y` and calculate:

   `C = (1.5L + 5) / (1 - Y)`

5. Bound the selected cycle to 40-150 seconds and allocate effective green
   (`C - L`) in proportion to the two critical ratios.

The illustrative default lost time `L` is 12 seconds per cycle. Invalid or
negative inputs raise clear errors. Zero demand returns the minimum cycle and
zero demand-based green allocations. `Y >= 1` is flagged as oversaturated. A
maximum-cycle result is also flagged as insufficient when `Y` exceeds the
fraction of that capped cycle available as effective green. These flags are
diagnostics, not proof of field performance.

## Phase 4G evidence and the dashboard data layer

Phase 4G is the authoritative evidence set for the portfolio figures and the dashboard.
`traffic_signals/dashboard_data.py` loads it and applies these checks before any value
is displayed:

- every period, strategy, seed (500 to 529) and movement is present exactly once;
- movement passages, vehicles, residual queues and total delay add up to every
  network record, with three junction passages per corridor vehicle and one per
  north-south vehicle;
- signal plans satisfy `EW green + NS green + lost time = cycle`, use one common
  cycle per strategy and period, and place NS green one 6-second interphase loss after
  EW green ends;
- strategy comparisons are paired on exact seed (and movement), and paired vehicle or
  passage counts must match;
- benchmark and candidate must be different strategies.

Bootstrap intervals use the same scheme as `workflows.stage4a_4e_corridor.bootstrap_mean_ci`
(4,000 percentile resamples, generator seed 2026). A regression test confirms that
the dashboard's recalculated means, intervals, seed counts and fairness verdicts
match `phase4g_network_summary.csv` and `phase4g_movement_summary.csv` exactly.

The fairness verdict is only assigned to the four protected movements (EB, WB, C_N,
C_S). Other movements are reported as "Not assessed". Two exceedances of the
+2-second upper-bound rule are present in the saved Phase 4G data, both against
Phase 3A:

| Period | Movement | Mean change (s/veh) | 95% interval (s/veh) |
|---|---|---:|---|
| PM peak | EB | +2.906 | +2.732 to +3.084 |
| AM peak | C_N | +0.919 | -0.287 to +2.197 |

The AM Integrated plan is identical to Phase 4E. Its offsets were selected in Phase
4E, before north-south traffic was modelled, so C_N was not a constraint in that
selection.

## Outputs

The workflow writes:

- `outputs/data/synthetic_traffic_15min.csv`
- `outputs/data/synthetic_webster_results.csv`
- `outputs/data/phase2_period_summary.csv` (after running `python -m workflows phase2_periods`)
- `outputs/data/phase2_fixed_plans.csv` (after running `python -m workflows phase2_plans`)
- `outputs/data/phase2_interval_comparison.csv` (after running `python -m workflows phase2_plans`)
- `outputs/data/phase2_plan_evaluation.csv` (after running `python -m workflows phase2_plans`)
- `outputs/data/phase2_green_split_plans.csv` (after running `python -m workflows phase2_splits`)
- `outputs/data/phase2_green_split_evaluation.csv` (after running `python -m workflows phase2_splits`)
- `outputs/data/phase2_green_split_summary.csv` (after running `python -m workflows phase2_splits`)
- `outputs/data/phase2_optimised_plans.csv` (after running `python -m workflows phase2_optimise`)
- `outputs/data/phase2_optimised_interval_evaluation.csv` (after running `python -m workflows phase2_optimise`)
- `outputs/data/phase3a_common_cycles.csv` (after running `python -m workflows phase3a`)
- `outputs/data/phase3a_signal_plans.csv` (after running `python -m workflows phase3a`)
- `outputs/data/phase3a_bandwidth.csv` (after running `python -m workflows phase3a`)
- `outputs/data/phase3b_bandwidth_verification.csv` (after running `python -m workflows phase3b`)
- `outputs/data/phase3c_speed_sensitivity.csv` (after running `python -m workflows phase3c`)
- `outputs/data/phase3d_optimised_offsets.csv` (after running `python -m workflows phase3d`)
- `outputs/data/phase3d_robustness_comparison.csv` (after running `python -m workflows phase3d`)
- `outputs/data/phase3e_bidirectional_evaluation.csv` (after running `python -m workflows phase3e`)
- `outputs/data/phase3e_direction_summary.csv` (after running `python -m workflows phase3e`)
- `outputs/data/phase3e_combined_summary.csv` (after running `python -m workflows phase3e`)
- `outputs/data/phase3f_direction_weights.csv` (after running `python -m workflows phase3f`)
- `outputs/data/phase3f_optimisation_summary.csv` (after running `python -m workflows phase3f`)
- `outputs/data/phase3f_speed_direction_comparison.csv` (after running `python -m workflows phase3f`)
- `outputs/data/phase4a_summary.csv` (after running `python -m workflows phase4a`)
- `outputs/data/phase4a_vehicle_journeys.csv` (after running `python -m workflows phase4a`)
- `outputs/data/phase4a_queue_summary.csv` (after running `python -m workflows phase4a`)
- `outputs/data/phase4a_combined_comparison.csv` (after running `python -m workflows phase4a`)
- `outputs/data/phase4b_replication_metrics.csv` (after running `python -m workflows phase4b`)
- `outputs/data/phase4b_paired_differences.csv` (after running `python -m workflows phase4b`)
- `outputs/data/phase4b_uncertainty_summary.csv` (after running `python -m workflows phase4b`)
- `outputs/data/phase4c_replication_metrics.csv` (after running `python -m workflows phase4c`)
- `outputs/data/phase4c_approach_queues.csv` (after running `python -m workflows phase4c`)
- `outputs/data/phase4c_paired_differences.csv` (after running `python -m workflows phase4c`)
- `outputs/data/phase4c_sensitivity_summary.csv` (after running `python -m workflows phase4c`)
- `outputs/data/phase4d_selected_plan.csv` (after running `python -m workflows phase4d1`)
- `outputs/data/phase4d_training_metrics.csv` (after running `python -m workflows phase4d1`)
- `outputs/data/phase4d_validation_metrics.csv` (after running `python -m workflows phase4d1`)
- `outputs/data/phase4d_validation_summary.csv` (after running `python -m workflows phase4d1`)
- `outputs/data/phase4d2_replication_metrics.csv` (after running `python -m workflows phase4d2`)
- `outputs/data/phase4d2_paired_differences.csv` (after running `python -m workflows phase4d2`)
- `outputs/data/phase4d2_stress_summary.csv` (after running `python -m workflows phase4d2`)
- `outputs/data/phase4d3_screening.csv` (after running `python -m workflows phase4d3`)
- `outputs/data/phase4d3_training_candidates.csv` (after running `python -m workflows phase4d3`)
- `outputs/data/phase4d3_selected_plan.csv` (after running `python -m workflows phase4d3`)
- `outputs/data/phase4d3_holdout_metrics.csv` (after running `python -m workflows phase4d3`)
- `outputs/data/phase4d3_holdout_summary.csv` (after running `python -m workflows phase4d3`)
- `outputs/data/phase4e_screening.csv` (after running `python -m workflows phase4e`)
- `outputs/data/phase4e_training_candidates.csv` (after running `python -m workflows phase4e`)
- `outputs/data/phase4e_selected_plans.csv` (after running `python -m workflows phase4e`)
- `outputs/data/phase4e_holdout_metrics.csv` (after running `python -m workflows phase4e`)
- `outputs/data/phase4e_holdout_summary.csv` (after running `python -m workflows phase4e`)
- `outputs/data/phase4f1_corridor_results.csv` (after running `python -m workflows phase4f1`)
- `outputs/data/phase4f1_ns_approaches.csv` (after running `python -m workflows phase4f1`)
- `outputs/data/phase4f1_network_summary.csv` (after running `python -m workflows phase4f1`)
- `outputs/data/phase4f1_signal_windows.csv` (after running `python -m workflows phase4f1`)
- `outputs/data/phase4f1_c_ns_capacity.csv` (after running `python -m workflows phase4f1`)
- `outputs/data/phase4f2_movement_metrics.csv` (after running `python -m workflows phase4f2`)
- `outputs/data/phase4f2_network_metrics.csv` (after running `python -m workflows phase4f2`)
- `outputs/data/phase4f2_paired_movements.csv` (after running `python -m workflows phase4f2`)
- `outputs/data/phase4f2_movement_summary.csv` (after running `python -m workflows phase4f2`)
- `outputs/data/phase4f2_paired_network.csv` (after running `python -m workflows phase4f2`)
- `outputs/data/phase4f2_network_summary.csv` (after running `python -m workflows phase4f2`)
- `outputs/data/phase4f2_c_capacity.csv` (after running `python -m workflows phase4f2`)
- `outputs/data/phase4f3a_movement_metrics.csv` (after running `python -m workflows phase4f3a`)
- `outputs/data/phase4f3a_network_metrics.csv` (after running `python -m workflows phase4f3a`)
- `outputs/data/phase4f3a_capacity_scenarios.csv` (after running `python -m workflows phase4f3a`)
- `outputs/data/phase4f3a_paired_network.csv` (after running `python -m workflows phase4f3a`)
- `outputs/data/phase4f3a_network_summary.csv` (after running `python -m workflows phase4f3a`)
- `outputs/data/phase4f3a_paired_movements.csv` (after running `python -m workflows phase4f3a`)
- `outputs/data/phase4f3a_movement_summary.csv` (after running `python -m workflows phase4f3a`)
- `outputs/data/phase4f3b_screening.csv` (after running `python -m workflows phase4f3b`)
- `outputs/data/phase4f3b_training_candidates.csv` (after running `python -m workflows phase4f3b`)
- `outputs/data/phase4f3b_training_fairness.csv` (after running `python -m workflows phase4f3b`)
- `outputs/data/phase4f3b_selected_plans.csv` (after running `python -m workflows phase4f3b`)
- `outputs/data/phase4f3b_holdout_movements.csv` (after running `python -m workflows phase4f3b`)
- `outputs/data/phase4f3b_holdout_network.csv` (after running `python -m workflows phase4f3b`)
- `outputs/data/phase4f3b_holdout_network_summary.csv` (after running `python -m workflows phase4f3b`)
- `outputs/data/phase4f3b_holdout_movement_summary.csv` (after running `python -m workflows phase4f3b`)
- `outputs/data/phase4g_signal_plans.csv` (after running `python -m workflows phase4g`)
- `outputs/data/phase4g_network_metrics.csv` (after running `python -m workflows phase4g`)
- `outputs/data/phase4g_movement_metrics.csv` (after running `python -m workflows phase4g`)
- `outputs/data/phase4g_network_summary.csv` (after running `python -m workflows phase4g`)
- `outputs/data/phase4g_movement_summary.csv` (after running `python -m workflows phase4g`)
- `outputs/data/phase4g_fairness_audit.csv` (after running `python -m workflows phase4g`)
- `outputs/figures/synthetic_demand_profile.png`
- `outputs/figures/synthetic_webster_cycles.png`
- `outputs/figures/phase3b_time_space_overnight.png`
- `outputs/figures/phase3b_time_space_am_peak.png`
- `outputs/figures/phase3b_time_space_interpeak.png`
- `outputs/figures/phase3b_time_space_pm_peak.png`
- `outputs/figures/phase3c_speed_sensitivity.png`
- `outputs/figures/phase3d_robustness_overnight.png`
- `outputs/figures/phase3d_robustness_am_peak.png`
- `outputs/figures/phase3d_robustness_interpeak.png`
- `outputs/figures/phase3d_robustness_pm_peak.png`
- `outputs/figures/phase3e_bidirectional_overnight.png`
- `outputs/figures/phase3e_bidirectional_am_peak.png`
- `outputs/figures/phase3e_bidirectional_interpeak.png`
- `outputs/figures/phase3e_bidirectional_pm_peak.png`
- `outputs/figures/phase3f_strategy_comparison.png`
- `outputs/figures/phase4b_delay_comparison.png`
- `outputs/figures/phase4c_demand_sensitivity_overnight.png`
- `outputs/figures/phase4c_demand_sensitivity_am_peak.png`
- `outputs/figures/phase4c_demand_sensitivity_interpeak.png`
- `outputs/figures/phase4c_demand_sensitivity_pm_peak.png`
- `outputs/figures/phase4d2_congestion_comparison.png`
- `outputs/figures/phase4d2_eastbound_fairness.png`
- `outputs/figures/phase4e_delay_overnight.png`
- `outputs/figures/phase4e_delay_am_peak.png`
- `outputs/figures/phase4e_delay_interpeak.png`
- `outputs/figures/phase4e_delay_pm_peak.png`
- `outputs/figures/phase4f2_network_delay.png`
- `outputs/figures/phase4f3a_network_delay_am_peak.png`
- `outputs/figures/phase4f3a_network_delay_pm_peak.png`
- `outputs/figures/phase4f3a_c_ns_queue_am_peak.png`
- `outputs/figures/phase4f3a_c_ns_queue_pm_peak.png`
- `outputs/figures/phase4f3b_holdout_am_peak.png`
- `outputs/figures/phase4f3b_holdout_pm_peak.png`
- `outputs/figures/phase4g_integrated_network.png`

Both figure titles identify the data/model as synthetic.

Portfolio figures are generated separately from the saved Phase 4G CSVs:

```bash
python scripts/make_portfolio_figures.py
```

This writes `figures/corridor_network.png` and `.svg`, `figures/network_performance.png`,
`figures/fairness_tradeoff.png`, `figures/signal_timing_pm.png` and
`figures/linkedin_summary.png`.

## Limitations and next steps

The model has no turning movements, lane groups, pedestrian timings, finite
queue storage, arrival dispersion, actuated control, intergreen detail or field
calibration.
Converting one 15-minute count to an hourly equivalent is an analysis convention
and is not a rolling-hour measurement. The current time-of-day boundaries are
manually defined starting assumptions, not data-derived or optimised boundaries.
The fixed cycles and green splits are capacity-based prototypes; phase-capacity
shortfalls are dimensionless screening results, not vehicle counts, queues or
delays. The aggregate-capacity lower bound is three warnings. The one-second
grid search found four warnings, but the 144.33-second conflicting-demand
calculation does not prove four as the minimum over every possible continuous
green split. Coordination assumes ideal constant-speed travel and effective
green windows, not displayed indications, stochastic vehicle trajectories or
queue interactions. The speed sensitivity analysis changes one constant speed
at a time while holding signal timings fixed; it is not a traffic simulation.
The offset search treats the five speeds equally and evaluates each operating
period independently. It does not model how controllers transition between
plans, and changing an offset changes the clock time at which green occurs. A
bidirectional evaluation now reports both directions but does not optimise them
jointly. Phase 3F adds joint demand-weighted optimisation, but the entry-approach
weights are synthetic proxies rather than observed corridor origin-destination
flows or a calibrated speed distribution. Phase 4A adds simplified vehicle and
queue movement for one AM hour and one random seed. It excludes turning, lane
changing, acceleration, stochastic speeds, opposing-queue interaction,
spillback, finite storage and timing-plan transitions. Its stopped-delay proxy
and waiting events are simulation measures, not field-observed performance.
Phase 4B extends the experiment to all periods and 30 seeds, but each period is
still represented by a standalone one-hour rate. It does not simulate a full
day, transitions between timing plans, or independent real-world validation.
Phase 4C varies synthetic east-west through demand while preserving the same
fixed timings. Its unlimited queues cannot spill back, block upstream junctions
or represent finite lane storage. Residual queues and delay changes are model
outputs, not evidence of real-world congestion or traffic improvement. The
Phase 4A-4E corridor comparisons exclude north-south demand and shared phase
constraints; Phase 4F-1 adds those movements only for its separate AM experiment.
Phase 4D-1 optimises one AM plan against three synthetic training seeds. Its
two-second directional fairness tolerance is an experimental constraint, and
validation on the remaining synthetic seeds is not field validation. Phase
4D-2 stress-tests only the AM candidate with additional synthetic seeds and
elevated demand; the offsets have not been applied to the other operating
periods. It does not change the unlimited-storage queue model, add spillback or
validate the timing plan against observed traffic. Phase 4D-3 strengthens the
training constraint and uses new synthetic holdout seeds, but it remains a
local offset search for the AM period. Its eight individual 95% bootstrap
checks are not a simultaneous 95% guarantee, and passing them does not establish
fairness for every possible arrival pattern or for field traffic. Phase 4E
extends the same method to four separately simulated representative hours; it
does not model transitions between timing plans or optimise the fixed cycles and
green splits. Its local candidate search is not proof of global optimality. In
the synthetic holdout results, lower simulated delay can coexist with a higher
residual queue, particularly when demand is high; unlimited queue storage means
these results do not represent spillback or network blocking. Phase 4F-1 adds
local north-south approaches for one synthetic AM hour and one random seed, but
it does not represent east-west vehicles entering or leaving at intermediate
junctions. It does not optimise green allocations, model turning movements or
finite storage. Phase 4F-2 adds four representative hours and 30 seeds per
period but is not a continuous full-day simulation. Its intervals describe
variation under the synthetic Poisson arrival process, not field uncertainty or
a simultaneous guarantee across periods and movements. Fixed green allocations
mean the diagnosed Junction C capacity exceedances are not resolved by this
offset-only comparison.
Phase 4F-3A varies only Junction C's green split over 12 synthetic arrival
seeds in two representative hours. It does not jointly optimise the split and
offset, use untouched holdout seeds, or establish an operational plan. Its
nominal capacities assume 1,800 vehicles/hour saturation flow per approach,
and equality between demand and nominal capacity provides no reserve. The
movement and queue trade-offs remain consequences of the simplified FIFO,
unlimited-storage model and are not evidence of real-world performance.
Phase 4F-3B searches only 35 local Junction C combinations in two representative
hours. Its training and holdout seeds are independent of one another but still
come from the same synthetic demand model. The four movement bootstrap
intervals are individual intervals rather than simultaneous guarantees, and
their two-second upper-bound limit is an experimental choice. Holdout results
do not establish performance under different demand assumptions, every vehicle
trajectory, full-day timing transitions or field conditions.
Phase 4G freezes the integrated plan before using fresh seeds, but it remains a
synthetic audit of four representative hours rather than continuous operation
or field validation. The PM integrated candidate passes the two-second
movement-level bootstrap-upper-bound checks against Phase 4E but fails the
eastbound check against Phase 3A. It must therefore be described as
network-beneficial within this model with a cumulative directional fairness
trade-off, not as fairness-protected against the original baseline. No plan was
retuned after inspecting the Phase 4G results.
Phase 5A only visualises the committed Phase 4G records. Its selectors recompute
paired summaries and uncertainty intervals from those records; they do not
create a new scenario, run the queue model or validate a signal plan.

## Proposed extensions

These are possible future work items. None is implemented.

- **Exploratory simulator page.** A separate, clearly labelled dashboard page could let a
  user enter a timing plan and run the existing queue model on a small number of seeds.
  Its results would need to be visually distinct from the validated Phase 4G evidence,
  labelled as unvalidated, and never written back to `outputs/data`.
- **Finite queue storage and spillback** between junctions.
- **Observed demand.** Replace synthetic counts with observed counts (for example from
  automatic traffic counters) and calibrate saturation flow and discharge headways.
- **Turning movements, lane groups and pedestrian phases.**
- **Continuous full-day simulation** with timing-plan transitions between periods.
- **Simultaneous inference** across movements and periods (for example a
  Bonferroni-adjusted or joint bootstrap check) rather than individual intervals.
