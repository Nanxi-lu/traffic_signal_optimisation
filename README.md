# Traffic Signal Optimisation and Green-Wave Coordination

This repository is a reproducible educational prototype for generating a hypothetical
day of traffic demand and applying a simplified Webster cycle calculation at
three consecutive intersections (A, B and C). All data and results in this
prototype are **synthetic**. The repository contains matched simulated baseline
comparisons, but no real-world traffic performance improvement is claimed.

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
- Junction C green-split sensitivity for the AM and PM peaks; and
- CSV outputs, figures and automated tests.

It deliberately does **not** yet implement a field-calibrated or microscopic
traffic simulation, stochastic speeds, finite queue storage, machine learning,
reinforcement learning or a dashboard.

## Setup

From the repository root, create an isolated environment and install the small
dependency set:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

## Run

```bash
python -m traffic_signals.demo
```

The default run uses seed 42, 12 seconds of lost time, and cycle bounds of
40--150 seconds. These can be changed without editing source code:

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

After running the Phase 1 demo, generate the Phase 2 time-of-day summary with:

```bash
python phase2.py
```

The initial illustrative periods are Overnight (20:00--06:00), AM peak
(06:00--10:00), Interpeak (10:00--16:00), and PM peak (16:00--20:00).

Select and evaluate the exploratory p85 fixed timing plans with:

```bash
python phase2_plans.py
```

This rounds each period's 85th-percentile Webster cycle upward to the next five
seconds, within the 40--150 second bounds. It is an exploratory heuristic, not
an established optimal timing rule.

Design and evaluate fixed EW/NS effective-green allocations with:

```bash
python phase2_green_splits.py
```

This uses each phase's period-level 85th-percentile critical flow ratio to
allocate the available effective green proportionally, subject to an
illustrative 10-second minimum for each phase. It then screens every interval
against the fixed phase allocations. The minimum is a prototype assumption,
not a field-deployment safety standard.

Optimise each period's cycle and EW/NS split by transparent grid search with:

```bash
python phase2_optimise.py
```

The search considers 40--150 second cycles in five-second increments and
one-second EW green increments, while preserving 12 seconds of lost time and at
least 10 seconds for each phase. It prioritises fewer phase warnings, smaller
total shortfall, shorter cycles, and finally a greater minimum capacity margin.

Create the initial common-cycle eastbound coordination plans with:

```bash
python phase3_coordination.py
```

This selects the longest independently optimised cycle in each period, assumes
400 m from A to B, 500 m from B to C and a constant 40 km/h progression speed,
then aligns EW green starts to the resulting 0, 36 and 81 second cumulative
travel times. The resulting bandwidth is theoretical and assumes exact travel
times with no queues or disturbances.

Generate and numerically verify the four design-speed time-space diagrams with:

```bash
python phase3_time_space.py
```

Evaluate the unchanged Phase 3A plans at 30, 35, 40, 45 and 50 km/h with:

```bash
python phase3_speed_sensitivity.py
```

For each period and speed, this samples every whole-second departure within one
signal cycle and reports the fraction that encounters effective EW green at A,
B and C. This is a theoretical progression-window measure, not an estimate of
the percentage of real vehicles completing the corridor without stopping.

Optimise the B and C offsets for equal-weighted robustness across the five
tested speeds with:

```bash
python phase3_offset_optimisation.py
```

The search keeps A at zero and checks every integer B/C offset pair. It first
maximises mean progression across 30, 35, 40, 45 and 50 km/h, then worst-speed
progression, then 40 km/h progression, then proximity to the Phase 3A offsets.
These are exploratory priorities rather than a measured vehicle-speed
distribution.

Evaluate the original and Phase 3D offsets for both eastbound and westbound
progression with:

```bash
python phase3_bidirectional.py
```

This applies the same whole-second arrival-window calculation in the A--B--C
and C--B--A directions at all five test speeds. The combined summary weights
the two directions and five speeds equally; it is not weighted by traffic
demand. The runner also verifies that all eastbound values reproduce the Phase
3D results before writing the Phase 3E outputs.

Optimise offsets using synthetic directional-demand proxies with:

```bash
python phase3_bidirectional_optimisation.py
```

The Phase 3F search derives period weights from eastbound counts entering at A
and westbound counts entering at C. It keeps A at zero, preserves every Phase
3A cycle and green allocation, and prevents either direction's 40 km/h
progression from falling more than five percentage points below its Phase 3A
value. The five-point threshold is an experimental modelling choice, not a
traffic engineering standard.

Compare the Phase 3A and Phase 3F AM offsets using the same generated vehicles:

```bash
python phase4_vehicle_simulation.py
```

Phase 4A estimates hourly entry demand from the 16 synthetic AM observations at
A eastbound and C westbound, draws Poisson vehicle counts with seed 42, and
distributes entry times uniformly over one hour. Vehicles travel at 40 km/h and
join unlimited FIFO queues with a two-second discharge headway. The AM timing
plan continues after the one-hour generation window until every vehicle exits.
`waiting_events` counts junctions with positive waiting time; it is not a count
of confirmed physical stops.

Repeat the matched comparison for all four periods and seeds 42--71 with:

```bash
python phase4b_multi_seed.py
```

Phase 4B simulates a representative hour for each period and calculates paired
Phase 3F-minus-Phase 3A delay differences. Its 95% bootstrap intervals resample
the 30 complete paired-seed differences. They quantify variation under the
synthetic arrival process and fixed model assumptions, not uncertainty about
real traffic conditions or model validity.

Test demand and congestion sensitivity at 75%, 100%, 125% and 150% of the
baseline demand with:

```bash
python phase4c_demand_sensitivity.py
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
python phase4d_delay_optimisation.py
```

Phase 4D-1 keeps the Phase 3A cycle and green allocations fixed. It uses only
seeds 42--44 to conduct a 10-second coarse offset search followed by one-second
refinement around selected candidates. Candidate directional mean delay may be
no more than two seconds above its Phase 3A training baseline. Seeds 45--71 are
reserved for a single unseen-sample comparison with Phase 3A and Phase 3F. The
search is not exhaustive, and its result is a provisional AM candidate rather
than a final or globally optimal timing plan.

Stress-test the frozen AM Phase 4D candidate with fresh seeds using:

```bash
python phase4d2_stress_validation.py
```

Phase 4D-2 does not search for new offsets. It compares the fixed 0/32/68-second
plan with Phase 3A and Phase 3F using seeds 72--101 at 75%, 100%, 125% and 150%
of baseline AM demand. All strategies receive identical nested arrival streams.
Directional fairness is reported at three distinct levels against Phase 3A:
the mean change, the upper bootstrap bound and the number of individual seeds
whose delay increase is no greater than two seconds.

Select and holdout-test a more robustly fair AM plan with:

```bash
python phase4d3_robust_fairness.py
```

Phase 4D-3 keeps the 150-second cycle and Phase 3A green allocations fixed and
searches a local B/C offset grid around earlier strategies. A relaxed four-seed
screen reduces the candidate pool. Final training uses seeds 102--113 at 75%,
100%, 125% and 150% demand. A plan is feasible only when the upper endpoint of
each of eight directional bootstrap intervals is no greater than two seconds
relative to Phase 3A and combined mean delay does not increase at any factor.
The selected offsets are frozen before comparison with Phase 3A, Phase 3F and
Phase 4D on untouched seeds 114--143. The four demand scenarios are weighted
equally during selection; their probabilities in real traffic are unknown.

Build and holdout-test the four-period Phase 4E strategy with:

```bash
python phase4e_four_period_optimisation.py
```

Phase 4E retains the frozen 0/36/70-second AM plan and independently searches
local B/C offset neighbourhoods around Phase 3A and Phase 3F for Overnight,
Interpeak and PM peak. It reuses the Phase 4D-3 relaxed four-seed screen: at
100% and 150% demand, directional mean delay changes must be no greater than
three seconds and combined delay must not increase. Full training then uses
seeds 144--155 and all four demand factors with the two-second bootstrap-upper-
endpoint constraint. The four selected plans are frozen before validation on
seeds 156--185. Signal cycles and effective green allocations remain unchanged.

Compare the AM Phase 3A and Phase 4E offsets with competing north-south traffic:

```bash
python phase4f1_full_junction.py
```

Phase 4F-1 uses the synthetic 08:00--09:00 counts without averaging them over
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
python phase4f2_multi_seed_full_junction.py
```

Phase 4F-2 evaluates Overnight at 02:00--03:00, AM peak at 08:00--09:00,
Interpeak at 12:00--13:00 and PM peak at 17:00--18:00. It preserves each hour's
four synthetic 15-minute profiles and compares the fixed Phase 3A and Phase 4E
plans using matched streams for seeds 200--229. Results are reported separately
for eight movements and for network delay per junction passage. The four hours
are representative snapshots and are not combined into a daily average.

Test Junction C green allocation while holding the Phase 4E offsets fixed with:

```bash
python phase4f3a_green_split_sensitivity.py
```

Phase 4F-3A transfers 0, 3, 5, 7 or 10 seconds of effective green from EW to
NS at Junction C in the representative AM and PM peak hours. Plans at A and B,
the 150-second common cycle, 12 seconds of lost time and C's EW offset remain
unchanged. Because the phases are consecutive, transferring green also moves
C's NS phase start earlier. Each scenario uses the same eight arrival streams
for seeds 300--311. This is a synthetic sensitivity experiment, not selection
or validation of a new operational timing plan. The earlier two-second
directional fairness threshold is reported only as a reference here, not
enforced as a constraint.

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

5. Bound the selected cycle to 40--150 seconds and allocate effective green
   (`C - L`) in proportion to the two critical ratios.

The illustrative default lost time `L` is 12 seconds per cycle. Invalid or
negative inputs raise clear errors. Zero demand returns the minimum cycle and
zero demand-based green allocations. `Y >= 1` is flagged as oversaturated. A
maximum-cycle result is also flagged as insufficient when `Y` exceeds the
fraction of that capped cycle available as effective green. These flags are
diagnostics, not proof of field performance.

## Outputs

The workflow writes:

- `outputs/data/synthetic_traffic_15min.csv`
- `outputs/data/synthetic_webster_results.csv`
- `outputs/data/phase2_period_summary.csv` (after running `phase2.py`)
- `outputs/data/phase2_fixed_plans.csv` (after running `phase2_plans.py`)
- `outputs/data/phase2_interval_comparison.csv` (after running `phase2_plans.py`)
- `outputs/data/phase2_plan_evaluation.csv` (after running `phase2_plans.py`)
- `outputs/data/phase2_green_split_plans.csv` (after running `phase2_green_splits.py`)
- `outputs/data/phase2_green_split_evaluation.csv` (after running `phase2_green_splits.py`)
- `outputs/data/phase2_green_split_summary.csv` (after running `phase2_green_splits.py`)
- `outputs/data/phase2_optimised_plans.csv` (after running `phase2_optimise.py`)
- `outputs/data/phase2_optimised_interval_evaluation.csv` (after running `phase2_optimise.py`)
- `outputs/data/phase3a_common_cycles.csv` (after running `phase3_coordination.py`)
- `outputs/data/phase3a_signal_plans.csv` (after running `phase3_coordination.py`)
- `outputs/data/phase3a_bandwidth.csv` (after running `phase3_coordination.py`)
- `outputs/data/phase3b_bandwidth_verification.csv` (after running `phase3_time_space.py`)
- `outputs/data/phase3c_speed_sensitivity.csv` (after running `phase3_speed_sensitivity.py`)
- `outputs/data/phase3d_optimised_offsets.csv` (after running `phase3_offset_optimisation.py`)
- `outputs/data/phase3d_robustness_comparison.csv` (after running `phase3_offset_optimisation.py`)
- `outputs/data/phase3e_bidirectional_evaluation.csv` (after running `phase3_bidirectional.py`)
- `outputs/data/phase3e_direction_summary.csv` (after running `phase3_bidirectional.py`)
- `outputs/data/phase3e_combined_summary.csv` (after running `phase3_bidirectional.py`)
- `outputs/data/phase3f_direction_weights.csv` (after running `phase3_bidirectional_optimisation.py`)
- `outputs/data/phase3f_optimisation_summary.csv` (after running `phase3_bidirectional_optimisation.py`)
- `outputs/data/phase3f_speed_direction_comparison.csv` (after running `phase3_bidirectional_optimisation.py`)
- `outputs/data/phase4a_summary.csv` (after running `phase4_vehicle_simulation.py`)
- `outputs/data/phase4a_vehicle_journeys.csv` (after running `phase4_vehicle_simulation.py`)
- `outputs/data/phase4a_queue_summary.csv` (after running `phase4_vehicle_simulation.py`)
- `outputs/data/phase4a_combined_comparison.csv` (after running `phase4_vehicle_simulation.py`)
- `outputs/data/phase4b_replication_metrics.csv` (after running `phase4b_multi_seed.py`)
- `outputs/data/phase4b_paired_differences.csv` (after running `phase4b_multi_seed.py`)
- `outputs/data/phase4b_uncertainty_summary.csv` (after running `phase4b_multi_seed.py`)
- `outputs/data/phase4c_replication_metrics.csv` (after running `phase4c_demand_sensitivity.py`)
- `outputs/data/phase4c_approach_queues.csv` (after running `phase4c_demand_sensitivity.py`)
- `outputs/data/phase4c_paired_differences.csv` (after running `phase4c_demand_sensitivity.py`)
- `outputs/data/phase4c_sensitivity_summary.csv` (after running `phase4c_demand_sensitivity.py`)
- `outputs/data/phase4d_selected_plan.csv` (after running `phase4d_delay_optimisation.py`)
- `outputs/data/phase4d_training_metrics.csv` (after running `phase4d_delay_optimisation.py`)
- `outputs/data/phase4d_validation_metrics.csv` (after running `phase4d_delay_optimisation.py`)
- `outputs/data/phase4d_validation_summary.csv` (after running `phase4d_delay_optimisation.py`)
- `outputs/data/phase4d2_replication_metrics.csv` (after running `phase4d2_stress_validation.py`)
- `outputs/data/phase4d2_paired_differences.csv` (after running `phase4d2_stress_validation.py`)
- `outputs/data/phase4d2_stress_summary.csv` (after running `phase4d2_stress_validation.py`)
- `outputs/data/phase4d3_screening.csv` (after running `phase4d3_robust_fairness.py`)
- `outputs/data/phase4d3_training_candidates.csv` (after running `phase4d3_robust_fairness.py`)
- `outputs/data/phase4d3_selected_plan.csv` (after running `phase4d3_robust_fairness.py`)
- `outputs/data/phase4d3_holdout_metrics.csv` (after running `phase4d3_robust_fairness.py`)
- `outputs/data/phase4d3_holdout_summary.csv` (after running `phase4d3_robust_fairness.py`)
- `outputs/data/phase4e_screening.csv` (after running `phase4e_four_period_optimisation.py`)
- `outputs/data/phase4e_training_candidates.csv` (after running `phase4e_four_period_optimisation.py`)
- `outputs/data/phase4e_selected_plans.csv` (after running `phase4e_four_period_optimisation.py`)
- `outputs/data/phase4e_holdout_metrics.csv` (after running `phase4e_four_period_optimisation.py`)
- `outputs/data/phase4e_holdout_summary.csv` (after running `phase4e_four_period_optimisation.py`)
- `outputs/data/phase4f1_corridor_results.csv` (after running `phase4f1_full_junction.py`)
- `outputs/data/phase4f1_ns_approaches.csv` (after running `phase4f1_full_junction.py`)
- `outputs/data/phase4f1_network_summary.csv` (after running `phase4f1_full_junction.py`)
- `outputs/data/phase4f1_signal_windows.csv` (after running `phase4f1_full_junction.py`)
- `outputs/data/phase4f1_c_ns_capacity.csv` (after running `phase4f1_full_junction.py`)
- `outputs/data/phase4f2_movement_metrics.csv` (after running `phase4f2_multi_seed_full_junction.py`)
- `outputs/data/phase4f2_network_metrics.csv` (after running `phase4f2_multi_seed_full_junction.py`)
- `outputs/data/phase4f2_paired_movements.csv` (after running `phase4f2_multi_seed_full_junction.py`)
- `outputs/data/phase4f2_movement_summary.csv` (after running `phase4f2_multi_seed_full_junction.py`)
- `outputs/data/phase4f2_paired_network.csv` (after running `phase4f2_multi_seed_full_junction.py`)
- `outputs/data/phase4f2_network_summary.csv` (after running `phase4f2_multi_seed_full_junction.py`)
- `outputs/data/phase4f2_c_capacity.csv` (after running `phase4f2_multi_seed_full_junction.py`)
- `outputs/data/phase4f3a_movement_metrics.csv` (after running `phase4f3a_green_split_sensitivity.py`)
- `outputs/data/phase4f3a_network_metrics.csv` (after running `phase4f3a_green_split_sensitivity.py`)
- `outputs/data/phase4f3a_capacity_scenarios.csv` (after running `phase4f3a_green_split_sensitivity.py`)
- `outputs/data/phase4f3a_paired_network.csv` (after running `phase4f3a_green_split_sensitivity.py`)
- `outputs/data/phase4f3a_network_summary.csv` (after running `phase4f3a_green_split_sensitivity.py`)
- `outputs/data/phase4f3a_paired_movements.csv` (after running `phase4f3a_green_split_sensitivity.py`)
- `outputs/data/phase4f3a_movement_summary.csv` (after running `phase4f3a_green_split_sensitivity.py`)
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

Both figure titles identify the data/model as synthetic.

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
Phase 4A--4E corridor comparisons exclude north-south demand and shared phase
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
