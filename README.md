# Traffic Signal Optimisation and Green-Wave Coordination

This repository is a reproducible educational prototype for generating a hypothetical
day of traffic demand and applying a simplified Webster cycle calculation at
three consecutive intersections (A, B and C). All data and results in this
prototype are **synthetic**. No traffic performance improvement is claimed because
there is not yet a measured or simulated baseline comparison.

## Scope

The completed work includes:

- reproducible 15-minute synthetic counts for eastbound, westbound,
  northbound and southbound approaches;
- transparent two-phase EW/NS Webster cycle and green-split calculations;
- CSV outputs and two figures; and
- unit tests for the formula, allocations, units and numerical safeguards; and
- an initial rule-based Phase 2 summary for four time-of-day periods.

It deliberately does **not** yet implement period-level timing-plan selection,
coordinated offsets, a traffic simulation, machine learning, reinforcement
learning or a dashboard.

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

Running the demo writes:

- `outputs/data/synthetic_traffic_15min.csv`
- `outputs/data/synthetic_webster_results.csv`
- `outputs/data/phase2_period_summary.csv` (after running `phase2.py`)
- `outputs/figures/synthetic_demand_profile.png`
- `outputs/figures/synthetic_webster_cycles.png`

Both figure titles identify the data/model as synthetic.

## Limitations and next steps

The model has no turning movements, lane groups, pedestrian timings, queues,
arrival dispersion, actuated control, intergreen detail or field calibration.
Converting one 15-minute count to an hourly equivalent is an analysis convention
and is not a rolling-hour measurement. The generated intersections are not yet
coordinated and no offsets are calculated. The current time-of-day boundaries
are manually defined starting assumptions, not data-derived or optimised
boundaries. A later step can select and evaluate representative period-level
plans before adding a reproducible baseline, travel-time assumptions and offset
evaluation or making any comparative performance claim.
