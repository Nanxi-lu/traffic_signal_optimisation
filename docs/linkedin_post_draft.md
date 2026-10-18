# LinkedIn post draft

**Status:** draft, not published.
**Framing:** an independent educational project. It is presented as learning and method,
not as a result for a real road.
**Recommended image:** `figures/linkedin_summary.png` (dark, 1200 × 1500 px, 4:5 portrait,
readable on mobile).
**Links:** the GitHub Pages site and the repository. Check the site loads before posting.

---

## Post

I have been working on an educational Python project to understand how traffic signal timing
shapes congestion, and what it takes to evaluate a timing change fairly.

The project models a synthetic three-junction corridor. I worked through the main steps a
traffic engineer would recognise: Webster cycle calculations, common-cycle coordination,
offsets between junctions, and green-time allocation. I then simulated individual vehicles
queueing at each junction, with corridor traffic competing against side roads for green time.

To compare strategies fairly, every signal plan was tested on the same 30 random traffic
patterns, so the differences come from the signals rather than luck. Bootstrap intervals show
how much uncertainty remains.

What the simulated PM peak showed:

• The best-performing strategy reduced average network delay by 8.6 seconds per junction
passage compared with the original coordinated plan, and was lower in all 30 runs.

• But eastbound traffic was delayed by 2.9 seconds more per vehicle, which breaks the
+2-second fairness limit I set before the final test.

• Compared with a different, fairness-protected baseline, the same increase was only 1.6
seconds and passed. The verdict depended on which baseline I chose.

The main thing I learned is that a network-wide average can hide who pays for an improvement.
Choosing the baseline and the fairness test is an engineering judgement, and it deserves as
much care as the optimisation itself.

This is a learning exercise, not a real-world result. The traffic is synthetic, the queue
model is simplified, and nothing has been calibrated or deployed.

It also connects to my earlier Liverpool bus project, where I used GTFS schedules and
real-time vehicle location data to measure how bus journeys compared with their timetables.
That data showed when buses ran late but could not show why, including how much was junction
delay. This project was my way of studying one of those mechanisms in a controlled setting.

Interactive results:
https://nanxi-lu.github.io/traffic_signal_optimisation/

Code and full methodology:
https://github.com/Nanxi-lu/traffic_signal_optimisation

I would welcome feedback from traffic engineers and network performance colleagues,
particularly on how fairness between movements is judged in practice.

#TrafficEngineering #TransportPlanning #IntelligentTransportSystems #Python #TransportAnalytics

---

## Shorter version (about 120 words)

I built an educational Python project to learn how traffic signal timing affects congestion,
and how to test a timing change fairly.

On a synthetic three-junction corridor, I simulated individual vehicles and tested each signal
plan on the same 30 random traffic patterns. In the PM peak, the best strategy cut average
network delay by 8.6 seconds per junction passage, but eastbound delay rose by 2.9 seconds per
vehicle, breaking the fairness limit I had set in advance.

The lesson: a network average can hide who pays for the gain.

The data is synthetic and nothing is deployed. It complements my Liverpool bus project, which
used real-time vehicle data to measure delays this project tries to explain.

Results: https://nanxi-lu.github.io/traffic_signal_optimisation/

---

## Notes on the wording

- **Numbers:** they match `outputs/data/phase4g_network_summary.csv` and
  `outputs/data/phase4g_movement_summary.csv`. They are rounded for readability: -8.611 s per
  passage, EB +2.906 s, EB vs Phase 4E +1.620 s.
- **"Best-performing strategy"** is the Integrated strategy. It is described in plain words
  because "Phase 3A" and "Integrated" mean nothing to a general reader.
- **Real-time data:** it is attributed only to the Liverpool bus project. This project uses
  no real-time or observed data.
- **What is left out:** the marginal AM peak C-N exceedance (+0.919 s mean, upper bound
  +2.197 s), to keep the post short. It is shown on the website and in the README. If asked
  whether only one movement failed, the accurate answer is two exceedances against Phase 3A:
  PM eastbound clearly, AM Junction C northbound marginally.
- **Claims avoided:** the post does not claim deployment, real-world benefit, field calibration,
  optimality, or that signal timing explains the Liverpool bus delays.
- **Formatting:** bullets use "•" because LinkedIn does not render Markdown lists.
