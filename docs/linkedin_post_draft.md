# LinkedIn post draft

**Status:** draft, not published.
**Image:** `figures/linkedin_summary.png` (1200 × 1500 px, dark).
**Links:** the website and the repository.

---

## Post

Can retiming traffic lights cut congestion without making anyone's journey worse?

I built an educational Python project to find out. It simulates a road with three
signalised junctions and tests different signal timings: cycle lengths from Webster's
method, green waves between junctions, and how green time is split with the side roads.
Every plan faces the same 30 random rush hours, so any difference comes from the lights.

What I found in the evening peak:

• The best plan cut the average wait by 8.6 seconds at each junction, and was faster in
all 30 runs.

• But eastbound drivers waited 2.9 seconds longer, breaking the 2-second fairness limit
I had set at the start.

• The cause was the baseline. My search had checked fairness against an improved plan,
not the original, so small increases quietly added up.

• Rerunning the search against the original plan, and testing it on 30 new rush hours,
found a timing that saves 10.5 seconds with every direction within the limit.

The lesson for me: an average can hide who pays for an improvement, and which baseline
you compare against matters as much as the optimisation itself.

It connects to my Liverpool bus project, where real-time bus location data showed when
journeys ran late but not why. This project explores one possible reason, junction delay,
in a controlled setting.

Results: https://nanxi-lu.github.io/traffic_signal_optimisation/
Code: https://github.com/Nanxi-lu/traffic_signal_optimisation

I'd welcome thoughts from traffic engineers on how fairness between directions is judged
in practice.

#TrafficEngineering #TransportPlanning #Python #TransportAnalytics

---

## Shorter version

I built an educational Python project to see whether retiming traffic lights can cut
congestion without making any direction worse.

On a simulated three-junction road, the best plan cut the evening-peak wait by 8.6 seconds
per junction, but eastbound drivers waited 2.9 seconds longer, over my 2-second limit. The
cause: I had checked fairness against an improved plan, not the original. Rerunning the
search against the original found a timing that saves 10.5 seconds with every direction
within the limit.

An average can hide who pays for a gain.

https://nanxi-lu.github.io/traffic_signal_optimisation/

---

## Notes

- Numbers come from `phase4g_*` (8.6 s, 2.9 s, 30/30) and `phase4h_*` (10.5 s) in
  `outputs/data/`. The 10.5 s is measured on a fresh set of 30 runs (seeds 620 to 649).
- "Educational project" and "simulates" carry the context; no further disclaimers needed.
- Bullets use "•" because LinkedIn does not render Markdown lists.
