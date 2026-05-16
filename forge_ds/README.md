# ForgeDS

Computation layer for the field-force forecasting study. Consumes
ForgeSynth output (or one of the public-dataset adapters) and runs
forecasting algorithms against it. Six algorithms are implemented:
five from the original spec2 plus an LP-based assignment baseline
added during method selection.

## Quickstart

```bash
# venv shared with ForgeSynth
python3 -m venv .venv
.venv/bin/python -m pip install numpy pandas scipy matplotlib pulp prophet torch

# Stage 2 main comparison: LP vs Markov vs Naive on three datasets, two seeds
PYTHONPATH=forge_ds .venv/bin/python forge_ds/experiments/stage2.py --workers 6

# Volume-matched comparison: adds Markov-VM, runs only the new cells
PYTHONPATH=forge_ds .venv/bin/python forge_ds/experiments/stage2_vm.py

# Four-pass data characterization across all four datasets
PYTHONPATH=forge_ds .venv/bin/python forge_ds/experiments/analyze_data.py

# Public dataset prep
PYTHONPATH=forge_ds .venv/bin/python -m forge_ds.datasets.foursquare \
    --city nyc --horizon 120 --out public_dataset/foursquare/nyc
PYTHONPATH=forge_ds .venv/bin/python -m forge_ds.datasets.foursquare \
    --city tky --horizon 120 --out public_dataset/foursquare/tokyo
PYTHONPATH=forge_ds .venv/bin/python forge_ds/datasets/sonar.py
```

## Layout

```
forge_ds/
  algorithms/    six algorithms (markov, prophet, neural_tpp, beam_tpp,
                 constrained_tpp, lp_assignment)
  datasets/      adapters for foursquare, sonar
  harness/       matrix, runner, cache, logger, aggregator, CLI
                 (used by the early spec2 ML runs; superseded for
                  method selection by stage2.py)
  figures/       per-figure CSV producers (held over from the spec2
                 paper-figure pipeline; not used after the reframe)
  validation/    spec2 A / D / H / F check suite
  results/       run logs, result CSVs, cached models
  experiments/   runnable record of the project's investigations:
                   analyze_data.py        four-pass dataset characterization
                   probe.py               Stage 1 diagnostic probes
                   probe_orthogonal.py    ablation probe used to test
                                          the spec2 paper claim
                   stage1.py              default-scale baseline runner
                                          (left in for record; the per-rep
                                          harness was the bottleneck and
                                          was replaced by stage2.py)
                   stage2.py              LP / Markov / Naive runner
                                          used for the main results
                   stage2_vm.py           volume-matched Markov experiment
public_dataset/
  foursquare/    raw zip + per-city spec1-schema CSVs (NYC, Tokyo)
  sonar/         raw streamed metadata + processed spec1-schema CSVs
```

## Algorithm files

| File | Spec § | Role |
|---|---|---|
| `algorithms/base.py` | 3 | Algorithm interface, ActivityHistory, PlanContext, Plan |
| `algorithms/markov.py` | 4 | Per-rep Markov chain with force-pooled fallback |
| `algorithms/prophet_agg.py` | 5 | Four Prophet series per rep, frequency-adherence disaggregation |
| `algorithms/neural_tpp.py` | 6 | Transformer TPP, factored mark heads, sampling inference |
| `algorithms/beam_tpp.py` | 7 | Same model, beam search with basic masks |
| `algorithms/constrained_tpp.py` | 8 | Same model, beam search with full constraints and soft fallback |
| `algorithms/lp_assignment.py` | (added Stage 2) | Per-rep LP/IP via PuLP+CBC, piecewise-linear lift, 30s solver cap |

## Stage 2 headline results

LP, Markov, and Naive on three datasets, two seeds each. Sales reported
as absolute units (Eq. 1 from spec1). For Foursquare the spec1-style
greedy ceiling is mis-specified for consumer check-in volumes, so
sales_norm against it is not meaningful; absolute numbers tell the
real story. Full table in `results/stage2.csv` and `results/stage2_vm.csv`.

| Dataset | Algorithm | sales_abs | coverage | rob_ratio | calls/rep/day |
|---|---|---|---|---|---|
| ForgeSynth default | LP | 99 391 | 0.989 | 0.982 | 15.0 |
|  | Markov-VM | 94 921 | 0.997 | 0.979 | 15.7 |
|  | Markov | 60 748 | 0.951 | 0.943 | 4.6 |
| Foursquare NYC | LP | 4 682 | 0.989 | 0.997 | 12.7 |
|  | Markov-VM | 4 654 | 1.000 | 0.996 | 11.3 |
|  | Markov | 3 270 | 0.948 | 0.969 | 2.1 |
| Foursquare Tokyo | LP | 6 760 | 0.980 | 0.996 | 14.5 |
|  | Markov-VM | 6 451 | 0.993 | 0.991 | 11.6 |
|  | Markov | 3 999 | 0.874 | 0.964 | 2.6 |

The volume-matched comparison (Markov-VM is Markov given the same
daily call budget as LP) shows the LP advantage over Markov is mostly
volume, not allocation. LP/Markov-VM ratios are 1.01-1.05 across all
three datasets. Detail in the top-level `RESULTS.md`.

## What's in `public_dataset/`

### Foursquare TSMC 2014

Yang et al., "Modeling User Activity Preference by Leveraging User
Spatial Temporal Characteristics in LBSNs", IEEE Trans. SMC 2015.

NYC: 1041 users, 4881 venues, 49 603 check-ins over 120 days.
Tokyo: 2153 users, 7578 venues, 122 706 check-ins over 120 days.

Adapter (`forge_ds/datasets/foursquare.py`) maps users to reps, venues
to accounts, check-ins to call events. Segments, brands, and priorities
are a synthetic overlay (no real correspondence in source data).
Reference plans (greedy_upper, naive) are emitted alongside the actual
scenario.

### SONAR (HPI-DHC 2023)

Lübbe et al., "SONAR: A Nursing Activity Dataset with Inertial Sensors",
Nature Sci Data 2023. Streamed from Zenodo (record `7881952`) by
`forge_ds/datasets/sonar.py`. 14 caregivers, 18 distinct activities,
254 recordings.

Mapping: caregiver to rep, activity type to account, recording to call
event. SONAR is small enough that it can't drive method selection; we
include it as an external-data spot check, not a primary benchmark.

