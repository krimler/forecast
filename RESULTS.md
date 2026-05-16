# Headline results

Method-selection comparison for day-ahead rep activity scheduling on
three datasets. LP, Markov, and a volume-matched Markov ("Markov-VM")
are compared on absolute Sales, coverage, robustness, and other
operational metrics. Two seeds per cell, single uncertainty realization
per (dataset, seed).

Raw data: `forge_ds/results/stage2.csv` (Stage 2 main run),
`forge_ds/results/stage2_vm.csv` (volume-matched extension).

## Stage 2 main comparison

LP vs Markov vs Naive, three datasets, two seeds each. Means across
both seeds.

| Dataset | Algorithm | sales_abs | coverage | rob_ratio | rob_abs_loss | disr_rate | calls / rep / day |
|---|---|---|---|---|---|---|---|
| ForgeSynth default | **LP** | **99 391** | 0.989 | 0.982 | 1 871 | 0.101 | 15.0 |
|  | Markov | 60 748 | 0.951 | 0.943 | 3 729 | 0.112 | 4.6 |
|  | Naive | 85 028 | 0.671 | 0.983 | 1 492 | 0.154 | 14.2 |
| Foursquare NYC | **LP** | **4 682** | 0.989 | 0.997 | 12 | 0.099 | 12.7 |
|  | Markov | 3 270 | 0.948 | 0.969 | 105 | 0.100 | 2.1 |
|  | Naive | 2 973 | 0.846 | 0.926 | 239 | 0.154 | 2.7 |
| Foursquare Tokyo | **LP** | **6 760** | 0.980 | 0.996 | 26 | 0.098 | 14.5 |
|  | Markov | 3 999 | 0.874 | 0.964 | 149 | 0.098 | 2.6 |
|  | Naive | 4 316 | 0.807 | 0.934 | 305 | 0.155 | 4.5 |

LP wins absolute sales on every dataset, ratio 1.4-1.7x over the
next-best baseline. Coverage also higher. Zero LP fallbacks across
all 12 cells (Stage 2 + Volume-Match), every per-(rep, window) LP
solved within the 30 s cap.

## Volume-matched comparison

The Stage 2 gap is partly because LP uses option-3 capacity (540 min/day,
no count cap) and so plans 12-15 calls per rep per day, while Markov
respects the per-rep daily count distribution learned from training
(~2-5 calls/day). To isolate "allocation quality" from "volume", we
ran Markov-VM: the same Markov as before but with a 12-call daily
budget (plus the same 540-min face-time cap).

| Dataset | LP / Markov (Stage 2) | LP / Markov-VM | Verdict |
|---|---|---|---|
| ForgeSynth default | 1.636 | **1.047** | weak |
| Foursquare NYC | 1.432 | **1.006** | weak |
| Foursquare Tokyo | 1.690 | **1.048** | weak |

Per the criterion agreed before the experiment (`LP / Markov-VM < 1.10`
is "weak"), **the LP advantage is almost entirely volume**, not
allocation. Once Markov is allowed to plan at the same daily call
budget, it captures 95-99% of LP's sales on every dataset. The
remaining 1-5% LP edge is real but small. It comes from cross-day
allocation: LP coordinates account choices across the 14-day window
via the piecewise-lift envelope, while Markov-VM is greedy per-day
with sequential sampling.

On ForgeSynth, Markov-VM actually beats LP on coverage (0.997 vs 0.989),
so even the small sales gap is partially traded for slightly less reach.

## Seed variance

Each dataset was generated at seeds 42 and 43 separately. Different
draws produce different uncertainty traces (verified):

| Dataset | sick s42 / s43 | vacation s42 / s43 | acct unavail s42 / s43 |
|---|---|---|---|
| ForgeSynth | 8 020 / 8 257 | 1 802 / 1 790 | 1 826 974 / 1 830 409 |
| Foursquare NYC | 2 567 / 2 628 | 569 / 558 | 29 454 / 29 361 |
| Foursquare Tokyo | 5 257 / 5 459 | 1 193 / 1 125 | 45 817 / 45 828 |

Resulting per-algorithm sales spreads:

| Algorithm | ForgeSynth default sales (s42 / s43) | Spread |
|---|---|---|
| LP | 97 267 / 103 639 | 6.5 % |
| Markov | 57 553 / 67 137 | 16.6 % |
| Markov-VM | 89 745 / 100 097 | 11.5 % |

Two seeds give a directional check, not confidence intervals. The
LP/Markov-VM ratio (~1.05) is robust to seed variance because the
spreads are correlated across algorithms within the same dataset
realisation.

## Honest framing

What the numbers support and don't support:

- **Supported:** an LP-based per-rep assignment is a viable baseline.
  It produces the highest absolute Sales on every dataset, no soft
  fallbacks, predictable solve time.
- **Supported:** at matched daily call volume, LP gives roughly a 1-5%
  uplift over a per-rep frequency-weighted random schedule. Whether
  that's worth the operational complexity at deployment scale is a
  domain question, not a methods question.
- **Not supported:** "LP gives a 40-70% Sales improvement over
  Markov". That's only true when LP gets a larger daily budget than
  Markov. The volume-matched comparison retracts this claim.
- **Not supported:** the original spec2 "Constrained TPP as paper
  contribution". The orthogonal-constraint probe showed the
  Constrained-vs-Beam gap was the over-call cap accidentally
  aligning with the Sales objective, not a property of the constraint
  mechanism.

## Data characterization (Stage 1)

Four-pass analysis of all four datasets in
`forge_ds/results/data_analysis.md`. Key signals:

| Dataset | calls/(rep, day) CV | R^2 of (rep + dow) on calls/day | Next-account top-1 (dow only) |
|---|---|---|---|
| ForgeSynth default | 0.27 | 0.576 | 0.009 |
| Foursquare NYC | 0.78 | 0.409 | 0.325 |
| Foursquare Tokyo | 0.83 | 0.329 | 0.282 |
| SONAR | 0.00 (one per day by construction) | 1.000 | n/a |

Adding lag-1, rolling-7, time-of-day, last-account, or recency features
does not push R^2 past ~0.5 on Foursquare and adds nothing on
ForgeSynth. This is consistent with the assignment-problem framing:
the per-day choice is "which subset of accounts to serve", not "which
account follows which".

## Reproducibility

```bash
# venv
python3 -m venv .venv
.venv/bin/python -m pip install numpy pandas scipy matplotlib pulp prophet torch gdown

# datasets
.venv/bin/python forge-synth/code/generate.py --workers 8 \
    --out forge-synth/dataset/output_default
.venv/bin/python forge-synth/code/generate.py --workers 8 \
    --config <(echo '{"seed": 43}') \
    --out forge-synth/dataset/output_default_s43
.venv/bin/python -m forge_ds.datasets.foursquare --city nyc --horizon 120 \
    --out public_dataset/foursquare/nyc
.venv/bin/python -m forge_ds.datasets.foursquare --city tky --horizon 120 \
    --out public_dataset/foursquare/tokyo
.venv/bin/python -m forge_ds.datasets.foursquare --city nyc --seed 43 \
    --horizon 120 --out public_dataset/foursquare/nyc_s43
.venv/bin/python -m forge_ds.datasets.foursquare --city tky --seed 43 \
    --horizon 120 --out public_dataset/foursquare/tokyo_s43

# Stage 2 (~75 min wall on M3 with 6 workers)
PYTHONPATH=forge_ds .venv/bin/python forge_ds/experiments/stage2.py --workers 6

# Volume-matched (~55 min wall)
PYTHONPATH=forge_ds .venv/bin/python forge_ds/experiments/stage2_vm.py

# Optional: data characterization (~5 min)
PYTHONPATH=forge_ds .venv/bin/python forge_ds/experiments/analyze_data.py
```

Total wall: ~2.5 hours from a clean state.
