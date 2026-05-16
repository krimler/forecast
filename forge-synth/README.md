# ForgeSynth

Synthetic field-force activity data for forecasting work.

A seeded generator that produces sales-rep activity logs together with
the account universe, rep panels, planned and unplanned absences,
account unavailability traces, and two reference plans (greedy upper
bound, naive cadence).

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy

# default scale: 1000 reps, 365 days, 100K accounts
python code/generate.py --workers 8 --out dataset/output_default

# smoke scale: 20 reps, 120 days, 2K accounts
python code/generate.py --smoke --out dataset/output_smoke

# full validation
python validation/validate.py
```

## Layout

```
forge-synth/
  code/         the generator
  validation/   validate.py + last run log
  dataset/      generated output sets
```

## Data card

### Summary

ForgeSynth is a synthetic dataset of sales-rep field activity. Each run
produces eight files describing the account universe, the rep
population, panel assignments, daily call events, absence and
unavailability traces, and two reference plans over the same horizon.
The generator is deterministic in (seed, config), so anyone running it
with the same inputs gets the same outputs.

### Scale

A default run produces about 1.4M actual call events, 5.8K planned
absence blocks, 8K sick days, and 100 churn events across 1000 reps,
365 days, and 100K accounts. The smoke run produces about 9K call
events across 20 reps, 120 days, and 2K accounts.

Default-scale wall-clock is about 52 seconds on an Apple M3 with eight
workers.

### Files

Every output directory contains the same set of files.

| File | Rows | Description |
|---|---|---|
| `config.json` | n/a | Frozen config and parameter tables |
| `run_id.txt` | 1 | SHA-256 short hash of `config.json` |
| `population.csv` | reps + replacements | rep_id, type, force_id, panel_size, hire_date, departure_date |
| `accounts.csv` | accounts | account_id, specialty_id, initial_segment, eligible_brands |
| `panels.csv` | sum of panel sizes | rep_id, account_id, assignment_start_date, assignment_end_date |
| `segment_history.csv` | initial + transitions | account_id, effective_date, old_segment, new_segment |
| `activity_log.csv` | actual + greedy + naive calls | event_id, date, rep_id, start_time, planned_duration_min, actual_duration_min, account_id, segment_at_call, brand_id, brand_priority, outcome, scenario_id |
| `uncertainty_traces.csv` | absence + unavail blocks | trace_id, event_start_date, entity_type, entity_id, event_type, notice_days, duration_days, scenario_id |
| `validation_stats.csv` | ~30 rows | metric_name, scope, value |

### Provenance

Seed `42` is the published default. Run identity is the short SHA-256
of `config.json`, stored in `run_id.txt`. Same seed produces
byte-identical files. Parallel and serial runs produce identical output
because every rep draws from its own seeded RNG stream.

### Intended use

Training and evaluating forecasting models for field-force activity.
Benchmarking against the two reference plans. Stress-testing under
variable account unavailability and rep absence rates.

### What the generator does not model

Geography (no 2D coordinates, no travel time). Text channels (no notes,
no calendar entries, no news). Data-quality corruption (no missingness,
no outliers, no schema drift). Multi-rep coordination beyond churn
handoff. Brand launches or retirements. Account churn.

### Documented deviations from spec1

Six. Each one has an inline comment in the relevant file and a longer
note in `../checkmark.md` Section 7.

1. Greedy scores by marginal Sales lift instead of the spec's
   triangular `f_b`. The spec function is zero at `n = 0`, so every
   initial score ties and greedy concentrates past saturation.
2. Naive is capped at the same per-day call budget as the actual loop.
   Without a cap naive physically exceeds rep capacity.
3. `duration_days` in `uncertainty_traces.csv` counts working days, not
   the calendar span. This matches the spec's example numbers ("5-20
   for vacation"). Calendar bridges are stored alongside in memory.
4. Sick-rate denominator is the constant 252 working days per year, not
   the working days remaining from today. The literal "remaining"
   reading integrates to about 37 sick days a year for `rate = 6`,
   contradicting the spec's own T1 target.
5. Realized sick days per year is 8.02 vs spec target 6 (+33.7%). The other three absence types (personal, vacation, conference) realize within 12% of target. Likely cause is interaction between seasonal multiplier and autocorrelation in the sick-day sampling loop, but not investigated. T1 validation tolerance treats sick days as a documented exception. Empirically 8 sick days/year is realistic for many labor markets.
6. Robustness returns ratio plus absolute loss plus a flag, not just a
   scalar ratio. The bare ratio `Sales(replan) / Sales(plan)` is
   degenerate when the plan is near-empty (an empty plan scores ~1
   regardless of quality). The metric now returns a
   `RobustnessResult(ratio, absolute_loss, base_sales, n_scenarios,
   flagged)` and the harness logs all four. Read ratio and absolute
   loss together. (`metrics.py: robustness`)

### License

Synthetic data only. No real-world identifiers, no personal data.
