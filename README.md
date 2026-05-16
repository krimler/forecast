# forecast

Method-selection study for day-ahead field-force activity scheduling.

The project compares classical, ML, and LP-based methods for choosing
which accounts each sales rep should call on each day of a 14-day
rolling window. It ships a synthetic data generator, adapters for two
public datasets (Foursquare and SONAR), six forecasting algorithms, a
harness, and the numbers that came out of the runs.

## Layout

```
forecast/
  forge-synth/      synthetic dataset and generator
  forge_ds/         algorithms, harness, public-dataset adapters, experiments
  public_dataset/   raw + processed external datasets
  RESULTS.md        Stage 2 and volume-matched comparison numbers
```

Each subdirectory has its own README.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy pandas scipy matplotlib pulp prophet torch gdown

# Generate a small example dataset
.venv/bin/python forge-synth/code/generate.py --smoke --out forge-synth/dataset/output_smoke

# Run one cell of the Stage 2 comparison on smoke
PYTHONPATH=forge_ds .venv/bin/python forge_ds/experiments/stage2.py \
    --datasets forge_synth_default --algorithms lp markov naive --seeds 42 --workers 4
```

Full reproduction of the headline numbers is documented in `RESULTS.md`
(about 2.5 hours wall-clock from a clean state).

## What's in each subproject

**`forge-synth/`** is the synthetic data generator. Eight CSV/JSON files
per run describing rep population, account universe, panel assignments,
quarterly segment transitions, call events, absence and unavailability
traces, and summary stats. Deterministic in (seed, config). 33-check
validation suite included.

**`forge_ds/`** is the computation layer. Six algorithms (Markov,
Prophet aggregate, Neural TPP, Beam TPP, Constrained TPP, LP
assignment), a harness, runnable experiments under
`forge_ds/experiments/`, and a four-pass data characterization tool.

**`public_dataset/`** holds the Foursquare TSMC 2014 adapter output and
the SONAR (nursing activity) adapter output, both reshaped into the
same eight-file schema ForgeSynth produces. Raw downloads are kept
locally but gitignored; the adapters can re-fetch.

## Headline result

LP wins on absolute Sales across all three datasets in the main Stage 2
comparison. The volume-matched follow-up shows most of that advantage
comes from LP being given a larger daily call budget than Markov. At
matched volume the gap collapses to 1.01-1.05. See `RESULTS.md` for the
full tables, caveats, and reproduction recipe.

## What's not in this repo

- Paper drafts or figures. The figure-generator code under
  `forge_ds/figures/` is held over from an earlier scope and isn't run
  by the current pipeline.
- The internal spec documents the project was originally written
  against. Code and docs were reworded to be self-contained.
- Large generated datasets (default-scale ForgeSynth, Foursquare raw,
  SONAR). All regenerable. `output_smoke/` is checked in as a small
  runnable example.

## License

Synthetic data only. No real-world identifiers. Public dataset
attributions are in `public_dataset/README.md`.
