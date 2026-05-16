"""Stage 1: default-scale baseline picture for method selection.

Three algorithms at default ForgeSynth scale, two seeds each.
Six cells total. Runs sequentially. Thread-limited per process so a
single cell can use the full machine without contention.

Append-only output to forge_ds/results/stage1.csv, so partial progress
is preserved if the run is killed.
"""
from __future__ import annotations

import csv
import os

# Limit numpy/PyTorch threading to keep one cell from saturating the
# machine yet still benefit from BLAS parallelism. Set before any heavy
# import.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from harness.matrix import Cell
from harness.runner import run_cell


DATASET_DIR = "forge-synth/dataset/output_default"
CACHE_ROOT = "forge_ds/results/cache_stage1"
LOG_PATH = "forge_ds/results/stage1.csv"
NUM_SCENARIOS = 5


# Order: cheapest first so Markov + Prophet results land early.
CELLS = []
for algo in ("markov", "prophet", "neural_tpp"):
    for seed in (42, 43):
        CELLS.append((algo, seed))


def _existing_cell_ids() -> set:
    if not os.path.exists(LOG_PATH):
        return set()
    with open(LOG_PATH) as f:
        return {r["cell_id"] for r in csv.DictReader(f)}


def _append_row(row: dict) -> None:
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    new_file = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)


def main():
    print(f"Stage 1 sequential: {len(CELLS)} cells")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Scenarios per cell: {NUM_SCENARIOS}")
    print(f"OMP/MKL threads: 4")
    print()

    done = _existing_cell_ids()
    for algo, seed in CELLS:
        cell_id = f"stage1::{algo}::{seed}"
        if cell_id in done:
            print(f"  skip {algo:12s} seed={seed}  (already in log)")
            continue
        t0 = time.perf_counter()
        cell = Cell(
            cell_id=cell_id,
            algorithm=algo, dataset="synthetic_default",
            uncertainty_level="default", priority_regime="balanced", seed=seed,
        )
        try:
            r = run_cell(cell, dataset_dir=DATASET_DIR,
                         cache_root=CACHE_ROOT, num_scenarios=NUM_SCENARIOS)
            r["wall_time_sec"] = round(time.perf_counter() - t0, 1)
            _append_row(r)
            print(f"  done {algo:12s} seed={seed}  "
                  f"sn={r['sales_norm']:+.4f}  cov={r['coverage']:.3f}  "
                  f"rob={r['robustness']:.3f}  rob_loss={r['robustness_absolute_loss']:.1f}  "
                  f"replan_med={r['replan_cost_median']:.3f}s  "
                  f"wall={r['wall_time_sec']:.0f}s")
        except Exception as e:
            print(f"  FAIL {algo} seed={seed}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    if not os.path.exists(LOG_PATH):
        print("\nNo cells completed.")
        return

    # Summary by algorithm, mean across seeds.
    rows = []
    with open(LOG_PATH) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        return

    print()
    print("=" * 90)
    print("Stage 1 raw results (mean across seeds)")
    print("=" * 90)
    print(f"{'algorithm':12s} {'sales_norm':>11s} {'coverage':>9s} "
          f"{'rob_ratio':>10s} {'rob_loss':>10s} {'replan_med':>11s} {'wall_avg':>9s} {'n':>3s}")
    by_algo = {}
    for r in rows:
        by_algo.setdefault(r["algorithm"], []).append(r)
    for algo in ("markov", "prophet", "neural_tpp"):
        if algo not in by_algo:
            continue
        grp = by_algo[algo]
        def avg(k):
            vals = []
            for r in grp:
                try:
                    vals.append(float(r[k]))
                except (ValueError, KeyError, TypeError):
                    pass
            return sum(vals) / max(1, len(vals))
        print(f"{algo:12s} {avg('sales_norm'):+.4f}      "
              f"{avg('coverage'):.3f}    "
              f"{avg('robustness'):.3f}     "
              f"{avg('robustness_absolute_loss'):.1f}     "
              f"{avg('replan_cost_median'):.3f}s    "
              f"{avg('wall_time_sec'):.0f}s    "
              f"{len(grp)}")


if __name__ == "__main__":
    main()
