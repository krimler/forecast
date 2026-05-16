"""Volume-matched comparison experiment.

Runs only the cells that aren't already in Stage 2's stage2.csv:
    Markov-VM on (forge_synth_default, foursquare_nyc, foursquare_tokyo) x seeds {42, 43}
    LP and Markov on those three datasets x seed=43 only (seed=43 datasets
        produce different uncertainty traces and need fresh runs)

Logs to forge_ds/results/stage2_vm.csv. Reuses cached snapshots from
stage2.py via _dataset_path so seed-43 datasets are picked up correctly.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))     # so stage2 (sibling script) imports
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))      # so harness/algorithms imports work

from stage2 import run_cell


LOG_PATH = "forge_ds/results/stage2_vm.csv"
DATASETS = ("foursquare_nyc", "foursquare_tokyo", "forge_synth_default")
NEW_CELLS = []
# Markov-VM: both seeds, all three datasets.
for ds in DATASETS:
    for s in (42, 43):
        NEW_CELLS.append(("markov_vm", ds, s))
# LP and Markov: only seed=43 (seed=42 already in Stage 2 stage2.csv).
for ds in DATASETS:
    for algo in ("lp", "markov"):
        NEW_CELLS.append((algo, ds, 43))


def main():
    print(f"Volume-matched comparison: {len(NEW_CELLS)} new cells")
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    new_file = not Path(LOG_PATH).exists()
    rows = []
    for algo, ds, seed in NEW_CELLS:
        t0 = time.perf_counter()
        try:
            r = run_cell(algo, ds, seed, workers=6)
            r["wall_time_sec"] = round(time.perf_counter() - t0, 1)
            rows.append(r)
            with open(LOG_PATH, "a", newline="") as f:
                cols = list(r.keys())
                w = csv.DictWriter(f, fieldnames=cols)
                if new_file:
                    w.writeheader()
                    new_file = False
                w.writerow(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"FAILED {algo}/{ds}/{seed}: {e}")
    print(f"\nWrote {len(rows)} rows to {LOG_PATH}")


if __name__ == "__main__":
    main()
