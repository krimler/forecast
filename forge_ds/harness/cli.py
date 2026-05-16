"""Harness CLI (§11.7, §11.8).

Subcommands:
    smoke   small matrix on forge-synth/dataset/output_smoke
    run     full or partial matrix over given dataset directories
    list    print the matrix without running
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

# Make sibling modules importable when this file is run as a script.
HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from harness.matrix import (
    Cell, UNCERTAINTY_LEVELS, build_smoke_matrix, build_full_matrix,
)
from harness import logger as log_mod
from harness import runner as run_mod


def _dataset_dir_for_cell(cell: Cell) -> str:
    """Map dataset tag in a cell to a directory path."""
    if cell.dataset == "synthetic_smoke":
        return "forge-synth/dataset/output_smoke"
    if cell.dataset == "synthetic_default":
        return "forge-synth/dataset/output_default"
    if cell.dataset == "foursquare_nyc":
        return "public_dataset/foursquare/nyc"
    if cell.dataset == "foursquare_tokyo":
        return "public_dataset/foursquare/tokyo"
    return cell.dataset


def _retag_cells(cells: List[Cell], new_dataset: str) -> List[Cell]:
    """Rewrite a list of smoke cells to point at a different dataset."""
    out = []
    for c in cells:
        tag = "default" if new_dataset == "synthetic_default" else c.dataset
        out.append(Cell(
            cell_id=c.cell_id.replace("smoke::", f"{tag}::").replace("syn::", f"{new_dataset}::"),
            algorithm=c.algorithm, dataset=new_dataset,
            uncertainty_level=c.uncertainty_level,
            priority_regime=c.priority_regime, seed=c.seed,
        ))
    return out


def _run_one(args):
    cell, dataset_dir, cache_root, num_scenarios = args
    return run_mod.run_cell(cell, dataset_dir=dataset_dir,
                            cache_root=cache_root,
                            num_scenarios=num_scenarios)


def main():
    p = argparse.ArgumentParser(description="ForgeDS harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    s_smoke = sub.add_parser("smoke", help="run smoke matrix")
    s_smoke.add_argument("--algorithms", nargs="+",
                         default=["markov", "prophet"])
    s_smoke.add_argument("--seeds", nargs="+", type=int, default=[42])
    s_smoke.add_argument("--log", default="forge_ds/results/results.csv")
    s_smoke.add_argument("--cache-root", default="forge_ds/results/cache")
    s_smoke.add_argument("--scenarios", type=int, default=5)
    s_smoke.add_argument("--workers", type=int, default=1)
    s_smoke.add_argument("--resume", action="store_true")
    s_smoke.add_argument("--dataset", default="synthetic_smoke",
                         choices=["synthetic_smoke", "synthetic_default",
                                  "foursquare_nyc", "foursquare_tokyo"],
                         help="point cells at a different dataset")

    s_list = sub.add_parser("list", help="print matrix")
    s_list.add_argument("--full", action="store_true")

    args = p.parse_args()

    if args.cmd == "list":
        cells = build_full_matrix() if args.full else build_smoke_matrix()
        for c in cells:
            print(c.cell_id)
        return

    if args.cmd == "smoke":
        cells = build_smoke_matrix(seeds=args.seeds, algorithms=args.algorithms)
        if args.dataset != "synthetic_smoke":
            cells = _retag_cells(cells, args.dataset)
        existing = log_mod.existing_cell_ids(args.log) if args.resume else set()
        cells = [c for c in cells if c.cell_id not in existing]
        if not cells:
            print("nothing to run")
            return

        tasks = [(c, _dataset_dir_for_cell(c), args.cache_root, args.scenarios)
                 for c in cells]

        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(_run_one, t) for t in tasks]
                for fut in as_completed(futs):
                    row = fut.result()
                    log_mod.write_row(args.log, row)
                    print("done", row["cell_id"], "sales_norm",
                          row["sales_norm"], "coverage", row["coverage"])
        else:
            for t in tasks:
                row = _run_one(t)
                log_mod.write_row(args.log, row)
                print("done", row["cell_id"], "sales_norm",
                      row["sales_norm"], "coverage", row["coverage"])


if __name__ == "__main__":
    main()
