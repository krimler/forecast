"""Orthogonal-constraint probe (parallelized).

Runs six variants in parallel processes. Each worker trains its own
Neural TPP (training takes ~2 s on smoke data per the probe 2 log) and
runs inference with the variant's overrides. Wall-clock is bounded by
the slowest single variant.

Variants:
    Neural TPP                  sampling, no constraints
    Beam TPP                    beam, no constraints
    Constrained A_spec          oc_thr=1.5, capacity=540
    Constrained A_current       oc_thr=1.5, capacity=360 (matches probe 2b)
    Constrained B_orthogonal    no over-call, capacity=180
    Constrained C_control       no over-call, capacity=540 (essentially none)
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))


DATASET_DIR = "forge_ds/results/probe2_data"
CACHE_ROOT = "forge_ds/results/cache_probe2"
SCENARIOS = 3


VARIANTS = [
    ("neural_tpp",                 "neural_tpp",        None),
    ("beam_tpp",                   "beam_tpp",          None),
    ("constrained_A_spec_cap540",  "constrained_tpp",
     {"capacity_minutes": 540}),
    ("constrained_A_current_cap360", "constrained_tpp",
     {"capacity_minutes": 360}),
    ("constrained_B_orth_cap180_no_oc", "constrained_tpp",
     {"capacity_minutes": 180, "disable_over_call": True}),
    ("constrained_C_control_no_constraints", "constrained_tpp",
     {"capacity_minutes": 540, "disable_over_call": True}),
]


def _worker(label_algo_overrides):
    label, algo, overrides = label_algo_overrides
    # Re-import inside the worker so spawn-mode children pick up the path.
    pkg = Path(__file__).resolve().parent.parent
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    from harness.matrix import Cell
    from harness.runner import run_cell

    cell = Cell(
        cell_id=f"probe1_orth::{label}",
        algorithm=algo, dataset="probe2_high_unc",
        uncertainty_level="high", priority_regime="balanced", seed=42,
    )
    r = run_cell(cell, dataset_dir=DATASET_DIR,
                 cache_root=CACHE_ROOT, num_scenarios=SCENARIOS,
                 algo_overrides=overrides)
    return label, r


def main():
    print(f"Running {len(VARIANTS)} variants in parallel...")
    rows = {}
    with ProcessPoolExecutor(max_workers=len(VARIANTS)) as ex:
        futs = {ex.submit(_worker, v): v[0] for v in VARIANTS}
        for fut in as_completed(futs):
            label = futs[fut]
            try:
                _, r = fut.result()
                rows[label] = r
                print(f"  done {label:38s} sn={r['sales_norm']:+.4f}  "
                      f"cov={r['coverage']:.3f}  rob={r['robustness']:.3f}  "
                      f"softfallback={r['softfallback_invocations']}")
            except Exception as e:
                print(f"  FAIL {label}: {type(e).__name__}: {e}")

    if "beam_tpp" not in rows:
        print("\nMissing beam_tpp row, cannot compute gaps")
        return

    print()
    print("=" * 60)
    print("Gaps vs Beam TPP (no-constraint reference)")
    print("=" * 60)
    base = rows["beam_tpp"]["sales_norm"]
    order = [
        "neural_tpp",
        "constrained_A_spec_cap540",
        "constrained_A_current_cap360",
        "constrained_B_orth_cap180_no_oc",
        "constrained_C_control_no_constraints",
    ]
    for k in order:
        if k not in rows:
            print(f"  {k:38s} (missing)")
            continue
        gap = rows[k]["sales_norm"] - base
        print(f"  {k:38s} gap = {gap:+.4f}")
    print()
    print("Reading guide:")
    print("  A_* close to original +0.456: replicates probe 2b.")
    print("  B_orthogonal meaningfully positive: constraint mechanism")
    print("    contributes regardless of over-call/Sales alignment.")
    print("  B_orthogonal near zero: original gap was driven by the")
    print("    over-call cap accidentally optimizing Sales.")
    print("  C_control near zero: sanity (no real constraints active).")


if __name__ == "__main__":
    main()
