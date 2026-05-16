"""Figure 4: main comparison.

5 algorithms x 3 metrics. For each (algorithm, metric) cell we emit the
per-seed values (for the box plot) plus the paired Wilcoxon test vs the
Constrained TPP baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

from .shared import write_csv, load_results

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.aggregator import paired_wilcoxon, bootstrap_ci


REFERENCE = "constrained_tpp"


def build(results_path: str, out_box: str, out_tests: str) -> dict:
    df = load_results(results_path)
    if df.empty:
        return {"box": write_csv(out_box, []),
                "tests": write_csv(out_tests, [])}
    f4 = df[df["dataset"].str.contains("synthetic", na=False) &
            (df["uncertainty_level"] == "default") &
            (df["priority_regime"] == "balanced")]

    metrics = ("sales_norm", "coverage", "robustness", "robustness_absolute_loss")
    box_rows = []
    test_rows = []

    ref = f4[f4["algorithm"] == REFERENCE].sort_values("seed")
    available_metrics = [m for m in metrics if m in f4.columns]
    for algo in sorted(f4["algorithm"].unique()):
        grp = f4[f4["algorithm"] == algo].sort_values("seed")
        for metric in available_metrics:
            vals = grp[metric].dropna().tolist()
            for s, v in zip(grp["seed"].tolist(), vals):
                box_rows.append({"algorithm": algo, "metric": metric,
                                 "seed": int(s), "value": float(v)})
            if algo == REFERENCE:
                continue
            a = grp[metric].tolist()
            b = ref[metric].tolist()
            n = min(len(a), len(b))
            p, delta = paired_wilcoxon(a[:n], b[:n])
            lo, hi = bootstrap_ci(vals)
            test_rows.append({"algorithm": algo, "vs": REFERENCE,
                              "metric": metric, "n": n,
                              "p_value": p, "cliffs_delta": delta,
                              "ci_low": lo, "ci_high": hi})
    return {"box": write_csv(out_box, box_rows),
            "tests": write_csv(out_tests, test_rows)}
