"""Figure 3: Pareto frontier."""
from __future__ import annotations

import pandas as pd

from .shared import write_csv, load_results
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.aggregator import bootstrap_ci


def build(results_path: str, out_path: str) -> str:
    df = load_results(results_path)
    if df.empty:
        return write_csv(out_path, [])
    f3 = df[df["dataset"].str.contains("synthetic", na=False) &
            (df["uncertainty_level"] == "default")]
    rows = []
    for (algo, reg), grp in f3.groupby(["algorithm", "priority_regime"]):
        for metric in ("sales_norm", "coverage", "robustness",
                       "robustness_absolute_loss"):
            if metric not in grp.columns:
                continue
            vals = grp[metric].dropna().tolist()
            if not vals:
                continue
            lo, hi = bootstrap_ci(vals)
            rows.append({"algorithm": algo, "priority_regime": reg,
                         "metric": metric, "mean": float(pd.Series(vals).mean()),
                         "std": float(pd.Series(vals).std()),
                         "ci_low": lo, "ci_high": hi, "n": len(vals)})
    return write_csv(out_path, rows)
