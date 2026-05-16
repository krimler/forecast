"""Figure 2: source decomposition."""
from __future__ import annotations

import pandas as pd

from .shared import write_csv, load_results


def build(results_path: str, out_path: str) -> str:
    df = load_results(results_path)
    if df.empty:
        return write_csv(out_path, [])
    # F2 cells: algorithm=neural_tpp x 4 uncertainty levels x N seeds.
    f2 = df[(df["algorithm"] == "neural_tpp") &
            df["dataset"].str.contains("synthetic", na=False)]
    rows = []
    for level, grp in f2.groupby("uncertainty_level"):
        for src in ("account", "rep_planned", "rep_unplanned"):
            col = f"disruption_rate_source_{src}"
            rows.append({"uncertainty_level": level, "source": src,
                         "mean": float(grp[col].mean()),
                         "std": float(grp[col].std())})
    return write_csv(out_path, rows)
