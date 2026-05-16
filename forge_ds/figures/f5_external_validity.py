"""Figure 5: external validity on Foursquare.

Same layout as Figure 4 but filtered to the foursquare_nyc and
foursquare_tokyo datasets.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

from .shared import write_csv, load_results
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.aggregator import paired_wilcoxon, bootstrap_ci


REFERENCE = "constrained_tpp"


def build(results_path: str, out_dir: str) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(results_path)
    if df.empty:
        return {}
    outs = {}
    for city_tag, fname in [("foursquare_nyc", "f5a_nyc"),
                             ("foursquare_tokyo", "f5b_tokyo")]:
        sub = df[df["dataset"] == city_tag]
        box_rows = []
        test_rows = []
        ref = sub[sub["algorithm"] == REFERENCE].sort_values("seed")
        for algo in sorted(sub["algorithm"].unique()):
            grp = sub[sub["algorithm"] == algo].sort_values("seed")
            for metric in ("sales_norm", "coverage", "robustness",
                           "robustness_absolute_loss"):
                if metric not in grp.columns:
                    continue
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
        outs[city_tag] = {
            "box": write_csv(str(out_dir / f"{fname}_box.csv"), box_rows),
            "tests": write_csv(str(out_dir / f"{fname}_tests.csv"), test_rows),
        }
    return outs
