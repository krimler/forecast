"""Result aggregation for figure CSVs (spec2 §11.9)."""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci(values: List[float], n_boot: int = 1000,
                 alpha: float = 0.05) -> Tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(0)
    sample_means = np.empty(n_boot)
    for i in range(n_boot):
        sample_means[i] = arr[rng.integers(0, len(arr), size=len(arr))].mean()
    lo = float(np.percentile(sample_means, 100 * alpha / 2))
    hi = float(np.percentile(sample_means, 100 * (1 - alpha / 2)))
    return lo, hi


def per_condition_summary(results: pd.DataFrame, group_cols: List[str],
                          metric: str) -> pd.DataFrame:
    rows = []
    for keys, grp in results.groupby(group_cols):
        vals = grp[metric].dropna().tolist()
        if not vals:
            continue
        lo, hi = bootstrap_ci(vals)
        row = dict(zip(group_cols, keys)) if isinstance(keys, tuple) else {group_cols[0]: keys}
        row.update({
            "metric": metric, "n": len(vals),
            "mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "ci_low": lo, "ci_high": hi,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def cliffs_delta(a: List[float], b: List[float]) -> float:
    """Cliff's delta effect size, in [-1, 1]."""
    if not a or not b:
        return 0.0
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def paired_wilcoxon(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Returns (p-value, Cliff's delta). Pairs by order."""
    if len(a) != len(b) or len(a) == 0:
        return 1.0, 0.0
    try:
        _, p = stats.wilcoxon(a, b, zero_method="zsplit")
    except ValueError:
        p = 1.0
    return float(p), cliffs_delta(a, b)
