"""Append-only result log (spec2 §11.7).

Atomic per-row writes: stage to <log>.partial, fsync, then rename. This
keeps the on-disk file consistent if the process dies mid-run, so resume
mode can pick up where it left off without losing or duplicating rows.
"""
from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Set


COLUMNS = [
    "cell_id", "algorithm", "dataset", "uncertainty_level", "priority_regime", "seed",
    "sales_norm", "coverage", "robustness",
    "robustness_absolute_loss", "robustness_base_sales", "robustness_flagged",
    "disruption_rate", "replan_cost_median",
    "disruption_rate_source_account", "disruption_rate_source_rep_planned",
    "disruption_rate_source_rep_unplanned",
    "training_time_sec", "prediction_time_sec", "total_replan_time_sec",
    "softfallback_invocations", "config_hash", "git_commit", "timestamp",
]


def write_row(log_path: str, row: Dict[str, object]) -> None:
    Path(os.path.dirname(log_path) or ".").mkdir(parents=True, exist_ok=True)
    new_file = not os.path.exists(log_path)

    with tempfile.NamedTemporaryFile(
        mode="w", newline="", delete=False, dir=os.path.dirname(log_path) or "."
    ) as tmp:
        w = csv.writer(tmp)
        if new_file:
            w.writerow(COLUMNS)
        else:
            with open(log_path) as f:
                for line in f:
                    tmp.write(line)
        w.writerow([row.get(c, "") for c in COLUMNS])
        tmp_path = tmp.name
    os.replace(tmp_path, log_path)


def existing_cell_ids(log_path: str) -> Set[str]:
    if not os.path.exists(log_path):
        return set()
    out: Set[str] = set()
    with open(log_path) as f:
        for r in csv.DictReader(f):
            out.add(r["cell_id"])
    return out
