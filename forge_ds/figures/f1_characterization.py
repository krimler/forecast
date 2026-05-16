"""Figure 1: data characterization (spec2 §12.1).

Reads the spec1 dataset directly (no harness needed) and writes the
six-panel summary stats. Each panel becomes one CSV.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Dict
import pandas as pd

from .shared import write_csv


def build(dataset_dir: str, out_dir: str) -> Dict[str, str]:
    p = Path(dataset_dir)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ev = pd.read_csv(p / "activity_log.csv")
    pop = pd.read_csv(p / "population.csv")
    u = pd.read_csv(p / "uncertainty_traces.csv")
    actual = ev[ev["scenario_id"] == "actual"]

    paths: Dict[str, str] = {}

    # 1a: calls/day box-plot data, by rep type.
    counts_per_day = (actual.groupby(["rep_id", "date"]).size()
                      .reset_index(name="calls"))
    counts_per_day = counts_per_day.merge(pop[["rep_id", "type"]], on="rep_id")
    paths["1a"] = write_csv(str(out_dir / "f1a_calls_by_type.csv"),
                            counts_per_day.to_dict("records"))

    # 1b: duration violin data, by segment.
    paths["1b"] = write_csv(str(out_dir / "f1b_duration_by_segment.csv"),
                            actual[["segment_at_call", "planned_duration_min"]]
                            .to_dict("records"))

    # 1c: mean calls Mon..Fri.
    actual_c = actual.copy()
    actual_c["dow"] = pd.to_datetime(actual_c["date"]).dt.day_name()
    dow_share = actual_c.groupby("dow").size().reset_index(name="calls")
    paths["1c"] = write_csv(str(out_dir / "f1c_calls_by_dow.csv"),
                            dow_share.to_dict("records"))

    # 1d: priority regimes (pull from config.json).
    import json
    cfg = json.loads((p / "config.json").read_text())
    rows = []
    for tag, weights in [("balanced_3", [0.34, 0.33, 0.33]),
                          ("moderate_3", [0.5, 0.3, 0.2]),
                          ("heavy_3", [0.6, 0.3, 0.1])]:
        for b, w in enumerate(weights):
            rows.append({"regime": tag, "brand_slot": b, "weight": w})
    paths["1d"] = write_csv(str(out_dir / "f1d_priority_regimes.csv"), rows)

    # 1e: 5 account availability traces over first 90 days.
    five = (u[(u["entity_type"] == "account")]
            .groupby("entity_id").size().sort_values(ascending=False)
            .head(5).index.tolist())
    subset = u[(u["entity_type"] == "account") & u["entity_id"].isin(five)]
    paths["1e"] = write_csv(str(out_dir / "f1e_account_availability.csv"),
                            subset.to_dict("records"))

    # 1f: absence type frequencies + notice histogram.
    rep_u = u[u["entity_type"] == "rep"]
    freq = rep_u.groupby("event_type")["duration_days"].agg(["count", "mean"])\
                .reset_index()
    freq["notice_mean"] = rep_u.groupby("event_type")["notice_days"].mean().values
    paths["1f"] = write_csv(str(out_dir / "f1f_absence_frequency.csv"),
                            freq.to_dict("records"))

    return paths
