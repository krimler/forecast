"""Three cheap probes before committing to a default-scale run.

Probe 1: A3 sanity ordering at no-uncertainty smoke.
    50 reps, 60 days, all uncertainty params = 0.
    Run all 5 algorithms. Check ordering naive <= markov <= prophet <= ntpp.

Probe 2: Constraint binding under stress.
    20 reps, 120 days, high-uncertainty config (p_acct_unavail=0.30, sick=12).
    Run Constrained TPP. Read softfallback_invocations.

Probe 3: Prophet on Foursquare.
    50 users from Foursquare NYC.
    Run all 5 algorithms. Compare Prophet sales_norm here vs synthetic smoke.

Outputs a single summary per probe so the signal is clear.
"""
from __future__ import annotations

import csv
import os
import shutil
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))
FORGE_SYNTH = ROOT / "forge-synth" / "code"
if str(FORGE_SYNTH) not in sys.path:
    sys.path.insert(0, str(FORGE_SYNTH))

import pandas as pd
import numpy as np

import config as fs_config
from generate import run as fs_run

from harness.matrix import Cell
from harness.runner import run_cell


def _make_synth(*, name: str, out_dir: str, overrides: dict) -> str:
    """Generate a small forge-synth dataset with overrides applied to Config."""
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    cfg = fs_config.Config(**overrides, output_dir=str(out))
    fs_run(cfg, str(out))
    return str(out)


def _subset_foursquare(src: str, dst: str, num_users: int = 50) -> str:
    """Copy a Foursquare-prepared directory and prune to the first N users."""
    src = Path(src); dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in ("config.json", "run_id.txt"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)

    pop = pd.read_csv(src / "population.csv")
    keep_reps = sorted(pop["rep_id"].astype(int).unique().tolist())[:num_users]
    keep_set = set(keep_reps)
    pop[pop["rep_id"].isin(keep_set)].to_csv(dst / "population.csv", index=False)

    panels = pd.read_csv(src / "panels.csv")
    panels = panels[panels["rep_id"].isin(keep_set)]
    panels.to_csv(dst / "panels.csv", index=False)

    keep_accounts = set(panels["account_id"].astype(int).unique().tolist())
    accounts = pd.read_csv(src / "accounts.csv")
    accounts = accounts[accounts["account_id"].isin(keep_accounts)]
    accounts.to_csv(dst / "accounts.csv", index=False)

    seg = pd.read_csv(src / "segment_history.csv")
    seg = seg[seg["account_id"].isin(keep_accounts)]
    seg.to_csv(dst / "segment_history.csv", index=False)

    act = pd.read_csv(src / "activity_log.csv")
    act = act[act["rep_id"].isin(keep_set) & act["account_id"].isin(keep_accounts)]
    act.to_csv(dst / "activity_log.csv", index=False)

    u = pd.read_csv(src / "uncertainty_traces.csv")
    rep_mask = (u["entity_type"] == "rep") & u["entity_id"].astype(int).isin(keep_set)
    acct_mask = (u["entity_type"] == "account") & u["entity_id"].astype(int).isin(keep_accounts)
    u[rep_mask | acct_mask].to_csv(dst / "uncertainty_traces.csv", index=False)

    if (src / "validation_stats.csv").exists():
        shutil.copy(src / "validation_stats.csv", dst / "validation_stats.csv")
    return str(dst)


def _cells_for(probe: str, algorithms: List[str], dataset_tag: str,
               uncertainty: str = "default", regime: str = "balanced",
               seed: int = 42) -> List[Cell]:
    cells = []
    for a in algorithms:
        cells.append(Cell(
            cell_id=f"{probe}::{a}::{dataset_tag}::{uncertainty}::{regime}::{seed}",
            algorithm=a, dataset=dataset_tag, uncertainty_level=uncertainty,
            priority_regime=regime, seed=seed,
        ))
    return cells


def _run_probe(cells: List[Cell], dataset_dir: str, log_path: str,
               cache_root: str, num_scenarios: int = 3) -> List[dict]:
    rows = []
    Path(os.path.dirname(log_path) or ".").mkdir(parents=True, exist_ok=True)
    for c in cells:
        t0 = time.perf_counter()
        try:
            r = run_cell(c, dataset_dir=dataset_dir,
                         cache_root=cache_root, num_scenarios=num_scenarios)
            r["elapsed_sec"] = round(time.perf_counter() - t0, 2)
            rows.append(r)
            print(f"  done {c.algorithm:18s} sn={r['sales_norm']:+.4f} "
                  f"cov={r['coverage']:.3f} rob={r['robustness']:.3f} "
                  f"softfallback={r['softfallback_invocations']}")
        except Exception as e:
            print(f"  FAIL {c.algorithm}: {e}")
    if rows:
        cols = list(rows[0].keys())
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    return rows


def probe1_a3_no_uncertainty():
    print("\n=== Probe 1: A3 sanity ordering at no-uncertainty smoke ===")
    print("Config: 50 reps, 60 days, 2K accounts, all uncertainty = 0.")
    out_dir = "forge_ds/results/probe1_data"
    _make_synth(
        name="probe1",
        out_dir=out_dir,
        overrides=dict(
            seed=42, horizon_days=60, warmup_days=20,
            num_reps=50, num_accounts_total=2000,
            p_account_unavail=0.0, sick_days_per_year_mean=0.0,
            personal_days_per_year_mean=0.0, vacation_days_per_year_mean=0.0,
            conference_days_per_year_mean=0.0, p_churn_annual=0.0,
        ),
    )
    cells = _cells_for("probe1",
                       ["markov", "prophet", "neural_tpp", "beam_tpp", "constrained_tpp"],
                       dataset_tag="probe1_no_unc", uncertainty="none")
    rows = _run_probe(cells, out_dir,
                      "forge_ds/results/probe1_results.csv",
                      "forge_ds/results/cache_probe1", num_scenarios=1)
    if rows:
        ordered = sorted(rows, key=lambda r: r["sales_norm"])
        print("\n  Ordering by sales_norm:")
        for r in ordered:
            print(f"    {r['algorithm']:18s} {r['sales_norm']:+.4f}")
        # A3 expects naive <= markov <= prophet <= ntpp.
        by_algo = {r["algorithm"]: r["sales_norm"] for r in rows}
        order_ok = (by_algo.get("markov", 0) >= 0
                    and by_algo.get("prophet", 0) >= by_algo.get("markov", 0)
                    and by_algo.get("neural_tpp", 0) >= by_algo.get("prophet", 0))
        verdict = "PASS" if order_ok else "FAIL"
        print(f"  A3 verdict: {verdict}")
    return rows


def probe2_constraint_binding():
    print("\n=== Probe 2: Constraint binding at high-uncertainty smoke ===")
    print("Config: 20 reps, 120 days, p_acct_unavail=0.30, sick=12, churn=0.20.")
    out_dir = "forge_ds/results/probe2_data"
    _make_synth(
        name="probe2",
        out_dir=out_dir,
        overrides=dict(
            seed=42, horizon_days=120, warmup_days=30,
            num_reps=20, num_accounts_total=2000,
            p_account_unavail=0.30, sick_days_per_year_mean=12.0,
            personal_days_per_year_mean=5.0, vacation_days_per_year_mean=25.0,
            conference_days_per_year_mean=8.0, p_churn_annual=0.20,
        ),
    )
    cells = _cells_for("probe2",
                       ["constrained_tpp"],
                       dataset_tag="probe2_high_unc", uncertainty="high")
    rows = _run_probe(cells, out_dir,
                      "forge_ds/results/probe2_results.csv",
                      "forge_ds/results/cache_probe2", num_scenarios=3)
    if rows:
        r = rows[0]
        fb = int(r["softfallback_invocations"])
        verdict = "BINDS" if fb > 0 else "DOES NOT BIND"
        print(f"  Constraint binding: {verdict} (softfallback_invocations={fb})")
    return rows


def probe3_prophet_foursquare():
    print("\n=== Probe 3: Prophet (and rest) on Foursquare NYC subset ===")
    print("Config: 50 users from Foursquare NYC, default uncertainty.")
    src = "public_dataset/foursquare/nyc"
    dst = "forge_ds/results/probe3_data"
    _subset_foursquare(src, dst, num_users=50)
    cells = _cells_for("probe3",
                       ["markov", "prophet", "neural_tpp", "beam_tpp", "constrained_tpp"],
                       dataset_tag="foursquare_nyc_50", uncertainty="default")
    rows = _run_probe(cells, dst,
                      "forge_ds/results/probe3_results.csv",
                      "forge_ds/results/cache_probe3", num_scenarios=3)
    if rows:
        by_algo = {r["algorithm"]: r for r in rows}
        prophet_sn = by_algo.get("prophet", {}).get("sales_norm")
        print(f"\n  Prophet sales_norm on Foursquare: {prophet_sn}")
        if prophet_sn is not None:
            if prophet_sn > 0:
                print("  Verdict: Prophet recovers on Foursquare (warmup-length diagnosis confirmed)")
            else:
                print("  Verdict: Prophet still fails. Issue is not just warmup length.")
    return rows


def main():
    print("Running three probes. Estimated ~30 minutes.")
    p1 = probe1_a3_no_uncertainty()
    p2 = probe2_constraint_binding()
    p3 = probe3_prophet_foursquare()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if p1:
        by_algo = {r["algorithm"]: r["sales_norm"] for r in p1}
        ordered_pass = (by_algo.get("markov", 0) >= 0
                        and by_algo.get("prophet", 0) >= by_algo.get("markov", 0)
                        and by_algo.get("neural_tpp", 0) >= by_algo.get("prophet", 0))
        print(f"  A3 at no-uncertainty: {'PASS' if ordered_pass else 'FAIL'}")
        for a in ("markov", "prophet", "neural_tpp", "beam_tpp", "constrained_tpp"):
            v = by_algo.get(a, "n/a")
            print(f"    {a:18s} sales_norm = {v}")

    if p2:
        r = p2[0]
        print(f"  Constraint binding at high uncertainty: "
              f"softfallback = {int(r['softfallback_invocations'])}")

    if p3:
        by_algo = {r["algorithm"]: r["sales_norm"] for r in p3}
        prophet_sn = by_algo.get("prophet")
        print(f"  Prophet on Foursquare subset: sales_norm = {prophet_sn}")
        for a in ("markov", "prophet", "neural_tpp", "beam_tpp", "constrained_tpp"):
            v = by_algo.get(a, "n/a")
            print(f"    {a:18s} sales_norm = {v}")


if __name__ == "__main__":
    main()
