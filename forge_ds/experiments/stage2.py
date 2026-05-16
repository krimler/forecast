"""Stage 2 runner: LP vs Markov vs Naive on three datasets.

Sidesteps the original harness (which had per-window pandas overhead at
default scale) by pre-building per-rep snapshots once at dataset load
and reusing them across all rolling windows. Parallel over reps.

Outputs forge_ds/results/stage2.csv with one row per (cell, seed) and
a printed summary.

Run:
    python forge_ds/stage2.py
    python forge_ds/stage2.py --algorithms lp markov naive --seeds 42 43
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parent           # forge_ds/
ROOT = PKG.parent           # project root
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))
FORGE_SYNTH = ROOT / "forge-synth" / "code"
if str(FORGE_SYNTH) not in sys.path:
    sys.path.insert(0, str(FORGE_SYNTH))

from algorithms.lp_assignment import (
    RepSnapshot, WindowMasks, solve_window, _greedy_fallback,
    SEG_VALUE, SEG_TARGET, SEG_LIFT_A, SEG_LIFT_B,
    expected_duration, DURATION_PROBS, DURATION_VALUES,
)


# ----------------------------------------------------------------------
# Dataset loading and per-rep snapshots.
# ----------------------------------------------------------------------

DATASETS = {
    "forge_synth_default": str(ROOT / "forge-synth/dataset/output_default"),
    "foursquare_nyc":      str(ROOT / "public_dataset/foursquare/nyc"),
    "foursquare_tokyo":    str(ROOT / "public_dataset/foursquare/tokyo"),
}


def _dataset_path(tag: str, seed: int) -> str:
    """Per-seed routing: prefer <base>_s{seed} if it exists, else fall back."""
    base = DATASETS[tag]
    seeded = f"{base}_s{seed}"
    if Path(seeded, "config.json").exists():
        return seeded
    return base


def _load_dataset(dataset_dir: str):
    p = Path(dataset_dir)
    cfg = json.loads((p / "config.json").read_text())
    pop = pd.read_csv(p / "population.csv")
    accts = pd.read_csv(p / "accounts.csv")
    panels = pd.read_csv(p / "panels.csv")
    act = pd.read_csv(p / "activity_log.csv")
    u = pd.read_csv(p / "uncertainty_traces.csv")
    return cfg, pop, accts, panels, act, u


def _build_snapshots(pop: pd.DataFrame, accts: pd.DataFrame,
                      panels: pd.DataFrame, act: pd.DataFrame
                      ) -> Dict[int, RepSnapshot]:
    """One RepSnapshot per rep. Eligibility / segments / panel are static
    for the whole horizon."""
    # Per-account: segment and eligible_brands.
    seg_of_acct: Dict[int, str] = {}
    elig_of_acct: Dict[int, List[int]] = {}
    for _, row in accts.iterrows():
        a = int(row["account_id"])
        seg_of_acct[a] = str(row["initial_segment"])
        raw = row["eligible_brands"]
        s = "" if pd.isna(raw) else str(raw)
        elig_of_acct[a] = [int(x) for x in s.split(";") if x.strip().isdigit()]

    # Per-force bag and priorities from the activity_log "actual" scenario.
    actual = act[act["scenario_id"] == "actual"]
    pop_ix = pop.set_index("rep_id")
    rep_to_force = {int(r): int(pop_ix.loc[r, "force_id"]) for r in pop_ix.index}
    merged = actual.merge(pop[["rep_id", "force_id"]], on="rep_id")
    force_brand_priority: Dict[int, Dict[int, float]] = {}
    for fid, grp in merged.groupby("force_id"):
        mean_pri = grp.groupby("brand_id")["brand_priority"].mean()
        force_brand_priority[int(fid)] = {int(b): float(p)
                                           for b, p in mean_pri.items()}

    snaps: Dict[int, RepSnapshot] = {}
    panel_groups = panels.groupby("rep_id")
    for rep_id, grp in panel_groups:
        rep_id = int(rep_id)
        rep_row = pop_ix.loc[rep_id]
        fid = int(rep_row["force_id"])
        bag_priority = force_brand_priority.get(fid, {})
        bag = sorted(bag_priority.keys())
        panel_ids = grp["account_id"].astype(int).tolist()
        elig_in_bag: Dict[int, List[int]] = {}
        seg_in_panel: Dict[int, str] = {}
        for a in panel_ids:
            in_bag = [b for b in elig_of_acct.get(a, []) if b in bag]
            elig_in_bag[a] = in_bag
            seg_in_panel[a] = seg_of_acct.get(a, "B")
        snaps[rep_id] = RepSnapshot(
            rep_id=rep_id, rep_type=str(rep_row["type"]),
            force_id=fid, panel=panel_ids, bag=bag,
            priorities={int(b): float(p) for b, p in bag_priority.items()},
            eligibility=elig_in_bag, segments=seg_in_panel,
        )
    return snaps


def _build_visibility(u: pd.DataFrame, start_date: date, horizon: int):
    """Pre-flatten uncertainty traces into per-day lookup tables keyed by
    entity_id, with the reveal date computed.

    Returns:
        rep_absent_by_rep[(rep_id)] = list of (reveal_day, start_day, end_day)
        acct_unavail_by_acct[(account_id)] = list of (reveal_day, start_day, end_day)
    """
    rep_abs: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    acct_un: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    u = u.copy()
    if u.empty:
        return rep_abs, acct_un
    u["_d"] = pd.to_datetime(u["event_start_date"]).dt.date
    u["_start_day"] = u["_d"].apply(lambda d: (d - start_date).days)
    u["_reveal_day"] = u["_start_day"] - u["notice_days"].astype(int)
    u["_end_day"] = u["_start_day"] + u["duration_days"].astype(int)
    for _, row in u.iterrows():
        eid = int(row["entity_id"])
        rec = (int(row["_reveal_day"]), int(row["_start_day"]),
               int(row["_end_day"]))
        if row["entity_type"] == "rep":
            rep_abs[eid].append(rec)
        else:
            acct_un[eid].append(rec)
    return rep_abs, acct_un


def _masks_for_window(snap: RepSnapshot, window_start_day: int, n_days: int,
                       rep_abs, acct_un, n_a_b_prior):
    """Compute the boolean masks visible to the algorithm at window start."""
    rep_absent = [False] * n_days
    for reveal, s, e in rep_abs.get(snap.rep_id, []):
        if reveal > window_start_day:
            continue
        for d in range(max(s, window_start_day),
                       min(e, window_start_day + n_days)):
            rep_absent[d - window_start_day] = True
    acct_unavail: Dict[int, List[bool]] = {}
    for a in snap.panel:
        events = acct_un.get(a, [])
        if not events:
            continue
        mask = [False] * n_days
        for reveal, s, e in events:
            if reveal > window_start_day:
                continue
            for d in range(max(s, window_start_day),
                           min(e, window_start_day + n_days)):
                mask[d - window_start_day] = True
        if any(mask):
            acct_unavail[a] = mask
    return WindowMasks(
        rep_absent_day=rep_absent,
        account_unavail_day=acct_unavail,
        n_a_b_prior=dict(n_a_b_prior),
    )


# ----------------------------------------------------------------------
# Algorithms.
# ----------------------------------------------------------------------

def algo_naive(snap: RepSnapshot, masks: WindowMasks, n_days: int,
               rng_seed: int) -> List[Dict]:
    """Cadence-based: per-segment annualized target / 252 working days per call.
    Capped by daily capacity."""
    out: List[Dict] = []
    n_a_b = dict(masks.n_a_b_prior)
    rng = np.random.default_rng(rng_seed)
    cadence_days = {s: max(1, int(252 / SEG_TARGET[s])) for s in SEG_TARGET}
    next_due: Dict[int, int] = {}
    for i, a in enumerate(snap.panel):
        seg = snap.segments.get(a, "B")
        next_due[a] = i % cadence_days[seg]
    for d in range(n_days):
        used = 0.0
        called_today: set = set()
        order = sorted(snap.panel, key=lambda a: (next_due.get(a, 9999), a))
        for a in order:
            if next_due.get(a, 9999) > d:
                continue
            if masks.account_unavail_day.get(a, [False] * n_days)[d]:
                next_due[a] = d + cadence_days[snap.segments.get(a, "B")]
                continue
            seg = snap.segments.get(a, "B")
            elig = [b for b in snap.eligibility.get(a, []) if b in snap.bag]
            if not elig:
                continue
            # Pick brand by priority.
            pris = [snap.priorities.get(b, 0.0) for b in elig]
            tot = sum(pris) or 1.0
            b = int(rng.choice(elig, p=np.array(pris) / tot))
            dur = expected_duration(snap.rep_type, seg)
            if used + dur > snap.capacity_min_per_day:
                break
            out.append({"day": d, "account_id": a, "brand_id": b,
                        "duration": int(round(dur))})
            called_today.add(a)
            used += dur
            n_a_b[(a, b)] = n_a_b.get((a, b), 0) + 1
            next_due[a] = d + cadence_days[seg]
    return out


def algo_markov_vm(snap: RepSnapshot, masks: WindowMasks, n_days: int,
                   rng_seed: int, model: dict,
                   daily_call_budget: int = 12) -> List[Dict]:
    """Volume-matched Markov.

    Same per-rep account-frequency model as algo_markov, but the daily
    call count comes from a fixed budget (12 by default, matching the
    LP's effective volume) rather than the empirical p(calls | dow)
    distribution. Still capped by face-time and panel availability.
    """
    out: List[Dict] = []
    rng = np.random.default_rng(rng_seed)
    p_acct = model.get("p_acct", {})
    if not p_acct:
        return out
    accts = list(p_acct.keys())
    probs = np.array([p_acct[a] for a in accts], dtype=float)
    if probs.sum() <= 0:
        return out
    probs = probs / probs.sum()

    for d in range(n_days):
        if masks.rep_absent_day[d]:
            continue
        used = 0.0
        called_today: set = set()
        n_made = 0
        attempts = 0
        max_attempts = daily_call_budget * 6   # safety against tight panels
        while (n_made < daily_call_budget
               and used < snap.capacity_min_per_day
               and attempts < max_attempts):
            attempts += 1
            a = int(rng.choice(accts, p=probs))
            if a in called_today:
                continue
            if masks.account_unavail_day.get(a, [False] * n_days)[d]:
                continue
            elig = [b for b in snap.eligibility.get(a, []) if b in snap.bag]
            if not elig:
                continue
            pris = [snap.priorities.get(b, 0.0) for b in elig]
            tot = sum(pris) or 1.0
            b = int(rng.choice(elig, p=np.array(pris) / tot))
            dur = expected_duration(snap.rep_type, snap.segments.get(a, "B"))
            if used + dur > snap.capacity_min_per_day:
                break
            out.append({"day": d, "account_id": a, "brand_id": b,
                        "duration": int(round(dur))})
            called_today.add(a)
            used += dur
            n_made += 1
    return out


def algo_markov(snap: RepSnapshot, masks: WindowMasks, n_days: int,
                rng_seed: int, model: dict) -> List[Dict]:
    """Per-rep Markov: P(calls/day | dow) from model, then sample accounts
    from per-rep account frequency."""
    out: List[Dict] = []
    rng = np.random.default_rng(rng_seed)
    panel = snap.panel
    p_acct = model.get("p_acct", {})
    p_calls = model.get("p_calls_dow", {})
    if not p_acct:
        return out

    accts = list(p_acct.keys())
    probs = np.array([p_acct[a] for a in accts], dtype=float)
    if probs.sum() <= 0:
        return out
    probs = probs / probs.sum()

    for d in range(n_days):
        if masks.rep_absent_day[d]:
            continue
        dow = (model["window_start_dow"] + d) % 7
        n_calls_dist = p_calls.get(dow, {0: 1.0})
        ks = list(n_calls_dist.keys())
        ps = np.array([n_calls_dist[k] for k in ks], dtype=float)
        n_calls = int(rng.choice(ks, p=ps / ps.sum()))
        used = 0.0
        called_today: set = set()
        for _ in range(n_calls):
            if used >= snap.capacity_min_per_day:
                break
            # Sample an account weighted by p_acct, with rejection for
            # already-called / unavailable.
            for _try in range(8):
                a = int(rng.choice(accts, p=probs))
                if a in called_today:
                    continue
                if masks.account_unavail_day.get(a, [False] * n_days)[d]:
                    continue
                break
            else:
                continue
            elig = [b for b in snap.eligibility.get(a, []) if b in snap.bag]
            if not elig:
                continue
            pris = [snap.priorities.get(b, 0.0) for b in elig]
            tot = sum(pris) or 1.0
            b = int(rng.choice(elig, p=np.array(pris) / tot))
            dur = expected_duration(snap.rep_type, snap.segments.get(a, "B"))
            if used + dur > snap.capacity_min_per_day:
                break
            out.append({"day": d, "account_id": a, "brand_id": b,
                        "duration": int(round(dur))})
            called_today.add(a)
            used += dur
    return out


def train_markov(actual_warmup: pd.DataFrame, snap: RepSnapshot) -> dict:
    """Fit per-rep account distribution and per-dow call-count distribution
    from warmup events. Force-pooled when the rep has too little history.
    """
    sub = actual_warmup[actual_warmup["rep_id"] == snap.rep_id]
    if len(sub) < 30:
        # Force-pool over the rep's force.
        sub = actual_warmup[actual_warmup["force_id"] == snap.force_id]
    if sub.empty:
        return {"p_acct": {}, "p_calls_dow": {}, "window_start_dow": 0}
    # P(account) over the rep's panel; uniform fallback elsewhere.
    panel_set = set(snap.panel)
    counts = sub[sub["account_id"].isin(panel_set)]["account_id"].value_counts()
    p_acct = {int(a): int(c) + 1 for a, c in counts.items()}  # Laplace
    # Cover panel accounts with no history.
    for a in snap.panel:
        p_acct.setdefault(int(a), 1)
    # P(calls/day | dow).
    sub2 = sub.copy()
    sub2["_d"] = pd.to_datetime(sub2["date"]).dt.date
    sub2["_dow"] = pd.to_datetime(sub2["date"]).dt.weekday
    rep_day = sub2.groupby(["rep_id", "_d", "_dow"]).size().reset_index(name="n")
    p_calls_dow: Dict[int, Dict[int, float]] = {}
    for dow, grp in rep_day.groupby("_dow"):
        c = Counter(int(x) for x in grp["n"].values)
        s = sum(c.values()) + len(c)
        p_calls_dow[int(dow)] = {k: (v + 1) / s for k, v in c.items()}
    return {"p_acct": p_acct, "p_calls_dow": p_calls_dow,
            "window_start_dow": 0}


# ----------------------------------------------------------------------
# Per-cell run.
# ----------------------------------------------------------------------

def _run_one_rep(args):
    (algo, snap, window_starts_days, window_size, rep_abs, acct_un,
     warmup_history, eval_start_day, seed) = args
    rng = np.random.default_rng((seed, snap.rep_id))
    rep_seed = int(rng.integers(0, 2 ** 31 - 1))

    if algo in ("markov", "markov_vm"):
        model = train_markov(warmup_history, snap)
    else:
        model = None

    all_calls: List[Dict] = []
    n_a_b: Dict[Tuple[int, int], int] = {}
    fb = 0
    cur_year = None

    for w_start_day in window_starts_days:
        n_days = window_size
        if algo == "markov":
            # Refresh dow on each window.
            model["window_start_dow"] = (w_start_day) % 7
        # Reset n_a_b at calendar year boundary (matches spec1 §8.2).
        # Approximate by zeroing every 252 working days (~365 calendar).
        if w_start_day // 365 != (w_start_day - window_size) // 365:
            n_a_b = {}
        masks = _masks_for_window(snap, w_start_day, n_days, rep_abs, acct_un,
                                  n_a_b)
        if algo == "lp":
            calls, fallback = solve_window(snap, masks, n_days,
                                            time_limit_sec=30)
            if fallback:
                fb += 1
        elif algo == "naive":
            calls = algo_naive(snap, masks, n_days, rep_seed + w_start_day)
            fallback = False
        elif algo == "markov_vm":
            calls = algo_markov_vm(snap, masks, n_days,
                                    rep_seed + w_start_day, model)
            fallback = False
        else:  # markov
            calls = algo_markov(snap, masks, n_days, rep_seed + w_start_day,
                                model)
            fallback = False

        # Translate per-window day index back to global day index.
        for c in calls:
            c["day"] = w_start_day + c["day"]
            c["rep_id"] = snap.rep_id
            n_a_b[(c["account_id"], c["brand_id"])] = \
                n_a_b.get((c["account_id"], c["brand_id"]), 0) + 1
            all_calls.append(c)

    return all_calls, fb


def _replay_against_actual(calls: List[Dict], rep_abs, acct_un):
    """Mark each planned call no_show if rep absent or account unavail on
    its day. Returns the productive subset and the failure attribution."""
    productive = []
    failures = {"rep_absent": 0, "account_unavail": 0, "both": 0}
    for c in calls:
        day = c["day"]
        rid = c["rep_id"]
        a = c["account_id"]
        rep_bad = any(s <= day < e for _, s, e in rep_abs.get(rid, []))
        acct_bad = any(s <= day < e for _, s, e in acct_un.get(a, []))
        if rep_bad and acct_bad:
            failures["both"] += 1
            c["outcome"] = "no_show"
        elif rep_bad:
            failures["rep_absent"] += 1
            c["outcome"] = "no_show"
        elif acct_bad:
            failures["account_unavail"] += 1
            c["outcome"] = "no_show"
        else:
            c["outcome"] = "completed"
            productive.append(c)
    return productive, failures


def _absolute_sales(productive: List[Dict], snap_of_rep: Dict[int, RepSnapshot]):
    """Sales(P) = sum_{a,b} v_seg * pi_b * lift(n_{a,b}, seg)."""
    counts: Dict[Tuple[int, int, int], int] = {}  # (rep, a, b) -> n
    seg_of_acct: Dict[int, str] = {}
    for c in productive:
        rep = c["rep_id"]
        a, b = c["account_id"], c["brand_id"]
        counts[(rep, a, b)] = counts.get((rep, a, b), 0) + 1
        if a not in seg_of_acct:
            seg_of_acct[a] = snap_of_rep[rep].segments.get(a, "B")

    # Aggregate per (account, brand) globally for Sales calculation.
    n_a_b: Dict[Tuple[int, int], int] = {}
    pi_b: Dict[int, float] = {}
    for (rep, a, b), n in counts.items():
        n_a_b[(a, b)] = n_a_b.get((a, b), 0) + n
        pi_b[b] = max(pi_b.get(b, 0.0), snap_of_rep[rep].priorities.get(b, 0.0))

    total = 0.0
    for (a, b), n in n_a_b.items():
        seg = seg_of_acct.get(a, "B")
        lift = SEG_LIFT_A[seg] * (1.0 - math.exp(-SEG_LIFT_B[seg] * n))
        total += SEG_VALUE[seg] * pi_b.get(b, 0.0) * lift
    return total


def _coverage(productive: List[Dict], snaps: Dict[int, RepSnapshot]) -> float:
    """Priority-weighted reach across all brands. eligibility from snaps."""
    brand_called: Dict[int, set] = defaultdict(set)
    for c in productive:
        brand_called[c["brand_id"]].add(c["account_id"])

    brand_priority: Dict[int, float] = {}
    eligible: Dict[int, set] = defaultdict(set)
    for s in snaps.values():
        for b, p in s.priorities.items():
            if p > brand_priority.get(b, 0.0):
                brand_priority[b] = p
        for a, brands in s.eligibility.items():
            for b in brands:
                eligible[b].add(a)

    total = 0.0
    total_pi = 0.0
    for b, p in brand_priority.items():
        eb = len(eligible.get(b, set()))
        if eb == 0:
            continue
        cov = len(brand_called.get(b, set())) / eb
        total += p * cov
        total_pi += p
    return 0.0 if total_pi <= 0 else total / total_pi


# ----------------------------------------------------------------------
# Cell-level driver.
# ----------------------------------------------------------------------

def run_cell(algo: str, dataset_tag: str, seed: int,
             window_size: int = 14, workers: int = 6,
             ds_overrides: Optional[dict] = None) -> dict:
    print(f"\n=== {algo} on {dataset_tag} seed={seed} ===")
    t0 = time.perf_counter()
    ds_path = _dataset_path(dataset_tag, seed)
    print(f"  dataset path: {ds_path}")
    cfg, pop, accts, panels, act, u = _load_dataset(ds_path)
    start_date = date.fromisoformat(str(cfg["start_date"]))
    horizon = int(cfg["horizon_days"])
    warmup = int(cfg["warmup_days"])

    snaps = _build_snapshots(pop, accts, panels, act)
    print(f"  loaded {len(snaps)} reps")

    rep_abs, acct_un = _build_visibility(u, start_date, horizon)

    # Warmup events for Markov training.
    actual_full = act[act["scenario_id"] == "actual"].copy()
    actual_full["_d"] = pd.to_datetime(actual_full["date"]).dt.date
    warmup_history = actual_full[actual_full["_d"] <
                                  start_date + timedelta(days=warmup)]
    warmup_history = warmup_history.merge(pop[["rep_id", "force_id"]], on="rep_id")

    eval_start_day = warmup
    window_starts = list(range(eval_start_day, horizon, window_size))

    # Per-rep parallel jobs.
    args = []
    for rid, snap in snaps.items():
        args.append((algo, snap, window_starts, window_size,
                     rep_abs, acct_un, warmup_history, eval_start_day, seed))

    all_calls: List[Dict] = []
    total_fallback = 0
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_run_one_rep, a) for a in args]
            for k, fut in enumerate(as_completed(futs)):
                calls, fb = fut.result()
                all_calls.extend(calls)
                total_fallback += fb
                if (k + 1) % 100 == 0:
                    print(f"  {k+1}/{len(args)} reps done, fallback={total_fallback}")
    else:
        for k, a in enumerate(args):
            calls, fb = _run_one_rep(a)
            all_calls.extend(calls)
            total_fallback += fb
            if (k + 1) % 50 == 0:
                print(f"  {k+1}/{len(args)} reps done")

    pred_time = time.perf_counter() - t0
    print(f"  prediction done in {pred_time:.0f}s, "
          f"{len(all_calls)} planned calls, fallbacks={total_fallback}")

    # Replay against actual uncertainty.
    productive, failures = _replay_against_actual(all_calls, rep_abs, acct_un)
    disr_rate = 1.0 - len(productive) / max(1, len(all_calls))
    print(f"  productive {len(productive)}, disruption rate {disr_rate:.3f}")

    # Metrics.
    sales_abs = _absolute_sales(productive, snaps)
    coverage = _coverage(productive, snaps)

    # Sales of greedy and naive from the dataset's stored scenarios (for
    # ForgeSynth these are meaningful; for Foursquare we still report them
    # but mark sales_norm as not meaningful).
    sales_greedy = _sales_of_scenario(act, "greedy_upper", snaps)
    sales_naive = _sales_of_scenario(act, "naive", snaps)
    if dataset_tag.startswith("forge_synth") and sales_greedy > sales_naive:
        sales_norm = (sales_abs - sales_naive) / max(1e-9, sales_greedy - sales_naive)
        sales_norm_note = "ok"
    else:
        sales_norm = float("nan")
        sales_norm_note = "not meaningful on Foursquare (greedy ceiling artifact)"

    # Robustness: ratio and absolute loss against the stored "actual"
    # uncertainty (a single sample). We don't resample uncertainty here
    # because Stage 2 is method comparison, not stochastic disruption study.
    # ratio = productive_sales / planned_sales_assuming_no_disruption.
    sales_planned = _absolute_sales_assuming_completed(all_calls, snaps)
    rob_ratio = sales_abs / max(1e-9, sales_planned)
    rob_abs_loss = sales_planned - sales_abs

    total_time = time.perf_counter() - t0
    print(f"  finished {algo}/{dataset_tag}/seed={seed} in {total_time:.0f}s")

    return {
        "algorithm": algo, "dataset": dataset_tag, "seed": seed,
        "n_reps": len(snaps),
        "horizon_days": horizon, "warmup_days": warmup,
        "n_planned_calls": len(all_calls),
        "n_productive_calls": len(productive),
        "sales_abs": round(sales_abs, 4),
        "sales_planned": round(sales_planned, 4),
        "sales_greedy": round(sales_greedy, 4),
        "sales_naive": round(sales_naive, 4),
        "sales_norm": (round(sales_norm, 6)
                       if not math.isnan(sales_norm) else "nan"),
        "sales_norm_note": sales_norm_note,
        "coverage": round(coverage, 6),
        "robustness_ratio": round(rob_ratio, 6),
        "robustness_abs_loss": round(rob_abs_loss, 4),
        "disruption_rate": round(disr_rate, 6),
        "failures_rep_absent": failures["rep_absent"],
        "failures_account_unavail": failures["account_unavail"],
        "failures_both": failures["both"],
        "lp_fallback_invocations": total_fallback,
        "prediction_time_sec": round(pred_time, 1),
        "total_time_sec": round(total_time, 1),
    }


def _sales_of_scenario(act: pd.DataFrame, scenario: str,
                       snaps: Dict[int, RepSnapshot]) -> float:
    df = act[act["scenario_id"] == scenario]
    if df.empty:
        return 0.0
    calls = []
    for _, r in df.iterrows():
        calls.append({"rep_id": int(r["rep_id"]),
                      "account_id": int(r["account_id"]),
                      "brand_id": int(r["brand_id"]),
                      "outcome": str(r["outcome"])})
    prod = [c for c in calls if c["outcome"] != "no_show"]
    return _absolute_sales(prod, snaps)


def _absolute_sales_assuming_completed(calls: List[Dict],
                                       snaps: Dict[int, RepSnapshot]) -> float:
    """Treat every planned call as completed (no disruption replay)."""
    fake = [{"rep_id": c["rep_id"], "account_id": c["account_id"],
             "brand_id": c["brand_id"], "outcome": "completed"} for c in calls]
    return _absolute_sales(fake, snaps)


# ----------------------------------------------------------------------
# CLI.
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algorithms", nargs="+",
                   default=["lp", "markov", "naive"],
                   choices=["lp", "markov", "markov_vm", "naive"])
    p.add_argument("--datasets", nargs="+",
                   default=list(DATASETS.keys()),
                   choices=list(DATASETS.keys()))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--log", default="forge_ds/results/stage2.csv")
    args = p.parse_args()

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    new_file = not Path(args.log).exists()

    rows = []
    for ds in args.datasets:
        for algo in args.algorithms:
            for seed in args.seeds:
                try:
                    r = run_cell(algo, ds, seed, workers=args.workers)
                    rows.append(r)
                    with open(args.log, "a", newline="") as f:
                        cols = list(r.keys())
                        w = csv.DictWriter(f, fieldnames=cols)
                        if new_file:
                            w.writeheader()
                            new_file = False
                        w.writerow(r)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"FAILED {algo}/{ds}/{seed}: {e}")

    # Print summary table.
    if not rows:
        return
    print("\n" + "=" * 100)
    print("Stage 2 summary")
    print("=" * 100)
    print(f"{'algo':8s} {'dataset':25s} {'seed':5s} "
          f"{'sales_abs':>11s} {'coverage':>10s} "
          f"{'rob_ratio':>10s} {'rob_loss':>11s} "
          f"{'disr':>7s} {'fbk':>5s} {'wall':>7s}")
    for r in rows:
        print(f"{r['algorithm']:8s} {r['dataset']:25s} {r['seed']:5d} "
              f"{r['sales_abs']:11.1f} {r['coverage']:10.4f} "
              f"{r['robustness_ratio']:10.4f} {r['robustness_abs_loss']:11.1f} "
              f"{r['disruption_rate']:7.3f} {r['lp_fallback_invocations']:5d} "
              f"{r['total_time_sec']:7.0f}s")


if __name__ == "__main__":
    main()
