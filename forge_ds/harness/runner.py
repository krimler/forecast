"""Single-cell runner (spec2 §11.4).

A "cell" is one (algorithm, dataset, uncertainty_level, priority_regime,
seed) configuration. Running a cell does:

1. Load the spec1-schema dataset.
2. Instantiate the algorithm and either train or reload from cache.
3. Roll a 14-day plan across the evaluation horizon, replanning daily.
4. Replay 100 disruption scenarios against the final plan; compute
   robustness ratio.
5. Compute the five metrics.
6. Return a result row.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Forge-synth's metrics module gives us Sales, Coverage, Robustness, and the
# reference replanner. We re-key the dataset onto the same in-memory types so
# we can call those functions directly.
HERE = Path(__file__).resolve().parent
FORGE_SYNTH_CODE = HERE.parent.parent / "forge-synth" / "code"
if str(FORGE_SYNTH_CODE) not in sys.path:
    sys.path.insert(0, str(FORGE_SYNTH_CODE))

import config as fs_config
import metrics as fs_metrics
from world import Population, Rep, ForceConfig
from uncertainty import build_availability

# Our own modules.
ALG_DIR = HERE.parent
if str(ALG_DIR) not in sys.path:
    sys.path.insert(0, str(ALG_DIR))

from algorithms.base import (
    Algorithm, ActivityHistory, PlanContext, Plan, PlannedCall,
)
from algorithms.markov import MarkovAlgorithm
from algorithms.prophet_agg import ProphetAlgorithm
from algorithms.neural_tpp import NeuralTPPAlgorithm
from algorithms.beam_tpp import BeamTPPAlgorithm
from algorithms.constrained_tpp import ConstrainedTPPAlgorithm
from .matrix import Cell, UNCERTAINTY_LEVELS
from . import cache as cache_mod


ALGO_REGISTRY = {
    "markov": MarkovAlgorithm,
    "prophet": ProphetAlgorithm,
    "neural_tpp": NeuralTPPAlgorithm,
    "beam_tpp": BeamTPPAlgorithm,
    "constrained_tpp": ConstrainedTPPAlgorithm,
}


# ---- Dataset loading.

def load_dataset_dir(dataset_dir: str) -> ActivityHistory:
    """Read the eight spec1 files into an ActivityHistory."""
    p = Path(dataset_dir)
    import json
    cfg = json.loads((p / "config.json").read_text())
    events = pd.read_csv(p / "activity_log.csv")
    # Only the actual scenario is used by algorithms (greedy / naive plans
    # in the same file are reference plans, not training data).
    events = events[events["scenario_id"] == "actual"].copy()
    uncertainty = pd.read_csv(p / "uncertainty_traces.csv")
    population = pd.read_csv(p / "population.csv")
    panels = pd.read_csv(p / "panels.csv")
    accounts = pd.read_csv(p / "accounts.csv")
    return ActivityHistory(events=events, uncertainty=uncertainty,
                           population=population, panels=panels,
                           accounts=accounts, config=cfg)


def dataset_hash(dataset_dir: str) -> str:
    """SHA-256 short hash of config.json + run_id.txt."""
    parts = []
    for name in ("config.json", "run_id.txt"):
        path = Path(dataset_dir) / name
        if path.exists():
            parts.append(path.read_bytes())
    h = hashlib.sha256(b"||".join(parts)).hexdigest()[:16]
    return h


# ---- PlanContext construction per rep.

def build_plan_context(history: ActivityHistory, rep_id: int,
                       horizon_start: date, horizon_end: date) -> PlanContext:
    pop_row = history.population[history.population["rep_id"] == rep_id].iloc[0]
    panel = history.panels[history.panels["rep_id"] == rep_id][
        "account_id"].astype(int).tolist()

    # Eligibility from accounts.csv: 'eligible_brands' is ';'-separated.
    elig: Dict[int, List[int]] = {}
    segs: Dict[int, str] = {}
    sub = history.accounts[history.accounts["account_id"].isin(panel)]
    for _, row in sub.iterrows():
        a = int(row["account_id"])
        raw = row["eligible_brands"]
        s = "" if pd.isna(raw) else str(raw)
        elig[a] = [int(x) for x in s.split(";") if x.strip().isdigit()]
        segs[a] = str(row["initial_segment"])

    # Bag and priorities: take from one rep in the same force in the activity
    # log. If the rep has no events we fall back to a uniform priority over
    # whatever brands appear in the dataset.
    fid = int(pop_row["force_id"])
    bag_priorities = (history.events.merge(
        history.population[["rep_id", "force_id"]], on="rep_id")
        .query("force_id == @fid")
        .groupby("brand_id")["brand_priority"].mean())
    bag = sorted(bag_priorities.index.astype(int).tolist())
    pris = [float(bag_priorities[b]) for b in bag]
    if not bag:
        bag = sorted(history.events["brand_id"].astype(int).unique().tolist())
        pris = [1.0 / len(bag)] * len(bag) if bag else []

    # Visibility: only events strictly before horizon_start are visible.
    ev = history.events.copy()
    ev["date"] = pd.to_datetime(ev["date"]).dt.date
    visible_events = ev[ev["date"] < horizon_start].copy()

    # Visible uncertainty: rows whose notice has elapsed by horizon_start.
    u = history.uncertainty.copy()
    u["event_start_date"] = pd.to_datetime(u["event_start_date"]).dt.date
    u["notice_days"] = u["notice_days"].astype(int)
    u["reveal_date"] = u.apply(
        lambda r: r["event_start_date"] - timedelta(days=int(r["notice_days"])), axis=1)
    visible_u = u[u["reveal_date"] <= horizon_start]

    known_abs = visible_u[visible_u["entity_type"] == "rep"]
    known_unav = visible_u[visible_u["entity_type"] == "account"]

    return PlanContext(
        rep_id=int(rep_id),
        rep_type=str(pop_row["type"]),
        force_id=fid,
        panel=panel,
        bag=bag,
        priorities=pris,
        eligibility=elig,
        segments=segs,
        known_absences=known_abs,
        known_unavailable=known_unav,
        history=ActivityHistory(
            events=visible_events, uncertainty=visible_u,
            population=history.population, panels=history.panels,
            accounts=history.accounts, config=history.config,
        ),
        horizon_start=horizon_start,
        horizon_end=horizon_end,
    )


# ---- Rolling 14-day plan generation.

def generate_rolling_plan(*, algo: Algorithm, history: ActivityHistory,
                          start: date, end: date, window_days: int = 14
                          ) -> Dict[int, List[PlannedCall]]:
    """For each rep, plan once per window. Daily replan is the algorithm's
    default replan_within_window (most algorithms just drop future calls
    and let the next window predict from scratch).
    """
    rep_ids = history.population["rep_id"].astype(int).tolist()
    out: Dict[int, List[PlannedCall]] = {r: [] for r in rep_ids}
    cur = start
    while cur < end:
        window_end = min(end, cur + timedelta(days=window_days))
        this_window = (window_end - cur).days
        for rid in rep_ids:
            ctx = build_plan_context(history, rid, cur, window_end)
            plan = algo.predict_window(ctx, cur, this_window)
            # Defensive clamp: drop any call past the dataset horizon.
            plan.calls = [c for c in plan.calls if cur <= c.date < window_end]
            out[rid].extend(plan.calls)
        cur = window_end
    return out


# ---- Disruption replay against a plan.

def replay_plan(*, calls_by_rep: Dict[int, List[PlannedCall]],
                availability: np.ndarray, absences_by_rep: Dict[int, set]
                ) -> List[PlannedCall]:
    """Walk every planned call. Mark as no_show if the rep was absent on
    the day or the account was unavailable. Otherwise mark completed.
    """
    out: List[PlannedCall] = []
    for rid, calls in calls_by_rep.items():
        for c in calls:
            day_idx = (c.date - c.date.replace(month=1, day=1)).days  # only for absent check
            # Caller passes day-of-horizon index sets, so we look up the
            # day directly off the call object.
            absent = c.date in absences_by_rep.get(rid, set())
            unav = False
            if availability is not None and 0 <= getattr(c, "horizon_day", -1) < availability.shape[1]:
                unav = not bool(availability[c.account_id, c.horizon_day])
            c2 = PlannedCall(
                date=c.date, rep_id=c.rep_id, start_minute=c.start_minute,
                planned_duration=c.planned_duration, account_id=c.account_id,
                segment_at_call=c.segment_at_call, brand_id=c.brand_id,
                brand_priority=c.brand_priority,
                outcome=("no_show" if (absent or unav) else "completed"),
            )
            out.append(c2)
    return out


# ---- Sales / Coverage / Robustness via forge-synth metrics.

def plan_to_callevents(plan_calls: List[PlannedCall], start_date: date):
    """Convert our PlannedCall objects to the forge-synth CallEvent dataclass."""
    from datetime import timedelta as _td
    from simulate import CallEvent
    from config import SEG_NAMES
    seg_to_idx = {s: i for i, s in enumerate(SEG_NAMES)}
    out = []
    for c in plan_calls:
        actual = c.planned_duration if c.outcome == "completed" \
            else (c.planned_duration // 2 if c.outcome == "abbreviated" else 0)
        out.append(CallEvent(
            date_idx=(c.date - start_date).days,
            rep_id=c.rep_id, start_minute=c.start_minute,
            planned_duration=c.planned_duration, actual_duration=actual,
            account_id=c.account_id,
            segment_at_call=seg_to_idx.get(c.segment_at_call, 1),
            brand_id=c.brand_id, brand_priority=c.brand_priority,
            outcome=c.outcome,
        ))
    return out


def build_forge_population(history: ActivityHistory) -> Population:
    """Re-key dataset metadata into the forge-synth Population shape so we
    can call the forge-synth metrics functions on our plans."""
    reps: List[Rep] = []
    for _, row in history.population.iterrows():
        panel_ids = history.panels[history.panels["rep_id"] == int(row["rep_id"])][
            "account_id"].astype(int).to_numpy()
        reps.append(Rep(
            rep_id=int(row["rep_id"]),
            rep_type=str(row["type"]),
            force_id=int(row["force_id"]),
            panel=panel_ids,
        ))

    # Force configs: take average priority per brand within each force.
    brand_priority_by_force: Dict[int, Dict[int, float]] = {}
    merged = history.events.merge(history.population[["rep_id", "force_id"]],
                                  on="rep_id")
    for fid, grp in merged.groupby("force_id"):
        mean_pri = grp.groupby("brand_id")["brand_priority"].mean()
        brand_priority_by_force[int(fid)] = {int(b): float(p)
                                              for b, p in mean_pri.items()}

    forces: List[ForceConfig] = []
    for fid in sorted(brand_priority_by_force.keys()):
        brand_map = brand_priority_by_force[fid]
        bag = sorted(brand_map.keys())
        forces.append(ForceConfig(
            force_id=fid, brands=bag,
            priorities=[brand_map[b] for b in bag],
            regime="from_data",
        ))
    return Population(reps=reps, forces=forces)


# ---- Cell execution.

def run_cell(cell: Cell, *, dataset_dir: str, cache_root: str,
             num_scenarios: int = 25, window_days: int = 14,
             algo_overrides: Dict[str, object] = None) -> Dict[str, object]:
    """Run one cell end-to-end. Returns a result row dict.

    `algo_overrides` is a dict of extra kwargs passed to the algorithm
    constructor. Used by ablation probes that need to vary algorithm
    hyperparameters without touching the matrix definition.
    """
    t_start = time.perf_counter()

    history = load_dataset_dir(dataset_dir)
    ds_hash = dataset_hash(dataset_dir)
    pop = build_forge_population(history)

    cfg = history.config
    start_date = date.fromisoformat(str(cfg["start_date"]))
    horizon_days = int(cfg["horizon_days"])
    warmup_days = int(cfg["warmup_days"])
    eval_start = start_date + timedelta(days=warmup_days)
    eval_end = start_date + timedelta(days=horizon_days)

    # Warmup history for training: events with date < eval_start.
    warmup_events = history.events[
        pd.to_datetime(history.events["date"]).dt.date < eval_start].copy()
    warmup_history = ActivityHistory(
        events=warmup_events, uncertainty=history.uncertainty,
        population=history.population, panels=history.panels,
        accounts=history.accounts, config=cfg,
    )

    # Algorithm: cached if compatible.
    algo_cls = ALGO_REGISTRY[cell.algorithm]
    key = cache_mod.make_key(ds_hash, cell.algorithm, cell.seed)
    cached = cache_mod.get(cache_root, key)
    t0 = time.perf_counter()
    algo_cfg = {"seed": cell.seed}
    if algo_overrides:
        algo_cfg.update(algo_overrides)
    # Cache hits only when there are no overrides (otherwise behavior depends
    # on the override values, which aren't part of the cache key).
    if cached is not None and cell.algorithm in ("markov", "prophet") and not algo_overrides:
        algo = cached
        training_time = 0.0
    else:
        algo = algo_cls(algo_cfg)
        algo.fit(warmup_history)
        try:
            if not algo_overrides:
                cache_mod.put(cache_root, key, algo)
        except Exception:
            pass
        training_time = time.perf_counter() - t0

    # Rolling plan over evaluation window.
    t0 = time.perf_counter()
    calls_by_rep = generate_rolling_plan(
        algo=algo, history=warmup_history, start=eval_start, end=eval_end,
        window_days=window_days,
    )
    pred_time = time.perf_counter() - t0

    # Flatten into one list and compute metrics against the realized
    # availability and absences (the "actual" run from the dataset is treated
    # as one disruption sample).
    rep_absences: Dict[int, set] = {}
    u = history.uncertainty.copy()
    u["event_start_date"] = pd.to_datetime(u["event_start_date"]).dt.date
    for _, row in u[u["entity_type"] == "rep"].iterrows():
        s = row["event_start_date"]
        for k in range(int(row["duration_days"])):
            rep_absences.setdefault(int(row["entity_id"]), set()).add(
                s + timedelta(days=k))

    # Mark outcomes against the realized uncertainty in the data.
    flat: List[PlannedCall] = []
    for rid, plan in calls_by_rep.items():
        for c in plan:
            absent = c.date in rep_absences.get(rid, set())
            unav = _account_unavailable(history.uncertainty, c.account_id, c.date)
            if absent or unav:
                c.outcome = "no_show"
            else:
                c.outcome = "completed"
            flat.append(c)

    # Metric computation via forge-synth.
    eval_window_idx = range(warmup_days, horizon_days)
    cevents = plan_to_callevents(flat, start_date)

    seg_init_idx = _segment_array(history, pop)
    sales_alg = fs_metrics.sales(cevents, pop, eval_window=eval_window_idx,
                                 account_segment=seg_init_idx)

    # Greedy and naive reference plans come from the spec1 dataset itself,
    # so we read them out of the activity log.
    ref_events = pd.read_csv(Path(dataset_dir) / "activity_log.csv")
    sales_star = _sales_for_scenario(ref_events, "greedy_upper", pop,
                                     eval_window_idx, seg_init_idx, start_date)
    sales_naive = _sales_for_scenario(ref_events, "naive", pop,
                                      eval_window_idx, seg_init_idx, start_date)
    if sales_star <= 0 and sales_alg > 0:
        sales_star = max(sales_alg, sales_naive + 1e-6)
    sn, _ = fs_metrics.sales_norm(sales_alg, sales_star, sales_naive)

    cov = fs_metrics.coverage(cevents, pop, _eligibility_matrix(history, pop),
                              eval_window=eval_window_idx)

    # Robustness: sample availability matrices from the spec1 distribution
    # and replay through the fixed reference replanner.
    rob_samples = []
    base_seed = cell.seed
    for i in range(num_scenarios):
        c2 = fs_config.Config(**{**cfg, "seed": base_seed + 10000 + i})
        avail = build_availability(c2, history.accounts.shape[0],
                                   history.config["horizon_days"])
        rob_samples.append(avail)

    sick_by_rep: Dict[int, set] = {}
    planned_by_rep: Dict[int, set] = {}
    u_rep = u[u["entity_type"] == "rep"]
    for _, row in u_rep.iterrows():
        rid = int(row["entity_id"])
        for k in range(int(row["duration_days"])):
            day_idx = (row["event_start_date"] + timedelta(days=k) - start_date).days
            target = sick_by_rep if str(row["event_type"]) == "sick" else planned_by_rep
            target.setdefault(rid, set()).add(day_idx)

    # Robustness returns ratio + absolute_loss + a flag for degenerate plans.
    # An empty plan would score ratio ~ 1 (nothing to disrupt), so we read
    # both numbers and surface the flag in the log.
    rob_ratio = 1.0
    rob_loss = 0.0
    rob_base = 0.0
    rob_flagged = False
    t_replan_start = time.perf_counter()
    if cfg.get("p_account_unavail", 0.10) > 0:
        rob = fs_metrics.robustness(
            cevents, pop,
            _eligibility_matrix(history, pop),
            _segment_per_day(history, pop, horizon_days),
            rob_samples, planned_by_rep, sick_by_rep,
            fs_config.Config(**cfg), eval_window=eval_window_idx,
        )
        rob_ratio = rob.ratio
        rob_loss = rob.absolute_loss
        rob_base = rob.base_sales
        rob_flagged = rob.flagged
    replan_time = time.perf_counter() - t_replan_start

    disr = fs_metrics.disruption_rate(cevents, eval_window=eval_window_idx)
    decomp = fs_metrics.decompose_disruptions(
        cevents,
        _availability_from_data(history, horizon_days),
        _absence_events(history, "planned"),
        _absence_events(history, "sick"),
        eval_window=eval_window_idx,
    )

    total = max(1, sum(decomp.values()))
    return {
        "cell_id": cell.cell_id, "algorithm": cell.algorithm,
        "dataset": cell.dataset, "uncertainty_level": cell.uncertainty_level,
        "priority_regime": cell.priority_regime, "seed": cell.seed,
        "sales_norm": round(sn, 6), "coverage": round(cov, 6),
        "robustness": round(rob_ratio, 6),
        "robustness_absolute_loss": round(rob_loss, 6),
        "robustness_base_sales": round(rob_base, 6),
        "robustness_flagged": bool(rob_flagged),
        "disruption_rate": round(disr, 6),
        "replan_cost_median": round(replan_time / max(1, num_scenarios), 6),
        "disruption_rate_source_account": decomp.get("account_unavail", 0) / total,
        "disruption_rate_source_rep_planned": decomp.get("rep_planned", 0) / total,
        "disruption_rate_source_rep_unplanned": decomp.get("rep_unplanned", 0) / total,
        "training_time_sec": round(training_time, 3),
        "prediction_time_sec": round(pred_time, 3),
        "total_replan_time_sec": round(replan_time, 3),
        "softfallback_invocations": getattr(algo, "softfallback_invocations", 0),
        "config_hash": ds_hash,
        "git_commit": "",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ---- Helpers.

def _account_unavailable(u: pd.DataFrame, account_id: int, d: date) -> bool:
    rows = u[(u["entity_type"] == "account") & (u["entity_id"].astype(int) == account_id)]
    if rows.empty:
        return False
    for _, ev in rows.iterrows():
        s = pd.to_datetime(ev["event_start_date"]).date()
        if s <= d < s + timedelta(days=int(ev["duration_days"])):
            return True
    return False


def _segment_array(history: ActivityHistory, pop: Population) -> np.ndarray:
    seg_to_idx = {"A": 0, "B": 1, "C": 2}
    arr = np.zeros(history.accounts["account_id"].max() + 1, dtype=np.int8)
    for _, row in history.accounts.iterrows():
        arr[int(row["account_id"])] = seg_to_idx.get(str(row["initial_segment"]), 1)
    return arr


def _eligibility_matrix(history: ActivityHistory, pop: Population) -> np.ndarray:
    n_accounts = int(history.accounts["account_id"].max()) + 1
    n_brands = max(2, int(history.config.get("num_brands_total", 6)))
    M = np.zeros((n_accounts, n_brands), dtype=bool)
    for _, row in history.accounts.iterrows():
        a = int(row["account_id"])
        raw = row["eligible_brands"]
        if pd.isna(raw):
            continue
        for b in str(raw).split(";"):
            if not b.strip().isdigit():
                continue
            bi = int(b)
            if 0 <= bi < n_brands:
                M[a, bi] = True
    return M


def _segment_per_day(history: ActivityHistory, pop: Population, horizon: int) -> np.ndarray:
    """Static segment grid (no quarterly transitions on real-data adapters)."""
    seg = _segment_array(history, pop)
    return np.tile(seg.reshape(-1, 1), (1, horizon))


def _availability_from_data(history: ActivityHistory, horizon: int) -> np.ndarray:
    """Reconstruct the realized account availability from uncertainty traces."""
    n_accounts = int(history.accounts["account_id"].max()) + 1
    avail = np.ones((n_accounts, horizon), dtype=bool)
    rows = history.uncertainty[history.uncertainty["entity_type"] == "account"]
    start = date.fromisoformat(str(history.config["start_date"]))
    for _, row in rows.iterrows():
        a = int(row["entity_id"])
        s = (pd.to_datetime(row["event_start_date"]).date() - start).days
        d = int(row["duration_days"])
        if 0 <= s < horizon:
            avail[a, s: min(horizon, s + d)] = False
    return avail


def _absence_events(history: ActivityHistory, kind: str):
    """Return forge-synth AbsenceEvent objects so we can pass them to
    decompose_disruptions. kind = 'planned' (notice >= 7) or 'sick'.
    """
    from uncertainty import AbsenceEvent
    out = []
    start = date.fromisoformat(str(history.config["start_date"]))
    rows = history.uncertainty[history.uncertainty["entity_type"] == "rep"]
    for _, row in rows.iterrows():
        if kind == "sick" and str(row["event_type"]) != "sick":
            continue
        if kind == "planned" and (str(row["event_type"]) == "sick" or int(row["notice_days"]) < 7):
            continue
        s = (pd.to_datetime(row["event_start_date"]).date() - start).days
        out.append(AbsenceEvent(
            rep_id=int(row["entity_id"]),
            event_type=str(row["event_type"]),
            start_day=s, duration_days=int(row["duration_days"]),
            notice_days=int(row["notice_days"]),
        ))
    return out


def _sales_for_scenario(activity_df, scenario_id, pop, eval_window_idx,
                        account_segment, start_date) -> float:
    """Compute Sales for a named scenario stored in the spec1 activity log."""
    df = activity_df[activity_df["scenario_id"] == scenario_id]
    if df.empty:
        return 0.0
    from simulate import CallEvent
    from config import SEG_NAMES
    seg_to_idx = {s: i for i, s in enumerate(SEG_NAMES)}
    cevents = []
    for _, row in df.iterrows():
        d = pd.to_datetime(row["date"]).date()
        cevents.append(CallEvent(
            date_idx=(d - start_date).days,
            rep_id=int(row["rep_id"]),
            start_minute=0,
            planned_duration=int(row["planned_duration_min"]),
            actual_duration=int(row["actual_duration_min"]),
            account_id=int(row["account_id"]),
            segment_at_call=seg_to_idx.get(str(row["segment_at_call"]), 1),
            brand_id=int(row["brand_id"]),
            brand_priority=float(row["brand_priority"]),
            outcome=str(row["outcome"]),
        ))
    return fs_metrics.sales(cevents, pop, eval_window=eval_window_idx,
                            account_segment=account_segment)
