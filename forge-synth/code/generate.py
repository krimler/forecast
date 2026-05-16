"""Top-level pipeline and CLI.

Builds the world, plans uncertainty, runs the daily loop for every rep in
parallel, generates the reference plans, writes the eight output files,
and produces a stats summary.

Usage:
    python generate.py                  # default config
    python generate.py --smoke          # smoke config (20 reps, 120 days)
    python generate.py --workers 8 --out output_default
    python generate.py --config my.json --out output_run
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Dict, List, Tuple

import numpy as np

from config import (
    Config, REP_TYPES, check_feasibility,
)
from world import build_world, build_population, Rep
from uncertainty import (
    build_availability, trace_account_blocks,
    build_planned_absences, plan_churn, AbsenceEvent,
)
from simulate import (
    CallEvent, build_segment_timeline, simulate_rep,
    greedy_upper_bound, naive_plan,
)
from output import (
    write_config, write_population, write_accounts, write_panels,
    write_segment_history, write_activity_log,
    write_uncertainty_traces, write_validation_stats,
)


def _active_window(rep: Rep, horizon: int) -> Tuple[int, int]:
    start = rep.hire_date_idx
    end = horizon if rep.departure_date_idx < 0 else rep.departure_date_idx + 1
    return (start, min(end, horizon))


def _run_one_rep(args):
    """Worker function. Pickled and sent to a child process."""
    (cfg, rep, force, eligibility, segment_per_day, availability,
     planned_abs, start_date, horizon, active_window) = args
    calls, sick = simulate_rep(
        cfg, rep, force, eligibility, segment_per_day, availability,
        planned_abs, start_date, horizon, active_window)
    greedy = greedy_upper_bound(
        cfg, rep, force, eligibility, segment_per_day, availability,
        planned_abs, sick, start_date, horizon, active_window)
    naive = naive_plan(
        cfg, rep, force, eligibility, segment_per_day, availability,
        planned_abs, sick, start_date, horizon, active_window)
    return rep.rep_id, calls, sick, greedy, naive


def run(cfg: Config, out_dir: str = None) -> Dict[str, object]:
    if out_dir is None:
        out_dir = cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.perf_counter()
    warnings = check_feasibility(cfg)

    start_date = date.fromisoformat(cfg.start_date)
    horizon = cfg.horizon_days

    world = build_world(cfg)
    pop = build_population(cfg, world)
    churn_events = plan_churn(cfg, pop, horizon)

    planned_by_rep: Dict[int, List[AbsenceEvent]] = {
        rep.rep_id: build_planned_absences(cfg, rep.rep_id, start_date, horizon)
        for rep in pop.reps
    }
    availability = build_availability(cfg, world.account_specialty.shape[0], horizon)
    segment_per_day, segment_history = build_segment_timeline(
        cfg, world.account_segment_initial, start_date, horizon)

    rep_args = [
        (cfg, rep, pop.forces[rep.force_id], world.eligibility,
         segment_per_day, availability, planned_by_rep[rep.rep_id],
         start_date, horizon, _active_window(rep, horizon))
        for rep in pop.reps
    ]

    all_calls: List[CallEvent] = []
    all_sick: List[AbsenceEvent] = []
    all_greedy: List[CallEvent] = []
    all_naive: List[CallEvent] = []

    if cfg.n_workers > 1:
        n = min(cfg.n_workers, len(rep_args))
        with ProcessPoolExecutor(max_workers=n) as ex:
            futures = [ex.submit(_run_one_rep, a) for a in rep_args]
            for fut in as_completed(futures):
                _, calls, sick, gr, nv = fut.result()
                all_calls.extend(calls)
                all_sick.extend(sick)
                all_greedy.extend(gr)
                all_naive.extend(nv)
    else:
        for a in rep_args:
            _, calls, sick, gr, nv = _run_one_rep(a)
            all_calls.extend(calls)
            all_sick.extend(sick)
            all_greedy.extend(gr)
            all_naive.extend(nv)

    # Stable order is what makes parallel == serial (validation D3).
    sort_key = lambda c: (c.date_idx, c.rep_id, c.start_minute)
    all_calls.sort(key=sort_key)
    all_greedy.sort(key=sort_key)
    all_naive.sort(key=sort_key)
    all_sick.sort(key=lambda e: (e.rep_id, e.start_day))

    run_id = write_config(cfg, out_dir)
    write_population(pop, start_date, out_dir)
    write_accounts(world.account_specialty, world.account_segment_initial,
                   world.eligibility, out_dir)
    write_panels(pop, start_date, out_dir)
    write_segment_history(segment_history, start_date, out_dir)
    write_activity_log(all_calls, all_greedy, all_naive, start_date, out_dir)

    flat_planned = [ev for evs in planned_by_rep.values() for ev in evs]
    write_uncertainty_traces(
        trace_account_blocks(availability, cfg),
        flat_planned + all_sick, churn_events,
        start_date, out_dir,
    )

    write_validation_stats(_validation_stats(cfg, pop, all_calls,
                                             flat_planned, all_sick,
                                             availability, churn_events,
                                             start_date),
                           out_dir)

    return {
        "run_id": run_id,
        "elapsed_s": time.perf_counter() - t0,
        "warnings": warnings,
        "calls_actual": len(all_calls),
        "calls_greedy": len(all_greedy),
        "calls_naive": len(all_naive),
        "sick_events": len(all_sick),
        "planned_absences": len(flat_planned),
        "churn_events": len(churn_events),
        "_inmem": {
            "world": world, "pop": pop, "availability": availability,
            "segment_per_day": segment_per_day, "segment_history": segment_history,
            "calls_actual": all_calls, "calls_greedy": all_greedy,
            "calls_naive": all_naive, "sick_events": all_sick,
            "planned_absences": flat_planned, "churn_events": churn_events,
        },
    }


def _validation_stats(cfg, pop, calls, planned, sick, availability, churn,
                      start_date) -> List[Dict[str, object]]:
    """Build the validation_stats.csv rows (Section 9.8 summary)."""
    stats: List[Dict[str, object]] = []
    n_active = sum(1 for r in pop.reps if r.replacement_of == -1)
    years = cfg.horizon_days / 365.0

    # Calls per rep per year, grouped by rep type.
    per_type: Dict[str, list] = {t: [] for t in REP_TYPES}
    rep_to_type = {r.rep_id: r.rep_type for r in pop.reps}
    per_rep: Dict[int, int] = {}
    for c in calls:
        per_rep[c.rep_id] = per_rep.get(c.rep_id, 0) + 1
    for rid, cnt in per_rep.items():
        per_type[rep_to_type[rid]].append(cnt)
    for t, arr in per_type.items():
        arr = arr or [0]
        stats.append({"metric_name": f"calls_per_rep_per_year::{t}",
                      "scope": "actual",
                      "value": float(np.mean(arr) / max(0.01, years))})
        stats.append({"metric_name": f"calls_per_rep_std::{t}",
                      "scope": "actual", "value": float(np.std(arr))})

    # Absence rates by type.
    targets = {"sick": cfg.sick_days_per_year_mean,
               "personal": cfg.personal_days_per_year_mean,
               "vacation": cfg.vacation_days_per_year_mean,
               "conference": cfg.conference_days_per_year_mean}
    all_events = planned + sick
    for tname in targets:
        days = sum(ev.duration_days for ev in all_events if ev.event_type == tname)
        stats.append({
            "metric_name": f"absence_days_per_year::{tname}",
            "scope": "actual",
            "value": float(days / max(1, n_active) / max(0.01, years)),
        })

    stats.append({"metric_name": "account_unavail_rate", "scope": "actual",
                  "value": float(1.0 - availability.mean())})
    stats.append({"metric_name": "realized_churn_rate", "scope": "actual",
                  "value": float(len(churn) / max(1, n_active))})

    dow_counts = np.zeros(7, dtype=int)
    for c in calls:
        dow_counts[(start_date + timedelta(days=c.date_idx)).weekday()] += 1
    total = max(1, dow_counts.sum())
    for i, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        stats.append({"metric_name": f"dow_share::{name}",
                      "scope": "actual",
                      "value": float(dow_counts[i] / total)})
    return stats


def _parse_cli():
    p = argparse.ArgumentParser(description="ForgeSynth generator")
    p.add_argument("--config", default=None, help="JSON file of Config overrides")
    p.add_argument("--out", default=None, help="output directory")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--smoke", action="store_true",
                   help="small-scale smoke config")
    return p.parse_args()


def main():
    args = _parse_cli()
    cfg = (Config(seed=42, horizon_days=120, warmup_days=30,
                  num_reps=20, num_accounts_total=2000)
           if args.smoke else Config())
    cfg.n_workers = args.workers
    if args.config:
        with open(args.config) as f:
            for k, v in json.load(f).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
    if args.out:
        cfg.output_dir = args.out

    info = run(cfg, args.out or cfg.output_dir)
    info.pop("_inmem", None)
    print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
