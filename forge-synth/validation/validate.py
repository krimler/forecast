"""Full validation suite for Runs every check in the sign-off list (S, C, T, D, M) plus the feasibility
rejection test. Writes a JSON summary alongside the outputs.

Run as a script:
    python validate.py
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

# The code modules live in a sibling directory. Insert that on sys.path so
# both this process and any spawned multiprocessing workers can import them.
_HERE = Path(__file__).resolve().parent
_CODE_DIR = (_HERE.parent / "code").resolve()
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from config import (
    Config, REP_TYPES, DOW_MULTIPLIER, FeasibilityError,
)
from world import Population
from uncertainty import (
    AbsenceEvent, build_availability, is_weekend,
)
from simulate import CallEvent
import metrics
from generate import run


# Smoke config keeps the validation fast (around a minute for the full suite).
SMOKE_KWARGS = dict(
    seed=42, horizon_days=120, warmup_days=30,
    num_reps=20, num_accounts_total=2000,
)

# All scratch outputs from validation land under forge-synth/dataset/.
DATASET_DIR = (_HERE.parent / "dataset").resolve()


def _dataset_path(name: str) -> str:
    return str(DATASET_DIR / name)


def smoke_config(**overrides) -> Config:
    cfg = Config(**SMOKE_KWARGS, output_dir=_dataset_path("output_smoke"))
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def zero_uncertainty_config() -> Config:
    return smoke_config(
        p_account_unavail=0.0,
        sick_days_per_year_mean=0.0,
        personal_days_per_year_mean=0.0,
        vacation_days_per_year_mean=0.0,
        conference_days_per_year_mean=0.0,
        p_churn_annual=0.0,
        output_dir=_dataset_path("output_zero"),
    )


# ---- Schema checks (S1, S2, S3).

EXPECTED_COLS = {
    "config.json": None,
    "population.csv": ["rep_id", "type", "force_id", "panel_size", "hire_date", "departure_date"],
    "accounts.csv": ["account_id", "specialty_id", "initial_segment", "eligible_brands"],
    "panels.csv": ["rep_id", "account_id", "assignment_start_date", "assignment_end_date"],
    "segment_history.csv": ["account_id", "effective_date", "old_segment", "new_segment"],
    "activity_log.csv": ["event_id", "date", "rep_id", "start_time",
                         "planned_duration_min", "actual_duration_min", "account_id",
                         "segment_at_call", "brand_id", "brand_priority", "outcome", "scenario_id"],
    "uncertainty_traces.csv": ["trace_id", "event_start_date", "entity_type", "entity_id",
                               "event_type", "notice_days", "duration_days", "scenario_id"],
    "validation_stats.csv": ["metric_name", "scope", "value"],
}


def validate_schemas(out_dir: str):
    results = []
    for fname, expected in EXPECTED_COLS.items():
        path = os.path.join(out_dir, fname)
        if not os.path.exists(path):
            results.append((f"S1::{fname}", False, "missing file"))
            continue
        if expected is None:
            results.append((f"S1::{fname}", True, "ok"))
            continue
        with open(path) as f:
            header = next(csv.reader(f), None)
        ok = header == expected
        results.append((f"S1::{fname}", ok, "ok" if ok else f"header mismatch {header}"))
    return results


def validate_orphans(out_dir: str):
    with open(os.path.join(out_dir, "population.csv")) as f:
        rep_ids = {row["rep_id"] for row in csv.DictReader(f)}
    with open(os.path.join(out_dir, "accounts.csv")) as f:
        acct_ids = {row["account_id"] for row in csv.DictReader(f)}
    orphan_reps = orphan_accts = 0
    with open(os.path.join(out_dir, "activity_log.csv")) as f:
        for row in csv.DictReader(f):
            if row["rep_id"] not in rep_ids:
                orphan_reps += 1
            if row["account_id"] not in acct_ids:
                orphan_accts += 1
    return [
        ("S2::activity_log.rep_id", orphan_reps == 0, f"{orphan_reps} orphans"),
        ("S2::activity_log.account_id", orphan_accts == 0, f"{orphan_accts} orphans"),
    ]


def validate_date_ranges(cfg: Config, out_dir: str):
    start = date.fromisoformat(cfg.start_date)
    end = start + timedelta(days=cfg.horizon_days)
    bad = 0
    with open(os.path.join(out_dir, "activity_log.csv")) as f:
        for row in csv.DictReader(f):
            d = date.fromisoformat(row["date"])
            if not (start <= d < end):
                bad += 1
    return [("S3::activity_log", bad == 0, f"{bad} out-of-range")]


# ---- Constraint checks (C1..C5).

def validate_constraints(cfg, pop, calls_actual, planned, sick, availability):
    results = []
    absent: Dict[int, set] = {}
    for ev in planned + sick:
        days = ev.absent_day_indices or list(range(ev.start_day, ev.start_day + ev.duration_days))
        for d in days:
            absent.setdefault(ev.rep_id, set()).add(d)

    c1 = sum(1 for c in calls_actual if c.date_idx in absent.get(c.rep_id, ()))
    results.append(("C1", c1 == 0, f"{c1} calls on absent days"))

    c2 = sum(1 for c in calls_actual
             if c.outcome != "no_show" and not availability[c.account_id, c.date_idx])
    results.append(("C2", c2 == 0, f"{c2} calls to unavailable accounts"))

    seen: Dict[Tuple[int, int, int], bool] = {}
    c3 = 0
    for c in calls_actual:
        key = (c.rep_id, c.date_idx, c.account_id)
        if key in seen:
            c3 += 1
        seen[key] = True
    results.append(("C3", c3 == 0, f"{c3} duplicates"))

    max_ms = max(rt["mu"] + 3 * rt["sigma"] for rt in REP_TYPES.values())
    cap = int(max_ms * max(DOW_MULTIPLIER) + 1)
    per_day: Dict[Tuple[int, int], int] = {}
    for c in calls_actual:
        per_day[(c.rep_id, c.date_idx)] = per_day.get((c.rep_id, c.date_idx), 0) + 1
    c4 = sum(1 for v in per_day.values() if v > cap)
    results.append(("C4", c4 == 0, f"{c4} days over cap ({cap})"))

    c5 = sum(1 for c in calls_actual
             if not (cfg.day_start_minute <= c.start_minute < cfg.day_end_minute))
    results.append(("C5", c5 == 0, f"{c5} out-of-hours starts"))
    return results


# ---- Statistical checks (T1..T4).

def validate_statistics(cfg, pop, calls_actual, planned, sick,
                        availability, segment_history):
    results = []
    years = cfg.horizon_days / 365.0
    n_active = sum(1 for r in pop.reps if r.replacement_of == -1)
    sample_scale = max(1.0, n_active * max(years, 0.01))

    targets = {
        "sick": cfg.sick_days_per_year_mean,
        "personal": cfg.personal_days_per_year_mean,
        "vacation": cfg.vacation_days_per_year_mean,
        "conference": cfg.conference_days_per_year_mean,
    }
    all_events = planned + sick
    # Tolerance is a 2-sigma binomial CI per event type. At the spec's
    # default scale (1000 reps x 365 d) this tightens to about 5%, which
    # is the spec's stated target.
    for tname, target in targets.items():
        obs_days = sum(ev.duration_days for ev in all_events if ev.event_type == tname)
        per_rep_per_year = obs_days / max(1, n_active) / max(0.01, years)
        rel = abs(per_rep_per_year - target) / max(1e-9, target)
        n_expected = max(1.0, sample_scale * target)
        tol = max(0.05, 2.0 / np.sqrt(n_expected))
        results.append((f"T1::{tname}", rel <= tol,
                        f"observed {per_rep_per_year:.2f} vs target {target:.2f} "
                        f"(rel {rel:.2%}, tol {tol:.2%})"))

    if cfg.p_account_unavail > 0:
        rate = 1.0 - availability.mean()
        rel = abs(rate - cfg.p_account_unavail) / cfg.p_account_unavail
        results.append(("T2", rel <= 0.20,
                        f"observed {rate:.4f} vs target {cfg.p_account_unavail:.4f}"))
    else:
        results.append(("T2", True, "p_account_unavail=0"))

    n_accounts = max(1, len(set(int(h["account_id"]) for h in segment_history)))
    transitions = [h for h in segment_history if h["old_segment"] is not None]
    rate_per_year = len(transitions) / n_accounts / max(0.01, years)
    results.append(("T3", 0.03 <= rate_per_year <= 0.30,
                    f"{rate_per_year:.2%} accounts shifted/yr"))

    dow_counts = np.zeros(7, dtype=int)
    start = date.fromisoformat(cfg.start_date)
    for c in calls_actual:
        dow_counts[(start + timedelta(days=c.date_idx)).weekday()] += 1
    total = dow_counts.sum()
    if total > 0:
        m = np.array(DOW_MULTIPLIER[:5])
        expected = m / m.sum()
        observed = dow_counts[:5] / total
        diff = float(np.abs(observed - expected).max())
        results.append(("T4", diff <= 0.15, f"max |obs-exp|={diff:.3f}"))
    else:
        results.append(("T4", True, "no calls"))
    return results


# ---- Top-level runner.

def main() -> int:
    results: List[Tuple[str, bool, str]] = []

    cfg = smoke_config()
    if os.path.isdir(cfg.output_dir):
        shutil.rmtree(cfg.output_dir)
    info = run(cfg, cfg.output_dir)
    inm = info["_inmem"]

    results += validate_schemas(cfg.output_dir)
    results += validate_orphans(cfg.output_dir)
    results += validate_date_ranges(cfg, cfg.output_dir)
    results += validate_constraints(
        cfg, inm["pop"], inm["calls_actual"],
        inm["planned_absences"], inm["sick_events"], inm["availability"],
    )
    results += validate_statistics(
        cfg, inm["pop"], inm["calls_actual"],
        inm["planned_absences"], inm["sick_events"], inm["availability"],
        inm["segment_history"],
    )

    # D1: same seed -> byte-identical files.
    cfg2 = smoke_config(output_dir=_dataset_path("output_smoke_rerun"))
    if os.path.isdir(cfg2.output_dir):
        shutil.rmtree(cfg2.output_dir)
    run(cfg2, cfg2.output_dir)
    d1 = all(
        (Path(cfg.output_dir) / f).read_bytes() == (Path(cfg2.output_dir) / f).read_bytes()
        for f in ["activity_log.csv", "population.csv", "accounts.csv",
                  "panels.csv", "uncertainty_traces.csv", "segment_history.csv"]
    )
    results.append(("D1", d1, "byte-identical" if d1 else "files differ"))

    # D2: different seed -> different output.
    cfg3 = smoke_config(seed=123, output_dir=_dataset_path("output_smoke_seed123"))
    if os.path.isdir(cfg3.output_dir):
        shutil.rmtree(cfg3.output_dir)
    run(cfg3, cfg3.output_dir)
    a = (Path(cfg.output_dir) / "activity_log.csv").read_bytes()
    b = (Path(cfg3.output_dir) / "activity_log.csv").read_bytes()
    results.append(("D2", a != b, "differ" if a != b else "identical (BAD)"))

    # D3: parallel run matches serial.
    cfg4 = smoke_config(output_dir=_dataset_path("output_smoke_par"))
    cfg4.n_workers = 4
    if os.path.isdir(cfg4.output_dir):
        shutil.rmtree(cfg4.output_dir)
    run(cfg4, cfg4.output_dir)
    par = (Path(cfg.output_dir) / "activity_log.csv").read_bytes()
    ser = (Path(cfg4.output_dir) / "activity_log.csv").read_bytes()
    results.append(("D3", par == ser, "byte-identical" if par == ser else "parallel != serial"))

    # Metrics M1..M6.
    eval_window = range(cfg.warmup_days, cfg.horizon_days)
    seg_init = inm["world"].account_segment_initial
    sales_actual = metrics.sales(inm["calls_actual"], inm["pop"],
                                 eval_window=eval_window, account_segment=seg_init)
    sales_greedy = metrics.sales(inm["calls_greedy"], inm["pop"],
                                 eval_window=eval_window, account_segment=seg_init)
    sales_naive = metrics.sales(inm["calls_naive"], inm["pop"],
                                eval_window=eval_window, account_segment=seg_init)
    sn_star, _ = metrics.sales_norm(sales_greedy, sales_greedy, sales_naive)
    sn_naive, _ = metrics.sales_norm(sales_naive, sales_greedy, sales_naive)
    results.append(("M1", abs(sn_star - 1.0) < 1e-6, f"SalesNorm(P*)={sn_star:.6f}"))
    results.append(("M2", abs(sn_naive - 0.0) < 1e-6, f"SalesNorm(P0)={sn_naive:.6f}"))

    cov = metrics.coverage(inm["calls_actual"], inm["pop"],
                           inm["world"].eligibility, eval_window=eval_window)
    results.append(("M3", 0.0 <= cov <= 1.0, f"Coverage={cov:.4f}"))

    # M4: Robustness = 1 when there is no uncertainty.
    cfgz = zero_uncertainty_config()
    if os.path.isdir(cfgz.output_dir):
        shutil.rmtree(cfgz.output_dir)
    info_z = run(cfgz, cfgz.output_dir)
    inm_z = info_z["_inmem"]
    rob_z = metrics.robustness(
        inm_z["calls_actual"], inm_z["pop"], inm_z["world"].eligibility,
        inm_z["segment_per_day"], [],
        _planned_by_rep(inm_z["planned_absences"]),
        _sick_by_rep(inm_z["sick_events"]),
        cfgz, eval_window=range(cfgz.warmup_days, cfgz.horizon_days),
    )
    results.append(("M4", abs(rob_z.ratio - 1.0) < 1e-9,
                    f"Robustness@0 ratio={rob_z.ratio:.6f} flagged={rob_z.flagged}"))

    # M5: Robustness should decrease as p_account_unavail goes up.
    rob_curve = []
    for p in [0.05, 0.10, 0.20]:
        c = smoke_config(p_account_unavail=p, output_dir=_dataset_path(f"output_p{p}"))
        if os.path.isdir(c.output_dir):
            shutil.rmtree(c.output_dir)
        info_p = run(c, c.output_dir)
        inm_p = info_p["_inmem"]
        samples = []
        for i in range(5):
            cc = replace(c, seed=c.seed + 10000 + i)
            samples.append(build_availability(
                cc, inm_p["world"].account_specialty.shape[0], c.horizon_days))
        rob = metrics.robustness(
            inm_p["calls_actual"], inm_p["pop"], inm_p["world"].eligibility,
            inm_p["segment_per_day"], samples,
            _planned_by_rep(inm_p["planned_absences"]),
            _sick_by_rep(inm_p["sick_events"]),
            c, eval_window=range(c.warmup_days, c.horizon_days),
        )
        rob_curve.append((p, rob.ratio, rob.absolute_loss, rob.flagged))
    vals = [r[1] for r in rob_curve]
    m5_ok = all(vals[i] >= vals[i + 1] - 0.05 for i in range(len(vals) - 1))
    results.append(("M5", m5_ok, f"curve={rob_curve}"))

    # M6: source-decomposed disruption counts add up to the total.
    decomp = metrics.decompose_disruptions(
        inm["calls_actual"], inm["availability"],
        inm["planned_absences"], inm["sick_events"], eval_window=eval_window,
    )
    total = sum(1 for c in inm["calls_actual"]
                if c.date_idx in eval_window and c.outcome != "completed")
    results.append(("M6", sum(decomp.values()) == total,
                    f"total={total} sum={sum(decomp.values())} {decomp}"))

    # Feasibility rejection: 10 accounts is below max panel size.
    bad_cfg = smoke_config(num_accounts_total=10, output_dir=_dataset_path("output_bad"))
    try:
        run(bad_cfg, bad_cfg.output_dir)
        feas_ok = False
        msg = "did not raise"
    except FeasibilityError:
        feas_ok = True
        msg = "rejected"
    results.append(("FEAS_REJECT", feas_ok, msg))

    # Print and save.
    ok = sum(1 for _, p, _ in results if p)
    bad = len(results) - ok
    print("=" * 60)
    print("ForgeSynth validation results")
    print("=" * 60)
    for name, passed, detail in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:30s}  {detail}")
    print("=" * 60)
    print(f"{ok} passed, {bad} failed")

    with open(Path(__file__).resolve().parent / "validation_results.json", "w") as f:
        json.dump([(n, bool(p), str(d)) for n, p, d in results], f, indent=2)

    return 0 if bad == 0 else 1


def _planned_by_rep(events: List[AbsenceEvent]) -> Dict[int, set]:
    out: Dict[int, set] = {}
    for ev in events:
        for d in (ev.absent_day_indices
                  or range(ev.start_day, ev.start_day + ev.duration_days)):
            out.setdefault(ev.rep_id, set()).add(d)
    return out


def _sick_by_rep(events: List[AbsenceEvent]) -> Dict[int, set]:
    out: Dict[int, set] = {}
    for ev in events:
        out.setdefault(ev.rep_id, set()).add(ev.start_day)
    return out


if __name__ == "__main__":
    sys.exit(main())
