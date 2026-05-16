"""CSV and JSON writers for the eight output files (spec1 Section 9)."""
from __future__ import annotations

import csv
import hashlib
import os
from datetime import date, timedelta
from typing import List, Dict
import numpy as np

from config import Config, SEG_NAMES
from world import Population
from uncertainty import AbsenceEvent, ChurnEvent
from simulate import CallEvent


def _date_str(start_date: date, day_idx: int) -> str:
    return (start_date + timedelta(days=day_idx)).isoformat()


def _fmt_time(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


def write_config(cfg: Config, out_dir: str) -> str:
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        f.write(cfg.to_json())
    h = hashlib.sha256(cfg.to_json().encode()).hexdigest()[:16]
    with open(os.path.join(out_dir, "run_id.txt"), "w") as f:
        f.write(h + "\n")
    return h


def write_population(pop: Population, start_date: date, out_dir: str) -> None:
    with open(os.path.join(out_dir, "population.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rep_id", "type", "force_id", "panel_size", "hire_date", "departure_date"])
        for rep in pop.reps:
            hire = _date_str(start_date, rep.hire_date_idx)
            dep = "" if rep.departure_date_idx < 0 else _date_str(start_date, rep.departure_date_idx)
            w.writerow([rep.rep_id, rep.rep_type, rep.force_id, rep.panel_size, hire, dep])


def write_accounts(account_specialty, account_segment_initial, eligibility, out_dir):
    with open(os.path.join(out_dir, "accounts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["account_id", "specialty_id", "initial_segment", "eligible_brands"])
        for a in range(account_specialty.shape[0]):
            elig = ";".join(str(b) for b in np.where(eligibility[a])[0])
            w.writerow([a, int(account_specialty[a]),
                        SEG_NAMES[int(account_segment_initial[a])], elig])


def write_panels(pop: Population, start_date: date, out_dir: str) -> None:
    with open(os.path.join(out_dir, "panels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rep_id", "account_id", "assignment_start_date", "assignment_end_date"])
        for rep in pop.reps:
            start = _date_str(start_date, rep.hire_date_idx)
            end = "" if rep.departure_date_idx < 0 else _date_str(start_date, rep.departure_date_idx)
            for a in rep.panel:
                w.writerow([rep.rep_id, int(a), start, end])


def write_segment_history(history: List[dict], start_date: date, out_dir: str) -> None:
    with open(os.path.join(out_dir, "segment_history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["account_id", "effective_date", "old_segment", "new_segment"])
        for row in history:
            w.writerow([
                row["account_id"],
                _date_str(start_date, row["effective_day"]),
                row["old_segment"] if row["old_segment"] is not None else "",
                row["new_segment"],
            ])


def write_activity_log(actual: List[CallEvent], greedy: List[CallEvent],
                       naive: List[CallEvent], start_date: date, out_dir: str) -> None:
    with open(os.path.join(out_dir, "activity_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "event_id", "date", "rep_id", "start_time",
            "planned_duration_min", "actual_duration_min", "account_id",
            "segment_at_call", "brand_id", "brand_priority", "outcome", "scenario_id",
        ])
        eid = 0
        for tag, calls in (("actual", actual), ("greedy_upper", greedy), ("naive", naive)):
            for c in calls:
                w.writerow([
                    eid, _date_str(start_date, c.date_idx), c.rep_id,
                    _fmt_time(c.start_minute), c.planned_duration,
                    c.actual_duration, c.account_id,
                    SEG_NAMES[c.segment_at_call], c.brand_id,
                    f"{c.brand_priority:.4f}", c.outcome, tag,
                ])
                eid += 1


def write_uncertainty_traces(account_blocks: List[dict],
                             rep_absences: List[AbsenceEvent],
                             churn_events: List[ChurnEvent],
                             start_date: date, out_dir: str,
                             scenario_id: str = "actual") -> None:
    with open(os.path.join(out_dir, "uncertainty_traces.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "trace_id", "event_start_date", "entity_type", "entity_id",
            "event_type", "notice_days", "duration_days", "scenario_id",
        ])
        tid = 0
        for blk in account_blocks:
            w.writerow([
                tid, _date_str(start_date, blk["event_start_day"]),
                "account", blk["entity_id"], "unavailable",
                blk["notice_days"], blk["duration_days"], scenario_id,
            ])
            tid += 1
        for ev in rep_absences:
            w.writerow([
                tid, _date_str(start_date, ev.start_day),
                "rep", ev.rep_id, ev.event_type,
                ev.notice_days, ev.duration_days, scenario_id,
            ])
            tid += 1
        for ch in churn_events:
            w.writerow([
                tid, _date_str(start_date, ch.announce_day),
                "rep", ch.rep_id, "churn_announce",
                0, ch.depart_day - ch.announce_day, scenario_id,
            ])
            tid += 1
            w.writerow([
                tid, _date_str(start_date, ch.depart_day),
                "rep", ch.rep_id, "churn_depart",
                ch.depart_day - ch.announce_day, 0, scenario_id,
            ])
            tid += 1


def write_validation_stats(stats: List[Dict], out_dir: str) -> None:
    with open(os.path.join(out_dir, "validation_stats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric_name", "scope", "value"])
        for row in stats:
            w.writerow([row["metric_name"], row["scope"], row["value"]])
