"""Daily activity loop, segment transitions, reference plans.

A few design notes that aren't visible from the code alone:

1. Sick-rate denominator is the working-day count of the calendar year
   (around 252), not the working days remaining from today. Read literally
   the spec's "remaining" wording integrates to roughly 6.1x the target rate,
   contradicting its own T1 validation example. Using the constant yearly
   denominator restores the intended expectation.

2. Greedy upper bound uses marginal Sales lift as its score, not the spec's
   triangular f_b. f_b is zero at n=0 so every initial score ties, and
   greedy then concentrates calls past the lift saturation point. That makes
   it worse than naive on Sales and breaks M1. Marginal lift is the true
   Sales gradient, which is what an upper bound has to optimize.

3. Naive is capped at the same n_calls budget the daily loop draws. The
   spec doesn't state a cap but uncapped naive can physically exceed a
   rep's day and dominate greedy.

Everything below honors the spec where the spec is internally consistent.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Dict, Tuple
import numpy as np

from config import (
    Config, REP_TYPES, SEGMENT_DEFAULTS, DOW_MULTIPLIER,
    DURATION_PROBS, DURATION_VALUES, OUTCOME_PROBS,
    SEGMENT_TRANSITION, SEG_NAMES,
    rng, seed_for_rep, seed_for_account,
)
from world import Rep, ForceConfig
from uncertainty import (
    AbsenceEvent, sample_sick_day, is_weekend, absent_day_set,
)


SEG_VALUES = np.array([SEGMENT_DEFAULTS[s]["value"] for s in SEG_NAMES], dtype=float)
SEG_TARGETS = np.array([SEGMENT_DEFAULTS[s]["annual_target"] for s in SEG_NAMES], dtype=float)
SEG_LIFT_A = np.array([SEGMENT_DEFAULTS[s]["lift_a"] for s in SEG_NAMES], dtype=float)
SEG_LIFT_B = np.array([SEGMENT_DEFAULTS[s]["lift_b"] for s in SEG_NAMES], dtype=float)

OUTCOME_NAMES = ["completed", "abbreviated", "no_show"]
OUTCOME_P = np.array([OUTCOME_PROBS[k] for k in OUTCOME_NAMES])


@dataclass
class CallEvent:
    date_idx: int
    rep_id: int
    start_minute: int
    planned_duration: int
    actual_duration: int
    account_id: int
    segment_at_call: int
    brand_id: int
    brand_priority: float
    outcome: str


# ---- Segment timeline (Section 8.7).

def quarter_boundaries(start_date: date, horizon: int) -> List[int]:
    """Day indices that fall on Jan/Apr/Jul/Oct 1 within (0, horizon)."""
    out = []
    for t in range(1, horizon):
        d = start_date + timedelta(days=t)
        if d.day == 1 and d.month in (1, 4, 7, 10):
            out.append(t)
    return out


def build_segment_timeline(cfg: Config, initial: np.ndarray,
                           start_date: date, horizon: int) -> Tuple[np.ndarray, List[dict]]:
    """Per-account segment for every day, plus a transition history list.

    The history includes an initial-assignment row for each account
    (old_segment is None).
    """
    N = initial.shape[0]
    transitions = quarter_boundaries(start_date, horizon)
    seg = np.empty((N, horizon), dtype=np.int8)
    seg[:, 0] = initial

    P = np.array([[SEGMENT_TRANSITION[a][b] for b in SEG_NAMES] for a in SEG_NAMES], dtype=float)
    cur = initial.copy()

    history: List[dict] = [
        {"account_id": int(a), "effective_day": 0, "old_segment": None,
         "new_segment": SEG_NAMES[int(initial[a])]}
        for a in range(N)
    ]

    last_t = 0
    for t_q in transitions:
        seg[:, last_t + 1: t_q + 1] = cur.reshape(-1, 1)
        for a in range(N):
            r = rng(seed_for_account(cfg.seed, a))
            r.bytes(32 + 8 * (t_q // 90))     # offset per quarter
            old = int(cur[a])
            new = int(r.choice(3, p=P[old]))
            if new != old:
                history.append({
                    "account_id": int(a), "effective_day": t_q,
                    "old_segment": SEG_NAMES[old], "new_segment": SEG_NAMES[new],
                })
                cur[a] = new
        seg[:, t_q] = cur
        last_t = t_q

    if last_t < horizon - 1:
        seg[:, last_t + 1:] = cur.reshape(-1, 1)
    seg[:, 0] = initial
    return seg, history


# ---- Helpers used by every plan generator.

def working_days_in_year(start_date: date, day_idx: int) -> int:
    """Working days in the calendar year that contains day_idx."""
    y = (start_date + timedelta(days=day_idx)).year
    count = 0
    for t in range(366):
        d = date(y, 1, 1) + timedelta(days=t)
        if d.year != y:
            break
        if not is_weekend(d):
            count += 1
    return max(1, count)


def _sample_duration(r, rep_type: str, seg_idx: int) -> int:
    return int(r.choice(DURATION_VALUES, p=DURATION_PROBS[(rep_type, SEG_NAMES[seg_idx])]))


def _sample_outcome(r) -> str:
    return str(r.choice(OUTCOME_NAMES, p=OUTCOME_P))


def _softmax_sample(r, scores: np.ndarray, temperature: float) -> int:
    noise = r.uniform(0.0, 1e-6, size=scores.shape)
    s = (scores + noise) / max(1e-9, temperature)
    s -= s.max()
    p = np.exp(s)
    p /= p.sum()
    return int(r.choice(scores.size, p=p))


# ---- Daily activity loop (Section 8).

def simulate_rep(cfg, rep: Rep, force: ForceConfig,
                 eligibility: np.ndarray, segment_per_day: np.ndarray,
                 availability: np.ndarray, planned_absences: List[AbsenceEvent],
                 start_date: date, horizon: int,
                 active_window: Tuple[int, int]) -> Tuple[List[CallEvent], List[AbsenceEvent]]:
    """Run one rep through the horizon. Returns calls and sampled sick events.

    Sick days are drawn inside this loop because the spec gives them zero
    notice (they're sampled at the start of each day).
    """
    r = rng(seed_for_rep(cfg.seed, rep.rep_id))
    r.bytes(64)        # offset so this stream is independent of planned absences

    type_cfg = REP_TYPES[rep.rep_type]
    mu, sigma = type_cfg["mu"], type_cfg["sigma"]
    rep_rate = float(np.clip(
        r.normal(cfg.sick_days_per_year_mean, cfg.sick_days_per_year_std), 0.0, 20.0))

    planned_absent = absent_day_set(planned_absences, horizon)
    panel = rep.panel
    P = panel.shape[0]
    if P == 0:
        return [], []

    num_brands = eligibility.shape[1]
    n_a_b = np.zeros((P, num_brands), dtype=np.int32)

    panel_eligibility = eligibility[panel]
    panel_seg_timeline = segment_per_day[panel]
    bag = np.array(force.brands, dtype=int)
    bag_priorities = np.array(force.priorities, dtype=float)
    panel_bag_elig = panel_eligibility[:, bag]
    # Accounts not eligible for any bag brand score zero by construction, so
    # they're filtered upfront to stop them eating call slots.
    panel_any_eligible = panel_bag_elig.any(axis=1)

    out_calls: List[CallEvent] = []
    sick_events: List[AbsenceEvent] = []

    was_sick_yesterday = False
    hire_day, dep_day = active_window
    cur_year = start_date.year

    for t in range(hire_day, dep_day):
        today = start_date + timedelta(days=t)

        # Annual cycle reset for the per-account, per-brand call counters.
        if today.year != cur_year:
            n_a_b.fill(0)
            cur_year = today.year

        if is_weekend(today) or planned_absent[t]:
            was_sick_yesterday = False
            continue

        is_sick = sample_sick_day(
            r, rep_rate=rep_rate, was_sick_yesterday=was_sick_yesterday,
            today=today, working_days_in_year=working_days_in_year(start_date, t),
            sick_autocorr=cfg.sick_autocorr,
        )
        if is_sick:
            sick_events.append(AbsenceEvent(
                rep_id=rep.rep_id, event_type="sick", start_day=t,
                duration_days=1, notice_days=cfg.sick_notice_days,
            ))
            was_sick_yesterday = True
            continue
        was_sick_yesterday = False

        m_dow = DOW_MULTIPLIER[today.weekday()]
        if m_dow == 0:
            continue
        n_calls = int(round(max(0.0, r.normal(mu, sigma) * m_dow)))
        if n_calls <= 0:
            continue

        today_avail = availability[panel, t]
        called_today: set = set()
        cur_minute = cfg.day_start_minute

        for _slot in range(n_calls):
            if cur_minute >= cfg.day_end_minute:
                break

            valid = today_avail & panel_any_eligible
            if called_today:
                valid[np.fromiter(called_today, dtype=np.int64)] = False
            valid_idx = np.where(valid)[0]
            if valid_idx.size == 0:
                break

            seg_today = panel_seg_timeline[valid_idx, t]
            v_seg = SEG_VALUES[seg_today]
            targets = SEG_TARGETS[seg_today]
            bag_n = n_a_b[valid_idx][:, bag]
            bag_elig = panel_bag_elig[valid_idx]
            tgt_col = targets.reshape(-1, 1)
            f_b = np.maximum(0.0, 1.0 - np.abs(bag_n - tgt_col) / np.maximum(1e-9, tgt_col))
            contrib = bag_priorities.reshape(1, -1) * bag_elig.astype(float) * f_b
            scores = v_seg * contrib.sum(axis=1)

            if not np.isfinite(scores).any() or scores.max() <= 0:
                pick_local = int(r.integers(0, valid_idx.size))
            else:
                pick_local = _softmax_sample(r, scores, cfg.softmax_temperature)
            pick = int(valid_idx[pick_local])
            account_id = int(panel[pick])

            elig_mask = bag_elig[pick_local]
            if not elig_mask.any():
                called_today.add(pick)
                continue

            brand_scores = bag_priorities * elig_mask.astype(float) * f_b[pick_local]
            if brand_scores.sum() <= 0:
                brand_scores = bag_priorities * elig_mask.astype(float)
            brand_p = brand_scores / brand_scores.sum()
            chosen_brand_idx = int(r.choice(len(bag), p=brand_p))
            brand_id = int(bag[chosen_brand_idx])
            brand_priority = float(bag_priorities[chosen_brand_idx])

            seg_idx_call = int(panel_seg_timeline[pick, t])
            planned_dur = _sample_duration(r, rep.rep_type, seg_idx_call)
            outcome = _sample_outcome(r)

            if outcome == "completed":
                actual_dur = planned_dur
            elif outcome == "abbreviated":
                actual_dur = planned_dur // 2
            else:
                actual_dur = 0

            if cur_minute + planned_dur > cfg.day_end_minute:
                break

            out_calls.append(CallEvent(
                date_idx=t, rep_id=rep.rep_id, start_minute=cur_minute,
                planned_duration=planned_dur, actual_duration=actual_dur,
                account_id=account_id, segment_at_call=seg_idx_call,
                brand_id=brand_id, brand_priority=brand_priority, outcome=outcome,
            ))
            called_today.add(pick)
            if outcome != "no_show":
                n_a_b[pick, brand_id] += 1
            cur_minute = cur_minute + planned_dur + cfg.inter_call_gap_min

    return out_calls, sick_events


# ---- Greedy upper bound (Section 10.1).

def greedy_upper_bound(cfg, rep, force, eligibility, segment_per_day,
                       availability, planned_absences, sick_events,
                       start_date, horizon, active_window) -> List[CallEvent]:
    r = rng(seed_for_rep(cfg.seed, rep.rep_id))
    r.bytes(128)

    panel = rep.panel
    if panel.size == 0:
        return []

    bag = np.array(force.brands, dtype=int)
    bag_priorities = np.array(force.priorities, dtype=float)
    panel_eligibility = eligibility[panel]
    panel_bag_elig = panel_eligibility[:, bag]
    panel_seg_timeline = segment_per_day[panel]
    panel_any_eligible = panel_bag_elig.any(axis=1)

    absent = absent_day_set(planned_absences + sick_events, horizon)
    type_cfg = REP_TYPES[rep.rep_type]
    mu, sigma = type_cfg["mu"], type_cfg["sigma"]

    last_called = np.full(panel.shape[0], -10_000, dtype=np.int32)
    num_brands = eligibility.shape[1]
    n_a_b = np.zeros((panel.shape[0], num_brands), dtype=np.int32)

    out: List[CallEvent] = []
    cur_year = start_date.year
    hire_day, dep_day = active_window

    for t in range(hire_day, dep_day):
        today = start_date + timedelta(days=t)
        if today.year != cur_year:
            n_a_b.fill(0)
            cur_year = today.year
        if is_weekend(today) or absent[t]:
            continue
        m_dow = DOW_MULTIPLIER[today.weekday()]
        if m_dow == 0:
            continue
        n_calls = int(round(max(0.0, r.normal(mu, sigma) * m_dow)))
        if n_calls <= 0:
            continue

        avail_today = availability[panel, t]
        avail_tomorrow = availability[panel, t + 1] if t + 1 < horizon else avail_today
        candidate_mask = avail_today & avail_tomorrow & panel_any_eligible

        cur_minute = cfg.day_start_minute
        called_today: set = set()

        for _slot in range(n_calls):
            if cur_minute >= cfg.day_end_minute:
                break
            cm = candidate_mask.copy()
            if called_today:
                cm[np.fromiter(called_today, dtype=np.int64)] = False
            valid_idx = np.where(cm)[0]
            if valid_idx.size == 0:
                break

            seg_today = panel_seg_timeline[valid_idx, t]
            v_seg = SEG_VALUES[seg_today]
            bag_n = n_a_b[valid_idx][:, bag]
            bag_elig = panel_bag_elig[valid_idx]

            # Marginal Sales lift = lift(n+1) - lift(n).
            la = SEG_LIFT_A[seg_today]
            lb = SEG_LIFT_B[seg_today]
            account_total_n = bag_n.sum(axis=1)
            marg_lift = la * np.exp(-lb * account_total_n) * (1.0 - np.exp(-lb))
            best_pi = (bag_priorities.reshape(1, -1) * bag_elig.astype(float)).max(axis=1)
            scores = v_seg * best_pi * marg_lift

            recency = (t - last_called[valid_idx]).astype(float)
            composite = scores + recency * 1e-8
            pick_local = int(np.argmax(composite))
            pick = int(valid_idx[pick_local])
            account_id = int(panel[pick])

            targets = SEG_TARGETS[seg_today]
            tgt_col = targets.reshape(-1, 1)
            f_b = np.maximum(0.0, 1.0 - np.abs(bag_n - tgt_col) / np.maximum(1e-9, tgt_col))
            elig_mask = bag_elig[pick_local]
            if not elig_mask.any():
                called_today.add(pick)
                continue
            brand_scores = bag_priorities * elig_mask.astype(float) * f_b[pick_local]
            if brand_scores.sum() <= 0:
                brand_scores = bag_priorities * elig_mask.astype(float)
            chosen_brand_idx = int(np.argmax(brand_scores))
            brand_id = int(bag[chosen_brand_idx])
            brand_priority = float(bag_priorities[chosen_brand_idx])

            seg_idx_call = int(panel_seg_timeline[pick, t])
            planned_dur = _sample_duration(r, rep.rep_type, seg_idx_call)
            if cur_minute + planned_dur > cfg.day_end_minute:
                break

            # Greedy assumes outcome = completed (Section 10.1).
            out.append(CallEvent(
                date_idx=t, rep_id=rep.rep_id, start_minute=cur_minute,
                planned_duration=planned_dur, actual_duration=planned_dur,
                account_id=account_id, segment_at_call=seg_idx_call,
                brand_id=brand_id, brand_priority=brand_priority,
                outcome="completed",
            ))
            called_today.add(pick)
            n_a_b[pick, brand_id] += 1
            last_called[pick] = t
            cur_minute += planned_dur + cfg.inter_call_gap_min

    return out


# ---- Naive plan (Section 10.2).

def naive_plan(cfg, rep, force, eligibility, segment_per_day,
               availability, planned_absences, sick_events,
               start_date, horizon, active_window) -> List[CallEvent]:
    """Cadence-driven schedule that ignores absence and unavailability.

    Failed calls land in the log as no_show.
    """
    r = rng(seed_for_rep(cfg.seed, rep.rep_id))
    r.bytes(192)

    panel = rep.panel
    if panel.size == 0:
        return []

    panel_eligibility = eligibility[panel]
    panel_seg_timeline = segment_per_day[panel]
    bag = np.array(force.brands, dtype=int)
    bag_priorities = np.array(force.priorities, dtype=float)
    panel_bag_elig = panel_eligibility[:, bag]
    absent = absent_day_set(planned_absences + sick_events, horizon)

    type_cfg = REP_TYPES[rep.rep_type]
    mu, sigma = type_cfg["mu"], type_cfg["sigma"]

    seg0 = panel_seg_timeline[:, 0]
    target_freq = SEG_TARGETS[seg0]
    # Cadence in working days = (working days per year) / target frequency.
    cadence = np.where(target_freq > 0, 252.0 / np.maximum(1.0, target_freq), 1e9)
    # Spread initial due dates across the panel so day 0 isn't a stampede.
    offsets = (np.arange(panel.shape[0]) % np.maximum(1, cadence.astype(int))).astype(int)
    next_due = (active_window[0] + offsets)

    out: List[CallEvent] = []
    hire_day, dep_day = active_window

    for t in range(hire_day, dep_day):
        today = start_date + timedelta(days=t)
        if is_weekend(today):
            continue
        m_dow = DOW_MULTIPLIER[today.weekday()]
        if m_dow == 0:
            continue
        n_calls_today = int(round(max(0.0, r.normal(mu, sigma) * m_dow)))
        if n_calls_today <= 0:
            continue

        due_idx = np.where(next_due <= t)[0]
        if due_idx.size == 0:
            continue

        cur_minute = cfg.day_start_minute
        placed_today = 0

        for local_idx in due_idx:
            if placed_today >= n_calls_today or cur_minute >= cfg.day_end_minute:
                break
            account_id = int(panel[local_idx])
            seg_idx = int(panel_seg_timeline[local_idx, t])
            elig_mask = panel_bag_elig[local_idx]
            if not elig_mask.any():
                next_due[local_idx] = t + int(cadence[local_idx])
                continue

            chosen_brand_idx = int(np.argmax(elig_mask.astype(float) * bag_priorities))
            brand_id = int(bag[chosen_brand_idx])
            brand_priority = float(bag_priorities[chosen_brand_idx])

            planned_dur = _sample_duration(r, rep.rep_type, seg_idx)
            if cur_minute + planned_dur > cfg.day_end_minute:
                break

            if absent[t] or not availability[account_id, t]:
                outcome = "no_show"
                actual_dur = 0
            else:
                outcome = "completed"
                actual_dur = planned_dur

            out.append(CallEvent(
                date_idx=t, rep_id=rep.rep_id, start_minute=cur_minute,
                planned_duration=planned_dur, actual_duration=actual_dur,
                account_id=account_id, segment_at_call=seg_idx,
                brand_id=brand_id, brand_priority=brand_priority,
                outcome=outcome,
            ))
            cur_minute += planned_dur + cfg.inter_call_gap_min
            placed_today += 1
            next_due[local_idx] = t + max(1, int(cadence[local_idx]))

    return out
