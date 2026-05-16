"""Uncertainty processes: account availability, rep absences, churn.

Spec1 Sections 7.1, 7.2, 7.3.

Account availability is a two-state Markov chain per account. Rep planned
absences (vacation, personal, conference) are sampled at simulation start.
Sick days are sampled inside the daily loop (see simulate.py) because they
have zero notice. Churn picks a single departure date per rep and creates
an inheriting replacement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List, Dict
import numpy as np

from config import Config, REP_TYPES, rng, seed_for_account, seed_for_rep
from world import Population, Rep, REP_TYPE_NAMES, REP_TYPE_PROBS


@dataclass
class AbsenceEvent:
    """One absence block.

    duration_days counts working days only. The exact set of calendar days
    (including weekend bridges) lives in absent_day_indices so the daily
    lookup is a single set membership test.
    """
    rep_id: int
    event_type: str
    start_day: int
    duration_days: int
    notice_days: int
    absent_day_indices: List[int] = field(default_factory=list)


@dataclass
class ChurnEvent:
    rep_id: int          # original rep
    announce_day: int
    depart_day: int      # last working day inclusive
    replacement_id: int  # rep_id of inheriting replacement


# ---- Account availability.

def build_availability(cfg: Config, num_accounts: int, horizon: int) -> np.ndarray:
    """Return a [N, horizon] bool array: True means the account is available.

    Transition probabilities are derived from the marginal unavailability
    rate and the persistence (Section 7.1).
    """
    if cfg.p_account_unavail <= 0.0:
        return np.ones((num_accounts, horizon), dtype=bool)

    pers = cfg.account_unavail_persistence
    p_av_to_un = (cfg.p_account_unavail * (1.0 - pers)) / max(1e-12, 1.0 - cfg.p_account_unavail)
    p_av_to_un = float(np.clip(p_av_to_un, 0.0, 1.0))

    out = np.empty((num_accounts, horizon), dtype=bool)
    for a in range(num_accounts):
        r = rng(seed_for_account(cfg.seed, a))
        is_avail = r.random() >= cfg.p_account_unavail   # draw from stationary
        u = r.random(horizon)
        row = out[a]
        for t in range(horizon):
            row[t] = is_avail
            if is_avail:
                is_avail = u[t] >= p_av_to_un
            else:
                is_avail = u[t] >= pers
    return out


def trace_account_blocks(avail: np.ndarray, cfg: Config) -> list:
    """Compress per-day availability into one record per unavailable block."""
    out = []
    N, H = avail.shape
    notice = cfg.account_notice_days
    for a in range(N):
        row = avail[a]
        t = 0
        while t < H:
            if row[t]:
                t += 1
                continue
            start = t
            while t < H and not row[t]:
                t += 1
            out.append({
                "entity_type": "account",
                "entity_id": a,
                "event_type": "unavailable",
                "event_start_day": start,
                "notice_days": notice,
                "duration_days": t - start,
            })
    return out


# ---- Calendar helpers.

def is_weekend(d: date) -> bool:
    return d.weekday() >= 5


def _sample_block(r: np.random.Generator, dist: Dict[int, float]) -> int:
    lengths = list(dist.keys())
    p = np.array([dist[k] for k in lengths], dtype=float)
    return int(r.choice(lengths, p=p / p.sum()))


def _stochastic_round(r: np.random.Generator, x: float) -> int:
    """Round x to floor or ceil so the expectation matches x exactly.

    Short horizons (under one year) need this. Deterministic rounding would
    bias the rate by up to 50%.
    """
    if x <= 0:
        return 0
    floor = int(x)
    return floor + int(r.random() < (x - floor))


def _expand_to_working_days(start_idx: int, length: int, start_date: date,
                            horizon: int) -> List[int]:
    """Walk forward from start_idx and collect `length` working days.

    A 5-working-day block starting Wednesday returns Wed, Thu, Fri, Mon, Tue.
    Stops at horizon.
    """
    days: List[int] = []
    cur = start_idx
    while len(days) < length and cur < horizon:
        if not is_weekend(start_date + timedelta(days=cur)):
            days.append(cur)
        cur += 1
    return days


def _season_vacation(d: date) -> float:
    return 2.0 if d.month in (6, 7, 8, 12) else 1.0


def _season_winter_sick(d: date) -> float:
    return 1.5 if d.month in (11, 12, 1, 2) else 1.0


def _dow_personal(d: date) -> float:
    return 1.5 if d.weekday() in (0, 4) else 1.0


def _try_place(r, *, event_type, rep_id, working_idx, weights, length,
               start_date, horizon, placed, notice_low, notice_high, max_tries=10):
    for _ in range(max_tries):
        probs = weights / weights.sum()
        start_day = int(r.choice(working_idx, p=probs))
        days = _expand_to_working_days(start_day, length, start_date, horizon)
        if not days:
            return None
        if any(set(days) & p for p in placed):
            continue
        placed.append(set(days))
        return AbsenceEvent(
            rep_id=rep_id, event_type=event_type, start_day=days[0],
            duration_days=len(days),
            notice_days=int(r.integers(notice_low, notice_high + 1)),
            absent_day_indices=days,
        )
    return None


def build_planned_absences(cfg: Config, rep_id: int, start_date: date,
                           horizon: int) -> List[AbsenceEvent]:
    """Vacation, personal, conference blocks for one rep.

    Uses the same per-rep seed as the daily loop will, with a fixed offset
    so the two streams stay independent.
    """
    r = rng(seed_for_rep(cfg.seed, rep_id))
    r.bytes(16)
    events: List[AbsenceEvent] = []

    horizon_dates = [start_date + timedelta(days=t) for t in range(horizon)]
    working_idx = np.array([t for t, d in enumerate(horizon_dates) if not is_weekend(d)])
    if working_idx.size == 0:
        return events

    placed: List[set] = []

    # Vacation: season-weighted (Jun-Aug and Dec).
    cd = cfg.vacation_chunk_distribution
    mean_chunk = sum(L * p for L, p in cd.items())
    n_chunks = _stochastic_round(
        r, (cfg.vacation_days_per_year_mean / max(1e-9, mean_chunk)) * (horizon / 365.0))
    season_w = np.array([_season_vacation(horizon_dates[t]) for t in working_idx])
    for _ in range(n_chunks):
        ev = _try_place(
            r, event_type="vacation", rep_id=rep_id,
            working_idx=working_idx, weights=season_w,
            length=_sample_block(r, cd),
            start_date=start_date, horizon=horizon, placed=placed,
            notice_low=cfg.vacation_notice_days_range[0],
            notice_high=cfg.vacation_notice_days_range[1],
        )
        if ev:
            events.append(ev)

    # Personal: Friday/Monday weighted for long weekends.
    pd_ = cfg.personal_block_lengths
    mean_p = sum(L * p for L, p in pd_.items())
    n_p = _stochastic_round(
        r, (cfg.personal_days_per_year_mean / max(1e-9, mean_p)) * (horizon / 365.0))
    dow_w = np.array([_dow_personal(horizon_dates[t]) for t in working_idx])
    for _ in range(n_p):
        ev = _try_place(
            r, event_type="personal", rep_id=rep_id,
            working_idx=working_idx, weights=dow_w,
            length=_sample_block(r, pd_),
            start_date=start_date, horizon=horizon, placed=placed,
            notice_low=cfg.personal_notice_days_range[0],
            notice_high=cfg.personal_notice_days_range[1],
        )
        if ev:
            events.append(ev)

    # Conference: uniform across working days.
    cd2 = cfg.conference_block_lengths
    mean_c = sum(L * p for L, p in cd2.items())
    n_c = _stochastic_round(
        r, (cfg.conference_days_per_year_mean / max(1e-9, mean_c)) * (horizon / 365.0))
    flat = np.ones_like(working_idx, dtype=float)
    for _ in range(n_c):
        ev = _try_place(
            r, event_type="conference", rep_id=rep_id,
            working_idx=working_idx, weights=flat,
            length=_sample_block(r, cd2),
            start_date=start_date, horizon=horizon, placed=placed,
            notice_low=cfg.conference_notice_days_range[0],
            notice_high=cfg.conference_notice_days_range[1],
        )
        if ev:
            events.append(ev)

    return events


def sample_sick_day(r, *, rep_rate, was_sick_yesterday, today,
                    working_days_in_year, sick_autocorr):
    if was_sick_yesterday:
        p = sick_autocorr
    else:
        p = rep_rate * _season_winter_sick(today) / max(1, working_days_in_year)
    return bool(r.random() < float(np.clip(p, 0.0, 1.0)))


def absent_day_set(events: List[AbsenceEvent], horizon: int) -> np.ndarray:
    """Bitset of absent calendar days from a list of AbsenceEvents.

    Planned events carry an explicit day list (weekend bridges included).
    Sick events have only start_day + duration_days = 1.
    """
    absent = np.zeros(horizon, dtype=bool)
    for ev in events:
        if ev.absent_day_indices:
            for d in ev.absent_day_indices:
                if 0 <= d < horizon:
                    absent[d] = True
        else:
            absent[ev.start_day:min(horizon, ev.start_day + ev.duration_days)] = True
    return absent


# ---- Churn (Section 7.3).

def plan_churn(cfg: Config, pop: Population, horizon: int) -> List[ChurnEvent]:
    """Decide which original reps churn and create their replacements.

    Mutates pop.reps by appending replacement reps and setting departure
    dates on the originals.
    """
    out: List[ChurnEvent] = []
    if cfg.p_churn_annual <= 0.0:
        return out

    original_n = len(pop.reps)
    next_id = original_n
    p_horizon = float(np.clip(cfg.p_churn_annual * horizon / 365.0, 0.0, 1.0))

    for rid in range(original_n):
        r = rng(seed_for_rep(cfg.seed, rid))
        r.bytes(8)        # offset from the planned-absence stream
        if r.random() >= p_horizon:
            continue

        depart_day = int(r.integers(1, horizon))
        announce_day = max(0, depart_day - cfg.churn_notice_days)
        original = pop.reps[rid]
        new_type = str(r.choice(REP_TYPE_NAMES, p=REP_TYPE_PROBS))

        replacement = Rep(
            rep_id=next_id, rep_type=new_type, force_id=original.force_id,
            panel=original.panel.copy(),
            hire_date_idx=depart_day + 1, departure_date_idx=-1,
            replacement_of=rid,
        )
        original.departure_date_idx = depart_day
        pop.reps.append(replacement)
        out.append(ChurnEvent(
            rep_id=rid, announce_day=announce_day,
            depart_day=depart_day, replacement_id=next_id,
        ))
        next_id += 1

    return out
