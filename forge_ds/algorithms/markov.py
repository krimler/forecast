"""Markov chain baseline.

A first-order Markov chain over rep activity. For each rep we fit five
conditional distributions: calls per day given dow, first account given
dow, next account given current account and dow, brand given account
and priorities, and duration given segment and rep type.

If a rep has fewer than 30 days of warmup history we fall back to a
force-pooled model fitted across all reps in the same force.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .base import (
    Algorithm, ActivityHistory, PlanContext, Plan, PlannedCall,
    DisruptionEvent, assign_start_times, sample_brand_for_account,
)


SEG_NAMES = ("A", "B", "C")
DURATION_VALUES = [30, 45, 60, 75]


def _laplace_dist(counts: Dict, alpha: float) -> Dict:
    """Return probabilities with Laplace smoothing applied. Returns a
    dict of key -> probability, plus a special key 'support' giving the
    smoothed denominator so a caller can sample over an extended domain.
    """
    if not counts:
        return {}
    total = sum(counts.values()) + alpha * len(counts)
    return {k: (v + alpha) / total for k, v in counts.items()}


class RepMarkov:
    """Fitted Markov model for a single rep or for a force pool."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.p_calls_given_dow: Dict[int, Counter] = defaultdict(Counter)
        self.p_first_given_dow: Dict[int, Counter] = defaultdict(Counter)
        self.p_next_given_prev_dow: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
        self.p_duration_given_seg_type: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        # Mark whether we've seen any data.
        self.empty = True

    def update(self, day_events: pd.DataFrame, dow: int, rep_type: str) -> None:
        """Add the events of one (rep, day) to the running counts."""
        if day_events.empty:
            self.p_calls_given_dow[dow][0] += 1
            return
        self.empty = False
        n = len(day_events)
        self.p_calls_given_dow[dow][n] += 1
        sorted_events = day_events.sort_values("start_time")
        accounts = sorted_events["account_id"].astype(int).tolist()
        segments = sorted_events["segment_at_call"].astype(str).tolist()
        durations = sorted_events["planned_duration_min"].astype(int).tolist()
        self.p_first_given_dow[dow][accounts[0]] += 1
        for i in range(1, len(accounts)):
            self.p_next_given_prev_dow[(accounts[i - 1], dow)][accounts[i]] += 1
        for s, d in zip(segments, durations):
            self.p_duration_given_seg_type[(s, rep_type)][d] += 1

    def n_calls(self, rng: np.random.Generator, dow: int) -> int:
        counts = self.p_calls_given_dow.get(dow, Counter())
        if not counts:
            return 0
        probs = _laplace_dist(counts, self.alpha)
        keys = list(probs.keys())
        ps = np.array([probs[k] for k in keys], dtype=float)
        ps = ps / ps.sum()
        return int(rng.choice(keys, p=ps))

    def first_account(self, rng: np.random.Generator, dow: int,
                      available: List[int]) -> Optional[int]:
        counts = self.p_first_given_dow.get(dow, Counter())
        candidates = {a: counts.get(a, 0) for a in available}
        if not candidates:
            return None
        probs = _laplace_dist(candidates, self.alpha)
        keys = list(probs.keys())
        ps = np.array([probs[k] for k in keys], dtype=float)
        ps = ps / ps.sum()
        return int(rng.choice(keys, p=ps))

    def next_account(self, rng: np.random.Generator, prev: int, dow: int,
                     available: List[int]) -> Optional[int]:
        counts = self.p_next_given_prev_dow.get((prev, dow), Counter())
        candidates = {a: counts.get(a, 0) for a in available}
        if not candidates:
            return None
        # If we've never seen `prev` before, fall back to first-account dist.
        if sum(candidates.values()) == 0:
            return self.first_account(rng, dow, available)
        probs = _laplace_dist(candidates, self.alpha)
        keys = list(probs.keys())
        ps = np.array([probs[k] for k in keys], dtype=float)
        ps = ps / ps.sum()
        return int(rng.choice(keys, p=ps))

    def duration(self, rng: np.random.Generator, segment: str, rep_type: str) -> int:
        counts = self.p_duration_given_seg_type.get((segment, rep_type), Counter())
        if not counts:
            return 45      # the dataset spec default mid-point
        probs = _laplace_dist(counts, self.alpha)
        keys = list(probs.keys())
        ps = np.array([probs[k] for k in keys], dtype=float)
        ps = ps / ps.sum()
        return int(rng.choice(keys, p=ps))


class MarkovAlgorithm(Algorithm):
    name = "markov"

    def __init__(self, config: dict = None):
        super().__init__(config or {})
        self.alpha = float(self.config.get("smoothing_alpha", 0.1))
        self.min_history_days = int(self.config.get("min_history_days", 30))
        self.per_rep: Dict[int, RepMarkov] = {}
        self.per_force: Dict[int, RepMarkov] = {}
        self.rep_force: Dict[int, int] = {}
        self.rep_type: Dict[int, str] = {}
        self.rep_history_days: Dict[int, int] = {}
        self.seed = int(self.config.get("seed", 42))

    def fit(self, history: ActivityHistory) -> None:
        events = history.events.copy()
        if events.empty:
            return
        events["date"] = pd.to_datetime(events["date"]).dt.date
        events["dow"] = pd.to_datetime(events["date"]).dt.weekday
        pop = history.population.set_index("rep_id")
        self.rep_force = {int(r): int(pop.loc[r, "force_id"]) for r in pop.index}
        self.rep_type = {int(r): str(pop.loc[r, "type"]) for r in pop.index}

        # Force-pooled first.
        for fid in set(self.rep_force.values()):
            self.per_force[fid] = RepMarkov(alpha=self.alpha)

        # Group by (rep, day) so we can update Markov stats one day at a time.
        for (rid, d), day_df in events.groupby(["rep_id", "date"]):
            rid = int(rid)
            rt = self.rep_type.get(rid, "mid-market")
            dow = int(pd.Timestamp(d).weekday())
            self.per_rep.setdefault(rid, RepMarkov(alpha=self.alpha))
            self.per_rep[rid].update(day_df, dow, rt)
            fid = self.rep_force.get(rid)
            if fid is not None:
                self.per_force[fid].update(day_df, dow, rt)
            self.rep_history_days[rid] = self.rep_history_days.get(rid, 0) + 1

    def _model_for(self, rep_id: int) -> RepMarkov:
        if self.rep_history_days.get(rep_id, 0) >= self.min_history_days:
            m = self.per_rep.get(rep_id)
            if m is not None and not m.empty:
                return m
        fid = self.rep_force.get(rep_id, 0)
        return self.per_force.get(fid) or RepMarkov(alpha=self.alpha)

    def predict_window(self, context: PlanContext,
                       window_start: date, window_days: int = 14) -> Plan:
        rng = np.random.default_rng((self.seed, context.rep_id,
                                     window_start.toordinal()))
        plan = Plan(rep_id=context.rep_id, window_start=window_start,
                    window_end=window_start + timedelta(days=window_days))
        model = self._model_for(context.rep_id)

        for offset in range(window_days):
            d = window_start + timedelta(days=offset)
            if d.weekday() >= 5:
                continue
            if context.is_rep_absent_on(d):
                continue
            available = context.available_accounts_on(d)
            if not available:
                continue
            dow = d.weekday()
            n = model.n_calls(rng, dow)
            n = max(0, min(n, len(available)))
            if n == 0:
                continue

            day_calls: List[PlannedCall] = []
            seen: set = set()
            first = model.first_account(rng, dow, available)
            if first is None:
                continue
            seen.add(first)
            prev = first
            picked = [first]
            for _ in range(n - 1):
                left = [a for a in available if a not in seen]
                if not left:
                    break
                nxt = model.next_account(rng, prev, dow, left)
                if nxt is None:
                    break
                picked.append(nxt)
                seen.add(nxt)
                prev = nxt

            for a in picked:
                seg = context.segments.get(a, "B")
                dur = model.duration(rng, seg, context.rep_type)
                brand_id, brand_p = sample_brand_for_account(
                    rng, a, context.bag, context.priorities, context.eligibility)
                day_calls.append(PlannedCall(
                    date=d, rep_id=context.rep_id, start_minute=0,
                    planned_duration=dur, account_id=a, segment_at_call=seg,
                    brand_id=brand_id, brand_priority=brand_p,
                ))
            day_calls = assign_start_times(day_calls)
            plan.calls.extend(day_calls)

        return plan

    def replan_within_window(self, current_plan: Plan,
                             disruptions: List[DisruptionEvent],
                             revealed_at: date) -> Plan:
        """Markov replan is stateless. Keep locked past, drop future, let
        the harness call predict_window again from revealed_at."""
        new_plan = Plan(rep_id=current_plan.rep_id,
                        window_start=current_plan.window_start,
                        window_end=current_plan.window_end)
        new_plan.calls = [c for c in current_plan.calls if c.date < revealed_at]
        return new_plan
