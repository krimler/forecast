"""Common interface and data structures shared by every algorithm.

Spec2 Section 3.

An Algorithm has three lifecycle methods. `fit` runs once on the warmup
window. `predict_window` produces a 14-day plan starting at a given date.
`replan_within_window` updates a plan when new disruptions arrive,
locking everything before `revealed_at`.

A PlanContext carries everything an algorithm needs to plan one rep's
window. Visibility rules are enforced by the harness before calling
predict/replan, so algorithms can trust the masked fields.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


# ---- Data structures.

@dataclass
class ActivityHistory:
    """Realized activity over a window of time.

    All dataframes follow the spec1 output schema. The harness builds
    this by reading the spec1 dataset directory and masking to the
    visibility window.
    """
    events: pd.DataFrame          # activity_log.csv subset (scenario_id="actual")
    uncertainty: pd.DataFrame     # uncertainty_traces.csv
    population: pd.DataFrame
    panels: pd.DataFrame
    accounts: pd.DataFrame
    config: dict                  # parsed config.json


@dataclass
class DisruptionEvent:
    """A single revealed uncertainty event."""
    revealed_at: date             # day on which the rep learns about this
    entity_type: str              # "account" or "rep"
    entity_id: int
    event_type: str               # "unavailable", "sick", "personal", ...
    start_day: date
    duration_days: int


@dataclass
class PlannedCall:
    """One scheduled call inside a plan (a row in activity_log)."""
    date: date
    rep_id: int
    start_minute: int
    planned_duration: int
    account_id: int
    segment_at_call: str
    brand_id: int
    brand_priority: float
    outcome: str = "completed"    # algorithms produce completed; outcome
                                  # is overwritten during disruption replay

    def end_minute(self) -> int:
        return self.start_minute + self.planned_duration


@dataclass
class Plan:
    """A plan over a window. Calls are sorted by (date, start_minute)."""
    rep_id: int
    window_start: date
    window_end: date              # exclusive
    calls: List[PlannedCall] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for c in self.calls:
            rows.append({
                "date": c.date.isoformat(),
                "rep_id": c.rep_id,
                "start_time": f"{c.start_minute // 60:02d}:{c.start_minute % 60:02d}",
                "planned_duration_min": c.planned_duration,
                "actual_duration_min": (c.planned_duration if c.outcome == "completed"
                                        else (c.planned_duration // 2 if c.outcome == "abbreviated"
                                              else 0)),
                "account_id": c.account_id,
                "segment_at_call": c.segment_at_call,
                "brand_id": c.brand_id,
                "brand_priority": c.brand_priority,
                "outcome": c.outcome,
            })
        return pd.DataFrame(rows)

    def calls_on(self, d: date) -> List[PlannedCall]:
        return [c for c in self.calls if c.date == d]


@dataclass
class PlanContext:
    """Everything an algorithm needs to plan one rep's 14-day window."""
    rep_id: int
    rep_type: str
    force_id: int
    panel: List[int]
    bag: List[int]
    priorities: List[float]
    eligibility: Dict[int, List[int]]   # account_id -> list of brand_ids
    segments: Dict[int, str]            # account_id -> "A"/"B"/"C"
    known_absences: pd.DataFrame        # uncertainty rows visible to plan
    known_unavailable: pd.DataFrame
    history: ActivityHistory            # actual realized events to date
    horizon_start: date
    horizon_end: date                   # exclusive

    def is_rep_absent_on(self, d: date) -> bool:
        if self.known_absences is None or self.known_absences.empty:
            return False
        for _, ev in self.known_absences.iterrows():
            if str(ev["entity_type"]) != "rep" or int(ev["entity_id"]) != self.rep_id:
                continue
            start = date.fromisoformat(str(ev["event_start_date"]))
            end = start + timedelta(days=int(ev["duration_days"]))
            if start <= d < end:
                return True
        return False

    def is_account_unavailable_on(self, account_id: int, d: date) -> bool:
        if self.known_unavailable is None or self.known_unavailable.empty:
            return False
        rows = self.known_unavailable[
            (self.known_unavailable["entity_type"] == "account")
            & (self.known_unavailable["entity_id"].astype(int) == int(account_id))
        ]
        for _, ev in rows.iterrows():
            start = date.fromisoformat(str(ev["event_start_date"]))
            end = start + timedelta(days=int(ev["duration_days"]))
            if start <= d < end:
                return True
        return False

    def available_accounts_on(self, d: date) -> List[int]:
        return [a for a in self.panel
                if not self.is_account_unavailable_on(a, d)]


# ---- Algorithm base class.

class Algorithm(ABC):
    """Abstract base. Every algorithm has a name and three methods."""
    name: str = "base"

    def __init__(self, config: dict):
        self.config = dict(config) if config else {}

    @abstractmethod
    def fit(self, history: ActivityHistory) -> None:
        """Train on the warmup window. Called once per (dataset, seed)."""

    @abstractmethod
    def predict_window(self, context: PlanContext,
                       window_start: date, window_days: int = 14) -> Plan:
        """Generate a plan for one rep over a window."""

    def replan_within_window(self, current_plan: Plan,
                             disruptions: List[DisruptionEvent],
                             revealed_at: date) -> Plan:
        """Default: re-run prediction from revealed_at forward.

        Subclasses with cheaper replan strategies override this.
        """
        return current_plan


# ---- Shared helpers.

def assign_start_times(day_calls: List[PlannedCall],
                       day_start_minute: int = 9 * 60,
                       day_end_minute: int = 18 * 60,
                       inter_call_gap: int = 15) -> List[PlannedCall]:
    """Pack calls into a day's clock, dropping ones that would run past 18:00.

    The clock starts at 9:00 and advances by (duration + 15-min gap) for
    every accepted call. Spec1 Section 8.5.
    """
    accepted: List[PlannedCall] = []
    cur = day_start_minute
    for c in day_calls:
        end = cur + c.planned_duration
        if end > day_end_minute:
            break
        c.start_minute = cur
        accepted.append(c)
        cur = end + inter_call_gap
    return accepted


def sample_brand_for_account(rng: np.random.Generator, account_id: int,
                             bag: List[int], priorities: List[float],
                             eligibility: Dict[int, List[int]]) -> Tuple[int, float]:
    """Sample a brand for the given account weighted by priority on the
    intersection of bag and eligibility[account_id]."""
    eligible = eligibility.get(account_id, [])
    options = [(b, p) for b, p in zip(bag, priorities) if b in eligible]
    if not options:
        return bag[0], priorities[0]
    bs, ps = zip(*options)
    arr = np.array(ps, dtype=float)
    arr = arr / arr.sum()
    idx = int(rng.choice(len(bs), p=arr))
    return int(bs[idx]), float(ps[idx])
