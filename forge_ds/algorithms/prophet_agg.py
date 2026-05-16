"""Prophet aggregate forecaster.

Per rep, we fit four daily series: total, A-tier, B-tier, C-tier. Each
series is the count of completed-or-abbreviated calls per day. Absent
days are encoded as NaN so Prophet treats them as missing.

Prediction runs Prophet to get daily counts per segment, then a
disaggregation step picks specific accounts using a frequency-adherence
score. Refit happens once per 14-day window; daily replanning re-runs
disaggregation against updated constraints without retraining Prophet.
"""
from __future__ import annotations

import io
import logging
import os
import sys
from contextlib import redirect_stdout, redirect_stderr
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .base import (
    Algorithm, ActivityHistory, PlanContext, Plan, PlannedCall,
    DisruptionEvent, assign_start_times, sample_brand_for_account,
)

# Prophet is chatty on import and during fit. Silence it.
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

try:
    from prophet import Prophet
except ImportError:
    Prophet = None


SEG_NAMES = ("A", "B", "C")
DURATION_VALUES = [30, 45, 60, 75]

# Per-segment frequency targets and lift params from SEGMENT_TARGET = {"A": 24, "B": 12, "C": 6}


def _frequency_adherence(n: int, target: int) -> float:
    """Triangular score from : peaks at target, zero at 0 or 2*target."""
    if target <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(n - target) / target)


def _silence_prophet():
    """Prophet emits stan output even with logging silenced. This catches stdout."""
    return redirect_stdout(io.StringIO())


class _RepProphet:
    """Four Prophet models for one rep."""

    def __init__(self, **prophet_kwargs):
        self.kwargs = prophet_kwargs
        self.models: Dict[str, Optional[Prophet]] = {}

    def fit(self, series_by_kind: Dict[str, pd.DataFrame]):
        for kind, df in series_by_kind.items():
            if df.empty or df["y"].notna().sum() < 14:
                self.models[kind] = None
                continue
            m = Prophet(**self.kwargs)
            try:
                with _silence_prophet(), redirect_stderr(io.StringIO()):
                    m.fit(df)
                self.models[kind] = m
            except Exception:
                self.models[kind] = None

    def predict(self, dates: List[date]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        future = pd.DataFrame({"ds": pd.to_datetime(dates)})
        for kind, m in self.models.items():
            if m is None:
                out[kind] = np.zeros(len(dates))
                continue
            with _silence_prophet(), redirect_stderr(io.StringIO()):
                fc = m.predict(future)
            yhat = np.maximum(0.0, fc["yhat"].to_numpy())
            out[kind] = yhat
        return out


class ProphetAlgorithm(Algorithm):
    name = "prophet"

    def __init__(self, config: dict = None):
        super().__init__(config or {})
        if Prophet is None:
            raise RuntimeError("prophet package not installed")
        self.kwargs = dict(
            seasonality_mode=self.config.get("seasonality_mode", "additive"),
            weekly_seasonality=self.config.get("weekly_seasonality", True),
            yearly_seasonality=self.config.get("yearly_seasonality", False),
            daily_seasonality=self.config.get("daily_seasonality", False),
            growth=self.config.get("growth", "linear"),
            changepoint_prior_scale=self.config.get("changepoint_prior_scale", 0.05),
            interval_width=self.config.get("interval_width", 0.8),
        )
        self.rep_models: Dict[int, _RepProphet] = {}
        self.rep_force: Dict[int, int] = {}
        self.rep_type: Dict[int, str] = {}
        self.seed = int(self.config.get("seed", 42))
        self.duration_dist: Dict[Tuple[str, str], List[int]] = {}
        self._panel_segments: Dict[int, Dict[int, str]] = {}

    def _build_series(self, history: ActivityHistory) -> Dict[int, Dict[str, pd.DataFrame]]:
        events = history.events.copy()
        events["date"] = pd.to_datetime(events["date"])
        productive = events[events["outcome"] != "no_show"]

        # All days from earliest event to latest.
        if productive.empty:
            return {}
        d_min = productive["date"].min()
        d_max = productive["date"].max()
        all_days = pd.date_range(d_min, d_max, freq="D")

        # Days where rep was absent should become NaN in the series.
        absences = history.uncertainty[
            (history.uncertainty["entity_type"] == "rep")
        ].copy()
        if not absences.empty:
            absences["event_start_date"] = pd.to_datetime(absences["event_start_date"])

        out: Dict[int, Dict[str, pd.DataFrame]] = {}
        for rid, rep_df in productive.groupby("rep_id"):
            rid = int(rid)
            counts_total = rep_df.groupby("date").size()
            counts_seg = {
                s: rep_df[rep_df["segment_at_call"] == s].groupby("date").size()
                for s in SEG_NAMES
            }

            rep_absent_days = set()
            if not absences.empty:
                rep_abs = absences[absences["entity_id"].astype(int) == rid]
                for _, row in rep_abs.iterrows():
                    s = row["event_start_date"].date()
                    for k in range(int(row["duration_days"])):
                        rep_absent_days.add(s + timedelta(days=k))

            series_by_kind: Dict[str, pd.DataFrame] = {}
            for kind, src in {"total": counts_total, **{s: counts_seg[s] for s in SEG_NAMES}}.items():
                y = []
                ds = []
                for d in all_days:
                    ds.append(d)
                    if d.date() in rep_absent_days:
                        y.append(np.nan)
                    else:
                        y.append(int(src.get(d, 0)))
                series_by_kind[kind] = pd.DataFrame({"ds": ds, "y": y})
            out[rid] = series_by_kind
        return out

    def _build_duration_dist(self, history: ActivityHistory) -> None:
        events = history.events[history.events["outcome"] != "no_show"]
        for (seg, rt), grp in events.merge(
            history.population[["rep_id", "type"]], on="rep_id"
        ).groupby(["segment_at_call", "type"]):
            self.duration_dist[(str(seg), str(rt))] = grp[
                "planned_duration_min"
            ].astype(int).tolist()

    def fit(self, history: ActivityHistory) -> None:
        pop = history.population.set_index("rep_id")
        self.rep_force = {int(r): int(pop.loc[r, "force_id"]) for r in pop.index}
        self.rep_type = {int(r): str(pop.loc[r, "type"]) for r in pop.index}

        series_by_rep = self._build_series(history)
        for rid, series in series_by_rep.items():
            model = _RepProphet(**self.kwargs)
            model.fit(series)
            self.rep_models[rid] = model

        self._build_duration_dist(history)

    def _sample_duration(self, rng: np.random.Generator, seg: str, rep_type: str) -> int:
        pool = self.duration_dist.get((seg, rep_type)) or self.duration_dist.get(("B", rep_type)) or DURATION_VALUES
        return int(rng.choice(pool))

    def predict_window(self, context: PlanContext,
                       window_start: date, window_days: int = 14) -> Plan:
        rng = np.random.default_rng((self.seed, context.rep_id,
                                     window_start.toordinal()))
        plan = Plan(rep_id=context.rep_id, window_start=window_start,
                    window_end=window_start + timedelta(days=window_days))

        dates_in_window = [window_start + timedelta(days=i)
                           for i in range(window_days)]
        model = self.rep_models.get(context.rep_id)
        if model is None:
            return plan
        pred = model.predict(dates_in_window)

        # Pre-compute segment partition of the panel for ranking.
        panel_by_seg: Dict[str, List[int]] = {s: [] for s in SEG_NAMES}
        for a in context.panel:
            panel_by_seg.setdefault(context.segments.get(a, "B"), []).append(a)

        # Track accumulated calls per (account, brand) so frequency-adherence
        # can guide ranking within the window.
        n_ab: Dict[Tuple[int, int], int] = {}

        for i, d in enumerate(dates_in_window):
            if d.weekday() >= 5:
                continue
            if context.is_rep_absent_on(d):
                continue
            n_total = int(round(pred.get("total", np.zeros(window_days))[i]))
            n_per_seg = {
                s: int(round(pred.get(s, np.zeros(window_days))[i]))
                for s in SEG_NAMES
            }
            # Reconcile per-segment counts with total (cap by total).
            total_seg = sum(n_per_seg.values())
            if total_seg > n_total and total_seg > 0:
                scale = n_total / total_seg
                n_per_seg = {s: int(round(c * scale)) for s, c in n_per_seg.items()}
            if max(n_per_seg.values(), default=0) == 0:
                continue

            available = set(context.available_accounts_on(d))
            day_calls: List[PlannedCall] = []
            seen: set = set()

            for s in SEG_NAMES:
                need = n_per_seg.get(s, 0)
                if need <= 0:
                    continue
                candidates = [a for a in panel_by_seg.get(s, [])
                              if a in available and a not in seen]
                if not candidates:
                    continue

                # Frequency-adherence score per candidate.
                target = SEGMENT_TARGET[s]
                def score(a):
                    n = sum(n_ab.get((a, b), 0) for b in context.bag)
                    return _frequency_adherence(n + 1, target)
                candidates.sort(key=score, reverse=True)
                picks = candidates[:need]
                for a in picks:
                    seen.add(a)
                    dur = self._sample_duration(rng, s, context.rep_type)
                    brand_id, brand_p = sample_brand_for_account(
                        rng, a, context.bag, context.priorities, context.eligibility)
                    day_calls.append(PlannedCall(
                        date=d, rep_id=context.rep_id, start_minute=0,
                        planned_duration=dur, account_id=a, segment_at_call=s,
                        brand_id=brand_id, brand_priority=brand_p,
                    ))
                    n_ab[(a, brand_id)] = n_ab.get((a, brand_id), 0) + 1

            day_calls = assign_start_times(day_calls)
            plan.calls.extend(day_calls)

        return plan

    def replan_within_window(self, current_plan: Plan,
                             disruptions: List[DisruptionEvent],
                             revealed_at: date) -> Plan:
        new_plan = Plan(rep_id=current_plan.rep_id,
                        window_start=current_plan.window_start,
                        window_end=current_plan.window_end)
        new_plan.calls = [c for c in current_plan.calls if c.date < revealed_at]
        return new_plan
