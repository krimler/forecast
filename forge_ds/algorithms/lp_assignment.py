"""LP-based assignment-problem baseline.

Per the algorithm spec Stage 2: maximize Sales over a 14-day window by selecting
(account, day, brand) assignments under a daily face-time capacity. No
within-window daily replans; we re-solve only at each window boundary.

Formulation per rep, per window:

    x_{a,d,b} ∈ [0, 1]    one continuous "call" indicator
    u_{a,b}   >= 0        piecewise-linear lift over total n_{a,b}

    maximize  sum_{a,b} v_seg(a) * pi_b * u_{a,b}
    subject to
        sum_b x_{a,d,b} <= feasible(a, d, b)         (eligibility, availability,
                                                      rep absence)
        sum_b x_{a,d,b} <= 1                         (one brand per a,d)
        sum_{a,b} dur(a,b,seg) * x_{a,d,b} <= cap    (daily face-time)
        u_{a,b} <= slope_k * (n_prior + sum_d x) + c_k    (concave envelope)

The lift is concave so the LP relaxation is tight on the u variables.
The x variables can come out fractional; we round greedily by descending
x*duration per (rep, day) keeping the daily cap.

If CBC can't solve in 30 seconds for a (rep, window) the algorithm
falls back to a value-greedy schedule and the caller increments a
fallback counter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import pulp


# segment table.
SEG_VALUE = {"A": 5.0, "B": 2.0, "C": 1.0}
SEG_TARGET = {"A": 24, "B": 12, "C": 6}
SEG_LIFT_A = {"A": 1.0, "B": 0.8, "C": 0.5}
SEG_LIFT_B = {"A": 0.12, "B": 0.18, "C": 0.30}

# Mean call duration per (rep_type, segment) from weighted by
# the probability table. Precomputed.
DURATION_PROBS = {
    ("specialty",   "A"): [0.05, 0.15, 0.40, 0.40],
    ("specialty",   "B"): [0.20, 0.40, 0.30, 0.10],
    ("specialty",   "C"): [0.50, 0.35, 0.15, 0.00],
    ("mid-market",  "A"): [0.10, 0.20, 0.40, 0.30],
    ("mid-market",  "B"): [0.30, 0.40, 0.20, 0.10],
    ("mid-market",  "C"): [0.60, 0.30, 0.10, 0.00],
    ("high-volume", "A"): [0.20, 0.30, 0.40, 0.10],
    ("high-volume", "B"): [0.50, 0.35, 0.15, 0.00],
    ("high-volume", "C"): [0.80, 0.20, 0.00, 0.00],
}
DURATION_VALUES = [30, 45, 60, 75]


def expected_duration(rep_type: str, segment: str) -> float:
    probs = DURATION_PROBS.get((rep_type, segment))
    if probs is None:
        return 45.0
    return float(sum(d * p for d, p in zip(DURATION_VALUES, probs)))


def lift_at(seg: str, n: float) -> float:
    return SEG_LIFT_A[seg] * (1.0 - math.exp(-SEG_LIFT_B[seg] * n))


def piecewise_lift_breakpoints(seg: str) -> List[Tuple[float, float, float]]:
    """Return (left_n, slope, intercept) triples for the concave envelope.

    Each segment k contributes a constraint:  u <= slope_k * n + intercept_k.
    With 4 breakpoints we get 3 segments, which under-approximates the
    concave lift everywhere and is exact at the breakpoints.
    """
    target = SEG_TARGET[seg]
    bps = [0.0, target / 3.0, 2.0 * target / 3.0, float(target)]
    segments = []
    for i in range(len(bps) - 1):
        x0, x1 = bps[i], bps[i + 1]
        y0, y1 = lift_at(seg, x0), lift_at(seg, x1)
        slope = (y1 - y0) / max(1e-9, (x1 - x0))
        intercept = y0 - slope * x0
        segments.append((x0, slope, intercept))
    return segments


# ----------------------------------------------------------------------
# Per-rep snapshot built by the stage2 runner; the LP doesn't touch
# pandas to keep solve-time predictable.
# ----------------------------------------------------------------------

@dataclass
class RepSnapshot:
    rep_id: int
    rep_type: str
    force_id: int
    panel: List[int]                      # account ids
    bag: List[int]                        # brand ids in this rep's bag
    priorities: Dict[int, float]          # brand_id -> priority
    eligibility: Dict[int, List[int]]     # account_id -> list of brand ids in bag eligible
    segments: Dict[int, str]              # account_id -> "A"/"B"/"C"
    capacity_min_per_day: int = 540


@dataclass
class WindowMasks:
    """Visibility-aware per-(rep, window) masks at window start."""
    rep_absent_day: List[bool]            # length = n_days
    account_unavail_day: Dict[int, List[bool]]  # account_id -> per-day mask
    # Prior counts of (account, brand) calls in this annual cycle for the rep.
    n_a_b_prior: Dict[Tuple[int, int], int] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Greedy fallback. Plain marginal-lift sort by (rep, day) until capacity.
# ----------------------------------------------------------------------

def _greedy_fallback(snap: RepSnapshot, masks: WindowMasks,
                     n_days: int) -> List[Dict]:
    out: List[Dict] = []
    n_a_b = dict(masks.n_a_b_prior)
    for d in range(n_days):
        if masks.rep_absent_day[d]:
            continue
        # Score every feasible (account, brand) by marginal lift.
        cand: List[Tuple[float, int, int, float]] = []
        for a in snap.panel:
            if masks.account_unavail_day.get(a, [False] * n_days)[d]:
                continue
            seg = snap.segments.get(a, "B")
            for b in snap.eligibility.get(a, []):
                if b not in snap.bag:
                    continue
                n = n_a_b.get((a, b), 0)
                marg = (SEG_LIFT_A[seg] * math.exp(-SEG_LIFT_B[seg] * n)
                        * (1.0 - math.exp(-SEG_LIFT_B[seg])))
                v = SEG_VALUE[seg] * snap.priorities.get(b, 0.0) * marg
                dur = expected_duration(snap.rep_type, seg)
                cand.append((v / dur, a, b, dur))
        cand.sort(reverse=True)
        used = 0.0
        called_today: set = set()
        for value_per_min, a, b, dur in cand:
            if a in called_today:
                continue
            if used + dur > snap.capacity_min_per_day:
                continue
            out.append({"day": d, "account_id": a, "brand_id": b,
                        "duration": int(round(dur))})
            called_today.add(a)
            used += dur
            n_a_b[(a, b)] = n_a_b.get((a, b), 0) + 1
    return out


# ----------------------------------------------------------------------
# LP solver. Returns (calls, fallback_used).
# ----------------------------------------------------------------------

def solve_window(snap: RepSnapshot, masks: WindowMasks, n_days: int,
                 time_limit_sec: int = 30) -> Tuple[List[Dict], bool]:
    prob = pulp.LpProblem(f"rep_{snap.rep_id}", pulp.LpMaximize)

    # ---- Variables.
    x: Dict[Tuple[int, int, int], pulp.LpVariable] = {}
    for a in snap.panel:
        avail = masks.account_unavail_day.get(a, [False] * n_days)
        for d in range(n_days):
            if masks.rep_absent_day[d] or avail[d]:
                continue
            for b in snap.eligibility.get(a, []):
                if b not in snap.bag:
                    continue
                x[(a, d, b)] = pulp.LpVariable(
                    f"x_{a}_{d}_{b}", lowBound=0, upBound=1)
    if not x:
        return [], False

    # u_{a,b}: piecewise-linear lift over total n_{a,b} = prior + window calls.
    u: Dict[Tuple[int, int], pulp.LpVariable] = {}
    ab_pairs: set = set()
    for (a, d, b) in x.keys():
        ab_pairs.add((a, b))
    for (a, b) in ab_pairs:
        u[(a, b)] = pulp.LpVariable(f"u_{a}_{b}", lowBound=0)

    # ---- Objective.
    obj_terms = []
    for (a, b), uv in u.items():
        seg = snap.segments.get(a, "B")
        coef = SEG_VALUE[seg] * snap.priorities.get(b, 0.0)
        obj_terms.append(coef * uv)
    prob += pulp.lpSum(obj_terms)

    # ---- Constraints.

    # 1) one brand per (a, d).
    one_brand: Dict[Tuple[int, int], List[pulp.LpVariable]] = {}
    for (a, d, b), v in x.items():
        one_brand.setdefault((a, d), []).append(v)
    for key, lst in one_brand.items():
        prob += pulp.lpSum(lst) <= 1, f"one_{key[0]}_{key[1]}"

    # 2) daily capacity.
    by_day: Dict[int, List[Tuple[float, pulp.LpVariable]]] = {}
    for (a, d, b), v in x.items():
        seg = snap.segments.get(a, "B")
        dur = expected_duration(snap.rep_type, seg)
        by_day.setdefault(d, []).append((dur, v))
    for d, terms in by_day.items():
        prob += pulp.lpSum(coef * var for coef, var in terms) \
            <= snap.capacity_min_per_day, f"cap_{d}"

    # 3) piecewise-linear lift envelope.
    for (a, b), uv in u.items():
        seg = snap.segments.get(a, "B")
        n_prior = masks.n_a_b_prior.get((a, b), 0)
        # n_total = n_prior + sum of x for this (a, b) over d.
        window_x = [x[(a, d, b)] for d in range(n_days) if (a, d, b) in x]
        n_sum = pulp.lpSum(window_x) if window_x else 0
        for x0, slope, intercept in piecewise_lift_breakpoints(seg):
            prob += uv <= slope * (n_prior + n_sum) + intercept, \
                f"lift_{a}_{b}_{int(x0)}"

    # ---- Solve.
    solver = pulp.PULP_CBC_CMD(timeLimit=time_limit_sec, msg=0)
    try:
        prob.solve(solver)
    except Exception:
        return _greedy_fallback(snap, masks, n_days), True

    status = pulp.LpStatus.get(prob.status, "")
    if prob.status not in (1,) or prob.objective is None:
        # Optimal=1; any other status (infeasible, unbounded, timeout w/o
        # feasible) falls back.
        return _greedy_fallback(snap, masks, n_days), True

    # ---- Greedy rounding of fractional x into integer call schedule.
    rounded: List[Dict] = []
    for d in range(n_days):
        if masks.rep_absent_day[d]:
            continue
        # Sort (a, b) descending by x_value × per-call value density.
        cand: List[Tuple[float, int, int, float]] = []
        for (a, dd, b), var in x.items():
            if dd != d:
                continue
            xv = var.value() or 0.0
            if xv <= 1e-6:
                continue
            seg = snap.segments.get(a, "B")
            n_prior = masks.n_a_b_prior.get((a, b), 0)
            marg = (SEG_LIFT_A[seg] * math.exp(-SEG_LIFT_B[seg] * n_prior)
                    * (1.0 - math.exp(-SEG_LIFT_B[seg])))
            v = SEG_VALUE[seg] * snap.priorities.get(b, 0.0) * marg
            dur = expected_duration(snap.rep_type, seg)
            cand.append((xv * v, a, b, dur))
        cand.sort(reverse=True)
        used = 0.0
        called_today: set = set()
        for _, a, b, dur in cand:
            if a in called_today:
                continue
            if used + dur > snap.capacity_min_per_day:
                continue
            rounded.append({"day": d, "account_id": a, "brand_id": b,
                            "duration": int(round(dur))})
            called_today.add(a)
            used += dur

    return rounded, False
