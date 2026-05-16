"""Evaluation metrics and the fixed reference replanner (spec1 Section 11).

Five metrics from the spec:
    Sales              Eq. 1, with SalesNorm (Eq. 2) for cross-scenario comparison
    Coverage           Eq. 3, priority-weighted brand reach
    Robustness         Eq. 4, expected Sales(Replan(P,D)) / Sales(P)
    DisruptionRate     fraction of calls with outcome != completed (Section 11.4)
    ReplanCost         median replan wall-clock time (Section 11.5)

Plus the source decomposition for Figure 2 (Section 11.6).
The reference replanner is plan-agnostic by spec design.
"""
from __future__ import annotations

import time
from copy import deepcopy
from statistics import median
from typing import List, Tuple, Dict
import numpy as np

from config import Config, SEGMENT_DEFAULTS, SEG_NAMES, rng, seed_for_replanner
from world import Population, Rep, ForceConfig
from uncertainty import AbsenceEvent
from simulate import (
    CallEvent, SEG_VALUES, SEG_TARGETS, SEG_LIFT_A, SEG_LIFT_B, _sample_duration,
)


SEG_VALUE = np.array([SEGMENT_DEFAULTS[s]["value"] for s in SEG_NAMES])
SEG_LA = np.array([SEGMENT_DEFAULTS[s]["lift_a"] for s in SEG_NAMES])
SEG_LB = np.array([SEGMENT_DEFAULTS[s]["lift_b"] for s in SEG_NAMES])


# ---- Sales (Eq. 1, Eq. 2).

def sales(calls: List[CallEvent], pop: Population, *, eval_window: range,
          account_segment: np.ndarray = None) -> float:
    """Sum over (account, brand) pairs of v_seg * pi_b * lift(n, seg).

    Only completed and abbreviated calls count toward n. Pass
    account_segment (a fixed per-account segment array) for stable
    cross-scenario comparison. Otherwise the segment seen at the first call
    in this plan is used as a fallback.
    """
    brand_priority: Dict[int, float] = {}
    for f in pop.forces:
        for b, p in zip(f.brands, f.priorities):
            brand_priority[b] = p

    counts: Dict[Tuple[int, int], int] = {}
    seg_seen: Dict[int, int] = {}
    for c in calls:
        if c.date_idx not in eval_window or c.outcome == "no_show":
            continue
        key = (c.account_id, c.brand_id)
        counts[key] = counts.get(key, 0) + 1
        if c.account_id not in seg_seen:
            seg_seen[c.account_id] = c.segment_at_call

    total = 0.0
    for (a, b), n in counts.items():
        seg_idx = int(account_segment[a]) if account_segment is not None else int(seg_seen[a])
        v = SEG_VALUE[seg_idx]
        pi = brand_priority.get(b, 0.0)
        lift_val = SEG_LA[seg_idx] * (1.0 - np.exp(-SEG_LB[seg_idx] * n))
        total += float(v * pi * lift_val)
    return total


def sales_norm(sales_p: float, sales_star: float, sales_naive: float) -> Tuple[float, bool]:
    """SalesNorm with the epsilon safeguard from B5.

    Returns (norm, flagged) where flagged is True if the upper-lower gap is
    smaller than 1% of the lower bound. In that case raw Sales is more
    informative than the normalized value.
    """
    eps = 0.01 * max(1e-9, sales_naive)
    denom = sales_star - sales_naive
    flagged = denom < eps
    return (sales_p - sales_naive) / max(eps, denom), flagged


# ---- Coverage (Eq. 3).

def coverage(calls: List[CallEvent], pop: Population, eligibility: np.ndarray,
             *, eval_window: range) -> float:
    """Priority-weighted average of per-brand reach.

    Reach for a brand is the fraction of its eligible accounts that got at
    least one non-no_show call. We normalize by the total priority so the
    result stays in [0, 1].
    """
    num_brands = eligibility.shape[1]
    called: Dict[int, set] = {b: set() for b in range(num_brands)}
    for c in calls:
        if c.date_idx not in eval_window or c.outcome == "no_show":
            continue
        called[c.brand_id].add(c.account_id)

    brand_priority: Dict[int, float] = {}
    for f in pop.forces:
        for b, p in zip(f.brands, f.priorities):
            brand_priority[b] = p

    total = 0.0
    total_pi = 0.0
    for b in range(num_brands):
        elig = int(eligibility[:, b].sum())
        if elig == 0:
            continue
        cov = len(called[b]) / elig
        pi = brand_priority.get(b, 0.0)
        total += pi * cov
        total_pi += pi
    return 0.0 if total_pi <= 0 else total / total_pi


# ---- Disruption rate and source decomposition (Sections 11.4, 11.6).

def disruption_rate(calls: List[CallEvent], *, eval_window: range) -> float:
    in_win = [c for c in calls if c.date_idx in eval_window]
    if not in_win:
        return 0.0
    bad = sum(1 for c in in_win if c.outcome != "completed")
    return bad / len(in_win)


def decompose_disruptions(calls: List[CallEvent], availability: np.ndarray,
                          planned: List[AbsenceEvent], sick: List[AbsenceEvent],
                          *, eval_window: range) -> Dict[str, int]:
    """Count disrupted calls by source.

    The 'other' bucket holds disruptions that can't be attributed to an
    observed source (typically stochastic outcomes in the actual log, since
    the daily loop doesn't schedule calls during known absences).
    """
    out = {"account_unavail": 0, "rep_planned": 0, "rep_unplanned": 0, "other": 0}
    planned_days: Dict[int, set] = {}
    unplanned_days: Dict[int, set] = {}
    for ev in planned:
        days = ev.absent_day_indices or list(range(ev.start_day, ev.start_day + ev.duration_days))
        bucket = planned_days if ev.notice_days >= 7 else unplanned_days
        for d in days:
            bucket.setdefault(ev.rep_id, set()).add(d)
    for ev in sick:
        unplanned_days.setdefault(ev.rep_id, set()).add(ev.start_day)

    for c in calls:
        if c.date_idx not in eval_window or c.outcome == "completed":
            continue
        if not availability[c.account_id, c.date_idx]:
            out["account_unavail"] += 1
        elif c.date_idx in planned_days.get(c.rep_id, ()):
            out["rep_planned"] += 1
        elif c.date_idx in unplanned_days.get(c.rep_id, ()):
            out["rep_unplanned"] += 1
        else:
            out["other"] += 1
    return out


# ---- Reference replanner (Section 11.3).

def replan(plan: List[CallEvent], pop: Population, eligibility: np.ndarray,
           segment_per_day: np.ndarray, availability: np.ndarray,
           planned_by_rep: Dict[int, set], sick_by_rep: Dict[int, set],
           cfg: Config, scenario_index: int) -> List[CallEvent]:
    """Mark disrupted calls as failed and try a greedy substitute.

    Plan-agnostic. The same replanner runs against every algorithm's plan so
    Robustness measures the plan, not the algorithm.
    """
    r = rng(seed_for_replanner(cfg.seed, scenario_index))
    out: List[CallEvent] = []
    rep_by_id = {rep.rep_id: rep for rep in pop.reps}

    # Group calls by (rep, day) so we can substitute within a day.
    by_rd: Dict[Tuple[int, int], List[CallEvent]] = {}
    for c in plan:
        by_rd.setdefault((c.rep_id, c.date_idx), []).append(c)

    for (rep_id, day), day_calls in by_rd.items():
        rep = rep_by_id.get(rep_id)
        if rep is None:
            continue
        force = pop.forces[rep.force_id]
        bag = np.array(force.brands, dtype=int)
        bag_priorities = np.array(force.priorities, dtype=float)
        panel = rep.panel
        panel_bag_elig = eligibility[panel][:, bag]
        panel_seg_today = segment_per_day[panel, day]
        avail_panel = availability[panel, day]

        rep_absent = (day in planned_by_rep.get(rep_id, set())) or \
                     (day in sick_by_rep.get(rep_id, set()))

        already_called: set = set()
        n_a_b_local = np.zeros((panel.shape[0], eligibility.shape[1]), dtype=np.int32)

        for c in day_calls:
            failed = rep_absent or not availability[c.account_id, c.date_idx]
            if not failed:
                out.append(c)
                idx = np.where(panel == c.account_id)[0]
                if idx.size:
                    n_a_b_local[int(idx[0]), c.brand_id] += 1
                    already_called.add(int(idx[0]))
                continue

            # Rep gone for the day: just record the failure, no substitute.
            if rep_absent:
                out.append(CallEvent(
                    date_idx=c.date_idx, rep_id=c.rep_id, start_minute=c.start_minute,
                    planned_duration=c.planned_duration, actual_duration=0,
                    account_id=c.account_id, segment_at_call=c.segment_at_call,
                    brand_id=c.brand_id, brand_priority=c.brand_priority,
                    outcome="no_show",
                ))
                continue

            # Account unavailable: try to swap in someone else on the panel.
            cand = avail_panel.copy()
            for idx in already_called:
                if 0 <= idx < cand.size:
                    cand[idx] = False
            cand_idx = np.where(cand)[0]
            if cand_idx.size == 0:
                out.append(CallEvent(
                    date_idx=c.date_idx, rep_id=c.rep_id, start_minute=c.start_minute,
                    planned_duration=c.planned_duration, actual_duration=0,
                    account_id=c.account_id, segment_at_call=c.segment_at_call,
                    brand_id=c.brand_id, brand_priority=c.brand_priority,
                    outcome="no_show",
                ))
                continue

            seg_now = panel_seg_today[cand_idx]
            v = SEG_VALUE[seg_now]
            targets = SEG_TARGETS[seg_now]
            bag_n = n_a_b_local[cand_idx][:, bag]
            bag_elig = panel_bag_elig[cand_idx]
            tgt_col = targets.reshape(-1, 1)
            f_b = np.maximum(0.0, 1.0 - np.abs(bag_n - tgt_col) / np.maximum(1e-9, tgt_col))
            scores = v * (bag_priorities.reshape(1, -1) * bag_elig.astype(float) * f_b).sum(axis=1)
            if scores.max() <= 0:
                continue

            pick_local = int(np.argmax(scores))
            pick = int(cand_idx[pick_local])
            elig_mask = bag_elig[pick_local]
            if not elig_mask.any():
                continue
            brand_scores = bag_priorities * elig_mask.astype(float) * f_b[pick_local]
            if brand_scores.sum() <= 0:
                brand_scores = bag_priorities * elig_mask.astype(float)
            chosen_brand_idx = int(np.argmax(brand_scores))
            brand_id = int(bag[chosen_brand_idx])
            brand_priority = float(bag_priorities[chosen_brand_idx])

            seg_idx = int(panel_seg_today[pick])
            planned_dur = _sample_duration(r, rep.rep_type, seg_idx)
            out.append(CallEvent(
                date_idx=c.date_idx, rep_id=c.rep_id, start_minute=c.start_minute,
                planned_duration=planned_dur, actual_duration=planned_dur,
                account_id=int(panel[pick]), segment_at_call=seg_idx,
                brand_id=brand_id, brand_priority=brand_priority,
                outcome="completed",
            ))
            n_a_b_local[pick, brand_id] += 1
            already_called.add(pick)

    return out


# ---- Robustness (Eq. 4).

from dataclasses import dataclass


@dataclass
class RobustnessResult:
    """Outcome of one robustness evaluation.

    ratio           mean of Sales(replan) / Sales(plan) across scenarios
    absolute_loss   mean of Sales(plan) - Sales(replan) in original units
    base_sales      Sales(plan), the denominator and the size signal
    n_scenarios     number of disruption samples averaged
    flagged         True when base_sales is below `min_sales` and the ratio
                    is therefore degenerate (a near-empty plan reports
                    ratio close to 1 regardless of how good it is)

    The ratio is the spec1 metric. absolute_loss is the companion view
    that does not reward empty plans. Read both, especially when comparing
    algorithms that produce plans of different sizes.
    """
    ratio: float
    absolute_loss: float
    base_sales: float
    n_scenarios: int
    flagged: bool

    def __float__(self) -> float:
        # Backwards compatible: anything that treated robustness as a scalar
        # still gets a float.
        return self.ratio


def robustness(plan: List[CallEvent], pop: Population, eligibility: np.ndarray,
               segment_per_day: np.ndarray,
               availability_samples: List[np.ndarray],
               planned_by_rep: Dict[int, set], sick_by_rep: Dict[int, set],
               cfg: Config, *, eval_window: range,
               min_sales: float = 1.0) -> RobustnessResult:
    """Compute robustness ratio + absolute loss.

    `min_sales` is the threshold below which the ratio is flagged as
    degenerate. The default of 1.0 sales units rules out essentially
    empty plans (a single A-tier completed call is worth ~0.28 by Eq. 1).
    Choose a domain-appropriate threshold for your run.
    """
    if cfg.p_account_unavail <= 0.0:
        return RobustnessResult(ratio=1.0, absolute_loss=0.0,
                                base_sales=0.0, n_scenarios=0, flagged=False)
    base = sales(plan, pop, eval_window=eval_window)
    if base <= 0:
        return RobustnessResult(ratio=1.0, absolute_loss=0.0,
                                base_sales=base, n_scenarios=0, flagged=True)

    ratios: List[float] = []
    losses: List[float] = []
    for i, sample in enumerate(availability_samples):
        replanned = replan(plan, pop, eligibility, segment_per_day, sample,
                           planned_by_rep, sick_by_rep, cfg, scenario_index=i)
        replan_sales = sales(replanned, pop, eval_window=eval_window)
        ratios.append(replan_sales / base)
        losses.append(base - replan_sales)

    flagged = base < min_sales
    return RobustnessResult(
        ratio=(float(np.mean(ratios)) if ratios else 1.0),
        absolute_loss=(float(np.mean(losses)) if losses else 0.0),
        base_sales=float(base),
        n_scenarios=len(ratios),
        flagged=flagged,
    )


# ---- Replan cost (Section 11.5).

def replan_cost(plan: List[CallEvent], pop: Population, eligibility: np.ndarray,
                segment_per_day: np.ndarray,
                availability_samples: List[np.ndarray],
                planned_by_rep: Dict[int, set], sick_by_rep: Dict[int, set],
                cfg: Config) -> float:
    times = []
    for i, sample in enumerate(availability_samples):
        t0 = time.perf_counter()
        replan(plan, pop, eligibility, segment_per_day, sample,
               planned_by_rep, sick_by_rep, cfg, scenario_index=i)
        times.append(time.perf_counter() - t0)
    return median(times) if times else 0.0
