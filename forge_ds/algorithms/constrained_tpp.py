"""Constrained Neural TPP.

Same model and weights as Neural TPP. Inference runs beam search with
six constraints active: panel, no-repetition, availability, eligibility,
capacity (face-time bound), and frequency over-call.

When no beam survives, a soft fallback relaxes constraints in order, and
if everything fails, the rep-day falls back to a naive plan. Every
invocation of the fallback path is logged so the harness can track how
often the soft path was needed.
"""
from __future__ import annotations

import math
from collections import Counter
from datetime import date
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from .base import PlanContext, Plan, PlannedCall
from .beam_tpp import BeamTPPAlgorithm, _Beam
from .neural_tpp import (
    DAY_START_MIN, DAY_END_MIN, INTER_CALL_GAP, DURATION_VALUES,
    REP_TYPE_TO_IDX, SEG_TO_IDX,
)


# annual call targets per segment.
SEGMENT_TARGET = {"A": 24, "B": 12, "C": 6}


class ConstrainedTPPAlgorithm(BeamTPPAlgorithm):
    name = "constrained_tpp"

    def __init__(self, config: dict = None):
        super().__init__(config or {})
        self.over_call_threshold = float(self.config.get("over_call_threshold", 1.5))
        # 360 = 6 hours actual face time. The earlier default of 540 matched
        # the full 9-to-6 clock, which made the capacity check identical to
        # the day-end check and never binding.
        self.capacity_minutes = int(self.config.get("capacity_minutes", 360))
        self.fallback_capacity_minutes = int(self.config.get("fallback_capacity_minutes", 480))
        # Window length is needed to scale the annual frequency target down
        # to a per-window cap. Defaults to 14 days (rolling).
        self.window_days = int(self.config.get("window_days", 14))
        # Ablation knob: when True, no attempt enforces the over-call cap.
        # Used to isolate capacity-only constraints from over-call constraints.
        self.disable_over_call = bool(self.config.get("disable_over_call", False))
        # Per-window running count of (account, brand) calls, used by the
        # frequency over-call constraint. Reset at predict_window entry.
        self._n_ab_window: Counter = Counter()
        # Log how many rep-days needed the soft fallback path.
        self.softfallback_invocations: int = 0

    def predict_window(self, context: PlanContext, window_start,
                       window_days: int = 14) -> Plan:
        self._n_ab_window = Counter()
        self.window_days = window_days
        return super().predict_window(context, window_start, window_days)

    def _infer_one_day(self, *, context: PlanContext, d: date, available: set,
                       panel: List[int], panel_to_local: Dict[int, int],
                       rng,
                       strategy: str) -> List[PlannedCall]:
        """Constrained beam search with soft fallback."""
        # First attempt: all constraints active. If disable_over_call is set,
        # we skip the over-call cap on every attempt and rely on capacity alone.
        if self.disable_over_call:
            attempts = [
                dict(over_call=None,
                     capacity=self.capacity_minutes,
                     enforce_over_call=False),
                dict(over_call=None,
                     capacity=self.fallback_capacity_minutes,
                     enforce_over_call=False),
            ]
        else:
            attempts = [
                dict(over_call=self.over_call_threshold,
                     capacity=self.capacity_minutes,
                     enforce_over_call=True),
                dict(over_call=2.0,
                     capacity=self.capacity_minutes,
                     enforce_over_call=True),
                dict(over_call=None,
                     capacity=self.capacity_minutes,
                     enforce_over_call=False),
                dict(over_call=None,
                     capacity=self.fallback_capacity_minutes,
                     enforce_over_call=False),
            ]

        for i, params in enumerate(attempts):
            calls = self._beam_with_constraints(
                context=context, d=d, available=available, panel=panel,
                panel_to_local=panel_to_local, params=params,
            )
            if calls:
                if i > 0:
                    self.softfallback_invocations += 1
                for c in calls:
                    self._n_ab_window[(c.account_id, c.brand_id)] += 1
                return calls

        # Total failure: naive plan for this rep-day.
        self.softfallback_invocations += 1
        return self._naive_day(context, d, available, panel, panel_to_local, rng)

    def _beam_with_constraints(self, *, context: PlanContext, d: date,
                               available: set, panel: List[int],
                               panel_to_local: Dict[int, int],
                               params: dict) -> List[PlannedCall]:
        beams = [_Beam()]
        dow = d.weekday()
        rt_idx = REP_TYPE_TO_IDX.get(context.rep_type, 1)
        capacity = params["capacity"]
        enforce_oc = params["enforce_over_call"]
        oc_thr = params["over_call"] or 0.0

        for _step in range(20):
            extensions: List[Tuple[_Beam, float]] = []
            any_active = False
            for beam in beams:
                if beam.cur_minute >= DAY_END_MIN:
                    extensions.append((beam, beam.score))
                    continue
                face_time = sum(c.planned_duration for c in beam.calls)
                if face_time >= capacity:
                    extensions.append((beam, beam.score))
                    continue
                any_active = True

                tokens, mask, acc_hist, brn_hist = self._build_running(
                    beam.seq_time, beam.seq_acc, beam.seq_brn, beam.seq_seg,
                    dow, rt_idx)
                with torch.no_grad():
                    out = self.model(tokens, mask, acc_hist, brn_hist)
                last = -1
                eod_p = torch.sigmoid(out["eod_logit"][0, last]).item()
                if eod_p > 0.5 and len(beam.calls) > 0:
                    extensions.append((beam, beam.score + math.log(eod_p + 1e-9)))
                    continue

                mix = F.softmax(out["time_mix_logits"][0, last], dim=-1).cpu().numpy()
                mu = out["time_mu"][0, last].cpu().numpy()
                dt = float(np.exp((mix * mu).sum()))
                next_minute = int(beam.cur_minute + max(0, dt))
                if next_minute >= DAY_END_MIN:
                    extensions.append((beam, beam.score))
                    continue

                acc_logits = out["account_logits"][0, last].cpu().numpy()
                acc_mask = self._constrained_account_mask(
                    acc_logits, available, beam.called, panel, panel_to_local,
                    context=context, enforce_oc=enforce_oc, oc_thr=oc_thr,
                )
                if (acc_mask == -1e9).all():
                    extensions.append((beam, beam.score))
                    continue
                scored = acc_logits + acc_mask
                top_acc = self._topk_indices(scored, self.beam_width)

                for li, acc_lp in top_acc:
                    account_id = panel[li]
                    seg = context.segments.get(account_id, "B")

                    br_logits = out["brand_logits"][0, last].cpu().numpy()
                    br_mask = np.full(self.num_brands, -1e9, dtype=np.float32)
                    # Annual target scaled to the 14-day window so the cap is
                    # reachable. A-tier becomes 24 * 14/365 ~= 0.92 per window.
                    window_target = SEGMENT_TARGET.get(seg, 12) * self.window_days / 365.0
                    for b in context.bag:
                        if b not in context.eligibility.get(account_id, []):
                            continue
                        if enforce_oc:
                            n = self._n_ab_window[(account_id, b)] \
                                + sum(1 for c in beam.calls
                                      if c.account_id == account_id and c.brand_id == b)
                            if n >= oc_thr * window_target:
                                continue
                        br_mask[min(b, self.num_brands - 1)] = 0.0
                    if (br_mask == -1e9).all():
                        continue
                    brand_id = int(np.argmax(br_logits + br_mask))
                    brand_lp = float(self._softmax_value(br_logits + br_mask, brand_id))
                    brand_priority = self._priority_for(context, brand_id)

                    dur_logits = out["duration_logits"][0, last].cpu().numpy()
                    dur_idx = int(np.argmax(dur_logits))
                    dur = int(DURATION_VALUES[dur_idx])
                    dur_lp = float(self._softmax_value(dur_logits, dur_idx))

                    if face_time + dur > capacity:
                        continue
                    end_min = next_minute + dur
                    if end_min > DAY_END_MIN:
                        continue

                    extended = beam.clone()
                    extended.calls.append(PlannedCall(
                        date=d, rep_id=context.rep_id, start_minute=next_minute,
                        planned_duration=dur, account_id=account_id,
                        segment_at_call=seg, brand_id=brand_id,
                        brand_priority=brand_priority,
                    ))
                    extended.called.add(account_id)
                    extended.cur_minute = end_min + INTER_CALL_GAP
                    extended.score = beam.score + acc_lp + brand_lp + dur_lp
                    extended.seq_acc.append(li)
                    extended.seq_brn.append(brand_id)
                    extended.seq_seg.append(SEG_TO_IDX.get(seg, 1))
                    extended.seq_time.append(math.log1p(max(0.0, dt)))
                    extensions.append((extended, extended.score))

            if not extensions:
                break
            extensions.sort(key=lambda x: x[1], reverse=True)
            beams = [b for b, _ in extensions[:self.beam_width]]
            if not any_active:
                break

        if not beams:
            return []
        beams.sort(key=lambda b: b.score, reverse=True)
        return beams[0].calls

    def _constrained_account_mask(self, acc_logits: np.ndarray, available: set,
                                  called: set, panel: List[int],
                                  panel_to_local: Dict[int, int], *,
                                  context: PlanContext, enforce_oc: bool,
                                  oc_thr: float) -> np.ndarray:
        mask = np.full_like(acc_logits, -1e9, dtype=np.float32)
        for a in available:
            if a in called:
                continue
            li = panel_to_local.get(a)
            if li is None or li >= len(mask):
                continue
            # Account is feasible only if at least one bag brand survives
            # the over-call test for this account.
            seg = context.segments.get(a, "B")
            window_target = SEGMENT_TARGET.get(seg, 12) * self.window_days / 365.0
            ok = False
            for b in context.bag:
                if b not in context.eligibility.get(a, []):
                    continue
                if not enforce_oc:
                    ok = True
                    break
                n = self._n_ab_window[(a, b)]
                if n < oc_thr * window_target:
                    ok = True
                    break
            if ok:
                mask[li] = 0.0
        return mask

    def _naive_day(self, context: PlanContext, d: date, available: set,
                   panel: List[int], panel_to_local: Dict[int, int],
                   rng) -> List[PlannedCall]:
        """Frequency-cadence pick over the panel, ignoring soft constraints."""
        from .base import sample_brand_for_account, assign_start_times
        ranked = sorted(available, key=lambda a: SEG_TO_IDX.get(
            context.segments.get(a, "B"), 1))
        calls: List[PlannedCall] = []
        for a in ranked[:8]:
            seg = context.segments.get(a, "B")
            brand_id, pri = sample_brand_for_account(
                rng, a, context.bag, context.priorities, context.eligibility)
            calls.append(PlannedCall(
                date=d, rep_id=context.rep_id, start_minute=0,
                planned_duration=45, account_id=a, segment_at_call=seg,
                brand_id=brand_id, brand_priority=pri,
            ))
        return assign_start_times(calls)
