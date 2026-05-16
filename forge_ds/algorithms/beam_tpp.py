"""Beam TPP ablation.

Same model and weights as Neural TPP. Inference uses beam search with
basic feasibility masks only (panel, no-rep, availability, eligibility).
No capacity or over-call constraints.

The point of this algorithm is to isolate the contribution of beam
search from the contribution of operational constraints. Figure 4
attributes the gap between Neural TPP and Beam TPP to search strategy,
and the gap between Beam TPP and Constrained TPP to constraints.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from .base import (
    PlanContext, Plan, PlannedCall, DisruptionEvent,
)
from .neural_tpp import (
    NeuralTPPAlgorithm, DAY_START_MIN, DAY_END_MIN, INTER_CALL_GAP,
    DURATION_VALUES, REP_TYPE_TO_IDX, SEG_TO_IDX, _gather_tokens,
)


class _Beam:
    """A partial plan being explored by beam search."""

    __slots__ = ("calls", "called", "cur_minute", "score",
                 "seq_acc", "seq_brn", "seq_seg", "seq_time")

    def __init__(self):
        self.calls: List[PlannedCall] = []
        self.called: set = set()
        self.cur_minute: int = DAY_START_MIN
        self.score: float = 0.0
        self.seq_acc: List[int] = [0]
        self.seq_brn: List[int] = [0]
        self.seq_seg: List[int] = [0]
        self.seq_time: List[float] = [0.0]

    def clone(self) -> "_Beam":
        b = _Beam()
        b.calls = list(self.calls)
        b.called = set(self.called)
        b.cur_minute = self.cur_minute
        b.score = self.score
        b.seq_acc = list(self.seq_acc)
        b.seq_brn = list(self.seq_brn)
        b.seq_seg = list(self.seq_seg)
        b.seq_time = list(self.seq_time)
        return b


class BeamTPPAlgorithm(NeuralTPPAlgorithm):
    name = "beam_tpp"

    def __init__(self, config: dict = None):
        super().__init__(config or {})
        self.beam_width = int(self.config.get("beam_width", 8))

    def _infer_one_day(self, *, context: PlanContext, d: date, available: set,
                       panel: List[int], panel_to_local: Dict[int, int],
                       rng: np.random.Generator,
                       strategy: str) -> List[PlannedCall]:
        """Beam search with basic feasibility masks only."""
        beams = [_Beam()]
        dow = d.weekday()
        rt_idx = REP_TYPE_TO_IDX.get(context.rep_type, 1)

        max_steps = 20
        for _step in range(max_steps):
            extensions: List[Tuple[_Beam, float]] = []
            any_active = False

            for beam in beams:
                if beam.cur_minute >= DAY_END_MIN:
                    extensions.append((beam, beam.score))
                    continue
                any_active = True

                tokens, mask, acc_hist, brn_hist = self._build_running(
                    beam.seq_time, beam.seq_acc, beam.seq_brn, beam.seq_seg,
                    dow, rt_idx)
                with torch.no_grad():
                    out = self.model(tokens, mask, acc_hist, brn_hist)
                last = -1

                # End-of-day decision: high eod terminates the beam.
                eod_p = torch.sigmoid(out["eod_logit"][0, last]).item()
                if eod_p > 0.5 and len(beam.calls) > 0:
                    extensions.append((beam, beam.score + math.log(eod_p + 1e-9)))
                    continue

                # Inter-event time: take the mixture mean as a point estimate.
                mix = F.softmax(out["time_mix_logits"][0, last], dim=-1).cpu().numpy()
                mu = out["time_mu"][0, last].cpu().numpy()
                dt = float(np.exp((mix * mu).sum()))
                next_minute = min(DAY_END_MIN, int(beam.cur_minute + max(0, dt)))
                if next_minute >= DAY_END_MIN:
                    extensions.append((beam, beam.score))
                    continue

                # Mask accounts by basic feasibility.
                acc_logits = out["account_logits"][0, last].cpu().numpy()
                acc_mask = self._account_mask(acc_logits, available, beam.called,
                                              panel, panel_to_local)
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
                    for b in context.bag:
                        if b in context.eligibility.get(account_id, []):
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

    @staticmethod
    def _topk_indices(scored: np.ndarray, k: int) -> List[Tuple[int, float]]:
        order = np.argsort(scored)[::-1]
        out = []
        for idx in order[:k]:
            v = scored[idx]
            if v <= -1e8:
                break
            out.append((int(idx), float(v)))
        return out

    @staticmethod
    def _account_mask(acc_logits: np.ndarray, available: set, called: set,
                      panel: List[int], panel_to_local: Dict[int, int]) -> np.ndarray:
        mask = np.full_like(acc_logits, -1e9, dtype=np.float32)
        for a in available:
            if a in called:
                continue
            li = panel_to_local.get(a)
            if li is not None and li < len(mask):
                mask[li] = 0.0
        return mask

    @staticmethod
    def _softmax_value(logits: np.ndarray, idx: int) -> float:
        x = logits - logits.max()
        e = np.exp(x)
        p = e / max(1e-12, e.sum())
        return math.log(max(1e-12, float(p[idx])))
