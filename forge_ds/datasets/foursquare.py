"""Foursquare TSMC 2014 adapter (spec2 Section 10.2).

The TSMC 2014 file is one tab-separated row per check-in. We map check-ins
to the spec1 schema as follows.

    rep            user
    account        venue
    panel          user's top 200 venues by visit count
    segment        per-user quantile of venue visit count (A 15%, B 35%, C 50%)
    call event     check-in (one row in activity_log)
    brand          random per-force assignment
    priority       drawn from spec1 §5 regimes
    duration       sampled from spec1 segment-rep-type distribution
    rep type       quantile of total check-ins per user

The segment / brand / priority overlay is synthetic; this dataset is for
structural-validity checks, not real-world fidelity (spec2 §10.1).

Uncertainty (absence, availability, churn) is injected on top using the
same procedures as spec1 §7.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Reach into forge-synth for the spec1 sampling logic so absence/availability/
# churn injection is exactly the same as the synthetic generator.
HERE = Path(__file__).resolve().parent
FORGE_SYNTH_CODE = HERE.parent.parent / "forge-synth" / "code"
if str(FORGE_SYNTH_CODE) not in sys.path:
    sys.path.insert(0, str(FORGE_SYNTH_CODE))

import config as fs_config              # noqa: E402
import uncertainty as fs_uncertainty    # noqa: E402
import simulate as fs_simulate          # noqa: E402
from world import Rep, ForceConfig      # noqa: E402


SEG_NAMES = ("A", "B", "C")
SEG_BOUNDARIES = (0.15, 0.50, 1.00)     # top 15% = A, next 35% = B, last 50% = C

REP_TYPE_BUCKETS = (0.20, 0.70, 1.00)   # by total check-ins; matches spec1 mix


# ---- Load raw.

def load_raw(path: str) -> pd.DataFrame:
    cols = ["user_id", "venue_id", "category_id", "category_name",
            "latitude", "longitude", "tz_offset_min", "utc_time"]
    df = pd.read_csv(path, sep="\t", header=None, names=cols,
                     dtype={"user_id": int, "venue_id": str},
                     encoding="latin-1")
    df["ts"] = pd.to_datetime(df["utc_time"], format="%a %b %d %H:%M:%S %z %Y", utc=True)
    df["local_ts"] = df["ts"] + pd.to_timedelta(df["tz_offset_min"], unit="m")
    df["date"] = df["local_ts"].dt.date
    df["start_minute"] = df["local_ts"].dt.hour * 60 + df["local_ts"].dt.minute
    return df


# ---- Filtering.

def filter_active(df: pd.DataFrame, min_user_checkins: int = 50,
                  min_venue_checkins: int = 10) -> pd.DataFrame:
    user_counts = df["user_id"].value_counts()
    df = df[df["user_id"].isin(user_counts[user_counts >= min_user_checkins].index)]
    venue_counts = df["venue_id"].value_counts()
    df = df[df["venue_id"].isin(venue_counts[venue_counts >= min_venue_checkins].index)]
    return df.reset_index(drop=True)


# ---- Panel and metadata construction.

def build_panel_and_segments(df: pd.DataFrame, panel_size: int = 200
                             ) -> Tuple[Dict[int, List[str]], Dict[int, Dict[str, str]]]:
    """For each user: top panel_size venues by visit count, plus per-venue
    quantile segment."""
    panels: Dict[int, List[str]] = {}
    segments: Dict[int, Dict[str, str]] = {}
    for user_id, ev in df.groupby("user_id"):
        counts = ev["venue_id"].value_counts()
        top = counts.head(panel_size)
        panels[int(user_id)] = top.index.tolist()
        n = len(top)
        if n == 0:
            segments[int(user_id)] = {}
            continue
        ranks = (np.arange(n) + 1) / n
        venue_to_seg: Dict[str, str] = {}
        for venue, rank in zip(top.index, ranks):
            if rank <= SEG_BOUNDARIES[0]:
                venue_to_seg[venue] = "A"
            elif rank <= SEG_BOUNDARIES[1]:
                venue_to_seg[venue] = "B"
            else:
                venue_to_seg[venue] = "C"
        segments[int(user_id)] = venue_to_seg
    return panels, segments


def assign_rep_types(df: pd.DataFrame, rng: np.random.Generator) -> Dict[int, str]:
    """User type by quantile of total check-ins."""
    counts = df.groupby("user_id").size().sort_values(ascending=False)
    n = len(counts)
    out: Dict[int, str] = {}
    for i, (uid, _) in enumerate(counts.items()):
        rank = (i + 1) / n
        if rank <= REP_TYPE_BUCKETS[0]:
            out[int(uid)] = "high-volume"
        elif rank <= REP_TYPE_BUCKETS[1]:
            out[int(uid)] = "mid-market"
        else:
            out[int(uid)] = "specialty"
    return out


def assign_forces(user_ids: List[int], num_forces: int,
                  rng: np.random.Generator) -> Dict[int, int]:
    return {int(u): int(rng.integers(0, num_forces)) for u in user_ids}


def build_brand_eligibility(venues: List[str], num_brands: int,
                            rng: np.random.Generator,
                            mean_eligibility: float = 0.30
                            ) -> Dict[str, List[int]]:
    """Each venue gets a random subset of brands, with each brand included
    independently at probability `mean_eligibility`."""
    elig: Dict[str, List[int]] = {}
    for v in venues:
        mask = rng.random(num_brands) < mean_eligibility
        if not mask.any():
            mask[int(rng.integers(0, num_brands))] = True
        elig[v] = [int(b) for b in np.where(mask)[0]]
    return elig


def assign_force_bag(num_forces: int, num_brands: int,
                     rng: np.random.Generator) -> Tuple[Dict[int, List[int]],
                                                          Dict[int, List[float]],
                                                          Dict[int, str]]:
    """Mirror spec1 §5 bag and priority sampling."""
    base, rem = divmod(num_brands, num_forces)
    force_brands: Dict[int, List[int]] = {}
    cursor = 0
    for f in range(num_forces):
        size = base + (1 if f < rem else 0)
        force_brands[f] = list(range(cursor, cursor + size))
        cursor += size

    regimes = {
        1: [("single", [1.0])],
        2: [("balanced", [0.5, 0.5]),
            ("moderate", [0.7, 0.3]),
            ("heavy",    [0.8, 0.2])],
        3: [("balanced", [0.34, 0.33, 0.33]),
            ("moderate", [0.5, 0.3, 0.2]),
            ("heavy",    [0.6, 0.3, 0.1])],
    }
    bag: Dict[int, List[int]] = {}
    pris: Dict[int, List[float]] = {}
    regime: Dict[int, str] = {}
    for f in range(num_forces):
        avail = force_brands[f]
        size = min(int(rng.choice([1, 2, 3], p=[0.2, 0.5, 0.3])), len(avail))
        chosen = sorted(int(b) for b in rng.choice(avail, size=size, replace=False))
        bag[f] = chosen
        rname, weights = regimes[size][int(rng.integers(0, len(regimes[size])))]
        pris[f] = list(weights)
        regime[f] = rname
    return bag, pris, regime


# ---- Duration sampling.

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


def sample_duration(rng: np.random.Generator, rep_type: str, seg: str) -> int:
    probs = DURATION_PROBS.get((rep_type, seg), [0.25] * 4)
    return int(rng.choice(DURATION_VALUES, p=np.array(probs) / sum(probs)))


# ---- Uncertainty injection.

def inject_uncertainty(*, cfg: fs_config.Config, df: pd.DataFrame,
                       panels: Dict[int, List[str]],
                       rep_types: Dict[int, str], forces: Dict[int, int],
                       start: date, horizon: int,
                       rng: np.random.Generator):
    """Plan rep-side absences, churn, and venue-side unavailability."""
    user_ids = sorted(panels.keys())
    rid_of_user = {uid: i for i, uid in enumerate(user_ids)}
    venue_ids = sorted({v for vs in panels.values() for v in vs})
    aid_of_venue = {v: i for i, v in enumerate(venue_ids)}

    # Rep absences and sick days, using spec1's calendars + daily sick draw.
    planned_by_rep: Dict[int, list] = {}
    sick_by_rep: Dict[int, list] = {}
    for uid in user_ids:
        rid = rid_of_user[uid]
        events = fs_uncertainty.build_planned_absences(cfg, rid, start, horizon)
        planned_by_rep[uid] = events

        rep_rng = fs_config.rng(fs_config.seed_for_rep(cfg.seed, rid))
        rep_rng.bytes(64)
        rep_rate = float(np.clip(
            rep_rng.normal(cfg.sick_days_per_year_mean, cfg.sick_days_per_year_std),
            0.0, 20.0))
        sick_events = []
        was_sick = False
        for t in range(horizon):
            d = start + timedelta(days=t)
            if d.weekday() >= 5:
                was_sick = False
                continue
            wdy = sum(1 for k in range(366)
                      if (date(d.year, 1, 1) + timedelta(days=k)).year == d.year
                      and (date(d.year, 1, 1) + timedelta(days=k)).weekday() < 5)
            sick = fs_uncertainty.sample_sick_day(
                rep_rng, rep_rate=rep_rate, was_sick_yesterday=was_sick,
                today=d, working_days_in_year=max(1, wdy),
                sick_autocorr=cfg.sick_autocorr,
            )
            if sick:
                sick_events.append(fs_uncertainty.AbsenceEvent(
                    rep_id=rid, event_type="sick", start_day=t,
                    duration_days=1, notice_days=0))
                was_sick = True
            else:
                was_sick = False
        sick_by_rep[uid] = sick_events

    # Venue availability: per venue Markov chain, vectorized using spec1's
    # logic by re-keying account ids.
    N = len(venue_ids)
    avail = np.ones((N, horizon), dtype=bool)
    if cfg.p_account_unavail > 0:
        pers = cfg.account_unavail_persistence
        p_av_to_un = (cfg.p_account_unavail * (1.0 - pers)) / max(1e-12, 1.0 - cfg.p_account_unavail)
        for vi, v in enumerate(venue_ids):
            r = fs_config.rng(fs_config.seed_for_account(cfg.seed, vi))
            is_avail = r.random() >= cfg.p_account_unavail
            u = r.random(horizon)
            row = avail[vi]
            for t in range(horizon):
                row[t] = is_avail
                if is_avail:
                    is_avail = u[t] >= p_av_to_un
                else:
                    is_avail = u[t] >= pers

    # Churn: same model as spec1 §7.3, fires once per user.
    churn_events = []
    p_horizon = float(np.clip(cfg.p_churn_annual * horizon / 365.0, 0.0, 1.0))
    for uid in user_ids:
        rid = rid_of_user[uid]
        r = fs_config.rng(fs_config.seed_for_rep(cfg.seed, rid))
        r.bytes(8)
        if r.random() >= p_horizon:
            continue
        depart_day = int(r.integers(1, horizon))
        announce_day = max(0, depart_day - cfg.churn_notice_days)
        churn_events.append((rid, announce_day, depart_day))

    return rid_of_user, aid_of_venue, planned_by_rep, sick_by_rep, avail, churn_events


SEG_NAME_TO_IDX = {"A": 0, "B": 1, "C": 2}


def _build_reference_structures(*, cfg: fs_config.Config,
                                users: List[int], venues: List[str],
                                panels: Dict[int, List[str]],
                                rep_types: Dict[int, str], forces: Dict[int, int],
                                bag: Dict[int, List[int]],
                                priorities: Dict[int, List[float]],
                                regime: Dict[int, str],
                                segments_global: Dict[str, str],
                                brand_eligibility: Dict[str, List[int]],
                                rid_of_user: Dict[int, int],
                                aid_of_venue: Dict[str, int],
                                venue_avail: np.ndarray, horizon: int):
    """Convert Foursquare adapter state into the shapes greedy / naive expect."""
    n_accts = len(venues)
    n_brands = cfg.num_brands_total

    eligibility = np.zeros((n_accts, n_brands), dtype=bool)
    for v, bs in brand_eligibility.items():
        if v not in aid_of_venue:
            continue
        ai = aid_of_venue[v]
        for b in bs:
            if 0 <= b < n_brands:
                eligibility[ai, b] = True

    segment_per_day = np.zeros((n_accts, horizon), dtype=np.int8)
    for v in venues:
        ai = aid_of_venue[v]
        idx = SEG_NAME_TO_IDX.get(segments_global.get(v, "B"), 1)
        segment_per_day[ai, :] = idx

    # Forces: one ForceConfig per force_id we've used.
    forces_list: List[ForceConfig] = []
    seen_forces = sorted(set(forces.values()))
    for fid in seen_forces:
        forces_list.append(ForceConfig(
            force_id=fid,
            brands=list(bag[fid]),
            priorities=list(priorities[fid]),
            regime=regime.get(fid, "balanced"),
        ))
    fid_to_force = {f.force_id: f for f in forces_list}

    # Reps: panel maps user-venue strings to account ids.
    reps: Dict[int, Rep] = {}
    for uid in users:
        rid = rid_of_user[uid]
        panel_ids = np.array(
            [aid_of_venue[v] for v in panels.get(uid, []) if v in aid_of_venue],
            dtype=np.int64,
        )
        panel_ids.sort()
        reps[rid] = Rep(
            rep_id=rid, rep_type=rep_types.get(uid, "mid-market"),
            force_id=forces.get(uid, 0),
            panel=panel_ids,
            hire_date_idx=0, departure_date_idx=-1, replacement_of=-1,
        )
    return reps, fid_to_force, eligibility, segment_per_day


def _events_to_rows(events, scenario_id: str, start: date, start_eid: int):
    """Turn forge-synth CallEvent dataclasses into activity_log CSV rows."""
    rows = []
    eid = start_eid
    seg_names = ("A", "B", "C")
    for c in events:
        d = (start + timedelta(days=c.date_idx)).isoformat()
        start_time = f"{c.start_minute // 60:02d}:{c.start_minute % 60:02d}"
        rows.append([
            eid, d, c.rep_id, start_time,
            c.planned_duration, c.actual_duration, c.account_id,
            seg_names[c.segment_at_call], c.brand_id,
            f"{c.brand_priority:.4f}", c.outcome, scenario_id,
        ])
        eid += 1
    return rows, eid


def _generate_reference_plans(*, cfg: fs_config.Config,
                              reps: Dict[int, Rep],
                              fid_to_force: Dict[int, ForceConfig],
                              eligibility: np.ndarray,
                              segment_per_day: np.ndarray,
                              venue_avail: np.ndarray,
                              planned_by_rep: Dict[int, list],
                              sick_by_rep: Dict[int, list],
                              rid_of_user: Dict[int, int],
                              start: date, horizon: int):
    """Run greedy_upper_bound and naive_plan for every rep."""
    greedy_all = []
    naive_all = []
    active_window = (0, horizon)
    user_of_rep = {rid: uid for uid, rid in rid_of_user.items()}
    for rid, rep in reps.items():
        force = fid_to_force[rep.force_id]
        uid = user_of_rep.get(rid)
        planned = planned_by_rep.get(uid, [])
        sick = sick_by_rep.get(uid, [])
        g = fs_simulate.greedy_upper_bound(
            cfg, rep, force, eligibility, segment_per_day, venue_avail,
            planned, sick, start, horizon, active_window,
        )
        n = fs_simulate.naive_plan(
            cfg, rep, force, eligibility, segment_per_day, venue_avail,
            planned, sick, start, horizon, active_window,
        )
        greedy_all.extend(g)
        naive_all.extend(n)
    greedy_all.sort(key=lambda c: (c.date_idx, c.rep_id, c.start_minute))
    naive_all.sort(key=lambda c: (c.date_idx, c.rep_id, c.start_minute))
    return greedy_all, naive_all


# ---- Output writing.

def emit_spec1_schema(*, out_dir: str, cfg: fs_config.Config,
                      df: pd.DataFrame, panels: Dict[int, List[str]],
                      segments: Dict[int, Dict[str, str]],
                      rep_types: Dict[int, str], forces: Dict[int, int],
                      bag: Dict[int, List[int]], priorities: Dict[int, List[float]],
                      regime: Dict[int, str],
                      brand_eligibility: Dict[str, List[int]],
                      rid_of_user: Dict[int, int], aid_of_venue: Dict[str, int],
                      planned_by_rep, sick_by_rep, venue_avail: np.ndarray,
                      churn_events,
                      start: date, horizon: int,
                      rng: np.random.Generator) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # config.json + run_id.txt
    cfg_text = cfg.to_json()
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        f.write(cfg_text)
    h = hashlib.sha256(cfg_text.encode()).hexdigest()[:16]
    with open(os.path.join(out_dir, "run_id.txt"), "w") as f:
        f.write(h + "\n")

    # population.csv
    users = sorted(rid_of_user.keys())
    with open(os.path.join(out_dir, "population.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rep_id", "type", "force_id", "panel_size", "hire_date", "departure_date"])
        for uid in users:
            rid = rid_of_user[uid]
            ps = len(panels.get(uid, []))
            w.writerow([rid, rep_types.get(uid, "mid-market"),
                        forces.get(uid, 0), ps, start.isoformat(), ""])
        # Churn replacements: keep things simple, no new rep types beyond
        # the originals; we just mark depart_date on the original row.
        # (For the harness this is enough; the spec1 representation creates
        # new rep_ids but the simplified mapping avoids extra accounting.)

    # accounts.csv
    venues = sorted(aid_of_venue.keys())
    with open(os.path.join(out_dir, "accounts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["account_id", "specialty_id", "initial_segment", "eligible_brands"])
        # Pick a segment that matches the most common assignment across users
        # for each venue. Falls back to "B".
        venue_to_seg_global: Dict[str, str] = {}
        from collections import Counter
        for venue in venues:
            votes = Counter()
            for uid, vmap in segments.items():
                if venue in vmap:
                    votes[vmap[venue]] += 1
            venue_to_seg_global[venue] = (votes.most_common(1)[0][0] if votes else "B")
            aid = aid_of_venue[venue]
            elig = ";".join(str(b) for b in brand_eligibility.get(venue, []))
            w.writerow([aid, 0, venue_to_seg_global[venue], elig])

    # panels.csv
    with open(os.path.join(out_dir, "panels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rep_id", "account_id", "assignment_start_date", "assignment_end_date"])
        for uid in users:
            rid = rid_of_user[uid]
            for v in panels.get(uid, []):
                w.writerow([rid, aid_of_venue[v], start.isoformat(), ""])

    # segment_history.csv (initial only; we don't simulate quarterly transitions for FSQ)
    with open(os.path.join(out_dir, "segment_history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["account_id", "effective_date", "old_segment", "new_segment"])
        for venue in venues:
            w.writerow([aid_of_venue[venue], start.isoformat(), "",
                        venue_to_seg_global.get(venue, "B")])

    # uncertainty_traces.csv: rep planned + sick + churn + venue unavail blocks
    with open(os.path.join(out_dir, "uncertainty_traces.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trace_id", "event_start_date", "entity_type", "entity_id",
                    "event_type", "notice_days", "duration_days", "scenario_id"])
        tid = 0
        # Venue unavailability blocks.
        for vi in range(venue_avail.shape[0]):
            row = venue_avail[vi]
            t = 0
            while t < horizon:
                if row[t]:
                    t += 1
                    continue
                s = t
                while t < horizon and not row[t]:
                    t += 1
                w.writerow([tid, (start + timedelta(days=s)).isoformat(),
                            "account", vi, "unavailable", cfg.account_notice_days,
                            t - s, "actual"])
                tid += 1
        # Rep absences.
        for uid in users:
            rid = rid_of_user[uid]
            for ev in planned_by_rep.get(uid, []) + sick_by_rep.get(uid, []):
                w.writerow([tid, (start + timedelta(days=ev.start_day)).isoformat(),
                            "rep", rid, ev.event_type, ev.notice_days,
                            ev.duration_days, "actual"])
                tid += 1
        # Churn.
        for rid, ann, dep in churn_events:
            w.writerow([tid, (start + timedelta(days=ann)).isoformat(),
                        "rep", rid, "churn_announce", 0, dep - ann, "actual"])
            tid += 1
            w.writerow([tid, (start + timedelta(days=dep)).isoformat(),
                        "rep", rid, "churn_depart", dep - ann, 0, "actual"])
            tid += 1

    # activity_log.csv: one row per check-in inside [start, start+horizon).
    end = start + timedelta(days=horizon)

    # Pre-build absence day sets per rep, so we can mark no_show on absent days.
    absent_days_by_rep: Dict[int, set] = {}
    for uid, evs in planned_by_rep.items():
        rid = rid_of_user[uid]
        s = absent_days_by_rep.setdefault(rid, set())
        for ev in evs:
            for d in (ev.absent_day_indices or range(ev.start_day, ev.start_day + ev.duration_days)):
                s.add(d)
    for uid, evs in sick_by_rep.items():
        rid = rid_of_user[uid]
        s = absent_days_by_rep.setdefault(rid, set())
        for ev in evs:
            s.add(ev.start_day)

    # Build the global per-venue segment vote here so we can reuse it
    # for the reference-plan structures below.
    from collections import Counter
    venues = sorted(aid_of_venue.keys())
    venue_to_seg_global: Dict[str, str] = {}
    for venue in venues:
        votes = Counter()
        for uid, vmap in segments.items():
            if venue in vmap:
                votes[vmap[venue]] += 1
        venue_to_seg_global[venue] = (votes.most_common(1)[0][0] if votes else "B")

    with open(os.path.join(out_dir, "activity_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "date", "rep_id", "start_time",
                    "planned_duration_min", "actual_duration_min", "account_id",
                    "segment_at_call", "brand_id", "brand_priority", "outcome",
                    "scenario_id"])
        eid = 0
        for _, row in df.iterrows():
            d = row["date"]
            if not (start <= d < end):
                continue
            day_idx = (d - start).days
            uid = int(row["user_id"])
            venue = row["venue_id"]
            if venue not in aid_of_venue:
                continue
            rid = rid_of_user[uid]
            aid = aid_of_venue[venue]
            seg = segments[uid].get(venue, "B")
            start_min = int(row["start_minute"])
            dur = sample_duration(rng, rep_types[uid], seg)

            # Brand from the rep's force bag, masked by venue eligibility.
            fid = forces[uid]
            options = [(b, p) for b, p in zip(bag[fid], priorities[fid])
                       if b in brand_eligibility.get(venue, [])]
            if not options:
                continue
            bs, ps = zip(*options)
            pp = np.array(ps, dtype=float)
            pp = pp / pp.sum()
            bi = int(rng.choice(len(bs), p=pp))
            brand_id, brand_priority = int(bs[bi]), float(ps[bi])

            # Outcome: no_show if rep absent OR venue unavailable today.
            outcome = "completed"
            if day_idx in absent_days_by_rep.get(rid, set()):
                outcome = "no_show"
            elif not venue_avail[aid, day_idx]:
                outcome = "no_show"
            actual = dur if outcome == "completed" else 0

            w.writerow([eid, d.isoformat(), rid, f"{start_min // 60:02d}:{start_min % 60:02d}",
                        dur, actual, aid, seg, brand_id, f"{brand_priority:.4f}",
                        outcome, "actual"])
            eid += 1

        # Reference plans: greedy_upper and naive scenarios.
        print("  generating greedy_upper + naive reference plans...")
        reps_struct, fid_to_force, eligibility_mtx, segment_per_day = \
            _build_reference_structures(
                cfg=cfg, users=users, venues=venues, panels=panels,
                rep_types=rep_types, forces=forces, bag=bag,
                priorities=priorities, regime=regime,
                segments_global=venue_to_seg_global,
                brand_eligibility=brand_eligibility,
                rid_of_user=rid_of_user, aid_of_venue=aid_of_venue,
                venue_avail=venue_avail, horizon=horizon,
            )
        greedy_events, naive_events = _generate_reference_plans(
            cfg=cfg, reps=reps_struct, fid_to_force=fid_to_force,
            eligibility=eligibility_mtx, segment_per_day=segment_per_day,
            venue_avail=venue_avail,
            planned_by_rep=planned_by_rep, sick_by_rep=sick_by_rep,
            rid_of_user=rid_of_user, start=start, horizon=horizon,
        )
        g_rows, eid = _events_to_rows(greedy_events, "greedy_upper", start, eid)
        for row in g_rows:
            w.writerow(row)
        n_rows, eid = _events_to_rows(naive_events, "naive", start, eid)
        for row in n_rows:
            w.writerow(row)
        print(f"  greedy_upper={len(greedy_events)}, naive={len(naive_events)}")

    # validation_stats.csv placeholder so downstream tools that read all 8
    # files don't trip.
    with open(os.path.join(out_dir, "validation_stats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric_name", "scope", "value"])
        w.writerow(["dataset", "foursquare", "1"])


# ---- Top-level.

def build_dataset(*, raw_path: str, out_dir: str, seed: int = 42,
                  panel_size: int = 200, num_brands: int = 6,
                  num_forces: int = 3,
                  start: date = date(2012, 4, 12),
                  horizon: int = 300) -> str:
    """Process a Foursquare TSV into spec1-schema CSVs. Returns out_dir."""
    rng = np.random.default_rng(seed)
    cfg = fs_config.Config(
        seed=seed, horizon_days=horizon, warmup_days=min(90, horizon // 3),
        start_date=start.isoformat(),
        num_brands_total=num_brands, num_forces=num_forces,
        output_dir=out_dir,
    )

    raw = load_raw(raw_path)
    raw = filter_active(raw)
    raw = raw[(raw["date"] >= start) & (raw["date"] < start + timedelta(days=horizon))]

    panels, segments = build_panel_and_segments(raw, panel_size=panel_size)
    rep_types = assign_rep_types(raw, rng)
    user_ids = sorted(panels.keys())
    forces = assign_forces(user_ids, num_forces, rng)
    bag, priorities, regime = assign_force_bag(num_forces, num_brands, rng)
    venues = sorted({v for vs in panels.values() for v in vs})
    brand_eligibility = build_brand_eligibility(venues, num_brands, rng)

    (rid_of_user, aid_of_venue, planned, sick, venue_avail,
     churn) = inject_uncertainty(
        cfg=cfg, df=raw, panels=panels, rep_types=rep_types,
        forces=forces, start=start, horizon=horizon, rng=rng,
    )

    emit_spec1_schema(
        out_dir=out_dir, cfg=cfg, df=raw, panels=panels, segments=segments,
        rep_types=rep_types, forces=forces, bag=bag, priorities=priorities,
        regime=regime, brand_eligibility=brand_eligibility,
        rid_of_user=rid_of_user, aid_of_venue=aid_of_venue,
        planned_by_rep=planned, sick_by_rep=sick, venue_avail=venue_avail,
        churn_events=churn, start=start, horizon=horizon, rng=rng,
    )
    return out_dir


def cli():
    import argparse
    p = argparse.ArgumentParser(description="Foursquare TSMC 2014 -> spec1 schema")
    p.add_argument("--city", choices=["nyc", "tky"], required=True)
    p.add_argument("--raw-root", default="public_dataset/foursquare/raw/dataset_tsmc2014")
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--panel-size", type=int, default=200)
    p.add_argument("--horizon", type=int, default=300)
    args = p.parse_args()

    fname = "dataset_TSMC2014_NYC.txt" if args.city == "nyc" else "dataset_TSMC2014_TKY.txt"
    raw_path = os.path.join(args.raw_root, fname)
    out_dir = args.out or f"public_dataset/foursquare/{args.city}"
    build_dataset(raw_path=raw_path, out_dir=out_dir, seed=args.seed,
                  panel_size=args.panel_size, horizon=args.horizon)
    print("wrote", out_dir)


if __name__ == "__main__":
    cli()
