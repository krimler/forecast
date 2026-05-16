"""Four-pass data characterization, run on every dataset.

Pass 1: basic distributions.
Pass 2: predictability (variance explained by rep+dow; next-account
        logistic regression with and without sequence context).
Pass 3: disruption impact (planned-vs-realized).
Pass 4: concrete examples (5 random reps, 5 random accounts, reference
        plan inspection).

Each dataset gets one section; each section runs all four passes. The
final report at forge_ds/results/data_analysis.md is plain markdown,
no figures.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent           # project root (HERE = forge_ds/experiments)


DATASETS = [
    ("ForgeSynth default",   str(ROOT / "forge-synth/dataset/output_default")),
    ("ForgeSynth smoke",     str(ROOT / "forge-synth/dataset/output_smoke")),
    ("Foursquare NYC",       str(ROOT / "public_dataset/foursquare/nyc")),
    ("Foursquare Tokyo",     str(ROOT / "public_dataset/foursquare/tokyo")),
]


def _load(d: str) -> Dict:
    cfg = json.loads(open(os.path.join(d, "config.json")).read())
    out = {"config": cfg}
    for name in ("population", "accounts", "panels", "activity_log",
                 "uncertainty_traces", "segment_history"):
        path = os.path.join(d, f"{name}.csv")
        if os.path.exists(path):
            out[name] = pd.read_csv(path)
        else:
            out[name] = pd.DataFrame()
    return out


def _pct(s, p):
    if len(s) == 0:
        return float("nan")
    return float(np.percentile(s, p))


def _hist_bins(values, edges):
    counts = pd.cut(values, bins=edges, include_lowest=True).value_counts().sort_index()
    return [(str(idx), int(cnt)) for idx, cnt in counts.items()]


# ---- Pass 1.

def pass1_distributions(act_actual: pd.DataFrame, pop: pd.DataFrame,
                        u: pd.DataFrame, accts: pd.DataFrame,
                        horizon: int) -> List[str]:
    L = ["### Pass 1: distributions\n"]

    productive = act_actual[act_actual["outcome"] != "no_show"]
    productive = productive.assign(
        _d=pd.to_datetime(productive["date"]).dt.date,
        _minute=(pd.to_datetime(productive["start_time"], format="%H:%M").dt.hour * 60
                 + pd.to_datetime(productive["start_time"], format="%H:%M").dt.minute),
    )

    # Calls per (rep, day).
    rd = productive.groupby(["rep_id", "_d"]).size()
    cv = rd.std() / max(1e-9, rd.mean())
    L.append("**Calls per (rep, day):**")
    L.append("")
    L.append(f"- mean {rd.mean():.2f}, std {rd.std():.2f}, CV {cv:.2f}")
    L.append(f"- p10 / p50 / p90 / p99: "
             f"{_pct(rd, 10):.0f} / {_pct(rd, 50):.0f} / "
             f"{_pct(rd, 90):.0f} / {_pct(rd, 99):.0f}")
    edges = [0, 1, 2, 4, 6, 8, 10, 15, 100]
    L.append(f"- histogram: " + ", ".join(
        f"{b}={c}" for b, c in _hist_bins(rd, edges)))
    L.append("")

    # Call durations.
    L.append("**Call duration (min):**")
    L.append("")
    dur = productive["planned_duration_min"].astype(int)
    L.append(f"- mean {dur.mean():.1f}, distribution: " + ", ".join(
        f"{int(v)}min={cnt}" for v, cnt in dur.value_counts().sort_index().items()))
    L.append("")

    # Day-of-week.
    productive["_dow"] = pd.to_datetime(productive["date"]).dt.day_name()
    dow = productive.groupby("_dow").size()
    tot = max(1, dow.sum())
    L.append("**Day-of-week call share:**")
    L.append("")
    for d in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
              "Saturday", "Sunday"):
        L.append(f"- {d}: {100 * dow.get(d, 0) / tot:.1f}%")
    L.append("")

    # Per-rep absence days.
    L.append("**Per-rep absence days (from uncertainty_traces, rep rows):**")
    L.append("")
    if not u.empty:
        rep_u = u[u["entity_type"] == "rep"]
        per_rep_absent = rep_u.groupby("entity_id")["duration_days"].sum().astype(int)
        if len(per_rep_absent) > 0:
            L.append(f"- mean {per_rep_absent.mean():.1f} days/rep over {horizon}-day horizon")
            L.append(f"- p10 / p50 / p90: {_pct(per_rep_absent, 10):.0f} / "
                     f"{_pct(per_rep_absent, 50):.0f} / {_pct(per_rep_absent, 90):.0f}")
            for etype in ("sick", "personal", "vacation", "conference"):
                ev = rep_u[rep_u["event_type"] == etype]
                if not ev.empty:
                    by_rep = ev.groupby("entity_id")["duration_days"].sum().astype(int)
                    L.append(f"- {etype}: mean {by_rep.reindex(per_rep_absent.index, fill_value=0).mean():.2f}")
        else:
            L.append("- no rep-side absence rows")
    else:
        L.append("- no uncertainty_traces")
    L.append("")

    # Account visit frequency.
    L.append("**Account visit frequency:**")
    L.append("")
    acct_calls = productive.groupby("account_id").size()
    pct_called = 100 * len(acct_calls) / max(1, len(accts))
    L.append(f"- {len(acct_calls)} of {len(accts)} accounts called at least once "
             f"({pct_called:.1f}%)")
    L.append(f"- never-called accounts: {len(accts) - len(acct_calls)}")
    if len(acct_calls) > 0:
        L.append(f"- calls per called-account: mean {acct_calls.mean():.1f}, "
                 f"p50 {_pct(acct_calls, 50):.0f}, p90 {_pct(acct_calls, 90):.0f}, "
                 f"max {acct_calls.max()}")
        top10 = acct_calls.sort_values(ascending=False).head(max(1, len(acct_calls) // 10)).sum()
        L.append(f"- top 10% of called accounts receive {100 * top10 / acct_calls.sum():.1f}% of calls")
    L.append("")

    # Inter-call gaps within a day.
    L.append("**Inter-call gap within day (min):**")
    L.append("")
    productive = productive.sort_values(["rep_id", "_d", "_minute"])
    gap = productive.groupby(["rep_id", "_d"])["_minute"].diff().dropna()
    if not gap.empty:
        L.append(f"- mean {gap.mean():.0f}, p10 {_pct(gap, 10):.0f}, "
                 f"p50 {_pct(gap, 50):.0f}, p90 {_pct(gap, 90):.0f}")
        L.append(f"- gap distribution: <30 min: {100 * (gap < 30).mean():.1f}%, "
                 f"30-60 min: {100 * ((gap >= 30) & (gap < 60)).mean():.1f}%, "
                 f">60 min: {100 * (gap >= 60).mean():.1f}%")
    L.append("")
    return L


# ---- Pass 2.

def pass2_predictability(act_actual: pd.DataFrame, pop: pd.DataFrame,
                         horizon: int, sample_reps: int = 50) -> List[str]:
    L = ["### Pass 2: predictability\n"]

    productive = act_actual[act_actual["outcome"] != "no_show"].copy()
    productive["_d"] = pd.to_datetime(productive["date"]).dt.date
    productive["_dow"] = pd.to_datetime(productive["date"]).dt.weekday
    productive["_t"] = (pd.to_datetime(productive["start_time"], format="%H:%M").dt.hour * 60
                        + pd.to_datetime(productive["start_time"], format="%H:%M").dt.minute)
    productive["_hour"] = productive["_t"] // 60
    seg_to_idx = {"A": 0, "B": 1, "C": 2}
    productive["_seg"] = productive["segment_at_call"].map(seg_to_idx).fillna(1).astype(int)

    # Part A: progressive variance-explained on calls/(rep, day).
    L.append("**Variance of calls/(rep, day) with progressive features:**")
    L.append("")
    grouped = (productive.groupby(["rep_id", "_d"]).size()
               .reset_index(name="y").sort_values(["rep_id", "_d"]))
    grouped["_dow"] = pd.to_datetime(grouped["_d"]).dt.weekday
    if not grouped.empty:
        # Lag-1 calls (yesterday) and 7-day rolling mean per rep.
        grouped["_lag1"] = grouped.groupby("rep_id")["y"].shift(1).fillna(0)
        grouped["_roll7"] = (grouped.groupby("rep_id")["y"]
                             .transform(lambda s: s.rolling(7, min_periods=1).mean().shift(1))
                             .fillna(0))
        total_var = grouped["y"].var()
        global_mean = grouped["y"].mean()

        def _r2(pred: pd.Series) -> float:
            resid_var = ((grouped["y"] - pred) ** 2).mean()
            return 1.0 - resid_var / max(1e-9, total_var)

        rep_mean = grouped.groupby("rep_id")["y"].transform("mean")
        dow_mean = grouped.groupby("_dow")["y"].transform("mean")

        # Cumulative feature sets:
        # 1) rep only
        # 2) rep + dow
        # 3) rep + dow + lag-1
        # 4) rep + dow + lag-1 + rolling-7
        pred_rep = rep_mean
        pred_rep_dow = rep_mean + dow_mean - global_mean
        # Linear-regression style: fit lag-1 coefficient with residual after rep+dow.
        resid_after_rep_dow = grouped["y"] - pred_rep_dow
        lag_norm = grouped["_lag1"] - grouped["_lag1"].mean()
        beta_lag = (lag_norm * resid_after_rep_dow).sum() / max(1e-9, (lag_norm ** 2).sum())
        pred_rep_dow_lag = pred_rep_dow + beta_lag * lag_norm

        resid_after_lag = grouped["y"] - pred_rep_dow_lag
        roll_norm = grouped["_roll7"] - grouped["_roll7"].mean()
        beta_roll = (roll_norm * resid_after_lag).sum() / max(1e-9, (roll_norm ** 2).sum())
        pred_full = pred_rep_dow_lag + beta_roll * roll_norm

        L.append(f"- variance of y = calls/day: {total_var:.2f}")
        L.append(f"- R^2 rep only:                       {_r2(pred_rep):.3f}")
        L.append(f"- R^2 rep + dow:                      {_r2(pred_rep_dow):.3f}")
        L.append(f"- R^2 rep + dow + lag-1:              {_r2(pred_rep_dow_lag):.3f}")
        L.append(f"- R^2 rep + dow + lag-1 + roll-7:     {_r2(pred_full):.3f}")
    L.append("")

    # Part B: next-account top-1 accuracy with progressive event-level features.
    L.append("**Next-account top-1 accuracy with progressive features:**")
    L.append("(per-rep multinomial logistic, time-ordered 80/20 split, mean across reps)")
    L.append("")

    rep_ids = productive["rep_id"].unique()
    rng = np.random.default_rng(42)
    chosen = rng.choice(rep_ids, size=min(sample_reps, len(rep_ids)), replace=False)

    feature_sets = [
        ("F1: dow only",                       ["_dow"]),
        ("F2: + hour-of-day",                  ["_dow", "_hour"]),
        ("F3: + last-account",                 ["_dow", "_hour", "_last_acct"]),
        ("F4: + recency",                      ["_dow", "_hour", "_last_acct", "_recency"]),
        ("F5: + last-segment",                 ["_dow", "_hour", "_last_acct", "_recency", "_last_seg"]),
    ]
    accs = {label: [] for label, _ in feature_sets}
    n_used = 0

    for rid in chosen:
        sub = productive[productive["rep_id"] == rid].sort_values(["_d", "_t"]).copy()
        if len(sub) < 30:
            continue
        sub["_last_acct"] = sub["account_id"].shift(1).fillna(-1).astype(int)
        sub["_last_seg"] = sub["_seg"].shift(1).fillna(1).astype(int)
        # Recency: days since this rep last called this account; -1 first time.
        days = pd.to_datetime(sub["_d"]).values.astype("datetime64[D]")
        rec = np.zeros(len(sub), dtype=np.int64)
        last_seen: Dict[int, int] = {}
        for i, (acct, d) in enumerate(zip(sub["account_id"].astype(int).values,
                                           days)):
            if acct in last_seen:
                rec[i] = int((d - last_seen[acct]).astype(int))
            else:
                rec[i] = -1
            last_seen[acct] = d
        sub["_recency"] = rec

        le = LabelEncoder()
        y = le.fit_transform(sub["account_id"].astype(int).values)
        if len(set(y)) < 2:
            continue
        cut = max(1, int(len(sub) * 0.8))
        if cut < 5 or len(sub) - cut < 2:
            continue
        n_used += 1
        for label, feats in feature_sets:
            X = sub[feats].astype(float).values
            try:
                m = LogisticRegression(max_iter=200, solver="lbfgs")
                m.fit(X[:cut], y[:cut])
                accs[label].append((m.predict(X[cut:]) == y[cut:]).mean())
            except Exception:
                pass

    L.append(f"- reps used: {n_used}")
    if n_used > 0:
        for label, vals in accs.items():
            if vals:
                L.append(f"- {label}: mean top-1 = {np.mean(vals):.3f}")
            else:
                L.append(f"- {label}: all per-rep fits failed")
    else:
        L.append("- not enough per-rep history for any rep")
    L.append("")
    return L


# ---- Pass 3.

def pass3_disruption(act_all: pd.DataFrame, pop: pd.DataFrame,
                     accts: pd.DataFrame, u: pd.DataFrame,
                     horizon: int, start: date) -> List[str]:
    L = ["### Pass 3: disruption impact\n"]

    actual = act_all[act_all["scenario_id"] == "actual"]
    naive = act_all[act_all["scenario_id"] == "naive"]
    greedy = act_all[act_all["scenario_id"] == "greedy_upper"]

    # Failure rate by source on the actual scenario.
    L.append("**Actual-scenario outcomes:**")
    L.append("")
    if not actual.empty:
        out = actual["outcome"].value_counts(normalize=True)
        L.append(f"- completed {100 * out.get('completed', 0):.1f}%, "
                 f"abbreviated {100 * out.get('abbreviated', 0):.1f}%, "
                 f"no_show {100 * out.get('no_show', 0):.1f}%")
    L.append("")

    # On the naive plan, how many calls failed and why.
    L.append("**Naive-plan failures by cause** (naive ignores uncertainty; "
             "this measures realised disruption pressure):")
    L.append("")
    if not naive.empty:
        n_total = len(naive)
        n_show = (naive["outcome"] == "no_show").sum()
        L.append(f"- {n_total} naive calls planned, "
                 f"{n_show} ({100 * n_show / n_total:.1f}%) failed")

        # Attribute failure: account unavailable on that day?
        if not u.empty:
            acct_u = u[u["entity_type"] == "account"]
            acct_unavail_days = {}
            for _, r in acct_u.iterrows():
                a = int(r["entity_id"])
                s = (pd.to_datetime(r["event_start_date"]).date() - start).days
                d = int(r["duration_days"])
                for k in range(d):
                    acct_unavail_days.setdefault(a, set()).add(s + k)

            rep_u = u[u["entity_type"] == "rep"]
            rep_absent_days = {}
            for _, r in rep_u.iterrows():
                rid = int(r["entity_id"])
                s = (pd.to_datetime(r["event_start_date"]).date() - start).days
                d = int(r["duration_days"])
                for k in range(d):
                    rep_absent_days.setdefault(rid, set()).add(s + k)

            fail_acct = 0
            fail_rep = 0
            fail_both = 0
            fail_other = 0
            naive_fail = naive[naive["outcome"] == "no_show"]
            for _, c in naive_fail.iterrows():
                day_idx = (pd.to_datetime(c["date"]).date() - start).days
                a = int(c["account_id"])
                rid = int(c["rep_id"])
                acct_bad = day_idx in acct_unavail_days.get(a, set())
                rep_bad = day_idx in rep_absent_days.get(rid, set())
                if acct_bad and rep_bad:
                    fail_both += 1
                elif acct_bad:
                    fail_acct += 1
                elif rep_bad:
                    fail_rep += 1
                else:
                    fail_other += 1
            L.append(f"- account-unavailable only: {fail_acct} "
                     f"({100 * fail_acct / max(1, n_show):.1f}% of failures)")
            L.append(f"- rep-absent only: {fail_rep} "
                     f"({100 * fail_rep / max(1, n_show):.1f}% of failures)")
            L.append(f"- both: {fail_both} "
                     f"({100 * fail_both / max(1, n_show):.1f}% of failures)")
            L.append(f"- neither (unaccounted): {fail_other} "
                     f"({100 * fail_other / max(1, n_show):.1f}% of failures)")
    else:
        L.append("- no naive plan rows in this dataset")
    L.append("")

    # Sales gap: greedy vs actual vs naive call counts.
    L.append("**Plan size gap (productive calls only):**")
    L.append("")
    for tag, df in (("actual", actual), ("greedy_upper", greedy), ("naive", naive)):
        prod = df[df["outcome"] != "no_show"] if not df.empty else df
        L.append(f"- {tag}: {len(prod)} productive calls")
    if not actual.empty and not greedy.empty:
        prod_act = (actual["outcome"] != "no_show").sum()
        prod_gr = (greedy["outcome"] != "no_show").sum()
        if prod_gr > 0:
            L.append(f"- realised fraction of greedy ceiling: "
                     f"{prod_act / prod_gr:.3f}")
    L.append("")
    return L


# ---- Pass 4.

def pass4_eyeball(act_all: pd.DataFrame, pop: pd.DataFrame,
                  accts: pd.DataFrame, horizon: int) -> List[str]:
    L = ["### Pass 4: concrete examples\n"]

    actual = act_all[act_all["scenario_id"] == "actual"].copy()
    productive = actual[actual["outcome"] != "no_show"]

    # 5 random reps.
    L.append("**Five random reps:**")
    L.append("")
    rng = np.random.default_rng(42)
    rep_ids = sorted(productive["rep_id"].unique())
    if len(rep_ids) > 0:
        chosen = rng.choice(rep_ids, size=min(5, len(rep_ids)), replace=False)
        for rid in chosen:
            sub = productive[productive["rep_id"] == rid]
            n_days = sub["date"].nunique()
            n_calls = len(sub)
            n_acct = sub["account_id"].nunique()
            top = sub["account_id"].value_counts().head(3)
            mean_day = n_calls / max(1, n_days)
            rep_row = pop[pop["rep_id"] == int(rid)]
            rep_type = rep_row["type"].iloc[0] if not rep_row.empty else "?"
            L.append(f"- rep_id={rid} ({rep_type}): {n_calls} calls over {n_days} active days "
                     f"(mean {mean_day:.1f}/day), {n_acct} distinct accounts. "
                     f"Top accounts: " + ", ".join(f"{int(a)}×{int(c)}" for a, c in top.items()))
    L.append("")

    # 5 random accounts.
    L.append("**Five random called accounts:**")
    L.append("")
    acct_ids = sorted(productive["account_id"].unique())
    if len(acct_ids) > 0:
        chosen_a = rng.choice(acct_ids, size=min(5, len(acct_ids)), replace=False)
        for aid in chosen_a:
            sub = productive[productive["account_id"] == aid]
            seg = accts[accts["account_id"] == int(aid)]["initial_segment"]
            seg = seg.iloc[0] if not seg.empty else "?"
            n_calls = len(sub)
            n_reps = sub["rep_id"].nunique()
            span_days = sub["date"].nunique()
            L.append(f"- account_id={aid} (seg {seg}): {n_calls} calls from "
                     f"{n_reps} reps across {span_days} days")
    L.append("")

    # Reference plans, do they make sense.
    L.append("**Reference plans sanity:**")
    L.append("")
    greedy = act_all[act_all["scenario_id"] == "greedy_upper"]
    naive = act_all[act_all["scenario_id"] == "naive"]
    if not greedy.empty:
        # Greedy should heavily favour high-value segments.
        seg_share_act = (actual.merge(accts[["account_id", "initial_segment"]],
                                       on="account_id")
                         .groupby("initial_segment").size())
        seg_share_gr = (greedy.merge(accts[["account_id", "initial_segment"]],
                                       on="account_id")
                        .groupby("initial_segment").size())
        L.append("- segment share comparison:")
        for s in ("A", "B", "C"):
            a = int(seg_share_act.get(s, 0))
            g = int(seg_share_gr.get(s, 0))
            L.append(f"  - segment {s}: actual {a}, greedy {g}")
    if not naive.empty:
        # Naive should call high-cadence segments more often.
        seg_share_nv = (naive.merge(accts[["account_id", "initial_segment"]],
                                     on="account_id")
                        .groupby("initial_segment").size())
        L.append("- naive segment counts: " + ", ".join(
            f"{s}={int(seg_share_nv.get(s, 0))}" for s in ("A", "B", "C")))
    L.append("")
    return L


def _analyze(label: str, d: str) -> List[str]:
    L = [f"## {label}\n", f"Path: `{d}`\n"]
    t0 = time.perf_counter()
    try:
        data = _load(d)
    except Exception as e:
        L.append(f"FAILED to load: {e}\n")
        return L

    cfg = data["config"]
    pop = data["population"]
    accts = data["accounts"]
    act = data["activity_log"]
    u = data["uncertainty_traces"]
    horizon = int(cfg.get("horizon_days", 0))
    start = date.fromisoformat(str(cfg.get("start_date", "2024-01-01")))

    if act.empty:
        L.append("No activity rows. Skipping.\n")
        return L

    act_actual = act[act["scenario_id"] == "actual"]

    L.append(f"Loaded {len(act):,} activity rows, {len(pop):,} reps, "
             f"{len(accts):,} accounts, horizon {horizon} days.\n")

    L.extend(pass1_distributions(act_actual, pop, u, accts, horizon))
    L.extend(pass2_predictability(act_actual, pop, horizon))
    L.extend(pass3_disruption(act, pop, accts, u, horizon, start))
    L.extend(pass4_eyeball(act, pop, accts, horizon))

    L.append(f"\n_Section runtime: {time.perf_counter() - t0:.1f} s_\n")
    return L


def main():
    out_path = ROOT / "forge_ds/results/data_analysis.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Dataset characterization\n",
             "Generated by `forge_ds/analyze_data.py`.\n",
             "Four passes per dataset: basic distributions, predictability, "
             "disruption impact, concrete examples.\n"]
    for label, d in DATASETS:
        if not os.path.isdir(d):
            lines.append(f"## {label}\n\n_Skipping: {d} does not exist._\n\n")
            continue
        print(f"analyzing {label} ...")
        try:
            lines.extend(_analyze(label, d))
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"FAILED on {label}: {e}")
            lines.append(f"## {label}\n\n_Analysis failed: {e}_\n\n```\n{tb}\n```\n\n")
        lines.append("\n---\n")
    out_path.write_text("\n".join(lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
