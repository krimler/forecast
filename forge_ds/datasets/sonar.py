"""SONAR (Sensor-based Nursing Activity Recognition) adapter.

The dataset (Lübbe et al, Nature Sci Data 2023) is 14 caregivers wearing
inertial sensors while doing labelled nursing tasks. Per spec2 §10.3 we
map caregivers to reps and activity types to accounts. Each recording
file in SONAR_ML is one labelled activity instance.

Because the sensor stream itself is irrelevant for the 4-pass data
analysis, we stream only the metadata of each file from the Zenodo zip
via HTTP range requests (around 500 KB total transfer instead of the
6 GB full download).

Run:
    python forge_ds/datasets/sonar.py --out public_dataset/sonar/processed
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import remotezip


ZENODO_ML_URL = "https://zenodo.org/records/7881952/files/SONAR_ML.zip?download=1"

# Map nursing activity strings to A/B/C segments by clinical complexity.
# A = high-complexity / high-skill (medication, vital signs, mobility).
# B = routine clinical care (hygiene, feeding, dressing).
# C = ancillary / administrative (documentation, walking around).
ACTIVITY_TO_SEGMENT = {
    "medication": "A",
    "vital_signs": "A",
    "blood_pressure": "A",
    "blood_sugar": "A",
    "mobility_assistance": "A",
    "transfer": "A",
    "lifting": "A",
    "patient_transfer": "A",
    "hygiene": "B",
    "personal_hygiene": "B",
    "wash": "B",
    "feeding": "B",
    "dressing": "B",
    "make_bed": "B",
    "morning_care": "B",
    "documentation": "C",
    "walk": "C",
    "walking": "C",
    "leisure": "C",
    "break": "C",
    "talk": "C",
    "talking": "C",
}


def _segment_for(activity: str) -> str:
    a = activity.lower().strip()
    if a in ACTIVITY_TO_SEGMENT:
        return ACTIVITY_TO_SEGMENT[a]
    # Heuristic fallback by keyword.
    for k, v in ACTIVITY_TO_SEGMENT.items():
        if k in a or a in k:
            return v
    return "B"


_NAME_RE = re.compile(r"(?:^|/)(\d+)_sub(\d+)\.csv$")


def _parse_name(path: str) -> Tuple[int, int]:
    m = _NAME_RE.search(path)
    if not m:
        raise ValueError(f"unexpected filename {path}")
    return int(m.group(1)), int(m.group(2))


def stream_metadata() -> List[Dict]:
    """Read the first data row of every SONAR_ML CSV via range requests.

    Returns one dict per recording: {recording_idx, subject, activity,
    n_rows_estimate, file_size}.
    """
    rows: List[Dict] = []
    with remotezip.RemoteZip(ZENODO_ML_URL) as z:
        infos = [i for i in z.infolist()
                 if i.filename.startswith("SONAR_ML/")
                 and i.filename.endswith(".csv")
                 and "__MACOSX" not in i.filename]
        infos.sort(key=lambda i: i.filename)
        print(f"  {len(infos)} recording files in SONAR_ML")
        for k, info in enumerate(infos):
            try:
                rec_idx, subject = _parse_name(info.filename)
            except ValueError:
                continue
            # Stream a small head: header + ~5 data rows.
            with z.open(info) as f:
                head = f.read(4096).decode("utf-8", errors="ignore")
            lines = head.splitlines()
            if len(lines) < 2:
                continue
            header_cols = lines[0].split(",")
            if "activity" not in header_cols:
                continue
            act_idx = header_cols.index("activity")
            data_row = lines[1].split(",")
            if len(data_row) <= act_idx:
                continue
            activity = data_row[act_idx].strip()
            # Estimate row count: file_size / first-data-row bytes.
            row_bytes = len(lines[1]) + 1
            n_rows_est = max(1, info.file_size // max(1, row_bytes))
            rows.append({
                "recording_idx": rec_idx,
                "subject_id": subject,
                "activity": activity,
                "n_rows_estimate": int(n_rows_est),
                "file_size": int(info.file_size),
            })
            if (k + 1) % 25 == 0:
                print(f"    streamed {k+1}/{len(infos)}")
    rows.sort(key=lambda r: (r["subject_id"], r["recording_idx"]))
    return rows


def emit_spec1_schema(rows: List[Dict], out_dir: str,
                      sample_hz: float = 60.0) -> None:
    """Lay the SONAR recordings out in the spec1 schema.

    rep_id            <- subject_id (renumbered 0..N-1)
    account_id        <- activity name (renumbered 0..M-1)
    date              <- one synthetic day per recording, in order
    start_time        <- 09:00 + 30 minutes between events
    planned_duration  <- n_rows_estimate / sample_hz / 60 (minutes)
    segment           <- ACTIVITY_TO_SEGMENT (or 'B' fallback)
    brand             <- single brand 0 per spec2 §10.3
    """
    os.makedirs(out_dir, exist_ok=True)

    subjects = sorted({r["subject_id"] for r in rows})
    rep_of_subject = {s: i for i, s in enumerate(subjects)}
    activities = sorted({r["activity"] for r in rows})
    acct_of_activity = {a: i for i, a in enumerate(activities)}
    activity_to_seg = {a: _segment_for(a) for a in activities}

    # Map recordings to dates. Each subject gets contiguous days with one
    # recording per "day". This is synthetic but preserves the per-subject
    # ordering, which is what the 4-pass analysis cares about.
    start_date = date(2024, 1, 1)
    horizon = 1 + max(
        sum(1 for r in rows if r["subject_id"] == s) for s in subjects
    )

    # config.json
    config = {
        "seed": 42, "horizon_days": int(horizon), "warmup_days": 0,
        "num_reps": len(subjects), "num_accounts_total": len(activities),
        "num_brands_total": 1, "num_forces": 1,
        "num_specialties": 1, "start_date": start_date.isoformat(),
        "min_eligible_per_brand": 0,
        "source": "SONAR_ML (Zenodo 7881952)",
        "note": ("Activity events from inertial sensor recordings. "
                 "Accounts = activity types, segments mapped from "
                 "ACTIVITY_TO_SEGMENT in forge_ds/datasets/sonar.py."),
    }
    Path(out_dir, "config.json").write_text(json.dumps(config, indent=2))
    Path(out_dir, "run_id.txt").write_text(
        hashlib.sha256(json.dumps(config).encode()).hexdigest()[:16] + "\n")

    # population.csv
    with open(Path(out_dir, "population.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rep_id", "type", "force_id", "panel_size",
                    "hire_date", "departure_date"])
        for s in subjects:
            n = sum(1 for r in rows if r["subject_id"] == s)
            rep_type = ("specialty" if n < 12 else
                        ("mid-market" if n < 25 else "high-volume"))
            w.writerow([rep_of_subject[s], rep_type, 0, n,
                        start_date.isoformat(), ""])

    # accounts.csv (one row per activity type)
    with open(Path(out_dir, "accounts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["account_id", "specialty_id", "initial_segment",
                    "eligible_brands"])
        for a in activities:
            w.writerow([acct_of_activity[a], 0,
                        activity_to_seg.get(a, "B"), "0"])

    # panels.csv (every subject has every activity in their panel)
    with open(Path(out_dir, "panels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rep_id", "account_id", "assignment_start_date",
                    "assignment_end_date"])
        for s in subjects:
            for a in activities:
                w.writerow([rep_of_subject[s], acct_of_activity[a],
                            start_date.isoformat(), ""])

    # segment_history.csv (initial only)
    with open(Path(out_dir, "segment_history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["account_id", "effective_date", "old_segment", "new_segment"])
        for a in activities:
            w.writerow([acct_of_activity[a], start_date.isoformat(), "",
                        activity_to_seg.get(a, "B")])

    # activity_log.csv (one row per recording, scenario=actual)
    with open(Path(out_dir, "activity_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "date", "rep_id", "start_time",
                    "planned_duration_min", "actual_duration_min",
                    "account_id", "segment_at_call", "brand_id",
                    "brand_priority", "outcome", "scenario_id"])
        eid = 0
        per_subject_day: Dict[int, int] = {s: 0 for s in subjects}
        for r in rows:
            s = r["subject_id"]
            day_idx = per_subject_day[s]
            d = start_date + timedelta(days=day_idx)
            per_subject_day[s] += 1
            # Duration: sensor sample count / Hz -> minutes, capped to 120.
            dur = max(1, min(120, int(r["n_rows_estimate"] / sample_hz / 60)))
            seg = activity_to_seg.get(r["activity"], "B")
            w.writerow([eid, d.isoformat(), rep_of_subject[s],
                        "09:00", dur, dur,
                        acct_of_activity[r["activity"]], seg, 0,
                        "1.0000", "completed", "actual"])
            eid += 1

    # uncertainty_traces.csv (empty in source data)
    with open(Path(out_dir, "uncertainty_traces.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trace_id", "event_start_date", "entity_type",
                    "entity_id", "event_type", "notice_days",
                    "duration_days", "scenario_id"])

    # validation_stats.csv (one placeholder row)
    with open(Path(out_dir, "validation_stats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric_name", "scope", "value"])
        w.writerow(["source", "sonar_ml", "1"])

    print(f"  wrote {out_dir}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="public_dataset/sonar/processed")
    p.add_argument("--cache", default="public_dataset/sonar/raw/metadata.json",
                   help="cache the streamed metadata so re-runs skip the network")
    args = p.parse_args()

    t0 = time.perf_counter()
    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        print(f"loading cached metadata from {cache_path}")
        rows = json.loads(cache_path.read_text())
    else:
        print("streaming SONAR_ML metadata from Zenodo...")
        rows = stream_metadata()
        cache_path.write_text(json.dumps(rows))
        print(f"  cached metadata to {cache_path}")

    print(f"  {len(rows)} recordings across "
          f"{len({r['subject_id'] for r in rows})} subjects")
    activities = sorted({r["activity"] for r in rows})
    print(f"  {len(activities)} distinct activities: " + ", ".join(activities[:10])
          + ("..." if len(activities) > 10 else ""))

    print("\nwriting spec1-schema CSVs...")
    emit_spec1_schema(rows, args.out)
    print(f"\ndone in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
