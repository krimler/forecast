"""Configuration, parameter tables, RNG seeding, and feasibility check.

All defaults follow spec1 Sections 3-8. The Config dataclass is the single
source of truth at run time. Tables (SEGMENT_DEFAULTS, REP_TYPES, etc.)
hold the constants the spec calls out separately because they live in
tables rather than scalar fields.

RNG seeding is per-entity (Section 2.4). Every (seed, rep_id) draws from
its own stream so the daily loop can run reps in parallel without
breaking determinism.
"""
import hashlib
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple


SEG_NAMES = ("A", "B", "C")


SEGMENT_DEFAULTS = {
    "A": {"prob": 0.15, "value": 5.0, "annual_target": 24, "lift_a": 1.0, "lift_b": 0.12},
    "B": {"prob": 0.35, "value": 2.0, "annual_target": 12, "lift_a": 0.8, "lift_b": 0.18},
    "C": {"prob": 0.50, "value": 1.0, "annual_target": 6,  "lift_a": 0.5, "lift_b": 0.30},
}

SEGMENT_TRANSITION = {
    "A": {"A": 0.92, "B": 0.07, "C": 0.01},
    "B": {"A": 0.04, "B": 0.90, "C": 0.06},
    "C": {"A": 0.01, "B": 0.06, "C": 0.93},
}

REP_TYPES = {
    "specialty":   {"prob": 0.20, "mu": 5.0, "sigma": 1.0, "panel": 200},
    "mid-market":  {"prob": 0.50, "mu": 7.0, "sigma": 1.0, "panel": 400},
    "high-volume": {"prob": 0.30, "mu": 8.0, "sigma": 1.0, "panel": 600},
}

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

# Index by datetime.weekday(): 0=Mon..6=Sun. Weekend entries are 0, so the
# daily loop naturally drops weekends without a separate check.
DOW_MULTIPLIER = [0.8, 1.0, 1.0, 1.0, 0.7, 0.0, 0.0]

OUTCOME_PROBS = {"completed": 0.85, "abbreviated": 0.10, "no_show": 0.05}


@dataclass
class Config:
    seed: int = 42
    horizon_days: int = 365
    warmup_days: int = 90
    num_reps: int = 1000
    num_brands_total: int = 6
    num_accounts_total: int = 100_000
    num_forces: int = 3
    num_specialties: int = 10
    start_date: str = "2024-01-01"
    min_eligible_per_brand: int = 30

    bag_size_probs: Tuple[float, float, float] = (0.2, 0.5, 0.3)

    # Account availability (Section 7.1).
    p_account_unavail: float = 0.10
    account_unavail_persistence: float = 0.5
    account_notice_days: int = 1

    # Rep absences (Section 7.2).
    sick_days_per_year_mean: float = 6.0
    sick_days_per_year_std: float = 2.0
    sick_winter_multiplier: float = 1.5
    sick_autocorr: float = 0.3
    sick_notice_days: int = 0

    personal_days_per_year_mean: float = 3.0
    personal_friday_monday_mult: float = 1.5
    personal_block_lengths: Dict[int, float] = field(
        default_factory=lambda: {1: 0.7, 2: 0.3})
    personal_notice_days_range: Tuple[int, int] = (1, 7)

    vacation_days_per_year_mean: float = 15.0
    vacation_chunk_distribution: Dict[int, float] = field(
        default_factory=lambda: {5: 0.40, 10: 0.40, 15: 0.15, 20: 0.05})
    vacation_summer_winter_mult: float = 2.0
    vacation_notice_days_range: Tuple[int, int] = (14, 90)

    conference_days_per_year_mean: float = 4.0
    conference_block_lengths: Dict[int, float] = field(
        default_factory=lambda: {2: 0.5, 3: 0.3, 5: 0.2})
    conference_notice_days_range: Tuple[int, int] = (30, 180)

    p_churn_annual: float = 0.10
    churn_notice_days: int = 14

    # Time of day (Section 8.5).
    day_start_minute: int = 9 * 60
    day_end_minute: int = 18 * 60
    inter_call_gap_min: int = 15

    softmax_temperature: float = 1.0
    n_workers: int = 1
    output_dir: str = "output"

    def to_json(self) -> str:
        d = asdict(self)
        # JSON keys must be strings, so int-keyed dicts get stringified.
        for k in ("personal_block_lengths", "vacation_chunk_distribution",
                  "conference_block_lengths"):
            d[k] = {str(kk): vv for kk, vv in getattr(self, k).items()}
        return json.dumps(d, sort_keys=True, indent=2)


# ---- RNG seeding.
# blake2b gives us a hash that is stable across Python processes. The
# built-in hash() is not, which would break multiprocessing reproducibility.

_MASK32 = (1 << 32) - 1
_MASK64 = (1 << 64) - 1


def _stable_hash(*parts) -> int:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(repr(p).encode())
        h.update(b"|")
    return int.from_bytes(h.digest(), "big") & _MASK64


def seed_for_rep(seed, rep_id):        return _stable_hash(seed, "rep", rep_id) & _MASK32
def seed_for_account(seed, acct_id):   return _stable_hash(seed, "account", acct_id) & _MASK32
def seed_for_world(seed):              return _stable_hash(seed, "world") & _MASK32
def seed_for_population(seed):         return _stable_hash(seed, "population") & _MASK32
def seed_for_force(seed, fid):         return _stable_hash(seed, "force", fid) & _MASK32
def seed_for_replanner(seed, idx):     return _stable_hash(seed, "replan", idx) & _MASK32


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---- Feasibility check (Section 12). Raises before anything expensive runs.

class FeasibilityError(ValueError):
    pass


def check_feasibility(cfg: Config) -> List[str]:
    """Return informational warnings. Raise FeasibilityError on hard issues."""
    warnings: List[str] = []
    errors: List[str] = []

    max_panel = max(rt["panel"] for rt in REP_TYPES.values())

    # Accounts are shared across rep panels (spec allows this), so the
    # binding sufficiency constraint is "universe holds one panel".
    if cfg.num_accounts_total < max_panel:
        errors.append(
            f"num_accounts_total ({cfg.num_accounts_total}) < max panel {max_panel}.")

    # A brand that targets only 2 specialties still needs at least
    # min_eligible_per_brand accounts in the union of those specialties.
    per_specialty = cfg.num_accounts_total / max(1, cfg.num_specialties)
    if 2 * per_specialty < cfg.min_eligible_per_brand:
        errors.append(
            f"Worst-case brand has ~{2*per_specialty:.0f} eligible accounts; "
            f"need at least {cfg.min_eligible_per_brand}.")

    # Demand vs capacity is informational. The spec accepts under-coverage.
    for tname, rt in REP_TYPES.items():
        demand = rt["panel"] * (0.15 * 24 + 0.35 * 12 + 0.5 * 6)
        capacity = rt["mu"] * 252 * 0.9
        if demand > capacity * 1.5:
            warnings.append(
                f"Rep type {tname}: demand {demand:.0f} > 1.5*capacity "
                f"{capacity:.0f}, expect under-coverage.")

    absence_total = (cfg.sick_days_per_year_mean + cfg.personal_days_per_year_mean
                     + cfg.vacation_days_per_year_mean + cfg.conference_days_per_year_mean)
    if absence_total >= 252:
        errors.append(f"Total absence days {absence_total} >= 252 working days/yr.")

    churn_frac = cfg.p_churn_annual * cfg.horizon_days / 365.0
    if churn_frac > 1.0:
        errors.append(f"Churn fraction over horizon = {churn_frac:.2f} > 1.")

    if cfg.num_brands_total < cfg.num_forces:
        errors.append(
            f"num_brands_total ({cfg.num_brands_total}) < num_forces ({cfg.num_forces}).")

    if errors:
        raise FeasibilityError("\n".join(errors))
    return warnings
