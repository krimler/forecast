"""Account universe, forces, reps, panels.

Builds the static structure of one simulation.
Two outputs:
    World      accounts, specialties, segments, brand eligibility, force layout
    Population reps with their type, force, and panel
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

from config import (
    Config, SEGMENT_DEFAULTS, REP_TYPES, SEG_NAMES,
    rng, seed_for_world, seed_for_force, seed_for_population, seed_for_rep,
)


REP_TYPE_NAMES = list(REP_TYPES.keys())
REP_TYPE_PROBS = np.array([REP_TYPES[t]["prob"] for t in REP_TYPE_NAMES])

# Priority regimes (Section 5.2). Sampled uniformly at random per force.
PRIORITY_REGIMES: Dict[int, List[Tuple[str, List[float]]]] = {
    1: [("single",   [1.0])],
    2: [("balanced", [0.5, 0.5]),
        ("moderate", [0.7, 0.3]),
        ("heavy",    [0.8, 0.2])],
    3: [("balanced", [0.34, 0.33, 0.33]),
        ("moderate", [0.5, 0.3, 0.2]),
        ("heavy",    [0.6, 0.3, 0.1])],
}


@dataclass
class World:
    account_specialty: np.ndarray          # int32 [N]
    account_segment_initial: np.ndarray    # int8  [N], 0=A 1=B 2=C
    brand_target_specialties: List[set]    # length num_brands
    brand_force: np.ndarray                # int32 [num_brands]
    force_brands: List[List[int]]          # brand ids per force
    eligibility: np.ndarray                # bool [N, num_brands]

    def eligible_accounts_for_brand(self, b: int) -> np.ndarray:
        return np.where(self.eligibility[:, b])[0]


@dataclass
class ForceConfig:
    force_id: int
    brands: List[int]
    priorities: List[float]
    regime: str

    def priority_of(self, brand_id: int) -> float:
        for b, p in zip(self.brands, self.priorities):
            if b == brand_id:
                return p
        return 0.0


@dataclass
class Rep:
    rep_id: int
    rep_type: str
    force_id: int
    panel: np.ndarray
    hire_date_idx: int = 0
    departure_date_idx: int = -1     # -1 means still active at horizon end
    replacement_of: int = -1         # rep_id this rep replaced, or -1

    @property
    def panel_size(self) -> int:
        return int(self.panel.shape[0])


@dataclass
class Population:
    reps: List[Rep]
    forces: List[ForceConfig]


# ---- World construction.

def build_world(cfg: Config) -> World:
    r = rng(seed_for_world(cfg.seed))
    N, K = cfg.num_accounts_total, cfg.num_specialties

    specialty = r.integers(0, K, size=N, dtype=np.int32)

    seg_probs = np.array([SEGMENT_DEFAULTS[s]["prob"] for s in SEG_NAMES])
    segment = r.choice(len(SEG_NAMES), size=N, p=seg_probs).astype(np.int8)

    # Brands are partitioned across forces (Section 4.3). When the total
    # doesn't divide evenly, the remainder goes to the first few forces.
    base, rem = divmod(cfg.num_brands_total, cfg.num_forces)
    force_brands: List[List[int]] = []
    cursor = 0
    for f in range(cfg.num_forces):
        size = base + (1 if f < rem else 0)
        force_brands.append(list(range(cursor, cursor + size)))
        cursor += size

    brand_force = np.zeros(cfg.num_brands_total, dtype=np.int32)
    for f, bs in enumerate(force_brands):
        for b in bs:
            brand_force[b] = f

    # Each brand targets k_b ~ Uniform{2,3,4} specialties.
    brand_targets: List[set] = []
    for _ in range(cfg.num_brands_total):
        k_b = min(int(r.integers(2, 5)), K)
        brand_targets.append(set(int(x) for x in r.choice(K, size=k_b, replace=False)))

    # Eligibility is fully determined by specialty.
    eligibility = np.zeros((N, cfg.num_brands_total), dtype=bool)
    for b, targets in enumerate(brand_targets):
        if not targets:
            continue
        eligibility[:, b] = np.isin(specialty, np.array(sorted(targets), dtype=np.int32))

    return World(
        account_specialty=specialty,
        account_segment_initial=segment,
        brand_target_specialties=brand_targets,
        brand_force=brand_force,
        force_brands=force_brands,
        eligibility=eligibility,
    )


# ---- Forces.

def build_forces(cfg: Config, world: World) -> List[ForceConfig]:
    forces: List[ForceConfig] = []
    for f in range(cfg.num_forces):
        r = rng(seed_for_force(cfg.seed, f))
        allocated = world.force_brands[f]
        max_bag = len(allocated)
        # Sampled bag size can exceed the brands the force was allocated, so
        # we clip down (Section 5.1 implementer note).
        size = min(int(r.choice([1, 2, 3], p=np.array(cfg.bag_size_probs))), max_bag)
        bag = sorted(int(b) for b in r.choice(allocated, size=size, replace=False))
        regimes = PRIORITY_REGIMES[size]
        regime_name, weights = regimes[int(r.integers(0, len(regimes)))]
        forces.append(ForceConfig(
            force_id=f, brands=bag, priorities=list(weights), regime=regime_name,
        ))
    return forces


# ---- Panels.

def _build_panel(cfg: Config, world: World, force: ForceConfig,
                 rep_type: str, r: np.random.Generator) -> np.ndarray:
    """Segment-balanced, brand-eligibility-aware panel for one rep (Section 6.3)."""
    target_size = REP_TYPES[rep_type]["panel"]
    chosen: set = set()
    seg_probs = np.array([SEGMENT_DEFAULTS[s]["prob"] for s in SEG_NAMES])

    # First pass: guarantee min_eligible_per_brand for every bag brand,
    # stratified by segment so we don't all-A or all-C.
    for b in force.brands:
        eligible = world.eligible_accounts_for_brand(b)
        if chosen:
            eligible = eligible[~np.isin(eligible, np.fromiter(chosen, dtype=np.int64))]
        if eligible.size == 0:
            continue

        per_seg = (cfg.min_eligible_per_brand * seg_probs).astype(int)
        # Push the rounding leftover to whichever segment has the largest
        # fractional remainder.
        deficit = cfg.min_eligible_per_brand - per_seg.sum()
        if deficit > 0:
            frac = cfg.min_eligible_per_brand * seg_probs - per_seg
            for i in np.argsort(-frac)[:deficit]:
                per_seg[i] += 1

        seg_of_eligible = world.account_segment_initial[eligible]
        for s_idx, n_need in enumerate(per_seg):
            if n_need <= 0:
                continue
            pool = eligible[seg_of_eligible == s_idx]
            if pool.size == 0:
                pool = eligible
            picks = r.choice(pool, size=min(n_need, pool.size), replace=False)
            chosen.update(int(p) for p in picks)

    # Second pass: fill the rest with segment-balanced sampling from the
    # whole universe.
    remaining = target_size - len(chosen)
    if remaining > 0:
        fill_per_seg = (remaining * seg_probs).astype(int)
        deficit = remaining - fill_per_seg.sum()
        if deficit > 0:
            frac = remaining * seg_probs - fill_per_seg
            for i in np.argsort(-frac)[:deficit]:
                fill_per_seg[i] += 1

        for s_idx, n_need in enumerate(fill_per_seg):
            if n_need <= 0:
                continue
            seg_pool = np.where(world.account_segment_initial == s_idx)[0]
            if chosen:
                seg_pool = seg_pool[~np.isin(seg_pool, np.fromiter(chosen, dtype=np.int64))]
            if seg_pool.size == 0:
                continue
            picks = r.choice(seg_pool, size=min(n_need, seg_pool.size), replace=False)
            chosen.update(int(p) for p in picks)

    panel = np.fromiter(chosen, dtype=np.int64)
    panel.sort()
    return panel


def build_population(cfg: Config, world: World) -> Population:
    forces = build_forces(cfg, world)
    pop_r = rng(seed_for_population(cfg.seed))
    rep_types = pop_r.choice(REP_TYPE_NAMES, size=cfg.num_reps, p=REP_TYPE_PROBS)
    force_ids = pop_r.integers(0, cfg.num_forces, size=cfg.num_reps)

    reps: List[Rep] = []
    for rid in range(cfg.num_reps):
        r = rng(seed_for_rep(cfg.seed, rid))
        force = forces[int(force_ids[rid])]
        rt = str(rep_types[rid])
        reps.append(Rep(
            rep_id=rid, rep_type=rt, force_id=int(force_ids[rid]),
            panel=_build_panel(cfg, world, force, rt, r),
        ))
    return Population(reps=reps, forces=forces)
