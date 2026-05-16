"""Experimental matrix definition (spec2 Section 11.3)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List


# Uncertainty intensity levels (spec2 §11.1).
UNCERTAINTY_LEVELS: Dict[str, Dict[str, float]] = {
    "none": {"p_account_unavail": 0.00, "sick_days_per_year_mean": 0,
             "vacation_days_per_year_mean": 0, "personal_days_per_year_mean": 0,
             "conference_days_per_year_mean": 0, "p_churn_annual": 0.0},
    "low":  {"p_account_unavail": 0.05, "sick_days_per_year_mean": 3,
             "vacation_days_per_year_mean": 8, "personal_days_per_year_mean": 1.5,
             "conference_days_per_year_mean": 2, "p_churn_annual": 0.05},
    "default": {"p_account_unavail": 0.10, "sick_days_per_year_mean": 6,
                "vacation_days_per_year_mean": 15, "personal_days_per_year_mean": 3,
                "conference_days_per_year_mean": 4, "p_churn_annual": 0.10},
    "high": {"p_account_unavail": 0.30, "sick_days_per_year_mean": 12,
             "vacation_days_per_year_mean": 25, "personal_days_per_year_mean": 5,
             "conference_days_per_year_mean": 8, "p_churn_annual": 0.20},
}

PRIORITY_REGIMES = ("balanced", "moderate", "heavy")
ALGORITHMS = ("markov", "prophet", "neural_tpp", "beam_tpp", "constrained_tpp")
DATASETS_PUBLIC = ("foursquare_nyc", "foursquare_tokyo")


@dataclass
class Cell:
    """One unique configuration we evaluate."""
    cell_id: str
    algorithm: str
    dataset: str
    uncertainty_level: str
    priority_regime: str
    seed: int

    def as_dict(self) -> Dict:
        return asdict(self)


def build_smoke_matrix(seeds: List[int] = (42,),
                       algorithms: List[str] = None) -> List[Cell]:
    """A small matrix that lets you run end-to-end on output_smoke."""
    if algorithms is None:
        algorithms = ["markov", "prophet"]
    cells: List[Cell] = []
    for algo in algorithms:
        for seed in seeds:
            cells.append(Cell(
                cell_id=f"smoke::{algo}::syn::default::balanced::{seed}",
                algorithm=algo, dataset="synthetic_smoke",
                uncertainty_level="default", priority_regime="balanced",
                seed=seed,
            ))
    return cells


def build_full_matrix(seeds: List[int] = (42, 43, 44, 45, 46)) -> List[Cell]:
    """The full deduplicated F1-F5 matrix from spec2 §11.3."""
    cells: List[Cell] = []

    # F1 (data characterization) is one synthetic-default cell.
    cells.append(Cell(
        cell_id="F1::syn::default::balanced::42",
        algorithm="markov", dataset="synthetic_default",
        uncertainty_level="default", priority_regime="balanced", seed=42,
    ))

    # F2 (source decomposition): neural_tpp x 4 uncertainty levels x 5 seeds.
    for level in UNCERTAINTY_LEVELS:
        for s in seeds:
            cells.append(Cell(
                cell_id=f"F2::neural_tpp::syn::{level}::balanced::{s}",
                algorithm="neural_tpp", dataset="synthetic_default",
                uncertainty_level=level, priority_regime="balanced", seed=s,
            ))

    # F3 (Pareto): 5 algos x 3 priority regimes x default uncertainty x 5 seeds.
    for algo in ALGORITHMS:
        for reg in PRIORITY_REGIMES:
            for s in seeds:
                cells.append(Cell(
                    cell_id=f"F3::{algo}::syn::default::{reg}::{s}",
                    algorithm=algo, dataset="synthetic_default",
                    uncertainty_level="default", priority_regime=reg, seed=s,
                ))

    # F4 cells overlap with F3 default-priority cells (spec2 §11.3 B10).

    # F5 (external validity): 5 algos x 2 cities x default x 5 seeds.
    for algo in ALGORITHMS:
        for city in DATASETS_PUBLIC:
            for s in seeds:
                cells.append(Cell(
                    cell_id=f"F5::{algo}::{city}::default::balanced::{s}",
                    algorithm=algo, dataset=city,
                    uncertainty_level="default", priority_regime="balanced", seed=s,
                ))
    return cells
