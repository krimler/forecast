"""Shared helpers for figure CSV writers (spec2 §12)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List
import pandas as pd


def write_csv(path: str, rows: List[Dict]) -> str:
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def load_results(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
