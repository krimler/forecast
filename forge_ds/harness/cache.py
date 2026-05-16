"""Tiny on-disk cache for trained algorithms.

Cache key = SHA-256 of (dataset_hash, algorithm_name, seed). Values are
pickled. Same trained Neural TPP weights get reused across F2/F3/F4 cells
that share dataset + seed.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path


def make_key(dataset_hash: str, algorithm: str, seed: int) -> str:
    raw = f"{dataset_hash}::{algorithm}::{seed}".encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def cache_path(root: str, key: str) -> str:
    Path(root).mkdir(parents=True, exist_ok=True)
    return os.path.join(root, f"{key}.pkl")


def get(root: str, key: str):
    path = cache_path(root, key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def put(root: str, key: str, obj) -> None:
    path = cache_path(root, key)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)
