"""Deterministic label splits and candidate-pass schedules (manuscript Section 5.3)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np


def class_balanced_split(labels: Sequence[int], fraction: float, seed: int) -> np.ndarray:
    """Class-balanced random subset containing ``round(fraction * N)`` indices."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1]")
    rng = np.random.default_rng(seed)
    total_target = int(round(fraction * len(labels)))
    classes = np.unique(labels)
    base = total_target // len(classes)
    remainder = total_target % len(classes)
    order = rng.permutation(len(classes))
    selected: list[np.ndarray] = []
    for rank, ci in enumerate(order):
        cls = classes[ci]
        idx = np.flatnonzero(labels == cls)
        n = min(len(idx), base + (1 if rank < remainder else 0))
        selected.append(rng.choice(idx, size=n, replace=False))
    out = np.sort(np.concatenate(selected))
    if len(out) < total_target:
        missing = total_target - len(out)
        remaining = np.setdiff1d(np.arange(len(labels)), out, assume_unique=True)
        out = np.sort(np.concatenate([out, rng.choice(remaining, size=missing, replace=False)]))
    return out


def stratified_subset(indices: np.ndarray, labels: Sequence[int], fraction: float, seed: int) -> np.ndarray:
    """Stratified subset of ``indices`` (used for the controller-validation carve-out)."""
    indices = np.asarray(indices)
    labels = np.asarray(labels)
    if fraction <= 0:
        return np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    y = labels[indices]
    target = int(round(fraction * len(indices)))
    classes = np.unique(y)
    per = max(1, target // len(classes))
    picks = []
    for c in classes:
        pool = indices[y == c]
        picks.append(rng.choice(pool, size=min(per, len(pool)), replace=False))
    out = np.concatenate(picks)
    if len(out) > target:
        out = rng.choice(out, size=target, replace=False)
    elif len(out) < target:
        remaining = np.setdiff1d(indices, out, assume_unique=True)
        out = np.concatenate([out, rng.choice(remaining, size=min(target - len(out), len(remaining)), replace=False)])
    return np.sort(out)


@dataclass(frozen=True)
class AcquisitionPass:
    pass_index: int
    candidate_indices: np.ndarray
    query_cap: int


def candidate_passes(unlabeled: np.ndarray, candidate_size: int, seed: int) -> list[np.ndarray]:
    """Shuffle the pool once and cut it into consecutive candidate chunks of ``candidate_size``.

    Consecutive passes therefore never revisit an example before the whole pool has been
    scored once (a superset of the manuscript's "repeat until budget met" rule that also
    maximises candidate coverage); the schedule is deterministic per seed and round.
    """
    pool = np.asarray(unlabeled, dtype=np.int64)
    if len(np.unique(pool)) != len(pool):
        raise ValueError("unlabeled indices must be unique")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(pool)
    size = int(min(candidate_size, len(perm)))
    if size <= 0:
        return []
    return [perm[s : s + size] for s in range(0, len(perm), size)]


def split_hash(indices: Sequence[int]) -> str:
    arr = np.asarray(sorted(int(i) for i in indices), dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]
