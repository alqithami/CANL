"""Atomic records and explicitly paired seed-level inference for the follow-up."""
from __future__ import annotations
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import numpy as np
from scipy import stats


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


atomic = atomic_json


def holm(pvalues):
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in pvalues):
        raise ValueError('Invalid p value')
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [None] * len(pvalues)
    largest = 0.0
    for rank, i in enumerate(order):
        largest = max(largest, min(1.0, (len(order) - rank) * pvalues[i]))
        adjusted[i] = largest
    return adjusted


def paired(target, reference):
    t, r = np.asarray(target, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if t.shape != r.shape or t.ndim != 1 or not 2 <= len(t) <= 20:
        raise ValueError('Expected equal one-dimensional paired seed vectors, 2..20 seeds')
    d = t - r
    if not np.isfinite(d).all():
        raise ValueError('Nonfinite paired outcome')
    n, mean, sd = len(d), float(d.mean()), float(d.std(ddof=1))
    if sd:
        se = sd / math.sqrt(n)
        half = float(stats.t.ppf(.975, n - 1)) * se
        lo, hi = mean - half, mean + half
        p = float(2 * stats.t.sf(abs(mean / se), n - 1))
    else:
        lo = hi = mean
        p = 1.0 if mean == 0 else 0.0
    extreme = sum(abs(float(np.dot(d, signs) / n)) >= abs(mean) - 1e-14
                  for signs in itertools.product((-1.0, 1.0), repeat=n))
    return dict(n_pairs=n, difference_pp=mean, sd_difference_pp=sd,
                ci95_low_pp=lo, ci95_high_pp=hi, paired_t_p=p,
                exact_signflip_p=extreme / 2 ** n,
                positive_seeds=int((d > 0).sum()), negative_seeds=int((d < 0).sum()),
                zero_seeds=int((d == 0).sum()),
                degenerate_note='Zero seed-difference variance; not independent precision evidence.' if not sd else None)
