#!/usr/bin/env python3
"""Independent numerical recomputation of the frozen focused confirmation.

Consumes released score records only; never opens data archives or models.
Does not replace the required prediction-file and release provenance audit.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

METHODS = ('ce', 'ratio_dcr', 'full_dcr', 'plain_fisher')
SEEDS = tuple(range(92001, 92011))
COUNTS = (4500, 9000, 18000, 27000)
FRACTIONS = (.1, .2, .4, .6)


def close(a, b):
    if not math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f'Independent number mismatch: {a!r} != {b!r}')


def holm(values):
    """Straight step-down implementation, independent of frozen vectorization."""
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.] * len(values)
    previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (len(values) - rank) * values[index]))
        adjusted[index] = previous
    return adjusted


def check(data, protocol, analysis, protocol_bytes):
    digest = hashlib.sha256(protocol_bytes).hexdigest()
    assert protocol['locked'] is True
    assert protocol['confirmation_seeds'] == list(SEEDS)
    assert protocol['methods'] == list(METHODS)
    assert protocol['label_counts'] == list(COUNTS)
    assert protocol['label_fractions'] == list(FRACTIONS)
    assert data['protocol_sha256'] == analysis['protocol_sha256'] == digest
    assert data['evaluation_partition'] == 'official_test'
    assert data['test_release_verified'] is True
    assert data['test_release_receipt_sha256'] == protocol['test_release_receipt_sha256']
    expected = set(itertools.product(METHODS, SEEDS, COUNTS))
    found = {}
    for r in data['records']:
        assert r['evaluation_partition'] == 'official_test'
        assert r['status'] == 'COMPLETED_VERIFIED'
        key = (r['method'], r['seed'], r['label_count'])
        assert key not in found
        assert r['label_fraction'] == FRACTIONS[COUNTS.index(r['label_count'])]
        assert math.isfinite(r['accuracy_percent']) and 0 <= r['accuracy_percent'] <= 100
        found[key] = r['accuracy_percent']
    assert set(found) == expected
    curves = {m: [[found[m, s, n] for n in COUNTS] for s in SEEDS] for m in METHODS}
    # Integrate adjacent trapezoids over theactual .1 to .6 horizontal interval.
    aulc = {m: np.asarray([sum((row[j] + row[j + 1]) * .5 *
                              (FRACTIONS[j + 1] - FRACTIONS[j])
                              for j in range(3)) / .5 for row in curves[m]])
            for m in METHODS}
    for method in METHODS:
        actual = analysis['methods'][method]
        for x, y in zip(aulc[method], actual['seed_AULC']):
            close(x, y)
        close(aulc[method].mean(), actual['mean_AULC'])
        close(aulc[method].std(ddof=1), actual['seed_SD_AULC'])
        for x, y in zip(np.asarray(curves[method]).mean(axis=0), actual['mean_accuracy_percent_by_budget']):
            close(x, y)
    rows = analysis['planned_contrasts']
    assert [(r['target'], r['control']) for r in rows] == [('ratio_dcr', c) for c in METHODS if c != 'ratio_dcr']
    raw_exact, raw_t = [], []
    for row in rows:
        d = aulc['ratio_dcr'] - aulc[row['control']]
        for x, y in zip(d, row['seed_differences_pp']):
            close(x, y)
        observed = abs(sum(d))
        tolerance = 16 * np.finfo(np.float64).eps * max(1., sum(abs(d)))
        outcomes = [abs(sum(sign * value for sign, value in zip(signs, d)))
                    for signs in itertools.product((-1, 1), repeat=10)]
        p = sum(value >= observed - tolerance for value in outcomes) / len(outcomes)
        close(p, row['exact_sign_flip_p_raw']); raw_exact.append(p)
        mean, sd = float(d.mean()), float(d.std(ddof=1))
        sensitivity = row['paired_t_sensitivity']
        close(mean, sensitivity['mean_difference_pp']); close(sd, sensitivity['seed_difference_sd_pp'])
        if sd == 0:
            ci = (mean, mean); p_t = 1. if mean == 0 else 0.
        else:
            se = sd / math.sqrt(10)
            ci = (mean - student_t.ppf(.975, 9) * se, mean + student_t.ppf(.975, 9) * se)
            p_t = float(2 * student_t.sf(abs(mean / se), 9))
        for x, y in zip(ci, sensitivity['ci95_unadjusted_pp']):
            close(x, y)
        close(p_t, sensitivity['p_raw']); raw_t.append(p_t)
    for row, pe, pt in zip(rows, holm(raw_exact), holm(raw_t)):
        close(pe, row['exact_sign_flip_p_Holm'])
        close(pt, row['paired_t_sensitivity']['p_Holm_separate_sensitivity_family'])
    return {'status': 'PASS', 'score_records': 160, 'paired_seeds': 10,
            'aulc_recomputed_by_adjacent_trapezoids': True,
            'sign_flip_recomputed_by_1024_explicit_sign_vectors': True,
            'both_holm_families_recomputed': True,
            'all_unadjusted_t_intervals_recomputed': True,
            'scope': 'Independent score-level numerical verification; prediction/provenance verification remains separate.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.protocol.read_bytes()
    result = check(json.loads(args.input.read_text()), json.loads(raw),
                   json.loads(args.analysis.read_text()), raw)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
