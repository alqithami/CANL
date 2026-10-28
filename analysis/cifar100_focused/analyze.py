#!/usr/bin/env python3
"""Strict, future-confirmation analysis; does not train or open test-label files.

Usage: python3 analyze.py RESULTS_INPUT.json --protocol LOCKED_PROTOCOL.json
The actual bytes of LOCKED_PROTOCOL.json are SHA-256 bound by the input.
Only a complete, independently locked confirmation contract is accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np


STUDY_ID = "caenl-cifar100-focused-confirmation-v1"
METHODS = ("ce", "ratio_dcr", "full_dcr", "plain_fisher")
FRACTIONS = (0.1, 0.2, 0.4, 0.6)
LABEL_COUNTS = (4500, 9000, 18000, 27000)
AULC_WEIGHTS = np.asarray((0.1, 0.3, 0.4, 0.2), dtype=np.float64)
CONTRASTS = tuple(("ratio_dcr", m) for m in METHODS if m != "ratio_dcr")
ANALYSIS_CONTRACT = {
    "endpoint": "normalized_trapezoidal_AULC_accuracy_percent",
    "primary_contrasts": [list(x) for x in CONTRASTS],
    "primary_test": "two_sided_exact_paired_sign_flip",
    "primary_adjustment": "Holm",
    "sensitivity_test": "paired_t",
    "sensitivity_adjustment": "Holm",
    "confidence_level": 0.95,
}
CONFIRMATION_SEEDS = tuple(range(92001,92011))
MAX_EXACT_SEEDS = 20


class ContractError(ValueError):
    """Incomplete, unlocked, mismatched, or wrong-stage evidence."""


def _require(condition, message):
    if not condition:
        raise ContractError(message)


def _integer(value, name, minimum=1):
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
             f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name, low=None, high=None):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(value), f"{name} must be a finite number")
    if low is not None:
        _require(value >= low, f"{name} must be >= {low}")
    if high is not None:
        _require(value <= high, f"{name} must be <= {high}")
    return float(value)


def _sha(value, name):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
             and value != "0" * 64, f"{name} must be a non-placeholder lowercase SHA-256")
    return value


def normalized_aulc(accuracies):
    a = np.asarray(accuracies, dtype=np.float64)
    _require(a.shape == (4,) and np.isfinite(a).all()
             and (a >= 0).all() and (a <= 100).all(),
             "AULC requires exactly four finite accuracy percentages in [0,100]")
    return float(a @ AULC_WEIGHTS)


def exact_sign_flip_p(differences):
    """Two-sided exact test of absolute paired sum; no Monte Carlo fallback.

    Zero differences can be omitted because their signs duplicate every outcome
    equally. Exchangeability under independent within-pair sign changes is an
    assumption of this test, not established by a small p-value.
    """
    d = np.asarray(differences, dtype=np.float64)
    _require(d.ndim == 1 and d.size >= 1 and np.isfinite(d).all(),
             "Sign-flip differences must be a nonempty finite vector")
    _require(d.size <= MAX_EXACT_SEEDS,
             f"Exact enumeration supports at most {MAX_EXACT_SEEDS} paired seeds")
    nonzero = d[d != 0]
    if not nonzero.size:
        return 1.0
    observed = abs(float(d.sum()))
    tolerance = 16 * np.finfo(np.float64).eps * max(1.0, float(np.abs(d).sum()))
    # Vectorized doubling gives all 2**n sums without an n-by-2**n sign matrix.
    sums = np.asarray((0.0,))
    for value in nonzero:
        sums = np.concatenate((sums + value, sums - value))
    return float(np.count_nonzero(np.abs(sums) >= observed - tolerance) / sums.size)


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=np.float64)
    _require(p.ndim == 1 and p.size > 0 and np.isfinite(p).all()
             and (p >= 0).all() and (p <= 1).all(), "Holm requires finite p-values in [0,1]")
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum(1.0, np.maximum.accumulate(p[order] * np.arange(p.size, 0, -1)))
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result.tolist()


def paired_t_summary(differences):
    try:
        from scipy.stats import t as student_t
    except ImportError as exc:
        raise RuntimeError("SciPy is required for the prespecified paired-t sensitivity analysis and confidence intervals") from exc
    d = np.asarray(differences, dtype=np.float64)
    _require(d.ndim == 1 and len(d) >= 2 and np.isfinite(d).all(),
             "Paired-t analysis requires at least two finite seed differences")
    n, mean, sd = len(d), float(d.mean()), float(d.std(ddof=1))
    if sd == 0:
        # Avoid nonfinite JSON; report the conventional zero-variance limit explicitly.
        return {"mean_difference_pp": mean, "seed_difference_sd_pp": sd,
                "ci95_unadjusted_pp": [mean, mean], "t_statistic": None,
                "degrees_of_freedom": n-1, "p_raw": 1.0 if mean == 0 else 0.0,
                "zero_variance_limit": True}
    se = sd / math.sqrt(n)
    statistic = mean / se
    critical = float(student_t.ppf(0.975, n-1))
    return {"mean_difference_pp": mean, "seed_difference_sd_pp": sd,
            "ci95_unadjusted_pp": [mean-critical*se, mean+critical*se],
            "t_statistic": statistic, "degrees_of_freedom": n-1,
            "p_raw": float(2*student_t.sf(abs(statistic), n-1)),
            "zero_variance_limit": False}


def validate_protocol(protocol):
    _require(isinstance(protocol, dict), "Locked protocol must be an object")
    _require(protocol.get("schema_version") == 1, "Protocol schema_version must be 1")
    _require(protocol.get("protocol_id") == STUDY_ID, "Wrong confirmation protocol ID")
    _require(protocol.get("locked") is True, "Protocol is not locked; no confirmation analysis")
    _require(protocol.get("stage") == "confirmation", "Protocol stage must be confirmation")
    _sha(protocol.get("source_binding_sha256"), "Protocol source_binding_sha256")
    _require(protocol.get("methods") == list(METHODS), "Protocol must freeze the four focused methods in declared order")
    _require(protocol.get("label_fractions") == list(FRACTIONS), "Protocol must freeze all four label fractions")
    _require(protocol.get("label_counts") == list(LABEL_COUNTS), "Protocol must freeze all four label counts")
    _require(protocol.get("analysis") == ANALYSIS_CONTRACT, "Protocol analysis contract does not match the prespecified three-contrast analysis")
    _sha(protocol.get("test_release_receipt_sha256"), "Protocol explicit test-release receipt SHA-256")
    seeds = protocol.get("confirmation_seeds")
    _require(isinstance(seeds, list) and 2 <= len(seeds) <= MAX_EXACT_SEEDS,
             f"Require 2..{MAX_EXACT_SEEDS} predeclared confirmation seeds")
    for seed in seeds:
        _integer(seed, "Confirmation seed")
    _require(len(set(seeds)) == len(seeds), "Duplicate confirmation seeds")
    _require(seeds == list(CONFIRMATION_SEEDS), "Confirmation seed family differs from frozen ten-seed plan")
    return sorted(seeds)


def validate_records(data, protocol, protocol_sha256):
    seeds = validate_protocol(protocol)
    _sha(protocol_sha256, "Expected protocol_sha256")
    _require(isinstance(data, dict) and data.get("schema_version") == 1,
             "Input schema_version must be 1")
    _require(data.get("stage") == "confirmation", "Engineering, development, and other non-confirmation input is refused")
    _require(data.get("status") == "COMPLETED_VERIFIED", "Input must declare a completed, verified confirmation")
    _require(data.get("protocol_id") == STUDY_ID, "Input confirmation protocol ID mismatch")
    _require(data.get("protocol_sha256") == protocol_sha256, "Input is not bound to the exact locked protocol bytes")
    _require(data.get("source_binding_sha256") == protocol["source_binding_sha256"], "Input source binding mismatch")
    _require(data.get("evaluation_partition") == "official_test" and data.get("test_release_verified") is True, "Only separately released official-test evaluations may enter confirmation analysis")
    _require(data.get("test_release_receipt_sha256") == protocol["test_release_receipt_sha256"], "Explicit test-release binding mismatch")
    records = data.get("records")
    _require(isinstance(records, list), "Confirmation records must be a list")
    expected = set(itertools.product(METHODS, seeds, range(4)))
    found = {}
    for record in records:
        _require(isinstance(record, dict), "Every confirmation record must be an object")
        _require(record.get("stage") == "confirmation" and record.get("status") == "COMPLETED_VERIFIED",
                 "Partial, engineering, development, or unverified record refused")
        _require(record.get("protocol_sha256") == protocol_sha256,
                 "Record protocol binding mismatch")
        _require(record.get("source_binding_sha256") == protocol["source_binding_sha256"],
                 "Record source binding mismatch")
        _sha(record.get("receipt_sha256"), "Record receipt_sha256")
        _require(record.get("evaluation_partition") == "official_test", "Development diagnostic is not an official-test record")
        method = record.get("method")
        _require(method in METHODS, f"Unknown method {method!r}")
        seed = _integer(record.get("seed"), "Record seed")
        _require(seed in seeds, "Record seed was not predeclared for confirmation")
        fraction = _number(record.get("label_fraction"), "Record label_fraction")
        _require(fraction in FRACTIONS, "Record must use an exact prespecified label fraction")
        index = FRACTIONS.index(fraction)
        count = _integer(record.get("label_count"), "Record label_count")
        _require(count == LABEL_COUNTS[index], "Record label_count/fraction mismatch")
        value = _number(record.get("accuracy_percent"), "Record accuracy_percent", 0, 100)
        key = (method, seed, index)
        _require(key not in found, "Duplicate method/seed/budget record")
        found[key] = value
    missing = expected - set(found)
    extra = set(found) - expected
    _require(not missing and not extra,
             f"Incomplete/unmatched seed-method-budget grid: {len(missing)} missing, {len(extra)} unexpected")
    return seeds, {m: np.asarray([[found[(m, s, j)] for j in range(4)] for s in seeds]) for m in METHODS}


def descriptive_costs(data, seeds):
    if "costs" not in data:
        return {"provided": False, "overhead_computed": False}
    costs = data["costs"]
    _require(isinstance(costs, dict) and isinstance(costs.get("scope"), str)
             and costs["scope"].strip(), "Costs require an explicit measurement scope")
    entries = costs.get("records")
    _require(isinstance(entries, list), "Cost records must be a list")
    found = {}
    fields = ("training_seconds", "end_to_end_seconds")
    for record in entries:
        _require(isinstance(record, dict), "Cost record must be an object")
        method, seed = record.get("method"), _integer(record.get("seed"), "Cost seed")
        _require(method in METHODS and seed in seeds, "Unmatched cost method/seed")
        key = (method, seed)
        _require(key not in found, "Duplicate cost method/seed record")
        values = {f: _number(record[f], f, 0) for f in fields if f in record}
        _require(values and all(v > 0 for v in values.values()), "Provided cost durations must be positive")
        found[key] = values
    _require(set(found) == set(itertools.product(METHODS, seeds)), "Provided costs must cover every paired seed and method")
    common = set.intersection(*(set(v) for v in found.values()))
    _require(common and all(set(v) == common for v in found.values()), "Cost fields must match across all methods and seeds")
    result = {"provided": True, "overhead_computed": True, "descriptive_only": True,
              "scope": costs["scope"], "costs_cover": "all four budget stages, once per method-seed trajectory",
              "fields": {}}
    for field in fields:
        if field not in common:
            continue
        by_method = {m: np.asarray([found[(m, s)][field] for s in seeds]) for m in METHODS}
        result["fields"][field] = {
            "method_mean_seconds": {m: float(v.mean()) for m, v in by_method.items()},
            "ratio_dcr_over_controls": {control: {
                "mean_paired_overhead_percent": float(np.mean(100*(by_method["ratio_dcr"]/by_method[control]-1))),
                "seed_overhead_percent": (100*(by_method["ratio_dcr"]/by_method[control]-1)).tolist(),
            } for _, control in CONTRASTS}}
    return result


def analyze(data, protocol, protocol_sha256):
    seeds, curves = validate_records(data, protocol, protocol_sha256)
    aulc = {m: values @ AULC_WEIGHTS for m, values in curves.items()}
    rows = []
    for target, control in CONTRASTS:
        differences = aulc[target] - aulc[control]
        rows.append({"target": target, "control": control,
                     "seed_differences_pp": differences.tolist(),
                     "exact_sign_flip_p_raw": exact_sign_flip_p(differences),
                     "paired_t_sensitivity": paired_t_summary(differences)})
    exact_adjusted = holm_adjust([r["exact_sign_flip_p_raw"] for r in rows])
    t_adjusted = holm_adjust([r["paired_t_sensitivity"]["p_raw"] for r in rows])
    for row, p_exact, p_t in zip(rows, exact_adjusted, t_adjusted):
        row["exact_sign_flip_p_Holm"] = p_exact
        row["paired_t_sensitivity"]["p_Holm_separate_sensitivity_family"] = p_t
    result = {"status": "CONFIRMATION_ANALYSIS_COMPLETE", "protocol_id": STUDY_ID,
              "protocol_sha256": protocol_sha256, "source_binding_sha256": protocol["source_binding_sha256"],
              "stage": "confirmation", "independent_unit": "paired seed", "seeds": seeds,
              "n_paired_seeds": len(seeds), "label_fractions": list(FRACTIONS),
              "label_counts": list(LABEL_COUNTS), "AULC_normalization_interval": [0.1,0.6],
              "AULC_weights": AULC_WEIGHTS.tolist(), "AULC_unit": "accuracy percentage points",
              "methods": {m: {"seed_AULC": aulc[m].tolist(), "mean_AULC": float(aulc[m].mean()),
                               "seed_SD_AULC": float(aulc[m].std(ddof=1)),
                               "mean_accuracy_percent_by_budget": curves[m].mean(axis=0).tolist()}
                          for m in METHODS},
              "planned_contrasts": rows, "primary_family_size": 3,
              "primary_test": ANALYSIS_CONTRACT["primary_test"],
              "multiplicity": "Holm across all three exact primary contrasts; separate Holm family for all three paired-t sensitivity contrasts",
              "confidence_intervals": "Two-sided 95% paired-t intervals, unadjusted; not simultaneous intervals",
              "sign_flip_assumption": "Paired seed differences are independent across seeds and jointly invariant to independent sign flips under the null",
              "costs": descriptive_costs(data, seeds),
              "annotation_savings_claim": "NOT_ESTABLISHED_BY_THIS_ANALYSIS; requires a separately predeclared target, performance margin, and interpolation protocol",
              "equivalence_claim": "NOT_TESTED; a nonsignificant difference does not establish equivalence or noninferiority",
              "verification_scope": "Checks complete confirmation grid and locked protocol/source/receipt binding declarations; does not independently recompute prediction files",
              "automatic_experiment_launch": False}
    # Catch numerical serialization failures before returning a purported finished analysis.
    json.dumps(result, allow_nan=False)
    return result


def read_and_analyze(input_path, protocol_path):
    raw_protocol = Path(protocol_path).read_bytes()
    protocol = json.loads(raw_protocol)
    return analyze(json.loads(Path(input_path).read_text()), protocol,
                   hashlib.sha256(raw_protocol).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = read_and_analyze(args.input, args.protocol)
    except (ContractError, KeyError, TypeError, json.JSONDecodeError, RuntimeError) as exc:
        parser.exit(2, f"Confirmation analysis refused: {exc}\n")
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
