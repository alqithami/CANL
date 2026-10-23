#!/usr/bin/env python3
"""Locked, complete-case-only analysis for the fresh AudioCaps confirmation.

This module does not run training, select a configuration, inspect pilot outcomes,
or use a random-number generator.  Its only inferential endpoint is CIDEr-D.
Run ``python study_stats.py ROOT`` after all 40 confirmation jobs are complete.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import statistics
from typing import Any, Iterable, Sequence

from scipy.stats import t as student_t


SEEDS = tuple(range(82001, 82011))
METHODS = ("baseline", "tuned_fixed", "macc_lite", "full_macc")
CONTRASTS = (
    ("tuned_fixed", "baseline"),
    ("macc_lite", "baseline"),
    ("full_macc", "baseline"),
    ("macc_lite", "tuned_fixed"),
    ("full_macc", "tuned_fixed"),
)
METRICS = (
    "cider_d", "bleu1", "bleu2", "bleu3", "bleu4", "rouge_l",
    "meteor", "spice", "spider", "test_token_nll", "development_token_nll",
)


class IncompleteStudyError(ValueError):
    """No aggregate may be published from an incomplete or mismatched study."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IncompleteStudyError(message)


def _read(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        raise IncompleteStudyError(f"Cannot read required JSON {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configuration_digest(job: dict[str, Any]) -> str:
    unbound = {key: value for key, value in job.items() if key != "binding"}
    encoded = (json.dumps(unbound, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def _is_metric_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_numbers(values: Iterable[Any], *, label: str) -> list[float]:
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} contains a nonnumeric value: {value!r}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label} contains a nonfinite value; outcomes are never dropped")
        result.append(number)
    if not result:
        raise ValueError(f"{label} is empty")
    return result


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, preserving the supplied contrast order."""
    values = _finite_numbers(p_values, label="p-values")
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("p-values must lie in [0, 1]")
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0] * len(values)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = previous
    return adjusted


def exact_sign_flip(differences: Sequence[float]) -> dict[str, Any]:
    """Enumerate all sign assignments, retaining equality in the two-sided tail.

    Fractions represent the supplied binary floating-point differences exactly.
    Thus an equal-magnitude statistic is included without a scale-dependent
    tolerance, and the observed all-plus/all-minus assignments cannot be lost.
    """
    values = _finite_numbers(differences, label="paired differences")
    if len(values) != len(SEEDS):
        raise ValueError("The locked exact test requires all ten paired seeds")
    fractions = [Fraction.from_float(value) for value in values]
    observed = abs(sum(fractions, Fraction(0)))
    extreme = 0
    total = 2 ** len(values)
    for signs in itertools.product((-1, 1), repeat=len(values)):
        statistic = abs(sum((value * sign for value, sign in zip(fractions, signs)), Fraction(0)))
        extreme += statistic >= observed
    return {
        "p_value": extreme / total,
        "extreme_assignments_including_ties": extreme,
        "enumerated_assignments": total,
        "exact": True,
        "alternative": "two-sided",
        "statistic": "absolute mean of paired target-minus-reference differences",
        "tail_rule": ">= observed, including ties",
        "minimum_possible_two_sided_p": 2 / total,
        "monte_carlo_correction": False,
    }


def paired_contrast(target: Sequence[float], reference: Sequence[float]) -> dict[str, Any]:
    """Paired t inference and exact sign flips for the ten locked paired seeds."""
    a = _finite_numbers(target, label="target")
    b = _finite_numbers(reference, label="reference")
    if len(a) != len(SEEDS) or len(b) != len(SEEDS):
        raise ValueError("The locked contrast requires ten outcomes per method")
    differences = [left - right for left, right in zip(a, b)]
    if not all(math.isfinite(value) for value in differences):
        raise ValueError("A paired difference is nonfinite")
    mean = statistics.fmean(differences)
    sample_sd = statistics.stdev(differences)
    standard_error = sample_sd / math.sqrt(len(differences))
    critical = float(student_t.ppf(0.975, df=9))
    margin = critical * standard_error
    t_note = None
    if sample_sd == 0.0:
        # The all-zero case is the no-difference null.  For a nonzero constant
        # effect use the conventional infinite-t limit, encoded without JSON NaN.
        t_statistic = 0.0 if mean == 0.0 else None
        t_p = 1.0 if mean == 0.0 else 0.0
        if mean != 0.0:
            t_note = "zero variance; infinite-t limit (" + ("positive" if mean > 0 else "negative") + ")"
    else:
        t_statistic = mean / standard_error
        t_p = float(2.0 * student_t.sf(abs(t_statistic), df=9))
    exact = exact_sign_flip(differences)
    result = {
        "n_pairs": len(differences),
        "seed_order": list(SEEDS),
        "paired_differences": differences,
        "difference_mean": mean,
        "difference_sample_sd": sample_sd,
        "difference_standard_error": standard_error,
        "confidence_interval_95_unadjusted": [mean - margin, mean + margin],
        "confidence_interval_method": "individual two-sided Student t interval, df=9; not simultaneous",
        "paired_t_statistic": t_statistic,
        "paired_t_df": 9,
        "paired_t_p_raw": t_p,
        "exact_sign_flip_p_raw": exact["p_value"],
        "exact_sign_flip": exact,
    }
    if t_note is not None:
        result["paired_t_note"] = t_note
    return result


def _safe_manifest_file(folder: Path, relative: str) -> Path:
    _require(isinstance(relative, str), "Artifact paths must be strings")
    relative_path = PurePosixPath(relative)
    _require(not relative_path.is_absolute() and ".." not in relative_path.parts and "\\" not in relative,
             f"Unsafe artifact path: {relative}")
    path = folder / relative
    _require(path.resolve().is_relative_to(folder.resolve()), f"Artifact escapes job directory: {relative}")
    _require(path.is_file(), f"Missing declared artifact: {path}")
    return path


def _validate_job(folder: Path, method: str, seed: int, expected_steps: int | None,
                  expected_test_clips: int | None) -> tuple[dict[str, Any], Any, dict[str, str]]:
    summary = _read(folder / "summary.json")
    done = _read(folder / "DONE.json")
    job = _read(folder / "job.json")
    _require(isinstance(summary, dict) and isinstance(done, dict) and isinstance(job, dict),
             f"Job metadata must be objects: {folder}")
    _require(done.get("status") == "COMPLETE", f"Job is not complete: {folder}")
    for name, payload in (("summary", summary), ("DONE", done)):
        _require(payload.get("method") == method and payload.get("seed") == seed,
                 f"{name} identity differs from locked directory: {folder}")
    _require(summary.get("phase") == "confirm", f"Nonconfirmatory outcome in confirmation tree: {folder}")
    _require(done.get("phase") == "confirm", f"Nonconfirmatory DONE in confirmation tree: {folder}")
    _require(job.get("phase") == "confirm" and job.get("seed") == seed
             and isinstance(job.get("method"), dict) and job["method"].get("name") == method,
             f"Job configuration identity differs from locked directory: {folder}")
    _require("input_binding" in job, f"Missing input_binding: {folder}/job.json")
    binding = job["input_binding"]
    _require(binding is not None, f"Null input binding: {folder}")
    _require(done.get("input_binding") == binding,
             f"DONE binding differs from job input binding: {folder}")
    _require("binding" in job and done.get("binding") == job["binding"],
             f"DONE configuration binding differs from job: {folder}")
    _require(job["binding"] == _configuration_digest(job), f"Configuration binding SHA-256 differs: {folder}")
    steps = summary.get("steps")
    _require(isinstance(steps, int) and not isinstance(steps, bool) and steps > 0,
             f"Missing completed training budget: {folder}")
    _require(steps == summary.get("steps_expected") == done.get("scheduled_steps"),
             f"Scheduled/actual training budgets differ: {folder}")
    if expected_steps is not None:
        _require(steps == expected_steps, f"Training budget differs from frozen study: {folder}")
    test_clips = summary.get("test_clips")
    _require(isinstance(test_clips, int) and not isinstance(test_clips, bool) and test_clips > 0
             and summary.get("official_test_caption_outcomes_computed") is True,
             f"Incomplete official test evaluation: {folder}")
    if expected_test_clips is not None:
        _require(test_clips == expected_test_clips, f"Test clip count differs from frozen study: {folder}")
    files = done.get("files")
    _require(isinstance(files, dict) and bool(files), f"Missing artifact integrity map: {folder}")
    _require("summary.json" in files and "job.json" in files,
             f"Integrity map must cover summary.json and job.json: {folder}")
    _require("artifacts/RESULT_AUDIT.json" in files,
             f"Integrity map must cover the independent saved-result audit: {folder}")
    _require("final_state.pt" in files,
             f"Integrity map must cover the final retained checkpoint: {folder}")
    hashes: dict[str, str] = {}
    for relative, digest in files.items():
        _require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                 f"Invalid SHA-256 for {folder}/{relative}")
        path = _safe_manifest_file(folder, relative)
        _require(_sha256(path) == digest, f"Artifact SHA-256 differs: {path}")
        hashes[relative] = digest
    audit = _read(folder / "artifacts/RESULT_AUDIT.json")
    _require(audit.get("status") == "PASS" and audit.get("saved_caption_metrics_recomputed") is True
             and audit.get("checkpoint_clean_replay") is True,
             f"Saved-caption metric/checkpoint replay audit has not passed: {folder}")
    _require(audit.get("job_binding") == job["binding"] and audit.get("input_binding") == binding
             and audit.get("summary_sha256") == files["summary.json"]
             and audit.get("final_state_sha256") == files["final_state.pt"],
             f"Saved-result audit is not bound to this job, summary and final checkpoint: {folder}")
    values = summary.get("final")
    _require(isinstance(values, dict), f"Missing final metrics: {folder}")
    _require(all(metric in values and _is_metric_number(values[metric]) for metric in METRICS),
             f"Missing/nonnumeric prespecified metric: {folder}")
    # Retain every numeric final metric. A new metric must be available for all
    # methods/seeds; there is no pairwise deletion or best-run selection.
    try:
        _finite_numbers((value for value in values.values() if _is_metric_number(value)), label=f"{folder}/final")
    except ValueError as exc:
        raise IncompleteStudyError(str(exc)) from exc
    return summary, binding, hashes


def analyze(root: str | Path, *, expected_input_binding: Any = None,
            expected_steps: int | None = None, expected_test_clips: int | None = None) -> dict[str, Any]:
    """Verify all 40 payloads, write ``analysis.json``, and return the aggregate.

    ``expected_input_binding`` is the SHA-256 string from the frozen input lock.
    ``expected_steps`` is the prespecified scheduled budget from that same lock.
    ``expected_test_clips`` is its frozen retained test split count.
    If not supplied, all jobs must still share one identical bound input object.
    The first failed validation raises IncompleteStudyError before any output.
    """
    root = Path(root)
    expected_paths = {Path("confirm") / method / f"s{seed}" / "summary.json" for method in METHODS for seed in SEEDS}
    actual_paths = {path.relative_to(root) for path in (root / "confirm").rglob("summary.json")}
    missing = sorted(str(path) for path in expected_paths - actual_paths)
    unexpected = sorted(str(path) for path in actual_paths - expected_paths)
    _require(not missing and not unexpected,
             f"Confirmation must contain exactly all 40 locked runs; missing={missing}; unexpected={unexpected}")
    rows = []
    by_method: dict[str, list[dict[str, float]]] = {method: [] for method in METHODS}
    binding = expected_input_binding
    scheduled_steps = expected_steps
    test_clips = expected_test_clips
    metric_names = None
    source_hashes = {}
    for seed in SEEDS:
        for method in METHODS:
            folder = root / "confirm" / method / f"s{seed}"
            summary, run_binding, hashes = _validate_job(folder, method, seed, scheduled_steps, test_clips)
            if scheduled_steps is None:
                scheduled_steps = summary["steps"]
            if test_clips is None:
                test_clips = summary["test_clips"]
            if binding is None:
                binding = run_binding
            _require(run_binding == binding, f"Input binding differs across locked jobs: {folder}")
            if metric_names is None:
                metric_names = tuple(sorted(key for key, value in summary["final"].items() if _is_metric_number(value)))
            _require(tuple(sorted(key for key, value in summary["final"].items() if _is_metric_number(value))) == metric_names,
                     f"Final numeric metric set differs: {folder}")
            values = {metric: float(summary["final"][metric]) for metric in metric_names}
            by_method[method].append(values)
            metadata = {key: value for key, value in summary["final"].items() if not _is_metric_number(value)}
            rows.append({"seed": seed, "method": method, "final_metrics": values, "final_metadata": metadata})
            source_hashes[f"confirm/{method}/s{seed}"] = {
                "done_sha256": _sha256(folder / "DONE.json"), "files": hashes,
            }
    grouped = {
        method: {
            metric: {
                "n": len(SEEDS),
                "mean": statistics.fmean(row[metric] for row in by_method[method]),
                "sample_sd": statistics.stdev(row[metric] for row in by_method[method]),
                "min": min(row[metric] for row in by_method[method]),
                "max": max(row[metric] for row in by_method[method]),
            }
            for metric in metric_names
        }
        for method in METHODS
    }
    contrasts = []
    for target, reference in CONTRASTS:
        comparison = paired_contrast(
            [row["cider_d"] for row in by_method[target]],
            [row["cider_d"] for row in by_method[reference]],
        )
        comparison.update(target=target, reference=reference, endpoint="CIDEr-D",
                          positive_favors_target=True, family_size=5,
                          primary_test="two-sided exact sign-flip test; Holm across five contrasts",
                          sensitivity_test="two-sided paired t test; separate five-contrast Holm adjustment")
        contrasts.append(comparison)
    t_adjusted = holm_adjust([row["paired_t_p_raw"] for row in contrasts])
    exact_adjusted = holm_adjust([row["exact_sign_flip_p_raw"] for row in contrasts])
    for index, comparison in enumerate(contrasts):
        comparison.update(paired_t_p_holm=t_adjusted[index], exact_sign_flip_p_holm=exact_adjusted[index])
    result = {
        "status": "COMPLETE",
        "analysis_version": "audiocaps-raw-confirm-10-paired-seeds-v1",
        "completed_method_runs": len(rows),
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "input_binding": binding,
        "scheduled_steps": scheduled_steps,
        "test_clips": test_clips,
        "primary_endpoint": "CIDEr-D",
        "primary_contrasts": contrasts,
        "multiplicity": {
            "method": "Holm step-down",
            "family_size": 5,
            "families": ["five paired-t p-values", "five exact-sign-flip p-values"],
            "primary_family": "five exact-sign-flip p-values",
            "sensitivity_family": "five paired-t p-values; never an alternative success criterion",
            "primary_familywise_alpha": 0.05,
            "no_either_family_success_rule": True,
            "confidence_intervals": "individual unadjusted 95%; not simultaneous",
        },
        "secondary_metrics": "descriptive only; no hypothesis tests or multiplicity-adjusted claims",
        "grouped_metrics": grouped,
        "per_seed": rows,
        "integrity": source_hashes,
        "interpretation_notes": [
            "Primary inference uses the five Holm-adjusted exact sign-flip tests at familywise alpha 0.05. Paired-t results are a separately adjusted sensitivity analysis, not an alternative route to significance.",
            "All ten prespecified seeds are paired by identity; no failed or nonfinite outcome is dropped.",
            "Two-sided paired-t inference assumes independent seed pairs and approximately normally distributed paired seed differences.",
            "The exact sign-flip test assumes sign exchangeability (symmetry) of paired differences under the null.",
            "Sign flips enumerate all 1024 configurations and include every statistic equal to the observed magnitude.",
            "The smallest possible two-sided exact p-value with ten nonzero pairs is 2/1024, before Holm adjustment.",
        ],
    }
    output = root / "analysis.json"
    temporary = output.with_name("analysis.json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-binding", type=Path, help="JSON file containing the frozen input binding")
    parser.add_argument("--expected-steps", type=int, help="Frozen scheduled optimizer budget")
    parser.add_argument("--expected-test-clips", type=int, help="Frozen retained test split count")
    args = parser.parse_args()
    binding = _read(args.expected_binding) if args.expected_binding is not None else None
    if isinstance(binding, dict) and "sha256" in binding:
        binding = binding["sha256"]
    result = analyze(args.root, expected_input_binding=binding, expected_steps=args.expected_steps,
                     expected_test_clips=args.expected_test_clips)
    print(json.dumps({"status": result["status"], "completed_method_runs": result["completed_method_runs"],
                      "analysis": str(args.root / "analysis.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
