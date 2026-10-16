#!/usr/bin/env python3
"""Reconstruct every planned contrast from the released method endpoints.

Requires Python 3, NumPy and SciPy. Does not train or execute checkpoint inference.
"""
import argparse
from collections import Counter, defaultdict
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats


def holm(p_values):
    order = np.argsort(p_values, kind="stable")
    adjusted = np.zeros(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, float(p_values[index]) * (len(p_values) - rank)))
        adjusted[index] = running
    return adjusted.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="?", type=Path, default=Path(__file__).with_name("RESULTS.json"))
    parser.add_argument("--json", action="store_true", help="Print reconstructed values as JSON.")
    args = parser.parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    records = data["records"]
    keys = {(r["stage"], r["seed"], r["method"]): r for r in records}
    problems = []
    maximum_numeric_error = 0.0

    def check(label, actual, expected):
        nonlocal maximum_numeric_error
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            error = abs(float(actual) - float(expected))
            maximum_numeric_error = max(maximum_numeric_error, error)
            if error > 1e-10:
                problems.append({"field": label, "actual": actual, "expected": expected})
        elif actual != expected:
            problems.append({"field": label, "actual": actual, "expected": expected})

    check("unique record count", len(keys), len(records))
    check("method count", len(records), data["planned"]["method_runs"])
    counts = dict(Counter(r["stage"] for r in records))
    for stage in ("tuning", "confirmation", "transfer"):
        check(stage + " count", counts[stage], data["planned"][stage])
        check(stage + " seeds", sorted({r["seed"] for r in records if r["stage"] == stage}), data["protocol"]["study"][stage + "_seeds"])

    contrasts = []
    for planned in data["analysis"]["contrasts"]:
        stage = "confirmation" if planned["stage"] == "mechanism" else planned["stage"]
        seeds = sorted({r["seed"] for r in records if r["stage"] == stage})
        def accuracy(seed, method):
            return 100.0 * keys[(stage, seed, method)]["official_validation"]["accuracy"]
        if planned["target"] == "factorial_interaction":
            differences = np.array([accuracy(s, "replay_fixed") - accuracy(s, "replay_ratio") - accuracy(s, "replay_separation") + accuracy(s, "entropy") for s in seeds])
        else:
            differences = np.array([accuracy(s, planned["target"]) - accuracy(s, planned["reference"]) for s in seeds])
        n = len(differences)
        mean = float(differences.mean())
        sd = float(differences.std(ddof=1))
        half_width = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n))
        signed_means = np.abs(np.array(list(itertools.product((-1.0, 1.0), repeat=n))) @ differences / n)
        reconstructed = {
            "stage": planned["stage"], "target": planned["target"], "reference": planned["reference"],
            "n_pairs": n, "difference_pp": mean, "sd_difference_pp": sd,
            "ci95_low_pp": mean - half_width, "ci95_high_pp": mean + half_width,
            "paired_t_p": float(stats.ttest_1samp(differences, 0.0).pvalue),
            "exact_signflip_p": float(np.count_nonzero(signed_means >= abs(mean) - 1e-12) / (2 ** n)),
            "positive_seeds": int(np.sum(differences > 1e-12)),
            "negative_seeds": int(np.sum(differences < -1e-12)),
            "zero_seeds": int(np.sum(np.abs(differences) <= 1e-12)),
        }
        contrasts.append(reconstructed)
    for adjusted_key, raw_key in (("holm_t_p", "paired_t_p"), ("holm_exact_p", "exact_signflip_p")):
        for contrast, adjusted in zip(contrasts, holm([c[raw_key] for c in contrasts])):
            contrast[adjusted_key] = adjusted
    check("contrast family size", len(contrasts), 17)
    for reconstructed, planned in zip(contrasts, data["analysis"]["contrasts"]):
        for field, value in reconstructed.items():
            check(reconstructed["stage"] + ":" + reconstructed["target"] + "-" + reconstructed["reference"] + "." + field, value, planned[field])

    groups = defaultdict(list)
    for record in records:
        groups[(record["stage"], record["method"])].append(record)
    group_means = []
    for (stage, method), group in sorted(groups.items()):
        key = "holdout" if stage == "tuning" else "official_validation"
        values = np.array([100.0 * r[key]["accuracy"] for r in group])
        group_means.append({"stage": stage, "method": method, "n": len(group), "accuracy_mean_percent": float(values.mean()), "accuracy_sd_pp": float(values.std(ddof=1))})
    selections = {}
    for family in ("fixed", "schedule", "supcon"):
        candidates = []
        for (stage, name), group in groups.items():
            if stage == "tuning" and group[0]["method_spec"]["family"] == family:
                candidates.append({"name": name, "accuracy": float(np.mean([r["holdout"]["accuracy"] for r in group])), "ce": float(np.mean([r["holdout"]["ce"] for r in group]))})
        best = max(c["accuracy"] for c in candidates)
        selected = min((c for c in candidates if best - c["accuracy"] <= 1e-12), key=lambda c: (c["ce"], c["name"]))
        selections[family] = selected["name"]
        check("selection " + family, selected["name"], data["selection"][family]["name"])
    for candidate in data["selection"]["ranking"]:
        name = candidate["candidate"]["name"]
        group = groups[("tuning", name)]
        check(name + " holdout mean", float(np.mean([r["holdout"]["accuracy"] for r in group])), candidate["mean_accuracy"])
        check(name + " holdout CE", float(np.mean([r["holdout"]["ce"] for r in group])), candidate["mean_ce"])

    cfg = data["protocol"]
    n_train = sum(data["data_audit"]["counts_by_class"]["train"])
    n_val = sum(data["data_audit"]["counts_by_class"]["val"])
    initial = n_train // 10
    holdout = initial // 20
    batch = cfg["batch_classes"] * cfg["samples_per_class"]
    phase_steps = [math.ceil((initial * (i + 1) - holdout) / batch) * cfg["round_epochs"] for i in range(1, 6)]
    initial_steps = math.ceil((initial - holdout) / batch) * cfg["initial_epochs"]
    check("training inventory", n_train, cfg["expected_train_count"])
    check("validation inventory", n_val, cfg["expected_val_count"])
    check("included holdout count", holdout, cfg["holdout_count"])
    check("postinitial phase count", len(records) * cfg["rounds"], data["planned"]["postinitial_phases"])
    check("initialization count", len({r["initial_checkpoint_sha256"] for r in records}), data["planned"]["initializations"])
    check("total optimizer steps", len(records) * sum(phase_steps) + data["planned"]["initializations"] * initial_steps, data["planned"]["optimizer_steps"])
    for record in records:
        check("record annotation count", record["annotation_count"], n_train if record["method"] == "full_labels" else initial * 6)
        check("record postinitial steps", record["training_steps"], sum(phase_steps))
        check("record holdout count", record["holdout"]["n"], holdout)
        check("record final reload flag", record["final_checkpoint_reloaded"], True)
        if record["stage"] == "tuning":
            check("no tuning validation", record["official_validation"], None)
        else:
            check("record validation count", record["official_validation"]["n"], n_val)
    result = {"status": "PASS" if not problems else "FAIL", "stage_counts": counts, "reconstructed_selection": selections, "contrasts": contrasts, "group_means": group_means, "maximum_numeric_error": maximum_numeric_error, "problems": problems, "scope": "Endpoint arithmetic reconstruction; no training, image audit, or checkpoint inference."}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("Endpoint statistical reconstruction: " + result["status"])
        print(f"{len(records)} runs; {len(contrasts)} contrasts; stages {counts}")
        print("Selected controls: " + "; ".join(f"{k}={v}" for k, v in selections.items()))
        print("All effects and individual 95% intervals are percentage points; p-values below are Holm-adjusted.")
        for c in contrasts:
            print(f'{c["stage"]}: {c["target"]} - {c["reference"]}: {c["difference_pp"]:+.6f} [{c["ci95_low_pp"]:+.6f}, {c["ci95_high_pp"]:+.6f}]; t p={c["holm_t_p"]:.8g}; exact p={c["holm_exact_p"]:.8g}')
        print("Scope: " + result["scope"])
        for problem in problems:
            print(json.dumps(problem))
    raise SystemExit(0 if not problems else 1)


if __name__ == "__main__":
    main()
