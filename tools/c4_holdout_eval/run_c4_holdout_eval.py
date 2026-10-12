#!/usr/bin/env python3
"""Evaluation-only C4 holdout audit for the completed CAENL confirmatory campaign.

This script does not train or modify any checkpoint.  It evaluates all 24 saved
checkpoints on a C4 validation shard that was not used by the calibration or
confirmatory jobs, then produces paired statistics and a checksum-verified lean
archive.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import itertools
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
from typing import Any

import numpy as np

EXPECTED_METHODS = ("baseline", "fixed_dcr", "macc_lite", "full_macc")
EXPECTED_SEEDS = (301, 302, 303, 304, 305, 306)
PRIMARY_METRIC = "validation_nll"


def sha256_file(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def holm_adjust(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [1.0] * n
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def exact_sign_flip_p(diffs: np.ndarray) -> float:
    observed = abs(float(np.mean(diffs)))
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(diffs)):
        null.append(abs(float(np.mean(diffs * np.asarray(signs)))))
    return float(np.mean(np.asarray(null) >= observed - 1e-18))


def paired_statistics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from scipy import stats

    by = {(r["method"], int(r["seed"])): r for r in rows}
    metrics = ("validation_nll", "perplexity", "token_ece", "token_accuracy")
    output: list[dict[str, Any]] = []
    for metric in metrics:
        metric_rows: list[dict[str, Any]] = []
        for method in EXPECTED_METHODS[1:]:
            diffs = np.asarray(
                [
                    float(by[(method, seed)][metric])
                    - float(by[("baseline", seed)][metric])
                    for seed in EXPECTED_SEEDS
                ],
                dtype=np.float64,
            )
            n = len(diffs)
            mean = float(np.mean(diffs))
            sd = float(np.std(diffs, ddof=1))
            se = sd / math.sqrt(n)
            tcrit = float(stats.t.ppf(0.975, n - 1))
            t_stat, p_t = stats.ttest_1samp(diffs, popmean=0.0)
            try:
                p_w = float(
                    stats.wilcoxon(
                        diffs,
                        zero_method="wilcox",
                        alternative="two-sided",
                        method="exact",
                    ).pvalue
                )
            except Exception:
                p_w = float("nan")
            metric_rows.append(
                {
                    "metric": metric,
                    "target": method,
                    "reference": "baseline",
                    "n_pairs": n,
                    "target_mean": float(
                        np.mean([by[(method, seed)][metric] for seed in EXPECTED_SEEDS])
                    ),
                    "reference_mean": float(
                        np.mean([by[("baseline", seed)][metric] for seed in EXPECTED_SEEDS])
                    ),
                    "paired_mean_difference": mean,
                    "paired_sd_difference": sd,
                    "paired_t_ci95_low": mean - tcrit * se,
                    "paired_t_ci95_high": mean + tcrit * se,
                    "paired_t_stat": float(t_stat),
                    "paired_t_p": float(p_t),
                    "exact_sign_flip_p": exact_sign_flip_p(diffs),
                    "exact_wilcoxon_p": p_w,
                    "cohens_dz": mean / sd if sd > 0 else float("nan"),
                    "target_lower_count": int(np.sum(diffs < 0)),
                    "target_higher_count": int(np.sum(diffs > 0)),
                }
            )
        adjusted = holm_adjust([float(r["paired_t_p"]) for r in metric_rows])
        for r, p_adj in zip(metric_rows, adjusted):
            r["paired_t_p_holm"] = p_adj
        output.extend(metric_rows)
    return output


def evaluate_model(
    *,
    model: Any,
    tokens: np.ndarray,
    seq_len: int,
    batch_seqs: int,
    device: Any,
    use_amp: bool,
    n_bins: int = 15,
) -> dict[str, float | int]:
    import torch
    import torch.nn.functional as F

    n_seq = (len(tokens) - 1) // seq_len
    if n_seq < 1:
        raise RuntimeError("Holdout token cache does not contain one complete sequence")

    nll_sum = 0.0
    token_count = 0
    correct_count = 0
    bin_count = torch.zeros(n_bins, dtype=torch.float64)
    bin_conf = torch.zeros(n_bins, dtype=torch.float64)
    bin_correct = torch.zeros(n_bins, dtype=torch.float64)

    model.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, n_seq, batch_seqs):
            ids = range(start, min(start + batch_seqs, n_seq))
            block = np.stack(
                [
                    np.asarray(tokens[i * seq_len : i * seq_len + seq_len + 1], dtype=np.int64)
                    for i in ids
                ]
            )
            t = torch.from_numpy(block).to(device, non_blocking=True)
            inp, target = t[:, :-1], t[:, 1:]
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                logits = model(input_ids=inp).logits
            flat_logits = logits.float().reshape(-1, logits.shape[-1])
            flat_target = target.reshape(-1)
            nll_sum += float(
                F.cross_entropy(flat_logits, flat_target, reduction="sum").item()
            )
            max_logits, pred = flat_logits.max(dim=-1)
            conf = torch.exp(max_logits - torch.logsumexp(flat_logits, dim=-1))
            correct = pred.eq(flat_target)
            idx = torch.clamp((conf * n_bins).to(torch.long), 0, n_bins - 1)
            counts = torch.bincount(idx, minlength=n_bins).to(torch.float64).cpu()
            conf_sums = torch.bincount(
                idx, weights=conf.to(torch.float64), minlength=n_bins
            ).cpu()
            correct_sums = torch.bincount(
                idx, weights=correct.to(torch.float64), minlength=n_bins
            ).cpu()
            bin_count += counts
            bin_conf += conf_sums
            bin_correct += correct_sums
            correct_count += int(correct.sum().item())
            token_count += int(flat_target.numel())
            del t, inp, target, logits, flat_logits, flat_target, max_logits, pred, conf, correct, idx

    elapsed = time.perf_counter() - started
    nll = nll_sum / token_count
    valid = bin_count > 0
    ece = float(
        (
            (bin_count[valid] / token_count)
            * (
                bin_conf[valid] / bin_count[valid]
                - bin_correct[valid] / bin_count[valid]
            ).abs()
        ).sum()
    )
    return {
        "validation_nll": nll,
        "perplexity": math.exp(min(nll, 50.0)),
        "token_ece": ece,
        "token_accuracy": correct_count / token_count,
        "n_eval_tokens": token_count,
        "n_eval_sequences": n_seq,
        "evaluation_time_s": elapsed,
        "tokens_per_second": token_count / max(elapsed, 1e-12),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def build_report(output_root: Path, rows: list[dict[str, Any]], paired: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    grouped: list[dict[str, Any]] = []
    for method in EXPECTED_METHODS:
        subset = [r for r in rows if r["method"] == method]
        rec: dict[str, Any] = {"method": method, "seeds": len(subset)}
        for metric in ("validation_nll", "perplexity", "token_ece", "token_accuracy", "evaluation_time_s"):
            vals = np.asarray([float(r[metric]) for r in subset])
            rec[f"{metric}_mean"] = float(vals.mean())
            rec[f"{metric}_std"] = float(vals.std(ddof=1))
        grouped.append(rec)
    write_csv(output_root / "grouped_metrics.csv", grouped)

    labels = {
        "baseline": "Baseline",
        "fixed_dcr": "Fixed DCR",
        "macc_lite": "MACC-Lite",
        "full_macc": "Full MACC",
    }
    lines = [
        "# C4 untouched-shard evaluation",
        "",
        f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        "",
        f"Source campaign: `{meta['source_campaign_root']}`",
        f"Holdout shard: `{meta['holdout_validation_shard']}`",
        f"Evaluated tokens per checkpoint: {rows[0]['n_eval_tokens']:,}",
        "",
        "| Method | Validation NLL | Perplexity | Token ECE | Token accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in grouped:
        lines.append(
            f"| {labels[r['method']]} | "
            f"{r['validation_nll_mean']:.6f} ± {r['validation_nll_std']:.6f} | "
            f"{r['perplexity_mean']:.5f} ± {r['perplexity_std']:.5f} | "
            f"{r['token_ece_mean']:.6f} ± {r['token_ece_std']:.6f} | "
            f"{r['token_accuracy_mean']:.6f} ± {r['token_accuracy_std']:.6f} |"
        )
    lines += ["", "## Paired comparisons against baseline", ""]
    for r in paired:
        if r["metric"] != PRIMARY_METRIC:
            continue
        lines.append(
            f"- **{labels[r['target']]}:** ΔNLL={r['paired_mean_difference']:+.8f}, "
            f"95% paired-t CI [{r['paired_t_ci95_low']:+.8f}, {r['paired_t_ci95_high']:+.8f}], "
            f"Holm-adjusted paired-t p={r['paired_t_p_holm']:.6g}, "
            f"exact sign-flip p={r['exact_sign_flip_p']:.5f}."
        )
    (output_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Evaluation of the six paired C4 checkpoints on an untouched validation shard. Values are mean $\pm$ standard deviation.}",
        r"\label{tab:c4_holdout}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Val. NLL $\downarrow$ & PPL $\downarrow$ & Token ECE $\downarrow$ & Token acc. $\uparrow$ \\",
        r"\midrule",
    ]
    for r in grouped:
        tex.append(
            f"{labels[r['method']]} & "
            f"{r['validation_nll_mean']:.6f} $\\pm$ {r['validation_nll_std']:.6f} & "
            f"{r['perplexity_mean']:.5f} $\\pm$ {r['perplexity_std']:.5f} & "
            f"{r['token_ece_mean']:.6f} $\\pm$ {r['token_ece_std']:.6f} & "
            f"{r['token_accuracy_mean']:.6f} $\\pm$ {r['token_accuracy_std']:.6f} \\\\" 
        )
    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip\footnotesize PPL is $\exp(\mathrm{NLL})$ and is reported descriptively. The preregistered primary comparison is Fixed DCR versus baseline on validation NLL.",
        r"\end{table}",
    ]
    (output_root / "c4_holdout_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def archive_results(output_root: Path, persistent_root: Path) -> tuple[Path, str]:
    archives = persistent_root / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"caenl-c4-confirmatory-v2-holdout-v1-{stamp}.tar.gz"
    archive = archives / name
    def _filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if Path(info.name).name == ".running":
            return None
        return info

    with tarfile.open(archive, "w:gz") as tf:
        tf.add(output_root, arcname=output_root.name, filter=_filter)
    digest = sha256_file(archive)
    (archive.with_suffix(archive.suffix + ".sha256")).write_text(
        f"{digest}  {archive.name}\n", encoding="utf-8"
    )
    pointer = archives / "LATEST_caenl-c4-confirmatory-v2-holdout-v1.txt"
    pointer.write_text(str(archive) + "\n", encoding="utf-8")
    return archive, digest


def self_test() -> None:
    p = holm_adjust([0.01, 0.04, 0.03])
    assert all(0 <= x <= 1 for x in p)
    assert exact_sign_flip_p(np.asarray([-1.0] * 6)) == 0.03125
    rows = []
    for seed in EXPECTED_SEEDS:
        for method, shift in (("baseline", 0.0), ("fixed_dcr", -0.1), ("macc_lite", 0.1), ("full_macc", 0.0)):
            rows.append({
                "method": method,
                "seed": seed,
                "validation_nll": 3.0 + shift,
                "perplexity": math.exp(3.0 + shift),
                "token_ece": 0.02 + shift / 100,
                "token_accuracy": 0.4 - shift / 100,
            })
    out = paired_statistics(rows)
    assert len(out) == 12
    print("SELF-TEST PASS")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, default=Path.home() / "caenl-v5.4.3" / "caenl_revision_pipeline_v5.4.3_c4_confirmatory")
    ap.add_argument("--campaign-root", type=Path, default=Path("/mnt/caenl/active/results/caenl-c4-confirmatory-v2"))
    ap.add_argument("--output-root", type=Path, default=Path("/mnt/caenl/active/results/caenl-c4-confirmatory-v2-holdout-v1"))
    ap.add_argument("--data-root", type=Path, default=Path("/mnt/caenl/active/data"))
    ap.add_argument("--cache-root", type=Path, default=Path("/mnt/caenl/active/cache"))
    ap.add_argument("--persistent-root", type=Path, default=Path("/mnt/caenl/persistent"))
    ap.add_argument("--validation-shard", default="en/c4-validation.00002-of-00008.json.gz")
    ap.add_argument("--validation-tokens", type=int, default=2_000_000)
    ap.add_argument("--batch-seqs", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    project = args.project.resolve()
    if not (project / "src" / "caenl" / "language" / "task.py").is_file():
        raise SystemExit(f"ERROR: v5.4.3 project not found: {project}")
    sys.path.insert(0, str(project / "src"))

    import torch
    from caenl.language.data import load_token_cache, prepare_c4
    from caenl.language.task import _build_model

    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA is unavailable")
    device = torch.device(args.device)
    if "L40S" not in torch.cuda.get_device_name(device):
        print(f"WARNING: expected L40S, found {torch.cuda.get_device_name(device)}", flush=True)

    campaign = args.campaign_root.resolve()
    jobs_root = campaign / "stages" / "c4_confirmatory" / "jobs"
    if not jobs_root.is_dir():
        raise SystemExit(f"ERROR: source campaign jobs not found: {jobs_root}")

    expected: list[tuple[str, int, Path, dict[str, Any]]] = []
    for seed in EXPECTED_SEEDS:
        for method in EXPECTED_METHODS:
            job_dir = jobs_root / f"{method}__seed{seed}"
            summary_file = job_dir / "summary.json"
            checkpoint = job_dir / "checkpoints" / "state.pt"
            if not summary_file.is_file():
                raise SystemExit(f"ERROR: missing summary: {summary_file}")
            if not checkpoint.is_file():
                raise SystemExit(f"ERROR: missing checkpoint: {checkpoint}")
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            if summary.get("method") != method or int(summary.get("seed", -1)) != seed:
                raise SystemExit(f"ERROR: source summary identity mismatch: {summary_file}")
            used_validation = set(summary.get("data", {}).get("validation_shards", []))
            if args.validation_shard in used_validation:
                raise SystemExit(
                    f"ERROR: requested holdout shard was already used by {method}/seed{seed}: "
                    f"{args.validation_shard}"
                )
            if int(summary.get("steps", -1)) != 12207 or int(summary.get("tokens_trained", -1)) != 99_999_744:
                raise SystemExit(f"ERROR: incomplete source job: {job_dir}")
            expected.append((method, seed, checkpoint, summary))

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".running"
    if lock.exists():
        raise SystemExit(f"ERROR: holdout lock already exists: {lock}")
    lock.write_text(f"pid={os.getpid()} time={dt.datetime.now(dt.timezone.utc).isoformat()}\n")

    try:
        print(f"Preparing untouched C4 shard: {args.validation_shard}", flush=True)
        info = prepare_c4(
            args.data_root,
            args.cache_root,
            {
                "tokenizer": "gpt2",
                "train_shards": ["en/c4-train.00000-of-01024.json.gz"],
                "validation_shards": [args.validation_shard],
                "sequence_length": 1024,
                "train_tokens": 2048,
                "validation_tokens": args.validation_tokens,
                "train_token_margin": 1.0,
            },
        )
        _, val_tokens = load_token_cache(info)
        if args.validation_shard in {"en/c4-validation.00000-of-00008.json.gz", "en/c4-validation.00001-of-00008.json.gz"}:
            raise SystemExit("ERROR: selected shard was referenced by the previous campaign preparation; use shard 00002 or later")

        first_summary = expected[0][3]
        seq_len = int(first_summary["config"]["language"]["sequence_length"])
        model = _build_model(
            "gpt2",
            "pretrained",
            int(info["vocab_size"]),
            seq_len,
            device,
            {},
        )
        model.eval()

        rows: list[dict[str, Any]] = []
        result_dir = output / "per_checkpoint"
        result_dir.mkdir(parents=True, exist_ok=True)
        for index, (method, seed, checkpoint, source_summary) in enumerate(expected, start=1):
            result_file = result_dir / f"{method}__seed{seed}.json"
            checkpoint_size = checkpoint.stat().st_size
            checkpoint_sha = sha256_file(checkpoint)
            if result_file.is_file():
                old = json.loads(result_file.read_text(encoding="utf-8"))
                if old.get("checkpoint_sha256") == checkpoint_sha and old.get("holdout_validation_shard") == args.validation_shard:
                    print(f"[{index:02d}/24] SKIP {method}/seed{seed}: verified existing result", flush=True)
                    rows.append(old)
                    continue
            print(f"[{index:02d}/24] EVAL {method}/seed{seed}", flush=True)
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if "model" not in state:
                raise RuntimeError(f"Checkpoint has no model state: {checkpoint}")
            model.load_state_dict(state["model"], strict=True)
            del state
            gc.collect()
            torch.cuda.empty_cache()
            metrics = evaluate_model(
                model=model,
                tokens=val_tokens,
                seq_len=seq_len,
                batch_seqs=args.batch_seqs,
                device=device,
                use_amp=True,
            )
            row = {
                "method": method,
                "seed": seed,
                **metrics,
                "holdout_validation_shard": args.validation_shard,
                "holdout_cache_dir": info["cache_dir"],
                "checkpoint": str(checkpoint),
                "checkpoint_size_bytes": checkpoint_size,
                "checkpoint_sha256": checkpoint_sha,
                "source_validation_shards": source_summary["data"]["validation_shards"],
                "source_validation_nll": source_summary["final"]["validation_nll"],
                "source_perplexity": source_summary["final"]["perplexity"],
                "source_token_ece": source_summary["final"]["token_ece"],
                "evaluated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            atomic_json(result_file, row)
            rows.append(row)
            print(
                f"          NLL={row['validation_nll']:.6f} PPL={row['perplexity']:.5f} "
                f"ECE={row['token_ece']:.6f} ({row['evaluation_time_s']:.1f}s)",
                flush=True,
            )

        rows.sort(key=lambda r: (int(r["seed"]), EXPECTED_METHODS.index(r["method"])))
        write_csv(output / "per_seed_metrics.csv", rows)
        paired = paired_statistics(rows)
        write_csv(output / "paired_comparisons.csv", paired)
        meta = {
            "source_campaign_root": str(campaign),
            "source_plan_sha256": sha256_file(campaign / "PLAN.yaml"),
            "source_publication_check_sha256": sha256_file(campaign / "PUBLICATION_CHECK.json"),
            "holdout_validation_shard": args.validation_shard,
            "holdout_info": info,
            "expected_methods": list(EXPECTED_METHODS),
            "expected_seeds": list(EXPECTED_SEEDS),
            "primary_metric": PRIMARY_METRIC,
            "primary_comparison": "fixed_dcr versus baseline",
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(device),
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        atomic_json(output / "METADATA.json", meta)
        build_report(output, rows, paired, meta)
        shutil.copy2(Path(__file__).resolve(), output / "run_c4_holdout_eval.py")
        checks = []
        for f in sorted(output.rglob("*")):
            if f.is_file() and f.name not in {"SHA256SUMS", ".running"}:
                checks.append(f"{sha256_file(f)}  {f.relative_to(output)}")
        (output / "SHA256SUMS").write_text("\n".join(checks) + "\n", encoding="utf-8")
        atomic_json(output / "STATUS.json", {"ok": True, "jobs": len(rows), "holdout_shard": args.validation_shard})
        archive, digest = archive_results(output, args.persistent_root.resolve())
        print("\nC4 HOLDOUT EVALUATION PASS", flush=True)
        print(f"Results:  {output}", flush=True)
        print(f"Archive:  {archive}", flush=True)
        print(f"SHA-256:  {digest}", flush=True)
        return 0
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
