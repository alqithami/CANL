#!/usr/bin/env python3
"""
Controlled C4/GPT-2 computational-overhead benchmark for CÆNL.

This benchmark measures five training paths on the same fixed token stream:
  1. plain: no hidden-state capture and no spectral monitor
  2. monitor_only: hidden-state capture + spectral monitoring
The missing measurement is the cost of hidden-state capture and spectral
monitoring relative to ordinary GPT-2 training. The already completed six-seed
confirmatory campaign supplies the incremental timings for Fixed DCR,
MACC-Lite, and Full MACC relative to its monitored baseline.

It is a timing benchmark, not a task-performance experiment. Model loading,
initialization, warm-up, report generation, and archive creation are excluded
from timed training intervals.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


MODES = ("plain", "monitor_only")
TARGET_RANKS = {6: 16, 12: 4}


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def run_text(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        return f"<command failed: {type(exc).__name__}: {exc}>"


def normalized_spectrum(
    hidden: torch.Tensor,
    *,
    sample_tokens: int,
    max_rank: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized eigenvalues and effective rank.

    hidden: [batch, sequence, hidden_dim]. The operation remains differentiable.
    """
    x = hidden.reshape(-1, hidden.shape[-1]).float()
    n = x.shape[0]
    if n > sample_tokens:
        g = torch.Generator(device=x.device)
        g.manual_seed(seed)
        idx = torch.randperm(n, generator=g, device=x.device)[:sample_tokens]
        x = x.index_select(0, idx)
    x = x - x.mean(dim=0, keepdim=True)

    # The token Gram matrix is substantially cheaper than a 768x768 covariance.
    denom = max(1, x.shape[0] - 1)
    gram = (x @ x.transpose(0, 1)) / float(denom)
    eig = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
    eig = eig[: min(max_rank, eig.numel())]
    p = eig / eig.sum().clamp_min(1e-12)
    entropy = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum()
    effective_rank = entropy.exp()
    return p, effective_rank


def target_spectrum(length: int, rank: int, device: torch.device) -> torch.Tensor:
    r = max(1, min(rank, length))
    target = torch.zeros(length, dtype=torch.float32, device=device)
    target[:r] = 1.0 / float(r)
    return target


def spectrum_loss(
    hidden_states: tuple[torch.Tensor, ...],
    *,
    lambdas: dict[int, float],
    sample_tokens: int,
    max_rank: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses: list[torch.Tensor] = []
    metrics: dict[str, float] = {}
    for layer in (6, 12):
        p, erank = normalized_spectrum(
            hidden_states[layer],
            sample_tokens=sample_tokens,
            max_rank=max_rank,
            seed=seed + layer,
        )
        target = target_spectrum(p.numel(), TARGET_RANKS[layer], p.device)
        raw = torch.mean((p - target) ** 2)
        losses.append(float(lambdas[layer]) * raw)
        metrics[f"h{layer}_effective_rank"] = float(erank.detach().cpu())
        metrics[f"h{layer}_spectrum_loss"] = float(raw.detach().cpu())
    return torch.stack(losses).sum(), metrics


@torch.no_grad()
def monitor_metrics(
    hidden_states: tuple[torch.Tensor, ...],
    *,
    sample_tokens: int,
    max_rank: int,
    seed: int,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for layer in (6, 12):
        p, erank = normalized_spectrum(
            hidden_states[layer],
            sample_tokens=sample_tokens,
            max_rank=max_rank,
            seed=seed + layer,
        )
        target = target_spectrum(p.numel(), TARGET_RANKS[layer], p.device)
        result[f"h{layer}_effective_rank"] = float(erank.cpu())
        result[f"h{layer}_spectrum_loss"] = float(
            torch.mean((p - target) ** 2).cpu()
        )
    return result


class FullMACCMicroPolicy(nn.Module):
    """Small policy used only to include learned-controller compute in timing."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.h6 = nn.Linear(64, 3)
        self.h12 = nn.Linear(64, 3)

    def distributions(self, state: torch.Tensor) -> tuple[torch.distributions.Categorical, ...]:
        z = self.net(state)
        return (
            torch.distributions.Categorical(logits=self.h6(z)),
            torch.distributions.Categorical(logits=self.h12(z)),
        )


def find_c4_shard(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"C4 shard not found: {path}")
        return path

    preferred = [
        Path("/mnt/caenl/active/data"),
        Path("/mnt/caenl/active/cache"),
        Path("/mnt/caenl/persistent"),
    ]
    names = (
        "c4-train.00000-of-01024.json.gz",
        "c4-validation.00000-of-00008.json.gz",
    )
    for base in preferred:
        if not base.exists():
            continue
        for name in names:
            matches = list(base.rglob(name))
            if matches:
                return matches[0].resolve()
    raise FileNotFoundError(
        "No local C4 shard was found under /mnt/caenl. "
        "Preserve the completed C4 cache or pass --c4-shard explicitly."
    )


def make_c4_batches(
    *,
    shard: Path,
    tokenizer: Any,
    count: int,
    micro_batch: int,
    seq_len: int,
) -> list[torch.Tensor]:
    """Build a deterministic fixed batch stream from a local C4 JSONL gzip shard."""
    needed_sequences = count * micro_batch
    needed_tokens = needed_sequences * seq_len
    token_buffer: list[int] = []
    eos = tokenizer.eos_token_id
    docs = 0

    with gzip.open(shard, "rt", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            ids = tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=False,
            )
            token_buffer.extend(ids)
            if eos is not None:
                token_buffer.append(int(eos))
            docs += 1
            if len(token_buffer) >= needed_tokens:
                break

    if len(token_buffer) < needed_tokens:
        raise RuntimeError(
            f"C4 shard supplied only {len(token_buffer):,} usable tokens; "
            f"{needed_tokens:,} are required"
        )

    token_array = torch.tensor(
        token_buffer[:needed_tokens],
        dtype=torch.long,
    ).view(needed_sequences, seq_len)

    batches = [
        token_array[i * micro_batch : (i + 1) * micro_batch].contiguous()
        for i in range(count)
    ]
    print(
        f"Prepared {needed_tokens:,} fixed C4 tokens from {docs:,} documents "
        f"in {shard}",
        flush=True,
    )
    return batches


def build_model(model_name: str, cache_dir: Path, device: torch.device) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    model.train()
    model.config.use_cache = False
    return model


def one_training_interval(
    *,
    mode: str,
    model_name: str,
    cache_dir: Path,
    device: torch.device,
    token_batches: list[torch.Tensor],
    warmup_steps: int,
    measured_steps: int,
    grad_accum: int,
    lr: float,
    monitor_cadence: int,
    sample_tokens: int,
    max_rank: int,
    repeat: int,
) -> dict[str, Any]:
    model = build_model(model_name, cache_dir, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    policy: FullMACCMicroPolicy | None = None
    policy_optimizer: torch.optim.Optimizer | None = None
    pending_log_prob: torch.Tensor | None = None
    reward_baseline = 0.0
    previous_val: float | None = None

    if mode == "full_macc":
        policy = FullMACCMicroPolicy().to(device)
        policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)

    lambdas = {6: 0.010, 12: 0.010}
    lambda_min, lambda_max = 0.0, 0.08
    eta_lambda = 0.002
    last_metrics = {
        "h6_effective_rank": float(TARGET_RANKS[6]),
        "h12_effective_rank": float(TARGET_RANKS[12]),
        "h6_spectrum_loss": 0.0,
        "h12_spectrum_loss": 0.0,
    }
    monitor_events = 0
    policy_updates = 0

    total_steps = warmup_steps + measured_steps
    micro_needed = total_steps * grad_accum + 8
    if len(token_batches) < micro_needed:
        raise ValueError("not enough fixed token batches")

    def validation_loss(batch: torch.Tensor) -> float:
        model.eval()
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            x = batch.to(device, non_blocking=True)
            out = model(input_ids=x, labels=x, output_hidden_states=False)
            value = float(out.loss.detach().float().cpu())
        model.train()
        return value

    def run_steps(n_steps: int, start_step: int, timed: bool) -> tuple[float, int]:
        nonlocal pending_log_prob, reward_baseline, previous_val
        nonlocal monitor_events, policy_updates, lambdas, last_metrics

        if timed:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            start_time = time.perf_counter()
        else:
            start_time = 0.0

        micro_index = start_step * grad_accum
        tokens = 0

        for local_step in range(n_steps):
            global_step = start_step + local_step
            optimizer.zero_grad(set_to_none=True)
            last_hidden: tuple[torch.Tensor, ...] | None = None

            for accum in range(grad_accum):
                batch = token_batches[micro_index].to(device, non_blocking=True)
                micro_index += 1
                need_hidden = mode != "plain"

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    out = model(
                        input_ids=batch,
                        labels=batch,
                        output_hidden_states=need_hidden,
                    )
                    loss = out.loss / float(grad_accum)

                    if mode in ("fixed_dcr", "macc_lite", "full_macc"):
                        assert out.hidden_states is not None
                        dcr, metrics = spectrum_loss(
                            out.hidden_states,
                            lambdas=lambdas,
                            sample_tokens=sample_tokens,
                            max_rank=max_rank,
                            seed=100000 * repeat + 1000 * global_step + accum,
                        )
                        loss = loss + dcr / float(grad_accum)
                        last_metrics = metrics

                loss.backward()
                if need_hidden:
                    assert out.hidden_states is not None
                    last_hidden = tuple(h.detach() for h in out.hidden_states)
                tokens += int(batch.numel())

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            event = ((global_step + 1) % monitor_cadence == 0)
            if event and mode != "plain":
                assert last_hidden is not None
                last_metrics = monitor_metrics(
                    last_hidden,
                    sample_tokens=sample_tokens,
                    max_rank=max_rank,
                    seed=200000 * repeat + global_step,
                )
                monitor_events += 1

                if mode == "macc_lite":
                    # Projected feedback. This is intentionally lightweight; the
                    # benchmark measures controller overhead, not controller quality.
                    for layer in (6, 12):
                        err = (
                            last_metrics[f"h{layer}_effective_rank"]
                            - float(TARGET_RANKS[layer])
                        ) / float(TARGET_RANKS[layer])
                        lambdas[layer] = float(
                            np.clip(
                                lambdas[layer] + eta_lambda * err,
                                lambda_min,
                                lambda_max,
                            )
                        )

                elif mode == "full_macc":
                    assert policy is not None and policy_optimizer is not None
                    val_batch = token_batches[micro_index]
                    current_val = validation_loss(val_batch)

                    target_error = (
                        abs(last_metrics["h6_effective_rank"] - TARGET_RANKS[6])
                        / TARGET_RANKS[6]
                        + abs(last_metrics["h12_effective_rank"] - TARGET_RANKS[12])
                        / TARGET_RANKS[12]
                    )

                    if previous_val is not None and pending_log_prob is not None:
                        reward = (previous_val - current_val) - 0.001 * target_error
                        advantage = reward - reward_baseline
                        reward_baseline = 0.95 * reward_baseline + 0.05 * reward

                        policy_optimizer.zero_grad(set_to_none=True)
                        policy_loss = -(pending_log_prob * float(advantage))
                        policy_loss.backward()
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                        policy_optimizer.step()
                        policy_updates += 1

                    state = torch.tensor(
                        [
                            last_metrics["h6_effective_rank"] / 128.0,
                            last_metrics["h12_effective_rank"] / 128.0,
                            last_metrics["h6_spectrum_loss"],
                            last_metrics["h12_spectrum_loss"],
                            (global_step + 1) / max(1, total_steps),
                        ],
                        dtype=torch.float32,
                        device=device,
                    )
                    d6, d12 = policy.distributions(state)
                    a6 = d6.sample()
                    a12 = d12.sample()
                    pending_log_prob = d6.log_prob(a6) + d12.log_prob(a12)
                    deltas = (-0.002, 0.0, 0.002)
                    lambdas[6] = float(
                        np.clip(lambdas[6] + deltas[int(a6.item())], lambda_min, lambda_max)
                    )
                    lambdas[12] = float(
                        np.clip(lambdas[12] + deltas[int(a12.item())], lambda_min, lambda_max)
                    )
                    previous_val = current_val

        if timed:
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start_time
        else:
            elapsed = 0.0
        return elapsed, tokens

    # Untimed warm-up, then restore fresh model/optimizer so timed modes begin
    # from the same pretrained weights and allocator state.
    if warmup_steps > 0:
        run_steps(warmup_steps, 0, timed=False)
        del model, optimizer, policy, policy_optimizer
        torch.cuda.empty_cache()
        model = build_model(model_name, cache_dir, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        policy = None
        policy_optimizer = None
        pending_log_prob = None
        reward_baseline = 0.0
        previous_val = None
        lambdas = {6: 0.010, 12: 0.010}
        last_metrics = {
            "h6_effective_rank": float(TARGET_RANKS[6]),
            "h12_effective_rank": float(TARGET_RANKS[12]),
            "h6_spectrum_loss": 0.0,
            "h12_spectrum_loss": 0.0,
        }
        monitor_events = 0
        policy_updates = 0
        if mode == "full_macc":
            policy = FullMACCMicroPolicy().to(device)
            policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)

    elapsed, tokens = run_steps(measured_steps, warmup_steps, timed=True)
    peak = int(torch.cuda.max_memory_allocated(device))
    reserved = int(torch.cuda.max_memory_reserved(device))

    result = {
        "mode": mode,
        "repeat": repeat,
        "measured_optimizer_steps": measured_steps,
        "tokens": tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens / elapsed,
        "milliseconds_per_step": 1000.0 * elapsed / measured_steps,
        "peak_allocated_gib": peak / 1024**3,
        "peak_reserved_gib": reserved / 1024**3,
        "monitor_events": monitor_events,
        "policy_updates": policy_updates,
        "lambda_h6_final": lambdas[6],
        "lambda_h12_final": lambdas[12],
    }

    del model, optimizer, policy, policy_optimizer
    torch.cuda.empty_cache()
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="/mnt/caenl/active/results/caenl-c4-overhead-v2",
    )
    parser.add_argument(
        "--persistent-root",
        default="/mnt/caenl/persistent",
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--c4-shard", default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--measure-steps", type=int, default=250)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--monitor-cadence", type=int, default=25)
    parser.add_argument("--sample-tokens", type=int, default=128)
    parser.add_argument("--max-rank", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA is unavailable")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    if "L40S" not in gpu_name:
        print(f"WARNING: benchmark expected L40S, found {gpu_name}", flush=True)

    output_root = Path(args.output_root).resolve()
    persistent_root = Path(args.persistent_root).resolve()
    cache_dir = persistent_root / "model-cache" / "huggingface"
    archive_dir = persistent_root / "archives"
    log_dir = persistent_root / "logs"
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    running_marker = output_root / "RUNNING"
    complete_marker = output_root / "COMPLETE"
    failed_marker = output_root / "FAILED"
    for p in (complete_marker, failed_marker):
        p.unlink(missing_ok=True)
    running_marker.write_text(f"{utc_stamp()}\n", encoding="utf-8")

    config = vars(args).copy()
    config.update(
        {
            "effective_batch_tokens": (
                args.seq_len * args.micro_batch * args.grad_accum
            ),
            "modes": list(MODES),
            "gpu": gpu_name,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "benchmark_scope": (
                "Controlled compute benchmark; model/data initialization and "
                "untimed warm-up excluded."
            ),
        }
    )
    (output_root / "config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "nvidia-smi.txt").write_text(
        run_text(["nvidia-smi"]),
        encoding="utf-8",
    )
    (output_root / "pip-freeze.txt").write_text(
        run_text([sys.executable, "-m", "pip", "freeze"]),
        encoding="utf-8",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=str(cache_dir),
    )
    total_micro = (
        (args.warmup_steps + args.measure_steps) * args.grad_accum + 8
    )
    c4_shard = find_c4_shard(args.c4_shard)
    config["c4_shard"] = str(c4_shard)
    config["c4_shard_size_bytes"] = c4_shard.stat().st_size
    (output_root / "config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    batches = make_c4_batches(
        shard=c4_shard,
        tokenizer=tokenizer,
        count=total_micro,
        micro_batch=args.micro_batch,
        seq_len=args.seq_len,
    )

    order_rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []

    try:
        for repeat in range(1, args.repeats + 1):
            order = list(MODES)
            order_rng.shuffle(order)
            print(
                f"[repeat {repeat}/{args.repeats}] order={','.join(order)}",
                flush=True,
            )
            for mode in order:
                print(f"  START {mode}", flush=True)
                result = one_training_interval(
                    mode=mode,
                    model_name=args.model,
                    cache_dir=cache_dir,
                    device=device,
                    token_batches=batches,
                    warmup_steps=args.warmup_steps,
                    measured_steps=args.measure_steps,
                    grad_accum=args.grad_accum,
                    lr=args.lr,
                    monitor_cadence=args.monitor_cadence,
                    sample_tokens=args.sample_tokens,
                    max_rank=args.max_rank,
                    repeat=repeat,
                )
                rows.append(result)
                print(
                    f"  DONE {mode}: {result['tokens_per_second']:.1f} tok/s, "
                    f"{result['peak_allocated_gib']:.2f} GiB",
                    flush=True,
                )
                write_csv(
                    output_root / "raw_runs.csv",
                    rows,
                    list(rows[0].keys()),
                )

        # Grouped descriptive results.
        plain = np.array(
            [r["elapsed_seconds"] for r in rows if r["mode"] == "plain"],
            dtype=float,
        )
        grouped: list[dict[str, Any]] = []
        for mode in MODES:
            subset = [r for r in rows if r["mode"] == mode]
            elapsed = np.array([r["elapsed_seconds"] for r in subset])
            tps = np.array([r["tokens_per_second"] for r in subset])
            peak = np.array([r["peak_allocated_gib"] for r in subset])
            paired_overhead = np.array(
                [
                    100.0
                    * (
                        next(
                            r["elapsed_seconds"]
                            for r in rows
                            if r["mode"] == mode and r["repeat"] == rep
                        )
                        / next(
                            r["elapsed_seconds"]
                            for r in rows
                            if r["mode"] == "plain" and r["repeat"] == rep
                        )
                        - 1.0
                    )
                    for rep in range(1, args.repeats + 1)
                ]
            )
            grouped.append(
                {
                    "mode": mode,
                    "repeats": len(subset),
                    "elapsed_mean_s": float(elapsed.mean()),
                    "elapsed_std_s": float(elapsed.std(ddof=1)),
                    "tokens_per_second_mean": float(tps.mean()),
                    "tokens_per_second_std": float(tps.std(ddof=1)),
                    "peak_allocated_gib_mean": float(peak.mean()),
                    "peak_allocated_gib_std": float(peak.std(ddof=1)),
                    "overhead_vs_plain_mean_pct": float(paired_overhead.mean()),
                    "overhead_vs_plain_std_pct": float(
                        paired_overhead.std(ddof=1)
                    ),
                }
            )

        write_csv(
            output_root / "grouped_metrics.csv",
            grouped,
            list(grouped[0].keys()),
        )

        # LaTeX table.
        tex = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{Controlled GPT-2/C4 computational overhead on one "
            r"NVIDIA L40S. Model initialization and warm-up are excluded. "
            r"Values are mean $\pm$ sample standard deviation over paired repeats.}",
            r"\label{tab:c4_overhead}",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Mode & Throughput (tok/s) $\uparrow$ & Peak memory (GiB) "
            r"& Overhead vs. plain $\downarrow$ \\",
            r"\midrule",
        ]
        display = {
            "plain": "Plain GPT-2",
            "monitor_only": "Monitoring only",
            "fixed_dcr": "Fixed DCR",
            "macc_lite": "MACC-Lite",
            "full_macc": "Full MACC",
        }
        for row in grouped:
            tex.append(
                f"{display[row['mode']]} & "
                f"{row['tokens_per_second_mean']:.1f} $\\pm$ "
                f"{row['tokens_per_second_std']:.1f} & "
                f"{row['peak_allocated_gib_mean']:.2f} $\\pm$ "
                f"{row['peak_allocated_gib_std']:.2f} & "
                f"{row['overhead_vs_plain_mean_pct']:+.2f}\\% $\\pm$ "
                f"{row['overhead_vs_plain_std_pct']:.2f}\\% \\\\"
            )
        tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (output_root / "c4_overhead_table.tex").write_text(
            "\n".join(tex) + "\n",
            encoding="utf-8",
        )

        summary = {
            "status": "complete",
            "completed_at": utc_stamp(),
            "config": config,
            "grouped": grouped,
            "notes": [
                "This is a controlled timing benchmark, not a performance run.",
                "Both modes use the same fixed C4 token tensors and effective batch.",
                "Mode order is randomized independently within each paired repeat.",
                "The measured interval excludes model loading and warm-up.",
            ],
        }
        (output_root / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )

        # Hash all retained result files before archiving.
        manifest_lines = []
        for path in sorted(output_root.rglob("*")):
            if path.is_file() and path.name not in {"SHA256SUMS", "RUNNING"}:
                manifest_lines.append(
                    f"{sha256_file(path)}  {path.relative_to(output_root)}"
                )
        (output_root / "SHA256SUMS").write_text(
            "\n".join(manifest_lines) + "\n",
            encoding="utf-8",
        )
        running_marker.unlink(missing_ok=True)
        complete_marker.write_text(f"{utc_stamp()}\n", encoding="utf-8")

        stamp = utc_stamp()
        archive = archive_dir / f"caenl-c4-overhead-v2-{stamp}.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(output_root, arcname=output_root.name)
        digest = sha256_file(archive)
        checksum = Path(str(archive) + ".sha256")
        checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
        pointer = archive_dir / "LATEST_caenl-c4-overhead-v2.txt"
        pointer.write_text(str(archive) + "\n", encoding="utf-8")

        print("C4 OVERHEAD BENCHMARK V2 PASS", flush=True)
        print(f"Archive:  {archive}", flush=True)
        print(f"Checksum: {checksum}", flush=True)
        return 0

    except Exception as exc:
        failed_marker.write_text(
            f"{utc_stamp()}\n{type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        running_marker.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
