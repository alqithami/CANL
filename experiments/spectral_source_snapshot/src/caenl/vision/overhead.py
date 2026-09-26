"""Computational/memory overhead benchmark (reviewer request).

Runs the *same* training-step code path as the active-learning jobs (``Trainer``) under
several configurations and reports median step time, throughput and peak memory, plus the
wall time of each acquisition strategy on a fixed candidate pool.

Configurations:

* ``baseline``    task loss only, no monitoring
* ``monitoring``  + EMA class-conditional monitoring every step (kappa, instability)
* ``fixed_dcr``   + spectrum DCR with fixed lambda
* ``macc_lite``   + MACC-Lite feedback controller (cadence N)
* ``full_macc``   + Full MACC contextual bandit (policy network, controller-validation evaluation)
* ``at_pgd10``    PGD-10 adversarial training (cost reference for the robustness discussion)
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..core.controllers import build_controller, kappa_target_from_strength
from ..core.dcr import DCR
from ..core.monitor import ClassConditionalMonitor
from ..utils.io import read_jsonl
from ..utils.jobctx import JobContext
from ..utils.seeding import derive_seed
from . import acquisition as acq
from .data import DATASET_INFO, DeviceArray, load_vision_dataset
from .models import build_model, count_parameters
from .train import Trainer

CONFIGS = {
    "baseline": {"regularizer": "none", "monitor": False, "adv": False},
    "monitoring": {"regularizer": "none", "monitor": True, "adv": False},
    "fixed_dcr": {"regularizer": "fixed_dcr", "monitor": True, "adv": False},
    "macc_lite": {"regularizer": "macc_lite", "monitor": True, "adv": False},
    "full_macc": {"regularizer": "full_macc", "monitor": True, "adv": False},
    "at_pgd10": {"regularizer": "none", "monitor": False, "adv": True},
}


def _random_data(n: int, res: int, num_classes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 256, size=(n, res, res, 3), dtype=np.uint8)
    y = rng.integers(0, num_classes, size=n).astype(np.int64)
    return x, y


def run(ctx: JobContext) -> dict[str, Any]:
    cfg = ctx.config
    oc = dict(cfg.get("overhead", {}))
    steps = int(oc.get("steps", 100))
    warmup = int(oc.get("warmup_steps", 20))
    repeats = int(oc.get("repeats", 3))
    bs = int(oc.get("batch_size", cfg.get("training", {}).get("batch_size", 256)))
    configs = list(oc.get("configs", list(CONFIGS)))
    dataset = cfg.get("dataset", "cifar10")
    info = DATASET_INFO[dataset]
    num_classes = int(cfg.get("dataset_options", {}).get("num_classes", info["num_classes"]))
    res = int(cfg.get("dataset_options", {}).get("train_resolution", info["train_res"]))
    cache_res = int(cfg.get("dataset_options", {}).get("cache_resolution", info.get("cache_res", info["train_res"])))
    n = bs * (steps + warmup)
    arch_opts = dict(cfg.get("arch_options", {}))
    arch_opts.setdefault("image_size", res)
    arch_opts.setdefault("hf_cache", str(Path(cfg["paths"].get("data_root", "data")) / "hf"))
    use_real = bool(oc.get("use_real_data", False))
    if use_real:
        ds = load_vision_dataset(dataset, cfg["paths"]["data_root"], cfg["paths"]["cache_root"], cfg.get("dataset_options", {}))
        idx = np.random.default_rng(0).permutation(len(ds.train_y))[:n]
        x, y = np.ascontiguousarray(ds.train_x[idx]), ds.train_y[idx]
        mean, std, aug = ds.mean, ds.std, ds.aug
    else:
        x, y = _random_data(n, cache_res, num_classes, 0)
        mean, std, aug = info["mean"], info["std"], info["aug"]
    data = DeviceArray(x, ctx.device)
    ctx.event("benchmark_setup", arch=cfg["arch"], dataset=dataset, n=n, batch_size=bs, steps=steps, warmup=warmup, configs=configs, real_data=use_real)
    results: list[dict[str, Any]] = []
    for name in configs:
        spec = CONFIGS[name]
        for rep in range(repeats):
            torch.manual_seed(derive_seed(ctx.seed, "init"))
            model = build_model(cfg["arch"], num_classes, mean, std, arch_opts).to(ctx.device)
            dims = model.feature_dims
            layers = [l for l in cfg.get("collapse", {}).get("monitored_layers", ["layer3", "layer4"]) if l in dims]
            layer_dims = {l: int(dims[l]) for l in layers}
            ccfg = cfg.get("collapse", {})
            monitor = ClassConditionalMonitor(layer_dims, num_classes, float(ccfg.get("covariance_ema_decay", 0.99)), float(ccfg.get("covariance_shrinkage_epsilon", 1e-5)), device=ctx.device)
            dcr = DCR.build(layer_dims, k=int(ccfg.get("truncated_spectral_rank", 128)), num_classes=num_classes, covariance=str(ccfg.get("dcr_covariance", "total")), device=ctx.device)
            tcs = ccfg.get("target_collapse_strength", 0.5)
            targets = {l: kappa_target_from_strength(float(tcs.get(l, 0.5) if isinstance(tcs, dict) else tcs)) for l in layers}
            ctrl_cfg = {"macc_lite": cfg.get("macc_lite", {}), "full_macc": cfg.get("full_macc", {}), "threshold_quantile": 0.9}
            controller = build_controller(spec["regularizer"], layers, targets, ctrl_cfg, total_controller_steps=max(1, steps // 10), device=ctx.device, seed=ctx.seed, state_extra=1, lambda_value=0.03)
            job_cfg = copy.deepcopy(dict(cfg))
            job_cfg.setdefault("training", {})["batch_size"] = bs
            job_cfg["training"]["log_every_steps"] = 10**9
            job_cfg.setdefault("collapse", {})["monitor_update_every_steps"] = 1 if spec["monitor"] else 10**9
            job_cfg["method"] = {"name": name, "regularizer": spec["regularizer"], "adversarial_training": spec["adv"]}
            sub = _SubContext(ctx, job_cfg)
            ctrl_val = np.arange(min(512, n)) if spec["regularizer"] == "full_macc" else None
            trainer = Trainer(sub, model, data, y, num_classes, res, aug, monitor, dcr, controller, spec["regularizer"], ctrl_val)
            trainer.total_steps_planned = steps + warmup
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            # warm-up phase (cudnn autotuning, allocator)
            trainer.train_phase(np.arange(bs * warmup), 1, 0.01, f"{name}-warmup{rep}", 0, 0.1, warmup_epochs=0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            stats = trainer.train_phase(np.arange(bs * warmup, n), 1, 0.01, f"{name}-timed{rep}", 0, 0.1, warmup_epochs=0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            extra = int(getattr(controller, "num_parameters", 0))
            row = {"config": name, "repeat": rep, "arch": cfg["arch"], "dataset": dataset, "batch_size": bs, "resolution": res, "steps": steps, "step_time_ms": 1000 * dt / steps, "images_per_s": bs * steps / dt, "peak_gpu_mem_gb": mem, "extra_params": extra, "model_params": count_parameters(model), "phase_images_per_s": stats.get("images_per_s")}
            results.append(row)
            ctx.event("benchmark_config", **{k: v for k, v in row.items() if k not in ("phase_images_per_s",)})
            del trainer, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    # acquisition benchmark
    ab = dict(oc.get("acquisition_bench", {}))
    acq_results: list[dict[str, Any]] = []
    if ab.get("enabled", True):
        n_c = int(ab.get("n_candidates", 10000))
        budget = int(ab.get("budget", max(1, n_c // 10)))
        strategies = list(ab.get("strategies", ["random", "entropy", "margin", "power_margin", "coreset", "badge", "bald_mcd", "noise_stability", "collapse"]))
        torch.manual_seed(derive_seed(ctx.seed, "init"))
        model = build_model(cfg["arch"], num_classes, mean, std, arch_opts).to(ctx.device).eval()
        pen = model.penultimate
        layers = [l for l in cfg.get("collapse", {}).get("monitored_layers", ["layer3", "layer4"]) if l in model.feature_dims]
        xc, yc = _random_data(n_c, cache_res, num_classes, 1)
        cdata = DeviceArray(xc, ctx.device)
        idx = np.arange(n_c)

        @torch.no_grad()
        def forward(ids: np.ndarray, want: list[str]):
            from .augment import eval_transform

            lg, fts = [], {l: [] for l in want}
            for s in range(0, len(ids), 512):
                xb = eval_transform(cdata.get(ids[s : s + 512]), res)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=ctx.device.type == "cuda"):
                    logits, feats = model.forward_features(xb)
                lg.append(logits.float())
                for l in want:
                    fts[l].append(feats[l].float())
            return torch.cat(lg), {l: torch.cat(v) for l, v in fts.items()}

        need = sorted(set(layers) | {pen})
        t0 = time.perf_counter()
        logits, feats = forward(idx, need)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_forward = time.perf_counter() - t0
        labeled_feats = feats[pen][: n_c // 5]
        ycs = torch.as_tensor(yc, device=ctx.device)
        class_stats = {l: acq.compute_class_stats(feats[l], ycs, num_classes, mode="diag") for l in layers}
        for strategy in strategies:
            cand = acq.CandidateBatch(indices=idx, logits=logits, feats=feats)
            actx = acq.AcquisitionContext(method={"acquisition": strategy}, params={"noise_samples": 10, "mc_samples": 20}, model=model, device=ctx.device, rng=np.random.default_rng(0), torch_gen=torch.Generator().manual_seed(0), penultimate=pen, forward_fn=lambda ids: forward(ids, [pen]), labeled_feats=labeled_feats, class_stats=class_stats, layer_weights={l: 1.0 / model.feature_dims[l] for l in layers}, threshold_quantile=0.9)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            acq.select(strategy, cand, budget, actx)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            acq_results.append({"strategy": strategy, "n_candidates": n_c, "budget": budget, "select_time_s": dt, "forward_time_s": t_forward, "arch": cfg["arch"], "dataset": dataset})
            ctx.event("acquisition_bench", strategy=strategy, select_time_s=round(dt, 3), forward_time_s=round(t_forward, 3))
    return {"task": "vision_overhead", "arch": cfg["arch"], "dataset": dataset, "seed": ctx.seed, "results": results, "acquisition_results": acq_results}


class _SubContext:
    """Light view of the job context with an overridden config (same logging/artifacts)."""

    def __init__(self, ctx: JobContext, config: dict[str, Any]):
        self._ctx = ctx
        self.config = config
        self.device = ctx.device
        self.job_dir = ctx.job_dir
        self.artifact_dir = ctx.artifact_dir
        self.seed = ctx.seed

    def get(self, dotted: str, default: Any = None):
        from ..utils.io import cfg_get

        return cfg_get(self.config, dotted, default)

    def log_metrics(self, record):
        self._ctx.log_metrics(record)

    def event(self, kind, **data):
        if kind in ("phase_started", "phase_finished"):
            return
        self._ctx.event(kind, **data)

    def timer(self, name):
        return self._ctx.timer(name)
