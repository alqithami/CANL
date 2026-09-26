"""AutoAttack pipeline sanity check (manuscript Sec. 6.2.1, "Sanity Check").

Evaluates a *reference* model under exactly the frozen robustness protocol of the campaign
(same ``fixed_evaluation_subset``, same AutoAttack version/eps/seed/batch size, inputs in
[0, 1] with normalisation inside the model).  With the torchvision ImageNet ResNet-50
(``IMAGENET1K_V1``) the job must reproduce ~76.1 % clean accuracy and ~0 % robust accuracy at
8/255; a configured tolerance turns a deviation into a job failure, so a mis-configured attack
(wrong input range, double normalisation, eval-mode bugs) is caught before any CAENL number is
reported.  On datasets without a pretrained reference (CIFAR-10 in the GPU smoke) the job only
exercises the code path with a randomly initialised network and records the numbers.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..utils.jobctx import JobContext
from ..utils.seeding import derive_seed, seed_everything
from .augment import eval_transform
from .data import DeviceArray, fixed_evaluation_subset, load_vision_dataset
from .diagnostics import evaluate_model
from .models import build_model
from .robustness import autoattack_evaluate


def run(ctx: JobContext) -> dict[str, Any]:
    cfg = ctx.config
    sc = dict(cfg.get("sanity", {}))
    rob = dict(cfg.get("robustness", {}))
    fin = dict(rob.get("final", {}))
    seed_everything(ctx.seed, deterministic=False)
    paths = cfg.get("paths", {})
    ds = load_vision_dataset(cfg["dataset"], paths.get("data_root", "data"), paths.get("cache_root", "cache"), cfg.get("dataset_options", {}))
    test = DeviceArray(ds.test_x, ctx.device, max_device_fraction=0.2)
    arch_opts = dict(cfg.get("arch_options", {}))
    arch_opts.setdefault("image_size", int(ds.train_res))
    arch_opts.setdefault("hf_cache", str(Path(paths.get("data_root", "data")) / "hf"))
    pretrained = bool(sc.get("pretrained", False))
    arch_opts["pretrained"] = pretrained
    if pretrained:
        arch_opts.setdefault("weights", sc.get("weights", "IMAGENET1K_V1"))
    torch.manual_seed(derive_seed(ctx.seed, "init"))
    model = build_model(cfg["arch"], ds.num_classes, ds.mean, ds.std, arch_opts).to(ctx.device).eval()
    ctx.event("model_ready", arch=cfg["arch"], pretrained=pretrained, weights=arch_opts.get("weights"), n_test=int(len(ds.test_y)))

    with ctx.timer("eval_clean"):
        ev = evaluate_model(model, test, ds.test_y, ds.test_res, bs=int(cfg.get("active_learning", {}).get("eval_batch_size", 256)), layers=[], amp=False)
    ctx.event("clean_evaluated", test_acc=ev.acc, test_top5=ev.top5)

    idx = fixed_evaluation_subset(ds.test_y, fin.get("n_examples"), ds.name, "autoattack")
    ctx.save_artifact_npy("aa_subset.npy", idx)
    n = int(len(idx))
    if n * 3 * ds.test_res * ds.test_res * 4 > 2e9:
        x = torch.cat([eval_transform(test.get(idx[s : s + 2000]), ds.test_res).cpu() for s in range(0, n, 2000)])
    else:
        x = eval_transform(test.get(idx), ds.test_res)
    y = torch.as_tensor(ds.test_y[idx], device=ctx.device)
    t0 = time.time()
    with ctx.timer("eval_autoattack"):
        res = autoattack_evaluate(
            model,
            x,
            y,
            eps=float(rob.get("eps", 8 / 255)),
            norm=str(rob.get("norm", "Linf")),
            version=str(fin.get("version", "standard")),
            attacks=fin.get("attacks"),
            bs=int(fin.get("batch_size", 250)),
            seed=derive_seed(ctx.seed, "attack"),
            log_path=str(ctx.job_dir / "autoattack_log.txt"),
            chunk=int(fin.get("chunk", 5000)),
            apgd_restarts=fin.get("apgd_restarts"),
            apgd_iterations=fin.get("apgd_iterations"),
        )
    ctx.save_artifact_npy("aa_robust_mask.npy", res["robust_mask"])
    ctx.save_artifact_npy("aa_clean_mask.npy", res["clean_mask"])
    aa = {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}
    summary: dict[str, Any] = {
        "task": "vision_aa_sanity",
        "dataset": ds.name,
        "arch": cfg["arch"],
        "seed": ctx.seed,
        "pretrained": pretrained,
        "weights": arch_opts.get("weights"),
        "clean_acc_full_test": float(ev.acc),
        "clean_top5_full_test": float(ev.top5),
        "aa": aa,
        "aa_n": n,
        "aa_time_s": time.time() - t0,
        "protocol": {"eps": float(rob.get("eps", 8 / 255)), "norm": rob.get("norm", "Linf"), "version": fin.get("version", "standard"), "n_examples": n},
        "dataset_fingerprint": ds.fingerprint,
    }
    exp = sc.get("expected")
    if exp:
        tol = float(sc.get("tolerance", 0.02))
        checks = {}
        if "clean_acc" in exp:
            checks["clean_acc"] = {"expected": float(exp["clean_acc"]), "observed": float(ev.acc), "ok": abs(float(ev.acc) - float(exp["clean_acc"])) <= tol}
        if "robust_acc_max" in exp:
            checks["robust_acc"] = {"max_expected": float(exp["robust_acc_max"]), "observed": float(aa["robust_acc"]), "ok": float(aa["robust_acc"]) <= float(exp["robust_acc_max"])}
        summary["checks"] = checks
        summary["sanity_ok"] = all(c["ok"] for c in checks.values())
        ctx.event("sanity_checked", ok=summary["sanity_ok"], **{k: v["observed"] for k, v in checks.items()})
        if not summary["sanity_ok"] and bool(sc.get("fail_on_mismatch", True)):
            raise RuntimeError(f"AutoAttack sanity check failed: {checks}")
    else:
        summary["sanity_ok"] = None
    return summary
