"""Adversarial robustness evaluation and margin / smoothness diagnostics.

* ``autoattack_evaluate``: official AutoAttack (``fra31/auto-attack``) on a ``NormalizedModel``
  (normalisation inside the module, inputs in [0,1]); ``version='standard'`` = APGD-CE, APGD-T,
  FAB-T, Square; per-example robust masks and adversarial predictions are returned so that
  paired per-example tests (McNemar) can be computed between methods.
* ``apgd_ce_evaluate``: AutoAttack's APGD-CE alone (cheap proxy used at intermediate rounds).
* ``pgd_evaluate``: plain PGD-k (L_inf) as an additional sanity/diagnostic number.
* ``margin_and_gradient_diagnostics``: logit-margin distribution and input-gradient norms of the
  margin (local Lipschitz proxy), supporting the manuscript's Proposition 4 discussion.
"""
from __future__ import annotations

import math
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _EvalWrapper(nn.Module):
    """fp32, eval-mode view of the model returning logits only."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x.float()).float()


@torch.no_grad()
def predict(model: nn.Module, x: Tensor, bs: int = 512) -> Tensor:
    outs = []
    for s in range(0, x.shape[0], bs):
        outs.append(model(x[s : s + bs].float()).argmax(dim=1))
    return torch.cat(outs)


def autoattack_evaluate(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    eps: float = 8 / 255,
    norm: str = "Linf",
    version: str = "standard",
    attacks: Optional[list[str]] = None,
    bs: int = 250,
    seed: int = 0,
    log_path: Optional[str] = None,
    n_examples: Optional[int] = None,
    chunk: int = 5000,
    apgd_restarts: Optional[int] = None,
    apgd_iterations: Optional[int] = None,
) -> dict[str, Any]:
    """Run AutoAttack on ``x`` (float [0,1], CPU or device) and return masks and accuracies."""
    from autoattack import AutoAttack

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    wrapped = _EvalWrapper(model).to(device).eval()
    n = int(x.shape[0]) if n_examples is None else int(min(n_examples, x.shape[0]))
    x = x[:n]
    y = y[:n]
    if version == "custom":
        adversary = AutoAttack(wrapped, norm=norm, eps=float(eps), version="custom", attacks_to_run=list(attacks or ["apgd-ce"]), seed=int(seed), verbose=False, device=str(device), log_path=log_path)
    else:
        adversary = AutoAttack(wrapped, norm=norm, eps=float(eps), version=version, seed=int(seed), verbose=False, device=str(device), log_path=log_path)
    if apgd_restarts is not None:
        adversary.apgd.n_restarts = int(apgd_restarts)
    if apgd_iterations is not None and version == "custom":
        adversary.apgd.n_iter = int(apgd_iterations)
    adversary.apgd.seed = int(seed)
    adversary.apgd_targeted.seed = int(seed)
    adversary.fab.seed = int(seed)
    adversary.square.seed = int(seed)
    t0 = time.time()
    robust_mask = np.zeros(n, dtype=bool)
    adv_pred = np.zeros(n, dtype=np.int64)
    clean_pred = np.zeros(n, dtype=np.int64)
    for s in range(0, n, chunk):
        xb = x[s : s + chunk].to(device).float()
        yb = y[s : s + chunk].to(device).long()
        with torch.no_grad():
            clean_pred[s : s + chunk] = predict(wrapped, xb, bs).cpu().numpy()
        x_adv = adversary.run_standard_evaluation(xb, yb, bs=int(bs))
        with torch.no_grad():
            p = predict(wrapped, x_adv.to(device), bs)
        adv_pred[s : s + chunk] = p.cpu().numpy()
        robust_mask[s : s + chunk] = (p == yb).cpu().numpy()
        # sanity: perturbation budget respected
        linf = float((x_adv.to(device) - xb).abs().max())
        if linf > float(eps) + 1e-5:
            raise RuntimeError(f"AutoAttack produced perturbation {linf} > eps {eps}")
    clean_mask = clean_pred == y.cpu().numpy()
    if was_training:
        model.train()
    return {
        "evaluator": f"autoattack-{version}",
        "attacks": list(attacks) if attacks else (["apgd-ce", "apgd-t", "fab-t", "square"] if version == "standard" else []),
        "norm": norm,
        "eps": float(eps),
        "n": n,
        "clean_acc": float(clean_mask.mean()),
        "robust_acc": float(robust_mask.mean()),
        "robust_mask": robust_mask,
        "clean_mask": clean_mask,
        "adv_pred": adv_pred,
        "clean_pred": clean_pred,
        "time_s": time.time() - t0,
        "seed": int(seed),
    }


def apgd_ce_evaluate(model, x, y, eps=8 / 255, bs=250, seed=0, n_examples=None, restarts=1, log_path=None, iterations=None) -> dict[str, Any]:
    out = autoattack_evaluate(model, x, y, eps=eps, version="custom", attacks=["apgd-ce"], bs=bs, seed=seed, n_examples=n_examples, log_path=log_path, apgd_restarts=restarts, apgd_iterations=iterations)
    out["evaluator"] = "apgd-ce"
    return out


def pgd_attack(model: nn.Module, x: Tensor, y: Tensor, eps: float, alpha: float, steps: int, random_start: bool = True, generator: Optional[torch.Generator] = None) -> Tensor:
    x = x.detach().float()
    delta = torch.zeros_like(x)
    if random_start:
        if generator is not None and generator.device == x.device:
            delta = (torch.rand(x.shape, generator=generator, device=x.device) * 2 - 1) * eps
        else:
            delta = torch.empty_like(x).uniform_(-eps, eps)
    delta = (x + delta).clamp(0, 1) - x
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.cross_entropy(model(x + delta).float(), y)
        (grad,) = torch.autograd.grad(loss, delta)
        delta = (delta.detach() + alpha * grad.sign()).clamp(-eps, eps)
        delta = (x + delta).clamp(0, 1) - x
    return (x + delta).detach()


def pgd_evaluate(model: nn.Module, x: Tensor, y: Tensor, eps: float = 8 / 255, alpha: float = 2 / 255, steps: int = 20, bs: int = 256, n_examples: Optional[int] = None, seed: int = 0) -> dict[str, Any]:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    n = int(x.shape[0]) if n_examples is None else int(min(n_examples, x.shape[0]))
    correct = 0
    mask = np.zeros(n, dtype=bool)
    t0 = time.time()
    for s in range(0, n, bs):
        xb = x[s : s + bs].to(device).float()
        yb = y[s : s + bs].to(device).long()
        x_adv = pgd_attack(model, xb, yb, eps, alpha, steps, generator=gen)
        with torch.no_grad():
            ok = (model(x_adv).argmax(dim=1) == yb)
        mask[s : s + bs] = ok.cpu().numpy()
        correct += int(ok.sum())
    if was_training:
        model.train()
    return {"evaluator": f"pgd{steps}", "eps": float(eps), "alpha": float(alpha), "steps": int(steps), "n": n, "robust_acc": correct / max(n, 1), "robust_mask": mask, "time_s": time.time() - t0}


def margin_and_gradient_diagnostics(model: nn.Module, x: Tensor, y: Tensor, bs: int = 256, n_examples: int = 1000, perturbation_eps: float = 8 / 255, n_random_dirs: int = 4, generator: Optional[torch.Generator] = None, seed: int = 0) -> dict[str, Any]:
    """Logit margins, input-gradient norms of the margin and empirical local Lipschitz estimates."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
    n = int(min(n_examples, x.shape[0]))
    margins: list[Tensor] = []
    grad_norms: list[Tensor] = []
    lip: list[Tensor] = []
    for s in range(0, n, bs):
        xb = x[s : s + bs].to(device).float().requires_grad_(True)
        yb = y[s : s + bs].to(device).long()
        logits = model(xb).float()
        true = logits.gather(1, yb[:, None]).squeeze(1)
        other = logits.clone()
        other.scatter_(1, yb[:, None], float("-inf"))
        m = true - other.max(dim=1).values
        (g,) = torch.autograd.grad(m.sum(), xb)
        gn = g.flatten(1).norm(dim=1)
        margins.append(m.detach())
        grad_norms.append(gn.detach())
        # empirical Lipschitz: |m(x+d)-m(x)| / ||d||_2 over random L_inf-bounded directions
        with torch.no_grad():
            best = torch.zeros_like(m)
            for _ in range(n_random_dirs):
                d = (torch.rand(xb.shape, generator=generator, device=device) * 2 - 1) * perturbation_eps
                xp = (xb + d).clamp(0, 1)
                lg = model(xp).float()
                t2 = lg.gather(1, yb[:, None]).squeeze(1)
                o2 = lg.clone()
                o2.scatter_(1, yb[:, None], float("-inf"))
                m2 = t2 - o2.max(dim=1).values
                ratio = (m2 - m.detach()).abs() / (xp - xb).flatten(1).norm(dim=1).clamp_min(1e-12)
                best = torch.maximum(best, ratio)
            lip.append(best)
    M = torch.cat(margins).cpu()
    G = torch.cat(grad_norms).cpu()
    L = torch.cat(lip).cpu()
    q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    out = {
        "n": n,
        "margin_mean": float(M.mean()),
        "margin_std": float(M.std(unbiased=False)),
        "margin_quantiles": {f"q{int(qq * 100)}": float(v) for qq, v in zip(q.tolist(), torch.quantile(M, q).tolist())},
        "margin_positive_fraction": float((M > 0).float().mean()),
        "grad_norm_mean": float(G.mean()),
        "grad_norm_median": float(G.median()),
        "grad_norm_quantiles": {f"q{int(qq * 100)}": float(v) for qq, v in zip(q.tolist(), torch.quantile(G, q).tolist())},
        "margin_over_gradnorm_median": float(torch.median(M.clamp_min(0) / G.clamp_min(1e-12))),
        "empirical_lipschitz_mean": float(L.mean()),
        "empirical_lipschitz_q90": float(torch.quantile(L, 0.9)),
    }
    # certified-style lower bound on the L2 radius from margin / gradient norm (first-order, diagnostic only)
    out["first_order_l2_radius_median"] = out["margin_over_gradnorm_median"]
    if was_training:
        model.train()
    return out
