"""Neural-collapse metrics (NC1-NC4), spectral summaries and the CAENL collapse strength.

Conventions (match the manuscript, Section 3.1):

* ``kappa = tr(Sigma_W) / tr(Sigma_B)`` is the *classification collapse metric*;
  lower kappa = stronger collapse.
* ``collapse_strength = tr(Sigma_B) / (tr(Sigma_W) + tr(Sigma_B)) = 1 / (1 + kappa)``
  is the bounded, monotone re-parameterisation in (0, 1) used for plots and for
  the controller targets.
* For generative / language representations the collapse index is the
  *effective rank* ``exp(H(p))`` of the trace-normalised covariance spectrum
  (``p_j = lambda_j / sum lambda``); lower = more collapsed.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor


@torch.no_grad()
def class_means(features: Tensor, labels: Tensor, num_classes: int) -> tuple[Tensor, Tensor]:
    """Return per-class means ``[C, d]`` and counts ``[C]`` (zero rows for unseen classes)."""
    if features.ndim != 2:
        raise ValueError(f"features must be [n, d], got {tuple(features.shape)}")
    if labels.shape != (features.shape[0],):
        raise ValueError("labels must be [n]")
    labels = labels.to(features.device).long()
    counts = torch.bincount(labels, minlength=num_classes).to(features.dtype)
    sums = torch.zeros(num_classes, features.shape[1], dtype=features.dtype, device=features.device)
    sums.index_add_(0, labels, features)
    means = sums / counts.clamp_min(1.0)[:, None]
    return means, counts


@torch.no_grad()
def scatter_traces(features: Tensor, labels: Tensor, num_classes: int, means: Optional[Tensor] = None) -> dict[str, Tensor]:
    """Traces of within-/between-class scatter (population normalisation, class-frequency weighted)."""
    if means is None:
        means, counts = class_means(features, labels, num_classes)
    else:
        counts = torch.bincount(labels.long(), minlength=num_classes).to(features.dtype)
    labels = labels.to(features.device).long()
    resid = features - means[labels]
    tr_w = resid.pow(2).sum() / float(features.shape[0])
    present = counts > 0
    w = counts[present] / counts.sum()
    global_mean = (w[:, None] * means[present]).sum(dim=0, keepdim=True)
    tr_b = (w * (means[present] - global_mean).pow(2).sum(dim=1)).sum()
    kappa = tr_w / tr_b.clamp_min(1e-12)
    return {
        "within_trace": tr_w,
        "between_trace": tr_b,
        "kappa": kappa,
        "collapse_strength": tr_b / (tr_w + tr_b).clamp_min(1e-12),
        "num_classes_present": present.sum(),
    }


@torch.no_grad()
def spectrum_summary(cov_or_eigs: Tensor, k: int = 128, eps: float = 1e-12, is_eigs: bool = False) -> dict[str, Any]:
    """Effective rank, normalised spectral entropy, participation ratio and the top-k normalised spectrum."""
    if is_eigs:
        eigs = cov_or_eigs.clamp_min(0)
    else:
        c = 0.5 * (cov_or_eigs + cov_or_eigs.T)
        eigs = torch.linalg.eigvalsh(c.double()).clamp_min(0)
    eigs = torch.sort(eigs, descending=True).values
    total = eigs.sum().clamp_min(eps)
    p = eigs / total
    nz = p[p > 0]
    entropy = -(nz * nz.log()).sum()
    d = eigs.numel()
    return {
        "effective_rank": float(entropy.exp()),
        "spectral_entropy_normalized": float(entropy / torch.log(torch.tensor(float(max(d, 2))))),
        "participation_ratio": float(total.pow(2) / eigs.pow(2).sum().clamp_min(eps)),
        "top1_fraction": float(p[0]) if d > 0 else float("nan"),
        "topk_spectrum": p[: min(k, d)].tolist(),
        "trace": float(total),
    }


@torch.no_grad()
def effective_rank(cov: Tensor, eps: float = 1e-12) -> Tensor:
    c = 0.5 * (cov + cov.T)
    eigs = torch.linalg.eigvalsh(c).clamp_min(0)
    p = eigs / eigs.sum().clamp_min(eps)
    nz = p[p > 0]
    return (-(nz * nz.log()).sum()).exp()


@torch.no_grad()
def nc_metrics(
    features: Tensor,
    labels: Tensor,
    num_classes: int,
    classifier_weight: Optional[Tensor] = None,
    predictions: Optional[Tensor] = None,
    topk: int = 128,
    compute_pinv_nc1: bool = True,
) -> dict[str, Any]:
    """Full NC1-NC4 panel on a feature matrix (typically the test set at the penultimate layer)."""
    features = features.double()
    labels = labels.to(features.device).long()
    C = int(num_classes)
    means, counts = class_means(features, labels, C)
    present = counts > 0
    traces = scatter_traces(features, labels, C, means)
    out: dict[str, Any] = {
        "kappa": float(traces["kappa"]),
        "collapse_strength": float(traces["collapse_strength"]),
        "within_trace": float(traces["within_trace"]),
        "between_trace": float(traces["between_trace"]),
        "num_classes_present": int(traces["num_classes_present"]),
    }
    w = counts[present] / counts.sum()
    global_mean = (w[:, None] * means[present]).sum(dim=0, keepdim=True)
    M = means[present] - global_mean  # centred class means [C', d]
    Cp = int(M.shape[0])

    # NC1 (Papyan): tr(Sigma_W Sigma_B^+) / C
    if compute_pinv_nc1 and features.shape[1] <= 4096:
        resid = features - means[labels]
        sigma_w = resid.T @ resid / float(features.shape[0])
        sigma_b = (M.T * w) @ M
        try:
            out["nc1_papyan"] = float(torch.trace(sigma_w @ torch.linalg.pinv(sigma_b, hermitian=True)) / Cp)
        except Exception:
            out["nc1_papyan"] = float("nan")

    # NC2: equinorm, equiangularity, ETF distance
    norms = M.norm(dim=1)
    out["nc2_equinorm_cv"] = float(norms.std(unbiased=False) / norms.mean().clamp_min(1e-12))
    Mn = M / norms.clamp_min(1e-12)[:, None]
    cos = Mn @ Mn.T
    off = cos[~torch.eye(Cp, dtype=torch.bool, device=cos.device)]
    out["nc2_equiangular_std"] = float(off.std(unbiased=False))
    out["nc2_mean_cosine"] = float(off.mean())
    out["nc2_target_cosine"] = float(-1.0 / max(Cp - 1, 1))
    gram = M @ M.T
    gram = gram / gram.norm().clamp_min(1e-12)
    etf = (torch.eye(Cp, device=gram.device, dtype=gram.dtype) - torch.full((Cp, Cp), 1.0 / Cp, device=gram.device, dtype=gram.dtype))
    etf = etf / etf.norm().clamp_min(1e-12)
    out["nc2_etf_distance"] = float((gram - etf).norm())

    # NC3: self-duality between classifier rows and centred class means
    if classifier_weight is not None and classifier_weight.shape[0] == C:
        W = classifier_weight.double().to(features.device)[present]
        A = W / W.norm().clamp_min(1e-12)
        B = M / M.norm().clamp_min(1e-12)
        out["nc3_self_duality"] = float((A - B).norm())

    # NC4: nearest-class-centre agreement with classifier prediction
    if predictions is not None:
        d2 = torch.cdist(features, means[present])
        ncc = torch.nonzero(present).squeeze(1)[d2.argmin(dim=1)]
        out["nc4_ncc_agreement"] = float((ncc == predictions.to(ncc.device).long()).double().mean())
        out["ncc_accuracy"] = float((ncc == labels).double().mean())

    # Spectral summaries of the total covariance
    centered = features - features.mean(dim=0, keepdim=True)
    n, d = centered.shape
    if n <= d:
        gram_t = centered @ centered.T / float(n)
        eigs = torch.linalg.eigvalsh(gram_t).clamp_min(0)
    else:
        cov_t = centered.T @ centered / float(n)
        eigs = torch.linalg.eigvalsh(cov_t).clamp_min(0)
    spec = spectrum_summary(eigs, k=topk, is_eigs=True)
    out.update({f"total_{k}": v for k, v in spec.items()})
    out["feature_norm_mean"] = float(features.norm(dim=1).mean())
    return out
