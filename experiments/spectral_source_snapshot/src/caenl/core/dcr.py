"""Spectrum-only Dynamic Collapse Regularisation (DCR), manuscript Section 4.3.

``L_DCR = sum_l lambda_l * || p_l - p_l^* ||_2^2`` where ``p_l`` is the trace-normalised
top-k eigenvalue vector of the layer-l activation covariance and ``p_l^*`` a fixed
target template:

* classification (ETF-like): equal mass on the first ``r = min(C-1, d, k)`` components;
* generative / language (low-rank near-isotropic): equal mass on the first
  ``r = min(r_cfg, d, k)`` components.

Only eigen*values* are penalised, so the gradient of ``eigvalsh`` is well defined even
when the target makes eigenvalues degenerate (the eigenvalue-only backward is
``V diag(g) V^T``, with no ``1/(lambda_i - lambda_j)`` terms).

Implementation notes
--------------------
* The batch covariance spectrum is obtained through the Gram trick: for a centred
  feature matrix ``Z`` of shape ``[n, d]`` the non-zero eigenvalues of ``Z^T Z / n``
  (``d x d``) equal those of ``Z Z^T / n`` (``n x n``); the smaller matrix is used.
* ``covariance='total'`` uses the unconditional covariance of the batch (the manuscript's
  generic ``Sigma_l``); ``covariance='between'`` uses the covariance of the batch class
  means (NC2 view). The frozen protocol uses ``total`` for every modality.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import torch
from torch import Tensor


def feasible_target_rank(k: int, dimension: int, num_classes: Optional[int] = None, configured_rank: Optional[int] = None) -> int:
    k_eff = min(int(k), int(dimension))
    if num_classes is not None:
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        return max(1, min(int(num_classes) - 1, int(dimension), k_eff))
    if configured_rank is None or configured_rank <= 0:
        raise ValueError("configured_rank is required for non-classification targets")
    return max(1, min(int(configured_rank), int(dimension), k_eff))


def target_spectrum(
    *,
    k: int,
    dimension: int,
    num_classes: Optional[int] = None,
    configured_rank: Optional[int] = None,
    device=None,
    dtype=None,
) -> Tensor:
    """Normalised target template of length ``min(k, d)`` with ``1/r`` on the first ``r`` entries."""
    if k <= 0 or dimension <= 0:
        raise ValueError("k and dimension must be positive")
    k_eff = min(int(k), int(dimension))
    r = feasible_target_rank(k, dimension, num_classes, configured_rank)
    out = torch.zeros(k_eff, device=device, dtype=dtype or torch.float32)
    out[:r] = 1.0 / float(r)
    return out


def covariance_eigenvalues(features: Tensor, center: bool = True) -> Tensor:
    """Descending eigenvalues of the (population) covariance of ``features`` ``[n, d]``.

    Differentiable w.r.t. ``features``.  Returns ``min(n, d)`` values.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be [n, d], got {tuple(features.shape)}")
    z = features.float()
    n, d = z.shape
    if center:
        z = z - z.mean(dim=0, keepdim=True)
    if n <= d:
        m = z @ z.T / float(n)
    else:
        m = z.T @ z / float(n)
    m = 0.5 * (m + m.T)
    eigs = torch.linalg.eigvalsh(m)
    return eigs.flip(0).clamp_min(0.0)


def normalized_topk_spectrum(features: Tensor, k: int, eps: float = 1e-8, center: bool = True) -> Tensor:
    """Trace-normalised (over the retained top-k) spectrum, zero padded to ``min(k, d)``."""
    eigs = covariance_eigenvalues(features, center=center)
    d = features.shape[1]
    k_eff = min(int(k), int(d))
    top = eigs[:k_eff]
    if top.numel() < k_eff:
        top = torch.cat([top, top.new_zeros(k_eff - top.numel())])
    return top / (top.sum() + eps)


def spectrum_loss(observed: Tensor, target: Tensor) -> Tensor:
    if observed.shape != target.shape:
        raise ValueError(f"spectrum shape mismatch: {tuple(observed.shape)} vs {tuple(target.shape)}")
    return torch.sum((observed - target.to(observed)) ** 2)


def batch_class_means(features: Tensor, labels: Tensor) -> Tensor:
    classes, inverse = torch.unique(labels, return_inverse=True)
    counts = torch.bincount(inverse, minlength=classes.numel()).to(features.dtype).clamp_min(1)
    sums = torch.zeros(classes.numel(), features.shape[1], dtype=features.dtype, device=features.device)
    sums.index_add_(0, inverse, features.to(sums.dtype))
    return sums / counts[:, None]


@dataclass
class DCRLayerSpec:
    name: str
    dimension: int
    target: Tensor
    rank: int


@dataclass
class DCR:
    """Spectrum-only DCR over a set of monitored layers."""

    layers: dict[str, DCRLayerSpec] = field(default_factory=dict)
    k: int = 128
    covariance: str = "total"  # total | between
    eps: float = 1e-8

    @classmethod
    def build(
        cls,
        layer_dims: Mapping[str, int],
        k: int,
        num_classes: Optional[int] = None,
        configured_rank: Optional[int | Mapping[str, int]] = None,
        covariance: str = "total",
        device=None,
    ) -> "DCR":
        specs: dict[str, DCRLayerSpec] = {}
        for name, d in layer_dims.items():
            cr = configured_rank.get(name) if isinstance(configured_rank, Mapping) else configured_rank
            tgt = target_spectrum(k=k, dimension=d, num_classes=num_classes, configured_rank=cr, device=device)
            specs[name] = DCRLayerSpec(name=name, dimension=int(d), target=tgt, rank=int((tgt > 0).sum()))
        return cls(layers=specs, k=int(k), covariance=covariance)

    def layer_loss(self, name: str, features: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        spec = self.layers[name]
        if self.covariance == "between":
            if labels is None:
                raise ValueError("between-class DCR requires labels")
            feats = batch_class_means(features, labels)
            if feats.shape[0] < 2:
                return features.new_zeros(())
        else:
            feats = features
        if feats.ndim > 2:
            feats = feats.flatten(1)
        p = normalized_topk_spectrum(feats, self.k, eps=self.eps)
        target = spec.target
        n_rank = int(feats.shape[0]) - 1  # a centred batch covariance has rank <= n-1
        if 0 < n_rank < spec.rank:
            target = torch.zeros_like(spec.target)
            target[:n_rank] = 1.0 / float(n_rank)
        return spectrum_loss(p, target)

    def __call__(
        self,
        features: Mapping[str, Tensor],
        lambdas: Mapping[str, float | Tensor],
        labels: Optional[Tensor] = None,
        flags: Optional[Mapping[str, float | Tensor]] = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Return ``(total_loss, {layer: unweighted_loss})``; skips layers with zero weight."""
        total = None
        per_layer: dict[str, float] = {}
        for name, feats in features.items():
            if name not in self.layers:
                continue
            lam = lambdas.get(name, 0.0)
            lam_v = float(lam) if not torch.is_tensor(lam) else float(lam.item())
            if flags is not None:
                fl = flags.get(name, 1.0)
                lam_v *= float(fl) if not torch.is_tensor(fl) else float(fl.item())
            if lam_v <= 0.0:
                per_layer[name] = float("nan")
                continue
            loss_l = self.layer_loss(name, feats, labels)
            per_layer[name] = float(loss_l.detach())
            term = lam_v * loss_l
            total = term if total is None else total + term
        if total is None:
            any_feat = next(iter(features.values()))
            total = any_feat.new_zeros((), dtype=torch.float32)
        return total, per_layer
