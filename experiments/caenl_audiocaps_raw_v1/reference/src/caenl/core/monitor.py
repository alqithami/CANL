"""Online collapse monitoring (manuscript Section 4, "Monitoring every N steps").

``ClassConditionalMonitor``
    Keeps EMA class-conditional means / diagonal variances per monitored layer (decay 0.99,
    shrinkage 1e-5 as in the manuscript) and EMA traces of within-/between-class scatter,
    giving ``kappa_l(t) = tr(Sigma_W,l)/tr(Sigma_B,l)`` and the instability
    ``dkappa_l(t) = |kappa_l(t) - EMA(kappa_l)(t)|`` (Definition 2).

``SpectralMonitor``
    For label-free representations (LM hidden states, diffusion latents, audio embeddings):
    EMA covariance per layer whose trace-normalised spectrum gives the effective rank
    ``kappa_l = exp(H(p))`` (manuscript Section 3.1, generalised collapse index).
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import torch
from torch import Tensor


class ClassConditionalMonitor:
    def __init__(
        self,
        layer_dims: Mapping[str, int],
        num_classes: int,
        ema_decay: float = 0.99,
        shrinkage: float = 1e-5,
        kappa_ema_decay: float = 0.9,
        device: torch.device | str = "cpu",
    ) -> None:
        self.layers = list(layer_dims.keys())
        self.dims = {k: int(v) for k, v in layer_dims.items()}
        self.C = int(num_classes)
        self.decay = float(ema_decay)
        self.shrinkage = float(shrinkage)
        self.kappa_decay = float(kappa_ema_decay)
        self.device = torch.device(device)
        self.mean = {l: torch.zeros(self.C, d, device=self.device) for l, d in self.dims.items()}
        self.var = {l: torch.ones(self.C, d, device=self.device) for l, d in self.dims.items()}
        self.seen = {l: torch.zeros(self.C, dtype=torch.bool, device=self.device) for l in self.layers}
        self.class_weight = {l: torch.zeros(self.C, device=self.device) for l in self.layers}
        self.tr_w = {l: float("nan") for l in self.layers}
        self.tr_b = {l: float("nan") for l in self.layers}
        self.kappa_last = {l: float("nan") for l in self.layers}
        self.kappa_ema = {l: float("nan") for l in self.layers}
        self.dkappa = {l: float("nan") for l in self.layers}
        self.updates = 0

    @torch.no_grad()
    def update(self, feats: Mapping[str, Tensor], labels: Tensor) -> None:
        labels = labels.to(self.device).long()
        present = torch.unique(labels)
        for l in self.layers:
            z = feats[l].detach().to(self.device, torch.float32)
            if z.ndim > 2:
                z = z.flatten(1)
            counts = torch.bincount(labels, minlength=self.C).float()
            sums = torch.zeros_like(self.mean[l])
            sums.index_add_(0, labels, z)
            bmean = sums[present] / counts[present, None]
            sq = torch.zeros_like(self.mean[l])
            sq.index_add_(0, labels, z * z)
            bvar = (sq[present] / counts[present, None] - bmean * bmean).clamp_min(0.0)
            seen = self.seen[l][present]
            # fresh classes: direct assignment, seen classes: EMA
            m_old = self.mean[l][present]
            v_old = self.var[l][present]
            m_new = torch.where(seen[:, None], self.decay * m_old + (1 - self.decay) * bmean, bmean)
            v_new = torch.where(seen[:, None], self.decay * v_old + (1 - self.decay) * bvar, bvar)
            self.mean[l][present] = m_new
            self.var[l][present] = v_new.clamp_min(self.shrinkage)
            self.seen[l][present] = True
            self.class_weight[l] = self.decay * self.class_weight[l] + (1 - self.decay) * counts / counts.sum()
            # traces with EMA centres
            resid = z - self.mean[l][labels]
            tr_w_b = float(resid.pow(2).sum(dim=1).mean())
            seen_all = self.seen[l]
            w = self.class_weight[l][seen_all]
            w = w / w.sum().clamp_min(1e-12)
            mu = self.mean[l][seen_all]
            gmean = (w[:, None] * mu).sum(dim=0, keepdim=True)
            tr_b = float((w * (mu - gmean).pow(2).sum(dim=1)).sum())
            if math.isnan(self.tr_w[l]):
                self.tr_w[l], self.tr_b[l] = tr_w_b, tr_b
            else:
                self.tr_w[l] = self.decay * self.tr_w[l] + (1 - self.decay) * tr_w_b
                self.tr_b[l] = self.decay * self.tr_b[l] + (1 - self.decay) * tr_b
            kappa = self.tr_w[l] / max(self.tr_b[l], 1e-12)
            self.kappa_last[l] = kappa
            if math.isnan(self.kappa_ema[l]):
                self.kappa_ema[l] = kappa
                self.dkappa[l] = 0.0
            else:
                self.dkappa[l] = abs(kappa - self.kappa_ema[l])
                self.kappa_ema[l] = self.kappa_decay * self.kappa_ema[l] + (1 - self.kappa_decay) * kappa
        self.updates += 1

    def kappa(self) -> dict[str, float]:
        return dict(self.kappa_last)

    def instability(self) -> dict[str, float]:
        return dict(self.dkappa)

    def strength(self) -> dict[str, float]:
        return {l: 1.0 / (1.0 + k) if math.isfinite(k) else float("nan") for l, k in self.kappa_last.items()}

    def record(self) -> dict[str, float]:
        rec: dict[str, float] = {}
        for l in self.layers:
            rec[f"kappa/{l}"] = self.kappa_last[l]
            rec[f"dkappa/{l}"] = self.dkappa[l]
            rec[f"strength/{l}"] = 1.0 / (1.0 + self.kappa_last[l]) if math.isfinite(self.kappa_last[l]) else float("nan")
            rec[f"tr_w/{l}"] = self.tr_w[l]
            rec[f"tr_b/{l}"] = self.tr_b[l]
        return rec

    @torch.no_grad()
    def diag_mahalanobis(self, layer: str, z: Tensor, predicted: Tensor) -> Tensor:
        """Squared diagonal Mahalanobis distance of ``z`` to the EMA centre of ``predicted`` class."""
        z = z.to(self.device, torch.float32)
        if z.ndim > 2:
            z = z.flatten(1)
        predicted = predicted.to(self.device).long()
        mean = self.mean[layer][predicted]
        var = self.var[layer][predicted].clamp_min(self.shrinkage)
        return ((z - mean) ** 2 / var).sum(dim=1)

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": {l: v.cpu() for l, v in self.mean.items()},
            "var": {l: v.cpu() for l, v in self.var.items()},
            "seen": {l: v.cpu() for l, v in self.seen.items()},
            "class_weight": {l: v.cpu() for l, v in self.class_weight.items()},
            "tr_w": dict(self.tr_w),
            "tr_b": dict(self.tr_b),
            "kappa_last": dict(self.kappa_last),
            "kappa_ema": dict(self.kappa_ema),
            "dkappa": dict(self.dkappa),
            "updates": self.updates,
        }

    def load_state_dict(self, s: Mapping[str, Any]) -> None:
        for l in self.layers:
            self.mean[l].copy_(s["mean"][l].to(self.device))
            self.var[l].copy_(s["var"][l].to(self.device))
            self.seen[l].copy_(s["seen"][l].to(self.device))
            self.class_weight[l].copy_(s["class_weight"][l].to(self.device))
        self.tr_w.update(s["tr_w"])
        self.tr_b.update(s["tr_b"])
        self.kappa_last.update(s["kappa_last"])
        self.kappa_ema.update(s["kappa_ema"])
        self.dkappa.update(s["dkappa"])
        self.updates = int(s.get("updates", 0))


class SpectralMonitor:
    def __init__(
        self,
        layer_dims: Mapping[str, int],
        ema_decay: float = 0.99,
        kappa_ema_decay: float = 0.9,
        max_dim: int = 2048,
        projection_seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        self.layers = list(layer_dims.keys())
        self.device = torch.device(device)
        self.decay = float(ema_decay)
        self.kappa_decay = float(kappa_ema_decay)
        self.proj: dict[str, Optional[Tensor]] = {}
        self.dims: dict[str, int] = {}
        g = torch.Generator(device="cpu").manual_seed(int(projection_seed))
        for l, d in layer_dims.items():
            d = int(d)
            if d > max_dim:
                P = torch.randn(d, max_dim, generator=g) / math.sqrt(max_dim)
                self.proj[l] = P.to(self.device)
                self.dims[l] = max_dim
            else:
                self.proj[l] = None
                self.dims[l] = d
        self.cov = {l: torch.zeros(d, d, device=self.device) for l, d in self.dims.items()}
        self.mean = {l: torch.zeros(d, device=self.device) for l, d in self.dims.items()}
        self.initialized = {l: False for l in self.layers}
        self.kappa_last = {l: float("nan") for l in self.layers}
        self.kappa_ema = {l: float("nan") for l in self.layers}
        self.dkappa = {l: float("nan") for l in self.layers}
        self.updates = 0
        self._sub_gen = torch.Generator(device="cpu").manual_seed(int(projection_seed) + 1)

    @torch.no_grad()
    def update(self, feats: Mapping[str, Tensor], labels: Optional[Tensor] = None, compute_kappa: bool = True, max_tokens: int = 8192) -> None:
        for l in self.layers:
            z = feats[l].detach().to(self.device, torch.float32)
            if z.ndim > 2:
                z = z.reshape(-1, z.shape[-1])
            if z.shape[0] > max_tokens:
                idx = torch.randperm(z.shape[0], generator=self._sub_gen)[:max_tokens].to(z.device)
                z = z[idx]
            if self.proj[l] is not None:
                z = z @ self.proj[l]
            m = z.mean(dim=0)
            zc = z - m
            c = zc.T @ zc / float(max(z.shape[0], 1))
            if not self.initialized[l]:
                self.mean[l].copy_(m)
                self.cov[l].copy_(c)
                self.initialized[l] = True
            else:
                self.mean[l].mul_(self.decay).add_(m, alpha=1 - self.decay)
                self.cov[l].mul_(self.decay).add_(c, alpha=1 - self.decay)
            if compute_kappa:
                eigs = torch.linalg.eigvalsh(0.5 * (self.cov[l] + self.cov[l].T)).clamp_min(0)
                p = eigs / eigs.sum().clamp_min(1e-12)
                nz = p[p > 0]
                kappa = float((-(nz * nz.log()).sum()).exp())
                self.kappa_last[l] = kappa
                if math.isnan(self.kappa_ema[l]):
                    self.kappa_ema[l], self.dkappa[l] = kappa, 0.0
                else:
                    self.dkappa[l] = abs(kappa - self.kappa_ema[l])
                    self.kappa_ema[l] = self.kappa_decay * self.kappa_ema[l] + (1 - self.kappa_decay) * kappa
        self.updates += 1

    def kappa(self) -> dict[str, float]:
        return dict(self.kappa_last)

    def instability(self) -> dict[str, float]:
        return dict(self.dkappa)

    def record(self) -> dict[str, float]:
        rec: dict[str, float] = {}
        for l in self.layers:
            rec[f"effective_rank/{l}"] = self.kappa_last[l]
            rec[f"dkappa/{l}"] = self.dkappa[l]
        return rec

    def state_dict(self) -> dict[str, Any]:
        return {
            "cov": {l: v.cpu() for l, v in self.cov.items()},
            "mean": {l: v.cpu() for l, v in self.mean.items()},
            "initialized": dict(self.initialized),
            "kappa_last": dict(self.kappa_last),
            "kappa_ema": dict(self.kappa_ema),
            "dkappa": dict(self.dkappa),
            "updates": self.updates,
        }

    def load_state_dict(self, s: Mapping[str, Any]) -> None:
        for l in self.layers:
            self.cov[l].copy_(s["cov"][l].to(self.device))
            self.mean[l].copy_(s["mean"][l].to(self.device))
        self.initialized.update(s["initialized"])
        self.kappa_last.update(s["kappa_last"])
        self.kappa_ema.update(s["kappa_ema"])
        self.dkappa.update(s["dkappa"])
        self.updates = int(s.get("updates", 0))
