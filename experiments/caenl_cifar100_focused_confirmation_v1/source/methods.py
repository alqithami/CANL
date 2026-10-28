"""Loss/model adapters for a later, not-yet-launched CIFAR-100 study.

This module does not define a training schedule, tune a method, acquire labels,
or evaluate the official test split. The pinned author losses remain unchanged
under vendor/. See README_METHODS.md and SOURCES.json.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
METHOD_NAMES = ("ce", "ratio_dcr", "full_dcr", "plain_fisher", "supcon", "nonl")


def verify_source_bytes() -> list[dict]:
    """Check downloaded bytes, including Git blob IDs where provided."""
    records = json.loads((ROOT / "SOURCES.json").read_text())["files"]
    for item in records:
        raw = (ROOT / item["local_path"]).read_bytes()
        if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Changed reference source: {item['local_path']}")
        if "git_blob_sha1" in item:
            blob = b"blob " + str(len(raw)).encode() + b"\0" + raw
            if hashlib.sha1(blob).hexdigest() != item["git_blob_sha1"]:
                raise RuntimeError(f"Git blob mismatch: {item['local_path']}")
    return records


def _reference(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise ImportError(relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


verify_source_bytes()
_supcontrast = _reference("_cifar_original_supcontrast", "vendor/supcontrast/losses.py")
_nc_reference = _reference("_cifar_nc_by_design", "vendor/nc_by_design/losses.py")
_aligned = _reference("_cifar_caenl_aligned", "vendor/caenl_reference/aligned.py")


@dataclass(frozen=True)
class GeometryConfig:
    scatter_epsilon: float = 1e-6
    feature_norm_epsilon: float = 1e-6
    huber_delta: float = 0.25
    floor_weight: float = 0.01
    require_balanced: bool = True

    def __post_init__(self):
        for name in ("scatter_epsilon", "feature_norm_epsilon", "huber_delta"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.floor_weight) or self.floor_weight < 0:
            raise ValueError("floor_weight must be finite and nonnegative")


@dataclass(frozen=True)
class FrozenTargets:
    log_kappa_target: float
    between_floor: float

    def __post_init__(self):
        if not math.isfinite(self.log_kappa_target):
            raise ValueError("log_kappa_target must be finite")
        if not math.isfinite(self.between_floor) or self.between_floor <= 0:
            raise ValueError("between_floor must be finite and positive")


def scatter_statistics(features: Tensor, labels: Tensor,
                       cfg: GeometryConfig = GeometryConfig()) -> dict[str, Tensor]:
    """Use the original class-weighted, unit-feature scatter implementation."""
    return _aligned.class_scatter(
        features, labels, eps=cfg.scatter_epsilon,
        norm_eps=cfg.feature_norm_epsilon, require_balanced=cfg.require_balanced,
    )


@torch.no_grad()
def calibrate_targets(anchor_batches: Iterable[tuple[Tensor, Tensor]], *,
                      cfg: GeometryConfig = GeometryConfig(),
                      expected_batches: int = 16) -> FrozenTargets:
    """Calibrate from supplied deterministic anchor features, never live updates.

    The caller must freeze/evaluate its shared initialization checkpoint, supply
    only labeled-training anchors, and bind that checkpoint and anchor IDs.
    """
    stats = [scatter_statistics(z, labels, cfg) for z, labels in anchor_batches]
    if len(stats) != expected_batches or expected_batches <= 0:
        raise ValueError(f"Expected exactly {expected_batches} anchor batches")
    q_mean = sum(float(s["log_kappa"]) for s in stats) / len(stats)
    between_mean = sum(float(s["between"]) for s in stats) / len(stats)
    if not math.isfinite(between_mean) or between_mean <= 10 * cfg.scatter_epsilon:
        raise ValueError("Degenerate anchor dispersion; do not change the target to continue")
    return FrozenTargets(q_mean + math.log(0.9), 0.5 * between_mean)


def geometry_penalty(method: str, features: Tensor, labels: Tensor, *,
                     targets: FrozenTargets | None = None,
                     ratio_weight: float = 0.1,
                     cfg: GeometryConfig = GeometryConfig()) -> tuple[Tensor, dict]:
    """Return a regularizer, not cross-entropy plus a regularizer.

    ratio_dcr: one-sided Huber excess above the frozen log-ratio target.
    full_dcr: ratio_dcr plus the original between-class dispersion safeguard.
    plain_fisher: unthresholded stabilized W/B on the same normalized features.
    ce: zero; skips scatter calculation so it does not hide instrumentation cost.

    The plain_fisher name denotes this explicitly defined trace-ratio control,
    not every method in the Fisher-discriminant literature. Tune its scale
    independently. SupCon and NONL are replacement training losses, not additive
    geometry penalties, and are intentionally rejected here.
    """
    if method not in {"ce", "ratio_dcr", "full_dcr", "plain_fisher"}:
        raise ValueError(f"Not an additive geometry method: {method}")
    if not math.isfinite(ratio_weight) or ratio_weight < 0:
        raise ValueError("ratio_weight must be finite and nonnegative")
    if method == "ce":
        return features.new_zeros((), dtype=torch.float32), {"method": "ce", "regularizer": 0.0}
    s = scatter_statistics(features, labels, cfg)
    ratio = features.new_zeros((), dtype=torch.float32)
    spread = features.new_zeros((), dtype=torch.float32)
    gap = None
    if method == "plain_fisher":
        ratio = s["kappa"]
    else:
        if targets is None:
            raise ValueError("FrozenTargets required for DCR")
        gap = s["log_kappa"] - targets.log_kappa_target
        ratio = _aligned.positive_huber(gap, cfg.huber_delta)
        if method == "full_dcr":
            spread_gap = math.log(targets.between_floor) - torch.log(s["between"] + cfg.scatter_epsilon)
            spread = _aligned.positive_huber(spread_gap, cfg.huber_delta)
    total = ratio_weight * ratio + cfg.floor_weight * spread
    records = {"method": method, **{k: float(v.detach()) for k, v in s.items()},
               "ratio_penalty": float(ratio.detach()), "spread_penalty": float(spread.detach()),
               "ratio_weight": ratio_weight, "regularizer": float(total.detach())}
    if gap is not None:
        records["log_gap"] = float(gap.detach())
    return total, records


def _features(encoder: nn.Module, inputs: Tensor, feature_dim: int) -> Tensor:
    features = encoder(inputs)
    if not isinstance(features, Tensor) or features.ndim != 2 or features.shape[1] != feature_dim:
        raise ValueError("Encoder must return pooled preclassifier features [N, feature_dim]")
    return features


class SupConEncoder(nn.Module):
    """Author-style encoder + D->D->128 MLP and normalized projection.

    Supply a backbone with its classifier removed. forward_views concatenates
    both views before the encoder, matching the original BatchNorm exposure.
    """
    def __init__(self, encoder: nn.Module, feature_dim: int, projection_dim: int = 128):
        super().__init__()
        if feature_dim <= 0 or projection_dim <= 0:
            raise ValueError("Feature dimensions must be positive")
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.projector = nn.Sequential(nn.Linear(feature_dim, feature_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(feature_dim, projection_dim))

    def encode(self, inputs: Tensor) -> Tensor:
        return _features(self.encoder, inputs, self.feature_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        return F.normalize(self.projector(self.encode(inputs)), p=2, dim=1)

    def forward_views(self, view1: Tensor, view2: Tensor) -> Tensor:
        if view1.shape != view2.shape or view1.shape[0] < 1:
            raise ValueError("Two views must have the same nonempty batch shape")
        projected = self(torch.cat((view1, view2), dim=0))
        first, second = projected.split(view1.shape[0], dim=0)
        return torch.stack((first, second), dim=1)


class SupConLoss(nn.Module):
    """Delegate to unchanged original author loss, including tau/base_tau."""
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(base_temperature) or base_temperature <= 0:
            raise ValueError("base_temperature must be finite and positive")
        self.reference = _supcontrast.SupConLoss(
            temperature=temperature, contrast_mode="all", base_temperature=base_temperature,
        )

    def forward(self, projections: Tensor, labels: Tensor) -> Tensor:
        if projections.ndim != 3 or projections.shape[1] != 2:
            raise ValueError("Full SupCon adapter requires exactly two views [N,2,D]")
        return self.reference(projections, labels)


class FrozenLinearProbe(nn.Module):
    """Readout on raw encoder features, with the projection head discarded."""
    def __init__(self, encoder: nn.Module, feature_dim: int, num_classes: int = 100):
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, inputs: Tensor) -> Tensor:
        with torch.no_grad():
            features = _features(self.encoder, inputs, self.feature_dim)
        return self.classifier(features)


class NormalizedClassifier(nn.Module):
    """Normalized features and bias-free normalized learnable class weights.

    forward(x) returns prediction logits. forward(x, return_features=True)
    returns (logits, raw_features); NONLLoss also receives classifier.weight.
    """
    def __init__(self, encoder: nn.Module, feature_dim: int, num_classes: int = 100,
                 temperature: float = 0.1):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)
        self.temperature = temperature

    def forward(self, inputs: Tensor, *, return_features: bool = False):
        features = _features(self.encoder, inputs, self.feature_dim)
        logits = F.linear(F.normalize(features, dim=1),
                          F.normalize(self.classifier.weight, dim=1)) / self.temperature
        return (logits, features) if return_features else logits


class NONLLoss(nn.Module):
    """Stable Eq.5 NONL: class-to-batch, exclude ALL same-class instances.

    Input weights must be [C,D], never an ambiguous transposed shape. This
    adapter normalizes raw features and weights itself; no bias is supported.
    Direct author-code equivalence is checked on nondegenerate finite batches
    in test_methods.py. This is not normalized cross-entropy or NTCE.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.temperature = temperature

    def forward(self, features: Tensor, weights: Tensor, labels: Tensor) -> Tensor:
        if features.ndim != 2 or weights.ndim != 2 or features.shape[1] != weights.shape[1]:
            raise ValueError("Expected features [N,D] and classifier weights [C,D]")
        if labels.ndim != 1 or labels.shape[0] != features.shape[0] or labels.dtype != torch.long:
            raise ValueError("Expected int64 labels [N]")
        if features.shape[0] < 2 or int(labels.min()) < 0 or int(labels.max()) >= weights.shape[0]:
            raise ValueError("Invalid/nonempty class labels")
        if labels.unique().numel() < 2:
            raise ValueError("NONL requires at least two represented classes per batch")
        unit_features = F.normalize(features, p=2, dim=1)
        unit_weights = F.normalize(weights, p=2, dim=1)
        anchor_weights = unit_weights[labels]
        logits = anchor_weights @ unit_features.T / self.temperature
        negative = labels[:, None] != labels[None, :]
        log_denominator = torch.logsumexp(logits.masked_fill(~negative, -torch.inf), dim=1)
        return (log_denominator - logits.diagonal()).mean()
