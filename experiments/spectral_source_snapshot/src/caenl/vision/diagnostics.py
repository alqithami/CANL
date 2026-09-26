"""Clean evaluation, calibration, NC panels and acquisition-score analyses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sps
from torch import Tensor

from ..core.nc import nc_metrics
from .augment import eval_transform
from .data import DeviceArray


@dataclass
class EvalResult:
    loss: float
    acc: float
    top5: float
    ece: float
    nll: float
    logits: np.ndarray
    preds: np.ndarray
    feats: dict[str, Tensor] = field(default_factory=dict)


def expected_calibration_error(probs: Tensor, labels: Tensor, n_bins: int = 15) -> float:
    conf, pred = probs.max(dim=1)
    correct = (pred == labels).float()
    bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros((), device=probs.device)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            ece += m.float().mean() * (conf[m].mean() - correct[m].mean()).abs()
    return float(ece)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data: DeviceArray,
    labels: np.ndarray,
    res: int,
    bs: int = 512,
    layers: Optional[list[str]] = None,
    indices: Optional[np.ndarray] = None,
    amp: bool = True,
) -> EvalResult:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    idx = np.arange(len(labels)) if indices is None else np.asarray(indices)
    y_all = torch.as_tensor(labels[idx], device=device).long()
    logits_all: list[Tensor] = []
    feats_all: dict[str, list[Tensor]] = {l: [] for l in (layers or [])}
    use_amp = amp and device.type == "cuda"
    data.prefetch([idx[s : s + bs] for s in range(0, len(idx), bs)])  # memmap mode only
    for s in range(0, len(idx), bs):
        xb = eval_transform(data.get(idx[s : s + bs]), res)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits, feats = model.forward_features(xb)
        logits_all.append(logits.float())
        for l in feats_all:
            feats_all[l].append(feats[l].float())
    logits = torch.cat(logits_all)
    probs = F.softmax(logits, dim=1)
    nll = float(F.cross_entropy(logits, y_all))
    preds = logits.argmax(dim=1)
    acc = float((preds == y_all).float().mean())
    k = min(5, logits.shape[1])
    top5 = float((logits.topk(k, dim=1).indices == y_all[:, None]).any(dim=1).float().mean())
    ece = expected_calibration_error(probs, y_all)
    if was_training:
        model.train()
    return EvalResult(
        loss=nll,
        acc=acc,
        top5=top5,
        ece=ece,
        nll=nll,
        logits=logits.cpu().numpy().astype(np.float16),
        preds=preds.cpu().numpy(),
        feats={l: torch.cat(v) for l, v in feats_all.items()},
    )


@torch.no_grad()
def nc_panel(feats: Mapping[str, Tensor], labels: np.ndarray, num_classes: int, classifier_weight: Optional[Tensor], preds: np.ndarray, penultimate: str, topk: int = 128) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    y = torch.as_tensor(labels)
    p = torch.as_tensor(preds)
    for l, z in feats.items():
        z = z.detach()
        y_dev = y.to(z.device)
        p_dev = p.to(z.device)
        W = classifier_weight.detach().to(z.device) if (l == penultimate and classifier_weight is not None) else None
        out[l] = nc_metrics(z, y_dev, num_classes, classifier_weight=W, predictions=p_dev, topk=topk, compute_pinv_nc1=z.shape[1] <= 2048)
    return out


def score_analysis(scores: Mapping[str, np.ndarray], selected_positions: Optional[np.ndarray] = None, top_fraction: float = 0.1) -> dict[str, Any]:
    """Spearman correlations between acquisition scores and top-set overlaps (candidate pass level)."""
    names = [k for k, v in scores.items() if v is not None and len(v) > 1]
    out: dict[str, Any] = {"n": int(len(scores[names[0]])) if names else 0, "spearman": {}, "top_overlap": {}}
    if not names:
        return out
    n = len(scores[names[0]])
    k = max(1, int(top_fraction * n))
    tops = {a: set(np.argsort(-np.asarray(scores[a]))[:k].tolist()) for a in names}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            try:
                rho = float(sps.spearmanr(scores[a], scores[b]).statistic)
            except Exception:
                rho = float("nan")
            out["spearman"][f"{a}|{b}"] = rho
            out["top_overlap"][f"{a}|{b}"] = len(tops[a] & tops[b]) / k
    if selected_positions is not None and len(selected_positions) > 0:
        sel = set(int(x) for x in selected_positions)
        for a in names:
            out["top_overlap"][f"selected|{a}"] = len(sel & tops[a]) / max(1, min(len(sel), k))
    return out


def class_histogram(labels: np.ndarray, num_classes: int) -> list[int]:
    return np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).tolist()
