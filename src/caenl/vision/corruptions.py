"""Synthetic common-corruption probe (download-free; *not* CIFAR-10-C/ImageNet-C).

Five corruption families x three severities applied on the GPU to a fixed evaluation subset:
Gaussian noise, impulse noise, Gaussian blur, contrast reduction, brightness shift.  The
reported ``corruption_mean_acc`` is the mean accuracy over all (family, severity) pairs.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

SEVERITY = {
    "gaussian_noise": [0.04, 0.08, 0.12],
    "impulse_noise": [0.02, 0.05, 0.10],
    "gaussian_blur": [0.6, 1.0, 1.5],
    "contrast": [0.6, 0.4, 0.25],
    "brightness": [0.1, 0.2, 0.3],
}


def _gaussian_kernel(sigma: float, device) -> Tensor:
    radius = max(1, int(3 * sigma))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(x**2) / (2 * sigma**2))
    k = k / k.sum()
    return k


def corrupt(x: Tensor, kind: str, severity: int, generator: torch.Generator | None = None) -> Tensor:
    level = SEVERITY[kind][severity - 1]
    if kind == "gaussian_noise":
        noise = torch.randn(x.shape, generator=generator, device=x.device) if generator is not None and generator.device == x.device else torch.randn_like(x)
        return (x + level * noise).clamp(0, 1)
    if kind == "impulse_noise":
        u = torch.rand(x.shape, generator=generator, device=x.device) if generator is not None and generator.device == x.device else torch.rand_like(x)
        out = x.clone()
        out[u < level / 2] = 0.0
        out[u > 1 - level / 2] = 1.0
        return out
    if kind == "gaussian_blur":
        k = _gaussian_kernel(level, x.device)
        c = x.shape[1]
        kx = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
        ky = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
        pad = k.numel() // 2
        out = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"), kx, groups=c)
        out = F.conv2d(F.pad(out, (0, 0, pad, pad), mode="reflect"), ky, groups=c)
        return out.clamp(0, 1)
    if kind == "contrast":
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        return ((x - mean) * level + mean).clamp(0, 1)
    if kind == "brightness":
        return (x + level).clamp(0, 1)
    raise ValueError(kind)


@torch.no_grad()
def corruption_evaluate(model: torch.nn.Module, x: Tensor, y: Tensor, severities: list[int] | None = None, bs: int = 256, seed: int = 0) -> dict[str, Any]:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    severities = severities or [1, 2, 3]
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    results: dict[str, float] = {}
    accs = []
    for kind in SEVERITY:
        for s in severities:
            correct = 0
            for i in range(0, x.shape[0], bs):
                xb = corrupt(x[i : i + bs].to(device).float(), kind, s, gen)
                pred = model(xb).argmax(dim=1)
                correct += int((pred == y[i : i + bs].to(device)).sum())
            acc = correct / x.shape[0]
            results[f"{kind}_s{s}"] = acc
            accs.append(acc)
    if was_training:
        model.train()
    return {"n": int(x.shape[0]), "per_corruption": results, "corruption_mean_acc": float(np.mean(accs))}
