"""Batched GPU augmentation on uint8 tensors ``[B,3,H,W]`` -> float ``[0,1]``."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def to_float(x: Tensor) -> Tensor:
    return x.float().div_(255.0) if x.dtype == torch.uint8 else x.float()


def cifar_augment(x: Tensor, pad: int = 4, flip: bool = True, generator: torch.Generator | None = None) -> Tensor:
    """Random crop with zero padding + horizontal flip (standard CIFAR recipe)."""
    B, C, H, W = x.shape
    xf = to_float(x)
    xp = F.pad(xf, (pad, pad, pad, pad))
    oy = torch.randint(0, 2 * pad + 1, (B,), device=x.device, generator=generator)
    ox = torch.randint(0, 2 * pad + 1, (B,), device=x.device, generator=generator)
    rows = (oy[:, None] + torch.arange(H, device=x.device)[None, :])  # B,H
    cols = (ox[:, None] + torch.arange(W, device=x.device)[None, :])  # B,W
    b = torch.arange(B, device=x.device)[:, None, None]
    out = xp.permute(0, 2, 3, 1)[b, rows[:, :, None], cols[:, None, :]]  # B,H,W,C
    out = out.permute(0, 3, 1, 2)
    if flip:
        mask = torch.rand(B, device=x.device, generator=generator) < 0.5
        out = torch.where(mask[:, None, None, None], out.flip(3), out)
    return out.contiguous()


def random_resized_crop(
    x: Tensor,
    out_size: int,
    scale: tuple[float, float] = (0.25, 1.0),
    ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    flip: bool = True,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Batched RandomResizedCrop via affine sampling grids (bilinear)."""
    B, C, H, W = x.shape
    xf = to_float(x)
    dev = x.device
    area = torch.empty(B, device=dev).uniform_(scale[0], scale[1], generator=generator)
    log_r = torch.empty(B, device=dev).uniform_(math.log(ratio[0]), math.log(ratio[1]), generator=generator)
    r = torch.exp(log_r)
    w = torch.sqrt(area * r).clamp(max=1.0)  # fraction of width
    h = torch.sqrt(area / r).clamp(max=1.0)
    cx = (torch.rand(B, device=dev, generator=generator) * (1 - w) + w / 2) * 2 - 1  # centre in [-1,1]
    cy = (torch.rand(B, device=dev, generator=generator) * (1 - h) + h / 2) * 2 - 1
    sx = w.clone()
    if flip:
        fl = torch.rand(B, device=dev, generator=generator) < 0.5
        sx = torch.where(fl, -sx, sx)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = sx
    theta[:, 0, 2] = cx
    theta[:, 1, 1] = h
    theta[:, 1, 2] = cy
    grid = F.affine_grid(theta, (B, C, out_size, out_size), align_corners=False)
    return F.grid_sample(xf, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def center_crop(x: Tensor, out_size: int) -> Tensor:
    B, C, H, W = x.shape
    xf = to_float(x)
    if H == out_size and W == out_size:
        return xf
    if H < out_size or W < out_size:
        return F.interpolate(xf, size=(out_size, out_size), mode="bilinear", align_corners=False)
    top, left = (H - out_size) // 2, (W - out_size) // 2
    return xf[:, :, top : top + out_size, left : left + out_size].contiguous()


def train_transform(x: Tensor, kind: str, out_size: int, generator: torch.Generator | None = None, rrc_scale=(0.25, 1.0)) -> Tensor:
    if kind == "cifar":
        return cifar_augment(x, pad=max(1, out_size // 8), flip=True, generator=generator)
    if kind == "rrc":
        return random_resized_crop(x, out_size, scale=tuple(rrc_scale), flip=True, generator=generator)
    if kind == "none":
        return center_crop(x, out_size)
    raise ValueError(f"unknown augmentation kind {kind}")


def eval_transform(x: Tensor, out_size: int) -> Tensor:
    return center_crop(x, out_size)
