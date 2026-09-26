"""Label-free initialisation of the torchvision ViT-B/16 from the MAE encoder (``facebook/vit-mae-base``).

MAE (He et al., CVPR 2022) pre-trains the encoder on ImageNet-1K *images only* (no labels), so
initialising the active learner from it does not leak the labels it is supposed to acquire.  The
Hugging Face checkpoint is converted tensor-by-tensor into torchvision's ``VisionTransformer``
(same architecture: 12 pre-norm blocks, 768 hidden, 12 heads, MLP 3072, patch 16); the conversion
is verified by :func:`check_equivalence` (cls-token features of both implementations agree) and by
``tests/test_vision.py``.  The classifier head is freshly initialised.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F

MAE_REPO = "facebook/vit-mae-base"


def _pick(state: Mapping[str, torch.Tensor], *names: str) -> torch.Tensor:
    for n in names:
        if n in state:
            return state[n]
    raise KeyError(f"none of {names} in the MAE state dict (keys like {list(state)[:5]})")


def convert_mae_state_dict(hf_state: Mapping[str, torch.Tensor], n_layers: int) -> dict[str, torch.Tensor]:
    """Map an HF ``ViTMAEModel`` state dict (transformers v4 or v5 key names) onto torchvision ``VisionTransformer`` keys."""
    s = {k.replace("vit.", "", 1) if k.startswith("vit.") else k: v for k, v in hf_state.items()}
    out: dict[str, torch.Tensor] = {}
    out["class_token"] = _pick(s, "embeddings.cls_token")
    out["encoder.pos_embedding"] = _pick(s, "embeddings.position_embeddings")
    out["conv_proj.weight"] = _pick(s, "embeddings.patch_embeddings.projection.weight")
    out["conv_proj.bias"] = _pick(s, "embeddings.patch_embeddings.projection.bias")
    for i in range(n_layers):
        v5 = f"layers.{i}."
        v4 = f"encoder.layer.{i}."
        q_w = _pick(s, v5 + "attention.q_proj.weight", v4 + "attention.attention.query.weight")
        k_w = _pick(s, v5 + "attention.k_proj.weight", v4 + "attention.attention.key.weight")
        v_w = _pick(s, v5 + "attention.v_proj.weight", v4 + "attention.attention.value.weight")
        q_b = _pick(s, v5 + "attention.q_proj.bias", v4 + "attention.attention.query.bias")
        k_b = _pick(s, v5 + "attention.k_proj.bias", v4 + "attention.attention.key.bias")
        v_b = _pick(s, v5 + "attention.v_proj.bias", v4 + "attention.attention.value.bias")
        p = f"encoder.layers.encoder_layer_{i}."
        out[p + "self_attention.in_proj_weight"] = torch.cat([q_w, k_w, v_w], dim=0)
        out[p + "self_attention.in_proj_bias"] = torch.cat([q_b, k_b, v_b], dim=0)
        out[p + "self_attention.out_proj.weight"] = _pick(s, v5 + "attention.o_proj.weight", v4 + "attention.output.dense.weight")
        out[p + "self_attention.out_proj.bias"] = _pick(s, v5 + "attention.o_proj.bias", v4 + "attention.output.dense.bias")
        out[p + "ln_1.weight"] = _pick(s, v5 + "layernorm_before.weight", v4 + "layernorm_before.weight")
        out[p + "ln_1.bias"] = _pick(s, v5 + "layernorm_before.bias", v4 + "layernorm_before.bias")
        out[p + "ln_2.weight"] = _pick(s, v5 + "layernorm_after.weight", v4 + "layernorm_after.weight")
        out[p + "ln_2.bias"] = _pick(s, v5 + "layernorm_after.bias", v4 + "layernorm_after.bias")
        out[p + "mlp.0.weight"] = _pick(s, v5 + "mlp.fc1.weight", v4 + "intermediate.dense.weight")
        out[p + "mlp.0.bias"] = _pick(s, v5 + "mlp.fc1.bias", v4 + "intermediate.dense.bias")
        out[p + "mlp.3.weight"] = _pick(s, v5 + "mlp.fc2.weight", v4 + "output.dense.weight")
        out[p + "mlp.3.bias"] = _pick(s, v5 + "mlp.fc2.bias", v4 + "output.dense.bias")
    out["encoder.ln.weight"] = _pick(s, "layernorm.weight")
    out["encoder.ln.bias"] = _pick(s, "layernorm.bias")
    return out


def resize_pos_embedding(pos: torch.Tensor, new_tokens: int) -> torch.Tensor:
    """Bicubic-resample a ``[1, 1 + g*g, D]`` position embedding (cls first) to ``1 + g'*g'`` tokens."""
    if pos.shape[1] == new_tokens:
        return pos
    cls, grid = pos[:, :1], pos[:, 1:]
    g_old = int(round(math.sqrt(grid.shape[1])))
    g_new = int(round(math.sqrt(new_tokens - 1)))
    if g_old * g_old != grid.shape[1] or g_new * g_new != new_tokens - 1:
        raise ValueError(f"cannot resample a {grid.shape[1]}-token grid to {new_tokens - 1} tokens")
    grid = grid.reshape(1, g_old, g_old, -1).permute(0, 3, 1, 2)
    grid = F.interpolate(grid.float(), size=(g_new, g_new), mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, g_new * g_new, -1).to(pos.dtype)
    return torch.cat([cls, grid], dim=1)


def load_mae_into_torchvision(net: torch.nn.Module, hf_state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Load converted MAE weights into a torchvision ``VisionTransformer`` (head left untouched)."""
    n_layers = len(net.encoder.layers)
    conv = convert_mae_state_dict(hf_state, n_layers)
    conv["encoder.pos_embedding"] = resize_pos_embedding(conv["encoder.pos_embedding"], net.encoder.pos_embedding.shape[1])
    target = net.state_dict()
    for k, v in conv.items():
        if k not in target:
            raise KeyError(f"converted key {k} not in the torchvision model")
        if tuple(target[k].shape) != tuple(v.shape):
            raise ValueError(f"shape mismatch for {k}: torchvision {tuple(target[k].shape)} vs MAE {tuple(v.shape)}")
    missing, unexpected = net.load_state_dict(conv, strict=False)
    missing = [m for m in missing if not m.startswith("heads.")]
    if missing or unexpected:
        raise RuntimeError(f"MAE conversion incomplete: missing={missing[:5]} unexpected={list(unexpected)[:5]}")
    return {"loaded": len(conv), "layers": n_layers, "pos_tokens": int(net.encoder.pos_embedding.shape[1])}


def fetch_mae_state_dict(repo: str = MAE_REPO, cache_dir: Optional[str] = None) -> dict[str, torch.Tensor]:
    """Download (or load from the HF cache) the MAE encoder weights."""
    from transformers import ViTMAEModel

    model = ViTMAEModel.from_pretrained(repo, cache_dir=cache_dir)
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def check_equivalence(hf_model: torch.nn.Module, tv_net: torch.nn.Module, x: torch.Tensor) -> float:
    """Max |difference| between HF's cls-token encoder output and torchvision's (after the final LayerNorm).

    ``hf_model`` must be a ``ViTMAEModel`` with ``config.mask_ratio == 0`` (its random patch shuffling does
    not change the cls-token output, which is permutation-invariant).
    """
    hf_model.eval()
    tv_net.eval()
    hf_cls = hf_model(pixel_values=x).last_hidden_state[:, 0]
    h = tv_net._process_input(x)
    n = h.shape[0]
    h = torch.cat([tv_net.class_token.expand(n, -1, -1), h], dim=1)
    h = tv_net.encoder(h)  # adds pos_embedding, runs the blocks and the final LayerNorm
    return float((h[:, 0] - hf_cls).abs().max())


def vit_param_layer_ids(named_params, n_layers: int) -> dict[str, int]:
    """Layer index per parameter name for layer-wise lr decay (0 = embeddings, n_layers + 1 = head/final norm)."""
    ids: dict[str, int] = {}
    for name, _ in named_params:
        m = re.search(r"encoder_layer_(\d+)\.", name)
        if m:
            ids[name] = int(m.group(1)) + 1
        elif any(t in name for t in ("conv_proj", "class_token", "pos_embedding")):
            ids[name] = 0
        else:
            ids[name] = n_layers + 1
    return ids
