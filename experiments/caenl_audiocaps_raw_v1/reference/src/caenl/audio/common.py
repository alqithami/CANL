"""Shared audio modules: token encoder over cached features, pair datasets, tokenizers."""
from __future__ import annotations

import math
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _debug_stop(step: int) -> None:
    v = os.environ.get("CAENL_DEBUG_STOP_AT_STEP")
    if v and step >= int(v):
        raise SystemExit(f"debug stop at step {step}")


class AudioTokenEncoder(nn.Module):
    """Trainable transformer encoder over frozen backbone tokens (the representation DCR shapes)."""

    def __init__(self, in_dim: int, d_model: int = 512, n_layers: int = 2, n_heads: int = 8, dropout: float = 0.1, max_tokens: int = 256):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """tokens [B,T,in_dim], mask [B,T] (True = valid) -> (sequence [B,T,d], pooled [B,d])."""
        x = self.proj(tokens.float()) + self.pos[:, : tokens.shape[1]]
        h = self.encoder(x, src_key_padding_mask=~mask)
        h = self.norm(h)
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        return h, pooled


class CharTokenizer:
    """Byte-level tokenizer for smoke tests (no downloads)."""

    def __init__(self):
        self.bos_token_id = 256
        self.eos_token_id = 257
        self.pad_token_id = 258
        self.vocab_size = 259

    def encode(self, text: str, max_len: int = 64) -> list[int]:
        return [self.bos_token_id] + list(text.lower().encode("utf-8"))[: max_len - 2] + [self.eos_token_id]

    def decode(self, ids) -> str:
        out = bytearray()
        for i in ids:
            i = int(i)
            if i == self.eos_token_id:
                break
            if i < 256:
                out.append(i)
        return out.decode("utf-8", errors="ignore").strip()


class HFTokenizer:
    def __init__(self, name: str, cache_dir: Optional[str] = None):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
        self.bos_token_id = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id
        self.eos_token_id = self.tok.eos_token_id
        self.pad_token_id = self.tok.eos_token_id
        self.vocab_size = len(self.tok)

    def encode(self, text: str, max_len: int = 48) -> list[int]:
        ids = self.tok(text.strip().lower(), add_special_tokens=False)["input_ids"][: max_len - 2]
        return [self.bos_token_id] + ids + [self.eos_token_id]

    def decode(self, ids) -> str:
        ids = [int(i) for i in ids]
        if self.eos_token_id in ids:
            ids = ids[: ids.index(self.eos_token_id)]
        return self.tok.decode(ids, skip_special_tokens=True).strip()


def pad_batch(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros(len(seqs), L, dtype=torch.bool)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = True
    return ids, mask


def split_indices(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    out: dict[str, list[int]] = {"train": [], "valid": [], "test": []}
    for i, r in enumerate(rows):
        out.setdefault(r["split"], []).append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}


def clip_batch(features: np.ndarray, n_tokens: np.ndarray, idx: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    T = int(max(1, n_tokens[idx].max()))
    x = torch.from_numpy(np.ascontiguousarray(features[idx, :T]).astype(np.float32)).to(device)
    mask = torch.arange(T)[None, :] < torch.from_numpy(n_tokens[idx])[:, None]
    return x, mask.to(device)


def cosine_lr(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * 0.5 * (1 + math.cos(math.pi * p))
