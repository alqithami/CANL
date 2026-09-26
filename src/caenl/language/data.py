"""C4 token caches (deterministic, shard-pinned) and a synthetic corpus for smoke tests."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..utils.io import atomic_write_json, ensure_dir, read_json


def _cache_dir(cache_root: Path, tokenizer: str, train_shards: list[str], val_shards: list[str], train_tokens: int, val_tokens: int, seq_len: int) -> Path:
    key = hashlib.sha256(json.dumps({"tok": tokenizer, "tr": train_shards, "va": val_shards, "nt": train_tokens, "nv": val_tokens, "L": seq_len}, sort_keys=True).encode()).hexdigest()[:12]
    return ensure_dir(Path(cache_root) / "language" / f"c4-{tokenizer.replace('/', '_')}-{key}")


def _tokenize_stream(ds, tokenizer, budget: int, eos_id: int, desc: str) -> np.ndarray:
    out = np.empty(budget, dtype=np.uint16)
    pos = 0
    buf: list[int] = []
    n_docs = 0
    for row in ds:
        ids = tokenizer(row["text"])["input_ids"]
        ids.append(eos_id)
        buf.extend(ids)
        n_docs += 1
        if len(buf) >= 1_000_000 or pos + len(buf) >= budget:
            take = min(len(buf), budget - pos)
            out[pos : pos + take] = np.asarray(buf[:take], dtype=np.uint16)
            pos += take
            buf = buf[take:]
            print(f"[c4] {desc}: {pos / 1e6:.1f}M / {budget / 1e6:.1f}M tokens ({n_docs} docs)", flush=True)
            if pos >= budget:
                break
    if pos < budget:
        take = min(len(buf), budget - pos)
        out[pos : pos + take] = np.asarray(buf[:take], dtype=np.uint16)
        pos += take
    if pos < budget:
        print(f"[c4] WARNING: only {pos} tokens available for {desc} (requested {budget})", flush=True)
    return out[:pos]


def prepare_c4(data_root: Path, cache_root: Path, options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    options = dict(options or {})
    tokenizer_name = str(options.get("tokenizer", "gpt2"))
    train_shards = list(options.get("train_shards", ["en/c4-train.00000-of-01024.json.gz"]))
    val_shards = list(options.get("validation_shards", ["en/c4-validation.00000-of-00008.json.gz"]))
    seq_len = int(options.get("sequence_length", 1024))
    train_tokens = int(options.get("train_tokens", 100_000_000))
    val_tokens = int(options.get("validation_tokens", 2_000_000))
    extra = float(options.get("train_token_margin", 1.02))
    out = _cache_dir(Path(cache_root), tokenizer_name, train_shards, val_shards, train_tokens, val_tokens, seq_len)
    meta = out / "meta.json"
    if meta.exists():
        return read_json(meta)
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0
    ds_tr = load_dataset("allenai/c4", data_files={"train": train_shards}, split="train", streaming=True, cache_dir=str(Path(data_root) / "hf"))
    ds_va = load_dataset("allenai/c4", data_files={"validation": val_shards}, split="validation", streaming=True, cache_dir=str(Path(data_root) / "hf"))
    tr = _tokenize_stream(ds_tr, tok, int(train_tokens * extra), eos, "train")
    va = _tokenize_stream(ds_va, tok, val_tokens, eos, "validation")
    np.save(out / "train_tokens.npy", tr)
    np.save(out / "val_tokens.npy", va)
    info = {"source": "allenai/c4", "tokenizer": tokenizer_name, "train_shards": train_shards, "validation_shards": val_shards, "n_train_tokens": int(len(tr)), "n_val_tokens": int(len(va)), "sequence_length": seq_len, "cache_dir": str(out), "vocab_size": int(len(tok))}
    atomic_write_json(meta, info)
    return info


def prepare_synthetic_text(cache_root: Path, options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Markov-chain token stream over a small vocabulary (smoke tests only)."""
    options = dict(options or {})
    vocab = int(options.get("vocab_size", 256))
    n_train = int(options.get("train_tokens", 200_000))
    n_val = int(options.get("validation_tokens", 20_000))
    seq_len = int(options.get("sequence_length", 64))
    out = ensure_dir(Path(cache_root) / "language" / f"synthetic-v{vocab}-n{n_train}-L{seq_len}")
    meta = out / "meta.json"
    if meta.exists():
        return read_json(meta)
    rng = np.random.default_rng(int(options.get("seed", 0)))
    trans = rng.dirichlet(np.full(vocab, 0.05), size=vocab)

    def gen(n):
        seq = np.empty(n, dtype=np.uint16)
        s = int(rng.integers(vocab))
        for i in range(n):
            s = int(rng.choice(vocab, p=trans[s]))
            seq[i] = s
        return seq

    np.save(out / "train_tokens.npy", gen(n_train))
    np.save(out / "val_tokens.npy", gen(n_val))
    info = {"source": "synthetic-markov", "tokenizer": "none", "n_train_tokens": n_train, "n_val_tokens": n_val, "sequence_length": seq_len, "cache_dir": str(out), "vocab_size": vocab}
    atomic_write_json(meta, info)
    return info


def load_token_cache(info: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    d = Path(info["cache_dir"])
    return np.load(d / "train_tokens.npy", mmap_mode="r"), np.load(d / "val_tokens.npy", mmap_mode="r")
