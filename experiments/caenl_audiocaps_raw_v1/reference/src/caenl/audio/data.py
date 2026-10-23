"""Audio data: Clotho v2.1 download/extraction, manifests, frozen feature caches, text embeddings.

Manifest format (jsonl, one clip per line)::

    {"id": "clip-0001", "path": "/abs/path/clip.wav", "split": "train|valid|test", "captions": ["...", ...]}

The same manifest format can describe AudioCaps (``prepare_manifest``) when the audio is
available locally; Clotho is the default because it is freely downloadable (Zenodo 4783391).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

from ..utils.io import atomic_save_npy, atomic_write_json, ensure_dir, file_lock, read_json, read_jsonl

CLOTHO_RECORD = "https://zenodo.org/records/4783391/files/{name}?download=1"
CLOTHO_FILES = {
    "development": ("clotho_audio_development.7z", "clotho_captions_development.csv"),
    "validation": ("clotho_audio_validation.7z", "clotho_captions_validation.csv"),
    "evaluation": ("clotho_audio_evaluation.7z", "clotho_captions_evaluation.csv"),
}
SPLIT_MAP = {"development": "train", "validation": "valid", "evaluation": "test"}


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[audio] downloading {url} -> {dest}", flush=True)
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    os.replace(tmp, dest)


def _extract_7z(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("7z"):
        subprocess.check_call(["7z", "x", "-y", f"-o{out_dir}", str(archive)], stdout=subprocess.DEVNULL)
        return
    import py7zr  # type: ignore

    with py7zr.SevenZipFile(str(archive), mode="r") as z:
        z.extractall(path=str(out_dir))


def prepare_clotho(data_root: Path, cache_root: Path, options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    options = dict(options or {})
    root = ensure_dir(Path(data_root) / "clotho")
    manifest = root / "manifest.jsonl"
    if manifest.exists():
        rows = read_jsonl(manifest)
        return {"manifest": str(manifest), "n_clips": len(rows), "splits": dict(_count(rows))}
    rows: list[dict[str, Any]] = []
    for split, (audio7z, caps_csv) in CLOTHO_FILES.items():
        _download(CLOTHO_RECORD.format(name=audio7z), root / audio7z)
        _download(CLOTHO_RECORD.format(name=caps_csv), root / caps_csv)
        audio_dir = root / "audio" / split
        if not audio_dir.exists() or not any(audio_dir.rglob("*.wav")):
            _extract_7z(root / audio7z, audio_dir)
        wavs = {p.name: p for p in audio_dir.rglob("*.wav")}
        with (root / caps_csv).open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                name = r["file_name"]
                if name not in wavs:
                    continue
                caps = [r[f"caption_{i}"].strip() for i in range(1, 6) if r.get(f"caption_{i}")]
                rows.append({"id": f"{split}/{name}", "path": str(wavs[name].resolve()), "split": SPLIT_MAP[split], "captions": caps})
        if bool(options.get("delete_archives", False)):
            (root / audio7z).unlink(missing_ok=True)
    with manifest.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return {"manifest": str(manifest), "n_clips": len(rows), "splits": dict(_count(rows))}


def _count(rows):
    from collections import Counter

    return Counter(r["split"] for r in rows)


def prepare_manifest(data_root: Path, cache_root: Path, options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    options = dict(options or {})
    path = Path(options["manifest"])
    rows = read_jsonl(path)
    missing = [r["path"] for r in rows if not Path(r["path"]).exists()]
    return {"manifest": str(path), "n_clips": len(rows), "missing_audio": len(missing), "splits": dict(_count(rows))}


def prepare_synthetic_audio(cache_root: Path, options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Tiny synthetic corpus (tones + noise with templated captions) for smoke tests."""
    import soundfile as sf

    options = dict(options or {})
    n = int(options.get("n_clips", 60))
    root = ensure_dir(Path(cache_root) / "audio" / f"synthetic-{n}")
    manifest = root / "manifest.jsonl"
    if manifest.exists():
        rows = read_jsonl(manifest)
        return {"manifest": str(manifest), "n_clips": len(rows), "splits": dict(_count(rows))}
    rng = np.random.default_rng(0)
    kinds = [("a low hum of an engine", 110.0), ("a high pitched whistle", 1760.0), ("a bird chirping quickly", 880.0), ("a bell ringing once", 440.0)]
    rows = []
    sr = 16000
    for i in range(n):
        k = i % len(kinds)
        text, f0 = kinds[k]
        t = np.arange(sr * 2) / sr
        wave = 0.3 * np.sin(2 * math.pi * f0 * t) * (1 + 0.2 * np.sin(2 * math.pi * (2 + k) * t)) + 0.05 * rng.normal(size=t.shape)
        p = root / f"clip{i:03d}.wav"
        sf.write(str(p), wave.astype(np.float32), sr)
        split = "train" if i < int(0.6 * n) else ("valid" if i < int(0.8 * n) else "test")
        caps = [text, text + " in the distance", "sound of " + text, text + " is heard", "there is " + text]
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        rows.append({"id": f"{split}/{p.stem}", "youtube_id": p.stem, "start_time": 0, "path": str(p.resolve()), "split": split, "captions": caps, "bytes": p.stat().st_size, "sha256": h})
    with manifest.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    atomic_write_json(root / "manifest_meta.json", {"dataset": "synthetic", "n_clips": len(rows), "coverage": None})
    return {"manifest": str(manifest), "n_clips": len(rows), "splits": dict(_count(rows))}


# ----------------------------------------------------------------------------- audio loading
def load_audio_16k(path: str, target_sr: int = 16000, max_seconds: float = 30.0) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != target_sr:
        g = math.gcd(int(sr), int(target_sr))
        wav = resample_poly(wav, target_sr // g, sr // g).astype(np.float32)
    max_len = int(max_seconds * target_sr)
    return wav[:max_len]


# ----------------------------------------------------------------------------- feature backbones
class MelCNNBackbone(torch.nn.Module):
    """Deterministic random-init log-mel CNN (smoke tests; no downloads)."""

    def __init__(self, n_mels: int = 64, dim: int = 128, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.n_mels = n_mels
        self.dim = dim
        self.net = torch.nn.Sequential(torch.nn.Conv2d(1, 32, 3, stride=(2, 2), padding=1), torch.nn.GELU(), torch.nn.Conv2d(32, 64, 3, stride=(2, 2), padding=1), torch.nn.GELU(), torch.nn.Conv2d(64, dim, 3, stride=(2, 1), padding=1), torch.nn.GELU())
        self.register_buffer("mel_fb", _mel_filterbank(400 // 2 + 1, n_mels, 16000))

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:  # [T] -> [tokens, dim]
        spec = torch.stft(wav, n_fft=400, hop_length=160, win_length=400, window=torch.hann_window(400, device=wav.device), return_complex=True).abs() ** 2
        mel = torch.log(self.mel_fb.T @ spec + 1e-6)  # [n_mels, frames]
        x = mel[None, None]
        h = self.net(x)  # [1, dim, f', t']
        return h.mean(dim=2).squeeze(0).T  # [t', dim]


def _mel_filterbank(n_freqs: int, n_mels: int, sr: int) -> torch.Tensor:
    def hz2mel(f):
        return 2595 * math.log10(1 + f / 700)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    m_pts = torch.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2)
    f_pts = torch.tensor([mel2hz(m) for m in m_pts])
    freqs = torch.linspace(0, sr / 2, n_freqs)
    fb = torch.zeros(n_freqs, n_mels)
    for i in range(n_mels):
        lo, c, hi = f_pts[i], f_pts[i + 1], f_pts[i + 2]
        fb[:, i] = torch.clamp(torch.minimum((freqs - lo) / (c - lo + 1e-9), (hi - freqs) / (hi - c + 1e-9)), min=0)
    return fb


class ASTBackbone(torch.nn.Module):
    """Frozen Audio Spectrogram Transformer (AudioSet-pretrained) producing patch tokens per 10.24 s window."""

    def __init__(self, name: str, cache_dir: Optional[str] = None, device: torch.device = torch.device("cpu")):
        super().__init__()
        from transformers import AutoFeatureExtractor, AutoModel

        self.extractor = AutoFeatureExtractor.from_pretrained(name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(name, cache_dir=cache_dir).to(device).eval()
        self.device = device
        self.dim = int(self.model.config.hidden_size)
        self.window = int(getattr(self.extractor, "max_length", 1024)) * 160  # samples per window at 10 ms hop

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        inputs = self.extractor(wav.cpu().numpy(), sampling_rate=16000, return_tensors="pt")
        out = self.model(input_values=inputs["input_values"].to(self.device))
        return out.last_hidden_state[0, 2:]  # drop CLS / distillation tokens -> [patches, dim]


def build_feature_cache(manifest: str | Path, cache_root: Path, backbone: str, tokens_per_window: int = 64, max_windows: int = 3, device: torch.device = torch.device("cpu"), hf_cache: Optional[str] = None) -> dict[str, Any]:
    """Extract (or load) frozen token features for every clip in the manifest."""
    rows = read_jsonl(manifest)
    key = hashlib.sha256(f"{Path(manifest).resolve()}|{backbone}|{tokens_per_window}|{max_windows}|{len(rows)}".encode()).hexdigest()[:12]
    out = ensure_dir(Path(cache_root) / "audio" / f"features-{backbone.replace('/', '_')}-{key}")
    meta = out / "meta.json"

    def _load():
        info = read_json(meta)
        info["features"] = np.load(out / "features.npy", mmap_mode="r")
        info["rows"] = rows
        return info

    if meta.exists():
        return _load()
    with file_lock(out / ".build.lock"):  # concurrent jobs of one stage: one builds, the others wait and load
        if meta.exists():
            return _load()
        return _build_feature_cache(rows, out, meta, backbone, tokens_per_window, max_windows, device, hf_cache)


def _build_feature_cache(rows, out: Path, meta: Path, backbone: str, tokens_per_window: int, max_windows: int, device: torch.device, hf_cache: Optional[str]) -> dict[str, Any]:
    if backbone == "mel_cnn":
        bb = MelCNNBackbone().to(device)
        window = 16000 * 10
    else:
        bb = ASTBackbone(backbone, cache_dir=hf_cache, device=device)
        window = bb.window
    dim = int(bb.dim)
    feats = np.zeros((len(rows), max_windows * tokens_per_window, dim), dtype=np.float16)
    n_tokens = np.zeros(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        wav = torch.from_numpy(load_audio_16k(r["path"])).to(device)
        chunks = []
        for w in range(max_windows):
            seg = wav[w * window : (w + 1) * window]
            if seg.numel() < 16000:  # < 1 s remainder
                break
            if seg.numel() < window and backbone == "mel_cnn":
                pass
            tok = bb(seg)  # [t, dim]
            pooled = torch.nn.functional.adaptive_avg_pool1d(tok.T[None], tokens_per_window)[0].T  # [tokens_per_window, dim]
            chunks.append(pooled)
        if not chunks:
            chunks.append(bb(torch.nn.functional.pad(wav, (0, max(0, 16000 - wav.numel()))))[:tokens_per_window])
        z = torch.cat(chunks, dim=0)
        feats[i, : z.shape[0]] = z.float().cpu().numpy().astype(np.float16)
        n_tokens[i] = z.shape[0]
        if i % 200 == 0:
            print(f"[audio] features {i}/{len(rows)}", flush=True)
    atomic_save_npy(out / "features.npy", feats)
    atomic_save_npy(out / "n_tokens.npy", n_tokens)
    info = {"backbone": backbone, "dim": dim, "tokens_per_window": tokens_per_window, "max_windows": max_windows, "n_clips": len(rows), "cache_dir": str(out)}
    atomic_write_json(meta, info)
    info["features"] = np.load(out / "features.npy", mmap_mode="r")
    info["rows"] = rows
    return info


def load_n_tokens(info: Mapping[str, Any]) -> np.ndarray:
    return np.load(Path(info["cache_dir"]) / "n_tokens.npy")


# ----------------------------------------------------------------------------- text embeddings (retrieval)
def text_embeddings(captions: list[str], encoder: str, cache_root: Path, device: torch.device, hf_cache: Optional[str] = None) -> np.ndarray:
    key = hashlib.sha256((encoder + "|" + "\n".join(captions)).encode()).hexdigest()[:12]
    out = ensure_dir(Path(cache_root) / "audio" / "text") / f"{encoder.replace('/', '_')}-{key}.npy"
    if out.exists():
        return np.load(out)
    with file_lock(out.with_suffix(".lock")):
        if out.exists():
            return np.load(out)
        emb = _compute_text_embeddings(captions, encoder, device, hf_cache)
        atomic_save_npy(out, emb)
        return emb


def _compute_text_embeddings(captions: list[str], encoder: str, device: torch.device, hf_cache: Optional[str]) -> np.ndarray:
    if encoder == "hashing":
        rng = np.random.default_rng(0)
        dim = 256
        emb = np.zeros((len(captions), dim), dtype=np.float32)
        for i, c in enumerate(captions):
            for tok in c.lower().split():
                h = int(hashlib.sha256(tok.encode()).hexdigest()[:8], 16)
                g = np.random.default_rng(h)
                emb[i] += g.normal(size=dim).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-6
    else:
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(encoder, cache_dir=hf_cache)
        mdl = AutoModel.from_pretrained(encoder, cache_dir=hf_cache).to(device).eval()
        embs = []
        with torch.no_grad():
            for s in range(0, len(captions), 256):
                batch = tok(captions[s : s + 256], padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
                h = mdl(**batch).last_hidden_state
                m = batch["attention_mask"].unsqueeze(-1).float()
                pooled = (h * m).sum(1) / m.sum(1).clamp_min(1)
                embs.append(torch.nn.functional.normalize(pooled, dim=1).cpu())
        emb = torch.cat(embs).numpy().astype(np.float32)
    return emb


# ----------------------------------------------------------------------------- dataset resolution
def resolve_audio_manifest(ac: Mapping[str, Any], paths: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve ``audio.dataset`` to a validated manifest.

    * ``audiocaps`` (manuscript dataset, default): ``audio.manifest`` must point at a manifest built by
      :func:`caenl.audio.audiocaps.prepare_audiocaps`; it is validated strictly (paths, splits, captions,
      cross-split clips, hashes, minimum coverage ``audio.manifest_min_coverage``) before any training.
    * ``clotho`` (supplementary): downloaded from Zenodo on first use.
    * ``synthetic``: smoke tests.
    * any other name with ``audio.manifest``: generic manifest, validated strictly as well.
    """
    from .audiocaps import manifest_provenance, validate_audio_manifest

    dataset = str(ac.get("dataset", "audiocaps"))
    data_root, cache_root = paths.get("data_root", "data"), paths.get("cache_root", "cache")
    if dataset == "synthetic":
        info = prepare_synthetic_audio(cache_root, ac.get("synthetic_options", {}))
        return {"dataset": dataset, **info}
    if dataset == "clotho":
        info = prepare_clotho(data_root, cache_root)
        return {"dataset": dataset, **info, "provenance": manifest_provenance(info["manifest"])}
    manifest = ac.get("manifest")
    if not manifest:
        raise FileNotFoundError(f"audio.dataset={dataset!r} needs audio.manifest (for AudioCaps run `caenl preflight --download` to build <data_root>/audiocaps/manifest.jsonl)")
    rep = validate_audio_manifest(manifest, min_coverage=ac.get("manifest_min_coverage"), verify_hashes=str(ac.get("manifest_verify_hashes", "sample")))
    if not rep["valid"]:
        raise RuntimeError(f"audio manifest {manifest} failed validation: {rep['errors']}")
    return {"dataset": dataset, "manifest": str(manifest), "n_clips": rep["n_clips"], "splits": rep["splits"], "provenance": {**manifest_provenance(manifest), "validation": {k: v for k, v in rep.items() if k in ("coverage", "warnings", "manifest_sha256")}}}
