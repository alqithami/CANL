"""Audio-text retrieval: contrastive (InfoNCE) training of an audio tower over frozen AST tokens
against frozen sentence embeddings; DCR on the audio embedding.  Metrics on the evaluation
split: text->audio and audio->text Recall@{1,5,10} and median rank.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.control_loop import ControlLoop
from ..utils.io import atomic_write_json
from ..utils.jobctx import JobContext
from ..utils.seeding import derive_seed, load_rng_state, rng_state_dict
from .common import AudioTokenEncoder, _debug_stop, clip_batch, cosine_lr, split_indices
from .data import build_feature_cache, load_n_tokens, resolve_audio_manifest, text_embeddings


class RetrievalModel(nn.Module):
    def __init__(self, in_dim: int, text_dim: int, d_model: int = 512, embed_dim: int = 512, enc_layers: int = 2, n_heads: int = 8, temperature_init: float = 0.07):
        super().__init__()
        self.audio = AudioTokenEncoder(in_dim, d_model=d_model, n_layers=enc_layers, n_heads=n_heads)
        self.audio_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, embed_dim))
        self.text_proj = nn.Sequential(nn.Linear(text_dim, d_model), nn.GELU(), nn.Linear(d_model, embed_dim))
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(1 / temperature_init))))

    def encode_audio(self, tokens, mask):
        _, pooled = self.audio(tokens, mask)
        z = self.audio_proj(pooled)
        return z, {"audio_embed": z, "audio_pool": pooled}

    def encode_text(self, t):
        return self.text_proj(t.float())

    def forward(self, tokens, mask, text):
        za, feats = self.encode_audio(tokens, mask)
        zt = self.encode_text(text)
        za_n, zt_n = F.normalize(za, dim=1), F.normalize(zt, dim=1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * za_n @ zt_n.T
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        return loss, feats


def recall_metrics(sim: np.ndarray, clip_of_caption: np.ndarray, n_clips: int) -> dict[str, float]:
    """sim: [n_captions, n_clips] similarities."""
    out: dict[str, float] = {}
    # text -> audio
    ranks = []
    order = np.argsort(-sim, axis=1)
    for i in range(sim.shape[0]):
        ranks.append(int(np.where(order[i] == clip_of_caption[i])[0][0]) + 1)
    ranks = np.asarray(ranks)
    for k in (1, 5, 10):
        out[f"t2a_r{k}"] = float((ranks <= k).mean())
    out["median_rank"] = float(np.median(ranks))
    out["mean_rank"] = float(ranks.mean())
    # audio -> text: best rank among the clip's captions
    order_a = np.argsort(-sim.T, axis=1)  # [n_clips, n_captions]
    best = []
    for c in range(n_clips):
        caps = np.flatnonzero(clip_of_caption == c)
        pos = np.where(np.isin(order_a[c], caps))[0]
        best.append(int(pos.min()) + 1 if pos.size else sim.shape[0])
    best = np.asarray(best)
    for k in (1, 5, 10):
        out[f"a2t_r{k}"] = float((best <= k).mean())
    out["a2t_median_rank"] = float(np.median(best))
    return out


class RetrievalJob:
    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        self.cfg = ctx.config
        self.ac = dict(self.cfg.get("audio", {}))
        self.rc = dict(self.ac.get("retrieval", {}))
        self.method = dict(self.cfg.get("method", {}))
        self.device = ctx.device

    def setup(self) -> None:
        ctx, ac, rc = self.ctx, self.ac, self.rc
        paths = self.cfg.get("paths", {})
        hf_cache = str(Path(paths.get("data_root", "data")) / "hf")
        info = resolve_audio_manifest(ac, paths)
        self.manifest_provenance = info.get("provenance", {})
        ctx.event("audio_manifest", dataset=info.get("dataset"), manifest=info["manifest"], n_clips=info.get("n_clips"), manifest_sha256=(info.get("provenance") or {}).get("manifest_sha256"))
        feat = build_feature_cache(info["manifest"], paths.get("cache_root", "cache"), str(ac.get("backbone", "MIT/ast-finetuned-audioset-10-10-0.4593")), int(ac.get("tokens_per_window", 64)), int(ac.get("max_windows", 3)), self.device, hf_cache)
        self.rows = feat["rows"]
        self.features = feat["features"]
        self.n_tokens = load_n_tokens(feat)
        self.splits = split_indices(self.rows)
        self.train_clips = [int(i) for i in self.splits["train"]]
        # one caption per clip per epoch (avoids in-batch false negatives); pairs re-drawn every epoch
        self.pairs = [(i, 0) for i in self.train_clips]
        all_caps = [c for r in self.rows for c in r["captions"]]
        self.cap_offset = np.cumsum([0] + [len(r["captions"]) for r in self.rows])
        self.text_emb = torch.from_numpy(text_embeddings(all_caps, str(rc.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2")), Path(paths.get("cache_root", "cache")), self.device, hf_cache)).to(self.device)
        torch.manual_seed(derive_seed(ctx.seed, "init"))
        d_model = int(rc.get("d_model", 512))
        self.model = RetrievalModel(int(feat["dim"]), int(self.text_emb.shape[1]), d_model=d_model, embed_dim=int(rc.get("embed_dim", 512)), enc_layers=int(rc.get("encoder_layers", 2)), n_heads=int(rc.get("heads", 8)), temperature_init=float(rc.get("temperature_init", 0.07))).to(self.device)
        self.bs = int(rc.get("batch_size", 128))
        self.epochs = int(rc.get("epochs", 20))
        self.steps_per_epoch = max(1, len(self.pairs) // self.bs)
        self.total_steps = self.steps_per_epoch * self.epochs
        self.loop = ControlLoop({"audio_embed": int(rc.get("embed_dim", 512)), "audio_pool": d_model}, str(self.method.get("regularizer", "none")), float(self.method.get("dcr_lambda", 0.0)), self.cfg, self.device, derive_seed(ctx.seed, "controller"), self.total_steps, k=int(self.cfg.get("collapse", {}).get("truncated_spectral_rank", 128)), configured_rank=int(ac.get("configured_rank", 96)), target_effective_rank=ac.get("target_effective_rank"), cadence=int(ac.get("monitoring_cadence_steps", 20)), error_mode=str(ac.get("controller_error_mode", "absolute_log_ratio")), ema_decay=float(self.cfg.get("collapse", {}).get("covariance_ema_decay", 0.99)))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(rc.get("lr", 3e-4)), weight_decay=float(rc.get("weight_decay", 0.01)))
        self.step = 0
        self.amp = self.device.type == "cuda"
        n_ctrl = int(min(int(rc.get("ctrl_val_clips", 128)), max(1, len(self.splits["valid"]) // 2)))
        self.ctrl_val = self.splits["valid"][:n_ctrl]  # Full MACC reward only
        self.report_val = self.splits["valid"][n_ctrl:]
        ctx.event("setup", n_train_pairs=len(self.pairs), n_valid=len(self.splits["valid"]), n_test=len(self.splits["test"]), steps=self.total_steps, params=sum(p.numel() for p in self.model.parameters()), text_dim=int(self.text_emb.shape[1]), dcr_ranks=self.loop.summary()["dcr_target_ranks"], targets=self.loop.targets)

    def batch(self, pairs):
        idx = np.asarray([i for i, _ in pairs])
        x, mask = clip_batch(self.features, self.n_tokens, idx, self.device)
        cap_ids = torch.as_tensor([int(self.cap_offset[i] + k) for i, k in pairs], device=self.device)
        return x, mask, self.text_emb[cap_ids]

    @torch.no_grad()
    def evaluate(self, clips: np.ndarray) -> dict[str, float]:
        self.model.eval()
        za = []
        for s in range(0, len(clips), 256):
            x, mask = clip_batch(self.features, self.n_tokens, clips[s : s + 256], self.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
                z, _ = self.model.encode_audio(x, mask)
            za.append(F.normalize(z.float(), dim=1))
        za = torch.cat(za)
        cap_ids, clip_of_cap = [], []
        for j, i in enumerate(clips):
            for k in range(len(self.rows[int(i)]["captions"])):
                cap_ids.append(int(self.cap_offset[int(i)] + k))
                clip_of_cap.append(j)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
            zt = F.normalize(self.model.encode_text(self.text_emb[torch.as_tensor(cap_ids, device=self.device)]).float(), dim=1)
        sim = (zt @ za.T).cpu().numpy()
        self.model.train()
        out = recall_metrics(sim, np.asarray(clip_of_cap), len(clips))
        # contrastive validation loss on clip-caption pairs (first caption)
        return out

    @torch.no_grad()
    def val_loss(self, clips: np.ndarray) -> float:
        self.model.eval()
        pairs = [(int(i), 0) for i in clips]
        total, n = 0.0, 0
        for s in range(0, len(pairs), self.bs):
            x, mask, t = self.batch(pairs[s : s + self.bs])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
                loss, _ = self.model(x, mask, t)
            total += float(loss) * len(pairs[s : s + self.bs])
            n += len(pairs[s : s + self.bs])
        self.model.train()
        return total / max(n, 1)

    def save(self) -> None:
        self.ctx.save_checkpoint("state.pt", {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "step": self.step, "loop": self.loop.state_dict(), "rng": rng_state_dict()})

    def restore(self) -> bool:
        st = self.ctx.load_checkpoint("state.pt")
        if st is None:
            return False
        self.model.load_state_dict(st["model"])
        self.optimizer.load_state_dict(st["optimizer"])
        self.step = int(st["step"])
        self.loop.load_state_dict(st["loop"])
        load_rng_state(st["rng"])
        self.ctx.event("resumed", step=self.step)
        return True

    def run(self) -> dict[str, Any]:
        ctx, rc = self.ctx, self.rc
        self.setup()
        self.restore()
        self.model.train()
        log_every = int(rc.get("log_every_steps", 20))
        ckpt_every = int(rc.get("checkpoint_every_steps", 500))
        warmup = int(rc.get("warmup_steps", max(1, self.steps_per_epoch)))
        t0 = time.perf_counter()
        order = None
        while self.step < self.total_steps:
            epoch = self.step // self.steps_per_epoch
            if order is None or self.step % self.steps_per_epoch == 0:
                erng = np.random.default_rng(derive_seed(ctx.seed, "epoch", epoch))
                self.pairs = [(i, int(erng.integers(len(self.rows[i]["captions"])))) for i in self.train_clips]
                order = erng.permutation(len(self.pairs))
            b = self.step % self.steps_per_epoch
            pairs = [self.pairs[j] for j in order[b * self.bs : (b + 1) * self.bs]]
            x, mask, t = self.batch(pairs)
            lr = cosine_lr(self.step, self.total_steps, float(rc.get("lr", 3e-4)), warmup)
            for g in self.optimizer.param_groups:
                g["lr"] = lr
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
                ce, feats = self.model(x, mask, t)
            dcr_loss, per_layer = self.loop.dcr_loss(feats)
            loss = ce + dcr_loss
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            info = self.loop.update({k: v.detach() for k, v in feats.items()}, value_fn=lambda: -self.val_loss(self.ctrl_val), extra_state=(self.step / max(1, self.total_steps),))
            self.step += 1
            if info:
                ctx.log_metrics({"kind": "controller", "step": self.step, **{k: v for k, v in info.items() if isinstance(v, (int, float)) or v is None}})
            if self.step % log_every == 0:
                ctx.log_metrics({"kind": "train", "step": self.step, "epoch": epoch, "loss": float(loss.detach()), "contrastive": float(ce.detach()), "dcr": float(dcr_loss.detach()), "logit_scale": float(self.model.logit_scale.exp()), **{f"dcr/{k}": v for k, v in per_layer.items()}, **self.loop.record()})
            if self.step % self.steps_per_epoch == 0:
                ev = self.evaluate(self.report_val)
                ctx.log_metrics({"kind": "eval", "step": self.step, "epoch": epoch, **ev, "val_loss": self.val_loss(self.report_val), **self.loop.record()})
                ctx.event("epoch", epoch=epoch, t2a_r1=round(ev["t2a_r1"], 4), a2t_r1=round(ev["a2t_r1"], 4), lambdas=self.loop.lambdas, eff_rank={k: round(v, 1) for k, v in self.loop.monitor.kappa().items()})
            if self.step % ckpt_every == 0 and self.step < self.total_steps:
                self.save()
            _debug_stop(self.step)
        train_time = time.perf_counter() - t0
        self.save()
        final = self.evaluate(self.splits["test"])
        final["val_loss"] = self.val_loss(self.report_val)
        atomic_write_json(ctx.artifact_dir / "controller_trajectory.json", {"regularizer": self.loop.regularizer, "targets": self.loop.targets, "history": self.loop.summary()["history"]})
        ctx.event("final", **{k: round(v, 4) for k, v in final.items()})
        return {
            "task": "audio_retrieval",
            "dataset": self.ac.get("dataset", "audiocaps"),
            "manifest_provenance": getattr(self, "manifest_provenance", {}),
            "method": self.method.get("name"),
            "regularizer": self.loop.regularizer,
            "dcr_lambda": self.method.get("dcr_lambda"),
            "seed": ctx.seed,
            "model_params": sum(p.numel() for p in self.model.parameters()),
            "controller_params": self.loop.summary()["controller_params"],
            "steps": self.step,
            "final": {**final, **{f"effective_rank/{k}": v for k, v in self.loop.monitor.kappa().items()}, **{f"lambda/{k}": v for k, v in self.loop.lambdas.items()}},
            "compute": {"train_time_s": train_time, "steps_per_s": self.step / max(train_time, 1e-6)},
            "control": {k: v for k, v in self.loop.summary().items() if k != "history"},
        }


def run(ctx: JobContext) -> dict[str, Any]:
    return RetrievalJob(ctx).run()
