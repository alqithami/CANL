"""Audio captioning: frozen AST tokens -> trainable transformer encoder -> learned prefix -> GPT-2.

DCR regularises the prefix embeddings (the audio-conditioned representation injected into the
language model) and the pooled encoder output.  Methods: baseline / fixed_dcr / macc_lite /
full_macc with identical data order, schedule and decoding (beam search).  Metrics on the
evaluation split: CIDEr-D, BLEU-4, ROUGE-L (+ METEOR/SPICE when Java is available), val loss.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.control_loop import ControlLoop
from ..utils.io import atomic_write_json
from ..utils.jobctx import JobContext
from ..utils.seeding import derive_seed, load_rng_state, rng_state_dict
from .common import AudioTokenEncoder, CharTokenizer, HFTokenizer, _debug_stop, clip_batch, cosine_lr, pad_batch, split_indices
from .data import build_feature_cache, load_n_tokens, resolve_audio_manifest
from .metrics import caption_metrics


class PrefixCaptioner(nn.Module):
    def __init__(self, in_dim: int, decoder: nn.Module, d_model: int, prefix_len: int = 8, enc_layers: int = 2, n_heads: int = 8):
        super().__init__()
        self.encoder = AudioTokenEncoder(in_dim, d_model=d_model, n_layers=enc_layers, n_heads=n_heads)
        self.queries = nn.Parameter(torch.randn(prefix_len, d_model) * 0.02)
        self.cross = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.prefix_norm = nn.LayerNorm(d_model)
        self.decoder = decoder
        self.prefix_len = prefix_len

    def prefix(self, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        seq, pooled = self.encoder(tokens, mask)
        q = self.queries[None].expand(tokens.shape[0], -1, -1)
        p, _ = self.cross(q, seq, seq, key_padding_mask=~mask)
        p = self.prefix_norm(p + q)
        return p, {"prefix": p.reshape(-1, p.shape[-1]), "audio_enc": pooled}

    def forward(self, tokens, mask, input_ids, attn_mask):
        prefix, feats = self.prefix(tokens, mask)
        wte = self.decoder.get_input_embeddings()
        emb = torch.cat([prefix, wte(input_ids)], dim=1)
        am = torch.cat([torch.ones(prefix.shape[:2], dtype=attn_mask.dtype, device=attn_mask.device), attn_mask], dim=1)
        out = self.decoder(inputs_embeds=emb, attention_mask=am)
        # positions P..P+L-2 (BOS, w1, ..., w_{L-2}) predict input_ids[1:] = (w1, ..., EOS)
        logits = out.logits[:, self.prefix_len : -1]
        targets = input_ids[:, 1:]
        tmask = attn_mask[:, 1:].bool()
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
        loss = (loss * tmask.reshape(-1).float()).sum() / tmask.float().sum().clamp_min(1)
        return loss, feats

    @torch.no_grad()
    def generate(self, tokens, mask, bos_id: int, eos_id: int, pad_id: int, max_new_tokens: int = 30, num_beams: int = 3) -> list[list[int]]:
        prefix, _ = self.prefix(tokens, mask)
        wte = self.decoder.get_input_embeddings()
        bos = torch.full((tokens.shape[0], 1), bos_id, dtype=torch.long, device=tokens.device)
        emb = torch.cat([prefix, wte(bos)], dim=1)
        am = torch.ones(emb.shape[:2], dtype=torch.long, device=tokens.device)
        out = self.decoder.generate(inputs_embeds=emb, attention_mask=am, max_new_tokens=max_new_tokens, num_beams=num_beams, eos_token_id=eos_id, pad_token_id=pad_id, do_sample=False, early_stopping=True if num_beams > 1 else False)
        return [row.tolist() for row in out]


class CaptioningJob:
    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        self.cfg = ctx.config
        self.ac = dict(self.cfg.get("audio", {}))
        self.cc = dict(self.ac.get("captioning", {}))
        self.method = dict(self.cfg.get("method", {}))
        self.device = ctx.device

    def setup(self) -> None:
        ctx, ac, cc = self.ctx, self.ac, self.cc
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
        dec_name = str(cc.get("decoder", "gpt2"))
        from transformers import GPT2Config, GPT2LMHeadModel, AutoModelForCausalLM

        torch.manual_seed(derive_seed(ctx.seed, "init"))
        if dec_name == "tiny_random":
            self.tok = CharTokenizer()
            conf = GPT2Config(vocab_size=self.tok.vocab_size, n_positions=128, n_embd=64, n_layer=2, n_head=2, bos_token_id=self.tok.bos_token_id, eos_token_id=self.tok.eos_token_id)
            self.decoder = GPT2LMHeadModel(conf)
        else:
            self.tok = HFTokenizer(dec_name, cache_dir=hf_cache)
            self.decoder = AutoModelForCausalLM.from_pretrained(dec_name, cache_dir=hf_cache)
        d_model = int(self.decoder.config.n_embd if hasattr(self.decoder.config, "n_embd") else self.decoder.config.hidden_size)
        self.model = PrefixCaptioner(int(feat["dim"]), self.decoder, d_model, prefix_len=int(cc.get("prefix_length", 8)), enc_layers=int(cc.get("encoder_layers", 2)), n_heads=int(cc.get("heads", 8 if d_model % 8 == 0 else 2))).to(self.device)
        self.pairs = [(int(i), c) for i in self.splits["train"] for c in self.rows[i]["captions"]]
        self.bs = int(cc.get("batch_size", 32))
        self.epochs = int(cc.get("epochs", 10))
        self.steps_per_epoch = max(1, len(self.pairs) // self.bs)
        self.total_steps = self.steps_per_epoch * self.epochs
        self.loop = ControlLoop({"prefix": d_model, "audio_enc": d_model}, str(self.method.get("regularizer", "none")), float(self.method.get("dcr_lambda", 0.0)), self.cfg, self.device, derive_seed(ctx.seed, "controller"), self.total_steps, k=int(self.cfg.get("collapse", {}).get("truncated_spectral_rank", 128)), configured_rank=int(ac.get("configured_rank", max(1, d_model // 8))), target_effective_rank=ac.get("target_effective_rank"), cadence=int(ac.get("monitoring_cadence_steps", 20)), error_mode=str(ac.get("controller_error_mode", "absolute_log_ratio")), ema_decay=float(self.cfg.get("collapse", {}).get("covariance_ema_decay", 0.99)))
        dec_params = list(self.decoder.parameters())
        dec_ids = set(id(p) for p in dec_params)
        head_params = [p for p in self.model.parameters() if id(p) not in dec_ids]
        self.optimizer = torch.optim.AdamW([{"params": head_params, "lr": float(cc.get("lr", 1e-4))}, {"params": dec_params, "lr": float(cc.get("decoder_lr", 5e-5))}], weight_decay=float(cc.get("weight_decay", 0.01)))
        self.base_lrs = [float(cc.get("lr", 1e-4)), float(cc.get("decoder_lr", 5e-5))]
        self.rng = np.random.default_rng(derive_seed(ctx.seed, "order"))
        self.step = 0
        self.amp = self.device.type == "cuda"
        self.max_len = int(cc.get("max_caption_tokens", 48))
        n_ctrl = int(min(int(cc.get("ctrl_val_clips", 64)), max(1, len(self.splits["valid"]) // 2)))
        self.ctrl_val = self.splits["valid"][:n_ctrl]  # Full MACC reward only
        self.report_val = self.splits["valid"][n_ctrl:]  # reported validation loss
        ctx.event("setup", n_train_pairs=len(self.pairs), n_valid=len(self.splits["valid"]), n_test=len(self.splits["test"]), steps=self.total_steps, decoder=dec_name, params=sum(p.numel() for p in self.model.parameters()), dcr_ranks=self.loop.summary()["dcr_target_ranks"], targets=self.loop.targets)

    def batch(self, pairs: list[tuple[int, str]]):
        idx = np.asarray([i for i, _ in pairs])
        x, mask = clip_batch(self.features, self.n_tokens, idx, self.device)
        ids, am = pad_batch([self.tok.encode(c, self.max_len) for _, c in pairs], self.tok.pad_token_id)
        return x, mask, ids.to(self.device), am.long().to(self.device)

    @torch.no_grad()
    def val_loss(self, clips: np.ndarray) -> float:
        self.model.eval()
        pairs = [(int(i), c) for i in clips for c in self.rows[i]["captions"]]
        total, n = 0.0, 0
        for s in range(0, len(pairs), self.bs):
            x, mask, ids, am = self.batch(pairs[s : s + self.bs])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
                loss, _ = self.model(x, mask, ids, am)
            total += float(loss) * len(pairs[s : s + self.bs])
            n += len(pairs[s : s + self.bs])
        self.model.train()
        return total / max(n, 1)

    @torch.no_grad()
    def decode_split(self, clips: np.ndarray) -> tuple[list[str], list[list[str]]]:
        self.model.eval()
        cands: list[str] = []
        refs: list[list[str]] = []
        bs = int(self.cc.get("decode_batch_size", 64))
        for s in range(0, len(clips), bs):
            idx = clips[s : s + bs]
            x, mask = clip_batch(self.features, self.n_tokens, idx, self.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
                outs = self.model.generate(x, mask, self.tok.bos_token_id, self.tok.eos_token_id, self.tok.pad_token_id, int(self.cc.get("max_new_tokens", 30)), int(self.cc.get("beam_size", 3)))
            for i, o in zip(idx, outs):
                cands.append(self.tok.decode(o))
                refs.append(list(self.rows[int(i)]["captions"]))
        self.model.train()
        return cands, refs

    def save(self) -> None:
        self.ctx.save_checkpoint("state.pt", {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "step": self.step, "loop": self.loop.state_dict(), "rng": rng_state_dict(), "np_rng": self.rng.bit_generator.state})

    def restore(self) -> bool:
        st = self.ctx.load_checkpoint("state.pt")
        if st is None:
            return False
        self.model.load_state_dict(st["model"])
        self.optimizer.load_state_dict(st["optimizer"])
        self.step = int(st["step"])
        self.loop.load_state_dict(st["loop"])
        load_rng_state(st["rng"])
        self.rng.bit_generator.state = st["np_rng"]
        self.ctx.event("resumed", step=self.step)
        return True

    def run(self) -> dict[str, Any]:
        ctx, cc = self.ctx, self.cc
        self.setup()
        self.restore()
        self.model.train()
        log_every = int(cc.get("log_every_steps", 20))
        ckpt_every = int(cc.get("checkpoint_every_steps", 500))
        warmup = int(cc.get("warmup_steps", max(1, self.steps_per_epoch // 2)))
        t0 = time.perf_counter()
        order = None
        while self.step < self.total_steps:
            epoch = self.step // self.steps_per_epoch
            if order is None or self.step % self.steps_per_epoch == 0:
                # deterministic per-epoch permutation (reproducible after resume)
                order = np.random.default_rng(derive_seed(ctx.seed, "epoch", epoch)).permutation(len(self.pairs))
            b = self.step % self.steps_per_epoch
            batch_pairs = [self.pairs[j] for j in order[b * self.bs : (b + 1) * self.bs]]
            x, mask, ids, am = self.batch(batch_pairs)
            lr_scale = cosine_lr(self.step, self.total_steps, 1.0, warmup)
            for g, base in zip(self.optimizer.param_groups, self.base_lrs):
                g["lr"] = base * lr_scale
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
                ce, feats = self.model(x, mask, ids, am)
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
                ctx.log_metrics({"kind": "train", "step": self.step, "epoch": epoch, "loss": float(loss.detach()), "ce": float(ce.detach()), "dcr": float(dcr_loss.detach()), **{f"dcr/{k}": v for k, v in per_layer.items()}, **self.loop.record()})
            if self.step % self.steps_per_epoch == 0:
                vl = self.val_loss(self.report_val)
                ctx.log_metrics({"kind": "eval", "step": self.step, "epoch": epoch, "val_loss": vl, **self.loop.record()})
                ctx.event("epoch", epoch=epoch, val_loss=round(vl, 4), lambdas=self.loop.lambdas, eff_rank={k: round(v, 1) for k, v in self.loop.monitor.kappa().items()})
            if self.step % ckpt_every == 0 and self.step < self.total_steps:
                self.save()
            _debug_stop(self.step)
        train_time = time.perf_counter() - t0
        self.save()
        with ctx.timer("decode"):
            cands, refs = self.decode_split(self.splits["test"])
        # official pycocoevalcap values are authoritative; a required metric that cannot be computed raises (no silent NaN)
        metrics = caption_metrics(cands, refs, java=bool(cc.get("java_metrics", True)), spice=bool(cc.get("spice", True)), official=bool(cc.get("official_metrics", True)))
        ctx.event("caption_metrics", source=metrics.get("metrics_source"), tokenizer=metrics.get("tokenizer"), native_within_tolerance=metrics.get("native_within_tolerance"), **{k: round(float(v), 5) for k, v in metrics.items() if k.startswith("discrepancy_")})
        metrics["val_loss"] = self.val_loss(self.report_val)
        metrics["test_loss"] = self.val_loss(self.splits["test"])
        atomic_write_json(ctx.artifact_dir / "captions_test.json", [{"id": self.rows[int(i)]["id"], "candidate": c, "references": r} for i, c, r in zip(self.splits["test"], cands, refs)])
        atomic_write_json(ctx.artifact_dir / "controller_trajectory.json", {"regularizer": self.loop.regularizer, "targets": self.loop.targets, "history": self.loop.summary()["history"]})
        ctx.event("final", **{k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})
        return {
            "task": "audio_captioning",
            "metrics_source": metrics.get("metrics_source"),
            "dataset": self.ac.get("dataset", "audiocaps"),
            "manifest_provenance": getattr(self, "manifest_provenance", {}),
            "method": self.method.get("name"),
            "regularizer": self.loop.regularizer,
            "dcr_lambda": self.method.get("dcr_lambda"),
            "seed": ctx.seed,
            "model_params": sum(p.numel() for p in self.model.parameters()),
            "controller_params": self.loop.summary()["controller_params"],
            "steps": self.step,
            "final": {**metrics, **{f"effective_rank/{k}": v for k, v in self.loop.monitor.kappa().items()}, **{f"lambda/{k}": v for k, v in self.loop.lambdas.items()}},
            "compute": {"train_time_s": train_time, "steps_per_s": self.step / max(train_time, 1e-6), "decode_time_s": ctx.timings.get("decode")},
            "control": {k: v for k, v in self.loop.summary().items() if k != "history"},
            "examples": [{"candidate": c, "reference": r[0]} for c, r in list(zip(cands, refs))[:10]],
        }


def run(ctx: JobContext) -> dict[str, Any]:
    return CaptioningJob(ctx).run()
