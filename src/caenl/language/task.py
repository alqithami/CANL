"""C4 causal language modelling with spectrum DCR on transformer hidden states.

Methods: ``baseline`` (no DCR), ``fixed_dcr``, ``macc_lite``, ``full_macc`` — identical data
order, token budget, optimiser and evaluation; only the regulariser differs.
Metrics (held-out C4 validation tokens): NLL (nats/token), perplexity, token-level ECE,
effective rank / anisotropy of the monitored hidden states; curves in ``metrics.jsonl``.
"""
from __future__ import annotations

import math
import time
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..core.control_loop import ControlLoop
from ..utils.io import atomic_write_json
from ..utils.jobctx import JobContext
from ..utils.seeding import derive_seed, load_rng_state, rng_state_dict
from .data import load_token_cache, prepare_c4, prepare_synthetic_text


def _debug_stop(step: int) -> None:
    """Testing hook: CAENL_DEBUG_STOP_AT_STEP=N aborts the process after step N (resume tests)."""
    import os

    v = os.environ.get("CAENL_DEBUG_STOP_AT_STEP")
    if v and step >= int(v):
        raise SystemExit(f"debug stop at step {step}")


def _build_model(name: str, init: str, vocab_size: int, seq_len: int, device: torch.device, options: dict[str, Any]):
    from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

    if name == "tiny_random":
        cfg = GPT2Config(vocab_size=vocab_size, n_positions=seq_len, n_embd=int(options.get("n_embd", 64)), n_layer=int(options.get("n_layer", 2)), n_head=int(options.get("n_head", 2)), bos_token_id=0, eos_token_id=0, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
        model = GPT2LMHeadModel(cfg)
    elif init == "pretrained":
        model = AutoModelForCausalLM.from_pretrained(name).float()  # version-agnostic (no torch_dtype/dtype kwarg)
    else:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(name)
        model = AutoModelForCausalLM.from_config(cfg)
    return model.to(device)


def _token_ece(logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> tuple[float, float]:
    probs = F.softmax(logits.float(), dim=-1)
    conf, pred = probs.max(dim=-1)
    correct = (pred == targets).float()
    bins = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    ece = torch.zeros((), device=logits.device)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            ece += m.float().mean() * (conf[m].mean() - correct[m].mean()).abs()
    return float(ece), float(correct.mean())


class LanguageJob:
    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        self.cfg = ctx.config
        self.lc = dict(self.cfg.get("language", {}))
        self.method = dict(self.cfg.get("method", {}))
        self.device = ctx.device

    def setup(self) -> None:
        ctx, lc = self.ctx, self.lc
        paths = self.cfg.get("paths", {})
        seq_len = int(lc.get("sequence_length", 1024))
        if lc.get("dataset", "allenai/c4") == "synthetic":
            self.info = prepare_synthetic_text(paths.get("cache_root", "cache"), {"vocab_size": lc.get("vocab_size", 256), "train_tokens": lc.get("train_tokens", 200000), "validation_tokens": lc.get("validation_tokens", 20000), "sequence_length": seq_len})
        else:
            self.info = prepare_c4(paths.get("data_root", "data"), paths.get("cache_root", "cache"), {"tokenizer": lc.get("model", "gpt2") if lc.get("model") != "tiny_random" else "gpt2", "train_shards": lc.get("train_shards"), "validation_shards": lc.get("validation_shards"), "sequence_length": seq_len, "train_tokens": lc.get("train_tokens", 100_000_000), "validation_tokens": lc.get("validation_tokens", 2_000_000)})
        self.train_tokens_arr, self.val_tokens_arr = load_token_cache(self.info)
        self.seq_len = seq_len
        n_train_seq = (len(self.train_tokens_arr) - 1) // seq_len
        n_val_seq = (len(self.val_tokens_arr) - 1) // seq_len
        self.n_train_seq, self.n_val_seq = n_train_seq, n_val_seq
        self.batch_seqs = max(1, int(lc.get("batch_tokens", 16384)) // seq_len)
        self.total_steps = int(int(lc.get("train_tokens", 100_000_000)) // (self.batch_seqs * seq_len))
        self.total_steps = max(1, min(self.total_steps, n_train_seq // self.batch_seqs))
        vocab = int(self.info.get("vocab_size", 50257))
        torch.manual_seed(derive_seed(ctx.seed, "init"))
        self.model = _build_model(str(lc.get("model", "gpt2")), str(lc.get("init", "pretrained")), vocab, seq_len, self.device, lc.get("model_options", {}))
        self.model.config.use_cache = False
        d = int(self.model.config.n_embd if hasattr(self.model.config, "n_embd") else self.model.config.hidden_size)
        n_layers = int(self.model.config.n_layer if hasattr(self.model.config, "n_layer") else self.model.config.num_hidden_layers)
        wanted = [int(i) for i in lc.get("monitored_layers", [n_layers // 2, n_layers])]
        self.mon_layers = [i for i in wanted if 1 <= i <= n_layers]
        if not self.mon_layers:
            self.mon_layers = [n_layers]
        layer_dims = {f"h{i}": d for i in self.mon_layers}

        def _layer_mapping(value: Any, default: int | float) -> int | float | dict[str, int | float]:
            if not isinstance(value, Mapping):
                return default if value is None else value
            out: dict[str, int | float] = {}
            for layer in layer_dims:
                idx = layer[1:] if layer.startswith("h") else layer
                candidate = value.get(layer, value.get(idx, value.get(int(idx) if str(idx).isdigit() else idx, default)))
                out[layer] = candidate
            return out

        configured_rank = _layer_mapping(lc.get("configured_rank"), max(1, d // 4))
        target_raw = lc.get("target_effective_rank")
        target_effective_rank = configured_rank if target_raw is None else _layer_mapping(target_raw, max(1, d // 4))
        self.loop = ControlLoop(
            layer_dims,
            str(self.method.get("regularizer", "none")),
            float(self.method.get("dcr_lambda", 0.0)),
            self.cfg,
            self.device,
            derive_seed(ctx.seed, "controller"),
            self.total_steps,
            k=int(self.cfg.get("collapse", {}).get("truncated_spectral_rank", 128)),
            configured_rank=configured_rank,
            target_effective_rank=target_effective_rank,
            cadence=int(lc.get("monitoring_cadence_steps", 50)),
            error_mode=str(lc.get("controller_error_mode", "absolute_log_ratio")),
            ema_decay=float(self.cfg.get("collapse", {}).get("covariance_ema_decay", 0.99)),
        )
        self.token_subsample = int(lc.get("dcr_token_subsample", 4096))
        decay, no_decay = [], []
        for n_, p in self.model.named_parameters():
            (no_decay if p.ndim <= 1 or n_.endswith("bias") or "ln" in n_ or "norm" in n_ else decay).append(p)
        self.optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": float(lc.get("weight_decay", 0.01))}, {"params": no_decay, "weight_decay": 0.0}], lr=float(lc.get("lr", 5e-5)), betas=(0.9, 0.95))
        self.order_gen = np.random.default_rng(derive_seed(ctx.seed, "order"))
        self.order = self.order_gen.permutation(n_train_seq)
        self.sub_gen = torch.Generator(device="cpu").manual_seed(derive_seed(ctx.seed, "token_subsample"))
        self.step = 0
        self.amp = self.device.type == "cuda" and bool(lc.get("amp", True))
        # fixed controller-validation subset (first sequences of the validation cache) and full validation set
        n_ctrl = int(min(int(lc.get("ctrl_val_sequences", 16)), max(1, n_val_seq // 4)))
        self.ctrl_val_seqs = np.arange(n_ctrl)  # Full MACC reward only; excluded from every reported metric
        self.eval_seqs = np.arange(n_ctrl, min(n_ctrl + int(lc.get("eval_sequences", 200)), n_val_seq))
        self.final_eval_seqs = np.arange(n_ctrl, n_val_seq)
        ctx.event("setup", model=lc.get("model"), params=sum(p.numel() for p in self.model.parameters()), n_train_seq=n_train_seq, n_val_seq=n_val_seq, steps=self.total_steps, batch_seqs=self.batch_seqs, seq_len=seq_len, monitored_layers=self.mon_layers, dcr_ranks=self.loop.summary()["dcr_target_ranks"], targets=self.loop.targets)

    def batch(self, seq_ids: np.ndarray, arr: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        L = self.seq_len
        x = np.stack([arr[i * L : i * L + L + 1].astype(np.int64) for i in seq_ids])
        t = torch.from_numpy(x).to(self.device)
        return t[:, :-1], t[:, 1:]

    def forward(self, inp: torch.Tensor, need_hidden: bool = True):
        out = self.model(input_ids=inp, output_hidden_states=need_hidden)
        feats = {}
        if need_hidden:
            hs = out.hidden_states
            for i in self.mon_layers:
                feats[f"h{i}"] = hs[i]
        return out.logits, feats

    def lr_at(self, step: int) -> float:
        base = float(self.lc.get("lr", 5e-5))
        warm = int(self.lc.get("warmup_steps", 200))
        if step < warm:
            return base * (step + 1) / max(1, warm)
        p = min(1.0, (step - warm) / max(1, self.total_steps - warm))
        return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    @torch.no_grad()
    def evaluate(self, seqs: np.ndarray, bs: Optional[int] = None, with_hidden: bool = False) -> dict[str, Any]:
        self.model.eval()
        bs = bs or self.batch_seqs
        nll_sum, n_tok, ece_acc = 0.0, 0, []
        rank_feats = {f"h{i}": [] for i in self.mon_layers}
        for s in range(0, len(seqs), bs):
            inp, tgt = self.batch(seqs[s : s + bs], self.val_tokens_arr)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
                logits, feats = self.forward(inp, need_hidden=with_hidden)
            logits = logits.float()
            nll_sum += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="sum"))
            n_tok += int(tgt.numel())
            ece_acc.append(_token_ece(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1)))
            if with_hidden:
                for k, v in feats.items():
                    flat = v.float().reshape(-1, v.shape[-1])
                    idx = torch.randperm(flat.shape[0], generator=self.sub_gen)[:2048].to(flat.device)
                    rank_feats[k].append(flat[idx])
        self.model.train()
        nll = nll_sum / max(n_tok, 1)
        out = {"validation_nll": nll, "perplexity": math.exp(min(nll, 50)), "token_ece": float(np.mean([e for e, _ in ece_acc])), "token_acc": float(np.mean([a for _, a in ece_acc])), "n_tokens": n_tok}
        if with_hidden:
            from ..core.nc import spectrum_summary

            for k, chunks in rank_feats.items():
                z = torch.cat(chunks)[:16384]
                z = z - z.mean(dim=0, keepdim=True)
                cov = z.T @ z / z.shape[0]
                sp = spectrum_summary(cov, k=int(self.cfg.get("collapse", {}).get("truncated_spectral_rank", 128)))
                out[f"effective_rank/{k}"] = sp["effective_rank"]
                out[f"top1_fraction/{k}"] = sp["top1_fraction"]
                zn = F.normalize(z[:4096], dim=1)
                out[f"mean_cosine/{k}"] = float((zn @ zn.T).mean())
        return out

    def ctrl_value(self) -> float:
        return -self.evaluate(self.ctrl_val_seqs)["validation_nll"]

    def save(self) -> None:
        self.ctx.save_checkpoint("state.pt", {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "step": self.step, "loop": self.loop.state_dict(), "rng": rng_state_dict(), "order": self.order})

    def restore(self) -> bool:
        st = self.ctx.load_checkpoint("state.pt")
        if st is None:
            return False
        self.model.load_state_dict(st["model"])
        self.optimizer.load_state_dict(st["optimizer"])
        self.step = int(st["step"])
        self.loop.load_state_dict(st["loop"])
        load_rng_state(st["rng"])
        order = np.asarray(st["order"])
        if len(order) == self.n_train_seq:
            self.order = order
        self.ctx.event("resumed", step=self.step)
        return True

    def run(self) -> dict[str, Any]:
        ctx, lc = self.ctx, self.lc
        self.setup()
        self.restore()
        self.model.train()
        eval_every = int(lc.get("eval_every_steps", 500))
        ckpt_every = int(lc.get("checkpoint_every_steps", 1000))
        log_every = int(lc.get("log_every_steps", 20))
        t0 = time.perf_counter()
        tokens_done = 0
        first_eval = self.evaluate(self.eval_seqs, with_hidden=True)
        ctx.log_metrics({"kind": "eval", "step": self.step, **first_eval, **self.loop.record()})
        while self.step < self.total_steps:
            ids = self.order[self.step * self.batch_seqs : (self.step + 1) * self.batch_seqs]
            inp, tgt = self.batch(ids, self.train_tokens_arr)
            lr = self.lr_at(self.step)
            for g in self.optimizer.param_groups:
                g["lr"] = lr
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
                logits, feats = self.forward(inp, need_hidden=True)
                ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1))
            sub = {}
            for k, v in feats.items():
                flat = v.reshape(-1, v.shape[-1])
                if flat.shape[0] > self.token_subsample:
                    idx = torch.randperm(flat.shape[0], generator=self.sub_gen)[: self.token_subsample].to(flat.device)
                    flat = flat[idx]
                sub[k] = flat
            dcr_loss, per_layer = self.loop.dcr_loss(sub)
            loss = ce + dcr_loss
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(lc.get("grad_clip", 1.0)))
            self.optimizer.step()
            info = self.loop.update({k: v.detach() for k, v in sub.items()}, value_fn=self.ctrl_value, extra_state=(self.step / max(1, self.total_steps),))
            self.step += 1
            tokens_done += int(inp.numel())
            if info:
                ctx.log_metrics({"kind": "controller", "step": self.step, **{k: v for k, v in info.items() if isinstance(v, (int, float)) or v is None}})
            if self.step % log_every == 0:
                el = time.perf_counter() - t0
                ctx.log_metrics({"kind": "train", "step": self.step, "loss": float(loss.detach()), "ce": float(ce.detach()), "dcr": float(dcr_loss.detach()), "lr": lr, "tokens_per_s": tokens_done / max(el, 1e-6), **{f"dcr/{k}": v for k, v in per_layer.items()}, **self.loop.record()})
            if self.step % eval_every == 0 or self.step == self.total_steps:
                ev = self.evaluate(self.eval_seqs, with_hidden=True)
                ctx.log_metrics({"kind": "eval", "step": self.step, **ev, **self.loop.record()})
                ctx.event("eval", step=self.step, nll=round(ev["validation_nll"], 4), ppl=round(ev["perplexity"], 2), ece=round(ev["token_ece"], 4), lambdas=self.loop.lambdas, eff_rank={k: round(v, 1) for k, v in self.loop.monitor.kappa().items()})
            if self.step % ckpt_every == 0 and self.step < self.total_steps:
                self.save()
            _debug_stop(self.step)
        train_time = time.perf_counter() - t0
        with ctx.timer("final_eval"):
            final = self.evaluate(self.final_eval_seqs, with_hidden=True)
        self.save()
        atomic_write_json(ctx.artifact_dir / "controller_trajectory.json", {"regularizer": self.loop.regularizer, "targets": self.loop.targets, "history": self.loop.summary()["history"]})
        ctx.event("final", **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in final.items()})
        return {
            "task": "language_c4",
            "dataset": lc.get("dataset", "allenai/c4"),
            "method": self.method.get("name"),
            "regularizer": self.loop.regularizer,
            "dcr_lambda": self.method.get("dcr_lambda"),
            "seed": ctx.seed,
            "model": lc.get("model"),
            "model_params": sum(p.numel() for p in self.model.parameters()),
            "controller_params": self.loop.summary()["controller_params"],
            "steps": self.step,
            "tokens_trained": self.step * self.batch_seqs * self.seq_len,
            "final": {**final, **{f"lambda/{k}": v for k, v in self.loop.lambdas.items()}, "effective_rank/final": final.get(f"effective_rank/h{self.mon_layers[-1]}")},
            "compute": {"train_time_s": train_time, "tokens_per_s": tokens_done / max(train_time, 1e-6), "steps_per_s": self.step / max(train_time, 1e-6)},
            "data": self.info,
            "control": {k: v for k, v in self.loop.summary().items() if k != "history"},
        }


def run(ctx: JobContext) -> dict[str, Any]:
    return LanguageJob(ctx).run()
