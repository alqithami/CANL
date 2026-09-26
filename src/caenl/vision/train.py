"""Phase trainer: task loss + spectrum DCR with online collapse monitoring and MACC control.

One *phase* = the initial training on the seed labeled set, or the warm-started fine-tuning
after an acquisition round.  The learning-rate schedule (linear warm-up + cosine) restarts
every phase; the model, monitor and controller states carry over.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..core.controllers import ControllerAction, FullMACC
from ..core.dcr import DCR
from ..core.monitor import ClassConditionalMonitor
from ..utils.jobctx import JobContext
from ..utils.seeding import derive_seed
from .augment import eval_transform, train_transform
from .data import DeviceArray
from .robustness import pgd_attack


#: parameters (matched by name suffix) that are excluded from weight decay on top of all 0-/1-d tensors: the ViT
#: class token and position embedding (torchvision names; the timm/MAE names are included for converted checkpoints)
NO_WEIGHT_DECAY_SUFFIXES = ("class_token", "pos_embedding", "cls_token", "pos_embed")


def cosine_warmup_lr(step: int, total: int, warmup: int, base_lr: float, min_factor: float = 0.0) -> float:
    if total <= 0:
        return base_lr
    if step < warmup:
        return base_lr * float(step + 1) / float(max(1, warmup))
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base_lr * (min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * progress)))


class Trainer:
    def __init__(
        self,
        ctx: JobContext,
        model: torch.nn.Module,
        train_data: DeviceArray,
        train_labels: np.ndarray,
        num_classes: int,
        train_res: int,
        aug_kind: str,
        monitor: ClassConditionalMonitor,
        dcr: DCR,
        controller: Any,
        regularizer: str,
        ctrl_val_indices: Optional[np.ndarray] = None,
    ) -> None:
        self.ctx = ctx
        self.model = model
        self.data = train_data
        self.labels_np = np.asarray(train_labels, dtype=np.int64)
        self.labels = torch.as_tensor(self.labels_np, device=ctx.device)
        self.num_classes = int(num_classes)
        self.train_res = int(train_res)
        self.aug_kind = aug_kind
        self.monitor = monitor
        self.dcr = dcr
        self.controller = controller
        self.regularizer = regularizer
        self.ctrl_val = np.asarray(ctrl_val_indices) if ctrl_val_indices is not None else None
        self.device = ctx.device
        tcfg = ctx.config.get("training", {})
        self.batch_size = int(tcfg.get("batch_size", 256))
        self.amp = bool(tcfg.get("amp", True)) and self.device.type == "cuda"
        self.channels_last = bool(tcfg.get("channels_last", True)) and self.device.type == "cuda"
        self.log_every = int(tcfg.get("log_every_steps", 50))
        self.cadence = int(ctx.get("collapse.monitoring_cadence_steps", 100))
        self.monitor_every = int(ctx.get("collapse.monitor_update_every_steps", 1))
        # pre-registered number of post-acquisition steps after which the acquisition action is rewarded
        self.acq_reward_steps = int(ctx.get("full_macc.acquisition_reward_steps", 2 * self.cadence))
        self.layers = list(dcr.layers.keys())
        self.global_step = 0
        self.total_steps_planned = 1
        init_lam = {l: float(getattr(controller, "lambdas", {}).get(l, 0.0)) for l in self.layers}
        self.current_action: ControllerAction = ControllerAction(lambdas=init_lam, flags={l: 1.0 for l in self.layers}, threshold_quantile=float(getattr(controller, "threshold_quantile", 0.9)))
        self.prev_ctrl_value: Optional[float] = None
        self.queries_since_last = 0
        at = ctx.config.get("method", {}).get("adversarial_training", False)
        self.adv_train = bool(at)
        self.at_cfg = dict(ctx.config.get("adversarial_training", {}))
        if self.channels_last:
            self.model.to(memory_format=torch.channels_last)

    # ------------------------------------------------------------------ helpers
    @property
    def lambdas(self) -> dict[str, float]:
        return self.current_action.effective_lambdas()

    @property
    def threshold_quantile(self) -> float:
        return float(self.current_action.threshold_quantile)

    def make_optimizer(self, lr: float):
        tcfg = self.ctx.config.get("training", {})
        opt = str(tcfg.get("optimizer", "sgd")).lower()
        wd = float(tcfg.get("weight_decay", 5e-4))
        layer_decay = tcfg.get("layer_decay")
        layer_ids: dict[str, int] = {}
        n_layers = 0
        inner = getattr(self.model, "model", self.model)
        if layer_decay and hasattr(inner, "param_layer_ids"):
            layer_ids = inner.param_layer_ids(prefix="model.")
            n_layers = int(getattr(inner, "n_layers", 0))
        buckets: dict[tuple[float, float], list] = {}
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            scale = 1.0
            if layer_ids:
                scale = float(layer_decay) ** (n_layers + 1 - layer_ids.get(n, n_layers + 1))
            # no weight decay on biases / normalisation weights (ndim <= 1) nor on the ViT class token and position
            # embedding (DeiT / MAE fine-tuning recipes); names are matched on their suffix so wrappers do not matter
            no_decay = p.ndim <= 1 or n.endswith(NO_WEIGHT_DECAY_SUFFIXES)
            key = (0.0 if no_decay else wd, scale)
            buckets.setdefault(key, []).append(p)
        groups = [{"params": ps, "weight_decay": k[0], "lr_scale": k[1], "lr": lr * k[1]} for k, ps in sorted(buckets.items())]
        if opt == "sgd":
            return torch.optim.SGD(groups, lr=lr, momentum=float(tcfg.get("momentum", 0.9)), nesterov=bool(tcfg.get("nesterov", True)))
        if opt == "adamw":
            return torch.optim.AdamW(groups, lr=lr, betas=tuple(tcfg.get("betas", (0.9, 0.999))))
        raise ValueError(f"unknown optimizer {opt}")

    @torch.no_grad()
    def controller_value(self) -> Optional[float]:
        """Negative loss on the controller-validation split (higher is better)."""
        if self.ctrl_val is None or len(self.ctrl_val) == 0:
            return None
        self.model.eval()
        total, n = 0.0, 0
        for s in range(0, len(self.ctrl_val), 512):
            idx = self.ctrl_val[s : s + 512]
            xb = eval_transform(self.data.get(idx), self.train_res)  # memmap mode: synchronous gather, training prefetch is kept
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
                logits = self.model(xb)
            total += float(F.cross_entropy(logits.float(), self.labels[idx], reduction="sum"))
            n += len(idx)
        self.model.train()
        return -total / max(n, 1)

    def controller_step(self, progress: float, extra_state: tuple[float, ...]) -> dict[str, Any]:
        kappa = self.monitor.kappa()
        info: dict[str, Any] = {}
        if isinstance(self.controller, FullMACC):
            value = self.controller_value()
            delta = None if (value is None or self.prev_ctrl_value is None) else (value - self.prev_ctrl_value)
            self.prev_ctrl_value = value
            self.current_action, info = self.controller.observe_and_act(kappa, self.monitor.instability(), progress, delta, 0, extra_state, step_index=self.global_step)
            info = {"ctrl_value": value, "ctrl_delta": delta, **{k: v for k, v in info.items()}}
        else:
            self.current_action = self.controller.step(kappa, step_index=self.global_step)
        return info

    # ------------------------------------------------------------------ main loop
    def train_phase(
        self,
        labeled: np.ndarray,
        epochs: int,
        lr: float,
        phase: str,
        round_index: int,
        label_fraction: float,
        warmup_epochs: float = 1.0,
        on_epoch_end: Optional[Callable[[int, dict[str, Any]], None]] = None,
    ) -> dict[str, Any]:
        labeled = np.asarray(labeled, dtype=np.int64)
        n = len(labeled)
        bs = min(self.batch_size, n)
        steps_per_epoch = max(1, n // bs)
        total_steps = steps_per_epoch * int(epochs)
        warmup_steps = int(round(warmup_epochs * steps_per_epoch))
        optimizer = self.make_optimizer(lr)
        gen = torch.Generator(device="cpu").manual_seed(derive_seed(self.ctx.seed, "order", phase))
        aug_gen = torch.Generator(device=self.device).manual_seed(derive_seed(self.ctx.seed, "aug", phase))
        self.model.train()
        labeled_t = torch.as_tensor(labeled)
        step_in_phase = 0
        t_phase = time.perf_counter()
        images_seen = 0
        ema_loss: Optional[float] = None
        last_log: dict[str, Any] = {}
        ce_fn = torch.nn.CrossEntropyLoss(label_smoothing=float(self.ctx.get("training.label_smoothing", 0.0)))
        # regularisation-action rewards never straddle a phase boundary (acquisition + LR restart);
        # the pending *acquisition* action is kept and rewarded after `acq_reward_steps` steps below
        self.prev_ctrl_value = None
        if isinstance(self.controller, FullMACC):
            self.controller.new_phase()
        acq_reward_at = min(self.acq_reward_steps, total_steps)
        self.ctx.event("phase_started", phase=phase, round=round_index, n_labeled=n, epochs=int(epochs), steps=total_steps, lr=lr, lambdas=self.lambdas)
        for epoch in range(int(epochs)):
            perm = labeled_t[torch.randperm(n, generator=gen)]
            self.data.prefetch([perm[b * bs : (b + 1) * bs] for b in range(steps_per_epoch)])  # no-op unless the data is a disk memmap
            t_epoch = time.perf_counter()
            epoch_loss, epoch_ce, epoch_dcr, epoch_correct, epoch_n = 0.0, 0.0, 0.0, 0, 0
            for b in range(steps_per_epoch):
                idx = perm[b * bs : (b + 1) * bs]
                x = train_transform(self.data.get(idx), self.aug_kind, self.train_res, generator=aug_gen, rrc_scale=tuple(self.ctx.get("training.rrc_scale", (0.25, 1.0))))
                y = self.labels[idx.to(self.device)]
                if self.channels_last:
                    x = x.contiguous(memory_format=torch.channels_last)
                if self.adv_train:
                    if self.at_cfg.get("attack_in_eval_mode", False):
                        self.model.eval()
                    x = pgd_attack(self.model, x, y, float(self.at_cfg.get("eps", 8 / 255)), float(self.at_cfg.get("alpha", 2 / 255)), int(self.at_cfg.get("steps", 10)), random_start=True, generator=aug_gen)
                    self.model.train()
                lr_now = cosine_warmup_lr(step_in_phase, total_steps, warmup_steps, lr, float(self.ctx.get("training.min_lr_factor", 0.0)))
                for g in optimizer.param_groups:
                    g["lr"] = lr_now * float(g.get("lr_scale", 1.0))  # layer-wise lr decay (ViT fine-tuning) scales per group
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
                    logits, feats = self.model.forward_features(x)
                    ce = ce_fn(logits.float(), y)
                lambdas = self.lambdas
                if any(v > 0 for v in lambdas.values()):
                    dcr_loss, dcr_per_layer = self.dcr({l: feats[l].float() for l in self.layers if l in feats}, lambdas, labels=y)
                else:
                    dcr_loss, dcr_per_layer = ce.new_zeros(()), {}
                loss = ce + dcr_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.ctx.get("training.grad_clip_norm"):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.ctx.get("training.grad_clip_norm")))
                optimizer.step()
                # ---- monitoring
                if self.global_step % self.monitor_every == 0:
                    self.monitor.update({l: feats[l].detach().float() for l in self.layers if l in feats}, y)
                step_in_phase += 1
                self.global_step += 1
                images_seen += int(idx.numel())
                # ---- acquisition credit assignment (Full MACC): reward tau with the post-acquisition change
                if isinstance(self.controller, FullMACC) and self.controller.has_pending_acquisition() and step_in_phase == acq_reward_at:
                    value_after = self.controller_value()
                    acq_info = self.controller.reward_acquisition(value_after, step_index=self.global_step)
                    if acq_info:
                        self.ctx.log_metrics({"kind": "acquisition_update", "step": self.global_step, "phase": phase, "round": round_index, **{k: v for k, v in acq_info.items() if isinstance(v, (int, float, bool)) or v is None}})
                        self.ctx.event("acquisition_rewarded", round=round_index, reward=round(float(acq_info.get("reward", 0.0)), 4), value_delta=round(float(acq_info.get("value_delta", 0.0)), 5), queries=acq_info.get("queries"), tau=acq_info.get("tau"), explored=acq_info.get("explored"))
                with torch.no_grad():
                    correct = int((logits.argmax(dim=1) == y).sum())
                lv, cev, dv = float(loss.detach()), float(ce.detach()), float(dcr_loss.detach())
                ema_loss = lv if ema_loss is None else 0.98 * ema_loss + 0.02 * lv
                epoch_loss += lv * len(idx)
                epoch_ce += cev * len(idx)
                epoch_dcr += dv * len(idx)
                epoch_correct += correct
                epoch_n += len(idx)
                # ---- controller
                ctrl_info: dict[str, Any] = {}
                if self.global_step % self.cadence == 0 and self.regularizer in ("macc_lite", "full_macc"):
                    progress = min(1.0, self.global_step / max(1, self.total_steps_planned))
                    ctrl_info = self.controller_step(progress, (float(label_fraction),))
                    self.ctx.log_metrics({"kind": "controller", "step": self.global_step, "phase": phase, "round": round_index, **{f"lambda/{l}": v for l, v in self.lambdas.items()}, "tau": self.threshold_quantile, **self.monitor.record(), **{k: v for k, v in ctrl_info.items() if isinstance(v, (int, float)) or v is None}})
                if self.global_step % self.log_every == 0 or step_in_phase == total_steps:
                    elapsed = time.perf_counter() - t_phase
                    last_log = {
                        "kind": "train",
                        "step": self.global_step,
                        "phase": phase,
                        "round": round_index,
                        "label_fraction": float(label_fraction),
                        "epoch": epoch,
                        "loss": lv,
                        "ce": cev,
                        "dcr": dv,
                        "loss_ema": ema_loss,
                        "lr": lr_now,
                        "batch_acc": correct / max(1, len(idx)),
                        "images_per_s": images_seen / max(elapsed, 1e-6),
                        **{f"dcr/{l}": v for l, v in dcr_per_layer.items()},
                        **{f"lambda/{l}": v for l, v in self.lambdas.items()},
                        **self.monitor.record(),
                    }
                    self.ctx.log_metrics(last_log)
            epoch_time = time.perf_counter() - t_epoch
            rec = {"kind": "epoch", "step": self.global_step, "phase": phase, "round": round_index, "epoch": epoch, "epoch_time_s": epoch_time, "train_loss": epoch_loss / max(epoch_n, 1), "train_ce": epoch_ce / max(epoch_n, 1), "train_dcr": epoch_dcr / max(epoch_n, 1), "train_acc": epoch_correct / max(epoch_n, 1), "images_per_s": epoch_n / max(epoch_time, 1e-6), **self.monitor.record(), **{f"lambda/{l}": v for l, v in self.lambdas.items()}}
            self.ctx.log_metrics(rec)
            if on_epoch_end is not None:
                on_epoch_end(epoch, rec)
        phase_time = time.perf_counter() - t_phase
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        out = {
            "phase": phase,
            "round": round_index,
            "n_labeled": n,
            "epochs": int(epochs),
            "steps": total_steps,
            "phase_time_s": phase_time,
            "images_per_s": images_seen / max(phase_time, 1e-6),
            "final_train_loss": last_log.get("loss_ema"),
            "lambdas": self.lambdas,
            "threshold_quantile": self.threshold_quantile,
            "kappa": self.monitor.kappa(),
            "instability": self.monitor.instability(),
            "collapse_strength": self.monitor.strength(),
        }
        self.ctx.event("phase_finished", **{k: v for k, v in out.items() if not isinstance(v, dict)}, lambdas=self.lambdas)
        return out
