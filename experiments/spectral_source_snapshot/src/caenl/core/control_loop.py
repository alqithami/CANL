"""Shared monitoring + DCR + controller loop for label-free representations (LM, diffusion, audio).

The collapse index is the effective rank of the EMA covariance spectrum; the DCR target is
the rank-``r`` near-isotropic template with ``r = min(r_cfg, d, k)`` (manuscript Eq. 12), and
MACC-Lite uses the absolute log-ratio error so that lambda grows while the effective rank is
away from the target rank and decays inside the deadband.  Full MACC is rewarded with the
decrease of a held-out validation loss supplied by the task.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

import torch
from torch import Tensor

from .controllers import ControllerAction, FullMACC, build_controller
from .dcr import DCR
from .monitor import SpectralMonitor


class ControlLoop:
    def __init__(
        self,
        layer_dims: Mapping[str, int],
        regularizer: str,
        dcr_lambda: float,
        cfg: Mapping[str, Any],
        device: torch.device,
        seed: int,
        total_steps: int,
        k: int = 128,
        configured_rank: int | Mapping[str, int] = 128,
        target_effective_rank: Optional[float | Mapping[str, float]] = None,
        cadence: int = 50,
        error_mode: str = "absolute_log_ratio",
        ema_decay: float = 0.99,
        state_extra: int = 1,
    ) -> None:
        self.layers = list(layer_dims.keys())
        self.layer_dims = {l: int(d) for l, d in layer_dims.items()}
        self.regularizer = str(regularizer)
        self.device = device
        self.cadence = int(cadence)
        self.monitor = SpectralMonitor(self.layer_dims, ema_decay=float(ema_decay), device=device, projection_seed=seed)
        self.dcr = DCR.build(self.layer_dims, k=int(k), num_classes=None, configured_rank=configured_rank, covariance="total", device=device)
        # targets: effective rank target defaults to the DCR target rank of each layer
        self.targets: dict[str, float] = {}
        for l in self.layers:
            if isinstance(target_effective_rank, Mapping):
                self.targets[l] = float(target_effective_rank.get(l, self.dcr.layers[l].rank))
            elif target_effective_rank is not None:
                self.targets[l] = float(target_effective_rank)
            else:
                self.targets[l] = float(self.dcr.layers[l].rank)
        ml = dict(cfg.get("macc_lite", {}))
        ml["error_mode"] = error_mode  # the task decides (absolute_log_ratio for rank targets), not the inherited vision block
        ml.setdefault("deadband", 0.1)
        ctrl_cfg = {"macc_lite": ml, "full_macc": cfg.get("full_macc", {}), "threshold_quantile": 0.9}
        self.controller = build_controller(self.regularizer, self.layers, self.targets, ctrl_cfg, total_controller_steps=max(1, int(total_steps) // self.cadence), device=device, seed=seed, state_extra=state_extra, lambda_value=float(dcr_lambda), quantile_head_enabled=False)
        init_lam = {l: float(getattr(self.controller, "lambdas", {}).get(l, 0.0)) for l in self.layers}
        self.action = ControllerAction(lambdas=init_lam, flags={l: 1.0 for l in self.layers}, threshold_quantile=0.9)
        self.prev_value: Optional[float] = None
        self.step = 0
        self.total_steps = int(total_steps)

    @property
    def lambdas(self) -> dict[str, float]:
        return self.action.effective_lambdas()

    def active(self) -> bool:
        return any(v > 0 for v in self.lambdas.values())

    def dcr_loss(self, feats: Mapping[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        if not self.active():
            any_feat = next(iter(feats.values()))
            return any_feat.new_zeros((), dtype=torch.float32), {}
        return self.dcr({l: f.float() for l, f in feats.items() if l in self.dcr.layers}, self.lambdas)

    def update(self, feats: Mapping[str, Tensor], value_fn: Optional[Callable[[], float]] = None, extra_state: tuple[float, ...] = ()) -> dict[str, Any]:
        """Call once per optimisation step after the forward pass. Returns controller info when it acted."""
        self.step += 1
        compute_kappa = (self.step % self.cadence == 0) or self.step <= 2
        with torch.no_grad():
            self.monitor.update({l: f.detach() for l, f in feats.items() if l in self.monitor.layers}, compute_kappa=compute_kappa)
        info: dict[str, Any] = {}
        if self.step % self.cadence == 0 and self.regularizer in ("macc_lite", "full_macc"):
            kappa = self.monitor.kappa()
            if isinstance(self.controller, FullMACC):
                value = value_fn() if value_fn is not None else None
                delta = None if (value is None or self.prev_value is None) else (value - self.prev_value)
                self.prev_value = value
                self.action, upd = self.controller.observe_and_act(kappa, self.monitor.instability(), min(1.0, self.step / max(1, self.total_steps)), delta, 0, extra_state, step_index=self.step)
                info = {"ctrl_value": value, "ctrl_delta": delta, **{k: v for k, v in upd.items() if isinstance(v, (int, float))}}
            else:
                self.action = self.controller.step(kappa, step_index=self.step)
            info.update({f"lambda/{l}": v for l, v in self.lambdas.items()})
            info.update(self.monitor.record())
        return info

    def record(self) -> dict[str, float]:
        rec = {f"lambda/{l}": v for l, v in self.lambdas.items()}
        rec.update(self.monitor.record())
        return rec

    def state_dict(self) -> dict[str, Any]:
        return {"monitor": self.monitor.state_dict(), "controller": self.controller.state_dict(), "action": {"lambdas": self.action.lambdas, "flags": self.action.flags, "tau": self.action.threshold_quantile}, "prev_value": self.prev_value, "step": self.step}

    def load_state_dict(self, s: Mapping[str, Any]) -> None:
        self.monitor.load_state_dict(s["monitor"])
        self.controller.load_state_dict(s["controller"])
        a = s["action"]
        self.action = ControllerAction(lambdas=dict(a["lambdas"]), flags=dict(a["flags"]), threshold_quantile=float(a["tau"]))
        self.prev_value = s.get("prev_value")
        self.step = int(s.get("step", 0))

    def summary(self) -> dict[str, Any]:
        return {"regularizer": self.regularizer, "targets_effective_rank": self.targets, "dcr_target_ranks": {l: s.rank for l, s in self.dcr.layers.items()}, "final_lambdas": self.lambdas, "final_effective_rank": self.monitor.kappa(), "controller_params": int(getattr(self.controller, "num_parameters", 0)), "history": list(getattr(self.controller, "history", []))}
