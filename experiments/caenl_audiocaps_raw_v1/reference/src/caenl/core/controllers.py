"""Meta-Adaptive Collapse Controllers.

Two instantiations are evaluated as *distinct* methods (reviewer request):

``MACCLite``
    Non-learned projected proportional feedback controller (manuscript Alg. 2, line 12)::

        lambda_l <- Proj_[lambda_min, lambda_max]( lambda_l + eta * e_l )

    with the per-layer error ``e_l`` given by ``error_mode``:

    * ``log_ratio``  (default, classification):  ``e_l = log(kappa_l / kappa_l^*)`` — scale-free
      version of the manuscript's ``kappa_l - kappa_l^*``; positive when collapse is weaker
      than the target, so regularisation increases.
    * ``difference``: the literal manuscript rule ``kappa_l - kappa_l^*``.
    * ``absolute_log_ratio`` (generative / language, rank targets): ``|log(kappa_l/kappa_l^*)| - deadband``
      so that lambda grows while the effective rank is away from the target rank (in either
      direction) and decays inside the deadband.

``FullMACC``
    Online contextual bandit (manuscript Alg. 1): an MLP policy maps the controller state to
    a discrete lambda-delta and on/off flag per monitored layer plus a query-threshold
    quantile; it is trained with REINFORCE against an EMA reward baseline using the
    manuscript reward ``R_t = dVal_t - beta * n_q - gamma * sum_l |log(kappa_l/kappa_l^*)|``.
    Epsilon-greedy exploration decays linearly over the run.  No meta-pretraining is used.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


def kappa_target_from_strength(strength: float) -> float:
    """Convert a collapse-strength target ``s* in (0,1)`` into a kappa target ``(1-s*)/s*``."""
    s = float(strength)
    if not 0.0 < s < 1.0:
        raise ValueError("collapse strength target must lie in (0,1)")
    return (1.0 - s) / s


@dataclass
class ControllerAction:
    lambdas: dict[str, float]
    flags: dict[str, float]
    threshold_quantile: float
    errors: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def effective_lambdas(self) -> dict[str, float]:
        return {k: float(self.lambdas[k]) * float(self.flags.get(k, 1.0)) for k in self.lambdas}


class MACCLite:
    def __init__(
        self,
        layers: Sequence[str],
        targets: Mapping[str, float],
        lambda_initial: float = 0.03,
        lambda_min: float = 0.0,
        lambda_max: float = 0.5,
        step_size: float = 0.01,
        threshold_quantile: float = 0.90,
        error_mode: str = "log_ratio",
        deadband: float = 0.0,
    ) -> None:
        if not layers:
            raise ValueError("at least one layer is required")
        if error_mode not in {"log_ratio", "difference", "absolute_log_ratio"}:
            raise ValueError(f"unknown error_mode {error_mode}")
        self.layers = list(layers)
        self.targets = {l: float(targets[l]) for l in self.layers}
        self.lambdas = {l: float(lambda_initial) for l in self.layers}
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.step_size = float(step_size)
        self.threshold_quantile = float(threshold_quantile)
        self.error_mode = error_mode
        self.deadband = float(deadband)
        self.history: list[dict[str, Any]] = []

    def error(self, layer: str, kappa: float) -> float:
        k = max(float(kappa), 1e-12)
        t = max(self.targets[layer], 1e-12)
        if self.error_mode == "difference":
            return k - t
        e = math.log(k / t)
        if self.error_mode == "absolute_log_ratio":
            return abs(e) - self.deadband
        return e

    def step(self, kappa: Mapping[str, float], step_index: Optional[int] = None) -> ControllerAction:
        errors: dict[str, float] = {}
        for l in self.layers:
            if l not in kappa or not math.isfinite(float(kappa[l])):
                errors[l] = 0.0
                continue
            e = self.error(l, float(kappa[l]))
            errors[l] = e
            new = self.lambdas[l] + self.step_size * e
            self.lambdas[l] = min(self.lambda_max, max(self.lambda_min, new))
        action = ControllerAction(
            lambdas=dict(self.lambdas),
            flags={l: 1.0 if self.lambdas[l] > 0 else 0.0 for l in self.layers},
            threshold_quantile=self.threshold_quantile,
            errors=errors,
        )
        self.history.append({"step": step_index, "kappa": {l: float(kappa.get(l, float("nan"))) for l in self.layers}, "errors": errors, "lambdas": dict(self.lambdas)})
        return action

    def state_dict(self) -> dict[str, Any]:
        return {"lambdas": dict(self.lambdas), "history": list(self.history)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.lambdas.update({k: float(v) for k, v in state.get("lambdas", {}).items()})
        self.history = list(state.get("history", []))


class FullMACCPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_layers: int,
        hidden_dims: Sequence[int] = (128, 128),
        lambda_delta_choices: Sequence[float] = (-0.02, -0.01, 0.0, 0.01, 0.02),
        threshold_quantiles: Sequence[float] = (0.80, 0.85, 0.90, 0.95),
    ) -> None:
        super().__init__()
        dims = [int(state_dim), *[int(h) for h in hidden_dims]]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.GELU()]
        self.backbone = nn.Sequential(*layers)
        h = dims[-1]
        self.num_layers = int(num_layers)
        self.register_buffer("delta_values", torch.tensor(list(lambda_delta_choices), dtype=torch.float32))
        self.register_buffer("quantile_values", torch.tensor(list(threshold_quantiles), dtype=torch.float32))
        self.delta_head = nn.Linear(h, self.num_layers * len(lambda_delta_choices))
        self.flag_head = nn.Linear(h, self.num_layers * 2)
        self.quantile_head = nn.Linear(h, len(threshold_quantiles))
        # start near "no change, all on, tau=0.9" for a gentle beginning
        with torch.no_grad():
            self.delta_head.bias.view(self.num_layers, -1)[:, len(lambda_delta_choices) // 2] += 1.0
            self.flag_head.bias.view(self.num_layers, 2)[:, 1] += 1.0

    def distributions(self, state: Tensor):
        if state.ndim == 1:
            state = state.unsqueeze(0)
        h = self.backbone(state).squeeze(0)
        d = Categorical(logits=self.delta_head(h).view(self.num_layers, -1))
        f = Categorical(logits=self.flag_head(h).view(self.num_layers, 2))
        q = Categorical(logits=self.quantile_head(h))
        return d, f, q


class FullMACC:
    """Online contextual-bandit MACC (manuscript Algorithm 1) with explicit credit assignment.

    Two kinds of actions are rewarded separately:

    * **regularisation actions** (per-layer lambda delta and on/off flag) taken at every
      controller step are rewarded at the *next* controller step with the controller-validation
      change observed in between (``update``);
    * the **acquisition action** (query-threshold quantile ``tau``) is the quantile of the last
      controller action before an acquisition round.  Its effect can only be observed after the
      newly acquired examples have been trained on, so it is stored (``mark_acquisition``) together
      with the controller-validation value measured immediately before acquisition and rewarded
      after a pre-registered number of post-acquisition training steps
      (``reward_acquisition``) with ``scale * (value_after - value_before) - beta * n_queries``.

    Pending actions are stored as detached, serialisable records (state, action indices,
    exploration flag); log-probabilities are recomputed from the current policy at reward time, so
    pending actions survive checkpoint/resume (``state_dict``/``load_state_dict``).  Uniformly
    explored (epsilon-greedy) actions are never used for policy-gradient updates.  Tasks without
    label acquisition construct the controller with ``quantile_head_enabled=False``: tau is fixed
    and the quantile head is excluded from the joint log-probability.
    """

    def __init__(
        self,
        layers: Sequence[str],
        targets: Mapping[str, float],
        state_dim: int,
        lambda_initial: float = 0.03,
        lambda_min: float = 0.0,
        lambda_max: float = 0.5,
        hidden_dims: Sequence[int] = (128, 128),
        lambda_delta_choices: Sequence[float] = (-0.02, -0.01, 0.0, 0.01, 0.02),
        threshold_quantiles: Sequence[float] = (0.80, 0.85, 0.90, 0.95),
        policy_lr: float = 3e-4,
        epsilon_start: float = 0.20,
        epsilon_end: float = 0.02,
        total_controller_steps: int = 1000,
        reward_baseline_ema: float = 0.05,
        entropy_coefficient: float = 1e-3,
        reward_value_weight: float = 1.0,
        reward_value_scale: float = 10.0,
        reward_query_beta: float = 0.0,
        reward_target_gamma: float = 0.1,
        device: torch.device | str = "cpu",
        seed: int = 0,
        quantile_head_enabled: bool = True,
        fixed_threshold_quantile: float = 0.90,
        value_delta_clip: Optional[float] = 2.0,
    ) -> None:
        self.layers = list(layers)
        self.targets = {l: float(targets[l]) for l in self.layers}
        self.lambda_min, self.lambda_max = float(lambda_min), float(lambda_max)
        self.lambdas = {l: float(lambda_initial) for l in self.layers}
        self.flags = {l: 1.0 for l in self.layers}
        self.device = torch.device(device)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.quantile_head_enabled = bool(quantile_head_enabled)
        if not self.quantile_head_enabled:
            threshold_quantiles = (float(fixed_threshold_quantile),)
        self.policy = FullMACCPolicy(state_dim, len(self.layers), hidden_dims, lambda_delta_choices, threshold_quantiles).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=float(policy_lr))
        self.epsilon_start, self.epsilon_end = float(epsilon_start), float(epsilon_end)
        self.total_controller_steps = max(1, int(total_controller_steps))
        self.reward_baseline_ema = float(reward_baseline_ema)
        self.entropy_coefficient = float(entropy_coefficient)
        self.reward_value_weight = float(reward_value_weight)
        self.reward_value_scale = float(reward_value_scale)
        self.reward_query_beta = float(reward_query_beta)
        self.reward_target_gamma = float(reward_target_gamma)
        # validation deltas are clipped (nats) so that a transient loss blow-up cannot produce a reward that
        # swamps the policy's running baseline; never active for the O(0.1) deltas of a healthy run
        self.value_delta_clip = None if value_delta_clip is None else float(value_delta_clip)
        self.reward_baseline: Optional[float] = None
        self.acq_reward_baseline: Optional[float] = None
        self.controller_step = 0
        self.threshold_quantile = round(float(self.policy.quantile_values[0]), 6) if not self.quantile_head_enabled else 0.90
        self._pending: Optional[dict[str, Any]] = None  # regularisation action awaiting its reward
        self._pending_acq: Optional[dict[str, Any]] = None  # acquisition action awaiting its reward
        self._last_action: Optional[dict[str, Any]] = None
        self.history: list[dict[str, Any]] = []

    # -------------------------------------------------------------- helpers
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.policy.parameters())

    def epsilon(self) -> float:
        frac = min(1.0, self.controller_step / float(self.total_controller_steps))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def _clip_delta(self, value_delta: float) -> float:
        v = float(value_delta)
        if self.value_delta_clip is not None:
            v = max(-self.value_delta_clip, min(self.value_delta_clip, v))
        return v

    def reward_components(self, value_delta: float, queries: int, kappa: Mapping[str, float]) -> dict[str, float]:
        """Return a decomposed reward record for reproducible controller diagnostics."""
        clipped_delta = self._clip_delta(value_delta)
        dev = 0.0
        for l in self.layers:
            k = kappa.get(l, float("nan"))
            if k is None or not math.isfinite(float(k)):
                continue
            dev += abs(math.log(max(float(k), 1e-12) / max(self.targets[l], 1e-12)))
        value_term = self.reward_value_weight * self.reward_value_scale * float(clipped_delta)
        query_term = -self.reward_query_beta * float(queries)
        target_term = -self.reward_target_gamma * dev
        return {
            "reward": float(value_term + query_term + target_term),
            "reward_value_term": float(value_term),
            "reward_query_term": float(query_term),
            "reward_target_term": float(target_term),
            "target_deviation": float(dev),
            "value_delta_clipped": float(clipped_delta),
        }

    def reward(self, value_delta: float, queries: int, kappa: Mapping[str, float]) -> float:
        return self.reward_components(value_delta, queries, kappa)["reward"]

    def build_state(
        self,
        kappa: Mapping[str, float],
        instability: Mapping[str, float],
        progress: float,
        extra: Sequence[float] = (),
    ) -> Tensor:
        feats: list[float] = []
        for l in self.layers:
            k = float(kappa.get(l, float("nan")))
            k = k if math.isfinite(k) else self.targets[l]
            inst = float(instability.get(l, 0.0))
            inst = inst if math.isfinite(inst) else 0.0
            feats += [
                math.log(max(k, 1e-12)),
                math.log(max(k, 1e-12) / max(self.targets[l], 1e-12)),
                inst / max(k, 1e-12),
                self.lambdas[l],
                self.flags[l],
            ]
        feats += [float(progress), *[float(x) for x in extra]]
        return torch.tensor(feats, dtype=torch.float32, device=self.device)

    @staticmethod
    def state_dim_for(num_layers: int, num_extra: int = 0) -> int:
        return 5 * int(num_layers) + 1 + int(num_extra)

    def _log_prob_entropy(self, rec: Mapping[str, Any], heads: Sequence[str]) -> tuple[Tensor, Tensor]:
        """Recompute log-probability and entropy of a stored action under the *current* policy."""
        d_dist, f_dist, q_dist = self.policy.distributions(torch.as_tensor(rec["state"], dtype=torch.float32, device=self.device))
        lp = torch.zeros((), device=self.device)
        ent = torch.zeros((), device=self.device)
        if "delta" in heads:
            idx = torch.as_tensor(rec["delta_idx"], device=self.device)
            lp = lp + d_dist.log_prob(idx).sum()
            ent = ent + d_dist.entropy().sum()
        if "flag" in heads:
            idx = torch.as_tensor(rec["flag_idx"], device=self.device)
            lp = lp + f_dist.log_prob(idx).sum()
            ent = ent + f_dist.entropy().sum()
        if "quantile" in heads and self.quantile_head_enabled:
            idx = torch.as_tensor(int(rec["quantile_idx"]), device=self.device)
            lp = lp + q_dist.log_prob(idx)
            ent = ent + q_dist.entropy()
        return lp, ent

    def _policy_step(self, rec: Mapping[str, Any], advantage: float, heads: Sequence[str]) -> dict[str, float]:
        with torch.enable_grad():
            lp, ent = self._log_prob_entropy(rec, heads)
            loss = -(float(advantage) * lp) - self.entropy_coefficient * ent
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
            self.optimizer.step()
        return {"policy_loss": float(loss.detach()), "entropy": float(ent.detach()), "log_prob": float(lp.detach())}

    # -------------------------------------------------------------- acting
    def act(self, state: Tensor, greedy: bool = False) -> ControllerAction:
        with torch.no_grad():
            d_dist, f_dist, q_dist = self.policy.distributions(state.detach())
        eps = self.epsilon()
        explore = (not greedy) and (torch.rand((), generator=self.generator).item() < eps)
        if greedy:
            d_idx = d_dist.logits.argmax(-1)
            f_idx = f_dist.logits.argmax(-1)
            q_idx = q_dist.logits.argmax(-1)
        elif explore:
            d_idx = torch.randint(d_dist.logits.shape[-1], (len(self.layers),), generator=self.generator).to(self.device)
            f_idx = torch.randint(2, (len(self.layers),), generator=self.generator).to(self.device)
            q_idx = torch.randint(q_dist.logits.shape[-1], (), generator=self.generator).to(self.device)
        else:
            # sample with the controller's private generator (reproducible independent of the global RNG)
            d_idx = torch.multinomial(d_dist.probs.detach().cpu(), 1, generator=self.generator).squeeze(1).to(self.device)
            f_idx = torch.multinomial(f_dist.probs.detach().cpu(), 1, generator=self.generator).squeeze(1).to(self.device)
            q_idx = torch.multinomial(q_dist.probs.detach().cpu(), 1, generator=self.generator).squeeze().to(self.device)
        if not self.quantile_head_enabled:
            q_idx = torch.zeros((), dtype=torch.long, device=self.device)
        deltas = self.policy.delta_values[d_idx]
        for i, l in enumerate(self.layers):
            new = self.lambdas[l] + round(float(deltas[i]), 6)
            self.lambdas[l] = round(min(self.lambda_max, max(self.lambda_min, new)), 6)
            self.flags[l] = float(f_idx[i])
        self.threshold_quantile = round(float(self.policy.quantile_values[int(q_idx)]), 6)
        rec = {
            "state": state.detach().cpu().tolist(),
            "delta_idx": [int(v) for v in d_idx.tolist()],
            "flag_idx": [int(v) for v in f_idx.tolist()],
            "quantile_idx": int(q_idx),
            "explored": bool(explore),
            "epsilon": eps,
            "controller_step": self.controller_step,
        }
        self._pending = rec
        self._last_action = dict(rec)
        self.controller_step += 1
        return ControllerAction(
            lambdas=dict(self.lambdas),
            flags=dict(self.flags),
            threshold_quantile=self.threshold_quantile,
            extra={"explored": bool(explore), "epsilon": eps, "delta_idx": rec["delta_idx"], "flag_idx": rec["flag_idx"], "quantile_idx": rec["quantile_idx"]},
        )

    # -------------------------------------------------------------- regularisation-action reward
    def update(self, reward: float) -> dict[str, Any]:
        """REINFORCE update of the regularisation heads for the most recent action."""
        if self._pending is None:
            return {}
        r = float(reward)
        baseline_old = r if self.reward_baseline is None else self.reward_baseline
        advantage = r - baseline_old
        # the baseline is updated *after* forming the advantage
        self.reward_baseline = r if self.reward_baseline is None else (1 - self.reward_baseline_ema) * self.reward_baseline + self.reward_baseline_ema * r
        info: dict[str, Any] = {"reward": r, "baseline": float(self.reward_baseline), "advantage": float(advantage), "explored": bool(self._pending["explored"])}
        if self._pending["explored"]:
            # epsilon-greedy (uniform) actions are off-policy: no REINFORCE update for them
            self._pending = None
            return info
        info.update(self._policy_step(self._pending, advantage, heads=("delta", "flag")))
        self._pending = None
        return info

    def observe_and_act(
        self,
        kappa: Mapping[str, float],
        instability: Mapping[str, float],
        progress: float,
        value_delta: Optional[float],
        queries_since_last: int = 0,
        extra_state: Sequence[float] = (),
        step_index: Optional[int] = None,
    ) -> tuple[ControllerAction, dict[str, Any]]:
        """Reward the previous regularisation action with the observed validation change, then act."""
        info: dict[str, Any] = {}
        if self._pending is not None and value_delta is not None:
            components = self.reward_components(value_delta, 0, kappa)
            info = self.update(components["reward"])
            info.update(components)
        state = self.build_state(kappa, instability, progress, extra_state)
        action = self.act(state)
        rec = {"kind": "step", "step": step_index, "kappa": {l: float(kappa.get(l, float("nan"))) for l in self.layers}, "lambdas": dict(self.lambdas), "flags": dict(self.flags), "tau": self.threshold_quantile, "update": info, **action.extra}
        self.history.append(rec)
        return action, info

    def new_phase(self) -> None:
        """Called at a phase boundary (acquisition + learning-rate restart).

        The regularisation action taken just before the boundary cannot be rewarded cleanly (its
        validation change would be confounded by the new labels and the restart), so it is dropped.
        The pending *acquisition* action is kept: it is rewarded by ``reward_acquisition``.
        """
        self._pending = None

    # -------------------------------------------------------------- acquisition-action credit assignment
    def mark_acquisition(self, value_before: Optional[float], queries: int, step_index: Optional[int] = None) -> None:
        """Remember the action whose tau governed this acquisition, with the pre-acquisition value."""
        src = self._pending if self._pending is not None else self._last_action
        if src is None or not self.quantile_head_enabled:
            return
        self._pending_acq = {"state": list(src["state"]), "quantile_idx": int(src["quantile_idx"]), "delta_idx": list(src["delta_idx"]), "flag_idx": list(src["flag_idx"]), "explored": bool(src["explored"]), "value_before": None if value_before is None else float(value_before), "queries": int(queries), "tau": round(float(self.policy.quantile_values[int(src["quantile_idx"])]), 6), "step": step_index}

    def has_pending_acquisition(self) -> bool:
        return self._pending_acq is not None

    def reward_acquisition(self, value_after: Optional[float], step_index: Optional[int] = None) -> dict[str, Any]:
        """Reward the stored acquisition action with the post-acquisition validation change."""
        if self._pending_acq is None:
            return {}
        rec = self._pending_acq
        if value_after is None or rec.get("value_before") is None:
            self._pending_acq = None
            return {}
        delta = self._clip_delta(float(value_after) - float(rec["value_before"]))
        value_term = self.reward_value_weight * self.reward_value_scale * delta
        query_term = -self.reward_query_beta * float(rec["queries"])
        r = value_term + query_term
        baseline_old = r if self.acq_reward_baseline is None else self.acq_reward_baseline
        advantage = r - baseline_old
        self.acq_reward_baseline = r if self.acq_reward_baseline is None else (1 - self.reward_baseline_ema) * self.acq_reward_baseline + self.reward_baseline_ema * r
        info: dict[str, Any] = {"kind": "acquisition_update", "step": step_index, "acquisition_step": rec.get("step"), "tau": rec["tau"], "quantile_idx": rec["quantile_idx"], "queries": rec["queries"], "value_before": rec["value_before"], "value_after": float(value_after), "value_delta": delta, "value_delta_clipped": delta, "reward_value_term": float(value_term), "reward_query_term": float(query_term), "reward_target_term": 0.0, "target_deviation": 0.0, "reward": r, "baseline": float(self.acq_reward_baseline), "advantage": float(advantage), "explored": bool(rec["explored"])}
        if not rec["explored"]:
            info.update(self._policy_step(rec, advantage, heads=("quantile",)))
        self.history.append(info)
        self._pending_acq = None
        return info

    # -------------------------------------------------------------- persistence
    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lambdas": dict(self.lambdas),
            "flags": dict(self.flags),
            "reward_baseline": self.reward_baseline,
            "acq_reward_baseline": self.acq_reward_baseline,
            "controller_step": self.controller_step,
            "threshold_quantile": self.threshold_quantile,
            "history": list(self.history),
            "generator": self.generator.get_state(),
            "pending": None if self._pending is None else dict(self._pending),
            "pending_acq": None if self._pending_acq is None else dict(self._pending_acq),
            "last_action": None if self._last_action is None else dict(self._last_action),
            "quantile_head_enabled": self.quantile_head_enabled,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.policy.load_state_dict(state["policy"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.lambdas.update({k: float(v) for k, v in state["lambdas"].items()})
        self.flags.update({k: float(v) for k, v in state["flags"].items()})
        self.reward_baseline = state.get("reward_baseline")
        self.acq_reward_baseline = state.get("acq_reward_baseline")
        self.controller_step = int(state.get("controller_step", 0))
        self.threshold_quantile = float(state.get("threshold_quantile", 0.9))
        self.history = list(state.get("history", []))
        if "generator" in state:
            self.generator.set_state(torch.as_tensor(state["generator"], dtype=torch.uint8).cpu())
        self._pending = dict(state["pending"]) if state.get("pending") else None
        self._pending_acq = dict(state["pending_acq"]) if state.get("pending_acq") else None
        self._last_action = dict(state["last_action"]) if state.get("last_action") else None


class FixedController:
    """No-learning stand-in that keeps lambda constant (fixed DCR) or zero (no DCR)."""

    def __init__(self, layers: Sequence[str], lambda_value: float = 0.0, threshold_quantile: float = 0.90) -> None:
        self.layers = list(layers)
        self.lambdas = {l: float(lambda_value) for l in self.layers}
        self.threshold_quantile = float(threshold_quantile)
        self.history: list[dict[str, Any]] = []

    def step(self, kappa: Mapping[str, float], step_index: Optional[int] = None) -> ControllerAction:
        return ControllerAction(lambdas=dict(self.lambdas), flags={l: 1.0 for l in self.layers}, threshold_quantile=self.threshold_quantile)

    def state_dict(self) -> dict[str, Any]:
        return {"lambdas": dict(self.lambdas)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.lambdas.update({k: float(v) for k, v in state.get("lambdas", {}).items()})


def build_controller(kind: str, layers: Sequence[str], targets: Mapping[str, float], cfg: Mapping[str, Any], total_controller_steps: int, device, seed: int, state_extra: int = 1, lambda_value: float = 0.03, quantile_head_enabled: bool = True):
    """Factory used by every modality. ``kind`` in {none, fixed_dcr, macc_lite, full_macc}."""
    kind = str(kind)
    if kind == "none":
        return FixedController(layers, 0.0, float(cfg.get("threshold_quantile", 0.9)))
    if kind == "fixed_dcr":
        return FixedController(layers, float(lambda_value), float(cfg.get("threshold_quantile", 0.9)))
    if kind == "macc_lite":
        ml = dict(cfg.get("macc_lite", {}))
        return MACCLite(
            layers,
            targets,
            lambda_initial=float(lambda_value if lambda_value is not None else ml.get("lambda_initial", 0.03)),
            lambda_min=float(ml.get("lambda_min", 0.0)),
            lambda_max=float(ml.get("lambda_max", 0.5)),
            step_size=float(ml.get("eta_lambda", ml.get("controller_step_size", 0.05))),
            threshold_quantile=float(ml.get("threshold_quantile", cfg.get("threshold_quantile", 0.9))),
            error_mode=str(ml.get("error_mode", "log_ratio")),
            deadband=float(ml.get("deadband", 0.0)),
        )
    if kind == "full_macc":
        fm = dict(cfg.get("full_macc", {}))
        rw = dict(fm.get("reward", {}))
        return FullMACC(
            layers,
            targets,
            state_dim=FullMACC.state_dim_for(len(layers), state_extra),
            lambda_initial=float(lambda_value if lambda_value is not None else fm.get("lambda_initial", 0.03)),
            lambda_min=float(fm.get("lambda_min", 0.0)),
            lambda_max=float(fm.get("lambda_max", 0.5)),
            hidden_dims=tuple(fm.get("hidden_dimensions", (128, 128))),
            lambda_delta_choices=tuple(fm.get("action_space", {}).get("lambda_delta_choices", (-0.02, -0.01, 0.0, 0.01, 0.02))),
            threshold_quantiles=tuple(fm.get("action_space", {}).get("threshold_quantiles", (0.80, 0.85, 0.90, 0.95))),
            policy_lr=float(fm.get("policy_learning_rate", 3e-4)),
            epsilon_start=float(fm.get("exploration_epsilon_start", 0.2)),
            epsilon_end=float(fm.get("exploration_epsilon_end", 0.02)),
            total_controller_steps=int(total_controller_steps),
            reward_baseline_ema=float(fm.get("reward_baseline_ema", 0.05)),
            entropy_coefficient=float(fm.get("entropy_coefficient", 1e-3)),
            reward_value_weight=float(rw.get("validation_delta_weight", 1.0)),
            reward_value_scale=float(rw.get("validation_delta_scale", 10.0)),
            reward_query_beta=float(rw.get("query_penalty_beta", 0.0)),
            reward_target_gamma=float(rw.get("target_deviation_gamma", 0.1)),
            device=device,
            seed=seed,
            quantile_head_enabled=bool(quantile_head_enabled),
            fixed_threshold_quantile=float(cfg.get("threshold_quantile", 0.9)),
            value_delta_clip=rw.get("value_delta_clip", 2.0),
        )
    raise ValueError(f"unknown controller kind: {kind}")
