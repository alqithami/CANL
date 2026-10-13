"""Task-aligned classification DCR and bounded feedback: proposed development variant.

The loss and controller both use log((within_trace+eps)/(between_trace+eps))
from class-balanced, per-example unit-normalized features. This is not a claim
of an ETF, canonical pseudoinverse NC1, input-Lipschitz control, or robustness.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor
import torch.nn.functional as F


def class_scatter(features: Tensor, labels: Tensor, *, eps: float = 1e-6,
                  norm_eps: float = 1e-6, require_balanced: bool = True):
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.numel():
        raise ValueError('Expected features [n,d] and labels [n]')
    if eps <= 0 or norm_eps <= 0:
        raise ValueError('Stabilizers must be positive')
    # Keep float64 for tests; use float32 for the actual BF16 model activations.
    z = features if features.dtype == torch.float64 else features.float()
    h = F.normalize(z, p=2, dim=1, eps=norm_eps)
    classes, inv, counts = torch.unique(labels, sorted=True, return_inverse=True, return_counts=True)
    if classes.numel() < 2 or int(counts.min()) < 2:
        raise ValueError('At least two classes with at least two examples each are required')
    if require_balanced and not torch.equal(counts, counts[:1].expand_as(counts)):
        raise ValueError('Loss batches must contain equally many examples per selected class')
    # Small dense class incidence matrix avoids CUDA scatter-add nondeterminism.
    incidence = F.one_hot(inv, classes.numel()).to(h.dtype)
    means = incidence.T @ h / counts.to(h.dtype)[:, None]
    weights = torch.ones_like(counts, dtype=h.dtype) / len(counts)
    grand = (weights[:, None] * means).sum(0)
    residual = (h - means[inv]).square().sum(1)
    per_class_w = incidence.T @ residual / counts.to(h.dtype)
    within = (weights * per_class_w).sum()
    between = (weights * (means - grand).square().sum(1)).sum()
    total = (weights * (incidence.T @ (h-grand).square().sum(1) / counts.to(h.dtype))).sum()
    log_kappa = torch.log(within + eps) - torch.log(between + eps)
    return {'within': within, 'between': between, 'total': total,
            'log_kappa': log_kappa, 'kappa': torch.exp(log_kappa)}


def positive_huber(gap: Tensor, delta: float = 0.25) -> Tensor:
    """One-sided Huber: derivative in [0,1], zero for satisfied upper-bound targets."""
    if delta <= 0:
        raise ValueError('delta must be positive')
    u = gap.clamp_min(0)
    return torch.where(u < delta, 0.5 * u.square() / delta, u - 0.5 * delta)


def aligned_loss(features: dict[str, Tensor], labels: Tensor,
                 targets: dict, lambdas: dict[str, float], cfg: dict):
    first = next(iter(features.values()))
    total = first.new_zeros((), dtype=torch.float32)
    records = {}
    for layer, z in features.items():
        s = class_scatter(z, labels, eps=cfg['scatter_epsilon'], norm_eps=cfg['feature_norm_epsilon'])
        gap = s['log_kappa'] - float(targets[layer]['log_kappa_target'])
        spread_gap = math.log(float(targets[layer]['between_floor'])) - torch.log(s['between'] + cfg['scatter_epsilon'])
        ratio_pen = positive_huber(gap, cfg['huber_delta'])
        floor_pen = positive_huber(spread_gap, cfg['huber_delta'])
        lam = float(lambdas[layer])
        if not math.isfinite(lam) or lam < 0:
            raise ValueError('Invalid lambda')
        # The fixed spread safeguard remains active even when a policy switches off ratio DCR.
        total = total + lam * ratio_pen + cfg['floor_weight'] * floor_pen
        records[layer] = {**{k:float(v.detach()) for k,v in s.items()},
                          'log_gap':float(gap.detach()), 'ratio_penalty':float(ratio_pen.detach()),
                          'spread_penalty':float(floor_pen.detach()), 'lambda':lam}
    return total, records


class LeakyMACCLite:
    """New bounded proportional feedback, NOT silently substituted for legacy MACC-Lite."""
    def __init__(self, layers, initial=.03, maximum=.1, smoothing=.2, scale=.25):
        if not (0 <= initial <= maximum and 0 < smoothing <= 1 and scale > 0):
            raise ValueError('Invalid controller bounds')
        self.layers=list(layers); self.maximum=float(maximum); self.smoothing=float(smoothing)
        self.scale=float(scale); self.lambdas={l:float(initial) for l in layers}; self.history=[]
    def step(self, log_gaps: dict, step: int):
        desired={}
        for l in self.layers:
            g=float(log_gaps[l])
            if not math.isfinite(g): raise FloatingPointError('Nonfinite control residual')
            desired[l]=self.maximum*math.tanh(max(0.0,g)/self.scale)
            self.lambdas[l]=(1-self.smoothing)*self.lambdas[l]+self.smoothing*desired[l]
            if not 0<=self.lambdas[l]<=self.maximum+1e-12:raise RuntimeError('Control bound violated')
        rec={'step':step,'log_gaps':dict(log_gaps),'desired':desired,'lambdas':dict(self.lambdas)}
        self.history.append(rec); return dict(self.lambdas)
    def state_dict(self):return {'lambdas':dict(self.lambdas),'history':list(self.history)}
    def load_state_dict(self,state):
        self.lambdas={k:float(v) for k,v in state['lambdas'].items()};self.history=list(state['history'])
        if set(self.lambdas)!=set(self.layers) or any(not 0<=v<=self.maximum for v in self.lambdas.values()):
            raise ValueError('Incompatible controller state')
