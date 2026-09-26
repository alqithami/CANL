"""Acquisition functions evaluated under identical settings.

Baselines (all share the same model, schedule, splits and candidate passes):

* ``random``
* ``entropy``            softmax entropy (Settles, 2009)
* ``margin``             smallest top-2 probability gap (strong baseline, Bahri et al., 2022)
* ``least_confidence``
* ``power_margin``       stochastic batch acquisition (Kirsch et al., TMLR 2023): Gumbel-top-k on beta*log score
* ``coreset``            k-Center greedy on penultimate features (Sener & Savarese, ICLR 2018)
* ``badge``              k-means++ seeding on hallucinated last-layer gradient embeddings (Ash et al., ICLR 2020);
                         exact factorised distances avoid materialising the ``C*d``-dimensional embeddings.
* ``bald_mcd``           BALD with Monte-Carlo feature dropout on the penultimate layer (Gal et al., 2017 / DropQuery-style)
* ``noise_stability``    parameter-noise output deviation + k-Center (Li et al., AAAI 2024)
* ``collapse``           CAENL collapse-instability score: layer-weighted (diagonal / shrunk full) Mahalanobis
                         distance to the predicted-class centre with quantile threshold (manuscript Eq. 7-8)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ClassStats:
    mode: str  # "diag" or "full"
    means: Tensor  # [C, d]
    var: Optional[Tensor] = None  # [C, d]  (diag)
    chol: Optional[Tensor] = None  # [C, d, d] (full; lower Cholesky of shrunk covariance)
    valid: Optional[Tensor] = None  # [C] bool
    dim: int = 0


@dataclass
class CandidateBatch:
    indices: np.ndarray
    logits: Tensor
    feats: dict[str, Tensor]


@dataclass
class AcquisitionContext:
    method: Mapping[str, Any]
    params: Mapping[str, Any]
    model: Any
    device: torch.device
    rng: np.random.Generator
    torch_gen: torch.Generator
    penultimate: str
    forward_fn: Optional[Callable[[np.ndarray], tuple[Tensor, dict[str, Tensor]]]] = None
    labeled_feats: Optional[Tensor] = None
    class_stats: Optional[dict[str, ClassStats]] = None
    layer_weights: Optional[dict[str, float]] = None
    threshold_quantile: float = 0.90


@dataclass
class SelectionResult:
    positions: np.ndarray
    scores: Optional[np.ndarray] = None
    info: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------- utilities
def _probs(logits: Tensor) -> Tensor:
    return F.softmax(logits.float(), dim=1)


def entropy_scores(logits: Tensor) -> Tensor:
    p = _probs(logits)
    return -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1)


def margin_scores(logits: Tensor) -> Tensor:
    p = _probs(logits)
    top2 = p.topk(2, dim=1).values
    return 1.0 - (top2[:, 0] - top2[:, 1])  # higher = more uncertain


def least_confidence_scores(logits: Tensor) -> Tensor:
    return 1.0 - _probs(logits).max(dim=1).values


def top_b(scores: Tensor, b: int) -> np.ndarray:
    b = int(min(b, scores.numel()))
    return scores.topk(b).indices.cpu().numpy()


def device_generator(gen: torch.Generator, device: torch.device) -> torch.Generator:
    """Derive a generator living on ``device`` from the (CPU) acquisition generator."""
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=gen).item())
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def rand_on(shape, gen: torch.Generator, device: torch.device) -> Tensor:
    g = device_generator(gen, device) if device.type != gen.device.type else gen
    return torch.rand(shape, generator=g, device=device)


def randn_on(shape, gen: torch.Generator, device: torch.device) -> Tensor:
    g = device_generator(gen, device) if device.type != gen.device.type else gen
    return torch.randn(shape, generator=g, device=device)


def rademacher_projection(
    in_dim: int,
    out_dim: int,
    gen: torch.Generator,
    device: torch.device,
    *,
    scale: bool = True,
) -> Tensor:
    """Return a deterministic dense Rademacher projection matrix.

    The matrix is generated from the acquisition RNG, so paired methods and
    resumed jobs remain reproducible.  Dense projections are intentional here:
    for the largest production case (2048 x 256) the matrix is small, while the
    resulting GEMM is substantially faster than running farthest-first or
    k-means++ in the original feature/gradient dimension.
    """

    if out_dim <= 0:
        raise ValueError("out_dim must be positive")
    g = device_generator(gen, device) if device.type != gen.device.type else gen
    r = torch.randint(0, 2, (int(in_dim), int(out_dim)), generator=g, device=device, dtype=torch.int8)
    r = r.to(torch.float32).mul_(2.0).sub_(1.0)
    if scale:
        r.mul_(1.0 / math.sqrt(float(out_dim)))
    return r


def maybe_project(X: Tensor, out_dim: int, gen: torch.Generator) -> tuple[Tensor, Optional[Tensor]]:
    """Project rows of ``X`` when ``out_dim`` is smaller than the input width."""

    X = X.float()
    if out_dim <= 0 or out_dim >= X.shape[1]:
        return X, None
    R = rademacher_projection(X.shape[1], out_dim, gen, X.device)
    return X @ R, R


def gumbel_top_b(log_scores: Tensor, b: int, beta: float, gen: torch.Generator) -> np.ndarray:
    u = rand_on(log_scores.shape, gen, log_scores.device).clamp(1e-12, 1 - 1e-12)
    g = -torch.log(-torch.log(u))
    return top_b(beta * log_scores + g, b)


def k_center_greedy(X: Tensor, b: int, init_min_dist: Optional[Tensor] = None, first: Optional[int] = None, chunk: int = 65536) -> tuple[np.ndarray, Tensor]:
    """Farthest-first traversal. ``init_min_dist`` = distance of each point to the existing centres."""
    n = X.shape[0]
    b = int(min(b, n))
    X = X.float()
    sqn = (X * X).sum(dim=1)
    if init_min_dist is None:
        min_d = torch.full((n,), float("inf"), device=X.device)
        if first is None:
            first = int(torch.argmax(sqn).item())
    else:
        min_d = init_min_dist.clone().float()
        first = int(torch.argmax(min_d).item()) if first is None else first
    selected: list[int] = []
    j = first
    for _ in range(b):
        selected.append(j)
        d2 = (sqn + sqn[j] - 2.0 * (X @ X[j])).clamp_min(0.0)
        min_d = torch.minimum(min_d, d2.sqrt())
        min_d[selected[-1]] = -1.0
        j = int(torch.argmax(min_d).item())
    return np.asarray(selected, dtype=np.int64), min_d


def min_dist_to_set(
    X: Tensor,
    Y: Tensor,
    chunk: int = 2048,
    reference_chunk: int = 4096,
) -> Tensor:
    """Min Euclidean distance from each row of ``X`` to ``Y``.

    Both axes are chunked.  The previous implementation chunked only ``X``;
    with ImageNet it could still construct a multi-gigabyte
    ``chunk x n_labeled`` distance matrix.  Two-dimensional chunking keeps peak
    memory bounded independently of the labeled-set size.
    """

    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError(f"expected compatible 2-D matrices, got X={tuple(X.shape)}, Y={tuple(Y.shape)}")
    if Y.shape[0] == 0:
        return torch.full((X.shape[0],), float("inf"), device=X.device, dtype=torch.float32)
    X = X.float()
    Y = Y.float()
    out = torch.empty(X.shape[0], device=X.device)
    for s in range(0, X.shape[0], chunk):
        xb = X[s : s + chunk]
        x2 = (xb * xb).sum(dim=1, keepdim=True)
        best = torch.full((xb.shape[0],), float("inf"), device=X.device)
        for t in range(0, Y.shape[0], reference_chunk):
            yb = Y[t : t + reference_chunk]
            y2 = (yb * yb).sum(dim=1)[None, :]
            d2 = (x2 + y2 - 2.0 * xb @ yb.T).clamp_min_(0)
            best = torch.minimum(best, d2.min(dim=1).values)
        out[s : s + chunk] = best.sqrt()
    return out


def kmeanspp_greedy(X: Tensor, b: int, gen: torch.Generator, rng: np.random.Generator) -> np.ndarray:
    """BADGE-style k-means++ seeding on an explicit embedding matrix."""

    X = X.float()
    n = int(X.shape[0])
    b = int(min(b, n))
    if b <= 0:
        return np.zeros(0, dtype=np.int64)
    sqn = (X * X).sum(dim=1)
    first = int(torch.argmax(sqn).item())
    selected = [first]
    selected_mask = torch.zeros(n, dtype=torch.bool, device=X.device)
    selected_mask[first] = True
    d2 = (sqn + sqn[first] - 2.0 * (X @ X[first])).clamp_min_(0)
    d2[first] = 0.0
    dev_gen = device_generator(gen, X.device) if X.device.type != gen.device.type else gen
    for _ in range(b - 1):
        mass = d2.masked_fill(selected_mask, 0.0).sum()
        if not torch.isfinite(mass) or float(mass) <= 0.0:
            remaining = torch.where(~selected_mask)[0].cpu().numpy()
            fill = rng.permutation(remaining)[: b - len(selected)]
            selected.extend(int(x) for x in fill)
            break
        probs = d2.masked_fill(selected_mask, 0.0) / mass
        j = int(torch.multinomial(probs, 1, generator=dev_gen).item())
        selected.append(j)
        selected_mask[j] = True
        new = (sqn + sqn[j] - 2.0 * (X @ X[j])).clamp_min_(0)
        d2 = torch.minimum(d2, new)
        d2[selected_mask] = 0.0
    return np.asarray(selected, dtype=np.int64)


# ----------------------------------------------------------------------------- strategies
def select_random(cand: CandidateBatch, b: int, ctx: AcquisitionContext) -> SelectionResult:
    n = len(cand.indices)
    return SelectionResult(positions=ctx.rng.permutation(n)[: min(b, n)])


def select_entropy(cand, b, ctx):
    s = entropy_scores(cand.logits)
    return SelectionResult(top_b(s, b), s.cpu().numpy())


def select_margin(cand, b, ctx):
    s = margin_scores(cand.logits)
    return SelectionResult(top_b(s, b), s.cpu().numpy())


def select_least_confidence(cand, b, ctx):
    s = least_confidence_scores(cand.logits)
    return SelectionResult(top_b(s, b), s.cpu().numpy())


def select_power_margin(cand, b, ctx):
    beta = float(ctx.params.get("power_beta", 1.0))
    s = margin_scores(cand.logits).clamp_min(1e-8)
    pos = gumbel_top_b(torch.log(s), b, beta, ctx.torch_gen)
    return SelectionResult(pos, s.cpu().numpy(), {"beta": beta})


def select_coreset(cand, b, ctx):
    X = cand.feats[ctx.penultimate].to(ctx.device).float()
    projection_dim = int(ctx.params.get("projection_dim", 0))
    Xp, R = maybe_project(X, projection_dim, ctx.torch_gen)
    init = None
    if ctx.labeled_feats is not None and ctx.labeled_feats.shape[0] > 0:
        Y = ctx.labeled_feats.to(ctx.device).float()
        if R is not None:
            Y = Y @ R
        init = min_dist_to_set(
            Xp,
            Y,
            chunk=int(ctx.params.get("distance_chunk", 2048)),
            reference_chunk=int(ctx.params.get("reference_chunk", 4096)),
        )
    pos, _ = k_center_greedy(Xp, b, init_min_dist=init)
    return SelectionResult(
        pos,
        None,
        {
            "labeled_reference": int(ctx.labeled_feats.shape[0]) if ctx.labeled_feats is not None else 0,
            "projection_dim": int(Xp.shape[1]),
            "original_feature_dim": int(X.shape[1]),
            "implementation": "jl_projected_kcenter" if R is not None else "exact_kcenter",
        },
    )


def select_badge(cand, b, ctx):
    logits = cand.logits.to(ctx.device).float()
    H = cand.feats[ctx.penultimate].to(ctx.device).float()
    P = F.softmax(logits, dim=1)
    pred = P.argmax(dim=1)
    U = P.clone()
    U[torch.arange(U.shape[0], device=U.device), pred] -= 1.0  # p - e_yhat
    sketch_dim = int(ctx.params.get("sketch_dim", 0))
    if sketch_dim > 0:
        # Random-Maclaurin sketch of the product kernel
        # <u_i⊗h_i, u_j⊗h_j> = <u_i,u_j><h_i,h_j>.  This avoids materialising
        # the C*d gradient embedding while retaining its geometry in expectation.
        Ru = rademacher_projection(U.shape[1], sketch_dim, ctx.torch_gen, U.device, scale=False)
        Rh = rademacher_projection(H.shape[1], sketch_dim, ctx.torch_gen, H.device, scale=False)
        G = ((U @ Ru) * (H @ Rh)) / math.sqrt(float(sketch_dim))
        selected = kmeanspp_greedy(G, b, ctx.torch_gen, ctx.rng)
        return SelectionResult(
            selected,
            None,
            {
                "embedding_dim": int(U.shape[1] * H.shape[1]),
                "sketch_dim": sketch_dim,
                "implementation": "random_maclaurin_gradient_sketch",
            },
        )

    # Exact factorised gradient-kernel distances.  Suitable for small candidate
    # pools; ImageNet plans explicitly enable the sketch above.
    usq = (U * U).sum(dim=1)
    hsq = (H * H).sum(dim=1)
    sqn = usq * hsq
    n = U.shape[0]
    b = int(min(b, n))
    first = int(torch.argmax(sqn).item())
    selected = [first]
    selected_mask = torch.zeros(n, dtype=torch.bool, device=U.device)
    selected_mask[first] = True
    d2 = (sqn + sqn[first] - 2.0 * (U @ U[first]) * (H @ H[first])).clamp_min_(0)
    d2[first] = 0.0
    gen = device_generator(ctx.torch_gen, U.device) if U.device.type != ctx.torch_gen.device.type else ctx.torch_gen
    for _ in range(b - 1):
        mass = d2.masked_fill(selected_mask, 0.0).sum()
        if not torch.isfinite(mass) or float(mass) <= 0.0:
            remaining = torch.where(~selected_mask)[0].cpu().numpy()
            fill = ctx.rng.permutation(remaining)[: b - len(selected)]
            selected.extend(int(x) for x in fill)
            break
        probs = d2.masked_fill(selected_mask, 0.0) / mass
        j = int(torch.multinomial(probs, 1, generator=gen).item())
        selected.append(j)
        selected_mask[j] = True
        new = (sqn + sqn[j] - 2.0 * (U @ U[j]) * (H @ H[j])).clamp_min_(0)
        d2 = torch.minimum(d2, new)
        d2[selected_mask] = 0.0
    return SelectionResult(np.asarray(selected, dtype=np.int64), None, {"embedding_dim": int(U.shape[1] * H.shape[1]), "implementation": "exact_factorised"})


def select_bald_mcd(cand, b, ctx):
    T = int(ctx.params.get("mc_samples", 20))
    p_drop = float(ctx.params.get("dropout_p", 0.3))
    H = cand.feats[ctx.penultimate].to(ctx.device).float()
    model = ctx.model
    probs_sum = torch.zeros(H.shape[0], cand.logits.shape[1], device=ctx.device)
    ent_sum = torch.zeros(H.shape[0], device=ctx.device)
    with torch.no_grad():
        for _ in range(T):
            mask = (rand_on(H.shape, ctx.torch_gen, H.device) >= p_drop).float() / (1 - p_drop)
            p = F.softmax(model.head(H * mask).float(), dim=1)
            probs_sum += p
            ent_sum += -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1)
    pm = probs_sum / T
    H_mean = -(pm * torch.log(pm.clamp_min(1e-12))).sum(dim=1)
    bald = H_mean - ent_sum / T
    return SelectionResult(top_b(bald, b), bald.cpu().numpy(), {"mc_samples": T, "dropout_p": p_drop})


def select_noise_stability(cand, b, ctx):
    if ctx.forward_fn is None:
        raise ValueError("noise_stability requires forward_fn")
    K = int(ctx.params.get("noise_samples", 10))
    zeta = float(ctx.params.get("noise_scale", 1e-3))
    model = ctx.model
    params = [p for p in model.parameters() if p.requires_grad]  # Li et al. perturb the full parameter vector theta
    base = F.softmax(cand.logits.to(ctx.device).float(), dim=1)
    with torch.no_grad():
        theta_norm = math.sqrt(sum(float((p.double() ** 2).sum()) for p in params))
    sketch_dim = int(ctx.params.get("sketch_dim", 0))
    sketch = torch.zeros(base.shape[0], sketch_dim, device=ctx.device) if sketch_dim > 0 else None
    devs = [] if sketch is None else None
    norm_sq = torch.zeros(base.shape[0], device=ctx.device)
    gen = ctx.torch_gen
    for k in range(K):
        with torch.no_grad():
            noise = [randn_on(p.shape, gen, p.device).to(p.dtype) for p in params]
            nn_ = math.sqrt(sum(float((u.double() ** 2).sum()) for u in noise))
            scale = zeta * theta_norm / max(nn_, 1e-12)
            for p, u in zip(params, noise):
                p.add_(u, alpha=scale)
            try:
                logits_k, _ = ctx.forward_fn(cand.indices)
                pk = F.softmax(logits_k.to(ctx.device).float(), dim=1)
            finally:
                for p, u in zip(params, noise):
                    p.sub_(u, alpha=scale)
        dev = (pk - base) / max(zeta * theta_norm, 1e-12)
        norm_sq += (dev * dev).sum(dim=1)
        if sketch is None:
            assert devs is not None
            devs.append(dev)
        else:
            # Project each K*C block independently and accumulate.  This is
            # algebraically equivalent to projecting the concatenated output
            # deviation with a block-partitioned random matrix, without ever
            # allocating the potentially multi-gigabyte [n, K*C] tensor.
            R = rademacher_projection(dev.shape[1], sketch_dim, gen, dev.device)
            sketch.add_(dev @ R)
    if sketch is None:
        assert devs is not None
        X = torch.cat(devs, dim=1)
        implementation = "exact_output_deviation"
    else:
        X = sketch / math.sqrt(float(K))
        implementation = "projected_output_deviation"
    pos, _ = k_center_greedy(X, b)
    norms = norm_sq.sqrt()
    return SelectionResult(pos, norms.cpu().numpy(), {"noise_samples": K, "noise_scale": zeta, "sketch_dim": int(X.shape[1]), "implementation": implementation})


def mahalanobis_scores(z: Tensor, pred: Tensor, stats: ClassStats) -> Tensor:
    z = z.float()
    out = torch.empty(z.shape[0], device=z.device)
    fallback_mean = z.mean(dim=0)
    fallback_var = z.var(dim=0, unbiased=False).clamp_min(1e-6)
    for c in torch.unique(pred).tolist():
        m = pred == c
        valid = bool(stats.valid[c]) if stats.valid is not None else True
        if not valid:
            out[m] = (((z[m] - fallback_mean) ** 2) / fallback_var).sum(dim=1)
            continue
        diff = z[m] - stats.means[c].to(z)
        if stats.mode == "full" and stats.chol is not None:
            L = stats.chol[c].to(z)
            y = torch.linalg.solve_triangular(L, diff.T, upper=False)
            out[m] = (y * y).sum(dim=0)
        else:
            out[m] = (diff * diff / stats.var[c].to(z)).sum(dim=1)
    return out


def select_collapse(cand, b, ctx):
    if not ctx.class_stats:
        raise ValueError("collapse acquisition requires class statistics")
    pred = cand.logits.argmax(dim=1).to(ctx.device)
    total = None
    per_layer: dict[str, Tensor] = {}
    for layer, stats in ctx.class_stats.items():
        if layer not in cand.feats:
            continue
        u = mahalanobis_scores(cand.feats[layer].to(ctx.device), pred, stats)
        w = float((ctx.layer_weights or {}).get(layer, 1.0))
        per_layer[layer] = u
        total = w * u if total is None else total + w * u
    if total is None:
        raise ValueError("no monitored layer features available for collapse scoring")
    tau = float(ctx.threshold_quantile)
    thr = torch.quantile(total, tau) if total.numel() > 1 else total.min()
    n_above = int((total >= thr).sum())
    cap = int(min(b, max(n_above, 1)))
    pos = top_b(total, cap)
    info = {"threshold_quantile": tau, "threshold_value": float(thr), "n_above_threshold": n_above}
    for layer, u in per_layer.items():
        info[f"mean_u/{layer}"] = float(u.mean())
    return SelectionResult(pos, total.cpu().numpy(), info)


STRATEGIES: dict[str, Callable[[CandidateBatch, int, AcquisitionContext], SelectionResult]] = {
    "random": select_random,
    "entropy": select_entropy,
    "margin": select_margin,
    "least_confidence": select_least_confidence,
    "power_margin": select_power_margin,
    "coreset": select_coreset,
    "badge": select_badge,
    "bald_mcd": select_bald_mcd,
    "noise_stability": select_noise_stability,
    "collapse": select_collapse,
}


def select(strategy: str, cand: CandidateBatch, budget: int, ctx: AcquisitionContext) -> SelectionResult:
    if strategy not in STRATEGIES:
        raise KeyError(f"unknown acquisition strategy '{strategy}'. Known: {sorted(STRATEGIES)}")
    if budget <= 0 or len(cand.indices) == 0:
        return SelectionResult(np.zeros(0, dtype=np.int64))
    res = STRATEGIES[strategy](cand, int(budget), ctx)
    pos = np.asarray(res.positions, dtype=np.int64)
    if len(np.unique(pos)) != len(pos):
        raise RuntimeError(f"strategy {strategy} returned duplicate positions")
    if pos.size > budget:
        pos = pos[:budget]
    res.positions = pos
    return res


@torch.no_grad()
def compute_class_stats(
    feats: Tensor,
    labels: Tensor,
    num_classes: int,
    mode: str = "diag",
    shrinkage_abs: float = 1e-5,
    shrinkage_rel: float = 0.1,
    min_count: int = 2,
    diag_relative_floor: float = 0.01,
) -> ClassStats:
    """Exact class-conditional statistics from labeled features (end-of-phase estimates)."""
    feats = feats.float()
    labels = labels.to(feats.device).long()
    C, d = int(num_classes), int(feats.shape[1])
    counts = torch.bincount(labels, minlength=C)
    means = torch.zeros(C, d, device=feats.device)
    means.index_add_(0, labels, feats)
    means = means / counts.clamp_min(1).to(means.dtype)[:, None]
    valid = counts >= min_count
    if mode == "full":
        chol = torch.zeros(C, d, d, device=feats.device)
        eye = torch.eye(d, device=feats.device)
        for c in range(C):
            if not valid[c]:
                continue
            diff = feats[labels == c] - means[c]
            cov = diff.T @ diff / float(diff.shape[0])
            tr = torch.trace(cov) / d
            cov = (1 - shrinkage_rel) * cov + (shrinkage_rel * tr + shrinkage_abs) * eye
            chol[c] = torch.linalg.cholesky(cov)
        return ClassStats(mode="full", means=means, chol=chol, valid=valid, dim=d)
    var = torch.zeros(C, d, device=feats.device)
    var.index_add_(0, labels, feats * feats)
    var = (var / counts.clamp_min(1).to(var.dtype)[:, None] - means * means).clamp_min(0.0)
    # absolute shrinkage (manuscript) plus a relative floor so dead/near-constant channels cannot dominate
    floor = diag_relative_floor * var[valid].mean(dim=1, keepdim=True).clamp_min(0.0) if bool(valid.any()) else 0.0
    var[valid] = var[valid] + floor
    var = var + shrinkage_abs
    return ClassStats(mode="diag", means=means, var=var, valid=valid, dim=d)
