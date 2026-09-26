"""Statistical reporting helpers (multi-seed means, CIs, paired tests, multiple-comparison control).

All paired procedures pair runs by seed: method A seed s versus method B seed s share the
same initial split, the same initial checkpoint (within a regulariser group) and the same
data order, so seed-level differences isolate the acquisition / regularisation effect.

With the five principal seeds an *exact* sign-flip permutation test has 2^5 = 32
configurations and cannot reach p < 0.0625 two-sided; the paired t-test is therefore the
primary seed-level test and bootstrap CIs / Cohen's d_z quantify magnitude.  Exact
permutation and Wilcoxon p-values are reported alongside for completeness.  Per-example
McNemar tests on AutoAttack robust masks provide a complementary, higher-powered view of
robustness differences for each seed.
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import stats as sps


def summarize(values: Iterable[float], confidence: float = 0.95) -> dict[str, float]:
    x = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    n = int(x.size)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "sem": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "min": float("nan"), "max": float("nan")}
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else 0.0
    if n > 1:
        tcrit = float(sps.t.ppf(0.5 + confidence / 2.0, df=n - 1))
        lo, hi = mean - tcrit * sem, mean + tcrit * sem
    else:
        lo, hi = float("nan"), float("nan")
    return {"n": n, "mean": mean, "std": std, "sem": sem, "ci_low": lo, "ci_high": hi, "min": float(x.min()), "max": float(x.max())}


def _pair(a: Sequence[float], b: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same length")
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]


def paired_bootstrap_ci(a: Sequence[float], b: Sequence[float], n_boot: int = 10000, confidence: float = 0.95, seed: int = 0) -> dict[str, float]:
    a, b = _pair(a, b)
    d = a - b
    n = d.size
    if n == 0:
        return {"diff_mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    boots = d[idx].mean(axis=1)
    alpha = 1.0 - confidence
    return {"diff_mean": float(d.mean()), "ci_low": float(np.quantile(boots, alpha / 2)), "ci_high": float(np.quantile(boots, 1 - alpha / 2))}


def paired_permutation_test(a: Sequence[float], b: Sequence[float], n_perm: int = 10000, seed: int = 0, exact_max_n: int = 14) -> dict[str, Any]:
    """Two-sided sign-flip test of mean(a-b)=0. Exact enumeration when n <= exact_max_n."""
    a, b = _pair(a, b)
    d = a - b
    n = d.size
    if n == 0:
        return {"p_value": float("nan"), "exact": False, "n_perm": 0}
    obs = abs(d.mean())
    if n <= exact_max_n:
        count = 0
        total = 0
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            total += 1
            if abs((d * np.asarray(signs)).mean()) >= obs - 1e-12:
                count += 1
        return {"p_value": count / total, "exact": True, "n_perm": total, "min_attainable_p": 2.0 / total}
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(int(n_perm), n))
    stat = np.abs((signs * d).mean(axis=1))
    p = (1 + int((stat >= obs - 1e-12).sum())) / (n_perm + 1)
    return {"p_value": float(p), "exact": False, "n_perm": int(n_perm)}


def paired_t_test(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    a, b = _pair(a, b)
    if a.size < 2:
        return {"t": float("nan"), "p_value": float("nan"), "df": float(a.size - 1)}
    d = a - b
    if np.allclose(d, d[0]):
        # zero variance of paired differences: the t statistic is undefined
        return {"t": float("nan"), "p_value": float("nan") if d[0] != 0 else 1.0, "df": float(a.size - 1), "note": "constant paired differences"}
    res = sps.ttest_rel(a, b)
    return {"t": float(res.statistic), "p_value": float(res.pvalue), "df": float(a.size - 1)}


def welch_t_test(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    a = np.asarray([x for x in a if np.isfinite(x)], dtype=float)
    b = np.asarray([x for x in b if np.isfinite(x)], dtype=float)
    if a.size < 2 or b.size < 2:
        return {"t": float("nan"), "p_value": float("nan")}
    res = sps.ttest_ind(a, b, equal_var=False)
    return {"t": float(res.statistic), "p_value": float(res.pvalue)}


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    a, b = _pair(a, b)
    d = a - b
    d = d[d != 0]
    if d.size < 6:
        return {"p_value": float("nan"), "note": "n<6: Wilcoxon cannot reach p<0.05"}
    try:
        res = sps.wilcoxon(d, alternative="two-sided")
        return {"p_value": float(res.pvalue)}
    except ValueError:
        return {"p_value": float("nan")}


def cohens_dz(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = _pair(a, b)
    d = a - b
    if d.size < 2:
        return float("nan")
    sd = d.std(ddof=1)
    if sd == 0:
        # A standardized effect is undefined when every paired difference is identical.
        # Preserve the raw paired difference elsewhere; never emit an infinite effect size.
        return float("nan")
    return float(d.mean() / sd)


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (NaNs are passed through and not counted)."""
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.where(np.isfinite(p))[0]
    m = valid.size
    if m == 0:
        return out.tolist()
    order = valid[np.argsort(p[valid])]
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * p[idx])
        running = max(running, adj)
        out[idx] = running
    return out.tolist()


def mcnemar_test(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict[str, Any]:
    """Exact (binomial) McNemar test on paired per-example correctness masks."""
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("masks must have equal length")
    n01 = int(np.sum(~a & b))  # only B correct
    n10 = int(np.sum(a & ~b))  # only A correct
    n = n01 + n10
    if n == 0:
        return {"n_discordant": 0, "a_only": n10, "b_only": n01, "p_value": 1.0, "method": "exact", "acc_diff": float(a.mean() - b.mean())}
    if n <= 2000:
        p = float(sps.binomtest(min(n01, n10), n, 0.5, alternative="two-sided").pvalue)
        method = "exact"
    else:
        chi2 = (abs(n10 - n01) - 1) ** 2 / n
        p = float(sps.chi2.sf(chi2, df=1))
        method = "chi2_cc"
    return {"n_discordant": n, "a_only": n10, "b_only": n01, "p_value": p, "method": method, "acc_diff": float(a.mean() - b.mean())}


def compare_paired(a: Sequence[float], b: Sequence[float], n_boot: int = 10000, n_perm: int = 10000, seed: int = 0, confidence: float = 0.95) -> dict[str, Any]:
    """Complete paired comparison of A minus B."""
    a_arr, b_arr = _pair(a, b)
    out: dict[str, Any] = {"n_pairs": int(a_arr.size), "mean_a": float(a_arr.mean()) if a_arr.size else float("nan"), "mean_b": float(b_arr.mean()) if b_arr.size else float("nan")}
    out.update({f"boot_{k}": v for k, v in paired_bootstrap_ci(a_arr, b_arr, n_boot, confidence, seed).items()})
    t = paired_t_test(a_arr, b_arr)
    out["t_stat"], out["p_ttest"] = t["t"], t["p_value"]
    perm = paired_permutation_test(a_arr, b_arr, n_perm, seed)
    out["p_perm"], out["perm_exact"] = perm["p_value"], perm["exact"]
    out["p_wilcoxon"] = wilcoxon_signed_rank(a_arr, b_arr)["p_value"]
    out["cohens_dz"] = cohens_dz(a_arr, b_arr)
    if a_arr.size < 3:
        out["inference_note"] = "descriptive only: fewer than 3 paired seeds"
    elif not math.isfinite(out["cohens_dz"]):
        out["effect_note"] = "Cohen d_z undefined because paired differences have zero variance"
    out["wins"] = int(np.sum(a_arr > b_arr)) if a_arr.size else 0
    out["losses"] = int(np.sum(a_arr < b_arr)) if a_arr.size else 0
    return out


def format_mean_std(mean: float, std: float, digits: int = 2, pm: str = "\\pm") -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "--"
    if std is None or (isinstance(std, float) and math.isnan(std)):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} {pm} {std:.{digits}f}"


def significance_marker(p: float) -> str:
    if p is None or not math.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""
