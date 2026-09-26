"""LaTeX tables generated from aggregated CSVs (booktabs, mean +- std over seeds)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..core.stats import significance_marker
from .style import DATASET_LABEL, METRIC_LABEL, PERCENT_METRICS, method_label, method_sort_key
from .context import human_count

_ACTIVE_BANNER = ""


def _fmt(mean: float, std: float, metric: str, digits: Optional[int] = None) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "--"
    scale = 100.0 if metric in PERCENT_METRICS else 1.0
    if metric == "kid":
        scale = 1000.0
    d = digits if digits is not None else (1 if metric in PERCENT_METRICS else (3 if metric in ("test_ece", "token_ece", "validation_nll", "kid") else 2))
    m = mean * scale
    if std is None or (isinstance(std, float) and math.isnan(std)) or std == 0:
        return f"{m:.{d}f}"
    return f"{m:.{d}f} $\\pm$ {std * scale:.{d}f}"


def _esc(s: str) -> str:
    return s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def _wrap(body: str, caption: str, label: str, colspec: str, header: str, note: str = "", small: bool = True) -> str:
    size = "\\small" if small else ""
    banner = _ACTIVE_BANNER.strip()
    full_caption = f"{banner}: {caption}" if banner else caption
    lines = ["\\begin{table}[t]", "\\centering", size, f"\\caption{{{full_caption}}}", f"\\label{{{label}}}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " " + chr(92) * 2, "\\midrule", body, "\\bottomrule", "\\end{tabular}"]
    if note:
        lines.append(f"\\par\\smallskip\\footnotesize {note}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


FULL_SUPERVISION_METHODS = {"supervised_full"}
AUDIO_DATASET_LABEL = {"audiocaps": "AudioCaps", "clotho": "Clotho v2.1", "synthetic": "Synthetic audio", "manifest_smoke": "Synthetic audio (manifest)"}


def _final_frac(grouped: pd.DataFrame, dataset: str, arch: str) -> float:
    """Final active-learning label fraction (full-supervision reference rows excluded)."""
    g = grouped[(grouped["dataset"] == dataset) & (grouped["arch"] == arch) & (grouped["round"] >= 0) & (~grouped["method"].isin(FULL_SUPERVISION_METHODS))]
    if g.empty:
        g = grouped[(grouped["dataset"] == dataset) & (grouped["arch"] == arch) & (grouped["round"] >= 0)]
    return float(g["label_fraction"].max()) if not g.empty else float("nan")


def _lookup(grouped: pd.DataFrame, dataset: str, arch: str, method: str, metric: str, label_fraction: Optional[float] = None, round_: Optional[int] = None) -> tuple[float, float, int]:
    g = grouped[(grouped["dataset"] == dataset) & (grouped["arch"] == arch) & (grouped["method"] == method) & (grouped["metric"] == metric)]
    if label_fraction is not None:
        g = g[np.isclose(g["label_fraction"].astype(float), label_fraction, atol=1e-6)]
    if round_ is not None:
        g = g[g["round"] == round_]
    if g.empty:
        return float("nan"), float("nan"), 0
    r = g.iloc[-1]
    return float(r["mean"]), float(r["std"]), int(r["n"])


def _marker(sig: pd.DataFrame, dataset: str, arch: str, target: str, ref: str, metric: str) -> str:
    if sig is None or sig.empty:
        return ""
    s = sig[(sig["dataset"] == dataset) & (sig["arch"] == arch) & (sig["target"] == target) & (sig["reference"] == ref) & (sig["metric"] == metric)]
    if s.empty:
        return ""
    p = float(s.iloc[0].get("p_ttest_holm", float("nan")))
    return significance_marker(p)


def _aa_scope(protocol: Optional[dict], dataset: str) -> str:
    """How many test examples the final AutoAttack evaluation used for ``dataset`` (from the frozen protocol)."""
    if not protocol:
        return "the test set"
    fin = dict((protocol.get("robustness", {}) or {}).get("final", {}) or {})
    fin.update(((protocol.get("dataset_overrides", {}) or {}).get(dataset, {}) or {}).get("robustness", {}).get("final", {}) or {})
    n = fin.get("n_examples")
    return "the full test set" if not n else f"a fixed class-stratified {int(n):,}-example test subset (identical for every method and seed)"


def main_results_table(grouped: pd.DataFrame, sig: pd.DataFrame, dataset: str, arch: str, target: str = "caenl_macc_lite", protocol: Optional[dict] = None) -> str:
    frac = _final_frac(grouped, dataset, arch)
    methods = sorted(grouped[(grouped["dataset"] == dataset) & (grouped["arch"] == arch)]["method"].unique(), key=method_sort_key)
    cols = [("test_acc", "Clean"), ("aulc_test_acc", "AULC"), ("aa_robust_acc", "AA ($8/255$)"), ("apgd_ce_subset_acc", "APGD-CE"), ("pgd_subset_acc", "PGD-20"), ("corruption_mean_acc", "Corrupt."), ("test_ece", "ECE")]
    rows = []
    for m in methods:
        f_m = None if m in FULL_SUPERVISION_METHODS else frac
        cells = [_esc(method_label(m))]
        for metric, _ in cols:
            if metric.startswith("aulc_"):
                mean, std, n = _lookup(grouped, dataset, arch, m, metric, round_=-1)
            else:
                mean, std, n = _lookup(grouped, dataset, arch, m, metric, label_fraction=f_m)
            cell = _fmt(mean, std, metric)
            if m != target and not math.isnan(mean) and m not in FULL_SUPERVISION_METHODS:
                cell += _marker(sig, dataset, arch, target, m, metric)
            cells.append(cell)
        _, _, n = _lookup(grouped, dataset, arch, m, "test_acc", label_fraction=f_m)
        cells.append(str(n))
        if m in FULL_SUPERVISION_METHODS:
            rows.append("\\midrule\n" + " & ".join(cells))
        else:
            rows.append(" & ".join(cells))
    header = "Method & " + " & ".join(c for _, c in cols) + " & seeds"
    note = (f"Mean $\\pm$ std over seeds at {frac * 100:.0f}\\% labels. Accuracies in \\%; AULC = mean clean accuracy over all label fractions; AA = AutoAttack standard ($L_\\infty$, $\\epsilon=8/255$) on {_aa_scope(protocol, dataset)}; "
            "APGD-CE / PGD-20 / corruption accuracies on a fixed 2k-example subset. Stars mark paired $t$-tests of CAENL (MACC-Lite) versus the row "
            "(Holm-corrected within each column): * $p<.05$, ** $p<.01$, *** $p<.001$. "
            "Label budgets include the controller-validation carve-out that every method pays for (labels available); gradient steps use the remainder (labels trained). "
            "Full supervision has 100\\% of the labels available and trains on all but the same carve-out.")
    return _wrap("\\\\\n".join(rows) + " \\\\", f"{DATASET_LABEL.get(dataset, dataset)} ({arch}) results under a fixed {frac * 100:.0f}\\% label budget.", f"tab:main_{dataset}_{arch}", "l" + "c" * (len(cols) + 1), header, note)


def curve_table(grouped: pd.DataFrame, dataset: str, arch: str, metric: str) -> str:
    g = grouped[(grouped["dataset"] == dataset) & (grouped["arch"] == arch) & (grouped["metric"] == metric) & (grouped["round"] >= 0)]
    if g.empty:
        return ""
    fracs = sorted(g["label_fraction"].unique())
    methods = sorted(g["method"].unique(), key=method_sort_key)
    rows = []
    for m in methods:
        cells = [_esc(method_label(m))]
        for f in fracs:
            mean, std, _ = _lookup(grouped, dataset, arch, m, metric, label_fraction=f)
            cells.append(_fmt(mean, std, metric))
        rows.append(" & ".join(cells))
    header = "Method & " + " & ".join(f"{f * 100:.0f}\\%" for f in fracs)
    return _wrap("\\\\\n".join(rows) + " \\\\", f"{METRIC_LABEL.get(metric, metric)} versus label fraction on {DATASET_LABEL.get(dataset, dataset)} (mean $\\pm$ std over seeds).", f"tab:curve_{metric}_{dataset}_{arch}", "l" + "c" * len(fracs), header)


def ablation_table(grouped: pd.DataFrame, sig: pd.DataFrame, dataset: str, arch: str) -> str:
    """Acquisition x regulariser factorial at the final budget."""
    frac = _final_frac(grouped, dataset, arch)
    grid = [
        ("Random", "none", "random"), ("Random", "fixed DCR", "dcr_fixed_random"), ("Random", "MACC-Lite", "dcr_lite_random"),
        ("Entropy", "none", "entropy"), ("Entropy", "MACC-Lite", "entropy_macc_lite"),
        ("Collapse (CAENL)", "none", "caenl_acq_only"), ("Collapse (CAENL)", "fixed DCR", "caenl_fixed_dcr"), ("Collapse (CAENL)", "MACC-Lite", "caenl_macc_lite"), ("Collapse (CAENL)", "Full MACC", "caenl_full_macc"),
    ]
    rows = []
    for acqn, regn, m in grid:
        mean, std, n = _lookup(grouped, dataset, arch, m, "test_acc", label_fraction=frac)
        if n == 0:
            continue
        cells = [acqn, regn]
        for metric in ("test_acc", "aa_robust_acc", "apgd_ce_subset_acc", "kappa_train/layer4", "nc/layer4/nc2_etf_distance"):
            mu, sd, _ = _lookup(grouped, dataset, arch, m, metric, label_fraction=frac)
            cells.append(_fmt(mu, sd, metric, digits=(1 if metric in PERCENT_METRICS else 3)))
        cells.append(str(n))
        rows.append(" & ".join(cells))
    header = "Acquisition & Regulariser & Clean & AA & APGD-CE & $\\kappa_{4}$ (train) & ETF dist. & seeds"
    return _wrap("\\\\\n".join(rows) + " \\\\", f"Component ablation on {DATASET_LABEL.get(dataset, dataset)}: acquisition signal $\\times$ collapse regulariser at {frac * 100:.0f}\\% labels.", f"tab:ablation_{dataset}_{arch}", "llcccccc", header)


def sweep_table(grouped: pd.DataFrame, dataset: str, arch: str) -> str:
    frac = _final_frac(grouped, dataset, arch)
    methods = [m for m in grouped[(grouped["dataset"] == dataset) & (grouped["arch"] == arch)]["method"].unique() if m == "caenl_acq_only" or m.startswith("caenl_fixed_dcr")]
    if len(methods) < 2:
        return ""

    def lam(m):
        if m == "caenl_acq_only":
            return 0.0
        if m == "caenl_fixed_dcr":
            return 0.03
        return float(m.split("_l")[-1])

    rows = []
    for m in sorted(methods, key=lam):
        cells = [f"{lam(m):g}"]
        for metric in ("test_acc", "aa_robust_acc", "apgd_ce_subset_acc", "corruption_mean_acc", "kappa_train/layer4", "nc/layer4/total_effective_rank", "margins/grad_norm_median"):
            mu, sd, _ = _lookup(grouped, dataset, arch, m, metric, label_fraction=frac)
            cells.append(_fmt(mu, sd, metric, digits=(1 if metric in PERCENT_METRICS else 3)))
        rows.append(" & ".join(cells))
    header = "$\\lambda$ & Clean & AA & APGD-CE & Corrupt. & $\\kappa_4$ & eff. rank & $\\|\\nabla_x m\\|$"
    return _wrap("\\\\\n".join(rows) + " \\\\", f"Fixed DCR strength sweep on {DATASET_LABEL.get(dataset, dataset)} (collapse acquisition; {frac * 100:.0f}\\% labels).", f"tab:sweep_{dataset}_{arch}", "lccccccc", header, "$\\kappa_4$ = within/between scatter ratio of the penultimate layer on training batches; eff. rank of the test-set feature covariance; median input-gradient norm of the logit margin.")


def controller_table(grouped: pd.DataFrame, dataset: str, arch: str) -> str:
    frac = _final_frac(grouped, dataset, arch)
    rows = []
    for m in ("caenl_fixed_dcr", "caenl_macc_lite", "caenl_full_macc"):
        mean, std, n = _lookup(grouped, dataset, arch, m, "test_acc", label_fraction=frac)
        if n == 0:
            continue
        cells = [_esc(method_label(m))]
        for metric in ("test_acc", "aa_robust_acc", "apgd_ce_subset_acc", "lambda/layer3", "lambda/layer4", "tau", "kappa_train/layer4", "controller_params", "total_train_time_s"):
            mu, sd, _ = _lookup(grouped, dataset, arch, m, metric, label_fraction=frac) if not metric.startswith(("controller", "total")) else _lookup(grouped, dataset, arch, m, metric, round_=-1)
            if metric == "controller_params":
                cells.append("--" if math.isnan(mu) else f"{int(mu):,}")
            elif metric == "total_train_time_s":
                cells.append("--" if math.isnan(mu) else f"{mu / 60:.1f}")
            else:
                cells.append(_fmt(mu, sd, metric, digits=(1 if metric in PERCENT_METRICS else 3)))
        rows.append(" & ".join(cells))
    if not rows:
        return ""
    header = "Controller & Clean & AA & APGD-CE & $\\lambda_3$ & $\\lambda_4$ & $\\tau$ & $\\kappa_4$ & policy params & train min"
    return _wrap("\\\\\n".join(rows) + " \\\\", f"MACC-Lite versus Full MACC on {DATASET_LABEL.get(dataset, dataset)} ({frac * 100:.0f}\\% labels): final regularisation strengths, query threshold, collapse and cost.", f"tab:controller_{dataset}_{arch}", "lccccccccc", header)


def nc_table(grouped: pd.DataFrame, dataset: str, arch: str, layer: str = "layer4") -> str:
    frac = _final_frac(grouped, dataset, arch)
    methods = sorted(grouped[(grouped["dataset"] == dataset) & (grouped["arch"] == arch)]["method"].unique(), key=method_sort_key)
    metrics = [(f"nc/{layer}/kappa", "$\\kappa$ (NC1)"), (f"nc/{layer}/nc1_papyan", "tr$(\\Sigma_W\\Sigma_B^\\dagger)/C$"), (f"nc/{layer}/nc2_etf_distance", "ETF dist. (NC2)"), (f"nc/{layer}/nc2_equinorm_cv", "equinorm CV"), (f"nc/{layer}/nc3_self_duality", "self-duality (NC3)"), (f"nc/{layer}/nc4_ncc_agreement", "NCC agree. (NC4)"), (f"nc/{layer}/total_effective_rank", "eff. rank"), ("margins/margin_mean", "margin"), ("margins/grad_norm_median", "$\\|\\nabla_x m\\|$")]
    rows = []
    for m in methods:
        mean, _, n = _lookup(grouped, dataset, arch, m, metrics[0][0], label_fraction=frac)
        if n == 0:
            continue
        cells = [_esc(method_label(m))]
        for metric, _ in metrics:
            mu, sd, _ = _lookup(grouped, dataset, arch, m, metric, label_fraction=frac)
            cells.append(_fmt(mu, sd, metric, digits=3))
        rows.append(" & ".join(cells))
    if not rows:
        return ""
    header = "Method & " + " & ".join(h for _, h in metrics)
    return _wrap("\\\\\n".join(rows) + " \\\\", f"Neural-collapse and margin diagnostics of the penultimate layer on the {DATASET_LABEL.get(dataset, dataset)} test set ({frac * 100:.0f}\\% labels).", f"tab:nc_{dataset}_{arch}", "l" + "c" * len(metrics), header, small=True)


def overhead_table(over: pd.DataFrame, vis_grouped: pd.DataFrame) -> str:
    parts = []
    if over is not None and not over.empty:
        g = over.groupby(["arch", "dataset", "config"], dropna=False).agg(step_ms=("step_time_ms", "mean"), step_ms_std=("step_time_ms", "std"), img_s=("images_per_s", "mean"), mem_gb=("peak_gpu_mem_gb", "max"), params=("extra_params", "max")).reset_index()
        rows = []
        for (arch, ds), gg in g.groupby(["arch", "dataset"]):
            base = gg[gg["config"] == "baseline"]
            base_ms = float(base["step_ms"].iloc[0]) if not base.empty else float("nan")
            for _, r in gg.iterrows():
                ov = (r["step_ms"] / base_ms - 1) * 100 if base_ms and not math.isnan(base_ms) else float("nan")
                rows.append(f"{_esc(str(arch))} & {_esc(DATASET_LABEL.get(ds, str(ds)))} & {_esc(str(r['config']))} & {r['step_ms']:.1f} $\\pm$ {0 if math.isnan(r['step_ms_std']) else r['step_ms_std']:.1f} & {r['img_s']:.0f} & {'--' if math.isnan(ov) else f'{ov:+.1f}'} & {r['mem_gb']:.2f} & {int(r['params']) if not math.isnan(r['params']) else 0:,}")
        header = "Arch. & Data & Configuration & ms / step & img / s & overhead (\\%) & peak GPU GB & extra params"
        parts.append(_wrap("\\\\\n".join(rows) + " \\\\", "Training-step cost of monitoring and control (median of timed steps, 3 repeats).", "tab:overhead_steps", "lllccccc", header))
    if vis_grouped is not None and not vis_grouped.empty:
        rows = []
        for (ds, arch), g in vis_grouped[vis_grouped["round"] == -1].groupby(["dataset", "arch"]):
            for m in sorted(g["method"].unique(), key=method_sort_key):
                cells = [_esc(DATASET_LABEL.get(ds, ds)), _esc(method_label(m))]
                for metric in ("total_train_time_s", "total_acquisition_time_s", "aa_total_time_s", "peak_gpu_mem_gb", "peak_host_rss_gb"):
                    mu, sd, _ = _lookup(vis_grouped, ds, arch, m, metric, round_=-1)
                    if metric.endswith("_s"):
                        cells.append("--" if math.isnan(mu) else f"{mu / 60:.1f} $\\pm$ {0 if math.isnan(sd) else sd / 60:.1f}")
                    else:
                        cells.append("--" if math.isnan(mu) else f"{mu:.2f}")
                rows.append(" & ".join(cells))
        header = "Data & Method & train (min) & acquisition (min) & AutoAttack (min) & peak GPU GB & peak host GB"
        parts.append(_wrap("\\\\\n".join(rows) + " \\\\", "End-to-end cost per active-learning run (mean $\\pm$ std over seeds, single GPU).", "tab:overhead_runs", "llccccc", header))
    return "\n".join(parts)


def significance_table(sig: pd.DataFrame, dataset: str, arch: str, metrics: tuple[str, ...] = ("test_acc", "aa_robust_acc", "apgd_ce_subset_acc")) -> str:
    if sig is None or sig.empty:
        return ""
    subset = sig[(sig["dataset"] == dataset) & (sig["arch"] == arch) & (sig["metric"].isin(metrics))]
    if subset.empty:
        return ""

    def num(value: Any, fmt: str = ".3g") -> str:
        try:
            x = float(value)
        except (TypeError, ValueError):
            return "--"
        return format(x, fmt) if math.isfinite(x) else "--"

    rows = []
    for _, row in subset.sort_values(["metric", "target", "reference"]).iterrows():
        scale = 100 if row["metric"] in PERCENT_METRICS else 1
        delta = num(float(row.get("boot_diff_mean", float("nan"))) * scale, "+.2f")
        low = num(float(row.get("boot_ci_low", float("nan"))) * scale, "+.2f")
        high = num(float(row.get("boot_ci_high", float("nan"))) * scale, "+.2f")
        rows.append(
            f"{_esc(METRIC_LABEL.get(row['metric'], row['metric']))} & "
            f"{_esc(method_label(row['target']))} & {_esc(method_label(row['reference']))} & "
            f"{delta} & [{low}, {high}] & {num(row.get('p_ttest'))} & "
            f"{num(row.get('p_ttest_holm'))} & {num(row.get('p_perm'))} & "
            f"{num(row.get('cohens_dz'), '.2f')} & {int(row['n_pairs'])}"
        )
    header = "Metric & Target & Reference & $\\Delta$ & 95\\% bootstrap CI & $p$ ($t$) & $p$ (Holm) & $p$ (perm.) & $d_z$ & $n$"
    note = (
        "Inferential rows are emitted only when the frozen protocol's minimum number of paired "
        "confirmatory seeds is met. Undefined effects or tests are shown as --."
    )
    return _wrap((chr(92) * 2 + "\n").join(rows) + " " + chr(92) * 2, f"Paired seed-level comparisons on {DATASET_LABEL.get(dataset, dataset)} at the final label budget ($\\Delta$ = target $-$ reference, percentage points for accuracies).", f"tab:sig_{dataset}_{arch}", "lllcccccccc", header, note)


def mcnemar_table(mcn: pd.DataFrame, dataset: str, arch: str) -> str:
    if mcn is None or mcn.empty:
        return ""
    s = mcn[(mcn["dataset"] == dataset) & (mcn["arch"] == arch)]
    if s.empty:
        return ""
    g = s.groupby(["metric", "target", "reference"]).agg(acc_diff=("acc_diff", "mean"), n_disc=("n_discordant", "mean"), p_max=("p_value", "max"), p_min=("p_value", "min"), seeds=("seed", "count")).reset_index()
    rows = [f"{_esc(r['metric'])} & {_esc(method_label(r['target']))} & {_esc(method_label(r['reference']))} & {r['acc_diff'] * 100:+.2f} & {r['n_disc']:.0f} & {r['p_min']:.2g} & {r['p_max']:.2g} & {int(r['seeds'])}" for _, r in g.iterrows()]
    header = "Metric & Target & Reference & mean $\\Delta$ (pp) & discordant & min $p$ & max $p$ & seeds"
    return _wrap("\\\\\n".join(rows) + " \\\\", f"Per-example exact McNemar tests between paired final models on {DATASET_LABEL.get(dataset, dataset)} (AutoAttack robust / clean correctness masks, one test per seed).", f"tab:mcnemar_{dataset}_{arch}", "lllccccc", header)


def crossmodal_table(cross_grouped: pd.DataFrame, cross_sig: pd.DataFrame, task: str, metrics: list[tuple[str, str]], caption: str, label: str, dataset: Optional[str] = None) -> str:
    if cross_grouped is None or cross_grouped.empty:
        return ""
    g = cross_grouped[cross_grouped["task"] == task]
    if dataset is not None and "dataset" in g.columns:
        g = g[g["dataset"] == dataset]
        if cross_sig is not None and not cross_sig.empty and "dataset" in cross_sig.columns:
            cross_sig = cross_sig[cross_sig["dataset"] == dataset]
    if g.empty:
        return ""
    methods = sorted(g["method"].unique(), key=method_sort_key)
    rows = []
    for m in methods:
        cells = [_esc(method_label(m))]
        n = 0
        for metric, _ in metrics:
            gm = g[(g["method"] == m) & (g["metric"] == metric)]
            if gm.empty:
                cells.append("--")
                continue
            r = gm.iloc[0]
            n = int(r["n"])
            cell = _fmt(float(r["mean"]), float(r["std"]), metric, digits=(3 if metric in ("validation_nll", "token_ece", "kid", "cider_d", "bleu4", "rouge_l", "meteor", "spice") else 2))
            if m != "baseline" and cross_sig is not None and not cross_sig.empty:
                s = cross_sig[(cross_sig["task"] == task) & (cross_sig["target"] == m) & (cross_sig["reference"] == "baseline") & (cross_sig["metric"] == metric)]
                if not s.empty:
                    cell += significance_marker(float(s.iloc[0]["p_ttest_holm"]))
            cells.append(cell)
        cells.append(str(n))
        rows.append(" & ".join(cells))
    header = "Method & " + " & ".join(h for _, h in metrics) + " & seeds"
    note = ""
    if cross_sig is not None and not cross_sig.empty:
        note = "Stars: paired $t$-test versus the baseline (no DCR), Holm-corrected within each column."
    return _wrap("\\\\\n".join(rows) + " \\\\", caption, label, "l" + "c" * (len(metrics) + 1), header, note)


def eps_sweep_table(grouped: pd.DataFrame, dataset: str, arch: str, protocol: Optional[dict] = None) -> str:
    frac = _final_frac(grouped, dataset, arch)
    g = grouped[(grouped["dataset"] == dataset) & (grouped["arch"] == arch) & grouped["metric"].str.startswith("eps_sweep/")]
    if g.empty:
        return ""
    eps = sorted(g["metric"].unique(), key=lambda s: float(s.split("/")[1]))
    methods = sorted(g["method"].unique(), key=method_sort_key)
    rows = []
    for m in methods:
        cells = [_esc(method_label(m))]
        for e in eps:
            mu, sd, _ = _lookup(grouped, dataset, arch, m, e, label_fraction=frac)
            cells.append(_fmt(mu, sd, "test_acc"))
        mu, sd, _ = _lookup(grouped, dataset, arch, m, "aa_robust_acc", label_fraction=frac)
        cells.append(_fmt(mu, sd, "aa_robust_acc"))
        rows.append(" & ".join(cells))
    header = "Method & " + " & ".join(f"$\\epsilon={float(e.split('/')[1]) * 255:.0f}/255$" for e in eps) + " & AA $8/255$"
    return _wrap("\\\\\n".join(rows) + " \\\\", f"Robust accuracy (\\%) versus perturbation budget on {DATASET_LABEL.get(dataset, dataset)} (APGD-CE on a fixed subset; AutoAttack at $8/255$ on {_aa_scope(protocol, dataset)}).", f"tab:eps_{dataset}_{arch}", "l" + "c" * (len(eps) + 1), header)


def write_all(out: Path, vis: pd.DataFrame, grouped: pd.DataFrame, sig: pd.DataFrame, mcn: pd.DataFrame, cross: pd.DataFrame, cross_grouped: pd.DataFrame, cross_sig: pd.DataFrame, over: pd.DataFrame, protocol: dict[str, Any], report_meta: Optional[dict[str, Any]] = None) -> list[Path]:
    global _ACTIVE_BANNER
    _ACTIVE_BANNER = str((report_meta or {}).get("banner", ""))
    out.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []

    def emit(name: str, content: str):
        if not content or not content.strip():
            return
        p = out / name
        p.write_text(content, encoding="utf-8")
        files.append(p)

    if grouped is not None and not grouped.empty:
        for (ds, arch), _ in grouped.groupby(["dataset", "arch"]):
            emit(f"main_{ds}_{arch}.tex", main_results_table(grouped, sig, ds, arch, protocol=protocol))
            emit(f"curve_clean_{ds}_{arch}.tex", curve_table(grouped, ds, arch, "test_acc"))
            emit(f"curve_apgd_{ds}_{arch}.tex", curve_table(grouped, ds, arch, "apgd_ce_subset_acc"))
            emit(f"curve_pgd_{ds}_{arch}.tex", curve_table(grouped, ds, arch, "pgd_subset_acc"))
            emit(f"ablation_{ds}_{arch}.tex", ablation_table(grouped, sig, ds, arch))
            emit(f"sweep_{ds}_{arch}.tex", sweep_table(grouped, ds, arch))
            emit(f"controller_{ds}_{arch}.tex", controller_table(grouped, ds, arch))
            emit(f"nc_{ds}_{arch}.tex", nc_table(grouped, ds, arch))
            emit(f"significance_{ds}_{arch}.tex", significance_table(sig, ds, arch))
            emit(f"mcnemar_{ds}_{arch}.tex", mcnemar_table(mcn, ds, arch))
            emit(f"eps_sweep_{ds}_{arch}.tex", eps_sweep_table(grouped, ds, arch, protocol=protocol))
    emit("overhead.tex", overhead_table(over, grouped))
    meta = report_meta or {}
    lmeta = meta.get("language", {}) or {}
    dmeta = meta.get("diffusion", {}) or {}
    lmodel = _esc(str(lmeta.get("model") or "language model"))
    ltokens = human_count(lmeta.get("train_tokens"))
    dmodel = _esc(str(dmeta.get("model") or "DDPM"))
    dsamples = human_count(dmeta.get("num_samples"))
    dsteps = dmeta.get("sampling_steps")
    dsampler = _esc(str(dmeta.get("sampler") or "sampler"))
    diffusion_eval = f"{dsamples} generated samples"
    if dsteps is not None:
        diffusion_eval += f", {dsampler}-{int(dsteps)}"
    emit(
        "language_c4.tex",
        crossmodal_table(
            cross_grouped,
            cross_sig,
            "language_c4",
            [("validation_nll", "Val. NLL"), ("perplexity", "PPL"), ("token_ece", "Token ECE"), ("effective_rank/final", "eff. rank (last)")],
            f"C4 language modelling ({lmodel}, {ltokens} training tokens): validation metrics, mean $\\pm$ std over seeds.",
            "tab:c4",
        ),
    )
    emit(
        "diffusion.tex",
        crossmodal_table(
            cross_grouped,
            cross_sig,
            "diffusion_ddpm",
            [("fid", "FID $\\downarrow$"), ("kid", "KID$\\times10^{{3}}$ $\\downarrow$"), ("inception_score", "IS $\\uparrow$"), ("val_loss", "val. $\\epsilon$-MSE")],
            f"CIFAR-10 {dmodel} fine-tuning: generation quality over {diffusion_eval}, mean $\\pm$ std over seeds.",
            "tab:ddpm",
        ),
    )
    audio_sets = sorted(cross_grouped[cross_grouped["task"].isin(["audio_captioning", "audio_retrieval"])]["dataset"].dropna().unique().tolist()) if (cross_grouped is not None and not cross_grouped.empty and "dataset" in cross_grouped.columns) else []
    for ds in audio_sets or [None]:
        dl = AUDIO_DATASET_LABEL.get(ds, str(ds)) if ds else "Audio"
        role = " (supplementary corpus)" if ds == "clotho" else ""
        suffix = f"_{ds}" if ds else ""
        emit(f"audio_captioning{suffix}.tex", crossmodal_table(cross_grouped, cross_sig, "audio_captioning", [("cider_d", "CIDEr-D"), ("bleu4", "BLEU-4"), ("rouge_l", "ROUGE-L"), ("meteor", "METEOR"), ("spice", "SPICE"), ("val_loss", "val. loss")], f"{dl} audio captioning{role} (AST features + GPT-2 prefix decoder), evaluation split, mean $\\pm$ std over seeds.", f"tab:captioning{suffix}", dataset=ds))
        emit(f"audio_retrieval{suffix}.tex", crossmodal_table(cross_grouped, cross_sig, "audio_retrieval", [("t2a_r1", "T$\\to$A R@1"), ("t2a_r5", "R@5"), ("t2a_r10", "R@10"), ("a2t_r1", "A$\\to$T R@1"), ("a2t_r5", "R@5"), ("a2t_r10", "R@10"), ("median_rank", "med. rank")], f"{dl} audio--text retrieval{role} (recall in \\%), evaluation split, mean $\\pm$ std over seeds.", f"tab:retrieval{suffix}", dataset=ds))
    return files
