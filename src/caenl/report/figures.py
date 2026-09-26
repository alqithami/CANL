"""Publication figures (PDF + PNG) from aggregated CSVs; colours bound to methods, CI bands over seeds."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from .style import DATASET_LABEL, LAYER_STYLE, METRIC_LABEL, PERCENT_METRICS, method_label, method_sort_key, method_style, setup_matplotlib

KEY_METHODS = ["random", "entropy", "margin", "badge", "coreset", "caenl_acq_only", "caenl_macc_lite", "caenl_full_macc"]


_ACTIVE_BANNER = ""


def _save(fig, out: Path, name: str, files: list[Path]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if _ACTIVE_BANNER:
        fig.text(0.5, 0.995, _ACTIVE_BANNER, ha="center", va="top", fontsize=8, fontweight="bold")
    for ext in ("pdf", "png"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        if ext == "pdf":
            files.append(p)
    import matplotlib.pyplot as plt

    plt.close(fig)


def _scale(metric: str) -> float:
    return 100.0 if metric in PERCENT_METRICS else 1.0


def _series(grouped: pd.DataFrame, dataset: str, arch: str, method: str, metric: str) -> pd.DataFrame:
    g = grouped[(grouped["dataset"] == dataset) & (grouped["arch"] == arch) & (grouped["method"] == method) & (grouped["metric"] == metric) & (grouped["round"] >= 0)]
    return g.sort_values("label_fraction")


def learning_curves(plt, grouped: pd.DataFrame, dataset: str, arch: str, methods: list[str], metrics: list[str], name: str, out: Path, files: list[Path]) -> None:
    avail = [m for m in methods if not _series(grouped, dataset, arch, m, metrics[0]).empty]
    if not avail:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 2.5), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        sc = _scale(metric)
        for m in avail:
            s = _series(grouped, dataset, arch, m, metric)
            if s.empty:
                continue
            st = method_style(m)
            if m == "supervised_full":
                ax.axhline(float(s["mean"].iloc[-1]) * sc, color=st["color"], ls=st["ls"], lw=1.0, label=st["label"])
                continue
            x = s["label_fraction"].astype(float) * 100
            y = s["mean"].astype(float) * sc
            lw = 2.0 if m == "caenl_macc_lite" else 1.3
            ax.plot(x, y, color=st["color"], ls=st["ls"], lw=lw, marker="o", ms=3, label=st["label"])
            lo, hi = s["ci_low"].astype(float) * sc, s["ci_high"].astype(float) * sc
            if np.isfinite(lo).all():
                ax.fill_between(x, lo, hi, color=st["color"], alpha=0.12, lw=0)
        ax.set_xlabel("Labeled fraction (%)")
        ax.set_ylabel(METRIC_LABEL.get(metric, metric))
        ax.set_title(f"{DATASET_LABEL.get(dataset, dataset)}")
    axes[0][-1].legend(loc="best", fontsize=6, ncol=1)
    _save(fig, out, name, files)


def pareto(plt, vis: pd.DataFrame, grouped: pd.DataFrame, dataset: str, arch: str, out: Path, files: list[Path]) -> None:
    g = grouped[(grouped["dataset"] == dataset) & (grouped["arch"] == arch) & (grouped["round"] >= 0) & (grouped["method"] != "supervised_full")]
    if g.empty:
        return
    frac = float(g["label_fraction"].max())
    robust_metric = "aa_robust_acc" if not g[g["metric"] == "aa_robust_acc"].empty else "apgd_ce_subset_acc"
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    final = vis[(vis["dataset"] == dataset) & (vis["arch"] == arch) & np.isclose(vis["label_fraction"].astype(float), frac, atol=1e-6)]
    sweep_pts = []
    for m in sorted(final["method"].unique(), key=method_sort_key):
        fx = final[(final["method"] == m) & (final["metric"] == "test_acc")].set_index("seed")["value"]
        fy = final[(final["method"] == m) & (final["metric"] == robust_metric)].set_index("seed")["value"]
        seeds = sorted(set(fx.index) & set(fy.index))
        if not seeds:
            continue
        st = method_style(m)
        ax.scatter(fx.loc[seeds] * 100, fy.loc[seeds] * 100, s=9, color=st["color"], alpha=0.45, lw=0)
        mx, my = float(fx.loc[seeds].mean()) * 100, float(fy.loc[seeds].mean()) * 100
        ex = float(fx.loc[seeds].std(ddof=1)) * 100 if len(seeds) > 1 else 0
        ey = float(fy.loc[seeds].std(ddof=1)) * 100 if len(seeds) > 1 else 0
        marker = "s" if m.startswith("caenl") or m.startswith("dcr") else "o"
        ax.errorbar(mx, my, xerr=ex, yerr=ey, fmt=marker, color=st["color"], ms=6, mec="white", mew=0.8, elinewidth=0.8, capsize=2, label=st["label"])
        if m == "caenl_acq_only" or m.startswith("caenl_fixed_dcr"):
            lam = 0.0 if m == "caenl_acq_only" else (0.03 if m == "caenl_fixed_dcr" else float(m.split("_l")[-1]))
            sweep_pts.append((lam, mx, my))
    if len(sweep_pts) >= 2:
        sweep_pts.sort()
        ax.plot([p[1] for p in sweep_pts], [p[2] for p in sweep_pts], color=method_style("caenl_fixed_dcr")["color"], ls=":", lw=1, zorder=0)
        for lam, mx, my in sweep_pts:
            ax.annotate(f"$\\lambda$={lam:g}", (mx, my), textcoords="offset points", xytext=(4, 3), fontsize=6, color="#52514e")
    ax.set_xlabel("Clean accuracy (%)")
    ax.set_ylabel(METRIC_LABEL.get(robust_metric, robust_metric))
    ax.set_title(f"{DATASET_LABEL.get(dataset, dataset)}, {frac * 100:.0f}% labels")
    ax.legend(fontsize=5.5, loc="best", ncol=1)
    _save(fig, out, f"pareto_{dataset}_{arch}", files)


def collapse_dynamics(plt, curves: pd.DataFrame, dataset: str, arch: str, out: Path, files: list[Path]) -> None:
    c = curves[(curves["dataset"] == dataset) & (curves["arch"] == arch)] if not curves.empty else curves
    if c is None or c.empty:
        return
    layers = [col.split("/")[1] for col in c.columns if col.startswith("strength/")]
    if not layers:
        return
    methods = [m for m in ("random", "caenl_acq_only", "caenl_fixed_dcr", "caenl_macc_lite", "caenl_full_macc") if m in set(c["method"])]
    fig, axes = plt.subplots(2, len(layers), figsize=(3.2 * len(layers), 4.0), squeeze=False, sharex=True)
    for j, layer in enumerate(layers):
        for m in methods:
            cm = c[c["method"] == m]
            if cm.empty:
                continue
            st = method_style(m)
            # bin by step and average over seeds
            bins = pd.cut(cm["step"], bins=min(80, max(5, cm["step"].nunique())))
            agg = cm.groupby(bins, observed=True).agg(step=("step", "mean"), s=(f"strength/{layer}", "mean"), lam=(f"lambda/{layer}", "mean")).dropna(subset=["step"])
            axes[0][j].plot(agg["step"], agg["s"], color=st["color"], ls=st["ls"], lw=1.2, label=st["label"])
            axes[1][j].plot(agg["step"], agg["lam"], color=st["color"], ls=st["ls"], lw=1.2)
        axes[0][j].set_title(f"{layer}")
        axes[0][j].set_ylabel("Collapse strength $1/(1+\\kappa)$")
        axes[1][j].set_ylabel("$\\lambda_l$")
        axes[1][j].set_xlabel("Training step")
    axes[0][0].legend(fontsize=6)
    fig.suptitle(f"{DATASET_LABEL.get(dataset, dataset)}: collapse dynamics and regularisation strength", fontsize=9)
    _save(fig, out, f"collapse_dynamics_{dataset}_{arch}", files)


def controller_figure(plt, traj: pd.DataFrame, dataset: str, arch: str, out: Path, files: list[Path]) -> None:
    if traj is None or traj.empty:
        return
    t = traj[(traj["dataset"] == dataset) & (traj["arch"] == arch)]
    t = t[t["method"].isin(["caenl_macc_lite", "caenl_full_macc"])]
    if t.empty:
        return
    lam_cols = [c for c in t.columns if c.startswith("lambda/")]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.5))
    for m, ax in zip(["caenl_macc_lite", "caenl_full_macc"], axes[:2]):
        tm = t[t["method"] == m]
        if tm.empty:
            ax.set_visible(False)
            continue
        for col in lam_cols:
            layer = col.split("/")[1]
            agg = tm.groupby("i")[col].agg(["mean", "std"]).reset_index()
            ax.plot(agg["i"], agg["mean"], color=LAYER_STYLE.get(layer, "#888"), lw=1.3, label=f"$\\lambda$ {layer}")
            ax.fill_between(agg["i"], agg["mean"] - agg["std"].fillna(0), agg["mean"] + agg["std"].fillna(0), color=LAYER_STYLE.get(layer, "#888"), alpha=0.15, lw=0)
        ax.set_title(method_label(m))
        ax.set_xlabel("Controller step")
        ax.set_ylabel("$\\lambda_l$")
        ax.legend(fontsize=6)
    tf = t[t["method"] == "caenl_full_macc"]
    if not tf.empty and "tau" in tf:
        agg = tf.groupby("i")["tau"].agg(["mean", "std"]).reset_index()
        axes[2].plot(agg["i"], agg["mean"], color=method_style("caenl_full_macc")["color"], lw=1.3)
        axes[2].fill_between(agg["i"], agg["mean"] - agg["std"].fillna(0), agg["mean"] + agg["std"].fillna(0), color=method_style("caenl_full_macc")["color"], alpha=0.15, lw=0)
        axes[2].set_title("Full MACC query threshold")
        axes[2].set_xlabel("Controller step")
        axes[2].set_ylabel("$\\tau$")
    else:
        axes[2].set_visible(False)
    _save(fig, out, f"controller_{dataset}_{arch}", files)


def nc_bars(plt, grouped: pd.DataFrame, dataset: str, arch: str, out: Path, files: list[Path]) -> None:
    g = grouped[(grouped["dataset"] == dataset) & (grouped["arch"] == arch) & (grouped["round"] >= 0) & (grouped["method"] != "supervised_full")]
    if g.empty:
        return
    frac = float(g["label_fraction"].max())
    gf = g[np.isclose(g["label_fraction"].astype(float), frac, atol=1e-6)]
    metrics = [("nc/layer4/kappa", "$\\kappa$ (NC1, lower = more collapsed)"), ("nc/layer4/nc2_etf_distance", "ETF distance (NC2)"), ("nc/layer4/total_effective_rank", "Effective rank"), ("margins/grad_norm_median", "Median $\\|\\nabla_x m\\|_2$")]
    metrics = [(k, l) for k, l in metrics if not gf[gf["metric"] == k].empty]
    if not metrics:
        return
    methods = sorted(gf["method"].unique(), key=method_sort_key)
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.6 * len(metrics), 2.6), squeeze=False)
    for ax, (metric, label) in zip(axes[0], metrics):
        ys, es, cs, names = [], [], [], []
        for m in methods:
            r = gf[(gf["method"] == m) & (gf["metric"] == metric)]
            if r.empty:
                continue
            ys.append(float(r.iloc[0]["mean"]))
            es.append(0 if math.isnan(float(r.iloc[0]["std"])) else float(r.iloc[0]["std"]))
            cs.append(method_style(m)["color"])
            names.append(method_label(m))
        x = np.arange(len(ys))
        ax.bar(x, ys, yerr=es, color=cs, width=0.7, capsize=2, error_kw={"lw": 0.7})
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=5.5)
        ax.set_title(label, fontsize=7)
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"{DATASET_LABEL.get(dataset, dataset)} penultimate-layer geometry at {frac * 100:.0f}% labels", fontsize=8)
    _save(fig, out, f"nc_{dataset}_{arch}", files)


def eps_sweep(plt, grouped: pd.DataFrame, dataset: str, arch: str, out: Path, files: list[Path]) -> None:
    g = grouped[(grouped["dataset"] == dataset) & (grouped["arch"] == arch) & (grouped["round"] >= 0) & (grouped["method"] != "supervised_full")]
    sw = g[g["metric"].str.startswith("eps_sweep/")]
    if sw.empty:
        return
    frac = float(g["label_fraction"].max())
    sw = sw[np.isclose(sw["label_fraction"].astype(float), frac, atol=1e-6)]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for m in sorted(sw["method"].unique(), key=method_sort_key):
        sm = sw[sw["method"] == m].copy()
        sm["eps"] = sm["metric"].str.split("/").str[1].astype(float) * 255
        sm = sm.sort_values("eps")
        st = method_style(m)
        ax.errorbar(sm["eps"], sm["mean"] * 100, yerr=sm["std"].fillna(0) * 100, color=st["color"], ls=st["ls"], marker="o", ms=3, lw=1.2, capsize=2, label=st["label"])
    ax.set_xlabel("$\\epsilon$ ($\\times$1/255)")
    ax.set_ylabel("APGD-CE robust accuracy (%)")
    ax.set_title(f"{DATASET_LABEL.get(dataset, dataset)}, {frac * 100:.0f}% labels")
    ax.legend(fontsize=5.5)
    _save(fig, out, f"eps_sweep_{dataset}_{arch}", files)


def selection_entropy(plt, grouped: pd.DataFrame, dataset: str, arch: str, out: Path, files: list[Path]) -> None:
    g = grouped[(grouped["dataset"] == dataset) & (grouped["arch"] == arch) & (grouped["metric"] == "selection_class_entropy_norm")]
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for m in sorted(g["method"].unique(), key=method_sort_key):
        s = g[g["method"] == m].sort_values("round")
        st = method_style(m)
        ax.plot(s["round"], s["mean"], color=st["color"], ls=st["ls"], marker="o", ms=3, lw=1.2, label=st["label"])
    ax.set_xlabel("Acquisition round")
    ax.set_ylabel("Class entropy of selection (norm.)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{DATASET_LABEL.get(dataset, dataset)}: class balance of queried batches")
    ax.legend(fontsize=5.5, ncol=2)
    _save(fig, out, f"selection_entropy_{dataset}_{arch}", files)


def score_correlation(plt, grouped: pd.DataFrame, dataset: str, arch: str, out: Path, files: list[Path]) -> None:
    g = grouped[(grouped["dataset"] == dataset) & (grouped["arch"] == arch) & grouped["metric"].str.startswith("score_spearman/")]
    if g.empty:
        return
    pairs = [p for p in g["metric"].unique() if "collapse" in p]
    if not pairs:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    colors = {"entropy": "#eb6834", "margin": "#eda100"}
    for pair in sorted(pairs):
        other = pair.split("/")[1].replace("collapse|", "").replace("|collapse", "")
        s = g[(g["metric"] == pair) & (g["method"] == "caenl_macc_lite")].sort_values("round")
        if s.empty:
            s = g[g["metric"] == pair].groupby("round").agg(mean=("mean", "mean"), std=("std", "mean")).reset_index()
        ax.errorbar(s["round"], s["mean"], yerr=s["std"].fillna(0), color=colors.get(other, "#7a7975"), marker="o", ms=3, lw=1.2, capsize=2, label=f"collapse score vs {other}")
    ax.axhline(0, color="#9a9995", lw=0.6)
    ax.set_xlabel("Acquisition round")
    ax.set_ylabel("Spearman $\\rho$ (candidates)")
    ax.set_ylim(-1, 1)
    ax.set_title(f"{DATASET_LABEL.get(dataset, dataset)}: collapse score vs. output uncertainty")
    ax.legend(fontsize=6)
    _save(fig, out, f"score_correlation_{dataset}_{arch}", files)


def overhead_figure(plt, over: pd.DataFrame, out: Path, files: list[Path]) -> None:
    if over is None or over.empty:
        return
    g = over.groupby(["arch", "config"]).agg(step_ms=("step_time_ms", "mean"), sd=("step_time_ms", "std")).reset_index()
    archs = sorted(g["arch"].unique())
    fig, axes = plt.subplots(1, len(archs), figsize=(3.0 * len(archs), 2.5), squeeze=False)
    colors = {"baseline": "#7a7975", "monitoring": "#eda100", "fixed_dcr": "#eb6834", "macc_lite": "#2a78d6", "full_macc": "#4a3aa7", "at_pgd10": "#e34948"}
    for ax, arch in zip(axes[0], archs):
        ga = g[g["arch"] == arch]
        base = float(ga[ga["config"] == "baseline"]["step_ms"].iloc[0]) if not ga[ga["config"] == "baseline"].empty else float("nan")
        x = np.arange(len(ga))
        ax.bar(x, ga["step_ms"], yerr=ga["sd"].fillna(0), color=[colors.get(c, "#999") for c in ga["config"]], width=0.7, capsize=2)
        for xi, (_, r) in zip(x, ga.iterrows()):
            if base and not math.isnan(base):
                ax.annotate(f"{(r['step_ms'] / base - 1) * 100:+.0f}%", (xi, r["step_ms"]), ha="center", va="bottom", fontsize=6, xytext=(0, 2), textcoords="offset points")
        ax.set_xticks(x)
        ax.set_xticklabels(ga["config"], rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("ms / training step")
        ax.set_title(arch)
        ax.grid(axis="x", visible=False)
    _save(fig, out, "overhead_steps", files)


def crossmodal_figure(plt, cross_grouped: pd.DataFrame, task: str, metric: str, ylabel: str, out: Path, files: list[Path]) -> None:
    """One bar chart per (task, dataset): audio tasks may carry AudioCaps and the supplementary Clotho corpus."""
    if cross_grouped is None or cross_grouped.empty:
        return
    g_all = cross_grouped[(cross_grouped["task"] == task) & (cross_grouped["metric"] == metric)]
    if g_all.empty:
        return
    datasets = sorted(g_all["dataset"].dropna().unique().tolist()) if "dataset" in g_all.columns else [None]
    for ds in datasets or [None]:
        g = g_all if ds is None else g_all[g_all["dataset"] == ds]
        if g.empty:
            continue
        _crossmodal_bars(plt, g, task, metric, ylabel, out, files, suffix=("" if ds is None or len(datasets) <= 1 else f"_{ds}"))


def _crossmodal_bars(plt, g: pd.DataFrame, task: str, metric: str, ylabel: str, out: Path, files: list[Path], suffix: str = "") -> None:
    methods = sorted(g["method"].unique(), key=method_sort_key)
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    x = np.arange(len(methods))
    sc = _scale(metric)
    ys = [float(g[g["method"] == m]["mean"].iloc[0]) * sc for m in methods]
    es = [0 if math.isnan(float(g[g["method"] == m]["std"].iloc[0])) else float(g[g["method"] == m]["std"].iloc[0]) * sc for m in methods]
    ax.bar(x, ys, yerr=es, color=[method_style(m)["color"] for m in methods], width=0.7, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([method_label(m) for m in methods], rotation=30, ha="right", fontsize=6)
    ax.set_ylabel(ylabel)
    lo = min(ys) - 3 * (max(es) if es else 0)
    hi = max(ys) + 3 * (max(es) if es else 0)
    if hi > lo:
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    ax.grid(axis="x", visible=False)
    _save(fig, out, f"{task}_{metric}{suffix}", files)


def write_all(out: Path, vis: pd.DataFrame, grouped: pd.DataFrame, cross: pd.DataFrame, cross_grouped: pd.DataFrame, over: pd.DataFrame, traj: pd.DataFrame, curves: pd.DataFrame, protocol: dict[str, Any], report_meta: Optional[dict[str, Any]] = None) -> list[Path]:
    global _ACTIVE_BANNER
    _ACTIVE_BANNER = str((report_meta or {}).get("banner", ""))
    plt = setup_matplotlib()
    files: list[Path] = []
    if grouped is not None and not grouped.empty:
        for (ds, arch), g in grouped.groupby(["dataset", "arch"]):
            all_methods = sorted(g["method"].unique(), key=method_sort_key)
            key = [m for m in KEY_METHODS if m in all_methods]
            try:
                learning_curves(plt, grouped, ds, arch, key, ["test_acc", "apgd_ce_subset_acc"], f"learning_curves_{ds}_{arch}", out, files)
                learning_curves(plt, grouped, ds, arch, all_methods, ["test_acc", "apgd_ce_subset_acc", "pgd_subset_acc"], f"learning_curves_all_{ds}_{arch}", out, files)
                pareto(plt, vis, grouped, ds, arch, out, files)
                collapse_dynamics(plt, curves, ds, arch, out, files)
                controller_figure(plt, traj, ds, arch, out, files)
                nc_bars(plt, grouped, ds, arch, out, files)
                eps_sweep(plt, grouped, ds, arch, out, files)
                selection_entropy(plt, grouped, ds, arch, out, files)
                score_correlation(plt, grouped, ds, arch, out, files)
            except Exception as exc:  # never let a figure kill the report
                print(f"[report] figure generation failed for {ds}/{arch}: {exc!r}", flush=True)
    try:
        overhead_figure(plt, over, out, files)
    except Exception as exc:
        print(f"[report] overhead figure failed: {exc!r}", flush=True)
    dmeta = (report_meta or {}).get("diffusion", {}) or {}
    n_samples = dmeta.get("num_samples")
    sampling_steps = dmeta.get("sampling_steps")
    sampler = str(dmeta.get("sampler") or "sampler").upper()
    fid_label = "FID"
    if n_samples is not None or sampling_steps is not None:
        parts = []
        if n_samples is not None:
            parts.append(f"{int(n_samples):,} samples")
        if sampling_steps is not None:
            parts.append(f"{sampler}-{int(sampling_steps)}")
        fid_label += " (" + ", ".join(parts) + ")"
    for task, metric, label in [("language_c4", "validation_nll", "Validation NLL (nats/token)"), ("language_c4", "perplexity", "Perplexity"), ("diffusion_ddpm", "fid", fid_label), ("audio_captioning", "cider_d", "CIDEr-D"), ("audio_retrieval", "t2a_r1", "Text-to-audio R@1 (%)")]:
        try:
            crossmodal_figure(plt, cross_grouped, task, metric, label, out, files)
        except Exception as exc:
            print(f"[report] cross-modal figure failed for {task}/{metric}: {exc!r}", flush=True)
    return files
