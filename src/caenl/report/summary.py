"""Markdown results summary: the numbers the revised manuscript should cite, with provenance."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.io import utc_now
from .style import DATASET_LABEL, METRIC_LABEL, PERCENT_METRICS, method_label, method_sort_key
from .context import campaign_role, report_dirname, role_banner


def _f(v: float, metric: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    sc = 100 if metric in PERCENT_METRICS else 1
    d = 2 if metric in PERCENT_METRICS else 4
    return f"{v * sc:.{d}f}"


def write_summary(path: Path, root: Path, jobs: list[dict[str, Any]], grouped: pd.DataFrame, sig: pd.DataFrame, mcn: pd.DataFrame, cross_grouped: pd.DataFrame, cross_sig: pd.DataFrame, over: pd.DataFrame, table_files: list[Path], figure_files: list[Path], protocol: dict[str, Any]) -> None:
    role = campaign_role(root)
    banner = role_banner(role)
    report_dir = report_dirname(role)
    lines: list[str] = [f"# CAENL revision campaign — results summary", ""]
    if banner:
        lines.extend([f"> **{banner}**", ""])
    lines.extend([f"Generated {utc_now()} from `{root}` ({len(jobs)} successfully completed analysis jobs; campaign role: `{role}`).", "", f"All values are means over included seeds (± std); see `aggregate/*.csv` for per-seed records and `{report_dir}/tables/*.tex` for typeset tables.", ""])
    tasks = {}
    for j in jobs:
        if j.get("task") in ("aggregate", "prepare_data"):
            continue
        tasks[j.get("task", "?")] = tasks.get(j.get("task", "?"), 0) + 1
    lines.append("## Job inventory")
    lines.append("")
    for t, n in sorted(tasks.items()):
        lines.append(f"- `{t}`: {n} job(s)")
    lines.append("")
    sanity = [j for j in jobs if j.get("task") == "vision_aa_sanity"]
    if sanity:
        lines.append("## AutoAttack pipeline sanity check")
        lines.append("")
        lines.append("Reference models evaluated under the frozen robustness protocol (same subset, attack version, eps and seed as every CAENL run).")
        lines.append("")
        lines.append("| Dataset | Arch | Weights | Clean acc (full test) | AA clean acc | AA robust acc | n | Check |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for j in sanity:
            aa = j.get("aa", {}) or {}
            ok = j.get("sanity_ok")
            lines.append(f"| {j.get('dataset')} | {j.get('arch')} | {j.get('weights') or 'random init'} | {100 * float(j.get('clean_acc_full_test', float('nan'))):.2f} | {100 * float(aa.get('clean_acc', float('nan'))):.2f} | {100 * float(aa.get('robust_acc', float('nan'))):.2f} | {aa.get('n', '--')} | {'passed' if ok else ('FAILED' if ok is False else 'n/a')} |")
        lines.append("")
    if grouped is not None and not grouped.empty:
        lines.append("## Vision active learning (final label budget)")
        lines.append("")
        for (ds, arch), g in grouped.groupby(["dataset", "arch"]):
            gg = g[g["round"] >= 0]
            if gg.empty:
                continue
            al = gg[gg["method"] != "supervised_full"]
            frac = float((al if not al.empty else gg)["label_fraction"].max())
            gf = gg[np.isclose(gg["label_fraction"].astype(float), frac, atol=1e-6) | (gg["method"] == "supervised_full")]
            lines.append(f"### {DATASET_LABEL.get(ds, ds)} / {arch} at {frac * 100:.0f}% labels")
            lines.append("")
            cols = ["test_acc", "aa_robust_acc", "apgd_ce_subset_acc", "pgd_subset_acc", "corruption_mean_acc", "test_ece"]
            lines.append("| Method | seeds | " + " | ".join(METRIC_LABEL.get(c, c) for c in cols) + " |")
            lines.append("|---|---|" + "---|" * len(cols))
            for m in sorted(gf["method"].unique(), key=method_sort_key):
                cells = []
                n = 0
                for c in cols:
                    r = gf[(gf["method"] == m) & (gf["metric"] == c)]
                    if r.empty:
                        cells.append("--")
                        continue
                    n = int(r.iloc[0]["n"])
                    cells.append(f"{_f(float(r.iloc[0]['mean']), c)} ± {_f(float(r.iloc[0]['std']), c)}")
                lines.append(f"| {method_label(m)} | {n} | " + " | ".join(cells) + " |")
            lines.append("")
            if sig is not None and not sig.empty:
                s = sig[(sig["dataset"] == ds) & (sig["arch"] == arch) & (sig["target"] == "caenl_macc_lite")]
                if not s.empty:
                    lines.append("Paired comparisons, CAENL (MACC-Lite) minus reference (percentage points for accuracies; paired t-test, Holm-corrected per metric):")
                    lines.append("")
                    lines.append("| Metric | Reference | Δ | 95% CI | p (t) | p (Holm) | d_z | wins/losses |")
                    lines.append("|---|---|---|---|---|---|---|---|")
                    for _, r in s.sort_values(["metric", "reference"]).iterrows():
                        sc = 100 if r["metric"] in PERCENT_METRICS else 1
                        def safe_num(value: Any, fmt: str) -> str:
                            try:
                                x = float(value)
                            except (TypeError, ValueError):
                                return "--"
                            return format(x, fmt) if math.isfinite(x) else "--"
                        lines.append(f"| {METRIC_LABEL.get(r['metric'], r['metric'])} | {method_label(r['reference'])} | {safe_num(r.get('boot_diff_mean') * sc, '+.2f')} | [{safe_num(r.get('boot_ci_low') * sc, '+.2f')}, {safe_num(r.get('boot_ci_high') * sc, '+.2f')}] | {safe_num(r.get('p_ttest'), '.3g')} | {safe_num(r.get('p_ttest_holm'), '.3g')} | {safe_num(r.get('cohens_dz'), '.2f')} | {int(r['wins'])}/{int(r['losses'])} |")
                    lines.append("")
            if mcn is not None and not mcn.empty:
                s = mcn[(mcn["dataset"] == ds) & (mcn["arch"] == arch) & (mcn["target"] == "caenl_macc_lite") & (mcn["metric"] == "aa_robust_acc")]
                if not s.empty:
                    agg = s.groupby("reference").agg(d=("acc_diff", "mean"), pmax=("p_value", "max"), n=("seed", "count")).reset_index()
                    lines.append("Per-example McNemar (AutoAttack robust masks), CAENL (MACC-Lite) vs reference: " + "; ".join(f"{method_label(r['reference'])}: Δ={r['d'] * 100:+.2f} pp, max p over {int(r['n'])} seeds = {r['pmax']:.2g}" for _, r in agg.iterrows()))
                    lines.append("")
    if cross_grouped is not None and not cross_grouped.empty:
        lines.append("## Cross-modal extensions")
        lines.append("")
        keys = ["task", "dataset"] if "dataset" in cross_grouped.columns else ["task"]
        for key, g in cross_grouped.groupby(keys):
            task, ds = (key if isinstance(key, tuple) else (key, None))[:2] if len(keys) == 2 else (key if not isinstance(key, tuple) else key[0], None)
            lines.append(f"### {task}" + (f" — {ds}" if ds else ""))
            lines.append("")
            metrics = [m for m in g["metric"].unique() if not m.endswith(("_s", "_gb", "params"))]
            lines.append("| Method | seeds | " + " | ".join(METRIC_LABEL.get(m, m) for m in metrics) + " |")
            lines.append("|---|---|" + "---|" * len(metrics))
            for m in sorted(g["method"].unique(), key=method_sort_key):
                cells, n = [], 0
                for met in metrics:
                    r = g[(g["method"] == m) & (g["metric"] == met)]
                    if r.empty:
                        cells.append("--")
                        continue
                    n = int(r.iloc[0]["n"])
                    cells.append(f"{_f(float(r.iloc[0]['mean']), met)} ± {_f(float(r.iloc[0]['std']), met)}")
                lines.append(f"| {method_label(m)} | {n} | " + " | ".join(cells) + " |")
            lines.append("")
    if over is not None and not over.empty:
        lines.append("## Overhead benchmark")
        lines.append("")
        g = over.groupby(["arch", "dataset", "config"]).agg(ms=("step_time_ms", "mean"), mem=("peak_gpu_mem_gb", "max")).reset_index()
        lines.append("| Arch | Data | Config | ms/step | peak GPU GB |")
        lines.append("|---|---|---|---|---|")
        for _, r in g.iterrows():
            lines.append(f"| {r['arch']} | {r['dataset']} | {r['config']} | {r['ms']:.1f} | {r['mem']:.2f} |")
        lines.append("")
    lines.append("## Generated tables")
    lines.append("")
    lines.extend(f"- `{p.name}`" for p in table_files)
    lines.append("")
    lines.append("## Generated figures")
    lines.append("")
    lines.extend(f"- `{p.name}`" for p in figure_files)
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
