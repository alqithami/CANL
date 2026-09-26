"""Aggregate recorded job outputs into CSVs, statistics, LaTeX tables, figures and a summary.

Only ``summary.json`` files of *succeeded* jobs (plus their artifacts) are read.  Nothing is
ever transcribed by hand: every number in ``manuscript/tables`` and ``manuscript/figures``
is traceable to a job directory listed in ``aggregate/all_jobs.csv``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..core.stats import compare_paired, holm_bonferroni, mcnemar_test, summarize
from ..utils.io import atomic_write_json, read_json, utc_now
from .context import campaign_role, derive_report_meta, report_dirname


# ----------------------------------------------------------------------------- collection
def _infer_analysis_role(stage_name: str, cfg: dict[str, Any], summary: dict[str, Any]) -> tuple[str, bool, bool]:
    """Return ``(role, include_in_science, include_in_inference)`` for one job.

    Old campaigns may lack explicit role fields.  Resume/fault/gate/smoke stages are
    conservatively treated as engineering evidence so they cannot contaminate scientific
    seed counts.
    """
    role = str(cfg.get("analysis_role") or summary.get("analysis_role") or cfg.get("campaign_role") or "").lower()
    name = stage_name.lower()
    engineering_name = any(tok in name for tok in ("resume", "fault", "gate_probe", "smoke_probe"))
    if not role:
        role = "engineering" if engineering_name else "confirmatory"
    if engineering_name:
        role = "engineering"
    include_science = bool(cfg.get("include_in_science", role in {"pilot", "confirmatory"})) and not engineering_name
    include_inference = bool(cfg.get("include_in_inference", role == "confirmatory")) and include_science
    return role, include_science, include_inference


def collect_jobs(root: Path, include_engineering: bool = False) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    stages = root / "stages"
    if not stages.exists():
        return jobs
    for sd in sorted(stages.iterdir()):
        jdir = sd / "jobs"
        if not jdir.exists():
            continue
        for jd in sorted(jdir.iterdir()):
            st = jd / "status.json"
            su = jd / "summary.json"
            jf = jd / "job.json"
            if not (st.exists() and su.exists()):
                continue
            try:
                if read_json(st).get("state") != "succeeded":
                    continue
                s = read_json(su)
                cfg = read_json(jf) if jf.exists() else dict(s.get("config") or {})
            except Exception:
                continue
            if s.get("fault") is not None:
                continue
            role, include_science, include_inference = _infer_analysis_role(sd.name, cfg, s)
            if not include_engineering and not include_science:
                continue
            s["_job_dir"] = str(jd)
            s["_stage"] = sd.name
            s["_config"] = cfg
            s["_analysis_role"] = role
            s["_include_in_science"] = include_science
            s["_include_in_inference"] = include_inference
            jobs.append(s)
    return jobs


def _flatten_nc(nc: Optional[dict], prefix: str = "nc") -> dict[str, float]:
    out: dict[str, float] = {}
    if not nc:
        return out
    for layer, metrics in nc.items():
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[f"{prefix}/{layer}/{k}"] = float(v)
    return out


def vision_long(jobs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for s in jobs:
        if s.get("task") != "vision_al" or s.get("mode") == "initial_only":
            continue
        base = {
            "stage": s["_stage"],
            "dataset": s["dataset"],
            "arch": s["arch"],
            "method": s["method"],
            "acquisition": s.get("acquisition"),
            "regularizer": s.get("regularizer"),
            "dcr_lambda": s.get("dcr_lambda"),
            "seed": int(s["seed"]),
            "job_dir": s["_job_dir"],
            "analysis_role": s.get("_analysis_role", "confirmatory"),
            "inference_eligible": bool(s.get("_include_in_inference", True)),
        }
        n_rounds = len(s.get("rounds", []))
        for i, r in enumerate(s.get("rounds", [])):
            rec = dict(base)
            rec.update({"round": int(r["round"]), "label_fraction": float(r["label_fraction"]), "is_final": i == n_rounds - 1})
            metrics: dict[str, Any] = {
                "test_acc": r.get("test_acc"),
                "test_top5": r.get("test_top5"),
                "test_loss": r.get("test_loss"),
                "test_ece": r.get("test_ece"),
                "ctrl_val_acc": r.get("ctrl_val_acc"),
                "apgd_ce_subset_acc": (r.get("apgd_ce_subset") or {}).get("robust_acc"),
                "pgd_subset_acc": (r.get("pgd_subset") or {}).get("robust_acc"),
                "aa_robust_acc": (r.get("autoattack") or {}).get("robust_acc"),
                "aa_clean_acc": (r.get("autoattack") or {}).get("clean_acc"),
                "aa_time_s": (r.get("autoattack") or {}).get("time_s"),
                "corruption_mean_acc": (r.get("corruptions") or {}).get("corruption_mean_acc"),
                "phase_time_s": (r.get("phase") or {}).get("phase_time_s"),
                "images_per_s": (r.get("phase") or {}).get("images_per_s"),
                "acquisition_time_s": (r.get("acquisition") or {}).get("time_s"),
                "acquisition_coverage": (r.get("acquisition") or {}).get("coverage_of_pool"),
                "eval_time_s": r.get("eval_time_s"),
                "tau": r.get("tau"),
            }
            for l, v in (r.get("kappa_train") or {}).items():
                metrics[f"kappa_train/{l}"] = v
            for l, v in (r.get("collapse_strength_train") or {}).items():
                metrics[f"strength_train/{l}"] = v
            for l, v in (r.get("lambdas") or {}).items():
                metrics[f"lambda/{l}"] = v
            metrics.update(_flatten_nc(r.get("nc")))
            for k, v in (r.get("margins") or {}).items():
                if isinstance(v, (int, float)):
                    metrics[f"margins/{k}"] = v
            for e, v in (r.get("epsilon_sweep") or {}).items():
                metrics[f"eps_sweep/{e}"] = v.get("robust_acc")
            for k, v in ((r.get("corruptions") or {}).get("per_corruption") or {}).items():
                metrics[f"corruption/{k}"] = v
            sa = (r.get("acquisition") or {}).get("score_analysis") or {}
            for pair, v in (sa.get("spearman") or {}).items():
                metrics[f"score_spearman/{pair}"] = v
            for pair, v in (sa.get("top_overlap") or {}).items():
                metrics[f"score_overlap/{pair}"] = v
            hist = (r.get("acquisition") or {}).get("class_histogram")
            if hist:
                h = np.asarray(hist, dtype=float)
                p = h / max(h.sum(), 1)
                nz = p[p > 0]
                metrics["selection_class_entropy_norm"] = float(-(nz * np.log(nz)).sum() / math.log(max(len(h), 2)))
            for k, v in metrics.items():
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                rr = dict(rec)
                rr["metric"] = k
                rr["value"] = float(v)
                rows.append(rr)
        comp = s.get("compute", {}) or {}
        tel = s.get("telemetry", {}) or {}
        accs = [r.get("test_acc") for r in s.get("rounds", []) if r.get("test_acc") is not None]
        rob = [(r.get("apgd_ce_subset") or {}).get("robust_acc") for r in s.get("rounds", [])]
        rob = [v for v in rob if v is not None]
        for k, v in {
            "aulc_test_acc": float(np.mean(accs)) if len(accs) > 1 else None,  # area under the learning curve (mean over label fractions)
            "aulc_apgd_acc": float(np.mean(rob)) if len(rob) > 1 else None,
            "total_train_time_s": comp.get("train_time_s"),
            "total_acquisition_time_s": comp.get("acquisition_time_s"),
            "wall_time_s": s.get("wall_time_s"),
            "peak_gpu_mem_gb": (tel.get("peak_gpu_mem_allocated_bytes") or 0) / 1e9 if tel.get("peak_gpu_mem_allocated_bytes") else None,
            "peak_host_rss_gb": (tel.get("peak_host_rss_bytes") or 0) / 1e9 if tel.get("peak_host_rss_bytes") else None,
            "model_params": s.get("model_params"),
            "controller_params": s.get("controller_params"),
            "aa_total_time_s": (s.get("timings_s") or {}).get("eval_autoattack"),
        }.items():
            if v is None:
                continue
            rr = dict(base)
            rr.update({"round": -1, "label_fraction": float("nan"), "is_final": True, "metric": k, "value": float(v)})
            rows.append(rr)
    return pd.DataFrame(rows)


def crossmodal_long(jobs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for s in jobs:
        task = s.get("task")
        if task not in ("language_c4", "diffusion_ddpm", "audio_captioning", "audio_retrieval"):
            continue
        base = {"stage": s["_stage"], "task": task, "dataset": s.get("dataset"), "method": s.get("method"), "regularizer": s.get("regularizer"), "seed": int(s.get("seed", 0)), "job_dir": s["_job_dir"], "analysis_role": s.get("_analysis_role", "confirmatory"), "inference_eligible": bool(s.get("_include_in_inference", True))}
        for k, v in (s.get("final") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None and not (isinstance(v, float) and math.isnan(v)):
                rr = dict(base)
                rr.update({"metric": k, "value": float(v)})
                rows.append(rr)
        comp = s.get("compute", {}) or {}
        tel = s.get("telemetry", {}) or {}
        for k, v in {"wall_time_s": s.get("wall_time_s"), "train_time_s": comp.get("train_time_s"), "steps_per_s": comp.get("steps_per_s"), "peak_gpu_mem_gb": (tel.get("peak_gpu_mem_allocated_bytes") or 0) / 1e9 if tel.get("peak_gpu_mem_allocated_bytes") else None, "controller_params": s.get("controller_params")}.items():
            if v is None:
                continue
            rr = dict(base)
            rr.update({"metric": k, "value": float(v)})
            rows.append(rr)
    return pd.DataFrame(rows)


def overhead_long(jobs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for s in jobs:
        if s.get("task") != "vision_overhead":
            continue
        for r in s.get("results", []):
            rr = {"stage": s["_stage"], "job_dir": s["_job_dir"], "seed": int(s.get("seed", 0)), "analysis_role": s.get("_analysis_role", "confirmatory"), "inference_eligible": bool(s.get("_include_in_inference", True))}
            rr.update({k: v for k, v in r.items() if not isinstance(v, (dict, list))})
            rows.append(rr)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- statistics
def grouped_stats(df: pd.DataFrame, keys: list[str], confidence: float = 0.95) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = []
    for k, g in df.groupby(keys, dropna=False):
        st = summarize(g["value"].tolist(), confidence)
        row = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        row.update(st)
        row["seeds"] = ",".join(str(x) for x in sorted(g["seed"].unique())) if "seed" in g else ""
        out.append(row)
    return pd.DataFrame(out)


def paired_significance(df: pd.DataFrame, comparisons: list[dict[str, Any]], metrics: list[str], group_keys: list[str], n_boot: int, n_perm: int, min_pairs: int = 3) -> pd.DataFrame:
    """Paired-by-seed comparisons with Holm correction per (group, metric) family.

    Engineering and pilot jobs may be reported descriptively, but inferential tests are
    generated only from rows marked ``inference_eligible`` and only when at least
    ``min_pairs`` paired seeds exist.
    """
    if df.empty:
        return pd.DataFrame()
    if "inference_eligible" in df.columns:
        df = df[df["inference_eligible"].astype(bool)]
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for gkey, g in df.groupby(group_keys, dropna=False):
        gdict = dict(zip(group_keys, gkey if isinstance(gkey, tuple) else (gkey,)))
        for metric in metrics:
            gm = g[g["metric"] == metric]
            if gm.empty:
                continue
            fam: list[dict[str, Any]] = []
            for comp in comparisons:
                tgt = comp["target"]
                t = gm[gm["method"] == tgt].groupby("seed")["value"].mean()
                if t.empty:
                    continue
                for ref in comp.get("references", []):
                    r = gm[gm["method"] == ref].groupby("seed")["value"].mean()
                    seeds = sorted(set(t.index) & set(r.index))
                    if len(seeds) < int(min_pairs):
                        continue
                    res = compare_paired(t.loc[seeds].tolist(), r.loc[seeds].tolist(), n_boot=n_boot, n_perm=n_perm)
                    row = dict(gdict)
                    row.update({"metric": metric, "target": tgt, "reference": ref, "seeds": ",".join(map(str, seeds))})
                    row.update(res)
                    fam.append(row)
            if fam:
                adj = holm_bonferroni([f["p_ttest"] for f in fam])
                for f, a in zip(fam, adj):
                    f["p_ttest_holm"] = a
                rows.extend(fam)
    return pd.DataFrame(rows)


def per_example_mcnemar(jobs: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> pd.DataFrame:
    """Per-seed McNemar tests on AutoAttack robust / clean masks (target vs reference)."""
    index: dict[tuple, dict[str, Any]] = {}
    for s in jobs:
        if s.get("task") != "vision_al" or s.get("mode") == "initial_only":
            continue
        index[(s["dataset"], s["arch"], s["method"], int(s["seed"]))] = s
    rows = []
    groups = sorted(set((k[0], k[1]) for k in index))
    seeds_all = sorted(set(k[3] for k in index))
    for ds, arch in groups:
        for comp in comparisons:
            tgt = comp["target"]
            for ref in comp.get("references", []):
                for seed in seeds_all:
                    a = index.get((ds, arch, tgt, seed))
                    b = index.get((ds, arch, ref, seed))
                    if a is None or b is None:
                        continue
                    ra = a["rounds"][-1]["round"]
                    rb = b["rounds"][-1]["round"]
                    for kind, fname in (("aa_robust_acc", "aa_robust_mask_round{}.npy"), ("clean_acc", "aa_clean_mask_round{}.npy")):
                        pa = Path(a["_job_dir"]) / "artifacts" / fname.format(ra)
                        pb = Path(b["_job_dir"]) / "artifacts" / fname.format(rb)
                        if not (pa.exists() and pb.exists()):
                            continue
                        ma, mb = np.load(pa), np.load(pb)
                        if ma.shape != mb.shape:
                            continue
                        res = mcnemar_test(ma, mb)
                        rows.append({"dataset": ds, "arch": arch, "target": tgt, "reference": ref, "seed": seed, "metric": kind, **res})
    return pd.DataFrame(rows)


def controller_trajectories(jobs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for s in jobs:
        if s.get("task") not in ("vision_al", "language_c4", "diffusion_ddpm", "audio_captioning", "audio_retrieval"):
            continue
        p = Path(s["_job_dir"]) / "artifacts" / "controller_trajectory.json"
        if not p.exists():
            continue
        try:
            t = read_json(p)
        except Exception:
            continue
        for i, h in enumerate(t.get("history", [])):
            row = {"task": s.get("task"), "dataset": s.get("dataset"), "arch": s.get("arch"), "method": s.get("method"), "seed": int(s.get("seed", 0)), "i": i, "step": h.get("step"), "tau": h.get("tau")}
            for l, v in (h.get("lambdas") or {}).items():
                row[f"lambda/{l}"] = v
            for l, v in (h.get("kappa") or {}).items():
                row[f"kappa/{l}"] = v
            rows.append(row)
    return pd.DataFrame(rows)


def training_curves(jobs: list[dict[str, Any]], max_points: int = 400) -> pd.DataFrame:
    """Down-sampled metrics.jsonl 'train' records for collapse-dynamics figures."""
    rows = []
    for s in jobs:
        if s.get("task") != "vision_al" or s.get("mode") == "initial_only":
            continue
        p = Path(s["_job_dir"]) / "metrics.jsonl"
        if not p.exists():
            continue
        recs = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("kind") == "train":
                    recs.append(rec)
        if not recs:
            continue
        stride = max(1, len(recs) // max_points)
        for rec in recs[::stride]:
            row = {"dataset": s["dataset"], "arch": s["arch"], "method": s["method"], "seed": int(s["seed"]), "step": rec.get("step"), "round": rec.get("round"), "loss": rec.get("loss"), "ce": rec.get("ce"), "dcr": rec.get("dcr")}
            for k, v in rec.items():
                if k.startswith(("kappa/", "strength/", "lambda/", "dkappa/")):
                    row[k] = v
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- driver
def run_report(root: Path | str, strict: bool = False, extra_roots: list[Path | str] | None = None, quick: bool = False) -> dict[str, Any]:
    """Aggregate one campaign (plus ``extra_roots``: campaigns of other pods that ran the same protocol) into
    ``<root>/aggregate`` and ``<root>/manuscript``; ``strict`` validates every campaign and the report.

    ``quick`` is the runner's between-stage monitoring refresh: the same tables/figures with the bootstrap and
    permutation replicates capped at 1,000 so that launches are not held up; the aggregate stage and
    ``caenl report`` always use the protocol's replicate counts."""
    root = Path(root)
    extra = [Path(r) for r in (extra_roots or [])]
    out_dir = root / "aggregate"
    role = campaign_role(root)
    report_dir_name = report_dirname(role)
    man_dir = root / report_dir_name
    (man_dir / "tables").mkdir(parents=True, exist_ok=True)
    (man_dir / "figures").mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_finished_jobs = collect_jobs(root, include_engineering=True)
    jobs = [j for j in all_finished_jobs if j.get("_include_in_science", False)]
    for r in extra:
        more = collect_jobs(r)
        for j in more:
            j["_campaign"] = str(r)
        jobs.extend(more)
    protocol: dict[str, Any] = {}
    fp = root / "frozen_protocol.yaml"
    if fp.exists():
        from ..utils.io import deep_update, load_yaml

        protocol = load_yaml(fp) or {}
        # Plan-level defaults are part of the frozen experiment contract and may
        # intentionally override the repository-wide protocol (for example the
        # number of paired confirmatory seeds and cross-modal comparison family).
        # Earlier versions ignored these defaults during reporting.
        plan_doc = load_yaml(root / "PLAN.yaml") if (root / "PLAN.yaml").exists() else {}
        plan_defaults = ((plan_doc or {}).get("plan", {}) or {}).get("defaults", {}) or {}
        protocol = deep_update(protocol, plan_defaults)
    stats_cfg = protocol.get("statistics", {})
    comparisons = stats_cfg.get("comparisons", [{"target": "caenl_macc_lite", "references": ["random", "entropy", "margin", "badge", "coreset"]}])
    metrics = stats_cfg.get("metrics", ["test_acc", "aa_robust_acc", "apgd_ce_subset_acc", "pgd_subset_acc", "test_ece"])
    n_boot = int(stats_cfg.get("bootstrap_replicates", 10000))
    n_perm = int(stats_cfg.get("permutation_replicates", 10000))
    if quick:
        n_boot, n_perm = min(n_boot, 1000), min(n_perm, 1000)
    confidence = float(stats_cfg.get("confidence_level", 0.95))
    min_pairs = int(stats_cfg.get("minimum_paired_seeds_for_inference", 3))
    report_meta = derive_report_meta(root, jobs, protocol)

    pd.DataFrame([{k: v for k, v in j.items() if not isinstance(v, (dict, list))} for j in all_finished_jobs]).to_csv(out_dir / "all_finished_jobs.csv", index=False)
    pd.DataFrame([{k: v for k, v in j.items() if not isinstance(v, (dict, list))} for j in jobs]).to_csv(out_dir / "all_jobs.csv", index=False)
    vis = vision_long(jobs)
    cross = crossmodal_long(jobs)
    over = overhead_long(jobs)
    vis.to_csv(out_dir / "vision_long.csv", index=False)
    cross.to_csv(out_dir / "crossmodal_long.csv", index=False)
    over.to_csv(out_dir / "overhead_long.csv", index=False)

    result: dict[str, Any] = {"root": str(root), "campaign_role": role, "report_dir": report_dir_name, "n_jobs": len(jobs), "n_finished_jobs_all_roles": len(all_finished_jobs), "generated_at": utc_now(), "quick": bool(quick), "bootstrap_replicates": n_boot, "permutation_replicates": n_perm, "minimum_paired_seeds_for_inference": min_pairs, "report_meta": report_meta}
    vis_grouped = grouped_stats(vis, ["dataset", "arch", "method", "round", "label_fraction", "metric"], confidence) if not vis.empty else pd.DataFrame()
    vis_grouped.to_csv(out_dir / "vision_grouped.csv", index=False)
    cross_grouped = grouped_stats(cross, ["task", "dataset", "method", "metric"], confidence) if not cross.empty else pd.DataFrame()
    cross_grouped.to_csv(out_dir / "crossmodal_grouped.csv", index=False)

    sig = pd.DataFrame()
    mcn = pd.DataFrame()
    if not vis.empty:
        final = vis[vis["is_final"] & ((vis["round"] >= 0) | vis["metric"].str.startswith("aulc_"))]
        sig = paired_significance(final, comparisons, metrics, ["dataset", "arch"], n_boot, n_perm, min_pairs=min_pairs)
        sig.to_csv(out_dir / "significance_final.csv", index=False)
        sig_curve = paired_significance(vis[vis["round"] >= 0], comparisons, ["test_acc", "apgd_ce_subset_acc"], ["dataset", "arch", "round"], min(n_boot, 2000), min(n_perm, 2000), min_pairs=min_pairs)
        sig_curve.to_csv(out_dir / "significance_by_round.csv", index=False)
        mcn = per_example_mcnemar(jobs, comparisons)
        mcn.to_csv(out_dir / "mcnemar_per_example.csv", index=False)
    cross_sig = pd.DataFrame()
    if not cross.empty:
        cross_comparisons = stats_cfg.get(
            "crossmodal_comparisons",
            [{"target": m, "references": ["baseline"]} for m in ("fixed_dcr", "macc_lite", "full_macc")]
            + [{"target": "full_macc", "references": ["macc_lite"]}],
        )
        cmetrics = sorted(cross["metric"].unique().tolist())
        cross_sig = paired_significance(cross, cross_comparisons, cmetrics, ["task", "dataset"], n_boot, n_perm, min_pairs=min_pairs)
        cross_sig.to_csv(out_dir / "crossmodal_significance.csv", index=False)
    traj = controller_trajectories(jobs)
    traj.to_csv(out_dir / "controller_trajectories.csv", index=False)
    curves = training_curves(jobs)
    curves.to_csv(out_dir / "training_curves.csv", index=False)

    from . import figures, tables
    from .summary import write_summary

    table_files = tables.write_all(man_dir / "tables", vis, vis_grouped, sig, mcn, cross, cross_grouped, cross_sig, over, protocol, report_meta=report_meta)
    figure_files = figures.write_all(man_dir / "figures", vis, vis_grouped, cross, cross_grouped, over, traj, curves, protocol, report_meta=report_meta)
    write_summary(man_dir / "results_summary.md", root, jobs, vis_grouped, sig, mcn, cross_grouped, cross_sig, over, table_files, figure_files, protocol)
    result.update({"tables": [str(p) for p in table_files], "figures": [str(p) for p in figure_files], "vision_rows": int(len(vis)), "crossmodal_rows": int(len(cross))})
    atomic_write_json(out_dir / "report_manifest.json", result)
    if strict:
        from ..publication import load_frozen_plan, validate_campaign

        check = validate_campaign(load_frozen_plan(root), require_report=True)
        checks = {str(root): check}
        for r in extra:  # other pods: every expected job of their plans must validate too (their own report is not required)
            checks[str(r)] = validate_campaign(load_frozen_plan(r), require_report=False)
        errors = [f"{k}: {e}" for k, c in checks.items() for e in c["errors"]]
        result["publication_check"] = {k: {kk: vv for kk, vv in c.items() if kk != "stages"} for k, c in checks.items()}
        result["publication_stages"] = {k: c["stages"] for k, c in checks.items()}
        if errors:
            raise RuntimeError("publication-readiness validation FAILED:\n  - " + "\n  - ".join(errors[:30]) + f"\n(see PUBLICATION_CHECK.json under each campaign root)")
    return result


def run_task(ctx) -> dict[str, Any]:
    """The ``aggregate`` stage: build the report, then verify the whole expected job matrix (strict)."""
    root = Path(ctx.config["paths"]["campaign_root"])
    return {"task": "aggregate", **run_report(root, strict=True)}
