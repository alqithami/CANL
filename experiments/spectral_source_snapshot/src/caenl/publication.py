"""Publication-readiness validation of finished stages and whole campaigns.

Nothing in the manuscript may rest on a job that merely *looks* finished.  After every stage the
runner (``publication_gate: true`` in the plan, the default) calls :func:`validate_stage`, which
checks every job the plan expected against its recorded outputs and writes
``stages/<stage>/STAGE_MANIFEST.json`` (job ids, states, SHA-256 of ``job.json`` and
``summary.json``, key metrics).  A stage that does not validate stops the campaign.
``caenl report --strict`` and the ``aggregate`` stage call :func:`validate_campaign`, which
re-checks every stage, the full expected job matrix, the recorded hashes and the report
artefacts, and fails (non-zero exit / failed aggregate job) on any problem.

Checks per job type
-------------------
* every job: ``status.json`` state ``succeeded``; ``summary.json`` present, parseable, same ``job_id``;
* ``vision_al`` (active learning): one record per round (initial + ``rounds``), final label fraction
  equal to the protocol's, finite ``test_acc``/``test_loss``/``test_ece`` (+ APGD/PGD/corruption
  proxies when enabled), AutoAttack at the final round under the frozen protocol (evaluator,
  version, eps = 8/255, the fixed subset size) with the robust/clean masks saved, ``labels`` block;
* ``vision_al`` (``initial_only``): one round record with finite ``test_acc``;
* ``vision_aa_sanity``: ``sanity_ok`` when an expectation is configured;
* ``vision_overhead``: a finite step time for every benchmarked configuration;
* ``language_c4`` / ``diffusion_ddpm`` / ``audio_retrieval``: the protocol's metrics finite;
* ``audio_captioning``: CIDEr-D/BLEU-4/ROUGE-L finite, METEOR/SPICE finite when required,
  official ``pycocoevalcap`` values when ``official_metrics`` is on;
* ``prepare_data`` / ``aggregate``: summary present.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional

from .plan import Plan
from .utils.io import atomic_write_json, read_json, utc_now
from .report.context import campaign_role, report_dirname


def _sha(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _job_dir(plan: Plan, stage_name: str, job_id: str) -> Path:
    from .runner import job_dir_for

    return job_dir_for(plan, stage_name, job_id)


# ----------------------------------------------------------------------------- per-job checks
def required_vision_metrics(cfg: dict[str, Any]) -> list[str]:
    """Final-round metrics a vision job must report, mirroring the defaults of ``VisionALJob.evaluate_round``."""
    rob = cfg.get("robustness", {})
    inter = rob.get("intermediate", {})
    req = ["test_acc", "test_loss", "test_ece"]
    if bool(inter.get("enabled", True)) and bool(inter.get("also_final", True)):
        req.append("apgd_ce_subset_acc")
        if int(inter.get("pgd_steps", 20)) > 0:
            req.append("pgd_subset_acc")
    if bool(rob.get("corruptions", {}).get("enabled", False)):
        req.append("corruption_mean_acc")
    if bool(rob.get("final", {}).get("enabled", True)):
        req += ["aa_robust_acc", "aa_clean_acc"]
    return req


def required_crossmodal_metrics(cfg: dict[str, Any]) -> list[str]:
    t = cfg["type"]
    if t == "language_c4":
        return list(cfg.get("language", {}).get("metrics", ["validation_nll", "perplexity", "token_ece"]))
    if t == "diffusion_ddpm":
        return list(cfg.get("diffusion", {}).get("metrics", ["fid", "kid", "inception_score"]))
    if t == "audio_retrieval":
        return list(cfg.get("audio", {}).get("retrieval", {}).get("metrics", ["t2a_r1", "t2a_r5", "t2a_r10", "a2t_r1", "a2t_r5", "a2t_r10", "median_rank"]))
    if t == "audio_captioning":
        cc = cfg.get("audio", {}).get("captioning", {})
        req = [m for m in cc.get("metrics", ["cider_d", "bleu4", "rouge_l", "meteor", "spice", "validation_loss"]) if m != "validation_loss"]
        req.append("val_loss")
        if not bool(cc.get("java_metrics", True)):
            req = [m for m in req if m not in ("meteor", "spice")]
        elif not bool(cc.get("spice", True)):
            req = [m for m in req if m != "spice"]
        return req
    return []


def check_job(job_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate one finished job directory against its frozen configuration."""
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {}
    status_path, summary_path = job_dir / "status.json", job_dir / "summary.json"
    state = None
    if not status_path.exists():
        errors.append("status.json missing")
    else:
        try:
            state = read_json(status_path).get("state")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"status.json unreadable: {exc}")
        if state != "succeeded":
            errors.append(f"state is {state!r}, not 'succeeded'")
    summary: dict[str, Any] = {}
    if not summary_path.exists():
        errors.append("summary.json missing")
    else:
        try:
            summary = read_json(summary_path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"summary.json unreadable: {exc}")
    if summary and summary.get("job_id") not in (None, cfg.get("job_id")):
        errors.append(f"summary job_id {summary.get('job_id')!r} != {cfg.get('job_id')!r}")
    t = cfg["type"]
    if t == "fault_injection":  # smoke-only: validate as the emulated type
        cfg = dict(cfg)
        cfg["type"] = t = str(cfg.get("fault", {}).get("emulate", "language_c4"))
    final = summary.get("final", {}) if isinstance(summary.get("final"), dict) else {}
    if summary and not errors:
        if t == "vision_al":
            rounds = summary.get("rounds", [])
            al = cfg.get("active_learning", {})
            full = bool(cfg.get("method", {}).get("full_supervision", False))
            init_only = cfg.get("mode") == "initial_only"
            n_rounds = 0 if (full or init_only) else int(al.get("rounds", 5))
            if len(rounds) != n_rounds + 1:
                errors.append(f"{len(rounds)} round records, expected {n_rounds + 1}")
            if init_only:
                if not _finite(rounds[0].get("test_acc") if rounds else None):
                    errors.append("initial round without finite test_acc")
                metrics["test_acc"] = rounds[0].get("test_acc") if rounds else None
            else:
                for m in required_vision_metrics(cfg):
                    v = final.get(m)
                    metrics[m] = v
                    if not _finite(v):
                        errors.append(f"final.{m} is not finite ({v!r})")
                if "labels" not in summary:
                    errors.append("labels block (labels_available / labels_trained) missing")
                if rounds:
                    frac = float(rounds[-1].get("label_fraction", float("nan")))
                    want = 1.0 if full else float(al.get("initial_fraction", 0.1)) + n_rounds * float(al.get("fraction_per_round", 0.1))
                    if not math.isfinite(frac) or abs(frac - want) > 0.011:
                        errors.append(f"final label fraction {frac} != protocol {want:.3f}")
                    rob = cfg.get("robustness", {})
                    fin = rob.get("final", {})
                    if fin.get("enabled", True):
                        aa = rounds[-1].get("autoattack") or {}
                        ev = str(aa.get("evaluator", ""))
                        if not ev.startswith("autoattack"):
                            errors.append(f"final AutoAttack missing (evaluator={ev!r})")
                        else:
                            want_ver = str(fin.get("version", "standard"))
                            if ev != f"autoattack-{want_ver}":
                                errors.append(f"AutoAttack version {ev} != protocol autoattack-{want_ver}")
                            if abs(float(aa.get("eps", -1)) - float(rob.get("eps", 8 / 255))) > 1e-9:
                                errors.append(f"AutoAttack eps {aa.get('eps')} != protocol {rob.get('eps', 8 / 255)}")
                            if str(aa.get("norm", "Linf")) != str(rob.get("norm", "Linf")):
                                errors.append("AutoAttack norm differs from the protocol")
                            n_test = int(summary.get("n_test", 0) or 0)
                            n_want = n_test if not fin.get("n_examples") else min(int(fin["n_examples"]), n_test)
                            if n_test and int(aa.get("n", -1)) != n_want:
                                errors.append(f"AutoAttack evaluated {aa.get('n')} examples, protocol requires {n_want}")
                            r_final = rounds[-1].get("round")
                            for art in (f"aa_robust_mask_round{r_final}.npy", f"aa_clean_mask_round{r_final}.npy", f"aa_subset_round{r_final}.npy"):
                                if not (job_dir / "artifacts" / art).exists():
                                    errors.append(f"artifact {art} missing")

                method = cfg.get("method", {}) or {}
                regularizer = str(method.get("regularizer", "none"))
                gates = cfg.get("quality_gates", {}) or {}
                if bool(gates.get("require_controller_updates", False)) and regularizer in {"macc_lite", "full_macc"}:
                    trajectory_path = job_dir / "artifacts" / "controller_trajectory.json"
                    minimum = int(gates.get("minimum_controller_updates", 1))
                    if not trajectory_path.exists():
                        errors.append("controller_trajectory.json missing for adaptive-controller run")
                    else:
                        try:
                            trajectory = read_json(trajectory_path)
                            history = trajectory.get("history", []) if isinstance(trajectory, dict) else []
                            metrics["controller_updates"] = len(history)
                            if len(history) < minimum:
                                errors.append(f"controller recorded {len(history)} updates; protocol requires at least {minimum}")
                            if regularizer == "full_macc":
                                learned = [h for h in history if isinstance(h, dict) and (h.get("update") or h.get("kind") == "acquisition_update")]
                                if len(learned) < minimum:
                                    errors.append(f"Full MACC recorded only {len(learned)} learned policy updates; protocol requires at least {minimum}")
                        except Exception as exc:  # noqa: BLE001
                            errors.append(f"controller trajectory unreadable: {exc}")
        elif t == "vision_aa_sanity":
            metrics["clean_acc"] = summary.get("clean_acc_full_test")
            metrics["robust_acc"] = (summary.get("aa") or {}).get("robust_acc")
            if not _finite(metrics["clean_acc"]) or not _finite(metrics["robust_acc"]):
                errors.append("sanity job without finite clean/robust accuracy")
            if cfg.get("sanity", {}).get("expected") and summary.get("sanity_ok") is not True:
                errors.append(f"AutoAttack sanity check did not pass: {summary.get('checks')}")
        elif t == "vision_overhead":
            res = summary.get("results", [])
            want = set(cfg.get("overhead", {}).get("configs", []))
            seen = {r.get("config") for r in res if _finite(r.get("step_time_ms"))}
            missing = want - seen
            if missing:
                errors.append(f"no finite step time for configs {sorted(missing)}")
            metrics["configs"] = sorted(seen)
        elif t in ("language_c4", "diffusion_ddpm", "audio_captioning", "audio_retrieval"):
            for m in required_crossmodal_metrics(cfg):
                v = final.get(m)
                metrics[m] = v
                if not _finite(v):
                    errors.append(f"final.{m} is not finite ({v!r})")
            if t == "audio_captioning":
                cc = cfg.get("audio", {}).get("captioning", {})
                if bool(cc.get("official_metrics", True)) and summary.get("metrics_source") != "pycocoevalcap":
                    errors.append(f"caption metrics source is {summary.get('metrics_source')!r}; the protocol requires the official pycocoevalcap values")
                if str(cfg.get("audio", {}).get("dataset", "audiocaps")) not in ("synthetic",) and not (summary.get("manifest_provenance") or {}).get("manifest_sha256"):
                    errors.append("manifest provenance (sha256) missing from the summary")

            # An adaptive-controller experiment is not valid if the controller never acted.
            method = cfg.get("method", {}) or {}
            regularizer = str(method.get("regularizer", "none"))
            gates = cfg.get("quality_gates", {}) or {}
            require_updates = bool(gates.get("require_controller_updates", False))
            if require_updates and regularizer in {"macc_lite", "full_macc"}:
                trajectory_path = job_dir / "artifacts" / "controller_trajectory.json"
                minimum = int(gates.get("minimum_controller_updates", 1))
                if not trajectory_path.exists():
                    errors.append("controller_trajectory.json missing for adaptive-controller run")
                else:
                    try:
                        trajectory = read_json(trajectory_path)
                        history = trajectory.get("history", []) if isinstance(trajectory, dict) else []
                        n_updates = len(history)
                        metrics["controller_updates"] = n_updates
                        if n_updates < minimum:
                            errors.append(f"controller recorded {n_updates} updates; protocol requires at least {minimum}")
                        if regularizer == "full_macc":
                            learned = [
                                h
                                for h in history
                                if isinstance(h, dict)
                                and (
                                    "policy_loss" in h
                                    or (
                                        isinstance(h.get("update"), dict)
                                        and "policy_loss" in h["update"]
                                    )
                                    or (
                                        h.get("kind") == "acquisition_update"
                                        and "policy_loss" in h
                                    )
                                )
                            ]
                            if len(learned) < minimum:
                                errors.append(f"Full MACC recorded only {len(learned)} learned policy updates; protocol requires at least {minimum}")
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"controller trajectory unreadable: {exc}")
        elif t == "prepare_data":
            if not summary.get("prepared"):
                errors.append("prepare_data summary lists nothing prepared")
        elif t == "aggregate":
            pass
    return {"job_id": cfg.get("job_id"), "type": t, "state": state, "ok": not errors, "errors": errors, "warnings": warnings, "metrics": metrics, "job_sha256": _sha(job_dir / "job.json"), "summary_sha256": _sha(summary_path), "job_dir": str(job_dir)}


# ----------------------------------------------------------------------------- stage / campaign
def expected_matrix(stage: dict[str, Any], jobs: list[dict[str, Any]]) -> dict[str, Any]:
    """What the plan promises for a stage: the job ids plus the (method, seed) grid if applicable."""
    out: dict[str, Any] = {"job_ids": [j["job_id"] for j in jobs]}
    mx = stage.get("matrix", {})
    if "seed" in mx:
        out["seeds"] = sorted({int(j["seed"]) for j in jobs if "seed" in j})
    if "method" in mx:
        out["methods"] = sorted({j["method"]["name"] for j in jobs if isinstance(j.get("method"), dict)})
    exp = stage.get("expected") or {}
    for key in ("seeds", "methods"):
        if key in exp:
            missing = set(exp[key]) - set(out.get(key, []))
            if missing:
                out.setdefault("errors", []).append(f"plan stage '{stage['name']}' expects {key} {sorted(missing)} that the matrix does not produce")
    return out


def validate_stage(plan: Plan, stage: dict[str, Any], jobs: list[dict[str, Any]], write_manifest: bool = True) -> dict[str, Any]:
    checks = []
    for j in jobs:
        jd = _job_dir(plan, stage["name"], j["job_id"])
        checks.append(check_job(jd, j))
    exp = expected_matrix(stage, jobs)
    errors = list(exp.get("errors", []))
    failed = [c for c in checks if not c["ok"]]
    for c in failed:
        errors.append(f"{c['job_id']}: " + "; ".join(c["errors"][:4]))
    report = {
        "stage": stage["name"],
        "type": stage["type"],
        "generated_at": utc_now(),
        "plan_fingerprint": plan.fingerprint(),
        "expected": {k: v for k, v in exp.items() if k != "errors"},
        "n_expected": len(jobs),
        "n_ok": len(jobs) - len(failed),
        "ok": not errors,
        "errors": errors,
        "jobs": checks,
    }
    if write_manifest:
        sd = plan.campaign_root / "stages" / stage["name"]
        sd.mkdir(parents=True, exist_ok=True)
        atomic_write_json(sd / "STAGE_MANIFEST.json", report)
    return report


def validate_campaign(plan: Plan, require_report: bool = True, stages: Optional[list[str]] = None) -> dict[str, Any]:
    """Every stage + the full expected matrix + hash stability + report artefacts."""
    reports = []
    errors: list[str] = []
    for stage, jobs in plan.expand():
        if stages and stage["name"] not in stages:
            continue
        if stage["type"] == "aggregate":
            continue
        prev_path = plan.campaign_root / "stages" / stage["name"] / "STAGE_MANIFEST.json"
        prev = read_json(prev_path) if prev_path.exists() else None
        rep = validate_stage(plan, stage, jobs, write_manifest=True)
        if bool(stage.get("expect_failure", False)):
            # smoke-only gate probe: it must be rejected; its jobs never enter the report
            if rep["ok"]:
                errors.append(f"stage {stage['name']} is a gate probe (expect_failure) but validated: the publication gate is not working")
            rep["ok"], rep["probe"] = not rep["ok"], True
            reports.append(rep)
            continue
        if prev and prev.get("ok"):
            old = {c["job_id"]: c.get("summary_sha256") for c in prev.get("jobs", [])}
            for c in rep["jobs"]:
                if c["job_id"] in old and old[c["job_id"]] and c.get("summary_sha256") != old[c["job_id"]]:
                    errors.append(f"{stage['name']}/{c['job_id']}: summary.json changed after the stage was validated ({old[c['job_id']][:12]} -> {str(c.get('summary_sha256'))[:12]})")
        if not rep["ok"]:
            errors.append(f"stage {stage['name']}: {len(rep['errors'])} problem(s): " + " | ".join(rep["errors"][:3]))
        reports.append(rep)
    role = campaign_role(plan.campaign_root)
    report_dir = report_dirname(role)
    if require_report:
        root = plan.campaign_root
        man = root / "aggregate" / "report_manifest.json"
        if not man.exists():
            errors.append("aggregate/report_manifest.json missing (run `caenl report`)")
        else:
            rm = read_json(man)
            has_vis = int(rm.get("vision_rows", 0)) > 0
            has_cross = int(rm.get("crossmodal_rows", 0)) > 0
            if (has_vis or has_cross) and not rm.get("tables"):
                errors.append("report produced no tables")
            if (has_vis or has_cross) and not rm.get("figures"):
                errors.append("report produced no figures")
            for p in rm.get("tables", []) + rm.get("figures", []):
                if not Path(p).exists():
                    errors.append(f"report artefact missing: {p}")
            if has_vis and role == "confirmatory":
                sig = root / "aggregate" / "significance_final.csv"
                if not sig.exists():
                    errors.append("significance_final.csv missing")
                else:
                    try:
                        import pandas as pd

                        try:
                            df = pd.read_csv(sig)
                        except pd.errors.EmptyDataError:
                            df = pd.DataFrame()
                        n_seeds = 0
                        for stage, jobs in plan.expand():
                            if stage["type"] == "vision_al" and "method" in stage.get("matrix", {}):
                                n_seeds = max(n_seeds, len({j.get("seed") for j in jobs}))
                        min_pairs = int((plan.protocol.get("statistics", {}) or {}).get("minimum_paired_seeds_for_inference", 3))
                        if df.empty and n_seeds >= min_pairs:
                            errors.append(f"significance_final.csv is empty although paired comparisons over >= {min_pairs} confirmatory seeds are configured")
                        elif not df.empty and n_seeds >= min_pairs and df["p_ttest"].isna().all():
                            errors.append(f"all paired t-tests are NaN despite >= {min_pairs} configured confirmatory seeds")
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"significance_final.csv unreadable: {exc}")
            if has_cross and role == "confirmatory":
                cross_sig = root / "aggregate" / "crossmodal_significance.csv"
                if not cross_sig.exists():
                    errors.append("crossmodal_significance.csv missing")
                else:
                    try:
                        import pandas as pd

                        try:
                            cdf = pd.read_csv(cross_sig)
                        except pd.errors.EmptyDataError:
                            cdf = pd.DataFrame()
                        n_seeds = 0
                        expected_methods: set[str] = set()
                        for stage, jobs in plan.expand():
                            if stage["type"] in {"language_c4", "diffusion_ddpm", "audio_captioning", "audio_retrieval"} and "method" in stage.get("matrix", {}):
                                n_seeds = max(n_seeds, len({j.get("seed") for j in jobs}))
                                expected_methods.update(str(j.get("method", {}).get("name")) for j in jobs if isinstance(j.get("method"), dict))
                        min_pairs = int((plan.base_config().get("statistics", {}) or {}).get("minimum_paired_seeds_for_inference", 3))
                        if n_seeds >= min_pairs and cdf.empty:
                            errors.append(f"crossmodal_significance.csv is empty although paired comparisons over >= {min_pairs} confirmatory seeds are configured")
                        elif not cdf.empty:
                            if "n_pairs" in cdf and int(cdf["n_pairs"].min()) < min_pairs:
                                errors.append(f"crossmodal_significance.csv contains comparisons with fewer than {min_pairs} paired seeds")
                            if "p_ttest" in cdf and "p_perm" in cdf and cdf["p_ttest"].isna().all() and cdf["p_perm"].isna().all():
                                errors.append("all cross-modal inferential p-values are NaN")
                            comparisons = (plan.base_config().get("statistics", {}) or {}).get("crossmodal_comparisons", [])
                            required_pairs = {(str(c["target"]), str(r)) for c in comparisons for r in c.get("references", [])}
                            present_pairs = set(zip(cdf.get("target", []), cdf.get("reference", [])))
                            missing_pairs = sorted(required_pairs - present_pairs)
                            if missing_pairs:
                                errors.append(f"crossmodal_significance.csv missing configured method pairs: {missing_pairs}")
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"crossmodal_significance.csv unreadable: {exc}")
        summary_path = root / report_dir / "results_summary.md"
        if not summary_path.exists():
            errors.append(f"{report_dir}/results_summary.md missing")
    out = {"campaign_root": str(plan.campaign_root), "campaign_role": role, "report_dir": report_dir, "generated_at": utc_now(), "ok": not errors, "errors": errors, "stages": [{k: v for k, v in r.items() if k != "jobs"} for r in reports]}
    check_name = "PUBLICATION_CHECK.json" if role == "confirmatory" else "CAMPAIGN_INTEGRITY_CHECK.json"
    atomic_write_json(plan.campaign_root / check_name, out)
    return out


def load_frozen_plan(root: Path | str) -> Plan:
    """Rebuild the Plan from the campaign's frozen ``PLAN.yaml`` (plan + protocol as run)."""
    from .utils.io import load_yaml

    root = Path(root).resolve()
    frozen = load_yaml(root / "PLAN.yaml")
    raw = dict(frozen.get("plan", {}))
    protocol = dict(frozen.get("protocol", {}))
    # validate the directory we were given, even if the campaign was moved or copied since it ran
    raw["results_root"] = str(root.parent)
    raw["campaign_id"] = root.name
    return Plan(raw, root / "PLAN.yaml", protocol=protocol)
