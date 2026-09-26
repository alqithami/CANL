"""Runner behaviour the review demands: a stage that fails publication validation stops the campaign,
forced interruptions must resume from checkpoints, and gate probes (expect_failure) work."""
import json
import os
from pathlib import Path

import pytest
import yaml

from caenl.runner import job_dir_for, job_state, run_campaign
from caenl.plan import Plan

ROOT = Path(__file__).resolve().parents[1]


def _plan(tmp_path: Path, stages: list[dict]) -> Path:
    raw = {
        "campaign_id": "gate-test",
        "results_root": str(tmp_path / "results"),
        "data_root": str(tmp_path / "data"),
        "cache_root": str(tmp_path / "cache"),
        "protocol": str(ROOT / "configs/protocol/revision_protocol.yaml"),
        "report_after_each_stage": False,
        "publication_gate": True,
        "defaults": {"reproducibility": {"save_pip_freeze": False, "telemetry_interval_s": 2}},
        "stages": stages,
    }
    p = tmp_path / "plan.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


@pytest.mark.timeout(600)
def test_nonfinite_metric_stops_the_campaign_before_the_next_stage(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    plan_path = _plan(tmp_path, [
        {"name": "ok_stage", "type": "fault_injection", "matrix": {"seed": [0], "method": ["baseline"]}, "job": {"fault": {"kind": "nan_metric", "emulate": "language_c4", "final": {"validation_nll": 3.1, "perplexity": 22.0, "token_ece": 0.05}}}},
        {"name": "bad_stage", "type": "fault_injection", "matrix": {"seed": [0], "method": ["baseline"]}, "job": {"fault": {"kind": "nan_metric", "emulate": "language_c4", "final": {"validation_nll": "nan", "perplexity": 22.0, "token_ece": 0.05}}}},
        {"name": "never_runs", "type": "fault_injection", "matrix": {"seed": [0], "method": ["baseline"]}, "job": {"fault": {"kind": "nan_metric", "emulate": "language_c4", "final": {"validation_nll": 3.0, "perplexity": 20.0, "token_ece": 0.05}}}},
    ])
    rc = run_campaign(str(plan_path), report=False)
    assert rc == 1
    plan = Plan.load(plan_path)
    assert job_state(job_dir_for(plan, "ok_stage", "baseline/seed0")) == "succeeded"
    assert job_state(job_dir_for(plan, "bad_stage", "baseline/seed0")) == "succeeded"  # the job ran, the gate rejected it
    man = json.loads((plan.campaign_root / "stages" / "bad_stage" / "STAGE_MANIFEST.json").read_text())
    assert not man["ok"] and any("validation_nll" in e for e in man["errors"])
    assert job_state(job_dir_for(plan, "never_runs", "baseline/seed0")) == "pending"


@pytest.mark.timeout(600)
def test_crashing_job_stops_the_campaign(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    plan_path = _plan(tmp_path, [
        {"name": "crash", "type": "fault_injection", "matrix": {"seed": [0], "method": ["baseline"]}, "job": {"fault": {"kind": "raise"}}},
        {"name": "after", "type": "fault_injection", "matrix": {"seed": [0], "method": ["baseline"]}, "job": {"fault": {"kind": "nan_metric", "final": {"validation_nll": 3.0, "perplexity": 20.0, "token_ece": 0.05}}}},
    ])
    rc = run_campaign(str(plan_path), report=False, max_retries=0)
    assert rc == 1
    plan = Plan.load(plan_path)
    assert job_state(job_dir_for(plan, "crash", "baseline/seed0")) == "failed"
    assert job_state(job_dir_for(plan, "after", "baseline/seed0")) == "pending"


@pytest.mark.timeout(600)
def test_expect_failure_probe_passes_only_when_the_gate_rejects(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    plan_path = _plan(tmp_path, [
        {"name": "probe", "type": "fault_injection", "expect_failure": True, "matrix": {"seed": [0], "method": ["baseline"]}, "job": {"fault": {"kind": "nan_metric", "final": {"validation_nll": "nan", "perplexity": 20.0, "token_ece": 0.05}}}},
        {"name": "after", "type": "fault_injection", "matrix": {"seed": [0], "method": ["baseline"]}, "job": {"fault": {"kind": "nan_metric", "final": {"validation_nll": 3.0, "perplexity": 20.0, "token_ece": 0.05}}}},
    ])
    assert run_campaign(str(plan_path), report=False) == 0
    plan = Plan.load(plan_path)
    assert job_state(job_dir_for(plan, "after", "baseline/seed0")) == "succeeded"
    # a probe that unexpectedly validates is itself a failure
    (tmp_path / "two").mkdir()
    plan_path2 = _plan(tmp_path / "two", [
        {"name": "probe", "type": "fault_injection", "expect_failure": True, "matrix": {"seed": [0], "method": ["baseline"]}, "job": {"fault": {"kind": "nan_metric", "final": {"validation_nll": 3.0, "perplexity": 20.0, "token_ece": 0.05}}}},
    ])
    assert run_campaign(str(plan_path2), report=False) == 1


@pytest.mark.timeout(600)
def test_report_refresh_failure_stops_the_campaign(tmp_path, monkeypatch):
    import caenl.report.aggregate as agg

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    def boom(root, strict=False, **kwargs):
        raise RuntimeError("simulated report failure")

    monkeypatch.setattr(agg, "run_report", boom)
    plan_path = _plan(tmp_path, [
        {"name": "first", "type": "fault_injection", "matrix": {"seed": [0], "method": ["baseline"]}, "job": {"fault": {"kind": "nan_metric", "final": {"validation_nll": 3.0, "perplexity": 20.0, "token_ece": 0.05}}}},
        {"name": "second", "type": "fault_injection", "matrix": {"seed": [0], "method": ["baseline"]}, "job": {"fault": {"kind": "nan_metric", "final": {"validation_nll": 3.0, "perplexity": 20.0, "token_ece": 0.05}}}},
    ])
    raw = yaml.safe_load(plan_path.read_text())
    raw["report_after_each_stage"] = True
    plan_path.write_text(yaml.safe_dump(raw))
    assert run_campaign(str(plan_path), report=True) == 1
    plan = Plan.load(plan_path)
    assert job_state(job_dir_for(plan, "first", "baseline/seed0")) == "succeeded"
    assert job_state(job_dir_for(plan, "second", "baseline/seed0")) == "pending"


def _events(job_dir: Path) -> list[dict]:
    return [json.loads(l) for l in (job_dir / "events.jsonl").read_text().splitlines() if l.strip()]


@pytest.mark.timeout(900)
def test_dependency_scheduler_overlaps_independent_stages_and_respects_gpu_share(tmp_path, monkeypatch):
    """Two independent stages run concurrently on the 'GPU' pool; dependants wait for validation;
    gpu_share lets two small jobs share a device; exclusive stages run alone."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    ok = {"kind": "nan_metric", "emulate": "language_c4", "final": {"validation_nll": 3.0, "perplexity": 20.0, "token_ece": 0.05}, "sleep_s": 6}
    plan_path = _plan(tmp_path, [
        {"name": "a", "type": "fault_injection", "depends_on": [], "parallel": 2, "gpu_share": 2, "matrix": {"seed": [0, 1], "method": ["baseline"]}, "job": {"fault": ok}},
        {"name": "b", "type": "fault_injection", "depends_on": [], "parallel": 2, "gpu_share": 2, "matrix": {"seed": [0, 1], "method": ["baseline"]}, "job": {"fault": ok}},
        {"name": "c", "type": "fault_injection", "depends_on": ["a"], "parallel": 1, "matrix": {"seed": [0], "method": ["baseline"]}, "job": {"fault": dict(ok, sleep_s=1)}},
        {"name": "x", "type": "fault_injection", "depends_on": ["a", "b"], "exclusive": True, "matrix": {"seed": [0], "method": ["baseline"]}, "job": {"fault": dict(ok, sleep_s=1)}},
    ])
    # two "GPUs": a and b each allow two jobs per GPU -> all four jobs run at once
    assert run_campaign(str(plan_path), report=False, gpus="0,1", poll_s=0.5) == 0
    plan = Plan.load(plan_path)
    spans = {}
    for stage in ("a", "b", "c", "x"):
        for seed in ([0, 1] if stage in ("a", "b") else [0]):
            ev = _events(job_dir_for(plan, stage, f"baseline/seed{seed}"))
            st = [e for e in ev if e["kind"] == "sleep_started"][0]
            fi = [e for e in ev if e["kind"] == "sleep_finished"][0]
            spans[(stage, seed)] = (st["t0"], fi["t1"], st["gpu"])
    a_b = [spans[k] for k in spans if k[0] in ("a", "b")]
    latest_start = max(s[0] for s in a_b)
    earliest_end = min(s[1] for s in a_b)
    assert latest_start < earliest_end, "stages a and b (independent) did not overlap"
    assert {s[2] for s in a_b} == {"0", "1"}, "jobs were not spread over both GPUs"
    # c starts only after every a job finished (a must be validated first)
    assert spans[("c", 0)][0] >= max(spans[("a", s)][1] for s in (0, 1))
    # the exclusive stage runs alone: no other job's interval intersects it
    x0, x1, _ = spans[("x", 0)]
    assert all(s[1] <= x0 + 1e-3 or s[0] >= x1 - 1e-3 for k, s in spans.items() if k[0] != "x")
    man = json.loads((plan.campaign_root / "stages" / "x" / "STAGE_MANIFEST.json").read_text())
    assert man["ok"]


@pytest.mark.timeout(600)
def test_sequential_default_when_no_depends_on(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    ok = {"kind": "nan_metric", "emulate": "language_c4", "final": {"validation_nll": 3.0, "perplexity": 20.0, "token_ece": 0.05}, "sleep_s": 2}
    plan_path = _plan(tmp_path, [
        {"name": "first", "type": "fault_injection", "parallel": 2, "matrix": {"seed": [0, 1], "method": ["baseline"]}, "job": {"fault": ok}},
        {"name": "second", "type": "fault_injection", "parallel": 2, "matrix": {"seed": [0], "method": ["baseline"]}, "job": {"fault": ok}},
    ])
    assert run_campaign(str(plan_path), report=False, poll_s=0.5) == 0
    plan = Plan.load(plan_path)
    first_end = max([e for e in _events(job_dir_for(plan, "first", f"baseline/seed{s}")) if e["kind"] == "sleep_finished"][0]["t1"] for s in (0, 1))
    second_start = [e for e in _events(job_dir_for(plan, "second", "baseline/seed0")) if e["kind"] == "sleep_started"][0]["t0"]
    assert second_start >= first_end
