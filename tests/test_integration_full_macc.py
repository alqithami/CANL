"""End-to-end checks of Full MACC credit assignment and resume reproducibility (synthetic data, CPU)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _job_config(tmp: Path, job_id: str, seed: int = 0) -> dict:
    return {
        "job_id": job_id,
        "type": "vision_al",
        "dataset": "synthetic",
        "arch": "resnet18_cifar",
        "seed": seed,
        "dataset_options": {"num_classes": 6, "train_size": 900, "test_size": 120, "res": 32},
        "arch_options": {"width_mult": 0.125},
        "paths": {"data_root": str(tmp / "data"), "cache_root": str(tmp / "cache"), "shared_root": str(tmp / "shared")},
        "method": {"name": "caenl_full_macc", "acquisition": "collapse", "regularizer": "full_macc", "dcr_lambda": 0.03},
        "active_learning": {"initial_fraction": 0.2, "rounds": 2, "fraction_per_round": 0.1, "candidate_size": 200, "selected_fraction_per_pass": 0.1, "threshold_quantile": 0.9, "controller_validation_fraction": 0.1, "acquisition_batch_size": 128, "eval_batch_size": 128, "mahalanobis_mode": "diag", "layer_weights": "dim_normalized", "record_scores": False, "record_collapse_scores_for_baselines": False},
        "training": {"initial_epochs": 2, "epochs_per_round": 2, "batch_size": 32, "lr": 0.05, "amp": False, "channels_last": False, "log_every_steps": 10, "warmup_epochs": 0.5, "warmup_epochs_round": 0.5},
        "collapse": {"monitored_layers": ["layer3", "layer4"], "monitoring_cadence_steps": 4, "covariance_ema_decay": 0.9, "covariance_shrinkage_epsilon": 1e-5, "truncated_spectral_rank": 16, "dcr_covariance": "total", "target_collapse_strength": {"layer3": 0.35, "layer4": 0.7}},
        "full_macc": {"lambda_initial": 0.03, "lambda_max": 0.5, "hidden_dimensions": [16, 16], "exploration_epsilon_start": 0.0, "exploration_epsilon_end": 0.0, "policy_learning_rate": 0.01, "acquisition_reward_steps": 6, "reward": {"validation_delta_scale": 10.0, "query_penalty_beta": 0.001, "target_deviation_gamma": 0.1}, "action_space": {"lambda_delta_choices": [-0.02, -0.01, 0.0, 0.01, 0.02], "threshold_quantiles": [0.8, 0.85, 0.9, 0.95]}},
        "robustness": {"eps": 8 / 255, "evaluate_rounds": [], "final": {"enabled": False}, "intermediate": {"enabled": False}, "epsilon_sweep": {"enabled": False}, "corruptions": {"enabled": False}},
        "diagnostics": {"nc_layers": ["layer4"], "margins": False, "save_test_logits": False},
        "checkpointing": {"keep": "all"},
        "shared_initial": {"use": True, "save": True},
        "reproducibility": {"save_pip_freeze": False, "telemetry_interval_s": 5, "deterministic_algorithms": True, "cudnn_benchmark": False},
    }


def _run(job_dir: Path, env: dict | None = None) -> int:
    e = dict(os.environ)
    e.update(env or {})
    e.setdefault("CUDA_VISIBLE_DEVICES", "")
    return subprocess.call([sys.executable, "-m", "caenl.cli", "run-job", "--job-dir", str(job_dir)], env=e, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _write(job_dir: Path, cfg: dict) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "job.json").write_text(json.dumps(cfg))


@pytest.mark.timeout(1500)
def test_full_macc_rewards_acquisition_and_resumes_identically(tmp_path):
    straight = tmp_path / "straight"
    _write(straight, _job_config(tmp_path, "straight"))
    assert _run(straight) == 0, (straight / "log.txt").read_text()[-2000:] if (straight / "log.txt").exists() else "no log"
    traj = json.loads((straight / "artifacts" / "controller_trajectory.json").read_text())
    acq = [h for h in traj["history"] if h.get("kind") == "acquisition_update"]
    assert len(acq) == 2, "one acquisition update per round expected"
    for h in acq:
        assert h["queries"] > 0 and "value_before" in h and "value_after" in h
        assert abs(h["reward"] - (10.0 * h["value_delta"] - 0.001 * h["queries"])) < 1e-6
        assert "policy_loss" in h  # non-explored (epsilon = 0) -> the quantile head was updated
    summary = json.loads((straight / "summary.json").read_text())
    assert summary["rounds"][-1]["acquisition"]["controller_value_before"] is not None
    # metrics stream contains the acquisition updates
    kinds = {json.loads(l).get("kind") for l in (straight / "metrics.jsonl").read_text().splitlines() if l.strip()}
    assert "acquisition_update" in kinds

    # interrupted after round 1, then resumed: must reproduce round 2 exactly
    interrupted = tmp_path / "interrupted"
    _write(interrupted, _job_config(tmp_path, "interrupted"))
    rc = _run(interrupted, {"CAENL_DEBUG_STOP_AT_ROUND": "1"})
    assert rc != 0
    assert (interrupted / "checkpoints" / "round_1.pt").exists()
    assert _run(interrupted) == 0
    events = [json.loads(l) for l in (interrupted / "events.jsonl").read_text().splitlines() if l.strip()]
    assert any(e.get("kind") == "resumed" and e.get("from_round") == 1 for e in events)
    sel_a = np.load(straight / "artifacts" / "selected_round2.npy")
    sel_b = np.load(interrupted / "artifacts" / "selected_round2.npy")
    assert np.array_equal(sel_a, sel_b)
    sa = json.loads((straight / "summary.json").read_text())
    sb = json.loads((interrupted / "summary.json").read_text())
    assert sa["final"]["lambdas"] if "lambdas" in sa["final"] else True
    assert sa["rounds"][-1]["lambdas"] == sb["rounds"][-1]["lambdas"]
    assert sa["rounds"][-1]["tau"] == sb["rounds"][-1]["tau"]
    assert abs(sa["rounds"][-1]["test_acc"] - sb["rounds"][-1]["test_acc"]) < 1e-6
    tb = json.loads((interrupted / "artifacts" / "controller_trajectory.json").read_text())
    acq_b = [h for h in tb["history"] if h.get("kind") == "acquisition_update"]
    assert len(acq_b) == 2 and abs(acq_b[-1]["reward"] - acq[-1]["reward"]) < 1e-6
