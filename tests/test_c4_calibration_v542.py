from pathlib import Path
import json

import pytest
import torch

from caenl.core.controllers import FullMACC, MACCLite, build_controller
from caenl.plan import Plan

ROOT = Path(__file__).resolve().parents[1]


def test_macc_lite_eta_lambda_takes_precedence() -> None:
    controller = build_controller(
        "macc_lite",
        ["h6"],
        {"h6": 16.0},
        {
            "macc_lite": {
                "eta_lambda": 0.002,
                "controller_step_size": 0.02,
                "lambda_min": 0.0,
                "lambda_max": 0.08,
                "error_mode": "difference",
            }
        },
        total_controller_steps=10,
        device="cpu",
        seed=1,
        lambda_value=0.005,
        quantile_head_enabled=False,
    )
    assert isinstance(controller, MACCLite)
    assert controller.step_size == pytest.approx(0.002)
    action = controller.step({"h6": 17.0}, step_index=1)
    assert action.lambdas["h6"] == pytest.approx(0.007)


def test_full_macc_reward_components_sum_exactly() -> None:
    controller = FullMACC(
        ["h6"],
        {"h6": 16.0},
        state_dim=FullMACC.state_dim_for(1, 1),
        reward_value_scale=5.0,
        reward_query_beta=0.1,
        reward_target_gamma=0.005,
        quantile_head_enabled=False,
        device="cpu",
    )
    c = controller.reward_components(0.01, 2, {"h6": 20.0})
    assert c["reward"] == pytest.approx(
        c["reward_value_term"] + c["reward_query_term"] + c["reward_target_term"]
    )
    assert c["target_deviation"] > 0


def test_c4_calibration_plan_expands_per_layer_targets() -> None:
    plan = Plan.load(ROOT / "configs/plans/one_c4_calibration_v2.yaml")
    stages = {s["name"]: jobs for s, jobs in plan.expand()}
    jobs = stages["c4_calibration"]
    assert len(jobs) == 12
    macc = next(j for j in jobs if j["method"]["name"] == "macc_lite_r16_4")
    assert macc["language"]["configured_rank"] == {"h6": 16, "h12": 4}
    assert macc["language"]["target_effective_rank"] == {"h6": 16, "h12": 4}
    assert macc["macc_lite"]["eta_lambda"] == pytest.approx(0.002)
    assert Path(macc["paths"]["results_root"]) == Path("/mnt/caenl/active/results")


def test_crossmodal_validator_reads_nested_policy_updates() -> None:
    text = (ROOT / "src/caenl/publication.py").read_text(encoding="utf-8")
    assert 'isinstance(h.get("update"), dict)' in text
    assert '"policy_loss" in h["update"]' in text


def test_storage_env_forces_derived_roots_from_active_root() -> None:
    text = (ROOT / "scripts/ibm_one_dataset_env.sh").read_text(encoding="utf-8")
    assert 'export CAENL_DATA_ROOT="$CAENL_ACTIVE_ROOT/data"' in text
    assert 'export CAENL_RESULTS_ROOT="$CAENL_ACTIVE_ROOT/results"' in text
    assert 'CAENL_DATA_ROOT="${CAENL_DATA_ROOT:-' not in text


def test_freeze_plan_writes_real_sha256_and_separate_fingerprint(tmp_path, monkeypatch) -> None:
    import hashlib
    import yaml
    from caenl.runner import freeze_plan

    protocol = ROOT / "configs/protocol/revision_protocol.yaml"
    plan_path = tmp_path / "plan.yaml"
    results_root = tmp_path / "results"
    plan_path.write_text(
        yaml.safe_dump(
            {
                "campaign_id": "hash-test",
                "campaign_role": "pilot",
                "results_root": str(results_root),
                "protocol": str(protocol),
                "stages": [],
            }
        ),
        encoding="utf-8",
    )
    plan = Plan.load(plan_path)
    root = freeze_plan(plan)
    plan_bytes = (root / "PLAN.yaml").read_bytes()
    expected = hashlib.sha256(plan_bytes).hexdigest()
    assert (root / "PLAN.sha256").read_text().split()[0] == expected
    assert len((root / "PLAN.fingerprint").read_text().split()[0]) == 16
    frozen = root / "frozen_protocol.yaml"
    expected_frozen = hashlib.sha256(frozen.read_bytes()).hexdigest()
    assert (root / "frozen_protocol.sha256").read_text().split()[0] == expected_frozen
    assert (root / "frozen_protocol.sha256").read_text().split()[1] == "frozen_protocol.yaml"
    source = root / "SOURCE_SNAPSHOT.json"
    source_hash = root / "SOURCE_SNAPSHOT.sha256"
    assert source.exists() and source_hash.exists()
    assert source_hash.read_text().split()[0] == hashlib.sha256(source.read_bytes()).hexdigest()
    source_meta = json.loads(source.read_text())
    assert len(source_meta["tree_sha256"]) == 64
    assert source_meta["file_count"] > 50
