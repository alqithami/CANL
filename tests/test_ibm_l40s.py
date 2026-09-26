from __future__ import annotations

from pathlib import Path

import yaml

from caenl.plan import Plan
from caenl.validate import validate_plan


ROOT = Path(__file__).resolve().parents[1]
PLAN_DIR = ROOT / "configs" / "plans"
IBM_PLANS = sorted(PLAN_DIR.glob("ibm_l40s_*.yaml"))


def load(name: str) -> Plan:
    return Plan.load(PLAN_DIR / name)


def jobs_by_stage(plan: Plan) -> dict[str, list[dict]]:
    return {stage["name"]: jobs for stage, jobs in plan.expand()}


def test_all_ibm_plans_validate_and_are_single_gpu_sequential() -> None:
    assert IBM_PLANS
    for path in IBM_PLANS:
        plan = Plan.load(path)
        report = validate_plan(plan)
        assert report["valid"], (path.name, report["errors"])
        hw = plan.raw.get("hardware", {})
        assert hw.get("expected_gpu_name_regex") == "L40S"
        assert float(hw.get("min_gpu_memory_gb", 0)) >= 44
        assert int(hw.get("min_nvidia_driver", 0)) >= 550
        assert str(hw.get("torch_cuda")) == "12.4"
        for stage in plan.stages:
            if stage.get("cpu_only"):
                continue
            assert int(stage.get("parallel", 1)) == 1, (path.name, stage["name"])


def test_ibm_smoke_exercises_every_controller_and_task_family() -> None:
    plan = load("ibm_l40s_smoke_core.yaml")
    by_stage = jobs_by_stage(plan)
    wanted = {"baseline", "fixed_dcr", "macc_lite", "full_macc"}
    for name in ("language", "diffusion", "audio_captioning", "audio_retrieval"):
        got = {(j.get("method") or {}).get("name") for j in by_stage[name]}
        assert got == wanted, (name, got)
    vision = {(j.get("method") or {}).get("name") for j in by_stage["vision_al"]}
    assert {"caenl_acq_only", "caenl_fixed_dcr", "caenl_macc_lite", "caenl_full_macc"} <= vision
    resume_stage = next(s for s in plan.stages if s["name"] == "vision_resume")
    assert resume_stage.get("interrupt", {}).get("verify", {}).get("full_macc_acquisition_updates") is True
    assert by_stage["autoattack_standard"][0]["robustness"]["final"]["version"] == "standard"


def test_ibm_resnet_plan_is_principal_imagenet_replication() -> None:
    plan = load("ibm_l40s_imagenet_r50.yaml")
    by_stage = jobs_by_stage(plan)
    al = by_stage["r50_al"]
    seeds = {int(j["seed"]) for j in al}
    methods = {(j.get("method") or {}).get("name") for j in al}
    assert seeds == set(range(6))
    assert {
        "random", "entropy", "coreset", "badge", "noise_stability",
        "caenl_acq_only", "dcr_fixed_random", "dcr_lite_random",
        "caenl_fixed_dcr", "caenl_macc_lite", "caenl_full_macc",
    } <= methods
    first = al[0]
    al_cfg = first["active_learning"]
    assert al_cfg["initial_fraction"] == 0.10
    assert al_cfg["rounds"] == 5
    assert al_cfg["fraction_per_round"] == 0.10
    assert al_cfg["candidate_size"] == 50_000
    assert first["training"]["epochs_per_round"] == 20
    assert int(first["training"]["batch_size"]) <= 128
    assert int(first["robustness"]["final"]["batch_size"]) <= 16
    assert {(j.get("method") or {}).get("name") for j in by_stage["r50_supervised"]} == {"supervised_full"}


def test_ibm_vit_plan_has_mae_and_scratch_confirmation() -> None:
    plan = load("ibm_l40s_imagenet_vit.yaml")
    by_stage = jobs_by_stage(plan)
    for stage, arch in (("mae_al", "vit_b_16_mae"), ("scratch_al", "vit_b_16")):
        jobs = by_stage[stage]
        assert {j["arch"] for j in jobs} == {arch}
        assert {int(j["seed"]) for j in jobs} == {0, 1, 2}
        assert {"caenl_fixed_dcr", "caenl_macc_lite", "caenl_full_macc"} <= {
            (j.get("method") or {}).get("name") for j in jobs
        }
        assert max(int(j["training"]["batch_size"]) for j in jobs) <= 64
        assert max(int(j["robustness"]["final"]["batch_size"]) for j in jobs) <= 8


def test_ibm_crossmodal_plans_keep_all_four_controller_conditions() -> None:
    specs = {
        "ibm_l40s_c4.yaml": ("c4", "language_c4"),
        "ibm_l40s_diffusion.yaml": ("diffusion", "diffusion_ddpm"),
        "ibm_l40s_audiocaps.yaml": ("captioning", "audio_captioning"),
        "ibm_l40s_clotho.yaml": ("captioning_clotho", "audio_captioning"),
    }
    wanted = {"baseline", "fixed_dcr", "macc_lite", "full_macc"}
    for filename, (stage_name, task_type) in specs.items():
        plan = load(filename)
        stage_jobs = jobs_by_stage(plan)[stage_name]
        assert {j["type"] for j in stage_jobs} == {task_type}
        assert {int(j["seed"]) for j in stage_jobs} == {0, 1, 2}
        assert {(j.get("method") or {}).get("name") for j in stage_jobs} == wanted


def test_audiocaps_is_primary_and_clotho_is_supplementary() -> None:
    ac = load("ibm_l40s_audiocaps.yaml")
    for name in ("captioning", "retrieval"):
        for j in jobs_by_stage(ac)[name]:
            assert j["audio"]["dataset"] == "audiocaps"
            assert str(j["audio"]["manifest"]).endswith("/audiocaps/manifest.jsonl")
    cl = load("ibm_l40s_clotho.yaml")
    for name in ("captioning_clotho", "retrieval_clotho"):
        assert {j["audio"]["dataset"] for j in jobs_by_stage(cl)[name]} == {"clotho"}


def test_ibm_scripts_refuse_unmounted_data_root() -> None:
    env = (ROOT / "scripts" / "ibm_env.sh").read_text()
    bootstrap = (ROOT / "scripts" / "bootstrap_ibm_l40s.sh").read_text()
    assert "CAENL_REQUIRE_MOUNT" in env
    assert "findmnt" in env
    assert "CAENL_REQUIRE_MOUNT=1" in bootstrap
    assert "--index-url https://download.pytorch.org/whl/cu124" in bootstrap


def test_version_and_pytorch_contract_are_consistent() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text()
    init = (ROOT / "src" / "caenl" / "__init__.py").read_text()
    docker = (ROOT / "docker" / "Dockerfile").read_text()
    assert 'version = "5.4.3"' in pyproject
    assert '__version__ = "5.4.3"' in init
    assert "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime" in docker
