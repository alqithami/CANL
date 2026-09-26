from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from caenl.audio.metrics import caption_metrics
from caenl.plan import Plan
from caenl.report.aggregate import grouped_stats, paired_significance
from caenl.vision.task import shared_initial_key

ROOT = Path(__file__).resolve().parents[1]


def test_full_plan_expands_and_shares_initial_checkpoints():
    plan = Plan.load(ROOT / "configs/plans/supporting_1gpu.yaml")
    init_keys, al = {}, []
    for stage, jobs in plan.expand():
        assert jobs, f"stage {stage['name']} expands to no jobs"
        for j in jobs:
            assert "job_id" in j and j["type"] == stage["type"]
            if j["type"] == "vision_al":
                k = shared_initial_key(j)
                if j.get("mode") == "initial_only":
                    init_keys[k] = j["job_id"]
                elif not j["method"].get("full_supervision") and not j["method"].get("adversarial_training"):
                    al.append(k)
    assert all(k in init_keys for k in al)


def test_smoke_plan_valid():
    from caenl.validate import validate_plan

    plan = Plan.load(ROOT / "configs/plans/smoke_cpu.yaml")
    rep = validate_plan(plan)
    assert rep["valid"], rep["errors"]


def test_method_resolution_defaults():
    plan = Plan.load(ROOT / "configs/plans/smoke_cpu.yaml")
    spec = plan.resolve_method("caenl_macc_lite")
    assert spec["acquisition"] == "collapse" and spec["regularizer"] == "macc_lite" and spec["dcr_lambda"] == 0.03
    assert plan.resolve_method("random")["regularizer"] == "none"


def test_grouped_and_significance():
    rows = []
    for m, base, slope in (("a", 0.8, 0.01), ("b", 0.7, 0.012)):
        for s in range(5):
            rows.append({"dataset": "d", "arch": "x", "method": m, "seed": s, "round": 2, "label_fraction": 0.6, "metric": "test_acc", "value": base + slope * s})
    df = pd.DataFrame(rows)
    g = grouped_stats(df, ["dataset", "arch", "method", "round", "label_fraction", "metric"])
    assert len(g) == 2 and g[g["method"] == "a"]["n"].iloc[0] == 5
    sig = paired_significance(df, [{"target": "a", "references": ["b"]}], ["test_acc"], ["dataset", "arch"], 200, 200)
    assert len(sig) == 1 and abs(sig.iloc[0]["boot_diff_mean"] - 0.096) < 1e-9 and sig.iloc[0]["p_ttest_holm"] < 0.05


def test_caption_metrics_monotone():
    refs = [["a dog barks in the yard"] * 5, ["water runs from a tap"] * 5]
    good = caption_metrics(["a dog barks in the yard", "water runs from a tap"], refs, java=False, official=False)
    bad = caption_metrics(["birds sing", "engine"], refs, java=False, official=False)
    assert good["bleu4"] > bad["bleu4"] and good["rouge_l"] > bad["rouge_l"] and good["cider_d"] >= bad["cider_d"]
    assert good["metrics_source"] == "native"


def test_native_caption_metrics_match_pycocoevalcap():
    """Official coco-caption values are authoritative; the native fallbacks must agree with them."""
    import random

    pytest.importorskip("pycocoevalcap")
    rng = random.Random(0)
    vocab = "a the dog cat barks meows runs water tap faucet running drips loudly softly engine hums bird chirps wind blows man woman speaks".split()

    def sent():
        return " ".join(rng.choice(vocab) for _ in range(rng.randint(4, 9)))

    refs = [[sent() for _ in range(5)] for _ in range(40)]
    cands = [rng.choice(r) if rng.random() < 0.3 else sent() for r in refs]
    m = caption_metrics(cands, refs, java=False, official=True)
    assert m["metrics_source"] == "pycocoevalcap"
    for k in ("cider_d", "bleu4", "rouge_l"):
        assert m[f"discrepancy_{k}"] < (0.02 if k != "cider_d" else 0.2), (k, m[k], m[f"native_{k}"])
    assert m["native_within_tolerance"]


def test_required_caption_metric_failure_is_fatal(monkeypatch):
    import shutil as _sh

    from caenl.audio.metrics import CaptionMetricError

    pytest.importorskip("pycocoevalcap")
    monkeypatch.setattr(_sh, "which", lambda name: None)  # no Java runtime
    refs = [["a dog barks in the yard"] * 5]
    with pytest.raises(CaptionMetricError):
        caption_metrics(["a dog barks"], refs, java=True, official=True)


def test_imagenet1k_production_plan_matches_the_review_requirements():
    plan = Plan.load(ROOT / "configs/plans/franklin_imagenet1k_revision.yaml")
    stages = {s["name"]: jobs for s, jobs in plan.expand()}
    assert list(stages) == ["prepare", "aa_sanity", "overhead_imagenet1k", "r50_init", "r50_al", "r50_supervised", "vit_init", "vit_al", "vit_supervised", "aggregate"]
    assert stages["aa_sanity"][0]["sanity"]["pretrained"] and stages["aa_sanity"][0]["robustness"]["final"]["n_examples"] == 5000
    al = stages["r50_al"]
    assert len(al) == 60 and sorted({j["seed"] for j in al}) == [0, 1, 2, 3, 4, 5]
    methods = {j["method"]["name"] for j in al}
    assert {"random", "entropy", "coreset", "badge", "noise_stability", "caenl_acq_only", "dcr_fixed_random", "caenl_fixed_dcr", "caenl_macc_lite", "caenl_full_macc"} == methods
    for j in al + stages["r50_supervised"] + stages["vit_al"]:
        assert j["dataset_options"]["train_resolution"] == 224 and j["dataset_options"]["test_resolution"] == 224 and j["dataset_options"]["cache_resolution"] == 256
        assert j["training"]["epochs_per_round"] == 20 and j["training"]["initial_epochs"] == 60 and j["active_learning"]["rounds"] == 5
        assert j["active_learning"]["initial_fraction"] == 0.1 and j["active_learning"]["fraction_per_round"] == 0.1
        assert j["robustness"]["final"]["enabled"] and j["robustness"]["final"]["version"] == "standard" and j["robustness"]["final"]["n_examples"] == 5000
    assert all(j["training"]["full_supervision_epochs"] == 90 for j in stages["r50_supervised"])
    # every AL job finds its shared initial checkpoint in the init stage of the same architecture
    init_keys = {shared_initial_key(j) for j in stages["r50_init"] + stages["vit_init"]}
    for j in al + stages["vit_al"]:
        assert shared_initial_key(j) in init_keys, j["job_id"]
    vit = stages["vit_al"]
    assert len(vit) == 18 and sorted({j["seed"] for j in vit}) == [0, 1, 2]
    assert all(j["training"]["optimizer"] == "adamw" for j in vit)
    # one job per GPU by default; timing stage is exclusive
    assert all(int(s.get("parallel", 1)) == 1 for s in plan.stages)
    assert any(s.get("exclusive") for s in plan.stages if s["type"] == "vision_overhead")


def test_audiocaps_manifest_validator_is_strict(tmp_path):
    import hashlib
    import json

    from caenl.audio.audiocaps import validate_audio_manifest

    def clip(name, data=b"RIFFfake"):
        p = tmp_path / name
        p.write_bytes(data)
        return p

    def row(i, split, path, caps=("a dog barks",), ytid=None):
        return {"id": f"{split}/{i}", "youtube_id": ytid or f"yt{i}", "start_time": 0, "split": split, "path": str(path), "captions": list(caps), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def write(rows, name="m.jsonl"):
        p = tmp_path / name
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return p

    paths = [clip(f"c{i}.wav", b"RIFF" + bytes([i])) for i in range(6)]
    good = [row(0, "train", paths[0]), row(1, "train", paths[1]), row(2, "valid", paths[2], caps=["a"] * 5), row(3, "test", paths[3], caps=["b"] * 5)]
    rep = validate_audio_manifest(write(good), verify_hashes="all")
    assert rep["valid"], rep["errors"]
    assert rep["splits"] == {"train": 2, "valid": 1, "test": 1} and rep["manifest_sha256"]
    # coverage gate against the official split sizes
    assert not validate_audio_manifest(write(good), min_coverage=0.9)["valid"]
    # missing file
    bad = good + [dict(row(4, "train", paths[4]), path=str(tmp_path / "nope.wav"))]
    assert any("missing" in e for e in validate_audio_manifest(write(bad))["errors"])
    # clip crossing splits (same youtube id)
    bad = good + [row(5, "test", paths[5], caps=["x"] * 5, ytid="yt0")]
    assert any("more than one split" in e for e in validate_audio_manifest(write(bad))["errors"])
    # empty split
    assert any("empty" in e for e in validate_audio_manifest(write([r for r in good if r["split"] != "test"]))["errors"])
    # missing caption
    bad = good + [row(5, "train", paths[5], caps=[" "])]
    assert any("no caption" in e for e in validate_audio_manifest(write(bad))["errors"])
    # changed audio (hash mismatch)
    paths[1].write_bytes(b"RIFFchanged")
    rep = validate_audio_manifest(write(good), verify_hashes="all")
    assert not rep["valid"] and any("changed" in e for e in rep["errors"])
    # missing manifest
    assert not validate_audio_manifest(tmp_path / "absent.jsonl")["valid"]


def test_publication_check_job_rejects_incomplete_or_nonfinite_results(tmp_path):
    import json
    import math

    import numpy as np

    from caenl.publication import check_job

    cfg = {
        "job_id": "cifar10/resnet18_cifar/caenl_macc_lite/seed0",
        "type": "vision_al",
        "method": {"name": "caenl_macc_lite"},
        "active_learning": {"rounds": 2, "initial_fraction": 0.1, "fraction_per_round": 0.1},
        "robustness": {"eps": 8 / 255, "norm": "Linf", "final": {"enabled": True, "version": "standard", "n_examples": 500}, "intermediate": {"enabled": True}, "corruptions": {"enabled": True}},
    }

    def write(job_dir, summary, state="succeeded", artifacts=True):
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "status.json").write_text(json.dumps({"state": state}))
        (job_dir / "summary.json").write_text(json.dumps(summary))
        if artifacts:
            (job_dir / "artifacts").mkdir(exist_ok=True)
            for name in ("aa_robust_mask_round2.npy", "aa_clean_mask_round2.npy", "aa_subset_round2.npy"):
                np.save(job_dir / "artifacts" / name, np.zeros(500, dtype=bool))

    aa = {"evaluator": "autoattack-standard", "eps": 8 / 255, "norm": "Linf", "n": 500, "robust_acc": 0.01, "clean_acc": 0.9}
    rounds = [{"round": r, "label_fraction": 0.1 + 0.1 * r, "test_acc": 0.8} for r in range(3)]
    rounds[-1]["autoattack"] = aa
    good = {"job_id": cfg["job_id"], "n_test": 10000, "rounds": rounds, "labels": {"labels_available": 15000, "labels_trained": 14500}, "final": {"test_acc": 0.9, "test_loss": 0.4, "test_ece": 0.02, "apgd_ce_subset_acc": 0.02, "pgd_subset_acc": 0.05, "corruption_mean_acc": 0.7, "aa_robust_acc": 0.01, "aa_clean_acc": 0.9}}
    write(tmp_path / "good", good)
    rep = check_job(tmp_path / "good", cfg)
    assert rep["ok"], rep["errors"]
    assert rep["summary_sha256"] and rep["metrics"]["aa_robust_acc"] == 0.01

    bad = json.loads(json.dumps(good))
    bad["final"]["aa_robust_acc"] = None
    write(tmp_path / "nan", bad)
    assert any("aa_robust_acc" in e for e in check_job(tmp_path / "nan", cfg)["errors"])

    bad = json.loads(json.dumps(good))
    bad["final"]["test_ece"] = float("nan")
    write(tmp_path / "nan2", bad)
    assert any("test_ece" in e for e in check_job(tmp_path / "nan2", cfg)["errors"])

    bad = json.loads(json.dumps(good))
    bad["rounds"][-1]["autoattack"]["evaluator"] = "autoattack-custom"
    write(tmp_path / "ver", bad)
    assert any("version" in e for e in check_job(tmp_path / "ver", cfg)["errors"])

    bad = json.loads(json.dumps(good))
    bad["rounds"][-1]["autoattack"]["n"] = 100
    write(tmp_path / "n", bad)
    assert any("examples" in e for e in check_job(tmp_path / "n", cfg)["errors"])

    bad = json.loads(json.dumps(good))
    bad["rounds"] = bad["rounds"][:2]
    write(tmp_path / "rounds", bad)
    assert any("round records" in e for e in check_job(tmp_path / "rounds", cfg)["errors"])

    write(tmp_path / "running", good, state="running")
    assert any("succeeded" in e for e in check_job(tmp_path / "running", cfg)["errors"])

    write(tmp_path / "noart", good, artifacts=False)
    assert any("artifact" in e for e in check_job(tmp_path / "noart", cfg)["errors"])

    cap_cfg = {"job_id": "baseline/seed0", "type": "audio_captioning", "audio": {"dataset": "audiocaps", "captioning": {"java_metrics": True, "spice": True, "official_metrics": True, "metrics": ["cider_d", "bleu4", "rouge_l", "meteor", "spice", "validation_loss"]}}}
    cap = {"job_id": "baseline/seed0", "metrics_source": "pycocoevalcap", "manifest_provenance": {"manifest_sha256": "ab"}, "final": {"cider_d": 0.3, "bleu4": 0.1, "rouge_l": 0.3, "meteor": 0.1, "spice": 0.1, "val_loss": 2.0}}
    write(tmp_path / "cap", cap, artifacts=False)
    assert check_job(tmp_path / "cap", cap_cfg)["ok"]
    cap2 = json.loads(json.dumps(cap))
    del cap2["final"]["spice"]
    write(tmp_path / "cap2", cap2, artifacts=False)
    assert any("spice" in e for e in check_job(tmp_path / "cap2", cap_cfg)["errors"])
    cap3 = json.loads(json.dumps(cap))
    cap3["metrics_source"] = "native"
    write(tmp_path / "cap3", cap3, artifacts=False)
    assert any("pycocoevalcap" in e for e in check_job(tmp_path / "cap3", cap_cfg)["errors"])


def test_plan_defaults_override_protocol_dataset_overrides():
    """Plan-level settings must win over the protocol's per-dataset defaults (smoke plans shorten schedules)."""
    plan = Plan.load(ROOT / "configs/plans/smoke_gpu.yaml")
    stages = {s["name"]: jobs for s, jobs in plan.expand()}
    j = stages["cifar10_al"][0]
    assert j["training"]["initial_epochs"] == 5 and j["training"]["epochs_per_round"] == 2 and j["active_learning"]["rounds"] == 2
    assert j["collapse"]["monitoring_cadence_steps"] == 20 and j["robustness"]["final"]["n_examples"] == 500
    # the stage job block wins over the plan defaults
    k = stages["imagefolder_r50"][0]
    assert k["training"]["initial_epochs"] == 2 and k["dataset_options"]["force_memmap"] is True and k["dataset_options"]["train_resolution"] == 224
    # protocol dataset overrides still apply where the plan is silent
    prod = Plan.load(ROOT / "configs/plans/franklin_imagenet1k_revision.yaml")
    r = {s["name"]: jobs for s, jobs in prod.expand()}["r50_al"][0]
    assert r["dataset_options"]["force_memmap"] is True and r["training"]["initial_epochs"] == 60 and r["reproducibility"]["max_device_data_fraction"] == 0.2


def test_complete_8gpu_plan_and_pod_split_are_consistent():
    """The one-pod complete plan carries the full ImageNet-1K matrix (14 arms x 6 seeds, ViT MAE + scratch) and every
    plan validates; the two generated pod plans partition its stages without losing any preparation job."""
    import subprocess
    import sys

    import yaml

    from caenl.validate import validate_plan

    full = Plan.load(ROOT / "configs/plans/franklin_complete_8gpu.yaml")
    rep = validate_plan(full)
    assert rep["valid"], rep["errors"]
    stages = {s["name"]: jobs for s, jobs in full.expand()}
    al = stages["r50_al"]
    assert len(al) == 84 and sorted({j["seed"] for j in al}) == [0, 1, 2, 3, 4, 5]
    methods = {j["method"]["name"] for j in al}
    assert {"random", "entropy", "margin", "power_margin", "bald_mcd", "coreset", "badge", "noise_stability", "caenl_acq_only", "dcr_fixed_random", "caenl_fixed_dcr", "caenl_macc_lite", "caenl_full_macc", "dcr_lite_random"} <= methods
    assert {j["arch"] for j in stages["vit_mae_al"]} == {"vit_b_16_mae"} and {j["arch"] for j in stages["vit_scratch_al"]} == {"vit_b_16"}
    assert all(j["arch_options"]["init"] == "mae" and j["training"].get("layer_decay") for j in stages["vit_mae_al"])
    assert {d["name"] for d in next(s for s in full.stages if s["name"] == "prepare_models")["matrix"]["dataset_spec"]} == {"mae_weights", "c4", "ddpm_cifar10", "clotho"}
    # regenerate the pod plans into a temporary copy and compare with the committed files
    out = subprocess.run([sys.executable, str(ROOT / "scripts/make_pod_plans.py")], capture_output=True, text=True, cwd=str(ROOT))
    assert out.returncode == 0, out.stderr
    a = Plan.load(ROOT / "configs/plans/franklin_pod_a_resnet.yaml")
    b = Plan.load(ROOT / "configs/plans/franklin_pod_b_vit.yaml")
    for p in (a, b):
        r = validate_plan(p)
        assert r["valid"], r["errors"]
        for st, jobs in p.expand():
            assert jobs, f"{p.campaign_id}: stage {st['name']} expands to no jobs"
    names_a = {s["name"] for s in a.stages}
    names_b = {s["name"] for s in b.stages}
    assert "vit_mae_al" not in names_a and "r50_al" not in names_b
    prep_a = {d["name"] for d in next(s for s in a.stages if s["name"] == "prepare_models")["matrix"]["dataset_spec"]}
    prep_b = {d["name"] for d in next(s for s in b.stages if s["name"] == "prepare_models")["matrix"]["dataset_spec"]}
    assert prep_a == {"c4", "ddpm_cifar10", "clotho"} and prep_b == {"mae_weights"}
    assert {d["name"] for d in next(s for s in b.stages if s["name"] == "prepare_vision")["matrix"]["dataset_spec"]} == {"imagenet1k"}
    # the complete plan's source file is untouched by the generator
    assert yaml.safe_load((ROOT / "configs/plans/franklin_complete_8gpu.yaml").read_text())["campaign_id"] == full.campaign_id


def test_validate_plan_rejects_empty_matrices_and_bad_parallelism(tmp_path):
    import yaml

    from caenl.validate import validate_plan

    raw = yaml.safe_load((ROOT / "configs/plans/smoke_cpu.yaml").read_text())
    raw["results_root"] = str(tmp_path)
    st = next(s for s in raw["stages"] if s["type"] == "vision_al")
    st["matrix"]["method"] = []
    st["parallel"] = 0
    other = next(s for s in raw["stages"] if s["type"] == "fault_injection")
    other["gpu_share"] = "two"
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw))
    rep = validate_plan(Plan.load(p))
    assert not rep["valid"]
    assert any("matrix list 'method' is empty" in e for e in rep["errors"])
    assert any("parallel must be >= 1" in e for e in rep["errors"])
    assert any("gpu_share must be an integer" in e for e in rep["errors"])


def test_budget_plan_is_the_minimum_defensible_subset_of_the_complete_plan():
    import subprocess
    import sys

    from caenl.validate import validate_plan

    out = subprocess.run([sys.executable, str(ROOT / "scripts/make_budget_plan.py")], capture_output=True, text=True, cwd=str(ROOT))
    assert out.returncode == 0, out.stderr
    plan = Plan.load(ROOT / "configs/plans/franklin_budget_8gpu.yaml")
    rep = validate_plan(plan)
    assert rep["valid"], rep["errors"]
    assert rep["jobs_total"] == 432
    stages = {s["name"]: jobs for s, jobs in plan.expand()}
    assert not any(n.startswith("vit_scratch") for n in stages)
    al = stages["r50_al"]
    assert len(al) == 30 and sorted({j["seed"] for j in al}) == [0, 1, 2]
    assert {j["method"]["name"] for j in al} == {"random", "entropy", "coreset", "badge", "noise_stability", "caenl_acq_only", "dcr_fixed_random", "caenl_fixed_dcr", "caenl_macc_lite", "caenl_full_macc"}
    vit = stages["vit_mae_al"]
    assert len(vit) == 12 and {j["method"]["name"] for j in vit} == {"random", "entropy", "caenl_macc_lite", "caenl_full_macc"}
    assert all(j["arch"] == "vit_b_16_mae" and j["arch_options"]["init"] == "mae" for j in vit)
    assert {j["arch"] for j in stages["overhead_imagenet1k"]} == {"resnet50", "vit_b_16_mae"}
    # the protocol is untouched: same schedule, resolution and AutoAttack subset as the complete plan
    full = {s["name"]: jobs for s, jobs in Plan.load(ROOT / "configs/plans/franklin_complete_8gpu.yaml").expand()}
    a, b = al[0], full["r50_al"][0]
    for key in ("training", "active_learning", "robustness", "dataset_options", "collapse"):
        assert a[key] == b[key], key
    # supporting experiments are unchanged
    for name in ("cifar10_al", "cifar100_al", "imagenet100_al", "c4", "diffusion", "audio_captioning", "audio_retrieval"):
        assert len(stages[name]) == len(full[name]), name
    # every AL job finds its shared initial checkpoint
    init_keys = {shared_initial_key(j) for n in ("r50_init", "vit_mae_init", "imagenet100_init", "cifar10_init", "cifar100_init", "cifar10_init_sweep") for j in stages[n]}
    for n in ("r50_al", "vit_mae_al", "imagenet100_al", "cifar10_al", "cifar100_al", "cifar10_sweep"):
        for j in stages[n]:
            assert shared_initial_key(j) in init_keys, j["job_id"]
