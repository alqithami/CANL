#!/usr/bin/env python
"""Derive the two-pod split plans from configs/plans/franklin_complete_8gpu.yaml.

Pod A (ResNet-50 + supporting + cross-modal) and pod B (ViT-B/16 MAE + from scratch) run the same
frozen protocol on two 8-GPU pods; their results are merged with
``caenl report --root <pod A root> --extra-roots <pod B root> --strict``.

    python scripts/make_pod_plans.py          # rewrites configs/plans/franklin_pod_{a_resnet,b_vit}.yaml
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "configs/plans/franklin_complete_8gpu.yaml"

POD_B_STAGES = {"prepare_vision", "prepare_models", "aa_sanity", "overhead_imagenet1k", "vit_mae_init", "vit_mae_al", "vit_mae_supervised", "vit_scratch_init", "vit_scratch_al", "vit_scratch_supervised", "aggregate"}
POD_A_EXCLUDE = {"vit_mae_init", "vit_mae_al", "vit_mae_supervised", "vit_scratch_init", "vit_scratch_al", "vit_scratch_supervised"}

HEADER_A = """# Two-pod split, POD A: ImageNet-1K / ResNet-50 (six seeds, 14 arms + full supervision), AutoAttack sanity,
# overhead (ResNet-50 + CIFAR), ImageNet-100, CIFAR-10/100, C4, DDPM, AudioCaps + Clotho.  ~490 H100-h ~ 2.6 days on 8 GPUs.
# Generated from franklin_complete_8gpu.yaml by scripts/make_pod_plans.py — edit that file, then regenerate.
# Run:   caenl run --plan configs/plans/franklin_pod_a_resnet.yaml --gpus 0,1,2,3,4,5,6,7
# Merge: caenl report --root $CAENL_RESULTS_ROOT/franklin-pod-a --extra-roots <pod B results>/franklin-pod-b --strict
"""
HEADER_B = """# Two-pod split, POD B: ImageNet-1K / ViT-B/16 with label-free MAE initialisation and from scratch (three seeds each,
# 6 arms + full supervision), ViT overhead benchmark, AutoAttack sanity.  ~490 H100-h ~ 2.6 days on 8 GPUs.
# Generated from franklin_complete_8gpu.yaml by scripts/make_pod_plans.py — edit that file, then regenerate.
# Needs its own copy of ImageNet-1K (scripts/get_imagenet_kaggle.sh) and the MAE weights (preflight --download).
# Run:   caenl run --plan configs/plans/franklin_pod_b_vit.yaml --gpus 0,1,2,3,4,5,6,7
"""


def _filter(plan: dict, keep) -> dict:
    """Return an independent copy of ``plan`` restricted to the stages accepted by ``keep``.

    The stages are deep-copied individually so that the per-pod edits in :func:`main` (dropping
    ``mae_weights`` from pod A, keeping only ImageNet-1K in pod B, ...) never leak into the other
    pod's plan through shared dict objects.
    """
    out = copy.deepcopy({k: v for k, v in plan.items() if k != "stages"})
    out["stages"] = [copy.deepcopy(s) for s in plan["stages"] if keep(s["name"])]
    names = {s["name"] for s in out["stages"]}
    for s in out["stages"]:
        if "depends_on" in s:
            s["depends_on"] = [d for d in s["depends_on"] if d in names]
    return out


def _check(plan: dict, label: str) -> None:
    """Fail loudly if a matrix list is empty (the aliasing bug this guards against produced plans
    whose preparation stages expanded to zero jobs)."""
    for s in plan["stages"]:
        for key, val in (s.get("matrix") or {}).items():
            if isinstance(val, list) and not val:
                raise SystemExit(f"{label}: stage {s['name']!r} has an empty matrix list {key!r}")


def main() -> None:
    plan = yaml.safe_load(SRC.read_text())
    a = _filter(plan, lambda n: n not in POD_A_EXCLUDE)
    a["campaign_id"] = "franklin-pod-a"
    for s in a["stages"]:
        if s["name"] == "overhead_imagenet1k":
            s["matrix"]["arch"] = ["resnet50"]
        if s["name"] == "prepare_models":
            s["matrix"]["dataset_spec"] = [d for d in s["matrix"]["dataset_spec"] if d["name"] != "mae_weights"]
    b = _filter(plan, lambda n: n in POD_B_STAGES)
    b["campaign_id"] = "franklin-pod-b"
    for s in b["stages"]:
        if s["name"] == "overhead_imagenet1k":
            s["matrix"]["arch"] = ["vit_b_16", "vit_b_16_mae"]
        if s["name"] == "prepare_vision":
            s["matrix"]["dataset_spec"] = [d for d in s["matrix"]["dataset_spec"] if d["name"] == "imagenet1k"]
        if s["name"] == "prepare_models":
            s["matrix"]["dataset_spec"] = [d for d in s["matrix"]["dataset_spec"] if d["name"] == "mae_weights"]
    _check(a, "pod A")
    _check(b, "pod B")
    for name, header, data in (("franklin_pod_a_resnet.yaml", HEADER_A, a), ("franklin_pod_b_vit.yaml", HEADER_B, b)):
        body = yaml.safe_dump(data, sort_keys=False, width=200, default_flow_style=None)
        (ROOT / "configs/plans" / name).write_text(header + body)
        print(f"wrote configs/plans/{name}: {len(data['stages'])} stages")


if __name__ == "__main__":
    main()
