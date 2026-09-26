#!/usr/bin/env python
"""Derive the *minimum-defensible* budget plan from configs/plans/franklin_complete_8gpu.yaml.

    python scripts/make_budget_plan.py      # rewrites configs/plans/franklin_budget_8gpu.yaml

The budget plan keeps every method family the pre-campaign review asked for and every supporting /
cross-modal experiment, and cuts only breadth that the manuscript can live without:

* ImageNet-1K / ResNet-50: **three** paired seeds (instead of six) and the ten reviewed arms
  (random, entropy, CoreSet, BADGE, Noise Stability, CAENL acq-only, random + fixed DCR,
  CAENL + fixed DCR, CAENL + MACC-Lite, CAENL + Full MACC) + full supervision — the four extra arms
  of the complete plan (Margin, Power-Margin, BALD-MC, random + MACC-Lite) stay in the CIFAR /
  ImageNet-100 supporting experiments only;
* ImageNet-1K / ViT-B/16: the label-free MAE-initialised programme only (no from-scratch programme),
  three paired seeds, four arms (random, entropy, CAENL + MACC-Lite, CAENL + Full MACC) + full
  supervision;
* overhead benchmark on ResNet-50 and the MAE ViT; everything else (AutoAttack sanity, CIFAR-10/100,
  ImageNet-100, C4, DDPM, AudioCaps + Clotho, strict aggregation) is unchanged.

~375 H100-hours  ~ 640 A100-hours: ~3.3 days on 8x A100 SXM (~US$1,000 at $1.59/GPU-h) or ~2 days on
8x H100 SXM (~US$1,250).  The same file runs on fewer GPUs (`--gpus 0,1`): only the wall-clock changes.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "configs/plans/franklin_complete_8gpu.yaml"
DST = ROOT / "configs/plans/franklin_budget_8gpu.yaml"

SEEDS = [0, 1, 2]
R50_ARMS = ["random", "entropy", "coreset", "badge", "noise_stability", "caenl_acq_only", "dcr_fixed_random", "caenl_fixed_dcr", "caenl_macc_lite", "caenl_full_macc"]
VIT_ARMS = ["random", "entropy", "caenl_macc_lite", "caenl_full_macc"]
DROP_STAGES = {"vit_scratch_init", "vit_scratch_al", "vit_scratch_supervised"}

HEADER = """# =============================================================================
# MINIMUM-DEFENSIBLE BUDGET CAMPAIGN for ONE 8-GPU POD / VM (8x A100 SXM 80 GB ~3.3 days ~US$1,000; 8x H100 ~2 days).
# Generated from franklin_complete_8gpu.yaml by scripts/make_budget_plan.py — edit that file, then regenerate.
#
# Versus the complete plan:
#   * ImageNet-1K / ResNet-50: THREE paired seeds, the ten reviewed arms + full supervision
#     (Margin, Power-Margin, BALD-MC and random+MACC-Lite remain in the CIFAR / ImageNet-100 supporting runs);
#   * ImageNet-1K / ViT-B/16: MAE-initialised programme only, three seeds, four arms + full supervision;
#   * everything else unchanged (AutoAttack sanity, overhead, CIFAR-10/100, ImageNet-100, C4, DDPM, AudioCaps, Clotho).
#
# Budget (A100 SXM ~2,200 img/s ResNet-50, ~900 img/s ViT-B/16; H100 ~1.7x faster):
#   r50_init 12 x 1.0 h | r50_al 30 x 7.7 h | r50_supervised 3 x 14.5 h                 ~ 285 A100-h
#   vit_mae_init 9 x 1.4 h | vit_mae_al 12 x 17 h | vit_mae_supervised 3 x 20 h          ~ 277 A100-h
#   ImageNet-100 ~17 | CIFAR ~26 | C4 + DDPM + audio ~29 | overhead + sanity ~3          ~  75 A100-h
#   => ~640 A100-h  ~ 3.3 days wall-clock on 8 GPUs (~US$1,000 at $1.59/GPU-h) + data preparation.
# Statistics: with three ImageNet seeds the paired tests have little power (exact sign-flip permutation
#   p >= 0.25, t-test with 2 d.o.f.); report mean +- std and bootstrap CIs, and lean on the 5-seed CIFAR /
#   3-seed ImageNet-100 results for significance claims.  The aggregate stage handles n = 3 (see docs/RUNBOOK.md).
# Run:  caenl run --plan configs/plans/franklin_budget_8gpu.yaml --gpus 0,1,2,3,4,5,6,7
# =============================================================================
"""


def main() -> None:
    plan = yaml.safe_load(SRC.read_text())
    out = copy.deepcopy({k: v for k, v in plan.items() if k != "stages"})
    out["campaign_id"] = "franklin-budget"
    out["stages"] = [copy.deepcopy(s) for s in plan["stages"] if s["name"] not in DROP_STAGES]
    names = {s["name"] for s in out["stages"]}
    for s in out["stages"]:
        if "depends_on" in s:
            s["depends_on"] = [d for d in s["depends_on"] if d in names]
        m = s.get("matrix") or {}
        if s["name"] in ("r50_init", "r50_al", "r50_supervised"):
            m["seed"] = list(SEEDS)
            if "expected" in s:
                s["expected"]["seeds"] = list(SEEDS)
        if s["name"] == "r50_al":
            m["method"] = list(R50_ARMS)
            s["expected"]["methods"] = list(R50_ARMS)
        if s["name"] == "vit_mae_al":
            m["method"] = list(VIT_ARMS)
            s["expected"]["methods"] = list(VIT_ARMS)
        if s["name"] == "overhead_imagenet1k":
            m["arch"] = ["resnet50", "vit_b_16_mae"]
    for s in out["stages"]:
        for key, val in (s.get("matrix") or {}).items():
            if isinstance(val, list) and not val:
                raise SystemExit(f"stage {s['name']!r}: empty matrix list {key!r}")
    body = yaml.safe_dump(out, sort_keys=False, width=200, default_flow_style=None)
    DST.write_text(HEADER + body)
    print(f"wrote {DST.relative_to(ROOT)}: {len(out['stages'])} stages")


if __name__ == "__main__":
    main()
