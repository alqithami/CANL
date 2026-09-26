from pathlib import Path
import json
import tempfile

import pandas as pd
import yaml

from caenl.plan import Plan
from caenl.runner import _source_snapshot

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs" / "plans" / "one_c4_confirmatory_v2.yaml"


def test_c4_confirmatory_plan_is_frozen_from_calibration_v2() -> None:
    raw = yaml.safe_load(PLAN_PATH.read_text())
    assert raw["campaign_id"] == "caenl-c4-confirmatory-v2"
    assert raw["campaign_role"] == "confirmatory"
    assert raw["defaults"]["language"]["train_tokens"] == 100_000_000
    assert raw["defaults"]["language"]["batch_tokens"] == 8192
    assert raw["defaults"]["language"]["dcr_token_subsample"] == 2048
    assert raw["defaults"]["language"]["configured_rank"] == {"h6": 16, "h12": 4}
    assert raw["methods"]["fixed_dcr"]["dcr_lambda"] == 0.01
    assert raw["defaults"]["macc_lite"]["eta_lambda"] == 0.002
    assert raw["defaults"]["statistics"]["minimum_paired_seeds_for_inference"] == 6
    stage = next(s for s in raw["stages"] if s["name"] == "c4_confirmatory")
    assert stage["matrix"]["seed"] == [301, 302, 303, 304, 305, 306]
    assert stage["matrix"]["method"] == ["baseline", "fixed_dcr", "macc_lite", "full_macc"]
    assert stage["include_in_inference"] is True


def test_c4_confirmatory_expands_to_26_jobs() -> None:
    plan = Plan.load(PLAN_PATH)
    expanded = {stage["name"]: jobs for stage, jobs in plan.expand()}
    assert len(expanded["prepare_c4"]) == 1
    assert len(expanded["c4_confirmatory"]) == 24
    assert len(expanded["aggregate"]) == 1
    assert sum(map(len, expanded.values())) == 26
    for job in expanded["c4_confirmatory"]:
        assert job["analysis_role"] == "confirmatory"
        assert job["include_in_inference"] is True
        assert job["paths"]["results_root"].startswith("/mnt/caenl/")


def test_crossmodal_comparisons_name_real_confirmatory_methods() -> None:
    plan = Plan.load(PLAN_PATH)
    cfg = plan.base_config()
    comps = cfg["statistics"]["crossmodal_comparisons"]
    pairs = {(c["target"], r) for c in comps for r in c["references"]}
    assert pairs == {
        ("fixed_dcr", "baseline"),
        ("macc_lite", "baseline"),
        ("full_macc", "baseline"),
        ("full_macc", "macc_lite"),
    }


def test_source_snapshot_is_deterministic_and_covers_plan() -> None:
    a = _source_snapshot(ROOT)
    b = _source_snapshot(ROOT)
    assert a["tree_sha256"] == b["tree_sha256"]
    assert a["file_count"] > 50
    paths = {x["path"] for x in a["files"]}
    assert "configs/plans/one_c4_confirmatory_v2.yaml" in paths
    assert "src/caenl/language/task.py" in paths
    assert len(a["tree_sha256"]) == 64


def test_crossmodal_significance_emits_all_configured_pairs() -> None:
    from caenl.report.aggregate import paired_significance

    plan = Plan.load(PLAN_PATH)
    cfg = plan.base_config()
    comparisons = cfg["statistics"]["crossmodal_comparisons"]
    rows = []
    offsets = {"baseline": 0.0, "fixed_dcr": -0.02, "macc_lite": -0.01, "full_macc": -0.015}
    for seed in range(301, 307):
        for method, offset in offsets.items():
            rows.append({
                "task": "language_c4",
                "dataset": "allenai/c4",
                "method": method,
                "seed": seed,
                "metric": "validation_nll",
                "value": 3.4 + 0.001 * (seed - 301) + offset,
                "inference_eligible": True,
            })
    out = paired_significance(
        pd.DataFrame(rows),
        comparisons,
        ["validation_nll"],
        ["task", "dataset"],
        n_boot=100,
        n_perm=100,
        min_pairs=6,
    )
    present = set(zip(out["target"], out["reference"]))
    assert present == {
        ("fixed_dcr", "baseline"),
        ("macc_lite", "baseline"),
        ("full_macc", "baseline"),
        ("full_macc", "macc_lite"),
    }
    assert set(out["n_pairs"]) == {6}


def test_report_uses_frozen_plan_defaults_for_inference_contract(tmp_path: Path) -> None:
    from caenl.report.aggregate import run_report

    root = tmp_path / "campaign"
    root.mkdir()
    (root / "frozen_protocol.yaml").write_text(
        yaml.safe_dump({"statistics": {"minimum_paired_seeds_for_inference": 3}}),
        encoding="utf-8",
    )
    (root / "PLAN.yaml").write_text(
        yaml.safe_dump(
            {
                "plan": {
                    "campaign_id": "report-default-test",
                    "campaign_role": "confirmatory",
                    "defaults": {
                        "statistics": {
                            "minimum_paired_seeds_for_inference": 6,
                            "crossmodal_comparisons": [
                                {"target": "fixed_dcr", "references": ["baseline"]}
                            ],
                        }
                    },
                },
                "protocol": {},
            }
        ),
        encoding="utf-8",
    )
    result = run_report(root, strict=False, quick=True)
    assert result["minimum_paired_seeds_for_inference"] == 6
