from pathlib import Path

from caenl.plan import Plan
from caenl.report.context import report_dirname, role_banner

ROOT = Path(__file__).resolve().parents[1]


def test_one_dataset_plans_are_sequential_and_role_tagged():
    plans = sorted((ROOT / "configs" / "plans").glob("one_*.yaml"))
    assert plans
    for path in plans:
        plan = Plan.load(path)
        assert plan.campaign_role in {"pilot", "confirmatory"}
        expanded = plan.expand()
        task_families = {
            stage["type"]
            for stage, _ in expanded
            if stage["type"] not in {"prepare_data", "aggregate", "vision_overhead", "vision_aa_sanity"}
        }
        # AudioCaps intentionally contains captioning + retrieval on the same dataset.
        assert len(task_families) == 1 or task_families == {"audio_captioning", "audio_retrieval"}
        for stage, jobs in expanded:
            assert int(stage.get("parallel", 1)) == 1
            for job in jobs:
                assert job["campaign_role"] == plan.campaign_role
                assert job["analysis_role"] == plan.campaign_role
                assert job["include_in_science"] is True
                assert job["include_in_inference"] is (plan.campaign_role == "confirmatory")


def test_report_directories_and_banners_are_role_safe():
    assert report_dirname("confirmatory") == "manuscript"
    assert report_dirname("pilot") == "pilot_report"
    assert report_dirname("smoke") == "diagnostics"
    assert role_banner("confirmatory") == ""
    assert "NOT CONFIRMATORY" in role_banner("pilot")
    assert "NOT FOR MANUSCRIPT" in role_banner("smoke")


def test_cleanup_is_allowlisted_and_requires_confirmation():
    script = (ROOT / "scripts" / "cleanup_caenl.sh").read_text()
    assert "require_yes" in script
    assert "safe_under" in script
    assert "verify_archive_for" in script
    assert "--one-file-system" in script
    assert "eval " not in script


def test_one_dataset_runner_blocks_imagenet1k_on_small_storage():
    script = (ROOT / "scripts" / "run_one_dataset.sh").read_text()
    assert "ImageNet-1K plan requires >=320 GB" in script
    assert "--parallel 1 --gpus 0 --fail-fast" in script
    assert "export_campaign_lean.sh" in script


def test_one_dataset_runner_has_global_lock_and_failure_record():
    script = (ROOT / "scripts" / "run_one_dataset.sh").read_text()
    assert "flock -n 9" in script
    assert ".one-dataset.lock" in script
    assert "ONE-DATASET RUN FAILED" in script


def test_export_verifies_checksum_from_archive_directory():
    script = (ROOT / "scripts" / "export_campaign_lean.sh").read_text()
    assert 'cd "$(dirname "$ARCHIVE")"' in script
    assert 'sha256sum -c "$(basename "$CHECKSUM")"' in script
    assert "resolved-PLAN.yaml" in script


def test_cleanup_protects_approved_active_roots_and_can_remove_transients():
    script = (ROOT / "scripts" / "cleanup_caenl.sh").read_text()
    assert "require_default_active_root" in script
    assert "/dev/shm/caenl-active" in script
    assert "/mnt/caenl/active" in script
    assert "clean-transient" in script
    assert "youtube-cookies.txt" in script


def test_legacy_model_cache_migration_is_non_destructive():
    script = (ROOT / "scripts" / "migrate_legacy_model_cache.sh").read_text()
    assert "/dev/shm/cache/huggingface/hub" in script
    assert "/dev/shm/data/hf" in script
    assert "--ignore-existing" in script
    assert "rm -rf" not in script
