"""Campaign-role and caption metadata for generated reports.

Reports are separated by evidentiary role:

* ``confirmatory`` -> ``manuscript/`` (eligible for final claims),
* ``pilot`` -> ``pilot_report/`` (descriptive calibration only),
* ``smoke``/``engineering`` -> ``diagnostics/`` (never manuscript evidence).

Every caption is derived from the frozen job configuration rather than hard-coded assumptions.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ..utils.io import load_yaml

VALID_ROLES = {"confirmatory", "pilot", "smoke", "engineering"}


def campaign_role(root: Path | str) -> str:
    root = Path(root)
    role = None
    plan_path = root / "PLAN.yaml"
    if plan_path.exists():
        try:
            frozen = load_yaml(plan_path) or {}
            raw = frozen.get("plan", frozen) or {}
            role = raw.get("campaign_role")
            campaign_id = str(raw.get("campaign_id", root.name))
        except Exception:
            campaign_id = root.name
    else:
        campaign_id = root.name
    if role is None:
        role = "smoke" if "smoke" in campaign_id.lower() else "confirmatory"
    role = str(role).lower().strip()
    return role if role in VALID_ROLES else "engineering"


def report_dirname(role: str) -> str:
    role = str(role).lower()
    if role == "confirmatory":
        return "manuscript"
    if role == "pilot":
        return "pilot_report"
    return "diagnostics"


def role_banner(role: str) -> str:
    role = str(role).lower()
    if role == "pilot":
        return "PILOT — NOT CONFIRMATORY"
    if role in {"smoke", "engineering"}:
        return "SMOKE/ENGINEERING TEST — NOT FOR MANUSCRIPT"
    return ""


def _job_config(job: dict[str, Any]) -> dict[str, Any]:
    cfg = job.get("_config") or job.get("config") or {}
    return cfg if isinstance(cfg, dict) else {}


def _common(values: Iterable[Any], default: Any = None) -> Any:
    vals = [v for v in values if v is not None]
    if not vals:
        return default
    counts = Counter(map(str, vals))
    winner = counts.most_common(1)[0][0]
    for v in vals:
        if str(v) == winner:
            return v
    return vals[0]


def human_count(value: Any) -> str:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:g}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:g}M"
    if n >= 1_000:
        return f"{n / 1_000:g}k"
    return f"{n:,}"


def derive_report_meta(root: Path | str, jobs: list[dict[str, Any]], protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    root = Path(root)
    role = campaign_role(root)
    cfgs = [_job_config(j) for j in jobs]

    lang_cfgs = [c.get("language", {}) for c, j in zip(cfgs, jobs) if j.get("task") == "language_c4"]
    diff_cfgs = [c.get("diffusion", {}) for c, j in zip(cfgs, jobs) if j.get("task") == "diffusion_ddpm"]
    aud_cfgs = [c.get("audio", {}) for c, j in zip(cfgs, jobs) if j.get("task") in {"audio_captioning", "audio_retrieval"}]

    language = {
        "model": _common((c.get("model") for c in lang_cfgs), "language model"),
        "train_tokens": _common((c.get("train_tokens") for c in lang_cfgs)),
        "validation_tokens": _common((c.get("validation_tokens") for c in lang_cfgs)),
        "sequence_length": _common((c.get("sequence_length") for c in lang_cfgs)),
    }
    diffusion = {
        "model": _common((c.get("model") for c in diff_cfgs), "DDPM"),
        "train_steps": _common((c.get("train_steps") for c in diff_cfgs)),
        "num_samples": _common((c.get("num_samples") for c in diff_cfgs)),
        "sampling_steps": _common((c.get("sampling_steps") for c in diff_cfgs)),
        "sampler": _common((c.get("sampler") for c in diff_cfgs), "DDIM"),
    }
    audio = {
        "dataset": _common((c.get("dataset") for c in aud_cfgs)),
        "caption_epochs": _common(((c.get("captioning") or {}).get("epochs") for c in aud_cfgs)),
        "retrieval_epochs": _common(((c.get("retrieval") or {}).get("epochs") for c in aud_cfgs)),
    }

    return {
        "role": role,
        "report_dir": report_dirname(role),
        "banner": role_banner(role),
        "language": language,
        "diffusion": diffusion,
        "audio": audio,
        "protocol_version": (protocol or {}).get("protocol_version"),
    }
