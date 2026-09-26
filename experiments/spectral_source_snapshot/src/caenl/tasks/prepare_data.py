"""Dataset preparation jobs (download + cache) so that GPU stages never wait on I/O."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..utils.jobctx import JobContext

VISION = {"cifar10", "cifar100", "tiny_imagenet", "imagenet100", "imagenet1k", "synthetic", "synthetic_imagefolder"}


def prepare(name: str, data_root: Path, cache_root: Path, options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    options = dict(options or {})
    if name in VISION:
        from ..vision.data import load_vision_dataset

        ds = load_vision_dataset(name, data_root, cache_root, options)
        return {"dataset": name, "n_train": int(len(ds.train_y)), "n_test": int(len(ds.test_y)), "num_classes": ds.num_classes, "fingerprint": ds.fingerprint, "cache_dir": ds.meta.get("cache_dir")}
    if name == "c4":
        from ..language.data import prepare_c4

        return prepare_c4(data_root, cache_root, options)
    if name == "clotho":
        from ..audio.data import prepare_clotho

        return prepare_clotho(data_root, cache_root, options)
    if name == "audiocaps":
        from ..audio.audiocaps import prepare_audiocaps

        info = prepare_audiocaps(data_root, cache_root, options)
        if not info.get("valid", False) and bool(options.get("strict", True)):
            raise RuntimeError(f"AudioCaps manifest did not validate: {info.get('errors')} (coverage {info.get('coverage')})")
        return info
    if name == "audio_manifest":
        from ..audio.audiocaps import validate_audio_manifest

        rep = validate_audio_manifest(options["manifest"], min_coverage=options.get("min_coverage"), verify_hashes=str(options.get("verify_hashes", "sample")))
        if not rep["valid"] and bool(options.get("strict", True)):
            raise RuntimeError(f"audio manifest {options['manifest']} did not validate: {rep['errors']}")
        return {"dataset": "audio_manifest", **{k: v for k, v in rep.items() if k != "warnings"}}
    if name == "mae_weights":
        from ..vision.mae import MAE_REPO, fetch_mae_state_dict

        repo = str(options.get("repo", MAE_REPO))
        sd = fetch_mae_state_dict(repo, cache_dir=str(Path(data_root) / "hf"))
        return {"dataset": "mae_weights", "repo": repo, "tensors": len(sd), "params": int(sum(v.numel() for v in sd.values()))}
    if name == "ddpm_cifar10":
        from ..diffusion.task import prepare_assets

        return prepare_assets(data_root, cache_root, options)
    raise ValueError(f"unknown dataset '{name}'")


def run(ctx: JobContext) -> dict[str, Any]:
    cfg = ctx.config
    paths = cfg.get("paths", {})
    out: dict[str, Any] = {"task": "prepare_data", "prepared": []}
    specs = list(cfg.get("datasets", []))
    if cfg.get("dataset_spec"):
        specs.append(cfg["dataset_spec"])
    for spec in specs:
        if isinstance(spec, str):
            name, options = spec, {}
        else:
            name, options = spec["name"], dict(spec.get("options", {}))
        with ctx.timer(f"prepare_{name}"):
            info = prepare(name, Path(paths.get("data_root", "data")), Path(paths.get("cache_root", "cache")), options)
        ctx.event("prepared", **{k: v for k, v in info.items() if isinstance(v, (int, float, str))})
        out["prepared"].append(info)
    return out
