"""Plan/protocol validation and the production pre-flight.

``preflight`` is a *conjunction* of named checks; ``ok`` is False (and ``caenl preflight`` exits
non-zero) as soon as one **required** check fails.  Smoke plans may relax the GPU and
AutoAttack-pin requirements through the plan's ``preflight:`` block; production plans cannot
relax data, dependency, disk, storage or manifest checks.
"""
from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import torch
import psutil

from .plan import Plan
from .tasks import TASKS

AUTOATTACK_PIN = "a39220048b3c9f2cca9a4d3a54604793c68eca7e"  # fra31/auto-attack, master, as pinned in requirements.txt
def _vision_hf_models(cfg: dict) -> list:
    opts = cfg.get("arch_options", {}) or {}
    if cfg.get("arch") == "vit_b_16_mae" or opts.get("init") == "mae":
        return [str(opts.get("mae_repo", "facebook/vit-mae-base"))]
    return []


HF_MODELS_BY_TYPE = {
    "vision_al": _vision_hf_models,
    "vision_overhead": _vision_hf_models,
    "language_c4": lambda cfg: [str(cfg.get("language", {}).get("model", "gpt2"))] if cfg.get("language", {}).get("init", "pretrained") == "pretrained" and cfg.get("language", {}).get("model") not in ("tiny_random",) else [],
    "diffusion_ddpm": lambda cfg: [str(cfg.get("diffusion", {}).get("model", "google/ddpm-cifar10-32"))] if cfg.get("diffusion", {}).get("model") not in ("tiny_random",) else [],
    "audio_captioning": lambda cfg: [m for m in (cfg.get("audio", {}).get("backbone"), cfg.get("audio", {}).get("captioning", {}).get("decoder", "gpt2")) if m and m not in ("mel_cnn", "tiny_random")],
    "audio_retrieval": lambda cfg: [m for m in (cfg.get("audio", {}).get("backbone"), cfg.get("audio", {}).get("retrieval", {}).get("text_encoder")) if m and m not in ("mel_cnn", "hashing", "tiny_random")],
}
DATASET_DISK_GB = {"imagenet1k": 275.0, "imagenet100": 10.0, "tiny_imagenet": 2.0, "cifar10": 1.0, "cifar100": 1.0, "synthetic": 0.1, "c4": 3.0, "ddpm_cifar10": 3.0, "clotho": 20.0, "audiocaps": 25.0}


# ----------------------------------------------------------------------------- static plan validation
def validate_plan(plan: Plan, check_data: bool = False, check_env: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    names = set()
    for stage in plan.stages:
        if "name" not in stage or "type" not in stage:
            errors.append(f"stage without name/type: {stage}")
            continue
        if stage["name"] in names:
            errors.append(f"duplicate stage name {stage['name']}")
        names.add(stage["name"])
        if stage["type"] not in TASKS:
            errors.append(f"stage {stage['name']}: unknown type {stage['type']}")
        if stage["type"] == "fault_injection":
            warnings.append(f"stage {stage['name']}: fault_injection is a smoke-test device and must not appear in a production plan")
        for m in stage.get("matrix", {}).get("method", []) or []:
            if m not in plan.methods:
                errors.append(f"stage {stage['name']}: method '{m}' not in registry")
        for key, val in (stage.get("matrix") or {}).items():
            if isinstance(val, list) and not val:
                errors.append(f"stage {stage['name']}: matrix list '{key}' is empty (the stage would expand to zero jobs)")
        for key in ("parallel", "gpu_share"):
            if key in stage:
                try:
                    if int(stage[key]) < 1:
                        errors.append(f"stage {stage['name']}: {key} must be >= 1 (got {stage[key]!r})")
                except (TypeError, ValueError):
                    errors.append(f"stage {stage['name']}: {key} must be an integer (got {stage[key]!r})")
        exp = stage.get("expected")
        if exp:
            mx = stage.get("matrix", {})
            for key in ("seeds", "methods"):
                want = set(exp.get(key, []) or [])
                have = set(mx.get(key[:-1], []) or [])
                if want - have:
                    errors.append(f"stage {stage['name']}: expected {key} {sorted(want - have)} are not in the matrix")
    try:
        from .runner import resolve_dependencies

        resolve_dependencies([st for st in plan.stages if "name" in st and "type" in st])
    except ValueError as exc:
        errors.append(str(exc))
    jobs_total = 0
    try:
        for stage, jobs in plan.expand():
            jobs_total += len(jobs)
            if not jobs:
                errors.append(f"stage {stage['name']}: expands to zero jobs")
            for j in jobs:
                if j["type"] == "vision_al":
                    al = j.get("active_learning", {})
                    for key in ("initial_fraction", "rounds", "fraction_per_round", "candidate_size"):
                        if key not in al:
                            errors.append(f"{j['job_id']}: active_learning.{key} missing")
                    if j.get("method", {}).get("acquisition") == "collapse" and not j.get("collapse", {}).get("monitored_layers"):
                        errors.append(f"{j['job_id']}: collapse acquisition needs monitored layers")
                    rob = j.get("robustness", {})
                    if abs(float(rob.get("eps", 8 / 255)) - 8 / 255) > 1e-9:
                        warnings.append(f"{j['job_id']}: eps differs from 8/255")
                if j["type"] in ("audio_captioning", "audio_retrieval"):
                    ac = j.get("audio", {})
                    if str(ac.get("dataset", "audiocaps")) not in ("synthetic", "clotho") and not ac.get("manifest"):
                        errors.append(f"{j['job_id']}: audio.dataset={ac.get('dataset')} requires audio.manifest")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"plan expansion failed: {exc!r}")
    st = plan.protocol.get("statistics", {})
    if st and len(st.get("principal_seeds", [])) < 3:
        warnings.append("fewer than 3 principal seeds")
    if check_env:
        for mod in ["autoattack", "transformers", "datasets", "diffusers", "torch_fidelity", "soundfile"]:
            try:
                importlib.import_module(mod)
            except Exception:
                warnings.append(f"optional module not importable: {mod}")
        if not torch.cuda.is_available():
            warnings.append("CUDA not available: only smoke plans are practical")
    if check_data:
        from .vision.data import DATASET_INFO

        for (ds, _o), cd in vision_caches_needed(plan).items():
            if ds not in DATASET_INFO:
                errors.append(f"unknown dataset {ds}")
            elif ds != "synthetic" and not (cd / "meta.json").exists():
                warnings.append(f"dataset cache missing for {ds}: {cd} (run `caenl preflight --download`)")
    return {"campaign_id": plan.campaign_id, "campaign_root": str(plan.campaign_root), "jobs_total": jobs_total, "fingerprint": plan.fingerprint(), "errors": errors, "warnings": warnings, "valid": not errors}


# ----------------------------------------------------------------------------- helpers for preflight
def expanded_jobs(plan: Plan) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    out = []
    for stage, jobs in plan.expand():
        for j in jobs:
            out.append((stage, j))
    return out


def vision_caches_needed(plan: Plan) -> dict[tuple[str, tuple], Path]:
    from .vision.data import DATASET_INFO, cache_dir_for

    needed: dict[tuple[str, tuple], Path] = {}
    for stage, j in expanded_jobs(plan):
        if j["type"] in ("vision_al", "vision_overhead", "vision_aa_sanity") and j.get("dataset"):
            if j["type"] == "vision_overhead" and not bool(j.get("overhead", {}).get("use_real_data", False)):
                continue  # benchmark on random tensors
            ds = j["dataset"]
            o = dict(j.get("dataset_options", {}))
            if ds in DATASET_INFO and ds != "synthetic":
                info = DATASET_INFO[ds]
                o.setdefault("cache_resolution", info.get("cache_res", info["train_res"]))
            key = (ds, tuple(sorted((k, str(v)) for k, v in o.items())))
            needed[key] = cache_dir_for(ds, plan.cache_root, o)
    return needed


def prepare_specs(plan: Plan) -> list[tuple[str, dict[str, Any]]]:
    specs: list[tuple[str, dict[str, Any]]] = []
    seen = set()
    for stage, j in expanded_jobs(plan):
        if stage["type"] != "prepare_data":
            continue
        for spec in list(j.get("datasets", [])) + ([j["dataset_spec"]] if j.get("dataset_spec") else []):
            name = spec if isinstance(spec, str) else spec["name"]
            opts = {} if isinstance(spec, str) else dict(spec.get("options", {}))
            key = (name, json.dumps(opts, sort_keys=True, default=str))
            if key not in seen:
                seen.add(key)
                specs.append((name, opts))
    return specs


def autoattack_provenance() -> dict[str, Any]:
    out: dict[str, Any] = {"installed": False, "commit": None, "source": None}
    try:
        dist = importlib_metadata.distribution("autoattack")
    except importlib_metadata.PackageNotFoundError:
        return out
    out["installed"] = True
    out["version"] = dist.version
    try:
        du = json.loads(dist.read_text("direct_url.json") or "{}")
    except Exception:
        du = {}
    out["source"] = du.get("url")
    out["commit"] = (du.get("vcs_info") or {}).get("commit_id")
    return out


def java_available() -> Optional[str]:
    exe = shutil.which("java")
    if not exe:
        return None
    try:
        res = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=30)
        text = (res.stderr or res.stdout).strip()
        if res.returncode != 0:
            return None
        lines = [l for l in text.splitlines() if "version" in l.lower()] or text.splitlines()  # skip "Picked up JAVA_TOOL_OPTIONS" noise
        return lines[0].strip() if lines else "java"
    except Exception:
        return None


def hf_model_available(repo: str, data_root: Path, timeout_s: float = 20.0) -> tuple[bool, str]:
    """True if the model is in the local HF cache or the Hub answers (download access)."""
    cache_dir = data_root / "hf"
    try:
        from huggingface_hub import HfApi, try_to_load_from_cache  # type: ignore

        for fname in ("config.json", "model_index.json"):
            for cd in (str(cache_dir), None):
                try:
                    hit = try_to_load_from_cache(repo, fname, cache_dir=cd)
                except Exception:
                    hit = None
                if isinstance(hit, str):
                    return True, f"cached ({hit})"
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            return False, "not cached and HF_HUB_OFFLINE=1"
        HfApi().model_info(repo, timeout=timeout_s, token=os.environ.get("HF_TOKEN"))
        return True, "reachable on the Hub"
    except Exception as exc:  # noqa: BLE001
        return False, f"not cached and Hub unreachable: {type(exc).__name__}: {str(exc)[:120]}"


def writable(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=str(path), prefix=".caenl_write_test_", delete=True) as f:
            f.write(b"ok")
            f.flush()
        return True, "writable"
    except Exception as exc:  # noqa: BLE001
        return False, f"not writable: {exc}"


def disk_requirement_gb(plan: Plan, n_jobs: int = 0) -> float:
    gb = 0.0
    prepared = {n for n, _ in prepare_specs(plan)}
    for name in prepared:
        gb += DATASET_DISK_GB.get(name, 1.0)
    for (ds, _), _cd in vision_caches_needed(plan).items():
        if ds not in prepared:
            gb += DATASET_DISK_GB.get(ds, 1.0)
    per_job = {"vision_al": {"imagenet1k": 1.0, "imagenet100": 0.8, "tiny_imagenet": 0.6}, "language_c4": 2.0, "diffusion_ddpm": 1.5, "audio_captioning": 0.6, "audio_retrieval": 0.3}
    for _, j in expanded_jobs(plan):
        t = j["type"]
        if t == "vision_al":
            gb += per_job["vision_al"].get(j.get("dataset"), 0.3)  # checkpoints (last2), logits, masks, logs
        else:
            gb += float(per_job.get(t, 0.1)) if not (j.get("language", {}).get("dataset") == "synthetic" or j.get("diffusion", {}).get("dataset") == "synthetic" or j.get("audio", {}).get("dataset") == "synthetic") else 0.05
    return gb


# ----------------------------------------------------------------------------- preflight
def preflight(plan: Plan, download: bool = False) -> dict[str, Any]:
    """Run every pre-flight check. ``ok`` is the conjunction of all *required* checks."""
    pf = dict(plan.raw.get("preflight", {}) or {})
    require_gpu = bool(pf.get("require_gpu", True))
    require_pin = bool(pf.get("require_pinned_autoattack", True))
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str = "", required: bool = True) -> None:
        checks.append({"name": name, "ok": bool(ok), "required": bool(required), "detail": detail})

    rep = validate_plan(plan, check_data=False, check_env=False)
    add("plan_valid", rep["valid"], "; ".join(rep["errors"]) or f"{rep['jobs_total']} jobs")
    jobs = expanded_jobs(plan)
    types = {j["type"] for _, j in jobs}

    # -- hardware
    hw = dict(plan.raw.get("hardware", {}) or {})
    cuda = torch.cuda.is_available()
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else []
    add("cuda_gpu", cuda, ", ".join(gpus) if cuda else "torch.cuda.is_available() is False", required=require_gpu)
    if cuda:
        free, total = torch.cuda.mem_get_info(0)
        min_gpu_gb = float(hw.get("min_gpu_memory_gb", 38.0))
        add(
            "gpu_memory",
            total >= min_gpu_gb * (1024**3),
            f"{total / (1024**3):.1f} GiB total, {free / (1024**3):.1f} GiB free on GPU 0 (required >= {min_gpu_gb:.1f} GiB)",
            required="min_gpu_memory_gb" in hw,
        )
        expected = hw.get("expected_gpu_name_regex")
        if expected:
            add("gpu_model", bool(re.search(str(expected), gpus[0], flags=re.IGNORECASE)), f"detected {gpus[0]!r}; expected /{expected}/")
        min_driver = hw.get("min_nvidia_driver")
        if min_driver is not None:
            try:
                driver = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    text=True,
                    timeout=15,
                ).splitlines()[0].strip()
                major = int(driver.split(".")[0])
                add("nvidia_driver", major >= int(min_driver), f"driver {driver}; required >= {min_driver}")
            except Exception as exc:  # noqa: BLE001
                add("nvidia_driver", False, f"could not query driver: {type(exc).__name__}: {exc}")
        expected_cuda = hw.get("torch_cuda")
        if expected_cuda:
            actual_cuda = str(torch.version.cuda or "none")
            add("torch_cuda_runtime", actual_cuda.startswith(str(expected_cuda)), f"torch CUDA runtime {actual_cuda}; expected {expected_cuda}.x")
    min_cpus = int(hw.get("min_logical_cpus", 0) or 0)
    if min_cpus:
        actual_cpus = int(os.cpu_count() or 0)
        add("logical_cpus", actual_cpus >= min_cpus, f"{actual_cpus} logical CPUs; required >= {min_cpus}")
    min_ram = float(hw.get("min_system_memory_gb", 0.0) or 0.0)
    if min_ram:
        actual_ram = psutil.virtual_memory().total / (1024**3)
        add("system_memory", actual_ram >= min_ram, f"{actual_ram:.1f} GiB RAM; required >= {min_ram:.1f} GiB")

    # -- python packages per task group
    groups = {
        "core": ["numpy", "scipy", "pandas", "yaml", "matplotlib", "torch", "torchvision", "psutil"],
        "robustness": ["autoattack"] if types & {"vision_al", "vision_aa_sanity"} else [],
        "language": ["transformers", "datasets"] if "language_c4" in types else [],
        "diffusion": ["diffusers", "torch_fidelity", "accelerate"] if "diffusion_ddpm" in types else [],
        "audio": ["transformers", "soundfile"] if types & {"audio_captioning", "audio_retrieval"} else [],
        "hf_datasets": ["datasets"] if any(j.get("dataset") in ("tiny_imagenet", "imagenet100") or (j.get("dataset") == "imagenet1k" and j.get("dataset_options", {}).get("source") == "hf") for _, j in jobs) else [],
    }
    for group, mods in groups.items():
        if not mods:
            continue
        missing = []
        for mod in mods:
            try:
                importlib.import_module(mod)
            except Exception as exc:  # noqa: BLE001
                missing.append(f"{mod} ({type(exc).__name__})")
        add(f"modules_{group}", not missing, "missing: " + ", ".join(missing) if missing else "ok")

    # -- AutoAttack pin
    if types & {"vision_al", "vision_aa_sanity"}:
        prov = autoattack_provenance()
        if not prov["installed"]:
            add("autoattack_pinned", False, "autoattack not installed (pip install 'git+https://github.com/fra31/auto-attack@" + AUTOATTACK_PIN + "')", required=True)
        elif prov["commit"]:
            add("autoattack_pinned", prov["commit"] == AUTOATTACK_PIN, f"installed commit {prov['commit']} (pinned {AUTOATTACK_PIN})", required=require_pin)
        else:
            add("autoattack_pinned", False, f"AutoAttack provenance not verifiable (installed from {prov.get('source')}); reinstall from the pinned commit {AUTOATTACK_PIN}", required=require_pin)

    # -- Java + pycocoevalcap when METEOR/SPICE are part of the protocol
    cap_jobs = [j for _, j in jobs if j["type"] == "audio_captioning"]
    needs_java = any(bool(j.get("audio", {}).get("captioning", {}).get("java_metrics", True)) and ({"meteor", "spice"} & set(j.get("audio", {}).get("captioning", {}).get("metrics", ["meteor", "spice"]))) for j in cap_jobs)
    if needs_java:
        jv = java_available()
        add("java_runtime", jv is not None, jv or "java not found (METEOR/SPICE are enabled: install default-jre-headless or set audio.captioning.java_metrics: false)")
        try:
            pcc = importlib.import_module("pycocoevalcap")
            add("pycocoevalcap", True, "ok")
            base = Path(list(pcc.__path__)[0])
            meteor_jar = base / "meteor" / "meteor-1.5.jar"
            meteor_data = base / "meteor" / "data" / "paraphrase-en.gz"
            add("meteor_assets", meteor_jar.exists() and meteor_data.exists(), "ok" if (meteor_jar.exists() and meteor_data.exists()) else f"METEOR jar/data missing under {base / 'meteor'} (reinstall pycocoevalcap)")
            needs_spice = any(bool(j.get("audio", {}).get("captioning", {}).get("spice", True)) and "spice" in set(j.get("audio", {}).get("captioning", {}).get("metrics", ["spice"])) for j in cap_jobs)
            if needs_spice:
                spice_jar = base / "spice" / "lib" / "stanford-corenlp-3.6.0.jar"
                if not spice_jar.exists() and download:
                    try:
                        from pycocoevalcap.spice.get_stanford_models import get_stanford_models  # type: ignore

                        get_stanford_models()
                    except Exception as exc:  # noqa: BLE001
                        add("spice_models_download", False, f"Stanford CoreNLP download failed: {type(exc).__name__}: {exc}")
                add("spice_models", spice_jar.exists(), "ok" if spice_jar.exists() else f"SPICE needs {spice_jar.name} (Stanford CoreNLP 3.6.0, ~400 MB from nlp.stanford.edu); run `caenl preflight --download` or set audio.captioning.spice: false")
        except Exception as exc:  # noqa: BLE001
            add("pycocoevalcap", False, f"pycocoevalcap not importable ({type(exc).__name__}); METEOR/SPICE and the official BLEU/CIDEr/ROUGE cross-check need it")

    # -- yt-dlp/ffmpeg for AudioCaps acquisition (only when the prepare stage asks for downloads)
    for name, opts in prepare_specs(plan):
        if name == "audiocaps" and bool(opts.get("download_audio", True)):
            have = shutil.which("yt-dlp") is not None and shutil.which("ffmpeg") is not None
            add("yt_dlp_ffmpeg", have, "ok" if have else "yt-dlp and/or ffmpeg missing: AudioCaps audio cannot be fetched here (install them, or place the clips under <data_root>/audiocaps/audio/<split>/)", required=False)

    # -- storage
    for label, path in (("results_root", plan.results_root), ("data_root", plan.data_root), ("cache_root", plan.cache_root)):
        ok, detail = writable(path)
        add(f"writable_{label}", ok, f"{path}: {detail}")
    need_gb = float(pf.get("min_free_gb", disk_requirement_gb(plan, len(jobs))))
    roots: dict[str, float] = {}
    for path in (plan.results_root, plan.data_root, plan.cache_root):
        try:
            roots[str(path)] = round(shutil.disk_usage(path).free / 1e9, 1)
        except Exception:
            roots[str(path)] = 0.0
    min_free = min(roots.values()) if roots else 0.0
    add("disk_space", min_free >= need_gb, f"min free across roots {min_free:.0f} GB, estimated need {need_gb:.0f} GB ({roots})")

    # -- downloads (run the plan's prepare stage now so that the data checks below see the result)
    prepared: list[dict[str, Any]] = []
    if download:
        from .tasks.prepare_data import prepare

        for name, opts in prepare_specs(plan):
            t0 = time.time()
            try:
                info = prepare(name, plan.data_root, plan.cache_root, opts)
                prepared.append({"dataset": name, "ok": True, "seconds": round(time.time() - t0, 1), **{k: v for k, v in info.items() if isinstance(v, (int, float, str, bool))}})
            except Exception as exc:  # noqa: BLE001
                prepared.append({"dataset": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
        add("downloads", all(p["ok"] for p in prepared), "; ".join(f"{p['dataset']}: {'ok' if p['ok'] else p['error']}" for p in prepared) or "nothing to prepare")

    # -- dataset caches / licensed data
    from .vision.data import DATASET_INFO

    for (ds, _opts), cd in vision_caches_needed(plan).items():
        if ds == "synthetic":
            continue
        present = (cd / "meta.json").exists() and (cd / "train_x.npy").exists()
        if present:
            add(f"data_{ds}", True, str(cd))
        elif ds == "imagenet1k":
            o = dict(_opts)
            for pname, popts in prepare_specs(plan):
                if pname == "imagenet1k" and "source" in popts:
                    o.setdefault("source", popts["source"])
            if o.get("source", "imagefolder") == "hf":
                tok = "HF_TOKEN is set" if os.environ.get("HF_TOKEN") else "HF_TOKEN is NOT set"
                add("data_imagenet1k", False, f"cache {cd} missing; the gated HF copy needs HF_TOKEN with access to ILSVRC/imagenet-1k ({tok}), then `caenl preflight --download` builds the cache")
            else:
                from .vision.data import resolve_imagenet_layout

                try:
                    lay = resolve_imagenet_layout(Path(o.get("imagenet_dir", plan.data_root / "imagenet")))
                    if download:
                        add("data_imagenet1k", False, f"raw ImageNet found ({lay['train'].parent}) but the cache {cd} was not built; see downloads")
                    else:
                        add("data_imagenet1k", False, f"raw ImageNet found ({lay['train'].parent}); build the 256 px cache first: `caenl preflight --plan ... --download` (~262 GB, 1-2 h)")
                except FileNotFoundError as exc:
                    add("data_imagenet1k", False, str(exc))
        else:
            add(f"data_{ds}", False, f"cache missing at {cd}; run `caenl preflight --plan ... --download`" if ds in DATASET_INFO else f"unknown dataset {ds}")

    # -- cross-modal data
    if any(j["type"] == "language_c4" and j.get("language", {}).get("dataset", "allenai/c4") != "synthetic" for _, j in jobs):
        lang_dir = plan.cache_root / "language"
        c4_ok = lang_dir.exists() and any(lang_dir.rglob("meta.json"))
        add("data_c4", c4_ok, "tokenised C4 shards present" if c4_ok else f"no tokenised C4 cache under {lang_dir}; run `caenl preflight --download`")
    from .vision.data import cache_dir_for

    for _, j in jobs:
        if j["type"] == "diffusion_ddpm" and j.get("diffusion", {}).get("dataset", "cifar10") != "synthetic":
            dname = str(j.get("diffusion", {}).get("dataset", "cifar10"))
            o = dict(j.get("diffusion", {}).get("dataset_options", {}))
            if dname in DATASET_INFO:
                o.setdefault("cache_resolution", DATASET_INFO[dname].get("cache_res", DATASET_INFO[dname]["train_res"]))
            cd = cache_dir_for(dname, plan.cache_root, o)
            ok = (cd / "meta.json").exists()
            add(f"data_diffusion_{dname}", ok, str(cd) if ok else f"image cache for the diffusion stage missing at {cd}; run `caenl preflight --download`")
            break
    manifests: dict[tuple[str, str], tuple[Path, Optional[float], str]] = {}
    for _, j in jobs:
        if j["type"] not in ("audio_captioning", "audio_retrieval"):
            continue
        ac = j.get("audio", {})
        ds = str(ac.get("dataset", "audiocaps"))
        if ds == "synthetic":
            continue
        if ds == "clotho":
            m = plan.data_root / "clotho" / "manifest.jsonl"
            manifests[("clotho", str(m))] = (m, None, "none")
        else:
            manifests[(ds, str(ac.get("manifest")))] = (Path(str(ac.get("manifest"))), ac.get("manifest_min_coverage"), str(ac.get("manifest_verify_hashes", "sample")))
    for (ds, _), (m, min_cov, verify) in manifests.items():
        if ds == "clotho" and not m.exists():
            add("data_clotho", False, f"{m} missing; `caenl preflight --download` fetches Clotho v2.1 from Zenodo")
            continue
        from .audio.audiocaps import validate_audio_manifest

        vr = validate_audio_manifest(m, min_coverage=min_cov, verify_hashes=verify)
        detail = "; ".join(vr["errors"][:4]) if not vr["valid"] else f"{vr['n_clips']} clips {vr['splits']} coverage {vr.get('coverage')}"
        if vr.get("warnings"):
            detail += f" | warnings: {vr['warnings']}"
        add(f"audio_manifest_{ds}", vr["valid"], detail)

    # -- model downloads / Hub access
    wanted: dict[str, None] = {}
    for _, j in jobs:
        fn = HF_MODELS_BY_TYPE.get(j["type"])
        if fn:
            for m in fn(j):
                wanted[m] = None
    if any(j["type"] == "vision_aa_sanity" and j.get("sanity", {}).get("pretrained") for _, j in jobs):
        wanted["torchvision:resnet50"] = None
    for repo in wanted:
        if repo.startswith("torchvision:"):
            from torchvision.models import ResNet50_Weights

            weights = ResNet50_Weights.IMAGENET1K_V1
            fname = Path(weights.url).name
            hub = Path(torch.hub.get_dir()) / "checkpoints"
            cached = (hub / fname).exists()
            if not cached and download:
                try:
                    weights.get_state_dict(progress=False)  # fetches into the torch hub cache
                    cached = (hub / fname).exists()
                except Exception as exc:  # noqa: BLE001
                    add("model_torchvision_resnet50_download", False, f"download of {weights.url} failed: {type(exc).__name__}: {str(exc)[:120]}")
            ok, detail = cached, (f"cached ({hub / fname})" if cached else "not cached")
            if not cached:
                try:
                    import urllib.request

                    req = urllib.request.Request(weights.url, method="HEAD")
                    with urllib.request.urlopen(req, timeout=15) as r:
                        ok, detail = r.status == 200, f"{weights.url} reachable (HTTP {r.status}); fetched by `caenl preflight --download`"
                except Exception as exc:  # noqa: BLE001
                    detail = f"not cached and {weights.url} unreachable ({type(exc).__name__})"
            add("model_torchvision_resnet50", ok, detail)
        else:
            ok, detail = hf_model_available(repo, plan.data_root)
            add(f"model_{repo.replace('/', '_')}", ok, detail)

    required_failed = [c["name"] for c in checks if c["required"] and not c["ok"]]
    optional_failed = [c["name"] for c in checks if not c["required"] and not c["ok"]]
    return {
        "campaign_id": plan.campaign_id,
        "plan": str(plan.path),
        "jobs_total": len(jobs),
        "checks": checks,
        "prepared": prepared,
        "failed_required": required_failed,
        "failed_optional": optional_failed,
        "validation": rep,
        "ok": not required_failed,
    }
