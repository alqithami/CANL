"""Private, byte-bound frozen AST extraction for the raw AudioCaps follow-up.

This module never downloads data or models and does not import the study runner.
The caller supplies the complete frozen input binding and canonical WAV rows.
Completed four-clip batches are committed independently, so an interrupted
uncommitted batch is recomputed rather than treated as extracted evidence.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time
import uuid
from collections import defaultdict
from typing import Any, Mapping, Sequence


SCHEMA = "audiocaps-raw-frozen-ast-v1"
AST_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 163840
BATCH_SIZE = 4
TOKEN_COUNT = 64
FEATURE_DIM = 768
EXTRACTION = {
    "model": AST_NAME,
    "model_path": "assets/ast",
    "model_local_files_only": True,
    "trust_remote_code": False,
    "model_forward_dtype": "float32",
    "model_frozen": True,
    "AST_forward_batch_size": BATCH_SIZE,
    "target_sr": SAMPLE_RATE,
    "source_audio_format": "original encoded WAV bytes unchanged",
    "canonical_channels": 1,
    "channel_mix": "soundfile float32 always_2d then arithmetic mean(axis=1), exactly the retained loader",
    "resampling": "scipy.signal.resample_poly(16000/gcd, source_sr/gcd).astype(float32), exactly the retained loader",
    "canonical_whole_waveform_hash": "SHA256(struct.pack('<I',16000) + contiguous little-endian float32 mono bytes), checked against frozen manifest",
    "amplitude_normalization": "none; retain canonical waveform amplitudes",
    "window_samples": WINDOW_SAMPLES,
    "window_seconds": 10.24,
    "clips_longer_than_window": "first 163840 samples only",
    "shorter_clips": "retain actual samples; frozen AST extractor pads its spectrogram",
    "drop_model_special_tokens": 2,
    "pooling": "adaptive_avg_pool1d over patch tokens",
    "tokens_per_window": TOKEN_COUNT,
    "max_windows": 1,
    "feature_dim": FEATURE_DIM,
    "cache_dtype": "float16",
    "n_tokens_dtype": "int64",
    "zero_waveforms": "retain unchanged and report; exempt only zero groups from cross-split duplicate rejection",
    "normalized_first_window_hash": "16000:(sample_count,):float32le + contiguous canonical first-window bytes; signed zeros canonicalized only for hashing",
    "official_test_feature_extraction": "fixed pretrained AST only; no task training or caption evaluation",
    "new_extraction": True,
}


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _read(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _sync_dir(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Any, *, immutable: bool = False) -> None:
    payload = _json_bytes(value)
    if immutable and path.exists():
        _require(path.read_bytes() == payload, "Frozen metadata differs: " + str(path))
        return
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npy(path: Path, value: Any) -> None:
    import numpy as np

    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _new_memmap(path: Path, shape: tuple[int, ...]) -> Any:
    """Publish a complete NPY header and file extent before writing any batch."""
    import numpy as np

    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        value = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=shape)
        value.flush()
        del value
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return np.load(path, mmap_mode="r+", allow_pickle=False)


@contextlib.contextmanager
def _locked(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".feature-extraction.lock").open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _source_path(root: Path, row: Mapping[str, Any]) -> Path:
    source = Path(str(row["path"]))
    if not source.is_absolute():
        source = root / source
    _require(source.suffix.lower() == ".wav", "Only canonical WAV input is allowed: " + str(source))
    _require(source.is_file(), "Missing canonical waveform: " + str(source))
    return source


def _verify_sources(root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Check every byte-bound source, including rows from already committed batches."""
    _require(bool(rows), "An empty feature cache is not a valid study input")
    identities = set()
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), "Invalid manifest row at " + str(index))
        identity = str(row["id"])
        _require(identity and identity not in identities, "Duplicate or empty manifest ID: " + identity)
        identities.add(identity)
        _require(row["split"] in ("train", "valid", "test"), "Invalid split: " + identity)
        expected = row.get("sha256")
        _require(isinstance(expected, str) and len(expected) == 64 and
                 all(c in "0123456789abcdef" for c in expected), "Missing/invalid source SHA256: " + identity)
        _row_canonical_sha(row)
        _require(_sha(_source_path(root, row)) == expected,
                 "Canonical recording changed since input freeze: " + identity)
        if (index + 1) % 2000 == 0:
            print("FEATURE SOURCE VERIFICATION: " + str(index + 1) + "/" + str(len(rows)), flush=True)


def _row_canonical_sha(row: Mapping[str, Any]) -> str:
    values = [row[key] for key in ("canonical_sha256", "pcm_float32_sha256") if key in row]
    _require(bool(values) and all(isinstance(value, str) and len(value) == 64 and
             all(c in "0123456789abcdef" for c in value) for value in values) and
             len(set(values)) == 1, "Missing/invalid canonical waveform SHA256: " + str(row["id"]))
    return values[0]


def _declared_ast_files(binding: Mapping[str, Any]) -> Mapping[str, str] | None:
    """Recognize the study's frozen-model inventory without importing its runner."""
    assets = binding.get("assets")
    if isinstance(assets, Mapping):
        models = assets.get("models")
        if isinstance(models, Mapping):
            record = models.get(AST_NAME)
            if isinstance(record, Mapping) and isinstance(record.get("files"), Mapping):
                return record["files"]
    return None


def _asset_inventory(root: Path, binding: Mapping[str, Any]) -> dict[str, str]:
    asset_root = root / "assets" / "ast"
    _require(asset_root.is_dir(), "Missing local frozen AST assets: " + str(asset_root))
    result = {}
    for source in sorted(asset_root.rglob("*")):
        _require(not source.is_symlink(), "AST assets must be local regular files: " + str(source))
        if source.is_file():
            result[source.relative_to(asset_root).as_posix()] = _sha(source)
    for name in ("config.json", "preprocessor_config.json"):
        _require(name in result, "Missing AST asset: " + name)
    _require(any(name in result for name in ("model.safetensors", "pytorch_model.bin",
                                            "model.safetensors.index.json", "pytorch_model.bin.index.json")),
             "Missing local AST weights")
    declared = _declared_ast_files(binding)
    _require(declared is not None, "Full input binding lacks the frozen AST file inventory")
    _require(set(result) == set(declared), "Local AST asset inventory differs from input binding")
    for name, expected in declared.items():
        relative = Path(str(name))
        _require(not relative.is_absolute() and ".." not in relative.parts,
                 "Unsafe bound AST asset filename")
        _require(result.get(relative.as_posix()) == expected,
                 "AST asset differs from frozen input binding: " + str(name))
    return result


def _extraction_binding(root: Path, rows: Sequence[Mapping[str, Any]],
                        binding: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(binding, Mapping), "Feature extraction requires the full input-binding JSON object")
    # Serialize now, so caller mutation and non-JSON objects cannot alter a freeze.
    frozen_input = json.loads(_json_bytes(dict(binding)))
    frozen_rows = json.loads(_json_bytes(list(rows)))
    return {
        "schema": SCHEMA,
        "input_binding": frozen_input,
        "input_binding_sha256": _digest(frozen_input),
        "ordered_full_rows_sha256": _digest(frozen_rows),
        "n_clips": len(rows),
        "extraction": json.loads(_json_bytes(EXTRACTION)),
        "ast_assets": _asset_inventory(root, frozen_input),
        "feature_helper_sha256": _sha(Path(__file__).resolve()),
    }


def _array_hash(value: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _load_first_window(root: Path, row: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    path = _source_path(root, row)
    # Recheck immediately before decoding as well as during the complete preflight.
    _require(_sha(path) == row["sha256"], "Canonical waveform changed before extraction: " + str(row["id"]))
    with sf.SoundFile(str(path), mode="r") as stream:
        source_sr = int(stream.samplerate)
        _require(source_sr > 0 and stream.channels > 0 and
                 8 <= stream.frames / source_sr <= 12,
                 "Input must be an eligible original WAV: " + str(row["id"]))
        decoded = stream.read(dtype="float32", always_2d=True)
    _require(np.isfinite(decoded).all(), "Nonfinite decoded source WAV: " + str(row["id"]))
    # These are precisely load_audio_16k's operations; resample the complete clip
    # before slicing, since a first-window-only resample changes boundary samples.
    canonical = decoded.mean(axis=1)
    if source_sr != SAMPLE_RATE:
        divisor = math.gcd(source_sr, SAMPLE_RATE)
        canonical = resample_poly(canonical, SAMPLE_RATE // divisor, source_sr // divisor).astype(np.float32)
    canonical = np.ascontiguousarray(canonical, dtype="<f4")
    _require(np.isfinite(canonical).all(), "Nonfinite canonical source WAV: " + str(row["id"]))
    canonical_sha = hashlib.sha256(struct.pack("<I", SAMPLE_RATE) + canonical.tobytes()).hexdigest()
    _require(canonical_sha == _row_canonical_sha(row),
             "Retained loader canonical waveform differs from frozen manifest: " + str(row["id"]))
    canonical_zero = not bool(np.any(canonical))
    for key in ("canonical_all_zero", "allzero", "all_zero_waveform"):
        if key in row:
            _require(isinstance(row[key], bool) and row[key] == canonical_zero,
                     "Canonical all-zero policy flag differs: " + str(row["id"]))
    wave = np.ascontiguousarray(canonical[:WINDOW_SAMPLES], dtype="<f4")
    _require(8 * SAMPLE_RATE <= wave.size <= WINDOW_SAMPLES and np.isfinite(wave).all(),
             "Invalid first-window waveform: " + str(row["id"]))
    zero = not bool(np.any(wave))
    # Signed zero carries no acoustic information; canonicalize only the hash copy.
    normalized = wave.copy()
    normalized[normalized == 0] = 0.0
    header = ("16000:" + str(normalized.shape) + ":float32le").encode("ascii")
    record = {"id": str(row["id"]), "split": str(row["split"]),
              "file_sha256": str(row["sha256"]),
              "canonical_sha256": canonical_sha, "canonical_all_zero": canonical_zero,
              "preprocessed_sha256": hashlib.sha256(header + normalized.tobytes()).hexdigest(),
              "samples": int(wave.size), "all_zero": zero}
    return wave, record


def _check_batch_marker(record: Mapping[str, Any], start: int, end: int,
                        rows: Sequence[Mapping[str, Any]], features: Any,
                        cache_binding: str) -> list[dict[str, Any]]:
    import numpy as np

    _require(record.get("schema") == SCHEMA and record.get("cache_binding") == cache_binding and
             record.get("start") == start and record.get("end") == end,
             "Feature batch binding or extent differs at " + str(start))
    inputs = record.get("inputs")
    _require(isinstance(inputs, list) and len(inputs) == end - start,
             "Feature batch input count differs at " + str(start))
    for source, saved in zip(rows[start:end], inputs):
        _require(saved.get("id") == str(source["id"]) and saved.get("split") == source["split"] and
                 saved.get("file_sha256") == source["sha256"] and
                 saved.get("canonical_sha256") == _row_canonical_sha(source) and
                 isinstance(saved.get("canonical_all_zero"), bool) and
                 isinstance(saved.get("all_zero"), bool) and
                 isinstance(saved.get("samples"), int) and 8 * SAMPLE_RATE <= saved["samples"] <= WINDOW_SAMPLES and
                 isinstance(saved.get("preprocessed_sha256"), str) and len(saved["preprocessed_sha256"]) == 64,
                 "Saved feature input identity differs at " + str(start))
    _require(_array_hash(features[start:end]) == record.get("features_sha256") and
             np.isfinite(features[start:end]).all(), "Committed feature batch changed at " + str(start))
    return inputs


def _audit_inputs(records: Sequence[Mapping[str, Any]], cache_binding: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if not record["all_zero"]:
            groups[str(record["preprocessed_sha256"])].append(record)
    crossing = {key: group for key, group in sorted(groups.items())
                if len({record["split"] for record in group}) > 1}
    return {"schema": SCHEMA, "cache_binding": cache_binding,
            "recordings": len(records),
            "all_zero_ids": [record["id"] for record in records if record["all_zero"]],
            "all_zero_count": sum(bool(record["all_zero"]) for record in records),
            "canonical_all_zero_ids": [record["id"] for record in records if record["canonical_all_zero"]],
            "canonical_all_zero_count": sum(bool(record["canonical_all_zero"]) for record in records),
            "normalized_first_window_sha256_records": list(records),
            "nonzero_exact_cross_split_groups": crossing,
            "test_features_are_fixed_backbone_only": True,
            "near_duplicates_not_certified": True}


def _load_ast(root: Path, device: Any) -> tuple[Any, Any]:
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    local = str(root / "assets" / "ast")
    extractor = AutoFeatureExtractor.from_pretrained(local, local_files_only=True,
                                                    trust_remote_code=False)
    model = AutoModel.from_pretrained(local, local_files_only=True,
                                      trust_remote_code=False).float().to(device).eval()
    model.requires_grad_(False)
    _require(int(model.config.hidden_size) == FEATURE_DIM and
             int(getattr(extractor, "max_length", 1024)) * 160 == WINDOW_SAMPLES,
             "Local AST feature dimensions differ from frozen protocol")
    _require(all(parameter.dtype == torch.float32 for parameter in model.parameters()),
             "Frozen AST parameters must be float32")
    return extractor, model


def _verify_locked(root: Path, rows: Sequence[Mapping[str, Any]],
                   extraction_binding: Mapping[str, Any], *, verify_sources: bool) -> dict[str, Any]:
    import numpy as np

    directory = root / "features"
    ready = _read(root / "FEATURES_READY.json")
    cache_binding = _digest(extraction_binding)
    _require(ready.get("schema") == SCHEMA and ready.get("status") == "READY" and
             ready.get("cache_binding") == cache_binding and ready.get("binding") == cache_binding and
             ready.get("new_extraction") is True and ready.get("n_clips") == len(rows),
             "Frozen feature cache binding differs")
    expected_files = {"features.npy", "n_tokens.npy", "meta.json", "EXTRACTION_BINDING.json",
                      "EXTRACTION_REPORT.json", "PREPROCESSED_INPUT_AUDIT.json", "PROGRESS.json"}
    files = ready.get("files", {})
    _require(isinstance(files, dict) and set(files) == expected_files,
             "Frozen feature payload inventory differs")
    for name, expected in files.items():
        path = directory / name
        _require(path.is_file() and _sha(path) == expected, "Frozen feature payload changed: " + name)
    _require(_read(directory / "EXTRACTION_BINDING.json") == extraction_binding,
             "Full feature extraction binding changed")
    if verify_sources:
        _verify_sources(root, rows)
    features = np.load(directory / "features.npy", mmap_mode="r", allow_pickle=False)
    counts = np.load(directory / "n_tokens.npy", mmap_mode="r", allow_pickle=False)
    _require(features.shape == (len(rows), TOKEN_COUNT, FEATURE_DIM) and features.dtype == np.float16,
             "Frozen feature shape or dtype differs")
    _require(counts.shape == (len(rows),) and counts.dtype == np.int64 and np.all(counts == TOKEN_COUNT),
             "Frozen feature token counts differ")
    records = []
    batch_dir = directory / "batches"
    expected_markers = set()
    marker_hashes = {}
    for start in range(0, len(rows), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(rows))
        name = "%09d.json" % start
        expected_markers.add(name)
        marker = batch_dir / name
        _require(marker.is_file(), "Missing committed feature batch: " + name)
        records.extend(_check_batch_marker(_read(marker), start, end, rows, features, cache_binding))
        marker_hashes[name] = _sha(marker)
    _require({path.name for path in batch_dir.glob("*.json")} == expected_markers,
             "Unexpected feature batch marker")
    report = _read(directory / "EXTRACTION_REPORT.json")
    _require(report.get("batch_marker_sha256") == marker_hashes and
             report.get("cache_binding") == cache_binding and report.get("n_clips") == len(rows) and
             report.get("completed_batches") == len(expected_markers) and report.get("status") == "COMPLETE",
             "Feature extraction report differs")
    audit = _audit_inputs(records, cache_binding)
    _require(_read(directory / "PREPROCESSED_INPUT_AUDIT.json") == audit and
             not audit["nonzero_exact_cross_split_groups"], "First-window preprocessing audit differs or leaks across splits")
    progress = _read(directory / "PROGRESS.json")
    _require(progress.get("cache_binding") == cache_binding and progress.get("next_row") == len(rows) and
             progress.get("completed_batches") == len(expected_markers), "Feature progress is incomplete")
    info = _read(directory / "meta.json")
    _require(info == _metadata(len(rows), cache_binding), "Frozen feature metadata differs")
    # Runtime absolute paths are returned, never frozen into portable metadata.
    return {**info, "backbone": str(root / "assets" / "ast"), "cache_dir": str(directory),
            "features": features, "rows": list(rows), "ready": ready}


def _metadata(n_rows: int, cache_binding: str) -> dict[str, Any]:
    return {"schema": SCHEMA, "cache_binding": cache_binding,
            "backbone": "assets/ast", "dim": FEATURE_DIM,
            "tokens_per_window": TOKEN_COUNT, "max_windows": 1,
            "n_clips": n_rows, "cache_dir": "features"}


def verify_features(root: str | Path, rows: Sequence[Mapping[str, Any]],
                    binding: Mapping[str, Any]) -> dict[str, Any]:
    """Verify frozen inputs and the complete cache; return CaptioningJob feature info.

    The ``features`` value is a read-only ``np.load(..., mmap_mode='r')`` array.
    The returned runtime ``cache_dir`` also supports caenl.audio.data.load_n_tokens.
    No model forward, download, or scientific optimizer update occurs here.
    """
    root = Path(root).resolve()
    rows = json.loads(_json_bytes(list(rows)))
    with _locked(root):
        complete_binding = _extraction_binding(root, rows, binding)
        return _verify_locked(root, rows, complete_binding, verify_sources=True)


def extract_features(root: str | Path, rows: Sequence[Mapping[str, Any]],
                     binding: Mapping[str, Any], device: Any) -> dict[str, Any]:
    """Extract/resume newly generated AST features; return FEATURES_READY JSON.

    Cache identity binds the *entire* supplied JSON input binding, full ordered
    manifest rows, actual local model asset bytes, preprocessing, and helper code.
    A cache from any other experiment or old path/count-only cache is rejected.
    """
    import numpy as np

    root = Path(root).resolve()
    rows = json.loads(_json_bytes(list(rows)))
    with _locked(root):
        started = time.monotonic()
        complete_binding = _extraction_binding(root, rows, binding)
        cache_binding = _digest(complete_binding)
        _verify_sources(root, rows)
        if (root / "FEATURES_READY.json").exists():
            return _verify_locked(root, rows, complete_binding, verify_sources=False)["ready"]
        directory = root / "features"
        directory.mkdir(exist_ok=True)
        binding_path = directory / "EXTRACTION_BINDING.json"
        if not binding_path.exists():
            # A kill before the very first atomic binding rename can leave only
            # its temporary file. No payload was ever authorized by that file.
            pending = list(directory.iterdir())
            _require(all(path.is_file() and not path.is_symlink() and
                         path.name.startswith("EXTRACTION_BINDING.json.tmp-") for path in pending),
                     "Refusing an unbound or historical feature cache")
            for path in pending:
                path.unlink()
        _atomic_json(binding_path, complete_binding, immutable=True)
        batches = directory / "batches"
        batches.mkdir(exist_ok=True)
        feature_path = directory / "features.npy"
        shape = (len(rows), TOKEN_COUNT, FEATURE_DIM)
        if feature_path.exists():
            features = np.load(feature_path, mmap_mode="r+", allow_pickle=False)
            _require(features.shape == shape and features.dtype == np.float16,
                     "Partial feature shape or dtype differs")
        else:
            _require(not any(batches.glob("*.json")), "Batch markers exist without feature payload")
            features = _new_memmap(feature_path, shape)
        allowed = {"%09d.json" % start for start in range(0, len(rows), BATCH_SIZE)}
        _require({path.name for path in batches.glob("*.json")} <= allowed,
                 "Unexpected feature batch marker")
        progress_path = directory / "PROGRESS.json"
        if progress_path.exists():
            progress = _read(progress_path)
            cursor = progress.get("next_row")
            _require(progress.get("cache_binding") == cache_binding and isinstance(cursor, int) and
                     0 <= cursor <= len(rows) and (cursor == len(rows) or cursor % BATCH_SIZE == 0) and
                     progress.get("completed_batches") == (cursor + BATCH_SIZE - 1) // BATCH_SIZE,
                     "Partial feature progress binding/extent differs")
            for start in range(0, cursor, BATCH_SIZE):
                _require((batches / ("%09d.json" % start)).is_file(),
                         "Committed progress references a missing feature batch")
        extractor = model = None
        records = []
        marker_hashes = {}
        committed = 0
        for start in range(0, len(rows), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(rows))
            marker = batches / ("%09d.json" % start)
            if marker.exists():
                inputs = _check_batch_marker(_read(marker), start, end, rows, features, cache_binding)
            else:
                import torch

                if model is None:
                    extractor, model = _load_ast(root, device)
                waves, inputs = [], []
                for row in rows[start:end]:
                    wave, record = _load_first_window(root, row)
                    waves.append(wave)
                    inputs.append(record)
                device_type = torch.device(device).type
                # Explicitly disable enclosing autocast: this backbone protocol is FP32.
                with torch.no_grad(), torch.autocast(device_type=device_type, enabled=False):
                    data = extractor(waves, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                    tokens = model(input_values=data["input_values"].to(device=device, dtype=torch.float32)).last_hidden_state[:, 2:]
                    _require(tokens.ndim == 3 and tokens.shape[0] == len(waves) and
                             tokens.shape[1] > 0 and tokens.shape[2] == FEATURE_DIM and
                             tokens.dtype == torch.float32 and bool(torch.isfinite(tokens).all()),
                             "Invalid FP32 frozen AST patch features")
                    pooled = torch.nn.functional.adaptive_avg_pool1d(tokens.transpose(1, 2), TOKEN_COUNT).transpose(1, 2)
                    values = pooled.float().cpu().numpy().astype(np.float16)
                _require(values.shape == (end - start, TOKEN_COUNT, FEATURE_DIM) and
                         np.isfinite(values).all(), "Nonfinite/invalid pooled frozen AST features")
                features[start:end] = values
                features.flush()
                with feature_path.open("rb") as stream:
                    os.fsync(stream.fileno())
                _atomic_json(marker, {"schema": SCHEMA, "cache_binding": cache_binding,
                                      "start": start, "end": end,
                                      "features_sha256": _array_hash(features[start:end]),
                                      "inputs": inputs}, immutable=True)
            records.extend(inputs)
            marker_hashes[marker.name] = _sha(marker)
            committed += 1
            _atomic_json(progress_path, {"schema": SCHEMA, "cache_binding": cache_binding,
                                        "next_row": end, "completed_batches": committed,
                                        "batch_size": BATCH_SIZE})
            if committed % 64 == 0 or end == len(rows):
                print("FROZEN AST FEATURES: " + str(end) + "/" + str(len(rows)), flush=True)
        del model, extractor
        audit = _audit_inputs(records, cache_binding)
        _atomic_json(directory / "PREPROCESSED_INPUT_AUDIT.json", audit, immutable=True)
        _require(not audit["nonzero_exact_cross_split_groups"],
                 "Nonzero canonical first windows are exact cross-split duplicates; no training is permitted")
        counts_path = directory / "n_tokens.npy"
        if counts_path.exists():
            counts = np.load(counts_path, allow_pickle=False)
            _require(counts.shape == (len(rows),) and counts.dtype == np.int64 and
                     np.all(counts == TOKEN_COUNT), "Existing frozen token counts differ")
        else:
            _atomic_npy(counts_path, np.full(len(rows), TOKEN_COUNT, dtype=np.int64))
        _atomic_json(directory / "meta.json", _metadata(len(rows), cache_binding), immutable=True)
        report = {"schema": SCHEMA, "status": "COMPLETE", "cache_binding": cache_binding,
                  "n_clips": len(rows), "completed_batches": committed,
                  "batch_marker_sha256": marker_hashes, "new_extraction": True,
                  "frozen_backbone_only": True, "all_zero_count": audit["all_zero_count"],
                  "canonical_all_zero_count": audit["canonical_all_zero_count"],
                  "nonzero_exact_cross_split_group_count": 0}
        _atomic_json(directory / "EXTRACTION_REPORT.json", report, immutable=True)
        file_names = ("features.npy", "n_tokens.npy", "meta.json", "EXTRACTION_BINDING.json",
                      "EXTRACTION_REPORT.json", "PREPROCESSED_INPUT_AUDIT.json", "PROGRESS.json")
        files = {name: _sha(directory / name) for name in file_names}
        ready = {"schema": SCHEMA, "status": "READY", "binding": cache_binding,
                 "cache_binding": cache_binding,
                 "input_binding_sha256": complete_binding["input_binding_sha256"],
                 "n_clips": len(rows), "new_extraction": True,
                 "files": files,
                 "feature_extraction_wall_seconds_this_attempt": time.monotonic() - started,
                 "timing_includes_resume_verification_and_final_hashing": True}
        _atomic_json(root / "FEATURES_READY.json", ready, immutable=True)
        # Full validation precedes successful return; extraction never counts as training.
        return _verify_locked(root, rows, complete_binding, verify_sources=False)["ready"]
