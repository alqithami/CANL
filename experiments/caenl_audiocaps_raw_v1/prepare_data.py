#!/usr/bin/env python3
"""CPU-only, two-pass AudioCaps V1 raw-source follow-up preparation.

The source is the same official AudioCaps V1 membership and captions, with a new
raw-audio source protocol. This does not replace any historical dataset or study.
No GPU, model, network, audio transcoding, amplitude adjustment, or padding is used.

Run: python prepare_data.py --root /srv/iga/caenl-audiocaps-raw-v1
Inspect without starting: python prepare_data.py --root ROOT --status
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import time
from typing import Any, Callable
import zipfile
import zlib

VERSION = "audiocaps-raw-source-followup-v1"
CAPTIONS_COMMIT = "d004db3ea1b01cf4fd0347dd8d27db90cadc8809"
PINNED = {
    "train": {"segments": 49838, "references": 1, "sha256": "c0c5223db682b3bf724ce7e7ce58d5b36929f74572e8526a7211f92d2eef7c8e"},
    "val": {"segments": 495, "references": 5, "sha256": "dab1c96641d5f3053ddb99dca3949450da9a75737bda53e11cc0aa8b102be0c3"},
    "test": {"segments": 975, "references": 5, "sha256": "b91c4b7ded2f4f6e7db5b9c4983dc1e1dca3d556f505b61e3fd65cac7e1c638a"},
}
SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}
EXPECTED_RECORDS = 51308
LOADER_SHA256 = "15587567bf11ddc5fe089c12cf23a7ace4ed44fb63c3965b758cbd499323ec20"
CHUNK = 1 << 20
CHECKPOINT_EVERY = 250
YT_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
POLICY = {
    "identity": "AudioCaps V1 raw-source follow-up; not a new dataset or historical replacement",
    "selection": "all and only pinned official V1 train/val/test segment memberships; all original references retained",
    "duration_seconds_inclusive": [8.0, 12.0],
    "decode": "soundfile.read(dtype='float32', always_2d=True); finite decoded and canonical samples required",
    "canonicalization": "arithmetic mean(axis=1); scipy.signal.resample_poly(target_sr/gcd, source_sr/gcd).astype(float32); target 16000 Hz; no amplitude adjustment, padding or truncation",
    "canonical_hash": "sha256(struct.pack('<I',16000) + C-contiguous little-endian float32 mono sample bytes)",
    "silence": "retain and report all-zero canonical audio; all-zero hashes do not trigger duplicate exclusion",
    "duplicates": "exclude every member of every cross-split nonzero canonical-hash group, including otherwise ineligible members",
    "youtube_ids": "exclude every member of every cross-split YouTube-ID group",
    "outcomes": "eligibility fixed before any model outcome; no random, best or outcome-dependent selection",
    "output": "original encoded WAV member bytes unchanged; no re-encoding",
    "minimum_coverage_per_split": 0.8,
    "loader_reference_sha256": LOADER_SHA256,
}


class PreparationBlocked(RuntimeError):
    """An explicit, non-destructive preparation stop requiring correction."""


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _hash_object(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def sha_file(path: Path, progress: Callable[[int], None] | None = None) -> str:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
            count += len(block)
            if progress and count % (256 * CHUNK) < CHUNK:
                progress(count)
    if progress:
        progress(count)
    return digest.hexdigest()


def safe_output(root: Path, relative: str | Path) -> Path:
    """Return a rooted output, refusing traversal and every symlink component."""
    root = root.absolute()
    relative = Path(relative)
    if relative.is_absolute() or not relative.parts or any(p in ("..", "") for p in relative.parts):
        raise PreparationBlocked(f"unsafe output path: {relative}")
    if root.is_symlink():
        raise PreparationBlocked(f"symlink output root refused: {root}")
    candidate = root / relative
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise PreparationBlocked(f"symlink output component refused: {current}")
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise PreparationBlocked(f"output escapes root: {candidate}")
    return candidate


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, _json_bytes(value), replace=True)


def _atomic_bytes(path: Path, payload: bytes, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise PreparationBlocked(f"symlink destination refused: {path}")
    if path.exists() and not replace:
        if path.read_bytes() != payload:
            raise PreparationBlocked(f"existing different file refused: {path}")
        return
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    if temp.exists() or temp.is_symlink():
        raise PreparationBlocked(f"existing temporary destination refused: {temp}")
    try:
        with temp.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temp, path)
        else:
            # link is an atomic no-clobber publish, unlike rename/replace.
            try:
                os.link(temp, path)
            except FileExistsError:
                if path.is_symlink() or path.read_bytes() != payload:
                    raise PreparationBlocked(f"concurrent different file refused: {path}")
    finally:
        temp.unlink(missing_ok=True)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PreparationBlocked(f"cannot read required JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreparationBlocked(f"JSON object required: {path}")
    return value


def _load_annotations(captions_dir: Path) -> tuple[list[dict], dict]:
    records: list[dict] = []
    report = {}
    for official_split, pin in PINNED.items():
        source = captions_dir / f"{official_split}.csv"
        if source.is_symlink() or not source.is_file():
            raise PreparationBlocked(f"missing regular pinned caption CSV: {source}")
        observed = sha_file(source)
        if observed != pin["sha256"]:
            raise PreparationBlocked(f"pinned caption SHA256 mismatch: {source.name}: {observed}")
        clips: dict[tuple[str, int], dict] = {}
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"youtube_id", "start_time", "caption", "audiocap_id"}
            if not required.issubset(reader.fieldnames or []):
                raise PreparationBlocked(f"required caption columns absent: {source}")
            for line, row in enumerate(reader, 2):
                yt, start = row["youtube_id"], row["start_time"]
                if not YT_RE.fullmatch(yt or "") or not re.fullmatch(r"[0-9]+", start or ""):
                    raise PreparationBlocked(f"invalid caption ID/start: {source}:{line}")
                try:
                    cap_id = int(row["audiocap_id"])
                except (TypeError, ValueError) as exc:
                    raise PreparationBlocked(f"invalid audiocap_id: {source}:{line}") from exc
                # Preserve caption text exactly as present in the pinned CSV.
                caption = row["caption"]
                if not isinstance(caption, str) or not caption.strip():
                    raise PreparationBlocked(f"empty original reference: {source}:{line}")
                start_int = int(start)
                split = SPLIT_MAP[official_split]
                item = clips.setdefault((yt, start_int), {
                    "id": f"{split}/{yt}_{start_int}", "youtube_id": yt,
                    "start_time": start_int, "split": split, "official_split": official_split,
                    "filename": f"{yt}_{start_int}.wav", "captions": [], "audiocap_ids": [],
                })
                item["captions"].append(caption)
                item["audiocap_ids"].append(cap_id)
        if len(clips) != pin["segments"] or any(len(v["captions"]) != pin["references"] for v in clips.values()):
            raise PreparationBlocked(f"pinned membership/reference count mismatch: {source}")
        records.extend(clips.values())
        report[official_split] = {"path": str(source.resolve()), "sha256": observed,
                                  "segment_count": len(clips), "caption_rows": sum(len(v["captions"]) for v in clips.values()),
                                  "references_per_segment": pin["references"]}
    records.sort(key=lambda r: r["id"])
    if len(records) != EXPECTED_RECORDS or len({r["id"] for r in records}) != EXPECTED_RECORDS:
        raise PreparationBlocked("unexpected total official AudioCaps V1 memberships")
    return records, report


def canonical_audio(raw_bytes: bytes) -> dict:
    """Audit the unchanged loader's canonical mono signal, without retaining it."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    wav, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=True)
    frames, channels = wav.shape
    if sr <= 0 or channels <= 0:
        raise ValueError("invalid decoded sample rate/channels")
    finite = bool(np.isfinite(wav).all())
    info = {"sr": int(sr), "channels": int(channels), "frames": int(frames),
            "duration": float(frames / sr), "finite": finite,
            "canonical_sr": 16000, "canonical_sha256": None,
            "canonical_all_zero": None, "allzero": None}
    if not finite:
        return info
    wav = wav.mean(axis=1)
    if sr != 16000:
        g = math.gcd(int(sr), 16000)
        wav = resample_poly(wav, 16000 // g, sr // g).astype(np.float32)
    info["canonical_finite"] = bool(np.isfinite(wav).all())
    info["canonical_frames"] = int(wav.shape[0])
    if not info["canonical_finite"]:
        return info
    canonical = np.ascontiguousarray(wav, dtype="<f4")
    digest = hashlib.sha256(struct.pack("<I", 16000))
    digest.update(canonical.tobytes(order="C"))
    allzero = bool(not np.any(canonical != 0))
    info.update(canonical_sha256=digest.hexdigest(), canonical_all_zero=allzero, allzero=allzero)
    return info


def audit_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, metadata: dict) -> dict:
    record = dict(metadata)
    record.update(zip_member=info.filename, zip_compressed_bytes=info.compress_size,
                  zip_uncompressed_bytes=info.file_size, crc32_expected=f"{info.CRC:08x}",
                  crc32_actual=None, crc_verified=False, sha256=None, bytes=0,
                  canonical_sha256=None, canonical_all_zero=None, allzero=None,
                  errors=[], eligibility_reasons=[])
    raw = io.BytesIO()
    digest = hashlib.sha256()
    crc, size = 0, 0
    try:
        with archive.open(info, "r") as stream:
            for block in iter(lambda: stream.read(CHUNK), b""):
                digest.update(block)
                crc = zlib.crc32(block, crc)
                size += len(block)
                raw.write(block)
        record.update(sha256=digest.hexdigest(), bytes=size, crc32_actual=f"{crc & 0xffffffff:08x}",
                      crc_verified=(crc & 0xffffffff) == info.CRC and size == info.file_size)
        if not record["crc_verified"]:
            record["eligibility_reasons"].append("zip_crc_or_size_mismatch")
        else:
            try:
                record.update(canonical_audio(raw.getvalue()))
                if not record["finite"] or not record.get("canonical_finite", False):
                    record["eligibility_reasons"].append("nonfinite_decoded_or_canonical_audio")
                if not 8.0 <= record["duration"] <= 12.0:
                    record["eligibility_reasons"].append("duration_outside_8_to_12_seconds_inclusive")
                if record["frames"] == 0:
                    record["eligibility_reasons"].append("empty_decoded_audio")
            except Exception as exc:
                record["eligibility_reasons"].append("audio_decode_error")
                record["errors"].append({"stage": "decode", "type": type(exc).__name__, "message": str(exc)})
    except Exception as exc:
        record.update(bytes=size, crc32_actual=f"{crc & 0xffffffff:08x}")
        record["eligibility_reasons"].append("zip_member_read_or_crc_error")
        record["errors"].append({"stage": "zip_read", "type": type(exc).__name__, "message": str(exc)})
    record["base_eligible"] = not record["eligibility_reasons"]
    return record


def duplicate_exclusions(records: list[dict]) -> tuple[dict[str, list[str]], dict]:
    """Compute global groups before eligibility; exclude ALL cross-split members."""
    by_hash: dict[str, list[dict]] = defaultdict(list)
    by_youtube: dict[str, list[dict]] = defaultdict(list)
    allzero = []
    for record in records:
        by_youtube[record["youtube_id"]].append(record)
        if record.get("canonical_sha256"):
            if record.get("canonical_all_zero"):
                allzero.append(record["id"])
            else:
                by_hash[record["canonical_sha256"]].append(record)
    reasons: dict[str, list[str]] = defaultdict(list)
    canonical_groups, youtube_groups = [], []
    for digest, members in sorted(by_hash.items()):
        if len({r["split"] for r in members}) > 1:
            canonical_groups.append({"canonical_sha256": digest, "members": [{"id": r["id"], "split": r["split"], "youtube_id": r["youtube_id"]} for r in sorted(members, key=lambda r: r["id"])]})
            for r in members:
                reasons[r["id"]].append("cross_split_nonzero_canonical_duplicate")
    for yt, members in sorted(by_youtube.items()):
        if len({r["split"] for r in members}) > 1:
            youtube_groups.append({"youtube_id": yt, "members": [{"id": r["id"], "split": r["split"]} for r in sorted(members, key=lambda r: r["id"])]})
            for r in members:
                reasons[r["id"]].append("cross_split_youtube_id")
    return dict(reasons), {"cross_split_nonzero_canonical_group_count": len(canonical_groups),
                           "cross_split_nonzero_canonical_groups": canonical_groups,
                           "cross_split_youtube_id_count": len(youtube_groups), "cross_split_youtube_ids": youtube_groups,
                           "all_zero_canonical_count": len(allzero), "all_zero_canonical_ids": sorted(allzero)}


def _safe_member(info: zipfile.ZipInfo) -> None:
    name = info.filename
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or "\x00" in name or ((info.external_attr >> 16) & 0o170000) == 0o120000:
        raise PreparationBlocked(f"unsafe selected ZIP member: {name!r}")


def _inventory(archive: zipfile.ZipFile, selected: list[dict]) -> tuple[dict[str, zipfile.ZipInfo], dict]:
    wanted = {r["filename"] for r in selected}
    members = {}
    wavs = 0
    for info in archive.infolist():
        base = info.filename.rsplit("/", 1)[-1]
        wavs += int(not info.is_dir() and base.lower().endswith(".wav"))
        if not info.is_dir() and base in wanted:
            _safe_member(info)
            if base in members:
                raise PreparationBlocked(f"ambiguous selected ZIP basename: {base}")
            if info.flag_bits & 1:
                raise PreparationBlocked(f"encrypted selected ZIP member: {base}")
            members[base] = info
    return members, {"total": len(archive.infolist()), "wav": wavs,
                     "selected_unique_present": len(members), "selected_unique_missing": len(wanted - members.keys()),
                     "unselected_members_not_read": len(archive.infolist()) - len(members)}


def _write_verified_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, destination: Path, expected: dict) -> bool:
    """Publish only verified original encoded bytes; never clobber a different file."""
    if destination.is_symlink():
        raise PreparationBlocked(f"symlink destination refused: {destination}")
    if destination.exists():
        if not destination.is_file() or destination.stat().st_size != expected["bytes"] or sha_file(destination) != expected["sha256"]:
            raise PreparationBlocked(f"existing different audio refused: {destination}")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".part")
    # A prior interrupted member is never treated as complete or silently reused.
    if temp.is_symlink():
        raise PreparationBlocked(f"symlink partial audio refused: {temp}")
    if temp.exists():
        if not temp.is_file():
            raise PreparationBlocked(f"non-file partial audio refused: {temp}")
        if temp.stat().st_size == expected["bytes"] and sha_file(temp) == expected["sha256"]:
            try:
                os.link(temp, destination)
            except FileExistsError:
                return _write_verified_member(archive, info, destination, expected)
            temp.unlink()
            return True
        # Explicitly block rather than hiding a member restart after interruption.
        raise PreparationBlocked(f"incomplete audio member {temp}; preserve or remove this partial explicitly, then resume")
    digest = hashlib.sha256()
    crc, count = 0, 0
    with temp.open("xb") as target:
        with archive.open(info, "r") as source:
            for block in iter(lambda: source.read(CHUNK), b""):
                target.write(block)
                digest.update(block)
                crc = zlib.crc32(block, crc)
                count += len(block)
        target.flush()
        os.fsync(target.fileno())
    if count != expected["bytes"] or digest.hexdigest() != expected["sha256"] or (crc & 0xffffffff) != info.CRC:
        raise PreparationBlocked(f"second-pass member byte/CRC mismatch; preserved {temp}")
    try:
        os.link(temp, destination)
    except FileExistsError:
        if destination.is_symlink() or destination.stat().st_size != count or sha_file(destination) != digest.hexdigest():
            raise PreparationBlocked(f"concurrent different audio refused: {destination}")
    temp.unlink()
    return True


def _audit_resume(journal: Path, checkpoint: Path, binding_sha256: str, selected: list[dict]) -> tuple[list[dict], Any, dict]:
    """Recover only the durable, hash-bound journal prefix; report tail removal."""
    if journal.is_symlink() or checkpoint.is_symlink():
        raise PreparationBlocked("symlink audit journal/checkpoint refused")
    if not checkpoint.exists():
        if journal.exists() and journal.stat().st_size:
            raise PreparationBlocked("nonempty audit journal has no durable checkpoint; refusing hidden restart")
        if not journal.exists():
            with journal.open("xb"):
                pass
        _atomic_json(checkpoint, {"binding_sha256": binding_sha256, "committed_records": 0,
                                  "committed_bytes": 0, "audit_prefix_sha256": hashlib.sha256().hexdigest()})
    cp = _read_json(checkpoint)
    if cp.get("binding_sha256") != binding_sha256:
        raise PreparationBlocked("audit checkpoint source/policy binding differs")
    offset = cp.get("committed_bytes")
    count = cp.get("committed_records")
    if not isinstance(offset, int) or not isinstance(count, int) or not 0 <= count <= len(selected) or offset < 0:
        raise PreparationBlocked("invalid audit checkpoint counters")
    digest = hashlib.sha256()
    records = []
    with journal.open("rb") as stream:
        while stream.tell() < offset:
            line = stream.readline()
            if not line.endswith(b"\n") or stream.tell() > offset:
                raise PreparationBlocked("audit checkpoint does not end at complete JSONL record")
            digest.update(line)
            try:
                records.append(json.loads(line))
            except ValueError as exc:
                raise PreparationBlocked("invalid committed audit record") from exc
        if stream.tell() != offset:
            raise PreparationBlocked("audit journal shorter than durable checkpoint")
    if len(records) != count or digest.hexdigest() != cp.get("audit_prefix_sha256"):
        raise PreparationBlocked("committed audit journal checksum/count mismatch")
    for record, expected in zip(records, selected):
        if any(record.get(k) != expected[k] for k in expected):
            raise PreparationBlocked("committed audit membership/reference mismatch")
    discarded = journal.stat().st_size - offset
    if discarded < 0:
        raise PreparationBlocked("audit journal shorter than checkpoint")
    if discarded:
        with journal.open("r+b") as stream:
            stream.truncate(offset)
            stream.flush()
            os.fsync(stream.fileno())
    return records, digest, {"resumed_committed_records": count, "discarded_uncommitted_audit_bytes": discarded}


def prepare(root: Path, progress: Callable[[dict], None] | None = None) -> dict:
    """Prepare raw source data; return DATA_READY or raise PreparationBlocked.

    Partial first-pass journals resume from durable checkpoints. Existing retained
    WAVs are rehashed and reused; different existing files cause an explicit stop.
    """
    root = Path(root).expanduser().absolute()
    if root.is_symlink():
        raise PreparationBlocked("symlink run root refused")
    root.mkdir(parents=True, exist_ok=True)
    data = safe_output(root, "data")
    data.mkdir(parents=True, exist_ok=True)
    lock_path = safe_output(root, "data/PREPARE.lock")
    if lock_path.is_symlink():
        raise PreparationBlocked("symlink preparation lock refused")
    lock = lock_path.open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise PreparationBlocked("preparation already running; inspect --status") from exc
    status_path = safe_output(root, "data/STATUS.json")
    start = time.monotonic()
    status = {"version": VERSION, "state": "VERIFYING_SOURCE", "started_at": utc(), "updated_at": utc(),
              "pid": os.getpid(), "cpu_only": True, "gpu_used": False, "model_outcomes_used": False,
              "expected_records": EXPECTED_RECORDS}

    def update(state: str | None = None, **values: Any) -> None:
        if state:
            status["state"] = state
        status.update(values)
        status.update(updated_at=utc(), elapsed_seconds=round(time.monotonic() - start, 3))
        _atomic_json(status_path, status)
        if progress:
            progress(dict(status))

    try:
        update()
        incoming = safe_output(root, "incoming")
        archive_path = safe_output(root, "incoming/audiocaps_raw_audio.zip")
        if not archive_path.is_file():
            raise PreparationBlocked(f"raw source archive absent: {archive_path}")
        source_path = safe_output(root, "incoming/SOURCE_READY.json")
        source = _read_json(source_path)
        if str(source.get("status", source.get("state", ""))).upper() not in ("VERIFIED", "SOURCE_READY", "READY"):
            raise PreparationBlocked("SOURCE_READY is not VERIFIED")
        expected_hash = source.get("archive_sha256", source.get("sha256"))
        expected_size = source.get("archive_bytes", source.get("bytes", source.get("size_bytes")))
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(expected_hash or "")) or not isinstance(expected_size, int):
            raise PreparationBlocked("SOURCE_READY requires archive_sha256 and integer archive_bytes")
        if source.get("archive_path") and Path(source["archive_path"]).resolve() != archive_path.resolve():
            raise PreparationBlocked("SOURCE_READY archive_path differs from required incoming archive")
        before = archive_path.stat()
        source_stat = {"bytes": before.st_size, "mtime_ns": before.st_mtime_ns, "inode": before.st_ino}
        if before.st_size != expected_size:
            raise PreparationBlocked("raw archive size differs from SOURCE_READY")
        archive_sha256 = sha_file(archive_path, lambda n: update(source_hash_bytes_read=n, source_hash_bytes_total=expected_size))
        if archive_sha256 != str(expected_hash).lower():
            raise PreparationBlocked("full raw archive SHA256 differs from SOURCE_READY")
        if archive_path.stat().st_size != before.st_size or archive_path.stat().st_mtime_ns != before.st_mtime_ns:
            raise PreparationBlocked("raw archive changed during full source hashing")
        update("VERIFYING_CAPTIONS_AND_REFERENCE", archive_sha256=archive_sha256, archive_bytes=before.st_size)
        selected, annotations = _load_annotations(incoming / "official_captions")
        loader_candidates = [root / "reference/src/caenl/audio/data.py", root / "src/caenl/audio/data.py"]
        loader = next((p for p in loader_candidates if p.is_file() and not p.is_symlink()), None)
        if loader is None or sha_file(loader) != LOADER_SHA256:
            raise PreparationBlocked("unchanged pinned src/caenl/audio/data.py reference missing or SHA256 mismatch")
        import numpy as np
        import scipy
        import soundfile as sf
        versions = {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
                    "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__}
        binding = {"version": VERSION, "archive_sha256": archive_sha256, "archive_bytes": before.st_size,
                   "captions_commit": CAPTIONS_COMMIT, "caption_csv_sha256": {s: p["sha256"] for s, p in PINNED.items()},
                   "loader_sha256": LOADER_SHA256, "policy": POLICY,
                   "implementation_sha256": sha_file(Path(__file__).resolve()), "decoder_versions": versions}
        binding_sha256 = _hash_object(binding)
        _atomic_bytes(safe_output(root, "data/PREPARATION_BINDING.json"), _json_bytes({"binding_sha256": binding_sha256, **binding}))
        ready_path = safe_output(root, "data/DATA_READY.json")
        if ready_path.exists():
            ready = _read_json(ready_path)
            if ready.get("binding_sha256") != binding_sha256:
                raise PreparationBlocked("existing DATA_READY source/policy binding differs")
            for name, key in (("manifest.jsonl", "manifest_sha256"), ("manifest_meta.json", "manifest_meta_sha256"),
                              ("AUDIT.jsonl", "audit_sha256"), ("EXCLUSIONS.jsonl", "exclusions_sha256")):
                artifact = safe_output(root, "data/" + name)
                if not artifact.is_file() or sha_file(artifact) != ready.get(key):
                    raise PreparationBlocked(f"existing DATA_READY artifact mismatch: {name}")
            rows = [json.loads(line) for line in (data / "manifest.jsonl").read_text().splitlines()]
            for index, row in enumerate(rows, 1):
                target = safe_output(root, row["relative_path"])
                if str(target.resolve()) != row["path"] or not target.is_file() or target.stat().st_size != row["bytes"] or sha_file(target) != row["sha256"]:
                    raise PreparationBlocked(f"existing retained audio mismatch: {row['id']}")
                if index % CHECKPOINT_EVERY == 0:
                    update("VERIFYING_EXISTING_READY", verified_retained=index, total_retained=len(rows))
            update("DATA_READY", **{k: ready[k] for k in ("manifest", "manifest_sha256", "n_clips", "splits", "coverage", "binding_sha256")})
            return ready
        with zipfile.ZipFile(archive_path, "r") as archive:
            members, inventory = _inventory(archive, selected)
            journal = safe_output(root, "data/AUDIT.jsonl")
            checkpoint = safe_output(root, "data/AUDIT_CHECKPOINT.json")
            records, audit_digest, resume = _audit_resume(journal, checkpoint, binding_sha256, selected)
            update("AUDITING_CANONICAL", **resume, processed=len(records), archive_inventory=inventory, binding_sha256=binding_sha256)
            with journal.open("ab") as stream:
                def commit() -> None:
                    stream.flush()
                    os.fsync(stream.fileno())
                    _atomic_json(checkpoint, {"binding_sha256": binding_sha256, "committed_records": len(records),
                                              "committed_bytes": stream.tell(), "audit_prefix_sha256": audit_digest.hexdigest()})
                    update(processed=len(records), last_id=records[-1]["id"] if records else None)
                for metadata in selected[len(records):]:
                    info = members.get(metadata["filename"])
                    if info is None:
                        record = {**metadata, "zip_member": None, "sha256": None, "bytes": 0,
                                  "crc_verified": False, "canonical_sha256": None, "canonical_all_zero": None,
                                  "allzero": None, "base_eligible": False, "errors": [], "eligibility_reasons": ["missing_zip_member"]}
                    else:
                        record = audit_member(archive, info, metadata)
                    payload = _json_bytes(record)
                    stream.write(payload)
                    audit_digest.update(payload)
                    records.append(record)
                    if len(records) % CHECKPOINT_EVERY == 0:
                        commit()
                commit()
            if len(records) != EXPECTED_RECORDS:
                raise PreparationBlocked("first pass did not audit all official memberships")
            duplicates, duplicate_report = duplicate_exclusions(records)
            retained, exclusions = [], []
            reason_counts = Counter()
            for record in records:
                reasons = sorted(set(record["eligibility_reasons"] + duplicates.get(record["id"], [])))
                if reasons:
                    reason_counts.update(reasons)
                    exclusions.append({**record, "exclusion_reasons": reasons})
                else:
                    relative = f"data/audio/{record['split']}/{record['filename']}"
                    target = safe_output(root, relative)
                    row = {key: record[key] for key in ("id", "youtube_id", "start_time", "split", "captions", "audiocap_ids", "sha256", "bytes", "sr", "channels", "frames", "duration", "canonical_sha256", "canonical_all_zero", "allzero")}
                    row.update(path=str(target.resolve()), relative_path=relative, canonical_sr=16000)
                    retained.append(row)
            splits = {s: sum(r["split"] == s for r in retained) for s in ("train", "valid", "test")}
            official = {SPLIT_MAP[s]: pin["segments"] for s, pin in PINNED.items()}
            coverage = {s: splits[s] / official[s] for s in official}
            # Publish reviewable deterministic audit/exclusions before coverage gate.
            _atomic_bytes(safe_output(root, "data/EXCLUSIONS.jsonl"), b"".join(_json_bytes(r) for r in exclusions))
            selection = {"binding_sha256": binding_sha256, "official": official, "retained": splits, "coverage": coverage,
                         "excluded": len(exclusions), "exclusion_reason_counts": dict(sorted(reason_counts.items())),
                         **duplicate_report}
            _atomic_bytes(safe_output(root, "data/SELECTION_REPORT.json"), _json_bytes(selection))
            if any(c < 0.8 for c in coverage.values()):
                raise PreparationBlocked(f"fixed-policy retained coverage below 0.8: {coverage}")
            update("WRITING_RETAINED_ORIGINAL_BYTES", n_clips=len(retained), splits=splits, coverage=coverage,
                   excluded=len(exclusions), written=0, reused=0)
            written, reused = 0, 0
            for index, row in enumerate(retained, 1):
                info = members[f"{row['youtube_id']}_{row['start_time']}.wav"]
                target = safe_output(root, row["relative_path"])
                if _write_verified_member(archive, info, target, row):
                    written += 1
                else:
                    reused += 1
                if index % CHECKPOINT_EVERY == 0:
                    update(written=written, reused=reused, retained_verified=index, last_id=row["id"])
            update("FINALIZING", written=written, reused=reused, retained_verified=len(retained))
        after = archive_path.stat()
        if (after.st_size, after.st_mtime_ns, after.st_ino) != (before.st_size, before.st_mtime_ns, before.st_ino):
            raise PreparationBlocked("raw archive changed during preparation")
        # Recheck every pinned metadata input and the immutable loader before readiness.
        if any(sha_file(incoming / "official_captions" / (s + ".csv")) != p["sha256"] for s, p in PINNED.items()) or sha_file(loader) != LOADER_SHA256:
            raise PreparationBlocked("caption or pinned loader source changed during preparation")
        manifest_path = safe_output(root, "data/manifest.jsonl")
        _atomic_bytes(manifest_path, b"".join(_json_bytes(r) for r in retained))
        manifest_sha256 = sha_file(manifest_path)
        audit_sha256 = sha_file(data / "AUDIT.jsonl")
        exclusions_sha256 = sha_file(data / "EXCLUSIONS.jsonl")
        meta = {"dataset": "audiocaps", "protocol": VERSION, "identity": POLICY["identity"],
                "binding_sha256": binding_sha256, "source": {"archive_path": str(archive_path.resolve()),
                "archive_sha256": archive_sha256, "archive_bytes": before.st_size,
                "source_ready_sha256": sha_file(source_path), "stat": source_stat},
                "captions_source": {"repository": "https://github.com/cdjkim/audiocaps", "commit": CAPTIONS_COMMIT,
                                    "csv_sha256": {s: p["sha256"] for s, p in PINNED.items()}},
                "annotation_csvs": annotations, "official_clips": official, "policy": POLICY,
                "loader_reference": {"path": str(loader.resolve()), "sha256": LOADER_SHA256},
                "versions": versions, "n_clips": len(retained), "splits": splits, "coverage": coverage,
                "manifest_sha256": manifest_sha256, "audit_sha256": audit_sha256,
                "exclusions_sha256": exclusions_sha256, "audit_records": len(records),
                "crc_verified": sum(bool(r["crc_verified"]) for r in records),
                "excluded": len(exclusions), "exclusion_reason_counts": dict(sorted(reason_counts.items())),
                "retained_all_zero": sum(bool(r["allzero"]) for r in retained),
                "archive_inventory": inventory, **duplicate_report,
                "cpu_only": True, "gpu_used": False, "model_outcomes_used": False}
        meta_path = safe_output(root, "data/manifest_meta.json")
        _atomic_bytes(meta_path, _json_bytes(meta))
        ready = {"status": "DATA_READY", "state": "DATA_READY", "version": VERSION,
                 "binding_sha256": binding_sha256, "manifest": str(manifest_path.resolve()),
                 "manifest_sha256": manifest_sha256, "manifest_meta_sha256": sha_file(meta_path),
                 "audit_sha256": audit_sha256, "exclusions_sha256": exclusions_sha256,
                 "archive_sha256": archive_sha256, "archive_bytes": before.st_size,
                 "source_sha256": archive_sha256, "policy_sha256": _hash_object(POLICY),
                 "caption_csv_sha256": {s: p["sha256"] for s, p in PINNED.items()},
                 "n_clips": len(retained), "splits": splits, "coverage": coverage,
                 "audited_records": len(records), "excluded_records": len(exclusions),
                 "retained_all_zero": meta["retained_all_zero"],
                 "selection_report_sha256": sha_file(data / "SELECTION_REPORT.json"),
                 "original_bytes_verified": True, "eligibility_fixed_before_model_outcomes": True,
                 "cpu_only": True, "gpu_used": False, "completed_at": utc()}
        _atomic_bytes(ready_path, _json_bytes(ready))
        update("DATA_READY", manifest=ready["manifest"], manifest_sha256=manifest_sha256,
               n_clips=len(retained), splits=splits, coverage=coverage, binding_sha256=binding_sha256)
        return ready
    except BaseException as exc:
        # Never remove final files, partial data, or evidence to hide a restart.
        update("BLOCKED", error_type=type(exc).__name__, error=str(exc), resume_instruction="Correct the explicit cause, then invoke the same --root. Durable audit and verified WAVs are reused; never delete another study.")
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, PreparationBlocked):
            raise
        raise PreparationBlocked(f"{type(exc).__name__}: {exc}") from exc
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--status", action="store_true", help="read status without starting or locking preparation")
    args = parser.parse_args()
    if args.status:
        path = args.root.expanduser().absolute() / "data/STATUS.json"
        if not path.is_file():
            print(json.dumps({"state": "NOT_STARTED", "status_path": str(path)}, sort_keys=True))
            return 0
        print(path.read_text(), end="")
        return 0
    try:
        result = prepare(args.root, lambda state: print(json.dumps(state, sort_keys=True), flush=True))
    except PreparationBlocked as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
