"""AudioCaps (Kim et al., NAACL 2019): official captions, best-effort audio acquisition, strict manifests.

AudioCaps is the manuscript's multimodal dataset.  Its captions are public (the official
repository ``cdjkim/audiocaps``, pinned to one commit below, SHA-256 verified); its audio is
10-second AudioSet segments from YouTube that are **not redistributable** and must be
obtained by the user (``yt-dlp`` + ``ffmpeg`` here, or a local archive).  Clip availability
changes over time, so every manifest records exactly which official clips are present
(coverage per split) and the hash of every audio file; the campaign refuses to run on a
manifest that does not validate.

Manifest rows (jsonl)::

    {"id": "train/r1nicOVtvkQ_130", "youtube_id": "r1nicOVtvkQ", "start_time": 130,
     "split": "train|valid|test", "path": "/abs/clip.wav", "captions": ["..."],
     "audiocap_ids": [91139], "bytes": 320044, "sha256": "..."}

``manifest_meta.json`` next to it records the caption-CSV hashes, the pinned commit, the
coverage per split, the manifest's own SHA-256 and the tool versions.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from ..utils.io import atomic_write_json, ensure_dir, read_json, read_jsonl, utc_now

AUDIOCAPS_REPO_COMMIT = "d004db3ea1b01cf4fd0347dd8d27db90cadc8809"  # cdjkim/audiocaps, dataset/ as of 2025-02-25
AUDIOCAPS_CSV_URL = "https://raw.githubusercontent.com/cdjkim/audiocaps/" + AUDIOCAPS_REPO_COMMIT + "/dataset/{split}.csv"
AUDIOCAPS_CSV_SHA256 = {
    "train": "c0c5223db682b3bf724ce7e7ce58d5b36929f74572e8526a7211f92d2eef7c8e",
    "val": "dab1c96641d5f3053ddb99dca3949450da9a75737bda53e11cc0aa8b102be0c3",
    "test": "b91c4b7ded2f4f6e7db5b9c4983dc1e1dca3d556f505b61e3fd65cac7e1c638a",
}
OFFICIAL_CLIPS = {"train": 49838, "val": 495, "test": 975}   # unique (youtube_id, start_time) per official split
SPLIT_NAMES = {"train": "train", "val": "valid", "test": "test"}
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".webm", ".opus")
REQUIRED_SPLITS = ("train", "valid", "test")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- captions
def fetch_caption_csvs(root: Path) -> dict[str, Path]:
    """Download (once) and verify the three official caption files."""
    out = {}
    csv_dir = ensure_dir(root / "csv")
    for split in ("train", "val", "test"):
        dest = csv_dir / f"{split}.csv"
        if not dest.exists() or sha256_file(dest) != AUDIOCAPS_CSV_SHA256[split]:
            url = AUDIOCAPS_CSV_URL.format(split=split)
            tmp = dest.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=60) as r, tmp.open("wb") as f:
                shutil.copyfileobj(r, f)
            os.replace(tmp, dest)
            got = sha256_file(dest)
            if got != AUDIOCAPS_CSV_SHA256[split]:
                raise RuntimeError(f"AudioCaps {split}.csv hash mismatch: {got} != {AUDIOCAPS_CSV_SHA256[split]} (pinned commit {AUDIOCAPS_REPO_COMMIT})")
        out[split] = dest
    return out


def read_official_clips(csvs: Mapping[str, Path]) -> dict[str, dict[tuple[str, int], dict[str, Any]]]:
    """``{split: {(youtube_id, start): {"captions": [...], "audiocap_ids": [...]}}}`` from the official CSVs."""
    clips: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for split, path in csvs.items():
        d: dict[tuple[str, int], dict[str, Any]] = {}
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = (r["youtube_id"], int(r["start_time"]))
                e = d.setdefault(key, {"captions": [], "audiocap_ids": []})
                cap = r["caption"].strip()
                if cap:
                    e["captions"].append(cap)
                e["audiocap_ids"].append(int(r["audiocap_id"]))
        clips[split] = d
    return clips


# ----------------------------------------------------------------------------- audio acquisition
def clip_filename(youtube_id: str, start: int) -> str:
    return f"{youtube_id}_{int(start)}.wav"


def _clip_duration_s(path: Path) -> Optional[float]:
    """Duration via libsndfile (wav/flac/ogg/mp3); None if the format cannot be inspected."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


def find_local_audio(audio_dirs: Iterable[Path], youtube_id: str, start: int, max_duration_s: float = 12.0) -> Optional[Path]:
    """Locate an already downloaded clip: ``<ytid>_<start>.*`` (segment naming), ``Y<ytid>.*`` (AudioSet style)
    or ``<ytid>.*``.  Files without the segment naming are accepted only if they are a ~10-s segment
    (duration <= ``max_duration_s``); full-length videos are ignored."""
    segment_stems = (f"{youtube_id}_{int(start)}", f"{youtube_id}_{int(start)}000")
    loose_stems = (f"Y{youtube_id}", youtube_id)
    for d in audio_dirs:
        for stem in segment_stems + loose_stems:
            for ext in AUDIO_EXTS:
                p = Path(d) / f"{stem}{ext}"
                if not (p.exists() and p.stat().st_size > 0):
                    continue
                if stem in loose_stems:
                    dur = _clip_duration_s(p)
                    if dur is None or dur > max_duration_s:
                        continue
                return p
    return None


def _ytdlp_version() -> Optional[str]:
    exe = shutil.which("yt-dlp")
    if not exe:
        return None
    try:
        return subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return "unknown"


def download_clip(youtube_id: str, start: int, dest: Path, duration: float = 10.0, sr: int = 16000, timeout_s: int = 180, cookies: Optional[str] = None) -> tuple[bool, str]:
    """Fetch the 10-s segment with yt-dlp + ffmpeg as 16 kHz mono WAV. Returns (ok, reason)."""
    exe = shutil.which("yt-dlp")
    if exe is None or shutil.which("ffmpeg") is None:
        return False, "yt-dlp/ffmpeg not installed"
    dest.parent.mkdir(parents=True, exist_ok=True)
    stem = dest.stem
    part = dest.parent / f"{stem}.part.wav"
    for old in dest.parent.glob(f"{stem}.part*"):
        old.unlink(missing_ok=True)
    cmd = [
        exe, "--quiet", "--no-warnings", "--no-playlist", "--force-overwrites", "-f", "bestaudio/best",
        "--download-sections", f"*{int(start)}-{int(start) + int(duration)}", "--force-keyframes-at-cuts",
        "--extract-audio", "--audio-format", "wav", "--postprocessor-args", f"ffmpeg:-ar {sr} -ac 1 -t {duration}",
        "-o", str(dest.parent / f"{stem}.part.%(ext)s"), f"https://www.youtube.com/watch?v={youtube_id}",
    ]
    cookies = cookies or os.environ.get("CAENL_YTDLP_COOKIES") or None
    if cookies:
        cmd += ["--cookies", cookies]
    extra = os.environ.get("CAENL_YTDLP_ARGS")  # e.g. "--extractor-args youtube:player_client=web --proxy socks5://..."
    if extra:
        cmd = cmd[:-1] + extra.split() + cmd[-1:]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        for old in dest.parent.glob(f"{stem}.part*"):
            old.unlink(missing_ok=True)
        return False, "timeout"
    if res.returncode != 0 or not part.exists() or part.stat().st_size == 0:
        err = (res.stderr or res.stdout or "").strip().splitlines()
        for old in dest.parent.glob(f"{stem}.part*"):
            old.unlink(missing_ok=True)
        return False, (err[-1][:200] if err else f"rc={res.returncode}")
    os.replace(part, dest)
    return True, "ok"


def acquire_audio(clips: Mapping[str, Mapping[tuple[str, int], Any]], root: Path, options: Mapping[str, Any]) -> dict[str, Any]:
    """Find or download every official clip. Never raises on individual failures; returns a coverage report."""
    audio_root = ensure_dir(root / "audio")
    extra_dirs = [Path(p) for p in (options.get("audio_dirs") or ([] if not options.get("audio_dir") else [options["audio_dir"]]))]
    do_download = bool(options.get("download_audio", True)) and shutil.which("yt-dlp") is not None and shutil.which("ffmpeg") is not None
    workers = int(options.get("download_workers", 4))
    limit = options.get("max_clips_per_split")  # smoke / partial runs
    report: dict[str, Any] = {"started": utc_now(), "yt_dlp": _ytdlp_version(), "download_enabled": do_download, "splits": {}, "failures": {}}
    found: dict[str, dict[tuple[str, int], Path]] = {}
    for split, d in clips.items():
        split_dir = ensure_dir(audio_root / split)
        keys = sorted(d.keys())
        if limit:
            keys = keys[: int(limit)]
        present: dict[tuple[str, int], Path] = {}
        todo = []
        for yt, st in keys:
            p = find_local_audio([split_dir] + extra_dirs, yt, st)
            if p is not None:
                present[(yt, st)] = p
            else:
                todo.append((yt, st))
        failures: dict[str, str] = {}
        if todo and do_download:
            print(f"[audiocaps] {split}: {len(present)} clips present, downloading {len(todo)} with {workers} workers", flush=True)
            t0 = time.time()
            done = 0
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(download_clip, yt, st, split_dir / clip_filename(yt, st), cookies=options.get("cookies")): (yt, st) for yt, st in todo}
                for fut in as_completed(futs):
                    yt, st = futs[fut]
                    try:
                        ok, reason = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        ok, reason = False, repr(exc)[:200]
                    if ok:
                        present[(yt, st)] = split_dir / clip_filename(yt, st)
                    else:
                        failures[f"{yt}_{st}"] = reason
                    done += 1
                    if done % 200 == 0:
                        print(f"[audiocaps] {split}: {done}/{len(todo)} attempted, {len(present)} available ({(time.time() - t0) / 60:.1f} min)", flush=True)
        elif todo:
            for yt, st in todo:
                failures[f"{yt}_{st}"] = "not found locally (download disabled or yt-dlp/ffmpeg missing)"
        found[split] = present
        reasons = Counter(v.split(":")[0][:80] for v in failures.values())
        report["splits"][split] = {"official": len(d), "requested": len(keys), "available": len(present), "coverage": round(len(present) / max(1, len(keys)), 4), "failed": len(failures), "failure_reasons": dict(reasons.most_common(20))}
        report["failures"][split] = dict(list(failures.items())[:500])  # sample; the split summary above has the full counts
    report["finished"] = utc_now()
    atomic_write_json(root / "download_report.json", report)
    return {"found": found, "report": report}


# ----------------------------------------------------------------------------- manifest
def write_manifest(clips: Mapping[str, Mapping[tuple[str, int], Any]], found: Mapping[str, Mapping[tuple[str, int], Path]], manifest: Path, hash_audio: bool = True) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        for (yt, st), p in sorted(found.get(split, {}).items()):
            e = clips[split][(yt, st)]
            row = {"id": f"{split}/{yt}_{st}", "youtube_id": yt, "start_time": int(st), "split": SPLIT_NAMES[split], "path": str(Path(p).resolve()), "captions": list(e["captions"]), "audiocap_ids": list(e["audiocap_ids"]), "bytes": int(Path(p).stat().st_size)}
            if hash_audio:
                row["sha256"] = sha256_file(Path(p))
            rows.append(row)
    tmp = manifest.with_suffix(".jsonl.part")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, manifest)
    return {"manifest": str(manifest), "n_clips": len(rows), "manifest_sha256": sha256_file(manifest)}


def prepare_audiocaps(data_root: Path, cache_root: Path, options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Captions (pinned, verified) + best-effort audio + strictly validated manifest under ``<data_root>/audiocaps``."""
    options = dict(options or {})
    root = ensure_dir(Path(options.get("root", Path(data_root) / "audiocaps")))
    manifest = Path(options.get("manifest", root / "manifest.jsonl"))
    if manifest.exists() and not bool(options.get("rebuild", False)):
        rep = validate_audio_manifest(manifest, min_coverage=options.get("min_coverage"), verify_hashes=str(options.get("verify_hashes", "sample")))
        meta = read_json(root / "manifest_meta.json") if (root / "manifest_meta.json").exists() else {}
        return {"manifest": str(manifest), "n_clips": rep["n_clips"], "splits": rep["splits"], "coverage": meta.get("coverage"), "valid": rep["valid"], "errors": rep["errors"], "manifest_sha256": rep["manifest_sha256"]}
    csvs = fetch_caption_csvs(root)
    clips = read_official_clips(csvs)
    for split, n in OFFICIAL_CLIPS.items():
        if len(clips[split]) != n:
            raise RuntimeError(f"official AudioCaps {split}.csv lists {len(clips[split])} clips, expected {n}")
    acq = acquire_audio(clips, root, options)
    info = write_manifest(clips, acq["found"], manifest, hash_audio=bool(options.get("hash_audio", True)))
    meta = {
        "dataset": "audiocaps",
        "captions_source": {"repository": "https://github.com/cdjkim/audiocaps", "commit": AUDIOCAPS_REPO_COMMIT, "csv_sha256": AUDIOCAPS_CSV_SHA256},
        "official_clips": OFFICIAL_CLIPS,
        "coverage": acq["report"]["splits"],
        "yt_dlp": acq["report"].get("yt_dlp"),
        "created": utc_now(),
        "manifest_sha256": info["manifest_sha256"],
        "n_clips": info["n_clips"],
    }
    atomic_write_json(root / "manifest_meta.json", meta)
    rep = validate_audio_manifest(manifest, min_coverage=options.get("min_coverage"), verify_hashes="none")
    return {"manifest": str(manifest), "n_clips": info["n_clips"], "splits": rep["splits"], "coverage": meta["coverage"], "valid": rep["valid"], "errors": rep["errors"], "manifest_sha256": info["manifest_sha256"], "download_report": str(root / "download_report.json")}


def validate_audio_manifest(
    manifest: str | Path,
    required_splits: Iterable[str] = REQUIRED_SPLITS,
    min_coverage: Optional[float] = None,
    verify_hashes: str = "sample",
    sample: int = 200,
    official_counts: Optional[Mapping[str, int]] = None,
) -> dict[str, Any]:
    """Strict manifest validation.  ``valid`` is False on ANY of:

    * the manifest file is missing/empty or a row lacks ``id``/``path``/``split``/``captions``;
    * a referenced audio file does not exist or is empty;
    * a required split is empty; duplicate ids; a clip (``id``, ``youtube_id`` or ``path``) in more than one split;
    * a clip without a non-empty caption;
    * a recorded ``sha256``/``bytes`` that no longer matches the file (``verify_hashes``: ``none`` | ``sample`` | ``all``);
    * coverage of an official split (``len(split) / official``) below ``min_coverage`` when given.
    """
    manifest = Path(manifest)
    errors: list[str] = []
    warnings: list[str] = []
    if not manifest.exists() or manifest.stat().st_size == 0:
        return {"valid": False, "errors": [f"manifest missing or empty: {manifest}"], "warnings": [], "n_clips": 0, "splits": {}, "manifest_sha256": None}
    rows = read_jsonl(manifest)
    counts = Counter()
    ids: Counter = Counter()
    by_key: dict[str, set[str]] = defaultdict(set)
    missing_files = 0
    no_caption = 0
    short_caps = 0
    for i, r in enumerate(rows):
        for k in ("id", "path", "split", "captions"):
            if k not in r:
                errors.append(f"row {i}: missing field {k}")
        if any(k not in r for k in ("id", "path", "split", "captions")):
            continue
        counts[r["split"]] += 1
        ids[r["id"]] += 1
        p = Path(r["path"])
        if not p.exists() or p.stat().st_size == 0:
            missing_files += 1
            if missing_files <= 5:
                errors.append(f"audio file missing or empty: {p}")
        caps = [c for c in (r.get("captions") or []) if isinstance(c, str) and c.strip()]
        if not caps:
            no_caption += 1
            if no_caption <= 5:
                errors.append(f"clip {r['id']} has no caption")
        if r["split"] in ("valid", "test") and 0 < len(caps) < 5:
            short_caps += 1
        for key in ("id", "youtube_id", "path"):
            if r.get(key):
                by_key[f"{key}:{r[key]}"].add(r["split"])
    if missing_files > 5:
        errors.append(f"... {missing_files} audio files missing in total")
    if no_caption > 5:
        errors.append(f"... {no_caption} clips without captions in total")
    dup = [k for k, n in ids.items() if n > 1]
    if dup:
        errors.append(f"{len(dup)} duplicate clip ids (e.g. {dup[:3]})")
    crossing = sorted(k for k, s in by_key.items() if len(s) > 1)
    if crossing:
        errors.append(f"{len(crossing)} clips appear in more than one split (e.g. {crossing[:3]})")
    for s in required_splits:
        if counts.get(s, 0) == 0:
            errors.append(f"required split '{s}' is empty")
    if short_caps:
        warnings.append(f"{short_caps} valid/test clips have fewer than 5 captions")
    # hashes
    hashed = [r for r in rows if r.get("sha256")]
    if verify_hashes != "none" and hashed:
        import random

        chosen = hashed if verify_hashes == "all" else random.Random(0).sample(hashed, min(sample, len(hashed)))
        bad = 0
        for r in chosen:
            p = Path(r["path"])
            if not p.exists():
                continue
            if r.get("bytes") is not None and p.stat().st_size != int(r["bytes"]):
                bad += 1
                continue
            if sha256_file(p) != r["sha256"]:
                bad += 1
        if bad:
            errors.append(f"{bad}/{len(chosen)} audio files changed since the manifest was built (hash/size mismatch)")
    elif not hashed:
        warnings.append("manifest carries no per-file sha256 (built with hash_audio=false)")
    # coverage against the official split sizes
    official = dict(official_counts or {SPLIT_NAMES[k]: v for k, v in OFFICIAL_CLIPS.items()})
    coverage = {s: round(counts.get(s, 0) / official[s], 4) for s in official if s in counts or s in required_splits}
    if min_coverage is not None:
        for s, c in coverage.items():
            if c < float(min_coverage):
                errors.append(f"split '{s}' covers only {c * 100:.1f}% of the official clips (minimum {float(min_coverage) * 100:.0f}%); lower audio.manifest_min_coverage to accept")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "n_clips": len(rows),
        "splits": dict(counts),
        "coverage": coverage,
        "manifest_sha256": sha256_file(manifest),
        "manifest": str(manifest),
    }


def manifest_provenance(manifest: str | Path) -> dict[str, Any]:
    """Compact provenance block for job summaries (hash, counts, coverage, meta if present)."""
    manifest = Path(manifest)
    rows = read_jsonl(manifest) if manifest.exists() else []
    counts = Counter(r.get("split") for r in rows)
    meta_path = manifest.parent / "manifest_meta.json"
    meta = read_json(meta_path) if meta_path.exists() else {}
    return {"manifest": str(manifest), "manifest_sha256": sha256_file(manifest) if manifest.exists() else None, "n_clips": len(rows), "splits": dict(counts), "coverage": meta.get("coverage"), "captions_commit": (meta.get("captions_source") or {}).get("commit")}
