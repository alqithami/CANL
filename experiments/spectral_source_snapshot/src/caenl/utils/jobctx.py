from __future__ import annotations

import json
import os
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .io import append_jsonl, atomic_write_json, cfg_get, read_json, to_jsonable, utc_now
from .telemetry import SystemMonitor, environment_snapshot


class JobContext:
    """Per-job filesystem contract and logging helpers.

    Directory layout::

        <job_dir>/job.json          resolved configuration (written by the planner)
        <job_dir>/status.json       {state: pending|running|succeeded|failed, ...}
        <job_dir>/events.jsonl      coarse events (phase boundaries, acquisitions, evaluations)
        <job_dir>/metrics.jsonl     fine-grained training metrics
        <job_dir>/system.csv        telemetry
        <job_dir>/environment.json  environment snapshot
        <job_dir>/summary.json      final summary (only present when succeeded)
        <job_dir>/checkpoints/      model checkpoints (resume + evaluation)
        <job_dir>/artifacts/        indices, predictions, robust masks, trajectories
    """

    def __init__(self, job_dir: Path | str, config: Mapping[str, Any] | None = None):
        self.job_dir = Path(job_dir)
        self.job_dir.mkdir(parents=True, exist_ok=True)
        if config is None:
            config = read_json(self.job_dir / "job.json")
        self.config: dict[str, Any] = dict(config)
        self.job_id = self.config.get("job_id", self.job_dir.name)
        self.checkpoint_dir = self.job_dir / "checkpoints"
        self.artifact_dir = self.job_dir / "artifacts"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.artifact_dir.mkdir(exist_ok=True)
        self._timers: dict[str, float] = {}
        self.timings: dict[str, float] = {}
        self._start_time = time.time()
        self.monitor: SystemMonitor | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._metrics_path = self.job_dir / "metrics.jsonl"
        self._events_path = self.job_dir / "events.jsonl"
        self._attempts_path = self.job_dir / "attempts.jsonl"
        self._prior_attempts = self._read_attempts()
        self.attempt = len(self._prior_attempts) + 1
        self._attempt_recorded = False

    # ------------------------------------------------------------------ config
    def get(self, dotted: str, default: Any = None) -> Any:
        return cfg_get(self.config, dotted, default)

    @property
    def seed(self) -> int:
        return int(self.config.get("seed", 0))

    # ------------------------------------------------------------------ logging
    def log_metrics(self, record: Mapping[str, Any]) -> None:
        rec = dict(record)
        rec.setdefault("time", time.time())
        append_jsonl(self._metrics_path, to_jsonable(rec))

    def event(self, kind: str, **data: Any) -> None:
        rec = {"time": utc_now(), "kind": kind}
        rec.update(data)
        append_jsonl(self._events_path, to_jsonable(rec))
        print(f"[{self.job_id}] {kind}: " + ", ".join(f"{k}={_short(v)}" for k, v in data.items()), flush=True)

    def write_status(self, state: str, **extra: Any) -> None:
        payload = {"state": state, "job_id": self.job_id, "time": utc_now(), "pid": os.getpid()}
        payload.update(extra)
        atomic_write_json(self.job_dir / "status.json", payload)

    def save_artifact_json(self, name: str, payload: Any) -> Path:
        p = self.artifact_dir / name
        atomic_write_json(p, payload)
        return p

    def save_artifact_npy(self, name: str, array: np.ndarray) -> Path:
        p = self.artifact_dir / name
        tmp = p.with_suffix(p.suffix + ".tmp.npy")
        np.save(tmp, np.asarray(array))
        os.replace(tmp, p)
        return p

    def load_artifact_npy(self, name: str) -> np.ndarray | None:
        p = self.artifact_dir / name
        return np.load(p) if p.exists() else None

    def _read_attempts(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if not self._attempts_path.exists():
            return records
        with self._attempts_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
        return records

    def _cumulative_timing(self, current_wall: float, current_timings: Mapping[str, float]) -> tuple[float, dict[str, float]]:
        wall = float(current_wall)
        timings: dict[str, float] = {str(k): float(v) for k, v in current_timings.items()}
        for attempt in self._prior_attempts:
            wall += float(attempt.get("wall_time_s", 0.0) or 0.0)
            for key, value in (attempt.get("timings_s") or {}).items():
                timings[str(key)] = timings.get(str(key), 0.0) + float(value or 0.0)
        return wall, timings

    def _record_attempt(self, state: str, wall_time_s: float, error: str | None = None) -> None:
        if self._attempt_recorded:
            return
        record = {
            "attempt": self.attempt,
            "state": state,
            "started_at_epoch": self._start_time,
            "finished_at": utc_now(),
            "wall_time_s": float(wall_time_s),
            "timings_s": dict(self.timings),
        }
        if error is not None:
            record["error"] = error
        append_jsonl(self._attempts_path, to_jsonable(record))
        self._attempt_recorded = True

    # ------------------------------------------------------------------ timing
    @contextmanager
    def timer(self, name: str):
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            self.timings[name] = self.timings.get(name, 0.0) + dt

    def elapsed(self) -> float:
        return time.time() - self._start_time

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.write_status("running", started_at=utc_now())
        atomic_write_json(self.job_dir / "environment.json", environment_snapshot(include_pip_freeze=self.get("reproducibility.save_pip_freeze", True)))
        interval = float(self.get("reproducibility.telemetry_interval_s", 10.0))
        self.monitor = SystemMonitor(self.job_dir / "system.csv", interval_s=interval).start()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.event("job_started", job_id=self.job_id, device=str(self.device), attempt=self.attempt)

    def finish(self, summary: Mapping[str, Any]) -> None:
        telemetry = self.monitor.stop() if self.monitor else {}
        attempt_wall = self.elapsed()
        cumulative_wall, cumulative_timings = self._cumulative_timing(attempt_wall, self.timings)
        self._record_attempt("succeeded", attempt_wall)
        payload = dict(summary)
        payload["job_id"] = self.job_id
        payload["config"] = to_jsonable(self.config)
        payload["attempt"] = self.attempt
        payload["attempt_count"] = self.attempt
        payload["attempt_timings_s"] = dict(self.timings)
        payload["timings_s"] = cumulative_timings
        payload["attempt_wall_time_s"] = attempt_wall
        payload["wall_time_s"] = cumulative_wall
        payload["telemetry"] = telemetry
        payload["finished_at"] = utc_now()
        atomic_write_json(self.job_dir / "summary.json", payload)
        self.write_status("succeeded", finished_at=utc_now(), attempt=self.attempt, wall_time_s=cumulative_wall)
        self.event("job_succeeded", attempt=self.attempt, wall_time_s=round(cumulative_wall, 1), attempt_wall_time_s=round(attempt_wall, 1))

    def fail(self, exc: BaseException) -> None:
        if self.monitor:
            self.monitor.stop()
        attempt_wall = self.elapsed()
        cumulative_wall, _ = self._cumulative_timing(attempt_wall, self.timings)
        tb = traceback.format_exc()
        self._record_attempt("failed", attempt_wall, error=repr(exc))
        self.write_status("failed", finished_at=utc_now(), attempt=self.attempt, error=repr(exc), traceback=tb, wall_time_s=cumulative_wall)
        self.event("job_failed", attempt=self.attempt, error=repr(exc))
        print(tb, flush=True)

    # ------------------------------------------------------------------ checkpoints
    def save_checkpoint(self, name: str, payload: Mapping[str, Any]) -> Path:
        p = self.checkpoint_dir / name
        tmp = p.with_suffix(p.suffix + ".tmp")
        torch.save(dict(payload), tmp)
        os.replace(tmp, p)
        return p

    def load_checkpoint(self, name: str, map_location: str | torch.device = "cpu") -> dict[str, Any] | None:
        p = self.checkpoint_dir / name
        if not p.exists():
            return None
        return torch.load(p, map_location=map_location, weights_only=False)


def _short(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    s = str(v)
    return s if len(s) < 80 else s[:77] + "..."
