from __future__ import annotations

import csv
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import psutil
import torch


def _nvml_handle():
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        idx = 0
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            try:
                idx = int(visible.split(",")[0])
            except ValueError:
                idx = 0
        return pynvml, pynvml.nvmlDeviceGetHandleByIndex(idx)
    except Exception:
        return None, None


class SystemMonitor:
    """Background sampler writing host/GPU memory and utilisation to ``system.csv``."""

    def __init__(self, path: Path, interval_s: float = 10.0):
        self.path = Path(path)
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc = psutil.Process(os.getpid())
        self.peak_host_rss_bytes = 0
        self._nvml, self._handle = _nvml_handle()

    def sample(self) -> dict[str, Any]:
        rss = self._proc.memory_info().rss
        self.peak_host_rss_bytes = max(self.peak_host_rss_bytes, rss)
        row: dict[str, Any] = {
            "time": time.time(),
            "host_rss_bytes": rss,
            "host_mem_used_bytes": psutil.virtual_memory().used,
            "cpu_percent": psutil.cpu_percent(interval=None),
        }
        if torch.cuda.is_available():
            row["gpu_mem_allocated_bytes"] = torch.cuda.memory_allocated()
            row["gpu_mem_reserved_bytes"] = torch.cuda.memory_reserved()
            row["gpu_mem_max_allocated_bytes"] = torch.cuda.max_memory_allocated()
            if self._nvml is not None and self._handle is not None:
                try:
                    util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                    mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                    row["gpu_util_percent"] = util.gpu
                    row["gpu_mem_used_total_bytes"] = mem.used
                    row["gpu_power_w"] = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                except Exception:
                    pass
        return row

    def _run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header_written = self.path.exists() and self.path.stat().st_size > 0
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = None
            while not self._stop.is_set():
                row = self.sample()
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if not header_written:
                        writer.writeheader()
                try:
                    writer.writerow(row)
                    f.flush()
                except ValueError:
                    # Field set changed (e.g. NVML became available); restart writer.
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    writer.writeheader()
                    writer.writerow(row)
                    f.flush()
                self._stop.wait(self.interval_s)

    def start(self) -> "SystemMonitor":
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="caenl-sysmon", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 5)
        out = {"peak_host_rss_bytes": int(self.peak_host_rss_bytes)}
        if torch.cuda.is_available():
            out["peak_gpu_mem_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            out["peak_gpu_mem_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        return out


def _cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=60).strip()
    except Exception as exc:  # pragma: no cover
        return f"unavailable: {exc}"


def environment_snapshot(include_pip_freeze: bool = True) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
        "host_mem_total_bytes": psutil.virtual_memory().total,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "env": {k: v for k, v in os.environ.items() if k.startswith("CAENL_")},
    }
    if torch.cuda.is_available():
        snap["gpus"] = [
            {
                "name": torch.cuda.get_device_name(i),
                "total_mem_bytes": torch.cuda.get_device_properties(i).total_memory,
                "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
            }
            for i in range(torch.cuda.device_count())
        ]
        snap["nvidia_smi"] = _cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"])
    for mod in ["torchvision", "transformers", "diffusers", "datasets", "torch_fidelity", "autoattack", "numpy", "scipy"]:
        try:
            m = __import__(mod)
            snap[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except Exception:
            snap[f"{mod}_version"] = None
    if include_pip_freeze:
        snap["pip_freeze"] = _cmd([sys.executable, "-m", "pip", "freeze", "--disable-pip-version-check"]).splitlines()
    snap["git_commit"] = _cmd(["git", "rev-parse", "HEAD"])
    snap["git_status"] = _cmd(["git", "status", "--porcelain"])
    return snap
