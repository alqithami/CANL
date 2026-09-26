"""Task registry: maps a job ``type`` to the function ``run(ctx) -> summary dict``."""
from __future__ import annotations

import importlib
from typing import Callable

TASKS: dict[str, str] = {
    "prepare_data": "caenl.tasks.prepare_data:run",
    "vision_al": "caenl.vision.task:run",
    "vision_overhead": "caenl.vision.overhead:run",
    "vision_aa_sanity": "caenl.vision.aa_sanity:run",
    "language_c4": "caenl.language.task:run",
    "diffusion_ddpm": "caenl.diffusion.task:run",
    "audio_captioning": "caenl.audio.captioning:run",
    "audio_retrieval": "caenl.audio.retrieval:run",
    "aggregate": "caenl.report.aggregate:run_task",
    "fault_injection": "caenl.tasks.fault_injection:run",  # smoke tests of the publication gate only
}


def resolve(task_type: str) -> Callable:
    if task_type not in TASKS:
        raise KeyError(f"unknown task type '{task_type}'. Known: {sorted(TASKS)}")
    mod, fn = TASKS[task_type].split(":")
    return getattr(importlib.import_module(mod), fn)
