"""Fault-injection job for smoke tests of the publication gate.

``type: fault_injection`` jobs emulate another job type's summary on purpose so that a smoke plan
can prove that a non-finite required metric (or a missing summary, or a crash) stops the campaign.
Never used by production plans; ``caenl validate`` warns when it is present.
"""
from __future__ import annotations

import math
from typing import Any

from ..utils.jobctx import JobContext


def run(ctx: JobContext) -> dict[str, Any]:
    fault = dict(ctx.config.get("fault", {}))
    kind = str(fault.get("kind", "nan_metric"))
    if float(fault.get("sleep_s", 0) or 0) > 0:  # scheduler tests: hold a slot for a while and record the interval
        import os
        import time

        t0 = time.time()
        ctx.event("sleep_started", t0=t0, gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), stage=os.environ.get("CAENL_STAGE"))
        time.sleep(float(fault["sleep_s"]))
        ctx.event("sleep_finished", t0=t0, t1=time.time(), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    if kind == "raise":
        raise RuntimeError("fault injection: deliberate job failure")
    emulate = str(fault.get("emulate", "language_c4"))
    final = {k: (float("nan") if v == "nan" else v) for k, v in dict(fault.get("final", {"validation_nll": "nan", "perplexity": 1.0, "token_ece": 0.1})).items()}
    ctx.event("fault_injected", fault_kind=kind, emulate=emulate, nonfinite=[k for k, v in final.items() if isinstance(v, float) and not math.isfinite(v)])
    return {"task": emulate, "method": ctx.config.get("method", {}).get("name", "baseline"), "seed": ctx.seed, "final": final, "fault": kind}
