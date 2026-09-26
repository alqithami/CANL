from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False, cudnn_benchmark: bool = True) -> None:
    """Seed python, numpy and torch (all devices).

    ``deterministic=True`` enables deterministic cuDNN kernels at a speed cost; the
    frozen protocol keeps it off and instead reports variability across seeds.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark) and not deterministic
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        except Exception:  # pragma: no cover
            pass


def rng_state_dict() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def load_rng_state(state: dict[str, Any]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8).cpu())
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([torch.as_tensor(s, dtype=torch.uint8).cpu() for s in state["cuda"]])
        except Exception:  # pragma: no cover
            pass


def derive_seed(base_seed: int, *tags: Any) -> int:
    """Derive an independent 31-bit seed from a base seed and string tags.

    Used to give the data split, model initialisation, data order, attack and
    acquisition sub-sampling their own seeds, all of which are logged.
    """
    import hashlib

    payload = f"{int(base_seed)}|" + "|".join(str(t) for t in tags)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16) % (2**31 - 1)
