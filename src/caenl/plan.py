"""Plan loading and job expansion.

A *plan* lists stages; each stage has a ``type`` (task implementation), a ``matrix`` whose
list-valued keys are expanded as a cartesian product, and a ``job`` block that is merged
on top of the protocol defaults.  Method names are resolved through the ``methods``
registry of the protocol file so that every job carries a fully resolved, self-contained
configuration (``job.json``).  The resolved plan and its SHA-256 are frozen into the
campaign directory before the first job starts.
"""
from __future__ import annotations

import copy
import itertools
import os
import re
from pathlib import Path
from typing import Any, Mapping

from .utils.io import deep_update, load_yaml, stable_hash, to_jsonable

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(m):
            return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else "")

        return _ENV_RE.sub(repl, value)
    if isinstance(value, Mapping):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def _resolve_relative(path_str: str, base: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    # Try relative to the plan file first, then to the repository root (parent of configs/)
    cand = (base / p).resolve()
    if cand.exists():
        return cand
    cand2 = (base.parent.parent / p).resolve()
    if cand2.exists():
        return cand2
    cand3 = (Path.cwd() / p).resolve()
    return cand3


class Plan:
    def __init__(self, raw: Mapping[str, Any], plan_path: Path, protocol: Mapping[str, Any] | None = None):
        self.path = Path(plan_path).resolve()
        self.raw = expand_env(copy.deepcopy(dict(raw)))
        protocol_path = self.raw.get("protocol")
        self.protocol: dict[str, Any] = {}
        self.protocol_path: Path | None = None
        if protocol is not None:
            # frozen campaign: the protocol as it was run is embedded (PLAN.yaml), no file lookup
            self.protocol = expand_env(copy.deepcopy(dict(protocol)))
        elif protocol_path:
            self.protocol_path = _resolve_relative(str(protocol_path), self.path.parent)
            if not self.protocol_path.exists():
                raise FileNotFoundError(f"protocol file not found: {self.protocol_path}")
            self.protocol = expand_env(load_yaml(self.protocol_path)) or {}
        self.campaign_id: str = str(self.raw.get("campaign_id", "caenl-campaign"))
        inferred_role = "smoke" if "smoke" in self.campaign_id.lower() else "confirmatory"
        self.campaign_role: str = str(self.raw.get("campaign_role", inferred_role)).strip().lower()
        if self.campaign_role not in {"smoke", "engineering", "pilot", "confirmatory"}:
            raise ValueError(f"unknown campaign_role {self.campaign_role!r}; expected smoke, engineering, pilot, or confirmatory")
        self.results_root = Path(str(self.raw.get("results_root", "results"))).expanduser()
        self.data_root = Path(str(self.raw.get("data_root", self.protocol.get("paths", {}).get("data_root", "data")))).expanduser()
        self.cache_root = Path(str(self.raw.get("cache_root", self.protocol.get("paths", {}).get("cache_root", "cache")))).expanduser()
        self.defaults: dict[str, Any] = dict(self.raw.get("defaults", {}))
        self.stages: list[dict[str, Any]] = list(self.raw.get("stages", []))
        self.report_after_each_stage: bool = bool(self.raw.get("report_after_each_stage", True))
        self.methods: dict[str, Any] = dict(self.protocol.get("methods", {}))
        self.methods.update(self.raw.get("methods", {}))

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Plan":
        p = Path(path)
        return cls(load_yaml(p), p)

    @property
    def campaign_root(self) -> Path:
        return self.results_root / self.campaign_id

    # ------------------------------------------------------------------ expansion
    def base_config(self, with_defaults: bool = True) -> dict[str, Any]:
        cfg = deep_update(self.protocol, self.defaults) if with_defaults else copy.deepcopy(dict(self.protocol))
        cfg["paths"] = deep_update(cfg.get("paths", {}), {
            "data_root": str(self.data_root),
            "cache_root": str(self.cache_root),
            "results_root": str(self.results_root),
            "campaign_root": str(self.campaign_root),
            "shared_root": str(self.campaign_root / "shared"),
        })
        cfg["campaign_id"] = self.campaign_id
        cfg["campaign_role"] = self.campaign_role
        return cfg

    def resolve_method(self, name: str) -> dict[str, Any]:
        if name not in self.methods:
            raise KeyError(f"method '{name}' is not defined in the protocol 'methods' registry")
        spec = copy.deepcopy(self.methods[name])
        spec["name"] = name
        spec.setdefault("acquisition", "random")
        spec.setdefault("regularizer", "none")
        spec.setdefault("dcr_lambda", 0.0)
        spec.setdefault("adversarial_training", False)
        spec.setdefault("full_supervision", False)
        return spec

    def expand_stage(self, stage: Mapping[str, Any]) -> list[dict[str, Any]]:
        stype = stage["type"]
        name = stage["name"]
        matrix = dict(stage.get("matrix", {}))
        job_block = dict(stage.get("job", {}))
        keys = list(matrix.keys())
        values = [v if isinstance(v, list) else [v] for v in (matrix[k] for k in keys)]
        jobs: list[dict[str, Any]] = []
        base = self.base_config(with_defaults=False)
        for combo in itertools.product(*values) if keys else [()]:
            assign = dict(zip(keys, combo))
            # precedence (lowest -> highest): protocol -> protocol dataset override -> protocol arch override
            # -> plan defaults -> stage job block -> per-method overrides.  Plan-level settings therefore
            # always win over the protocol's per-dataset defaults (smoke plans shorten every schedule).
            cfg = base
            ds = assign.get("dataset")
            if ds and "dataset_overrides" in cfg and ds in cfg["dataset_overrides"]:
                cfg = deep_update(cfg, cfg["dataset_overrides"][ds])
            arch = assign.get("arch")
            if arch and "arch_overrides" in cfg and arch in cfg["arch_overrides"]:
                cfg = deep_update(cfg, cfg["arch_overrides"][arch])
            cfg = deep_update(cfg, self.defaults)
            cfg = deep_update(cfg, job_block)
            method_name = assign.get("method")
            if method_name is not None:
                spec = self.resolve_method(str(method_name))
                cfg["method"] = deep_update(cfg.get("method", {}), spec)
                # per-method overrides (e.g. more epochs for AT)
                mo = stage.get("overrides_by_method", {}).get(method_name)
                if mo:
                    cfg = deep_update(cfg, mo)
            for k, v in assign.items():
                if k == "method":
                    continue
                if k in ("regularizer", "dcr_lambda", "acquisition"):
                    cfg.setdefault("method", {})
                    cfg["method"][k] = v
                    if k == "regularizer" and v != "none" and "dcr_lambda" not in assign:
                        cfg["method"].setdefault("dcr_lambda", 0.03)
                    continue
                cfg[k] = v
            cfg["stage"] = name
            cfg["type"] = stype
            role = str(stage.get("analysis_role", stage.get("role", self.campaign_role))).strip().lower()
            if role not in {"smoke", "engineering", "pilot", "confirmatory"}:
                raise ValueError(f"stage {name!r} has unknown analysis_role {role!r}")
            cfg["campaign_role"] = self.campaign_role
            cfg["analysis_role"] = role
            cfg["include_in_science"] = bool(stage.get("include_in_science", role in {"pilot", "confirmatory"}))
            cfg["include_in_inference"] = bool(stage.get("include_in_inference", role == "confirmatory"))
            cfg["job_id"] = self.job_id(stype, cfg, assign)
            jobs.append(to_jsonable(cfg))
        # de-duplicate identical job ids (e.g. init jobs where several methods share a group)
        seen: dict[str, dict[str, Any]] = {}
        for j in jobs:
            seen.setdefault(j["job_id"], j)
        return list(seen.values())

    @staticmethod
    def job_id(stype: str, cfg: Mapping[str, Any], assign: Mapping[str, Any]) -> str:
        parts: list[str] = []
        if stype == "prepare_data" and isinstance(assign.get("dataset_spec"), Mapping):
            return "prepare-" + str(assign["dataset_spec"].get("name", "data"))
        for key in ("dataset", "arch"):
            if key in assign:
                parts.append(str(assign[key]))
        if stype == "vision_al" and cfg.get("mode") == "initial_only":
            parts.append("init-" + str(assign.get("regularizer", cfg.get("method", {}).get("regularizer", "none"))))
            lam = assign.get("dcr_lambda", cfg.get("method", {}).get("dcr_lambda"))
            if lam not in (None, 0, 0.0) and assign.get("regularizer", "none") not in ("none",):
                parts[-1] += f"-l{lam}"
        elif "method" in assign:
            parts.append(str(assign["method"]))
        for key, val in assign.items():
            if key in ("dataset", "arch", "method", "seed", "regularizer", "dcr_lambda"):
                continue
            parts.append(f"{key}-{val}")
        if "seed" in assign:
            parts.append(f"seed{assign['seed']}")
        if not parts:
            parts.append(stype)
        return "/".join(parts)

    def expand(self) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
        return [(stage, self.expand_stage(stage)) for stage in self.stages]

    def fingerprint(self) -> str:
        return stable_hash({"raw": self.raw, "protocol": self.protocol}, 16)
