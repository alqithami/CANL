"""Vision active-learning job (one dataset x architecture x method x seed).

Protocol (frozen, manuscript Section 5.3 with the revision's clarifications):

1. class-balanced initial labeled set (``initial_fraction`` of the training set);
   a stratified controller-validation subset is carved out of it *for every method* and never
   used for gradient updates (label budget identical across methods);
2. initial training phase (shared across methods of the same regulariser group for a seed);
3. ``rounds`` acquisition rounds, each adding ``fraction_per_round`` of the original training-set
   size, scored in candidate passes of ``candidate_size`` examples with per-pass selection of the
   top ``selected_fraction_per_pass`` (CAENL: quantile threshold ``tau`` from the controller);
   selected examples leave the pool immediately and every pass is recorded;
4. warm-started fine-tuning for ``epochs_per_round`` epochs after each round;
5. evaluation after every phase (clean accuracy/loss/ECE, NC panel, APGD-CE/PGD proxies on a fixed
   subset) and full AutoAttack at the final label fraction (optionally at every round).
"""
from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..core.controllers import FullMACC, build_controller, kappa_target_from_strength
from ..core.dcr import DCR
from ..core.monitor import ClassConditionalMonitor
from ..utils.io import atomic_write_json, read_json, stable_hash, to_jsonable
from ..utils.jobctx import JobContext
from ..utils.seeding import derive_seed, load_rng_state, rng_state_dict
from . import acquisition as acq
from .augment import eval_transform
from .data import DeviceArray, fixed_evaluation_subset, load_vision_dataset
from .diagnostics import class_histogram, evaluate_model, nc_panel, score_analysis
from .models import build_model, count_parameters
from .robustness import apgd_ce_evaluate, autoattack_evaluate, margin_and_gradient_diagnostics, pgd_evaluate
from .schedule import candidate_passes, class_balanced_split, split_hash, stratified_subset
from .train import Trainer


# ----------------------------------------------------------------------------- helpers
def _debug_stop_round(r: int) -> None:
    """Testing hook: CAENL_DEBUG_STOP_AT_ROUND=r aborts the process right after round r is checkpointed."""
    v = os.environ.get("CAENL_DEBUG_STOP_AT_ROUND")
    if v and int(r) >= int(v):
        raise SystemExit(f"debug stop after round {r}")


def _targets_for(layers: list[str], cfg: dict[str, Any]) -> dict[str, float]:
    tcs = cfg.get("collapse", {}).get("target_collapse_strength", 0.5)
    out: dict[str, float] = {}
    for l in layers:
        s = tcs.get(l, tcs.get("default", 0.5)) if isinstance(tcs, dict) else tcs
        out[l] = kappa_target_from_strength(float(s))
    return out


def _layer_weights(layers: list[str], dims: dict[str, int], spec: Any) -> dict[str, float]:
    if isinstance(spec, dict):
        return {l: float(spec.get(l, 0.0)) for l in layers}
    if spec == "dim_normalized":
        return {l: 1.0 / float(dims[l]) for l in layers}
    return {l: 1.0 for l in layers}


def shared_initial_key(cfg: dict[str, Any]) -> str:
    m = cfg.get("method", {})
    reg = m.get("regularizer", "none")
    relevant = {
        "dataset": cfg.get("dataset"),
        "dataset_options": cfg.get("dataset_options", {}),
        "arch": cfg.get("arch"),
        "arch_options": cfg.get("arch_options", {}),
        "seed": cfg.get("seed"),
        "regularizer": reg,
        "dcr_lambda": m.get("dcr_lambda", 0.0) if reg != "none" else 0.0,
        "adversarial_training": bool(m.get("adversarial_training", False)),
        "adversarial_training_cfg": cfg.get("adversarial_training", {}) if m.get("adversarial_training") else None,
        "training": {k: v for k, v in cfg.get("training", {}).items() if k not in ("epochs_per_round", "round_lr_factor", "log_every_steps")},
        "collapse": cfg.get("collapse", {}),
        "macc_lite": cfg.get("macc_lite", {}) if reg == "macc_lite" else None,
        "full_macc": cfg.get("full_macc", {}) if reg == "full_macc" else None,
        "active_learning": {k: cfg.get("active_learning", {}).get(k) for k in ("initial_fraction", "controller_validation_fraction", "rounds", "fraction_per_round")},
        "full_supervision": bool(m.get("full_supervision", False)),
    }
    tag = f"{cfg.get('dataset')}-{cfg.get('arch')}-seed{cfg.get('seed')}-{reg}"
    if reg != "none":
        tag += f"-l{m.get('dcr_lambda', 0.0)}"
    if m.get("adversarial_training"):
        tag += "-at"
    if m.get("full_supervision"):
        tag += "-full"
    return f"{tag}-{stable_hash(relevant, 10)}"


# ----------------------------------------------------------------------------- job
class VisionALJob:
    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        self.cfg = ctx.config
        self.device = ctx.device
        self.seed = ctx.seed
        self.method = dict(self.cfg.get("method", {}))
        self.al = dict(self.cfg.get("active_learning", {}))
        self.tr = dict(self.cfg.get("training", {}))
        self.rob = dict(self.cfg.get("robustness", {}))
        self.diag = dict(self.cfg.get("diagnostics", {}))
        self.mode = str(self.cfg.get("mode", "full"))
        self.rounds_records: list[dict[str, Any]] = []
        self.acq_records: list[dict[str, Any]] = []
        self.initial_job: Optional[str] = None

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        ctx, cfg = self.ctx, self.cfg
        paths = cfg.get("paths", {})
        with ctx.timer("data_load"):
            self.ds = load_vision_dataset(cfg["dataset"], paths.get("data_root", "data"), paths.get("cache_root", "cache"), cfg.get("dataset_options", {}))
            rep = cfg.get("reproducibility", {})
            force_memmap = bool(cfg.get("dataset_options", {}).get("force_memmap", False))  # smoke: exercise the ImageNet-1K memmap path
            self.train = DeviceArray(self.ds.train_x, self.device, max_device_fraction=float(rep.get("max_device_data_fraction", 0.45)), max_host_fraction=float(rep.get("max_host_data_fraction", 0.4)), force_memmap=force_memmap)
            self.test = DeviceArray(self.ds.test_x, self.device, max_device_fraction=0.2, max_host_fraction=float(rep.get("max_host_data_fraction", 0.4)))
        self.num_classes = self.ds.num_classes
        ctx.event("data_ready", dataset=self.ds.name, n_train=len(self.ds.train_y), n_test=len(self.ds.test_y), num_classes=self.num_classes, fingerprint=self.ds.fingerprint[:16], train_location=self.train.location)

        # ---- splits
        split_seed = derive_seed(self.seed, "split")
        n_total = len(self.ds.train_y)
        initial = class_balanced_split(self.ds.train_y, float(self.al.get("initial_fraction", 0.1)), split_seed)
        ctrl_frac = float(self.al.get("controller_validation_fraction", 0.1))
        self.ctrl_val = stratified_subset(initial, self.ds.train_y, ctrl_frac, split_seed + 1)
        all_idx = np.arange(n_total)
        if self.method.get("full_supervision", False):
            self.labeled = np.setdiff1d(all_idx, self.ctrl_val)
            self.unlabeled = np.zeros(0, dtype=np.int64)
            self.n_rounds = 0
        else:
            self.labeled = np.setdiff1d(initial, self.ctrl_val)
            self.unlabeled = np.setdiff1d(all_idx, initial)
            self.n_rounds = int(self.al.get("rounds", 5))
        if self.mode == "initial_only":
            self.n_rounds = 0
        self.round_budget = int(round(float(self.al.get("fraction_per_round", 0.1)) * n_total))
        ctx.save_artifact_npy("labeled_initial.npy", self.labeled)
        ctx.save_artifact_npy("ctrl_val.npy", self.ctrl_val)
        ctx.event("splits", n_initial_labeled=int(len(self.labeled)), n_ctrl_val=int(len(self.ctrl_val)), n_unlabeled=int(len(self.unlabeled)), round_budget=self.round_budget, split_hash=split_hash(self.labeled), split_seed=split_seed)

        # ---- model
        torch.manual_seed(derive_seed(self.seed, "init"))
        arch_opts = dict(cfg.get("arch_options", {}))
        arch_opts.setdefault("image_size", int(self.ds.train_res))  # ViT patch grid follows the training resolution
        arch_opts.setdefault("hf_cache", str(Path(paths.get("data_root", "data")) / "hf"))  # MAE weights share the HF cache
        self.model = build_model(cfg["arch"], self.num_classes, self.ds.mean, self.ds.std, arch_opts).to(self.device)
        self.init_report = dict(getattr(self.model.model, "init_report", {"init": "scratch"}))
        self.n_params = count_parameters(self.model)
        dims = self.model.feature_dims
        wanted = list(cfg.get("collapse", {}).get("monitored_layers", ["layer2", "layer3", "layer4"]))
        self.layers = [l for l in wanted if l in dims]
        if not self.layers:
            raise ValueError(f"none of the monitored layers {wanted} exist in {cfg['arch']} ({list(dims)})")
        self.layer_dims = {l: int(dims[l]) for l in self.layers}
        ccfg = cfg.get("collapse", {})
        self.monitor = ClassConditionalMonitor(self.layer_dims, self.num_classes, ema_decay=float(ccfg.get("covariance_ema_decay", 0.99)), shrinkage=float(ccfg.get("covariance_shrinkage_epsilon", 1e-5)), kappa_ema_decay=float(ccfg.get("kappa_ema_decay", 0.9)), device=self.device)
        self.dcr = DCR.build(self.layer_dims, k=int(ccfg.get("truncated_spectral_rank", 128)), num_classes=self.num_classes, covariance=str(ccfg.get("dcr_covariance", "total")), device=self.device)
        self.targets = _targets_for(self.layers, cfg)
        self.regularizer = str(self.method.get("regularizer", "none"))
        planned = self.planned_steps()
        cadence = int(ccfg.get("monitoring_cadence_steps", 100))
        ctrl_cfg = {"macc_lite": cfg.get("macc_lite", {}), "full_macc": cfg.get("full_macc", {}), "threshold_quantile": self.al.get("threshold_quantile", 0.9)}
        self.controller = build_controller(self.regularizer, self.layers, self.targets, ctrl_cfg, total_controller_steps=max(1, planned // cadence), device=self.device, seed=derive_seed(self.seed, "controller"), state_extra=1, lambda_value=float(self.method.get("dcr_lambda", 0.0)))
        self.trainer = Trainer(ctx, self.model, self.train, self.ds.train_y, self.num_classes, self.ds.train_res, self.ds.aug, self.monitor, self.dcr, self.controller, self.regularizer, self.ctrl_val)
        self.trainer.total_steps_planned = planned
        self.acq_rng = np.random.default_rng(derive_seed(self.seed, "acq"))
        self.acq_gen = torch.Generator(device="cpu").manual_seed(derive_seed(self.seed, "acq_torch"))
        self.attack_seed = derive_seed(self.seed, "attack")
        rob_n = int(self.rob.get("intermediate", {}).get("n_examples", 2000))
        self.robust_subset = fixed_evaluation_subset(self.ds.test_y, rob_n, self.ds.name, "robust_subset")  # same subset for all seeds/methods
        self.aa_subset = fixed_evaluation_subset(self.ds.test_y, self.rob.get("final", {}).get("n_examples"), self.ds.name, "autoattack")
        self.layer_weights = _layer_weights(self.layers, self.layer_dims, self.al.get("layer_weights", "dim_normalized"))
        ctx.event("model_ready", arch=cfg["arch"], params=self.n_params, layers=self.layers, layer_dims=self.layer_dims, regularizer=self.regularizer, acquisition=self.method.get("acquisition"), dcr_ranks={l: s.rank for l, s in self.dcr.layers.items()}, kappa_targets=self.targets, controller_params=getattr(self.controller, "num_parameters", 0), init=self.init_report.get("init"), init_repo=self.init_report.get("repo"))

    def planned_steps(self) -> int:
        """Total optimisation steps of a complete run (initial phase + all rounds).

        Computed from the protocol's round count irrespective of ``mode`` so that the
        controller schedules (epsilon decay, progress input) of an ``initial_only`` job are
        identical to those of the active-learning jobs that reuse its checkpoint.
        """
        bs = int(self.tr.get("batch_size", 256))
        n0 = len(self.labeled)
        steps = max(1, n0 // bs) * int(self.initial_epochs())
        rounds = 0 if self.method.get("full_supervision", False) else int(self.al.get("rounds", 5))
        for r in range(1, rounds + 1):
            n = n0 + r * self.round_budget
            steps += max(1, n // bs) * int(self.tr.get("epochs_per_round", 20))
        return int(steps)

    def initial_epochs(self) -> int:
        if self.method.get("full_supervision", False):
            fe = self.tr.get("full_supervision_epochs")
            if fe:
                return int(fe)
            return int(self.tr.get("initial_epochs", 100)) + int(self.al.get("rounds", 5)) * int(self.tr.get("epochs_per_round", 20))
        return int(self.tr.get("initial_epochs", 100))

    # ------------------------------------------------------------------ state
    def state_payload(self, round_index: int) -> dict[str, Any]:
        return {
            "round": round_index,
            "model": self.model.state_dict(),
            "monitor": self.monitor.state_dict(),
            "controller": self.controller.state_dict(),
            "controller_action": {"lambdas": self.trainer.current_action.lambdas, "flags": self.trainer.current_action.flags, "tau": self.trainer.current_action.threshold_quantile},
            "global_step": self.trainer.global_step,
            "prev_ctrl_value": self.trainer.prev_ctrl_value,
            "labeled": self.labeled,
            "unlabeled": self.unlabeled,
            "rounds_records": self.rounds_records,
            "acq_records": self.acq_records,
            "rng": rng_state_dict(),
            "acq_rng": self.acq_rng.bit_generator.state,
            "acq_gen": self.acq_gen.get_state(),
        }

    def restore(self, state: dict[str, Any]) -> None:
        from ..core.controllers import ControllerAction

        self.model.load_state_dict(state["model"])
        self.monitor.load_state_dict(state["monitor"])
        self.controller.load_state_dict(state["controller"])
        ca = state.get("controller_action")
        if ca:
            self.trainer.current_action = ControllerAction(lambdas=dict(ca["lambdas"]), flags=dict(ca["flags"]), threshold_quantile=float(ca["tau"]))
        self.trainer.global_step = int(state.get("global_step", 0))
        self.trainer.prev_ctrl_value = state.get("prev_ctrl_value")
        self.labeled = np.asarray(state["labeled"], dtype=np.int64)
        self.unlabeled = np.asarray(state["unlabeled"], dtype=np.int64)
        self.rounds_records = list(state.get("rounds_records", []))
        self.acq_records = list(state.get("acq_records", []))
        if "rng" in state:
            load_rng_state(state["rng"])
        if "acq_rng" in state:
            self.acq_rng.bit_generator.state = state["acq_rng"]
        if "acq_gen" in state:
            self.acq_gen.set_state(torch.as_tensor(state["acq_gen"], dtype=torch.uint8))

    def save_round_checkpoint(self, r: int) -> None:
        self.ctx.save_checkpoint(f"round_{r}.pt", self.state_payload(r))
        keep = str(self.cfg.get("checkpointing", {}).get("keep", "last2"))
        if keep != "all":
            n_keep = 2 if keep == "last2" else 1
            for p in sorted(self.ctx.checkpoint_dir.glob("round_*.pt"), key=lambda q: int(q.stem.split("_")[1]))[:-n_keep]:
                p.unlink(missing_ok=True)
        atomic_write_json(self.ctx.artifact_dir / "rounds.json", {"rounds": self.rounds_records, "acquisitions": self.acq_records})

    def latest_round_checkpoint(self) -> Optional[int]:
        found = [int(p.stem.split("_")[1]) for p in self.ctx.checkpoint_dir.glob("round_*.pt")]
        return max(found) if found else None

    # ------------------------------------------------------------------ shared initial phase
    def shared_dir(self) -> Path:
        root = Path(self.cfg.get("paths", {}).get("shared_root", self.ctx.job_dir.parent.parent / "shared"))
        return root / "initial" / shared_initial_key(self.cfg)

    def try_load_shared_initial(self) -> bool:
        sd = self.shared_dir()
        done = sd / "done.json"
        if not done.exists():
            return False
        state = torch.load(sd / "state.pt", map_location="cpu", weights_only=False)
        self.restore(state)
        self.labeled = np.asarray(state["labeled"], dtype=np.int64)
        rec = read_json(done)
        self.rounds_records = [rec["round_record"]]
        self.initial_job = rec.get("job_id")
        ps = rec.get("planned_steps")
        if ps is not None and int(ps) != int(self.trainer.total_steps_planned):
            self.ctx.event("planned_steps_mismatch", shared=int(ps), this_job=int(self.trainer.total_steps_planned))
        # carry the initial-phase training trace into this job's metrics for dynamics figures
        src_metrics = sd / "metrics_initial.jsonl"
        dst_metrics = self.ctx.job_dir / "metrics.jsonl"
        if src_metrics.exists() and (not dst_metrics.exists() or dst_metrics.stat().st_size < 1024):
            with src_metrics.open("r", encoding="utf-8") as fsrc, dst_metrics.open("a", encoding="utf-8") as fdst:
                for line in fsrc:
                    fdst.write(line)
        self.ctx.event("initial_loaded", path=str(sd), n_labeled=int(len(self.labeled)), split_hash=split_hash(self.labeled), initial_job=rec.get("job_id"))
        return True

    def save_shared_initial(self, record: dict[str, Any]) -> None:
        sd = self.shared_dir()
        if (sd / "done.json").exists():
            # another job finished the same initial phase first: adopt it (paired design)
            if self.try_load_shared_initial():
                self.ctx.event("initial_adopted_after_race", path=str(sd))
            return
        sd.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=sd.name + ".tmp-", dir=str(sd.parent)))
        torch.save(self.state_payload(0), tmp / "state.pt")
        m = self.ctx.job_dir / "metrics.jsonl"
        if m.exists():
            shutil.copy2(m, tmp / "metrics_initial.jsonl")
        atomic_write_json(tmp / "done.json", {"key": sd.name, "job_id": self.ctx.job_id, "round_record": to_jsonable(record), "split_hash": split_hash(self.labeled), "planned_steps": self.trainer.total_steps_planned, "global_step": self.trainer.global_step})
        try:
            os.replace(tmp, sd)
            self.ctx.event("initial_saved", path=str(sd))
        except OSError:
            # another job won the race: adopt the canonical checkpoint so that pairing holds
            shutil.rmtree(tmp, ignore_errors=True)
            if self.try_load_shared_initial():
                self.ctx.event("initial_adopted_after_race", path=str(sd))

    # ------------------------------------------------------------------ evaluation
    def evaluate_round(self, r: int, phase_stats: dict[str, Any], acquisition: Optional[dict[str, Any]] = None, final: bool = False) -> dict[str, Any]:
        ctx = self.ctx
        frac = (len(self.labeled) + len(self.ctrl_val)) / len(self.ds.train_y)
        t0 = time.perf_counter()
        nc_layers = [l for l in self.diag.get("nc_layers", ["layer4"]) if l in self.layers or l == self.model.penultimate]
        with ctx.timer("eval_clean"):
            ev = evaluate_model(self.model, self.test, self.ds.test_y, self.ds.test_res, bs=int(self.al.get("eval_batch_size", 512)), layers=nc_layers, amp=bool(self.al.get("eval_amp", False)))
        rec: dict[str, Any] = {
            "round": r,
            "label_fraction": round(frac, 4),
            "n_labeled": int(len(self.labeled)),
            "n_labeled_incl_ctrl": int(len(self.labeled) + len(self.ctrl_val)),
            "test_acc": ev.acc,
            "test_top5": ev.top5,
            "test_loss": ev.loss,
            "test_ece": ev.ece,
            "phase": phase_stats,
            "acquisition": acquisition,
            "lambdas": self.trainer.lambdas,
            "tau": self.trainer.threshold_quantile,
            "kappa_train": self.monitor.kappa(),
            "collapse_strength_train": self.monitor.strength(),
            "instability_train": self.monitor.instability(),
            "global_step": self.trainer.global_step,
        }
        if self.diag.get("save_test_logits", True):
            ctx.save_artifact_npy(f"test_logits_round{r}.npy", ev.logits)
        if len(self.ctrl_val):
            cv = evaluate_model(self.model, self.train, self.ds.train_y, self.ds.train_res, bs=512, layers=[], indices=self.ctrl_val, amp=bool(self.al.get("eval_amp", False)))
            rec["ctrl_val_acc"], rec["ctrl_val_loss"] = cv.acc, cv.loss
        with ctx.timer("eval_nc"):
            rec["nc"] = nc_panel(ev.feats, self.ds.test_y, self.num_classes, self.model.fc.weight, ev.preds, self.model.penultimate, topk=int(self.cfg.get("collapse", {}).get("truncated_spectral_rank", 128)))
        # robustness proxies on the fixed subset
        inter = dict(self.rob.get("intermediate", {}))
        do_inter = bool(inter.get("enabled", True)) and (not final or bool(inter.get("also_final", True)))
        if do_inter:
            xs = eval_transform(self.test.get(self.robust_subset), self.ds.test_res)
            ys = torch.as_tensor(self.ds.test_y[self.robust_subset], device=self.device)
            eps = float(self.rob.get("eps", 8 / 255))
            with ctx.timer("eval_apgd"):
                ap = apgd_ce_evaluate(self.model, xs, ys, eps=eps, bs=int(inter.get("batch_size", 250)), seed=self.attack_seed, restarts=int(inter.get("apgd_restarts", 1)), log_path=str(ctx.job_dir / "autoattack_log.txt"), iterations=inter.get("apgd_iterations"))
            rec["apgd_ce_subset"] = {"robust_acc": ap["robust_acc"], "clean_acc": ap["clean_acc"], "n": ap["n"], "time_s": ap["time_s"]}
            ctx.save_artifact_npy(f"apgd_mask_round{r}.npy", ap["robust_mask"])
            if int(inter.get("pgd_steps", 20)) > 0:
                with ctx.timer("eval_pgd"):
                    pg = pgd_evaluate(self.model, xs, ys, eps=eps, alpha=float(inter.get("pgd_alpha", 2 / 255)), steps=int(inter.get("pgd_steps", 20)), bs=int(inter.get("batch_size", 250)), seed=self.attack_seed)
                rec["pgd_subset"] = {"robust_acc": pg["robust_acc"], "n": pg["n"], "steps": pg["steps"], "time_s": pg["time_s"]}
            if self.diag.get("margins", True):
                with ctx.timer("eval_margins"):
                    rec["margins"] = margin_and_gradient_diagnostics(self.model, xs, ys, n_examples=int(self.diag.get("margin_examples", 1000)), perturbation_eps=eps, seed=self.attack_seed)
        if final:
            sweep = dict(self.rob.get("epsilon_sweep", {}))
            if sweep.get("enabled", False):
                n_sw = int(min(int(sweep.get("n_examples", 2000)), len(self.robust_subset)))
                xs = eval_transform(self.test.get(self.robust_subset[:n_sw]), self.ds.test_res)
                ys = torch.as_tensor(self.ds.test_y[self.robust_subset[:n_sw]], device=self.device)
                rec["epsilon_sweep"] = {}
                with ctx.timer("eval_eps_sweep"):
                    for e in sweep.get("epsilons", []):
                        ap = apgd_ce_evaluate(self.model, xs, ys, eps=float(e), bs=int(inter.get("batch_size", 250)), seed=self.attack_seed, restarts=1, log_path=str(ctx.job_dir / "autoattack_log.txt"), iterations=inter.get("apgd_iterations"))
                        rec["epsilon_sweep"][f"{float(e):.6f}"] = {"robust_acc": ap["robust_acc"], "n": ap["n"]}
            corr = dict(self.rob.get("corruptions", {}))
            if corr.get("enabled", False):
                from .corruptions import corruption_evaluate

                n_c = int(min(int(corr.get("n_examples", 2000)), len(self.robust_subset)))
                xs = eval_transform(self.test.get(self.robust_subset[:n_c]), self.ds.test_res)
                ys = torch.as_tensor(self.ds.test_y[self.robust_subset[:n_c]], device=self.device)
                with ctx.timer("eval_corruptions"):
                    rec["corruptions"] = corruption_evaluate(self.model, xs, ys, severities=list(corr.get("severities", [1, 2, 3])), seed=self.attack_seed)
        rec["eval_time_s"] = time.perf_counter() - t0
        self.ctx.event("round_evaluated", round=r, label_fraction=rec["label_fraction"], test_acc=ev.acc, test_loss=ev.loss, apgd=rec.get("apgd_ce_subset", {}).get("robust_acc"), pgd=rec.get("pgd_subset", {}).get("robust_acc"), kappa=self.monitor.kappa())
        return rec

    def final_autoattack(self, r: int) -> Optional[dict[str, Any]]:
        fin = dict(self.rob.get("final", {}))
        if not fin.get("enabled", True):
            return None
        idx = self.aa_subset  # fixed, class-stratified, identical for every method and seed (never a class-sorted prefix)
        n = int(len(idx))
        self.ctx.save_artifact_npy(f"aa_subset_round{r}.npy", idx)
        if n * 3 * self.ds.test_res * self.ds.test_res * 4 > 2e9:
            x = torch.cat([eval_transform(self.test.get(idx[s : s + 2000]), self.ds.test_res).cpu() for s in range(0, n, 2000)])
        else:
            x = eval_transform(self.test.get(idx), self.ds.test_res)
        y = torch.as_tensor(self.ds.test_y[idx], device=self.device)
        with self.ctx.timer("eval_autoattack"):
            res = autoattack_evaluate(self.model, x, y, eps=float(self.rob.get("eps", 8 / 255)), norm=str(self.rob.get("norm", "Linf")), version=str(fin.get("version", "standard")), attacks=fin.get("attacks"), bs=int(fin.get("batch_size", 250)), seed=self.attack_seed, log_path=str(self.ctx.job_dir / "autoattack_log.txt"), chunk=int(fin.get("chunk", 5000)), apgd_restarts=fin.get("apgd_restarts"), apgd_iterations=fin.get("apgd_iterations"))
        self.ctx.save_artifact_npy(f"aa_robust_mask_round{r}.npy", res["robust_mask"])
        self.ctx.save_artifact_npy(f"aa_clean_mask_round{r}.npy", res["clean_mask"])
        self.ctx.save_artifact_npy(f"aa_adv_pred_round{r}.npy", res["adv_pred"])
        out = {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}
        out["round"] = r
        self.ctx.event("autoattack_done", round=r, clean_acc=out["clean_acc"], robust_acc=out["robust_acc"], n=out["n"], time_s=round(out["time_s"], 1))
        return out

    # ------------------------------------------------------------------ acquisition
    @torch.no_grad()
    def forward_indices(self, indices: np.ndarray, layers: list[str]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self.model.eval()
        bs = int(self.al.get("acquisition_batch_size", 512))
        logits_l: list[torch.Tensor] = []
        feats_l: dict[str, list[torch.Tensor]] = {l: [] for l in layers}
        use_amp = bool(self.al.get("acquisition_amp", False)) and self.device.type == "cuda"
        self.train.prefetch([indices[s : s + bs] for s in range(0, len(indices), bs)])  # memmap mode only
        for s in range(0, len(indices), bs):
            xb = eval_transform(self.train.get(indices[s : s + bs]), self.ds.train_res)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits, feats = self.model.forward_features(xb)
            logits_l.append(logits.float())
            for l in layers:
                feats_l[l].append(feats[l].float())
        self.model.train()
        return torch.cat(logits_l), {l: torch.cat(v) for l, v in feats_l.items()}

    def class_statistics(self, layers: list[str]) -> dict[str, acq.ClassStats]:
        mode = str(self.al.get("mahalanobis_mode", "diag"))
        max_n = int(self.al.get("class_stats_max_examples", 60000))
        idx = self.labeled if len(self.labeled) <= max_n else np.sort(self.acq_rng.choice(self.labeled, size=max_n, replace=False))
        _, feats = self.forward_indices(idx, layers)
        y = torch.as_tensor(self.ds.train_y[idx], device=self.device)
        stats: dict[str, acq.ClassStats] = {}
        for l in layers:
            m = mode
            if m == "auto":
                m = "full" if self.num_classes * self.layer_dims[l] ** 2 <= 2**28 else "diag"
            stats[l] = acq.compute_class_stats(feats[l], y, self.num_classes, mode=m, shrinkage_abs=float(self.cfg.get("collapse", {}).get("covariance_shrinkage_epsilon", 1e-5)), shrinkage_rel=float(self.al.get("full_cov_shrinkage", 0.1)))
        return stats

    def acquire(self, r: int) -> dict[str, Any]:
        ctx = self.ctx
        strategy = str(self.method.get("acquisition", "random"))
        t0 = time.perf_counter()
        budget = int(min(self.round_budget, len(self.unlabeled)))
        remaining = budget
        pen = self.model.penultimate
        need_layers = sorted(set(self.layers) | {pen})
        labeled_feats = None
        labeled_reference_total = 0
        labeled_reference_used = 0
        class_stats = None
        if strategy == "coreset":
            # Exact CoreSet distances to the complete ImageNet labeled pool are
            # prohibitively expensive and can exceed device memory late in the
            # campaign.  The scalable implementation uses a deterministic
            # labeled reference subset and a Johnson--Lindenstrauss projection;
            # both sizes are written into the acquisition record.
            params = dict(self.method.get("params", {}))
            labeled_reference_total = int(len(self.labeled))
            max_ref = int(params.get("reference_size", labeled_reference_total))
            if max_ref > 0 and labeled_reference_total > max_ref:
                rr = np.random.default_rng(derive_seed(self.seed, "coreset_reference", r))
                ref_idx = np.sort(rr.choice(self.labeled, size=max_ref, replace=False))
            else:
                ref_idx = self.labeled
            labeled_reference_used = int(len(ref_idx))
            _, lf = self.forward_indices(ref_idx, [pen])
            labeled_feats = lf[pen]
        if strategy == "collapse":
            class_stats = self.class_statistics(self.layers)
        value_before = self.trainer.controller_value() if isinstance(self.controller, FullMACC) else None
        passes = candidate_passes(self.unlabeled, int(self.al.get("candidate_size", 50000)), derive_seed(self.seed, "passes", r))
        sel_frac = float(self.al.get("selected_fraction_per_pass", 0.10))
        selected_all: list[np.ndarray] = []
        pass_records: list[dict[str, Any]] = []
        first_scores: Optional[dict[str, np.ndarray]] = None
        newly = set()
        scored_unique: set[int] = set()
        diag_time = 0.0
        p = 0
        while remaining > 0 and p < 10 * max(1, len(passes)):
            cand_idx = passes[p % len(passes)]
            cand_idx = cand_idx[~np.isin(cand_idx, list(newly))] if newly else cand_idx
            if len(cand_idx) == 0:
                p += 1
                continue
            logits, feats = self.forward_indices(cand_idx, need_layers)
            scored_unique.update(int(i) for i in cand_idx)
            cand = acq.CandidateBatch(indices=cand_idx, logits=logits, feats=feats)
            if strategy == "collapse":
                tau = float(self.trainer.threshold_quantile)
                per_pass_cap = max(1, int(math.floor(len(cand_idx) * (1.0 - tau))))
            else:
                per_pass_cap = max(1, int(math.floor(len(cand_idx) * sel_frac)))
            cap = int(min(per_pass_cap, remaining))
            actx = acq.AcquisitionContext(method=self.method, params=dict(self.method.get("params", {})), model=self.model, device=self.device, rng=self.acq_rng, torch_gen=self.acq_gen, penultimate=pen, forward_fn=lambda ids: self.forward_indices(ids, [pen]), labeled_feats=labeled_feats, class_stats=class_stats, layer_weights=self.layer_weights, threshold_quantile=float(self.trainer.threshold_quantile))
            with ctx.timer("acquisition_select"):
                res = acq.select(strategy, cand, cap, actx)
            chosen = cand_idx[res.positions]
            selected_all.append(chosen)
            newly.update(int(i) for i in chosen)
            remaining -= len(chosen)
            if labeled_feats is not None:
                labeled_feats = torch.cat([labeled_feats, feats[pen][torch.as_tensor(res.positions, device=feats[pen].device)]])
            rec = {"pass": p, "n_candidates": int(len(cand_idx)), "cap": cap, "n_selected": int(len(chosen)), **{k: v for k, v in res.info.items() if isinstance(v, (int, float, str))}}
            if strategy == "coreset":
                rec["labeled_reference_total"] = labeled_reference_total
                rec["labeled_reference_used"] = labeled_reference_used
            pass_records.append(rec)
            if first_scores is None and bool(self.al.get("record_scores", True)):
                sc = {"entropy": acq.entropy_scores(logits).cpu().numpy(), "margin": acq.margin_scores(logits).cpu().numpy()}
                if res.scores is not None:
                    sc[strategy] = np.asarray(res.scores)
                if class_stats is None and strategy != "collapse" and bool(self.al.get("record_collapse_scores_for_baselines", True)):
                    t_diag = time.perf_counter()
                    try:
                        cs = self.class_statistics(self.layers)
                        tmp_ctx = acq.AcquisitionContext(method=self.method, params={}, model=self.model, device=self.device, rng=self.acq_rng, torch_gen=self.acq_gen, penultimate=pen, class_stats=cs, layer_weights=self.layer_weights)
                        pred = logits.argmax(dim=1)
                        tot = None
                        for l, st in cs.items():
                            u = acq.mahalanobis_scores(feats[l].to(self.device), pred.to(self.device), st)
                            tot = self.layer_weights[l] * u if tot is None else tot + self.layer_weights[l] * u
                        sc["collapse"] = tot.cpu().numpy()
                    except Exception as exc:  # diagnostics must not break acquisition
                        ctx.event("collapse_score_diag_failed", error=repr(exc))
                    diag_time += time.perf_counter() - t_diag
                first_scores = sc
                analysis = score_analysis(sc, res.positions)
                rec["score_analysis"] = analysis
                np.savez_compressed(ctx.artifact_dir / f"scores_round{r}.npz", candidates=cand_idx, **{k: v.astype(np.float32) for k, v in sc.items()})
            p += 1
        selected = np.concatenate(selected_all) if selected_all else np.zeros(0, dtype=np.int64)
        if len(np.unique(selected)) != len(selected):
            raise RuntimeError("duplicate acquisitions detected")
        if np.intersect1d(selected, self.labeled).size:
            raise RuntimeError("acquired an already-labeled example")
        self.labeled = np.sort(np.concatenate([self.labeled, selected]))
        self.unlabeled = np.setdiff1d(self.unlabeled, selected, assume_unique=True)
        ctx.save_artifact_npy(f"selected_round{r}.npy", selected)
        scored = sum(int(x["n_candidates"]) for x in pass_records)
        rec = {
            "round": r,
            "strategy": strategy,
            "budget": budget,
            "n_selected": int(len(selected)),
            "n_passes": len(pass_records),
            "candidates_scored": scored,
            "candidates_scored_unique": int(len(scored_unique)),
            "pool_size_before": int(len(self.unlabeled) + len(selected)),
            "coverage_of_pool": len(scored_unique) / max(1, len(self.unlabeled) + len(selected)),
            "repeat_rate": 1.0 - len(scored_unique) / max(1, scored),
            "time_s": time.perf_counter() - t0 - diag_time,
            "diagnostic_time_s": diag_time,
            "class_histogram": class_histogram(self.ds.train_y[selected], self.num_classes),
            "tau": float(self.trainer.threshold_quantile),
            "score_analysis": (pass_records[0].get("score_analysis") if pass_records else None),
            "passes": pass_records,
        }
        if isinstance(self.controller, FullMACC):
            # the tau of the latest controller action governed this acquisition; reward it after training begins
            if self.controller.has_pending_acquisition():
                ctx.event("acquisition_reward_skipped", round=r, reason="previous acquisition action was never rewarded (phase shorter than full_macc.acquisition_reward_steps?)")
            self.controller.mark_acquisition(value_before, int(len(selected)), step_index=self.trainer.global_step)
            rec["controller_value_before"] = value_before
            if not self.controller.has_pending_acquisition():
                ctx.event("acquisition_not_rewardable", round=r, reason="the policy has not acted yet (no controller step before this round); tau came from the initial action")
        ctx.event("acquired", round=r, strategy=strategy, n_selected=int(len(selected)), n_passes=len(pass_records), time_s=round(rec["time_s"], 1), n_labeled=int(len(self.labeled)))
        return rec

    # ------------------------------------------------------------------ run
    def run(self) -> dict[str, Any]:
        ctx = self.ctx
        self.setup()
        start_round = 0
        resumed = False
        latest = self.latest_round_checkpoint()
        if latest is not None:
            state = ctx.load_checkpoint(f"round_{latest}.pt")
            if state is not None:
                self.restore(state)
                start_round = latest + 1
                resumed = True
                ctx.event("resumed", from_round=latest, n_labeled=int(len(self.labeled)))
        if not resumed:
            loaded = False
            if self.mode != "initial_only" and bool(self.cfg.get("shared_initial", {}).get("use", True)):
                loaded = self.try_load_shared_initial()
            if not loaded:
                stats = self.trainer.train_phase(self.labeled, self.initial_epochs(), float(self.tr.get("lr", 0.1)), "initial", 0, (len(self.labeled) + len(self.ctrl_val)) / len(self.ds.train_y), warmup_epochs=float(self.tr.get("warmup_epochs", 1.0)))
                rec = self.evaluate_round(0, stats, final=(self.n_rounds == 0 and self.mode != "initial_only"))
                self.rounds_records = [rec]
                if bool(self.cfg.get("shared_initial", {}).get("save", True)) and not self.method.get("full_supervision", False):
                    self.save_shared_initial(rec)
            self.save_round_checkpoint(0)
            start_round = 1
        for r in range(start_round, self.n_rounds + 1):
            acq_rec = self.acquire(r)
            self.acq_records.append(acq_rec)
            frac = (len(self.labeled) + len(self.ctrl_val)) / len(self.ds.train_y)
            stats = self.trainer.train_phase(self.labeled, int(self.tr.get("epochs_per_round", 20)), float(self.tr.get("lr", 0.1)) * float(self.tr.get("round_lr_factor", 0.5)), f"round{r}", r, frac, warmup_epochs=float(self.tr.get("warmup_epochs_round", 0.5)))
            rec = self.evaluate_round(r, stats, acquisition={k: v for k, v in acq_rec.items() if k != "passes"}, final=(r == self.n_rounds))
            rounds_to_aa = self.rob.get("evaluate_rounds", ["final"])
            if rounds_to_aa == "all" or (isinstance(rounds_to_aa, list) and r in rounds_to_aa):
                rec["autoattack"] = self.final_autoattack(r)
            self.rounds_records.append(rec)
            self.save_round_checkpoint(r)
            _debug_stop_round(r)
        # final AutoAttack on the last model
        final_r = self.n_rounds
        if "autoattack" not in self.rounds_records[-1] or self.rounds_records[-1].get("autoattack") is None:
            aa = self.final_autoattack(final_r)
            self.rounds_records[-1]["autoattack"] = aa
            atomic_write_json(ctx.artifact_dir / "rounds.json", {"rounds": self.rounds_records, "acquisitions": self.acq_records})
        ctx.save_checkpoint("final_model.pt", {"model": self.model.state_dict(), "arch": self.cfg["arch"], "num_classes": self.num_classes, "mean": self.ds.mean, "std": self.ds.std})
        traj = getattr(self.controller, "history", [])
        atomic_write_json(ctx.artifact_dir / "controller_trajectory.json", {"regularizer": self.regularizer, "targets": self.targets, "history": traj})
        final = self.rounds_records[-1]
        summary = {
            "task": "vision_al",
            "dataset": self.ds.name,
            "arch": self.cfg["arch"],
            "method": self.method.get("name"),
            "acquisition": self.method.get("acquisition"),
            "regularizer": self.regularizer,
            "dcr_lambda": self.method.get("dcr_lambda"),
            "seed": self.seed,
            "mode": self.mode,
            "num_classes": self.num_classes,
            "n_train": int(len(self.ds.train_y)),
            "n_test": int(len(self.ds.test_y)),
            "dataset_fingerprint": self.ds.fingerprint,
            "model_params": self.n_params,
            "model_init": getattr(self, "init_report", {"init": "scratch"}),
            "controller_params": int(getattr(self.controller, "num_parameters", 0)),
            "layers": self.layers,
            "layer_dims": self.layer_dims,
            "kappa_targets": self.targets,
            "dcr_target_ranks": {l: s.rank for l, s in self.dcr.layers.items()},
            "rounds": self.rounds_records,
            "acquisitions": [{k: v for k, v in a.items() if k != "passes"} for a in self.acq_records],
            "labels": {
                # labels *available* to the method (its annotation budget, carve-out included) versus labels
                # actually used for gradient steps; they differ by the controller-validation carve-out for
                # every method including supervised_full (100% budget, trained on all but the carve-out).
                "labels_available": int(len(self.labeled) + len(self.ctrl_val)),
                "labels_trained": int(len(self.labeled)),
                "controller_validation": int(len(self.ctrl_val)),
                "fraction_available": round((len(self.labeled) + len(self.ctrl_val)) / len(self.ds.train_y), 4),
                "fraction_trained": round(len(self.labeled) / len(self.ds.train_y), 4),
                "full_supervision": bool(self.method.get("full_supervision", False)),
            },
            "final": {
                "label_fraction": final["label_fraction"],
                "labels_available": int(len(self.labeled) + len(self.ctrl_val)),
                "labels_trained": int(len(self.labeled)),
                "test_acc": final["test_acc"],
                "test_loss": final["test_loss"],
                "test_ece": final["test_ece"],
                "aa_robust_acc": (final.get("autoattack") or {}).get("robust_acc"),
                "aa_clean_acc": (final.get("autoattack") or {}).get("clean_acc"),
                "aa_n": (final.get("autoattack") or {}).get("n"),
                "aa_time_s": (final.get("autoattack") or {}).get("time_s"),
                "apgd_ce_subset_acc": (final.get("apgd_ce_subset") or {}).get("robust_acc"),
                "pgd_subset_acc": (final.get("pgd_subset") or {}).get("robust_acc"),
                "kappa_train": final.get("kappa_train"),
                "nc": final.get("nc"),
                "margins": final.get("margins"),
                "corruption_mean_acc": (final.get("corruptions") or {}).get("corruption_mean_acc"),
                "epsilon_sweep": final.get("epsilon_sweep"),
            },
            "compute": {
                "train_time_s": sum(float(r_["phase"].get("phase_time_s", 0.0)) for r_ in self.rounds_records if r_.get("phase")),
                "acquisition_time_s": sum(float(a.get("time_s", 0.0)) for a in self.acq_records),
                "images_per_s_by_phase": {r_["phase"]["phase"]: r_["phase"].get("images_per_s") for r_ in self.rounds_records if r_.get("phase")},
                "total_steps": self.trainer.global_step,
            },
            "shared_initial_key": shared_initial_key(self.cfg),
            "initial_job": getattr(self, "initial_job", self.ctx.job_id),
        }
        return summary


def run(ctx: JobContext) -> dict[str, Any]:
    return VisionALJob(ctx).run()
