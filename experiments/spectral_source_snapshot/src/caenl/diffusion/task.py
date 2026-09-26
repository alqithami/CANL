"""DDPM fine-tuning on CIFAR-10 with spectrum DCR on the U-Net mid-block representation.

* model: ``google/ddpm-cifar10-32`` (pretrained, Apache-2.0) fine-tuned for ``train_steps`` under
  each method with identical data order, noise schedule and EMA; ``tiny_random`` for smoke tests;
* DCR: the spatially pooled mid-block activation of samples whose timestep lies in
  ``dcr_timestep_range`` is regularised toward a rank-r near-isotropic spectrum (manuscript
  Section 4.5: "DCR at select diffusion timesteps");
* controller value (Full MACC): decrease of the epsilon-prediction MSE on a fixed held-out set
  of test images with fixed noise and timesteps;
* metrics: FID / KID / IS over ``num_samples`` DDIM samples (torch-fidelity, InceptionV3) against
  the CIFAR-10 training set, plus the validation denoising loss and effective rank.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..core.control_loop import ControlLoop
from ..utils.io import file_lock, atomic_write_json
from ..utils.jobctx import JobContext
from ..utils.seeding import derive_seed, load_rng_state, rng_state_dict
from ..vision.data import DeviceArray, load_vision_dataset


def _debug_stop(step: int) -> None:
    v = os.environ.get("CAENL_DEBUG_STOP_AT_STEP")
    if v and step >= int(v):
        raise SystemExit(f"debug stop at step {step}")


#: torch-fidelity 0.4 feature extractor used by FID/KID/IS ("inception-v3-compat"); the reference set's cached
#: files are ``<input>-<extractor>-features-<layer>.pt`` (one per requested feature layer: "2048" for FID/KID and
#: "logits_unbiased" for IS) plus the FID statistics ``<input>-<extractor>-stat-fid-2048.pt``.  torch-fidelity
#: recomputes *all* feature layers when any one of them is missing, so the warm-up must request exactly the metric
#: set the jobs request (fid + kid + isc) and the lock guard must check the exact files.
FIDELITY_EXTRACTOR = "inception-v3-compat"
FIDELITY_REFERENCE_FILES = ("features-2048.pt", "features-logits_unbiased.pt", "stat-fid-2048.pt")


def fidelity_reference_files(cache_root: Path, input_name: str) -> list[Path]:
    return [Path(cache_root) / f"{input_name}-{FIDELITY_EXTRACTOR}-{suffix}" for suffix in FIDELITY_REFERENCE_FILES]


def fidelity_reference_cached(cache_root: Path, input_name: str) -> bool:
    return all(p.exists() for p in fidelity_reference_files(cache_root, input_name))


def prepare_assets(data_root: Path, cache_root: Path, options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Download the pretrained DDPM weights and the CIFAR-10 cache ahead of time."""
    options = dict(options or {})
    out: dict[str, Any] = {}
    model = str(options.get("model", "google/ddpm-cifar10-32"))
    if model != "tiny_random":
        from diffusers import DDPMScheduler, UNet2DModel

        UNet2DModel.from_pretrained(model, cache_dir=str(Path(data_root) / "hf"))
        DDPMScheduler.from_pretrained(model, cache_dir=str(Path(data_root) / "hf"))
        out["model"] = model
    ds = load_vision_dataset(str(options.get("dataset", "cifar10")), data_root, cache_root, options.get("dataset_options", {}))
    out["dataset"] = {"name": ds.name, "n_train": int(len(ds.train_y))}
    if bool(options.get("warm_fidelity_cache", True)) and str(options.get("dataset", "cifar10")) == "cifar10":
        # torch-fidelity caches the Inception features/statistics of the registered reference set; computing them once
        # here (CPU is fine) spares every diffusion job the same work and avoids concurrent writes to the cache.  The
        # call requests the same metrics as DiffusionJob.fidelity so that every cached file the jobs look up exists.
        import torch_fidelity

        fc_root = Path(cache_root) / "torch_fidelity"
        with file_lock(fc_root / ".ref.lock"):
            m = torch_fidelity.calculate_metrics(input1="cifar10-train", input2="cifar10-train", fid=True, kid=True, isc=True, cuda=torch.cuda.is_available(), verbose=False, cache_root=str(fc_root), datasets_root=str(Path(data_root) / "torch_fidelity"), samples_find_deep=False)
        missing = [p.name for p in fidelity_reference_files(fc_root, "cifar10-train") if not p.exists()]
        if missing:
            raise RuntimeError(f"torch-fidelity reference cache incomplete after warm-up: missing {missing} under {fc_root}")
        out["fidelity_reference"] = {"fid_self": float(m.get("frechet_inception_distance", float("nan"))), "isc_reference": float(m.get("inception_score_mean", float("nan"))), "cache_root": str(fc_root), "files": [p.name for p in fidelity_reference_files(fc_root, "cifar10-train")]}
    return out


class _EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def copy_to(self, model: torch.nn.Module) -> None:
        sd = {k: (v.to(model.state_dict()[k].dtype)) for k, v in self.shadow.items()}
        model.load_state_dict(sd)

    def state_dict(self):
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, s, device):
        self.shadow = {k: v.to(device).float() for k, v in s.items()}


class _Uint8Dataset(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray):
        self.x = x  # N,H,W,3 uint8

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, i):
        return torch.from_numpy(np.array(self.x[i], copy=True)).permute(2, 0, 1).contiguous()  # writable copy (memmap-safe)


class DiffusionJob:
    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        self.cfg = ctx.config
        self.dc = dict(self.cfg.get("diffusion", {}))
        self.method = dict(self.cfg.get("method", {}))
        self.device = ctx.device

    def setup(self) -> None:
        from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel

        ctx, dc = self.ctx, self.dc
        paths = self.cfg.get("paths", {})
        self.ds = load_vision_dataset(str(dc.get("dataset", "cifar10")), paths.get("data_root", "data"), paths.get("cache_root", "cache"), dc.get("dataset_options", {}))
        self.train = DeviceArray(self.ds.train_x, self.device, max_device_fraction=0.3)
        self.test = DeviceArray(self.ds.test_x, self.device, max_device_fraction=0.1)
        self.res = int(self.ds.train_x.shape[1])
        name = str(dc.get("model", "google/ddpm-cifar10-32"))
        torch.manual_seed(derive_seed(ctx.seed, "init"))
        if name == "tiny_random":
            mo = dc.get("model_options", {})
            self.unet = UNet2DModel(sample_size=self.res, in_channels=3, out_channels=3, layers_per_block=1, block_out_channels=tuple(mo.get("channels", (32, 64))), down_block_types=("DownBlock2D", "AttnDownBlock2D"), up_block_types=("AttnUpBlock2D", "UpBlock2D"), norm_num_groups=8).to(self.device)
            self.noise_sched = DDPMScheduler(num_train_timesteps=int(dc.get("num_train_timesteps", 1000)))
        else:
            self.unet = UNet2DModel.from_pretrained(name, cache_dir=str(Path(paths.get("data_root", "data")) / "hf")).to(self.device)
            self.noise_sched = DDPMScheduler.from_pretrained(name, cache_dir=str(Path(paths.get("data_root", "data")) / "hf"))
        self.T = int(self.noise_sched.config.num_train_timesteps)
        self.alphas_cumprod = self.noise_sched.alphas_cumprod.to(self.device)
        self.sampler = DDIMScheduler.from_config(self.noise_sched.config)
        # feature tap on the mid block
        self._feat: dict[str, torch.Tensor] = {}
        layer = str(dc.get("dcr_layer", "mid_block"))
        module = getattr(self.unet, layer)
        module.register_forward_hook(lambda m, i, o: self._feat.__setitem__("mid", o))
        with torch.no_grad():
            x0 = torch.zeros(2, 3, self.res, self.res, device=self.device)
            self.unet(x0, torch.tensor([0, 1], device=self.device))
        d = int(self._feat["mid"].shape[1])
        self.steps_total = int(dc.get("train_steps", 10000))
        self.loop = ControlLoop({"mid": d}, str(self.method.get("regularizer", "none")), float(self.method.get("dcr_lambda", 0.0)), self.cfg, self.device, derive_seed(ctx.seed, "controller"), self.steps_total, k=int(self.cfg.get("collapse", {}).get("truncated_spectral_rank", 128)), configured_rank=int(dc.get("configured_rank", max(1, d // 4))), target_effective_rank=dc.get("target_effective_rank"), cadence=int(dc.get("monitoring_cadence_steps", 50)), error_mode=str(dc.get("controller_error_mode", "absolute_log_ratio")), ema_decay=float(self.cfg.get("collapse", {}).get("covariance_ema_decay", 0.99)))
        self.t_lo, self.t_hi = [int(v) for v in dc.get("dcr_timestep_range", [0, 600])]
        self.optimizer = torch.optim.AdamW(self.unet.parameters(), lr=float(dc.get("lr", 2e-5)), weight_decay=float(dc.get("weight_decay", 0.0)))
        self.ema = _EMA(self.unet, float(dc.get("ema_decay", 0.9995)))
        self.bs = int(dc.get("batch_size", 128))
        self.step = 0
        self.amp = self.device.type == "cuda" and bool(dc.get("amp", True))
        self.order_gen = torch.Generator(device="cpu").manual_seed(derive_seed(ctx.seed, "order"))
        self.noise_gen = torch.Generator(device=self.device).manual_seed(derive_seed(ctx.seed, "noise"))
        # fixed validation batch (test images, fixed noise and timesteps) shared by all methods
        g = torch.Generator(device="cpu").manual_seed(derive_seed(0, "ddpm_val"))
        n_val = int(min(int(dc.get("val_examples", 512)), len(self.ds.test_y) // 2))
        n_rep = int(min(int(dc.get("report_val_examples", 1024)), len(self.ds.test_y) - n_val))
        perm = torch.randperm(len(self.ds.test_y), generator=g)
        # controller set (Full MACC reward) and the reported validation set are disjoint
        self.val_idx = perm[:n_val].numpy()
        self.val_noise = torch.randn(n_val, 3, self.res, self.res, generator=g).to(self.device)
        self.val_t = torch.randint(0, self.T, (n_val,), generator=g).to(self.device)
        self.rep_idx = perm[n_val : n_val + n_rep].numpy()
        self.rep_noise = torch.randn(n_rep, 3, self.res, self.res, generator=g).to(self.device)
        self.rep_t = torch.randint(0, self.T, (n_rep,), generator=g).to(self.device)
        ctx.event("setup", model=name, params=sum(p.numel() for p in self.unet.parameters()), steps=self.steps_total, batch_size=self.bs, mid_dim=d, dcr_rank=self.loop.summary()["dcr_target_ranks"], target=self.loop.targets, timestep_range=[self.t_lo, self.t_hi], T=self.T)

    def to_model_space(self, x_uint8: torch.Tensor, flip: bool = False) -> torch.Tensor:
        x = x_uint8.float() / 127.5 - 1.0
        if flip:
            mask = torch.rand(x.shape[0], device=x.device, generator=self.noise_gen) < 0.5
            x = torch.where(mask[:, None, None, None], x.flip(3), x)
        return x

    def denoise_loss(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xt = self.noise_sched.add_noise(x0, noise, t)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
            pred = self.unet(xt, t).sample
        return F.mse_loss(pred.float(), noise), self._feat["mid"]

    @torch.no_grad()
    def val_loss(self, report: bool = False) -> float:
        """Denoising MSE on the controller set (``report=False``) or the disjoint reported set."""
        self.unet.eval()
        idx_all, noise_all, t_all = (self.rep_idx, self.rep_noise, self.rep_t) if report else (self.val_idx, self.val_noise, self.val_t)
        total, n = 0.0, 0
        for s in range(0, len(idx_all), 256):
            idx = idx_all[s : s + 256]
            x0 = self.to_model_space(self.test.get(idx))
            loss, _ = self.denoise_loss(x0, noise_all[s : s + 256], t_all[s : s + 256])
            total += float(loss) * len(idx)
            n += len(idx)
        self.unet.train()
        return total / max(n, 1)

    def ctrl_value(self) -> float:
        return -self.val_loss()

    def save(self) -> None:
        self.ctx.save_checkpoint("state.pt", {"unet": self.unet.state_dict(), "ema": self.ema.state_dict(), "optimizer": self.optimizer.state_dict(), "step": self.step, "loop": self.loop.state_dict(), "rng": rng_state_dict(), "order_gen": self.order_gen.get_state(), "noise_gen": self.noise_gen.get_state().cpu(), "perm": getattr(self, "_perm", None), "pos": int(getattr(self, "_pos", 0))})

    def restore(self) -> bool:
        st = self.ctx.load_checkpoint("state.pt")
        if st is None:
            return False
        self.unet.load_state_dict(st["unet"])
        self.ema.load_state_dict(st["ema"], self.device)
        self.optimizer.load_state_dict(st["optimizer"])
        self.step = int(st["step"])
        self.loop.load_state_dict(st["loop"])
        load_rng_state(st["rng"])
        self.order_gen.set_state(torch.as_tensor(st["order_gen"], dtype=torch.uint8))
        self.noise_gen.set_state(torch.as_tensor(st["noise_gen"], dtype=torch.uint8).cpu())
        self._perm = st.get("perm")
        self._pos = int(st.get("pos", 0))
        self.ctx.event("resumed", step=self.step)
        return True

    @torch.no_grad()
    def sample(self, n: int, steps: int, bs: int = 500) -> np.ndarray:
        from diffusers import UNet2DModel

        ema_model = UNet2DModel.from_config(self.unet.config).to(self.device)
        self.ema.copy_to(ema_model)
        ema_model.eval()
        self.sampler.set_timesteps(int(steps))
        g = torch.Generator(device=self.device).manual_seed(derive_seed(self.ctx.seed, "sampling"))
        out = np.empty((n, self.res, self.res, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        for s in range(0, n, bs):
            b = min(bs, n - s)
            x = torch.randn(b, 3, self.res, self.res, device=self.device, generator=g)
            for t in self.sampler.timesteps:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
                    eps = ema_model(x, t).sample
                x = self.sampler.step(eps.float(), t, x, eta=0.0).prev_sample
            img = ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            out[s : s + b] = img
            if (s // bs) % 10 == 0:
                self.ctx.event("sampling_progress", done=s + b, total=n, elapsed_s=round(time.perf_counter() - t0, 1))
        del ema_model
        return out

    def fidelity(self, samples: np.ndarray) -> dict[str, Any]:
        import torch_fidelity

        ref = self.dc.get("fid_reference", "cifar10-train")
        input2: Any
        if ref == "cifar10-train" and self.ds.name == "cifar10":
            input2 = "cifar10-train"
        else:
            input2 = _Uint8Dataset(np.asarray(self.ds.train_x))
        kw = dict(input1=_Uint8Dataset(samples), input2=input2, cuda=self.device.type == "cuda", fid=True, kid=True, isc=True, verbose=False, kid_subset_size=int(min(1000, len(samples))), cache_root=str(Path(self.cfg.get("paths", {}).get("cache_root", "cache")) / "torch_fidelity"), datasets_root=str(Path(self.cfg.get("paths", {}).get("data_root", "data")) / "torch_fidelity"), samples_find_deep=False)
        if len(samples) < 2048:
            kw["kid_subset_size"] = max(2, len(samples) // 2)
            kw["isc_splits"] = 1
        cache_dir = Path(kw["cache_root"])
        if isinstance(input2, str) and not fidelity_reference_cached(cache_dir, input2):
            # first computation of the reference features/statistics (normally done by the prepare stage): one job at
            # a time, so that concurrent jobs never race on the same cache files
            with file_lock(cache_dir / ".ref.lock"):
                m = torch_fidelity.calculate_metrics(**kw)
        else:
            m = torch_fidelity.calculate_metrics(**kw)
        return {"fid": float(m.get("frechet_inception_distance", float("nan"))), "kid": float(m.get("kernel_inception_distance_mean", float("nan"))), "kid_std": float(m.get("kernel_inception_distance_std", float("nan"))), "inception_score": float(m.get("inception_score_mean", float("nan"))), "inception_score_std": float(m.get("inception_score_std", float("nan"))), "n_samples": int(len(samples)), "reference": str(ref)}

    def run(self) -> dict[str, Any]:
        ctx, dc = self.ctx, self.dc
        self.setup()
        self.restore()
        self.unet.train()
        n_train = len(self.ds.train_y)
        log_every = int(dc.get("log_every_steps", 20))
        eval_every = int(dc.get("eval_every_steps", 500))
        ckpt_every = int(dc.get("checkpoint_every_steps", 1000))
        t0 = time.perf_counter()
        ctx.log_metrics({"kind": "eval", "step": self.step, "val_loss": self.val_loss(report=True), **self.loop.record()})
        perm = getattr(self, "_perm", None)
        pos = int(getattr(self, "_pos", 0))
        if perm is None:
            perm = torch.randperm(n_train, generator=self.order_gen)
            pos = 0
        while self.step < self.steps_total:
            if pos + self.bs > n_train:
                perm = torch.randperm(n_train, generator=self.order_gen)
                pos = 0
            idx = perm[pos : pos + self.bs].numpy()
            pos += self.bs
            self._perm, self._pos = perm, pos
            x0 = self.to_model_space(self.train.get(idx), flip=True)
            noise = torch.randn(x0.shape, device=self.device, generator=self.noise_gen)
            t = torch.randint(0, self.T, (x0.shape[0],), device=self.device, generator=self.noise_gen)
            mse, mid = self.denoise_loss(x0, noise, t)
            mask = (t >= self.t_lo) & (t <= self.t_hi)
            feats = mid.float().mean(dim=(2, 3))
            sel = feats[mask] if int(mask.sum()) >= 8 else feats
            dcr_loss, per_layer = self.loop.dcr_loss({"mid": sel})
            loss = mse + dcr_loss
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), float(dc.get("grad_clip", 1.0)))
            self.optimizer.step()
            self.ema.update(self.unet)
            info = self.loop.update({"mid": sel.detach()}, value_fn=self.ctrl_value, extra_state=(self.step / max(1, self.steps_total),))
            self.step += 1
            if info:
                ctx.log_metrics({"kind": "controller", "step": self.step, **{k: v for k, v in info.items() if isinstance(v, (int, float)) or v is None}})
            if self.step % log_every == 0:
                el = time.perf_counter() - t0
                ctx.log_metrics({"kind": "train", "step": self.step, "loss": float(loss.detach()), "mse": float(mse.detach()), "dcr": float(dcr_loss.detach()), "steps_per_s": self.step / max(el, 1e-6), **{f"dcr/{k}": v for k, v in per_layer.items()}, **self.loop.record()})
            if self.step % eval_every == 0 or self.step == self.steps_total:
                vl = self.val_loss(report=True)
                ctx.log_metrics({"kind": "eval", "step": self.step, "val_loss": vl, **self.loop.record()})
                ctx.event("eval", step=self.step, val_loss=round(vl, 5), lambdas=self.loop.lambdas, eff_rank={k: round(v, 1) for k, v in self.loop.monitor.kappa().items()})
            if self.step % ckpt_every == 0 and self.step < self.steps_total:
                self.save()
            _debug_stop(self.step)
        train_time = time.perf_counter() - t0
        self.save()
        n_samples = int(dc.get("num_samples", 50000))
        with ctx.timer("sampling"):
            samples = self.sample(n_samples, int(dc.get("sampling_steps", 100)), bs=int(dc.get("sampling_batch_size", 500)))
        np.save(ctx.artifact_dir / "samples_uint8.npy", samples[: int(dc.get("save_samples", 1000))])
        with ctx.timer("fidelity"):
            fid = self.fidelity(samples)
        final_val = self.val_loss(report=True)
        atomic_write_json(ctx.artifact_dir / "controller_trajectory.json", {"regularizer": self.loop.regularizer, "targets": self.loop.targets, "history": self.loop.summary()["history"]})
        ctx.event("final", **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in fid.items()}, val_loss=round(final_val, 5))
        return {
            "task": "diffusion_ddpm",
            "dataset": self.ds.name,
            "method": self.method.get("name"),
            "regularizer": self.loop.regularizer,
            "dcr_lambda": self.method.get("dcr_lambda"),
            "seed": ctx.seed,
            "model": dc.get("model"),
            "model_params": sum(p.numel() for p in self.unet.parameters()),
            "controller_params": self.loop.summary()["controller_params"],
            "steps": self.step,
            "final": {**fid, "val_loss": final_val, **{f"effective_rank/{k}": v for k, v in self.loop.monitor.kappa().items()}, **{f"lambda/{k}": v for k, v in self.loop.lambdas.items()}},
            "compute": {"train_time_s": train_time, "steps_per_s": self.step / max(train_time, 1e-6), "sampling_time_s": ctx.timings.get("sampling"), "fidelity_time_s": ctx.timings.get("fidelity")},
            "sampler": {"type": "ddim", "steps": int(dc.get("sampling_steps", 100)), "eta": 0.0, "num_samples": n_samples},
            "control": {k: v for k, v in self.loop.summary().items() if k != "history"},
        }


def run(ctx: JobContext) -> dict[str, Any]:
    return DiffusionJob(ctx).run()
