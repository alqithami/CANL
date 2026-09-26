"""Vision datasets with uint8 array caches and device-resident batching.

Supported ``dataset`` names
---------------------------
``cifar10``, ``cifar100``
    torchvision downloads; native 32x32.
``tiny_imagenet``
    Hugging Face ``zh-plus/tiny-imagenet`` (200 classes, 100k/10k, 64x64).
``imagenet100``
    Hugging Face ``clane9/imagenet-100`` (100 classes, 126,689 train / 5,000 val). Images are
    resized (shorter side) and centre-cropped to ``cache_resolution`` (default 144) once; training
    uses GPU random-resized crops to ``train_resolution`` (default 128) and evaluation a centre crop.
``imagenet1k``
    Local ImageNet-1K under ``<data_root>/imagenet`` (class-folder val, flat val + ``LOC_val_solution.csv``,
    or the Kaggle ``ILSVRC/Data/CLS-LOC`` tree; see :func:`resolve_imagenet_layout`) or the gated Hugging
    Face ``ILSVRC/imagenet-1k`` (needs ``HF_TOKEN``).  The production protocol caches at 256 px (~250 GB,
    served from the memmap by :class:`DeviceArray` when it does not fit in RAM).
``synthetic``
    Structured random data for smoke tests (no downloads).

Every cache directory holds ``train_x.npy`` (N,H,W,3 uint8), ``train_y.npy``, ``test_x.npy``,
``test_y.npy`` and ``meta.json`` with a fingerprint recorded in each job summary.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch import Tensor

from ..utils.io import atomic_write_json, ensure_dir, read_json

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

DATASET_INFO: dict[str, dict[str, Any]] = {
    "cifar10": {"num_classes": 10, "mean": CIFAR10_MEAN, "std": CIFAR10_STD, "train_res": 32, "test_res": 32, "aug": "cifar", "expected_train": 50000, "expected_test": 10000},
    "cifar100": {"num_classes": 100, "mean": CIFAR100_MEAN, "std": CIFAR100_STD, "train_res": 32, "test_res": 32, "aug": "cifar", "expected_train": 50000, "expected_test": 10000},
    "tiny_imagenet": {"num_classes": 200, "mean": IMAGENET_MEAN, "std": IMAGENET_STD, "train_res": 64, "test_res": 64, "aug": "rrc", "cache_res": 64, "expected_train": 100000, "expected_test": 10000},
    "imagenet100": {"num_classes": 100, "mean": IMAGENET_MEAN, "std": IMAGENET_STD, "train_res": 128, "test_res": 128, "aug": "rrc", "cache_res": 144, "expected_train": 126689, "expected_test": 5000},
    "imagenet1k": {"num_classes": 1000, "mean": IMAGENET_MEAN, "std": IMAGENET_STD, "train_res": 224, "test_res": 224, "aug": "rrc", "cache_res": 256, "expected_train": 1281167, "expected_test": 50000},
    "synthetic": {"num_classes": 10, "mean": (0.5, 0.5, 0.5), "std": (0.25, 0.25, 0.25), "train_res": 32, "test_res": 32, "aug": "cifar", "expected_train": None, "expected_test": None},
    # smoke test of the ImageNet loader: JPEG trees in the ImageNet layouts, decoded by the same builder as ImageNet-1K
    "synthetic_imagefolder": {"num_classes": 10, "mean": IMAGENET_MEAN, "std": IMAGENET_STD, "train_res": 56, "test_res": 56, "aug": "rrc", "cache_res": 64, "expected_train": None, "expected_test": None},
}


@dataclass
class VisionDataset:
    name: str
    num_classes: int
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    train_res: int
    test_res: int
    aug: str
    fingerprint: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def cache_res(self) -> int:
        return int(self.train_x.shape[1])


def _fingerprint(train_y: np.ndarray, test_y: np.ndarray, train_x: np.ndarray, test_x: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(train_y, dtype=np.int64).tobytes())
    h.update(np.asarray(test_y, dtype=np.int64).tobytes())
    # sample of pixels (cheap, deterministic)
    n = train_x.shape[0]
    idx = np.linspace(0, n - 1, num=min(n, 256), dtype=np.int64)
    h.update(np.ascontiguousarray(train_x[idx]).tobytes())
    m = test_x.shape[0]
    idx = np.linspace(0, m - 1, num=min(m, 256), dtype=np.int64)
    h.update(np.ascontiguousarray(test_x[idx]).tobytes())
    return h.hexdigest()


def cache_dir_for(name: str, cache_root: Path, options: Mapping[str, Any]) -> Path:
    tag = name
    res = options.get("cache_resolution")
    if res:
        tag += f"-r{int(res)}"
    if name == "synthetic":
        tag += f"-c{options.get('num_classes', 10)}-n{options.get('train_size', 4000)}-t{options.get('test_size', 1000)}-s{options.get('res', 32)}"
    if name == "synthetic_imagefolder":
        tag += f"-c{options.get('num_classes', 10)}-n{options.get('per_class', 40)}-{options.get('layout', 'class_folders')}"
    return ensure_dir(Path(cache_root) / "vision" / tag)


def build_synthetic_imagefolder(root: Path, num_classes: int = 10, per_class: int = 40, per_class_val: int = 8, layout: str = "class_folders", size: int = 80, seed: int = 0) -> Path:
    """Write a tiny ImageNet-shaped JPEG tree (wnid folders; val either in class folders or flat + LOC_val_solution.csv)."""
    from PIL import Image

    root = Path(root)
    if (root / ".complete").exists():
        return root
    rng = np.random.default_rng(seed)
    wnids = [f"n{1000000 + i:08d}" for i in range(num_classes)]
    protos = rng.uniform(0.1, 0.9, size=(num_classes, 3))
    yy, xx = np.mgrid[0:size, 0:size] / size

    def image(c: int) -> Image.Image:
        freq = 1 + c % 5
        pattern = 0.5 + 0.5 * np.sin(2 * np.pi * freq * (xx * np.cos(c) + yy * np.sin(c)) + rng.uniform(0, 6.28))
        arr = protos[c][None, None, :] * (0.5 + 0.5 * pattern[..., None]) + rng.normal(0, 0.08, size=(size, size, 3))
        return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

    train = root / "train"
    for c, w in enumerate(wnids):
        (train / w).mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            image(c).save(train / w / f"{w}_{i}.JPEG", quality=90)
    val = root / "val"
    rows = []
    k = 0
    for c, w in enumerate(wnids):
        for i in range(per_class_val):
            k += 1
            if layout == "class_folders":
                (val / w).mkdir(parents=True, exist_ok=True)
                image(c).save(val / w / f"ILSVRC2012_val_{k:08d}.JPEG", quality=90)
            else:
                val.mkdir(parents=True, exist_ok=True)
                name = f"ILSVRC2012_val_{k:08d}"
                image(c).save(val / f"{name}.JPEG", quality=90)
                rows.append((name, f"{w} 1 1 10 10"))
    if layout != "class_folders":
        (root / "LOC_val_solution.csv").write_text("ImageId,PredictionString\n" + "\n".join(f"{a},{b}" for a, b in rows) + "\n")
    (root / ".complete").write_text("ok")
    return root


# ----------------------------------------------------------------------------- builders
def _build_cifar(name: str, data_root: Path, out: Path) -> None:
    import torchvision

    cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
    tr = cls(root=str(data_root / "torchvision"), train=True, download=True)
    te = cls(root=str(data_root / "torchvision"), train=False, download=True)
    np.save(out / "train_x.npy", np.asarray(tr.data, dtype=np.uint8))
    np.save(out / "train_y.npy", np.asarray(tr.targets, dtype=np.int64))
    np.save(out / "test_x.npy", np.asarray(te.data, dtype=np.uint8))
    np.save(out / "test_y.npy", np.asarray(te.targets, dtype=np.int64))
    atomic_write_json(out / "meta.json", {"source": f"torchvision.{cls.__name__}", "classes": list(tr.classes)})


def _resize_center_crop(img, res: int):
    from PIL import Image

    img = img.convert("RGB")
    w, h = img.size
    scale = res / min(w, h)
    nw, nh = max(res, round(w * scale)), max(res, round(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    left, top = (nw - res) // 2, (nh - res) // 2
    return np.asarray(img.crop((left, top, left + res, top + res)), dtype=np.uint8)


class _HFDecode(torch.utils.data.Dataset):
    def __init__(self, hf_ds, image_col: str, label_col: str, res: int):
        self.ds, self.image_col, self.label_col, self.res = hf_ds, image_col, label_col, res

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        img = row[self.image_col]
        arr = _resize_center_crop(img, self.res)
        return torch.from_numpy(np.array(arr, copy=True)), int(row[self.label_col])


def _materialize(loader_ds, out_x: Path, out_y: Path, n: int, res: int, workers: int, desc: str) -> None:
    x = np.lib.format.open_memmap(out_x, mode="w+", dtype=np.uint8, shape=(n, res, res, 3))
    y = np.zeros(n, dtype=np.int64)
    dl = torch.utils.data.DataLoader(loader_ds, batch_size=256, num_workers=workers, shuffle=False)
    pos = 0
    t0 = time.time()
    for xb, yb in dl:
        b = xb.shape[0]
        x[pos : pos + b] = xb.numpy()
        y[pos : pos + b] = yb.numpy()
        pos += b
        if pos % (256 * 40) == 0:
            print(f"[data] {desc}: {pos}/{n} ({pos / max(time.time() - t0, 1e-6):.0f} img/s)", flush=True)
    x.flush()
    del x
    np.save(out_y, y)


def _build_hf(name: str, data_root: Path, out: Path, res: int, workers: int, options: Mapping[str, Any]) -> None:
    from datasets import load_dataset

    if name == "tiny_imagenet":
        repo, train_split, test_split = "zh-plus/tiny-imagenet", "train", "valid"
    elif name == "imagenet100":
        repo, train_split, test_split = options.get("hf_repo", "clane9/imagenet-100"), "train", "validation"
    elif name == "imagenet1k":
        repo, train_split, test_split = "ILSVRC/imagenet-1k", "train", "validation"
    else:
        raise ValueError(name)
    token = os.environ.get("HF_TOKEN")
    ds = load_dataset(repo, cache_dir=str(data_root / "hf"), token=token)
    tr, te = ds[train_split], ds[test_split]
    image_col = "image" if "image" in tr.column_names else tr.column_names[0]
    label_col = "label" if "label" in tr.column_names else [c for c in tr.column_names if c != image_col][0]
    _materialize(_HFDecode(tr, image_col, label_col, res), out / "train_x.npy", out / "train_y.npy", len(tr), res, workers, f"{name} train")
    _materialize(_HFDecode(te, image_col, label_col, res), out / "test_x.npy", out / "test_y.npy", len(te), res, workers, f"{name} test")
    names = None
    try:
        names = list(tr.features[label_col].names)
    except Exception:
        pass
    atomic_write_json(out / "meta.json", {"source": f"hf:{repo}", "classes": names, "cache_resolution": res})


def resolve_imagenet_layout(root: Path) -> dict[str, Any]:
    """Locate ImageNet-1K train/val images under ``root``.

    Accepted layouts (checked in this order):

    * ``root/train/<wnid>/*.JPEG`` and ``root/val/<wnid>/*.JPEG`` (validation already sorted into
      class folders, e.g. with the usual ``valprep.sh``);
    * ``root/train/<wnid>/*.JPEG`` and a *flat* ``root/val/*.JPEG`` plus ``root/LOC_val_solution.csv``;
    * the Kaggle ``imagenet-object-localization-challenge`` tree
      ``root/ILSVRC/Data/CLS-LOC/{train,val}`` with ``root/LOC_val_solution.csv``.
    """
    candidates = [root, root / "ILSVRC" / "Data" / "CLS-LOC"]
    for base in candidates:
        train, val = base / "train", base / "val"
        if not train.is_dir() or not val.is_dir():
            continue
        val_has_classes = any(p.is_dir() for p in val.iterdir())
        csv = None
        for c in (root / "LOC_val_solution.csv", base / "LOC_val_solution.csv", base.parent / "LOC_val_solution.csv"):
            if c.exists():
                csv = c
                break
        if val_has_classes or csv is not None:
            return {"train": train, "val": val, "val_sorted": val_has_classes, "val_csv": csv}
    raise FileNotFoundError(
        f"ImageNet-1K not found under {root}: expected train/<wnid>/ and val/<wnid>/ (or flat val/ + LOC_val_solution.csv, "
        "or the Kaggle ILSVRC/Data/CLS-LOC layout). Set dataset option 'imagenet_dir' or use source: hf with HF_TOKEN."
    )


def _build_imagefolder(name: str, data_root: Path, out: Path, res: int, workers: int, options: Mapping[str, Any]) -> None:
    import torchvision

    root = Path(options.get("imagenet_dir", data_root / "imagenet"))
    layout = resolve_imagenet_layout(root)
    tr = torchvision.datasets.ImageFolder(str(layout["train"]))
    class_to_idx = dict(tr.class_to_idx)
    if layout["val_sorted"]:
        te = torchvision.datasets.ImageFolder(str(layout["val"]))
        if list(te.classes) != list(tr.classes):
            raise RuntimeError("ImageNet val class folders do not match the train class folders")
        te_samples = list(te.samples)
    else:
        import csv as _csv

        labels: dict[str, str] = {}
        with open(layout["val_csv"], newline="") as fh:
            for row in _csv.DictReader(fh):
                labels[row["ImageId"]] = row["PredictionString"].split()[0]
        te_samples = []
        for img in sorted(p for ext in ("*.JPEG", "*.jpeg", "*.jpg", "*.JPG") for p in layout["val"].glob(ext)):
            wnid = labels.get(img.stem)
            if wnid is None or wnid not in class_to_idx:
                raise RuntimeError(f"no label for validation image {img.name} in {layout['val_csv']}")
            te_samples.append((str(img), class_to_idx[wnid]))
        if not te_samples:
            raise RuntimeError(f"no validation images found under {layout['val']}")
        if name == "imagenet1k" and len(te_samples) != 50000:
            print(f"[data] WARNING: {len(te_samples)} validation images found (expected 50000)", flush=True)

    class _Wrap(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            path, label = self.samples[i]
            from PIL import Image

            with Image.open(path) as img:
                arr = _resize_center_crop(img, res)
            return torch.from_numpy(np.array(arr, copy=True)), int(label)

    _materialize(_Wrap(tr.samples), out / "train_x.npy", out / "train_y.npy", len(tr.samples), res, workers, f"{name} train")
    _materialize(_Wrap(te_samples), out / "test_x.npy", out / "test_y.npy", len(te_samples), res, workers, f"{name} test")
    atomic_write_json(out / "meta.json", {"source": f"imagefolder:{layout['train'].parent}", "classes": list(tr.classes), "cache_resolution": res, "val_layout": "class_folders" if layout["val_sorted"] else f"csv:{layout['val_csv']}"})


def _build_synthetic(out: Path, options: Mapping[str, Any]) -> None:
    C = int(options.get("num_classes", 10))
    n_train = int(options.get("train_size", 4000))
    n_test = int(options.get("test_size", 1000))
    res = int(options.get("res", 32))
    rng = np.random.default_rng(int(options.get("seed", 1234)))
    # class prototypes: smooth colour blobs + class-specific frequency pattern
    protos = rng.uniform(0.15, 0.85, size=(C, 3))
    yy, xx = np.mgrid[0:res, 0:res] / res

    def make(n):
        y = rng.integers(0, C, size=n)
        x = np.zeros((n, res, res, 3), dtype=np.float32)
        for i in range(n):
            c = y[i]
            freq = 1 + c % 4
            phase = rng.uniform(0, 2 * np.pi)
            pattern = 0.5 + 0.5 * np.sin(2 * np.pi * freq * (xx * np.cos(c) + yy * np.sin(c)) + phase)
            base = protos[c][None, None, :] * (0.6 + 0.4 * pattern[..., None])
            shift = rng.integers(-3, 4, size=2)
            base = np.roll(base, shift, axis=(0, 1))
            x[i] = np.clip(base + rng.normal(0, 0.12, size=base.shape), 0, 1)
        return (x * 255).astype(np.uint8), y.astype(np.int64)

    xtr, ytr = make(n_train)
    xte, yte = make(n_test)
    np.save(out / "train_x.npy", xtr)
    np.save(out / "train_y.npy", ytr)
    np.save(out / "test_x.npy", xte)
    np.save(out / "test_y.npy", yte)
    atomic_write_json(out / "meta.json", {"source": "synthetic", "classes": [f"c{i}" for i in range(C)], "num_classes": C})


def prepare_vision_dataset(name: str, data_root: Path, cache_root: Path, options: Mapping[str, Any] | None = None, workers: int | None = None) -> Path:
    options = dict(options or {})
    info = DATASET_INFO[name]
    res = int(options.get("cache_resolution", info.get("cache_res", info["train_res"])))
    if name != "synthetic":
        options["cache_resolution"] = res
    out = cache_dir_for(name, Path(cache_root), options)
    done = out / "meta.json"
    if done.exists() and (out / "train_x.npy").exists() and (out / "test_y.npy").exists():
        return out
    workers = int(workers or max(2, min(16, (os.cpu_count() or 4) - 1)))
    print(f"[data] building cache for {name} at {out}", flush=True)
    if name in ("cifar10", "cifar100"):
        _build_cifar(name, Path(data_root), out)
    elif name in ("tiny_imagenet", "imagenet100"):
        _build_hf(name, Path(data_root), out, res, workers, options)
    elif name == "imagenet1k":
        if options.get("source", "imagefolder") == "hf":
            _build_hf(name, Path(data_root), out, res, workers, options)
        else:
            _build_imagefolder(name, Path(data_root), out, res, workers, options)
    elif name == "synthetic":
        _build_synthetic(out, options)
    elif name == "synthetic_imagefolder":
        tree = build_synthetic_imagefolder(Path(cache_root) / "vision" / f"synthetic_imagefolder_tree-{options.get('layout', 'class_folders')}", int(options.get("num_classes", 10)), int(options.get("per_class", 40)), int(options.get("per_class_val", 8)), str(options.get("layout", "class_folders")))
        _build_imagefolder(name, Path(data_root), out, res, workers, {"imagenet_dir": str(tree)})
    else:
        raise ValueError(f"unknown dataset {name}")
    return out


def stratified_subset_indices(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Class-stratified subset of size ``n`` (sorted indices), deterministic in ``seed``."""
    if n is None or n >= len(y):
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per = n // len(classes)
    picks = [rng.choice(np.flatnonzero(y == c), size=min(per, int((y == c).sum())), replace=False) for c in classes]
    out = np.concatenate(picks)
    if len(out) < n:
        rest = np.setdiff1d(np.arange(len(y)), out)
        out = np.concatenate([out, rng.choice(rest, size=n - len(out), replace=False)])
    return np.sort(out)


def load_vision_dataset(name: str, data_root: Path | str, cache_root: Path | str, options: Mapping[str, Any] | None = None, mmap: bool = True) -> VisionDataset:
    options = dict(options or {})
    info = DATASET_INFO[name]
    out = prepare_vision_dataset(name, Path(data_root), Path(cache_root), options)
    mode = "r" if mmap else None
    train_x = np.load(out / "train_x.npy", mmap_mode=mode)
    train_y = np.load(out / "train_y.npy")
    test_x = np.load(out / "test_x.npy", mmap_mode=mode)
    test_y = np.load(out / "test_y.npy")
    meta = read_json(out / "meta.json")
    num_classes = int(options.get("num_classes", meta.get("num_classes", info["num_classes"])))
    exp_tr, exp_te = info.get("expected_train"), info.get("expected_test")
    if exp_tr and len(train_y) != exp_tr:
        print(f"[data] WARNING: {name} train size {len(train_y)} != expected {exp_tr}", flush=True)
    if exp_te and len(test_y) != exp_te:
        print(f"[data] WARNING: {name} test size {len(test_y)} != expected {exp_te}", flush=True)
    sub_tr = options.get("train_subset")
    sub_te = options.get("test_subset")
    if sub_tr:
        idx = stratified_subset_indices(train_y, int(sub_tr), int(options.get("subset_seed", 0)))
        train_x, train_y = np.ascontiguousarray(train_x[idx]), train_y[idx]
    if sub_te:
        idx = stratified_subset_indices(test_y, int(sub_te), int(options.get("subset_seed", 0)) + 1)
        test_x, test_y = np.ascontiguousarray(test_x[idx]), test_y[idx]
    fp = _fingerprint(train_y, test_y, train_x, test_x)
    train_res = int(options.get("train_resolution", info["train_res"]))
    test_res = int(options.get("test_resolution", info["test_res"]))
    return VisionDataset(
        name=name,
        num_classes=num_classes,
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        mean=tuple(info["mean"]),
        std=tuple(info["std"]),
        train_res=train_res,
        test_res=test_res,
        aug=str(options.get("aug", info["aug"])),
        fingerprint=fp,
        meta={"cache_dir": str(out), "source": meta.get("source"), "n_train": int(len(train_y)), "n_test": int(len(test_y))},
    )


# ----------------------------------------------------------------------------- device storage
def physical_memory_bytes() -> int:
    """RAM available to this process: min(physical RAM, cgroup limit) — containers do not virtualise /proc/meminfo."""
    try:
        total = int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        total = 64 * 1024**3
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
            if raw and raw != "max":
                lim = int(raw)
                if 0 < lim < total:
                    total = lim
        except (OSError, ValueError):
            continue
    return total


def host_budget_bytes(max_host_fraction: float) -> int:
    """Host-RAM budget of one job: a fraction of the RAM divided by the number of jobs sharing the pod (CAENL_PARALLEL)."""
    parallel = max(1, int(os.environ.get("CAENL_PARALLEL", "1") or 1))
    return int(max_host_fraction * physical_memory_bytes() / parallel)


def _chunked_nchw(x: np.ndarray, device: Optional[torch.device], pin: bool, chunk: int = 2048) -> Tensor:
    """Materialise ``[N,H,W,3]`` uint8 (possibly a memmap) as a ``[N,3,H,W]`` tensor without a 2x transient copy."""
    n, h, w, c = x.shape
    if device is not None and device.type == "cuda":
        out = torch.empty((n, c, h, w), dtype=torch.uint8, device=device)
    else:
        out = torch.empty((n, c, h, w), dtype=torch.uint8, pin_memory=bool(pin))
    for s in range(0, n, chunk):
        blk = torch.from_numpy(np.array(x[s : s + chunk], copy=True)).permute(0, 3, 1, 2)
        out[s : s + blk.shape[0]].copy_(blk)
    return out


def _as_index_array(idx: np.ndarray | Tensor) -> np.ndarray:
    if torch.is_tensor(idx):
        idx = idx.detach().cpu().numpy()
    return np.asarray(idx, dtype=np.int64).reshape(-1)


class _MemmapPrefetcher:
    """Background reader for ``DeviceArray`` in ``memmap`` mode.

    ``schedule(batches)`` hands over the exact sequence of index batches the consumer will
    request (the trainer knows every batch of an epoch in advance because the order is
    seeded); a worker thread gathers them from the memmap into pinned host buffers ``depth``
    batches ahead.  ``take(idx)`` returns the next prefetched batch if it matches ``idx``
    exactly, otherwise ``None`` (the caller then reads synchronously).  Row gathers are done
    in sorted order for locality and un-permuted on the fly.
    """

    def __init__(self, array: np.ndarray, pin: bool, depth: int = 6):
        import queue
        import threading

        self.array = array
        self.pin = pin
        self.depth = int(depth)
        self._queue_cls = queue.Queue
        self._thread_cls = threading.Thread
        self.ready: "queue.Queue" = queue.Queue(maxsize=self.depth)
        self.free: "queue.Queue" = queue.Queue()
        self.thread = None
        self.stop_flag = False
        self.batches: list[np.ndarray] = []
        self.misses = 0
        self._head: Optional[tuple[np.ndarray, Tensor, np.ndarray]] = None

    # -- buffers
    def _buffer(self, shape: tuple[int, ...]) -> Tensor:
        try:
            buf = self.free.get_nowait()
            if tuple(buf.shape) == tuple(shape):
                return buf
        except Exception:
            pass
        t = torch.empty(shape, dtype=torch.uint8)
        if self.pin:
            try:
                t = t.pin_memory()
            except Exception:
                pass
        return t

    def _worker(self, batches: list[np.ndarray], ready) -> None:
        # `batches`/`ready` are bound per schedule so that an orphaned worker can never write into a newer queue
        try:
            for idx in batches:
                if self.stop_flag:
                    break
                order = np.argsort(idx, kind="stable")
                buf = self._buffer((len(idx),) + tuple(self.array.shape[1:]))
                # np.take releases the GIL during the copy, so page faults on the memmap do not stall the trainer;
                # mode="clip" avoids the buffered double copy of the default bounds-checked path
                np.take(self.array, idx[order], axis=0, out=buf.numpy(), mode="clip")
                ready.put((idx, buf, np.argsort(order)))
        except BaseException as exc:  # noqa: BLE001  -> surfaced in take(); never leave the consumer waiting
            ready.put(("error", exc))
            return
        ready.put(None)

    def schedule(self, batches: list[np.ndarray]) -> None:
        self.cancel()
        self.batches = [np.asarray(b, dtype=np.int64) for b in batches]
        self.stop_flag = False
        self.ready = self._queue_cls(maxsize=self.depth)
        self.misses = 0
        self.thread = self._thread_cls(target=self._worker, args=(self.batches, self.ready), daemon=True, name="caenl-prefetch")
        self.thread.start()

    def take(self, idx: np.ndarray) -> Optional[tuple[Tensor, np.ndarray]]:
        """Next prefetched batch as ``(sorted_rows, inverse_order)`` if it is exactly ``idx``, else ``None``.

        A non-matching request (e.g. a controller-validation pass in the middle of an epoch)
        leaves the prefetched head in place, so the training schedule continues afterwards.
        """
        if self.thread is None:
            return None
        if self._head is None:
            item = self.ready.get()
            if item is None:
                self.thread = None
                return None
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "error":
                self.thread = None
                raise RuntimeError("memmap prefetch worker failed") from item[1]
            self._head = item
        got, buf, inv = self._head
        if len(got) != len(idx) or not np.array_equal(got, idx):
            self.misses += 1
            return None
        self._head = None
        return buf, inv

    def release(self, buf: Tensor) -> None:
        self.free.put(buf)

    def cancel(self) -> None:
        if self._head is not None:
            self.release(self._head[1])
            self._head = None
        if self.thread is None:
            return
        self.stop_flag = True
        self._drain()
        self.thread.join(timeout=60)
        self._drain()
        self.thread = None

    def _drain(self) -> None:
        try:
            while True:
                item = self.ready.get_nowait()
                if item is not None and not (isinstance(item, tuple) and len(item) == 2 and item[0] == "error"):
                    self.release(item[1])
        except Exception:
            pass


class DeviceArray:
    """uint8 image array ``[N,H,W,3]`` served as ``[B,3,H,W]`` device batches.

    Storage is chosen from the array size: on the GPU when it fits
    (``max_device_fraction`` of the free memory), otherwise in pinned host memory when it fits
    (``max_host_fraction`` of the physical RAM), otherwise the array stays a NumPy memmap on
    disk (``location == "memmap"``; ImageNet-1K at 256 px is ~250 GB) and batches are gathered
    on demand, optionally prefetched by a background thread via :meth:`prefetch`.
    """

    def __init__(self, x: np.ndarray, device: torch.device, max_device_fraction: float = 0.45, force_host: bool = False, max_host_fraction: float = 0.4, force_memmap: bool = False):
        self.n = int(x.shape[0])
        self.device = device
        self.shape = tuple(x.shape)
        nbytes = int(np.prod(x.shape))
        self.nbytes = nbytes
        on_device = False
        if device.type == "cuda" and not force_host and not force_memmap:
            free, total = torch.cuda.mem_get_info(device)
            on_device = nbytes < max_device_fraction * free
        self.host_budget = host_budget_bytes(max_host_fraction)
        in_host = (not force_memmap) and nbytes < self.host_budget
        self._prefetcher: Optional[_MemmapPrefetcher] = None
        self._pinned: dict[tuple[int, ...], list[Tensor]] = {}
        if on_device:
            self.x = _chunked_nchw(x, device, pin=False)  # N,3,H,W on the GPU, built chunk by chunk
            self.location = "device"
        elif in_host:
            pinned = device.type == "cuda"
            try:
                self.x = _chunked_nchw(x, None, pin=pinned)
            except RuntimeError:  # pinning can fail on hosts with locked-memory limits
                self.x = _chunked_nchw(x, None, pin=False)
            self.location = "host"
        else:
            self.x = x  # memmap (or plain array) kept as [N,H,W,3] uint8
            self.location = "memmap"
            self._prefetcher = _MemmapPrefetcher(x, pin=device.type == "cuda")

    def __len__(self) -> int:
        return self.n

    # ------------------------------------------------------------------ memmap helpers
    def prefetch(self, batches: list[np.ndarray | Tensor]) -> None:
        """Announce the exact sequence of index batches that will be requested next (memmap mode only)."""
        if self._prefetcher is None:
            return
        self._prefetcher.schedule([_as_index_array(b) for b in batches])

    def cancel_prefetch(self) -> None:
        if self._prefetcher is not None:
            self._prefetcher.cancel()

    def _staging(self, shape: tuple[int, ...]) -> Tensor:
        pool = self._pinned.setdefault(tuple(shape), [])
        if len(pool) < 2:
            t = torch.empty(shape, dtype=torch.uint8)
            if self.device.type == "cuda":
                try:
                    t = t.pin_memory()
                except Exception:
                    pass
            pool.append(t)
        pool.append(pool.pop(0))  # rotate: the buffer returned was used least recently
        return pool[-1]

    def _gather_memmap(self, idx: np.ndarray) -> tuple[Tensor, np.ndarray]:
        order = np.argsort(idx, kind="stable")
        buf = self._staging((len(idx),) + tuple(self.shape[1:]))
        np.take(self.x, idx[order], axis=0, out=buf.numpy())
        return buf, np.argsort(order)

    def get(self, idx: np.ndarray | Tensor) -> Tensor:
        """Return uint8 [B,3,H,W] on the target device (row order preserved)."""
        if self.location == "device":
            idx_t = torch.as_tensor(np.asarray(idx) if not torch.is_tensor(idx) else idx, device=self.device, dtype=torch.long)
            return self.x.index_select(0, idx_t)
        if self.location == "host":
            idx_t = torch.as_tensor(np.asarray(idx) if not torch.is_tensor(idx) else idx.cpu(), dtype=torch.long)
            return self.x.index_select(0, idx_t).to(self.device, non_blocking=True)
        idx_np = _as_index_array(idx)
        item = self._prefetcher.take(idx_np) if self._prefetcher is not None else None
        from_prefetch = item is not None
        buf, inv = item if item is not None else self._gather_memmap(idx_np)
        rows = buf.to(self.device, non_blocking=False)  # synchronous copy: staging buffers are reused immediately
        inv_t = torch.as_tensor(inv, device=self.device, dtype=torch.long)
        out = rows.index_select(0, inv_t).permute(0, 3, 1, 2).contiguous()  # always a fresh tensor (also on CPU, where .to() aliases)
        if from_prefetch:
            self._prefetcher.release(buf)
        return out


def fixed_evaluation_subset(test_y: np.ndarray, n: Optional[int], dataset: str, tag: str) -> np.ndarray:
    """Evaluation subset shared by every method and seed of a dataset (class-stratified, seeded by ``tag``).

    Used for the final AutoAttack subset (``tag="autoattack"``) and the per-round proxies; it
    must never be a prefix of the test set, whose order is class-sorted for ImageNet.
    """
    from ..utils.seeding import derive_seed

    if n is None or int(n) >= len(test_y):
        return np.arange(len(test_y))
    seed = derive_seed(0, tag, dataset)
    idx = stratified_subset_indices(np.asarray(test_y), int(n), seed)
    return idx[np.random.default_rng(seed + 1).permutation(len(idx))]  # shuffled so that every prefix is itself a fair subsample
