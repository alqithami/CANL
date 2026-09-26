"""Classification backbones with named feature taps.

Every model exposes ``forward_features(x) -> (logits, feats)`` where ``feats`` maps the tap
names ``layer1..layer4`` (globally average-pooled block outputs; ``layer4`` is the penultimate
representation fed to the linear classifier) to ``[B, d]`` tensors, and ``feature_dims``.
``NormalizedModel`` puts input normalisation *inside* the module so attacks operate in
``[0,1]`` pixel space (manuscript/AutoAttack convention).
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNetCIFAR(nn.Module):
    """ResNet-18/34 variant for 32x32 inputs (3x3 stem, no max-pool)."""

    def __init__(self, num_classes: int, depth: int = 18, width_mult: float = 1.0, dropout: float = 0.0):
        super().__init__()
        blocks = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3]}[depth]
        w = [max(8, int(round(c * width_mult))) for c in (64, 128, 256, 512)]
        self.in_planes = w[0]
        self.conv1 = nn.Conv2d(3, w[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(w[0])
        self.layer1 = self._make(w[0], blocks[0], 1)
        self.layer2 = self._make(w[1], blocks[1], 2)
        self.layer3 = self._make(w[2], blocks[2], 2)
        self.layer4 = self._make(w[3], blocks[3], 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(w[3], num_classes)
        self.feature_dims = {"layer1": w[0], "layer2": w[1], "layer3": w[2], "layer4": w[3]}
        self.penultimate = "layer4"

    def _make(self, planes: int, n: int, stride: int) -> nn.Sequential:
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward_features(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        feats: dict[str, Tensor] = {}
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        feats["layer1"] = out.mean(dim=(2, 3))
        out = self.layer2(out)
        feats["layer2"] = out.mean(dim=(2, 3))
        out = self.layer3(out)
        feats["layer3"] = out.mean(dim=(2, 3))
        out = self.layer4(out)
        pooled = out.mean(dim=(2, 3))
        feats["layer4"] = pooled
        logits = self.fc(self.dropout(pooled))
        return logits, feats

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_features(x)[0]

    def head(self, feats: Tensor) -> Tensor:
        return self.fc(feats)


class TorchvisionResNet(nn.Module):
    def __init__(self, num_classes: int, depth: int = 50, pretrained: bool = False, weights: str | None = None):
        super().__init__()
        import torchvision

        fn = {18: torchvision.models.resnet18, 34: torchvision.models.resnet34, 50: torchvision.models.resnet50, 101: torchvision.models.resnet101}[depth]
        w = (weights or "DEFAULT") if pretrained else None
        net = fn(weights=w)
        if net.fc.out_features != num_classes:
            if pretrained:
                raise ValueError(f"pretrained weights {w} have {net.fc.out_features} outputs but the dataset has {num_classes} classes")
            net.fc = nn.Linear(net.fc.in_features, num_classes)
        self.net = net
        exp = 4 if depth >= 50 else 1
        self.feature_dims = {"layer1": 64 * exp, "layer2": 128 * exp, "layer3": 256 * exp, "layer4": 512 * exp}
        self.penultimate = "layer4"

    @property
    def fc(self) -> nn.Linear:
        return self.net.fc

    def forward_features(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        n = self.net
        feats: dict[str, Tensor] = {}
        out = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        out = n.layer1(out)
        feats["layer1"] = out.mean(dim=(2, 3))
        out = n.layer2(out)
        feats["layer2"] = out.mean(dim=(2, 3))
        out = n.layer3(out)
        feats["layer3"] = out.mean(dim=(2, 3))
        out = n.layer4(out)
        pooled = out.mean(dim=(2, 3))
        feats["layer4"] = pooled
        return n.fc(pooled), feats

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_features(x)[0]

    def head(self, feats: Tensor) -> Tensor:
        return self.net.fc(feats)


class TorchvisionViT(nn.Module):
    """ViT-B/16 (torchvision) with taps on the class token after encoder blocks 3, 6, 9, 12.

    ``init="scratch"`` (default) trains from random initialisation; ``init="mae"`` loads the label-free
    MAE encoder (``facebook/vit-mae-base``, see :mod:`caenl.vision.mae`) with a fresh classifier head.
    """

    def __init__(self, num_classes: int, image_size: int = 128, pretrained: bool = False, variant: str = "vit_b_16", init: str = "scratch", mae_repo: str = "facebook/vit-mae-base", hf_cache: str | None = None):
        super().__init__()
        import torchvision

        fn = getattr(torchvision.models, variant)
        weights = "DEFAULT" if pretrained else None
        net = fn(weights=weights, image_size=image_size) if not pretrained else fn(weights=weights)
        self.init_report: dict = {"init": init}
        if init == "mae":
            from .mae import fetch_mae_state_dict, load_mae_into_torchvision

            self.init_report.update(load_mae_into_torchvision(net, fetch_mae_state_dict(mae_repo, hf_cache)))
            self.init_report["repo"] = mae_repo
        elif init != "scratch":
            raise ValueError(f"unknown ViT init {init!r} (scratch | mae)")
        net.heads = nn.Linear(net.hidden_dim, num_classes)
        self.net = net
        n_layers = len(net.encoder.layers)
        self.n_layers = n_layers
        self.taps = {f"layer{i + 1}": int(round((i + 1) * n_layers / 4)) - 1 for i in range(4)}
        self.feature_dims = {k: net.hidden_dim for k in self.taps}
        self.penultimate = "layer4"

    def param_layer_ids(self, prefix: str = "") -> dict[str, int]:
        """Layer index per parameter (for layer-wise lr decay), keyed by the parameter names of ``self``."""
        from .mae import vit_param_layer_ids

        return {prefix + k: v for k, v in vit_param_layer_ids(self.named_parameters(), self.n_layers).items()}

    @property
    def fc(self) -> nn.Linear:
        return self.net.heads

    def forward_features(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        n = self.net
        x = n._process_input(x)
        b = x.shape[0]
        cls = n.class_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + n.encoder.pos_embedding
        x = n.encoder.dropout(x)
        feats: dict[str, Tensor] = {}
        inv = {v: k for k, v in self.taps.items()}
        for i, layer in enumerate(n.encoder.layers):
            x = layer(x)
            if i in inv:
                feats[inv[i]] = x[:, 0]
        x = n.encoder.ln(x)
        feats["layer4"] = x[:, 0]
        return n.heads(x[:, 0]), feats

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_features(x)[0]

    def head(self, feats: Tensor) -> Tensor:
        return self.net.heads(feats)


class NormalizedModel(nn.Module):
    def __init__(self, model: nn.Module, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    @property
    def feature_dims(self) -> dict[str, int]:
        return self.model.feature_dims

    @property
    def penultimate(self) -> str:
        return self.model.penultimate

    @property
    def fc(self) -> nn.Linear:
        return self.model.fc

    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std

    def forward_features(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        return self.model.forward_features(self.normalize(x))

    def forward(self, x: Tensor) -> Tensor:
        return self.model(self.normalize(x))

    def head(self, feats: Tensor) -> Tensor:
        return self.model.head(feats)

    def feature_parameters(self):
        """Parameters of the feature extractor (everything except the linear head)."""
        head = set(id(p) for p in self.fc.parameters())
        return [p for p in self.model.parameters() if id(p) not in head]


def build_model(arch: str, num_classes: int, mean, std, options: Mapping[str, Any] | None = None) -> NormalizedModel:
    options = dict(options or {})
    if arch in ("resnet18_cifar", "resnet34_cifar"):
        depth = 18 if arch == "resnet18_cifar" else 34
        net = ResNetCIFAR(num_classes, depth=depth, width_mult=float(options.get("width_mult", 1.0)), dropout=float(options.get("dropout", 0.0)))
    elif arch in ("resnet18", "resnet34", "resnet50", "resnet101"):
        net = TorchvisionResNet(num_classes, depth=int(arch[6:]), pretrained=bool(options.get("pretrained", False)), weights=options.get("weights"))
    elif arch in ("vit_b_16", "vit_b_32", "vit_l_16"):
        net = TorchvisionViT(num_classes, image_size=int(options.get("image_size", 128)), pretrained=bool(options.get("pretrained", False)), variant=arch, init=str(options.get("init", "scratch")), mae_repo=str(options.get("mae_repo", "facebook/vit-mae-base")), hf_cache=options.get("hf_cache"))
    elif arch == "vit_b_16_mae":  # same network, label-free MAE initialisation (distinct arch name => separate shared-initial groups and table rows)
        net = TorchvisionViT(num_classes, image_size=int(options.get("image_size", 224)), pretrained=False, variant="vit_b_16", init="mae", mae_repo=str(options.get("mae_repo", "facebook/vit-mae-base")), hf_cache=options.get("hf_cache"))
    else:
        raise ValueError(f"unknown architecture {arch}")
    return NormalizedModel(net, mean, std)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
