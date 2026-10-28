"""Finite tensor tests against byte-pinned author implementations; no dataset access."""
from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
import time

import torch
from torch import nn
import torch.nn.functional as F

import methods as m


def _close(actual, expected, *, name, atol=2e-11, rtol=2e-10):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=name)


def _paired_grad(loss1, variables1, loss2, variables2, name):
    _close(loss1, loss2, name=f"{name}: value")
    for i, (grad1, grad2) in enumerate(zip(torch.autograd.grad(loss1, variables1),
                                         torch.autograd.grad(loss2, variables2))):
        _close(grad1, grad2, name=f"{name}: gradient {i}")


def test_source_bytes(device):
    assert len(m.verify_source_bytes()) == 5


def test_supcon_author_value_and_gradient(device):
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long, device=device)
    for tau in (0.07, 0.1, 0.2):
        raw = torch.randn(6, 2, 8, dtype=torch.float64, device=device)
        a = raw.clone().requires_grad_()
        b = raw.clone().requires_grad_()
        adapter = m.SupConLoss(tau, base_temperature=0.07)
        author = m._supcontrast.SupConLoss(temperature=tau, contrast_mode="all", base_temperature=0.07)
        ours = adapter(F.normalize(a, dim=2), labels)
        expected = author(F.normalize(b, dim=2), labels)
        _paired_grad(ours, (a,), expected, (b,), f"SupCon tau={tau}")


def test_supcon_two_view_wrapper(device):
    encoder = nn.Sequential(nn.BatchNorm1d(6), nn.Linear(6, 5)).to(device=device, dtype=torch.float64)
    model = m.SupConEncoder(encoder, feature_dim=5, projection_dim=128).to(device=device, dtype=torch.float64)
    first = torch.randn(4, 6, dtype=torch.float64, device=device)
    second = torch.randn(4, 6, dtype=torch.float64, device=device)
    result = model.forward_views(first, second)
    assert result.shape == (4, 2, 128)
    assert int(encoder[0].num_batches_tracked) == 1, "Views must share one encoder call"
    manual = F.normalize(model.projector(encoder(torch.cat((first, second)))), dim=1)
    manual = torch.stack((manual[:4], manual[4:]), dim=1)
    _close(result, manual, name="two-view author construction")
    _close(result.norm(dim=2), torch.ones(4, 2, dtype=torch.float64, device=device), name="unit projections")


def test_nonl_author_value_and_gradient(device):
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long, device=device)
    for tau in (0.07, 0.1, 0.2):
        raw_features = torch.randn(6, 8, dtype=torch.float64, device=device)
        raw_weights = torch.randn(4, 8, dtype=torch.float64, device=device)
        a, wa = raw_features.clone().requires_grad_(), raw_weights.clone().requires_grad_()
        b, wb = raw_features.clone().requires_grad_(), raw_weights.clone().requires_grad_()
        ours = m.NONLLoss(tau)(a, wa, labels)
        author = m._nc_reference.NONL(temperature=tau, symmetric=True, bidirectional=False)
        expected = author(F.normalize(b, dim=1), F.normalize(wb, dim=1).T, labels)
        _paired_grad(ours, (a, wa), expected, (b, wb), f"NONL tau={tau}")


def test_nonl_stability_and_degeneracy(device):
    features = torch.randn(6, 8, dtype=torch.float64, device=device, requires_grad=True)
    weights = torch.randn(4, 8, dtype=torch.float64, device=device, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long, device=device)
    loss = m.NONLLoss(1e-4)(features, weights, labels)
    assert bool(torch.isfinite(loss))
    assert all(bool(torch.isfinite(g).all()) for g in torch.autograd.grad(loss, (features, weights)))
    try:
        m.NONLLoss()(features, weights, torch.zeros(6, dtype=torch.long, device=device))
    except ValueError:
        pass
    else:
        raise AssertionError("Single-class NONL must fail before a divide-by-zero")


def test_dcr_author_value_and_gradient(device):
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long, device=device)
    raw = torch.randn(9, 8, dtype=torch.float64, device=device)
    initial = m.scatter_statistics(raw, labels)
    targets = m.FrozenTargets(float(initial["log_kappa"]) - 0.6, float(initial["between"]) * 1.4)
    cfg = m.GeometryConfig()
    for method, floor in (("ratio_dcr", 0.0), ("full_dcr", cfg.floor_weight)):
        a, b = raw.clone().requires_grad_(), raw.clone().requires_grad_()
        ours, records = m.geometry_penalty(method, a, labels, targets=targets, ratio_weight=0.1, cfg=cfg)
        reference_cfg = dict(scatter_epsilon=cfg.scatter_epsilon, feature_norm_epsilon=cfg.feature_norm_epsilon,
                             huber_delta=cfg.huber_delta, floor_weight=floor)
        expected, _ = m._aligned.aligned_loss(
            {"feature": b}, labels,
            {"feature": {"log_kappa_target": targets.log_kappa_target, "between_floor": targets.between_floor}},
            {"feature": 0.1}, reference_cfg,
        )
        _paired_grad(ours, (a,), expected, (b,), method)
        assert records["ratio_penalty"] > 0
        if method == "full_dcr":
            assert records["spread_penalty"] > 0


def test_plain_fisher_and_invariances(device):
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long, device=device)
    raw = torch.randn(6, 7, dtype=torch.float64, device=device)
    a, b = raw.clone().requires_grad_(), raw.clone().requires_grad_()
    ours, _ = m.geometry_penalty("plain_fisher", a, labels, ratio_weight=0.03)
    stats = m._aligned.class_scatter(b, labels)
    expected = 0.03 * (stats["within"] + 1e-6) / (stats["between"] + 1e-6)
    _paired_grad(ours, (a,), expected, (b,), "plain Fisher trace ratio")
    q, _ = torch.linalg.qr(torch.randn(7, 7, dtype=torch.float64, device=device))
    transformed = m.scatter_statistics(100 * raw @ q, labels * 3 + 7)
    original = m.scatter_statistics(raw, labels)
    for key in ("within", "between", "log_kappa"):
        _close(transformed[key], original[key], name=f"scatter invariance {key}")


def test_calibration_and_zero_gate(device):
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long, device=device)
    anchors = [(torch.randn(6, 7, dtype=torch.float64, device=device), labels) for _ in range(16)]
    targets = m.calibrate_targets(anchors)
    expected_q = sum(float(m.scatter_statistics(x, y)["log_kappa"]) for x, y in anchors) / 16 + math.log(0.9)
    assert abs(targets.log_kappa_target - expected_q) < 1e-12
    features = anchors[0][0].clone().requires_grad_()
    current = m.scatter_statistics(features.detach(), labels)
    satisfied = m.FrozenTargets(float(current["log_kappa"]) + 1, 0.001)
    loss, _ = m.geometry_penalty("ratio_dcr", features, labels, targets=satisfied)
    assert float(loss) == 0
    assert torch.count_nonzero(torch.autograd.grad(loss, features)[0]) == 0


def test_frozen_probe_and_normalized_classifier(device):
    backbone = nn.Sequential(nn.BatchNorm1d(6), nn.Linear(6, 5)).to(device=device, dtype=torch.float64)
    probe = m.FrozenLinearProbe(backbone, 5, 3).to(device=device, dtype=torch.float64).train()
    assert not backbone.training and probe.classifier.training
    inputs = torch.randn(6, 6, dtype=torch.float64, device=device)
    targets = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long, device=device)
    F.cross_entropy(probe(inputs), targets).backward()
    assert all(p.grad is None and not p.requires_grad for p in backbone.parameters())
    assert probe.classifier.weight.grad is not None
    normalized = m.NormalizedClassifier(nn.Linear(6, 5), 5, 3, temperature=0.1).to(device=device, dtype=torch.float64)
    logits, raw = normalized(inputs, return_features=True)
    expected = F.normalize(raw, dim=1) @ F.normalize(normalized.classifier.weight, dim=1).T / 0.1
    _close(logits, expected, name="normalized-classifier logits")
    assert normalized.classifier.bias is None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    torch.manual_seed(731)
    torch.set_num_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; no passing report written")
    results = []
    started = time.monotonic()
    for name, test in sorted(globals().copy().items()):
        if name.startswith("test_") and callable(test):
            test(torch.device(args.device))
            results.append({"name": name, "status": "PASS"})
    report = {"status": "PASS", "scope": "tensor implementation tests, no scientific finding or training run",
              "device": args.device, "torch": torch.__version__, "python": platform.python_version(),
              "tests": results, "elapsed_seconds": time.monotonic() - started}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
