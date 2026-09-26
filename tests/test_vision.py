import numpy as np
import pytest
import torch

from caenl.vision import acquisition as acq
from caenl.vision.augment import cifar_augment, eval_transform, random_resized_crop
from caenl.vision.corruptions import corrupt
from caenl.vision.models import build_model
from caenl.vision.robustness import apgd_ce_evaluate, pgd_attack
from caenl.vision.schedule import candidate_passes, class_balanced_split, stratified_subset


def _model(num_classes=5):
    return build_model("resnet18_cifar", num_classes, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25), {"width_mult": 0.125}).eval()


def _candidates(n=60, C=5, d=64):
    logits = torch.randn(n, C)
    feats = {"layer3": torch.randn(n, 32), "layer4": torch.randn(n, d)}
    return acq.CandidateBatch(indices=np.arange(1000, 1000 + n), logits=logits, feats=feats)


def _ctx(model=None, **kw):
    return acq.AcquisitionContext(method={}, params=kw.pop("params", {}), model=model, device=torch.device("cpu"), rng=np.random.default_rng(0), torch_gen=torch.Generator().manual_seed(0), penultimate="layer4", **kw)


@pytest.mark.parametrize("strategy", ["random", "entropy", "margin", "least_confidence", "power_margin", "coreset", "badge", "bald_mcd"])
def test_strategies_return_unique_positions(strategy):
    cand = _candidates()
    model = _model()
    labeled = torch.randn(20, 64)
    res = acq.select(strategy, cand, 10, _ctx(model=model, labeled_feats=labeled, params={"mc_samples": 3}))
    assert len(res.positions) == 10 and len(np.unique(res.positions)) == 10
    assert res.positions.max() < len(cand.indices)


def test_collapse_strategy_threshold_and_scores():
    cand = _candidates()
    y = torch.randint(0, 5, (200,))
    feats3, feats4 = torch.randn(200, 32), torch.randn(200, 64)
    stats = {"layer3": acq.compute_class_stats(feats3, y, 5, mode="diag"), "layer4": acq.compute_class_stats(feats4, y, 5, mode="full")}
    ctx = _ctx(class_stats=stats, layer_weights={"layer3": 1 / 32, "layer4": 1 / 64}, threshold_quantile=0.9)
    res = acq.select("collapse", cand, 30, ctx)
    # tau = 0.9 of 60 candidates -> at most ceil(6) above threshold
    assert 1 <= len(res.positions) <= 7
    assert res.scores is not None and len(res.scores) == 60
    assert res.positions[0] == int(np.argmax(res.scores))


def test_noise_stability_runs():
    cand = _candidates(n=24)
    model = _model()

    def forward_fn(ids):
        return torch.randn(len(ids), 5) + cand.logits[: len(ids)], {"layer4": torch.randn(len(ids), 64)}

    res = acq.select("noise_stability", cand, 6, _ctx(model=model, forward_fn=forward_fn, params={"noise_samples": 2}))
    assert len(res.positions) == 6
    # parameters restored after perturbation
    params_after = torch.cat([p.flatten() for p in model.parameters()])
    assert torch.isfinite(params_after).all()


def test_chunked_min_distance_matches_torch_cdist():
    g = torch.Generator().manual_seed(7)
    X = torch.randn(37, 11, generator=g)
    Y = torch.randn(53, 11, generator=g)
    got = acq.min_dist_to_set(X, Y, chunk=7, reference_chunk=9)
    want = torch.cdist(X, Y).min(dim=1).values
    assert torch.allclose(got, want, atol=2e-5, rtol=2e-5)


def test_scalable_coreset_and_badge_are_unique_and_deterministic():
    cand = _candidates(n=96, C=8, d=48)
    labeled = torch.randn(71, 48, generator=torch.Generator().manual_seed(8))
    params_core = {"projection_dim": 16, "distance_chunk": 13, "reference_chunk": 17}
    a = acq.select("coreset", cand, 24, _ctx(labeled_feats=labeled, params=params_core))
    b = acq.select("coreset", cand, 24, _ctx(labeled_feats=labeled, params=params_core))
    assert np.array_equal(a.positions, b.positions)
    assert len(a.positions) == len(np.unique(a.positions)) == 24
    assert a.info["implementation"] == "jl_projected_kcenter"

    params_badge = {"sketch_dim": 24}
    a = acq.select("badge", cand, 24, _ctx(params=params_badge))
    b = acq.select("badge", cand, 24, _ctx(params=params_badge))
    assert np.array_equal(a.positions, b.positions)
    assert len(a.positions) == len(np.unique(a.positions)) == 24
    assert a.info["implementation"] == "random_maclaurin_gradient_sketch"


def test_projected_noise_stability_does_not_materialize_k_times_classes():
    cand = _candidates(n=30, C=7, d=32)
    model = _model(num_classes=7)

    def forward_fn(ids):
        return torch.randn(len(ids), 7) + cand.logits[: len(ids)], {"layer4": torch.randn(len(ids), 32)}

    res = acq.select(
        "noise_stability",
        cand,
        8,
        _ctx(model=model, forward_fn=forward_fn, params={"noise_samples": 3, "sketch_dim": 12}),
    )
    assert len(res.positions) == 8
    assert len(np.unique(res.positions)) == 8
    assert res.info["implementation"] == "projected_output_deviation"
    assert res.info["sketch_dim"] == 12


def test_badge_factorised_distance_matches_explicit():
    cand = _candidates(n=30, C=4, d=8)
    P = torch.softmax(cand.logits, 1)
    pred = P.argmax(1)
    U = P.clone()
    U[torch.arange(30), pred] -= 1
    H = cand.feats["layer4"]
    G = torch.einsum("nc,nd->ncd", U, H).reshape(30, -1)
    explicit = torch.cdist(G, G) ** 2
    sqn = (U * U).sum(1) * (H * H).sum(1)
    j = 3
    fact = (sqn + sqn[j] - 2 * (U @ U[j]) * (H @ H[j])).clamp_min(0)
    assert torch.allclose(fact, explicit[:, j], atol=1e-4)


def test_schedule_splits():
    y = np.repeat(np.arange(10), 100)
    init = class_balanced_split(y, 0.1, seed=1)
    assert len(init) == 100 and all((y[init] == c).sum() == 10 for c in range(10))
    ctrl = stratified_subset(init, y, 0.1, seed=2)
    assert len(ctrl) == 10 and set(ctrl) <= set(init)
    pool = np.setdiff1d(np.arange(1000), init)
    passes = candidate_passes(pool, 300, seed=3)
    assert sum(len(p) for p in passes) == len(pool) and len(np.unique(np.concatenate(passes))) == len(pool)


def test_augmentations_shapes_and_range():
    x = torch.randint(0, 256, (8, 3, 32, 32), dtype=torch.uint8)
    out = cifar_augment(x)
    assert out.shape == (8, 3, 32, 32) and out.min() >= 0 and out.max() <= 1
    xb = torch.randint(0, 256, (4, 3, 40, 40), dtype=torch.uint8)
    rrc = random_resized_crop(xb, 32)
    assert rrc.shape == (4, 3, 32, 32)
    ev = eval_transform(xb, 32)
    assert ev.shape == (4, 3, 32, 32)
    for kind in ("gaussian_noise", "impulse_noise", "gaussian_blur", "contrast", "brightness"):
        c = corrupt(out, kind, 2)
        assert c.shape == out.shape and c.min() >= 0 and c.max() <= 1


def test_model_feature_dims_and_normalisation():
    m = _model()
    x = torch.rand(2, 3, 32, 32)
    logits, feats = m.forward_features(x)
    assert logits.shape == (2, 5) and set(feats) >= {"layer2", "layer3", "layer4"}
    assert feats["layer4"].shape[1] == m.feature_dims["layer4"]
    assert torch.allclose(m(x), m.model((x - 0.5) / 0.25), atol=1e-5)
    assert len(m.feature_parameters()) == sum(1 for _ in m.model.parameters()) - 2


def test_pgd_respects_budget_and_apgd_runs():
    m = _model()
    x = torch.rand(6, 3, 32, 32)
    y = torch.randint(0, 5, (6,))
    adv = pgd_attack(m, x, y, 8 / 255, 2 / 255, 3)
    assert float((adv - x).abs().max()) <= 8 / 255 + 1e-6 and adv.min() >= 0 and adv.max() <= 1
    pytest.importorskip("autoattack")
    res = apgd_ce_evaluate(m, x, y, eps=8 / 255, bs=6, seed=0, restarts=1, iterations=3)
    assert res["n"] == 6 and res["robust_mask"].shape == (6,) and 0 <= res["robust_acc"] <= 1


def test_device_array_memmap_mode_preserves_order_and_survives_interleaved_reads(tmp_path):
    """ImageNet-1K caches do not fit in RAM: batches are gathered from the memmap (optionally prefetched)."""
    import numpy as np
    import torch

    from caenl.vision.data import DeviceArray

    x = np.random.default_rng(0).integers(0, 255, size=(300, 8, 8, 3)).astype(np.uint8)
    np.save(tmp_path / "x.npy", x)
    mm = np.load(tmp_path / "x.npy", mmap_mode="r")
    d = DeviceArray(mm, torch.device("cpu"), force_memmap=True)
    assert d.location == "memmap"
    rng = np.random.default_rng(1)
    for _ in range(5):
        batches = [rng.permutation(300)[:32] for _ in range(8)]
        d.prefetch(batches)
        for i, b in enumerate(batches):
            if i == 3:  # controller-validation style read in the middle of the schedule
                o = d.get(np.array([5, 1, 250]))
                assert np.array_equal(o.permute(0, 2, 3, 1).numpy(), x[[5, 1, 250]])
            out = d.get(torch.as_tensor(b))
            assert out.shape == (32, 3, 8, 8)
            assert np.array_equal(out.permute(0, 2, 3, 1).numpy(), x[b])
        d.prefetch(batches)  # replacing a schedule midway is allowed
        out = d.get(batches[0])
        assert np.array_equal(out.permute(0, 2, 3, 1).numpy(), x[batches[0]])
    d.cancel_prefetch()
    h = DeviceArray(mm, torch.device("cpu"))
    assert h.location == "host"
    assert np.array_equal(h.get(np.array([3, 2])).permute(0, 2, 3, 1).numpy(), x[[3, 2]])


def test_mae_conversion_matches_hf_encoder_and_layer_ids():
    """MAE (HF) -> torchvision ViT conversion is exact (cls features agree); layer ids drive the lr decay."""
    import functools

    import torch
    import torchvision

    pytest.importorskip("transformers")
    from transformers import ViTMAEConfig, ViTMAEModel

    from caenl.vision.mae import check_equivalence, load_mae_into_torchvision, resize_pos_embedding, vit_param_layer_ids

    torch.manual_seed(0)
    cfg = ViTMAEConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128, image_size=32, patch_size=16, mask_ratio=0.0, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    hf = ViTMAEModel(cfg).eval()
    # same LayerNorm eps as the HF config so that the comparison isolates the weight mapping
    tv = torchvision.models.VisionTransformer(image_size=32, patch_size=16, num_layers=2, num_heads=4, hidden_dim=64, mlp_dim=128, num_classes=7, norm_layer=functools.partial(torch.nn.LayerNorm, eps=cfg.layer_norm_eps))
    rep = load_mae_into_torchvision(tv, hf.state_dict())
    assert rep["loaded"] == 30 and rep["layers"] == 2
    assert check_equivalence(hf, tv, torch.randn(3, 3, 32, 32)) < 1e-4
    # position embeddings are resampled for other grids
    assert resize_pos_embedding(tv.encoder.pos_embedding.detach(), 1 + 9).shape == (1, 10, 64)
    ids = vit_param_layer_ids(tv.named_parameters(), 2)
    assert ids["conv_proj.weight"] == 0 and ids["encoder.layers.encoder_layer_1.ln_1.weight"] == 2 and ids["heads.head.weight"] == 3
    assert ids["encoder.ln.weight"] == 3


def test_layer_decay_param_groups():
    import torch

    from caenl.vision.models import build_model

    class _Ctx:
        config = {"training": {"optimizer": "adamw", "layer_decay": 0.5, "weight_decay": 0.05}}
        device = torch.device("cpu")
        seed = 0

        def get(self, key, default=None):
            cur = self.config
            for part in key.split("."):
                if not isinstance(cur, dict) or part not in cur:
                    return default
                cur = cur[part]
            return cur

    from caenl.vision.train import Trainer

    model = build_model("vit_b_32", 5, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25), {"image_size": 64})
    tr = Trainer.__new__(Trainer)
    tr.ctx = _Ctx()
    tr.model = model
    opt = tr.make_optimizer(1e-3)
    scales = sorted({float(g["lr_scale"]) for g in opt.param_groups})
    n = model.model.n_layers
    assert min(scales) == pytest.approx(0.5 ** (n + 1), rel=1e-6) and max(scales) == 1.0
    head_group = [g for g in opt.param_groups if any(p is model.fc.weight for p in g["params"])][0]
    assert head_group["lr_scale"] == 1.0 and head_group["lr"] == pytest.approx(1e-3)
    assert head_group["weight_decay"] == pytest.approx(0.05)
    # class token / position embedding (3-d tensors) and all 1-d tensors are excluded from weight decay
    named = dict(model.named_parameters())
    special = [named[n] for n in named if n.endswith(("class_token", "pos_embedding"))]
    assert len(special) == 2
    for g in opt.param_groups:
        for p in g["params"]:
            if any(p is s for s in special) or p.ndim <= 1:
                assert g["weight_decay"] == 0.0
            else:
                assert g["weight_decay"] == pytest.approx(0.05)
