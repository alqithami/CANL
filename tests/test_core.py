import math

import numpy as np
import pytest
import torch

from caenl.core.controllers import FullMACC, MACCLite, build_controller, kappa_target_from_strength
from caenl.core.dcr import DCR, covariance_eigenvalues, normalized_topk_spectrum, target_spectrum
from caenl.core.monitor import ClassConditionalMonitor, SpectralMonitor
from caenl.core.nc import nc_metrics, scatter_traces
from caenl.core.stats import compare_paired, holm_bonferroni, mcnemar_test, paired_permutation_test, summarize
from conftest import make_etf_features


def test_nc_metrics_on_etf():
    feats, labels, means = make_etf_features()
    m = nc_metrics(feats, labels, 10, classifier_weight=means, predictions=labels)
    assert m["kappa"] < 0.02
    assert m["collapse_strength"] > 0.98
    assert m["nc2_etf_distance"] < 0.05
    assert m["nc2_equinorm_cv"] < 0.05
    assert abs(m["nc2_mean_cosine"] - (-1 / 9)) < 0.02
    assert m["nc3_self_duality"] < 0.05
    assert m["nc4_ncc_agreement"] == 1.0
    assert 8.5 < m["total_effective_rank"] < 9.6


def test_scatter_traces_consistency():
    feats, labels, _ = make_etf_features(noise=0.5)
    tr = scatter_traces(feats, labels, 10)
    assert abs(float(tr["collapse_strength"]) - 1 / (1 + float(tr["kappa"]))) < 1e-6


def test_target_spectrum_feasible_rank():
    t = target_spectrum(k=128, dimension=2048, num_classes=1000)
    assert t.shape == (128,) and torch.isclose(t.sum(), torch.tensor(1.0)) and torch.allclose(t, torch.full((128,), 1 / 128))
    t = target_spectrum(k=128, dimension=512, num_classes=10)
    assert int((t > 0).sum()) == 9
    t = target_spectrum(k=128, dimension=64, configured_rank=192)
    assert t.shape == (64,) and int((t > 0).sum()) == 64


def test_gram_trick_matches_full_covariance():
    z = torch.randn(40, 300)
    e1 = covariance_eigenvalues(z)
    zc = z - z.mean(0, keepdim=True)
    e2 = torch.linalg.eigvalsh(zc.T @ zc / 40).flip(0)[:40]
    assert torch.allclose(e1, e2, atol=1e-4)


def test_dcr_zero_on_etf_and_gradients_flow():
    feats, labels, _ = make_etf_features(C=10, d=64, noise=0.001)
    dcr = DCR.build({"l4": 64}, k=32, num_classes=10)
    assert dcr.layers["l4"].rank == 9
    assert float(dcr.layer_loss("l4", feats)) < 1e-3
    x = torch.randn(128, 64, requires_grad=True)
    loss, per = dcr({"l4": x}, {"l4": 0.5})
    loss.backward()
    assert x.grad is not None and float(x.grad.norm()) > 0
    assert per["l4"] > 0
    # zero lambda -> no loss, no graph
    loss0, per0 = dcr({"l4": x}, {"l4": 0.0})
    assert float(loss0) == 0.0 and math.isnan(per0["l4"])


def test_macc_lite_directions_and_clipping():
    lite = MACCLite(["a", "b"], {"a": 1.0, "b": 1.0}, lambda_initial=0.1, lambda_max=0.2, step_size=1.0)
    act = lite.step({"a": 4.0, "b": 0.25})
    assert act.lambdas["a"] == 0.2  # kappa above target -> increase, clipped
    assert act.lambdas["b"] == 0.0  # kappa below target -> decrease, clipped at 0
    absl = MACCLite(["a"], {"a": 16.0}, lambda_initial=0.0, step_size=0.1, error_mode="absolute_log_ratio", deadband=0.1)
    assert absl.step({"a": 64.0}).lambdas["a"] > 0
    assert absl.step({"a": 16.0}).lambdas["a"] < absl.lambdas["a"] + 1e-9


def test_full_macc_learns_and_roundtrips():
    fm = FullMACC(["a", "b"], {"a": 1.0, "b": 0.5}, state_dim=FullMACC.state_dim_for(2, 1), total_controller_steps=20)
    before = {k: v.clone() for k, v in fm.policy.state_dict().items()}
    for i in range(6):
        fm.observe_and_act({"a": 2.0, "b": 0.7}, {"a": 0.1, "b": 0.0}, i / 6, 0.05, extra_state=[0.1], step_index=i)
    changed = any(not torch.equal(before[k], v) for k, v in fm.policy.state_dict().items() if v.dtype.is_floating_point)
    assert changed
    sd = fm.state_dict()
    fm2 = FullMACC(["a", "b"], {"a": 1.0, "b": 0.5}, state_dim=FullMACC.state_dim_for(2, 1), total_controller_steps=20)
    fm2.load_state_dict(sd)
    assert fm2.controller_step == fm.controller_step and fm2.lambdas == fm.lambdas


def test_build_controller_kinds():
    for kind in ("none", "fixed_dcr", "macc_lite", "full_macc"):
        c = build_controller(kind, ["l"], {"l": 1.0}, {"macc_lite": {}, "full_macc": {}}, 10, torch.device("cpu"), 0, 1, 0.03)
        assert hasattr(c, "lambdas")


def test_kappa_target_from_strength():
    assert abs(kappa_target_from_strength(0.5) - 1.0) < 1e-12
    assert abs(kappa_target_from_strength(0.8) - 0.25) < 1e-12


def test_class_conditional_monitor_tracks_collapse():
    feats, labels, _ = make_etf_features(noise=0.05)
    mon = ClassConditionalMonitor({"l": 64}, 10, ema_decay=0.8)
    for _ in range(30):
        idx = torch.randperm(feats.shape[0])[:64]
        mon.update({"l": feats[idx]}, labels[idx])
    exact = float(scatter_traces(feats, labels, 10)["kappa"])
    assert abs(mon.kappa()["l"] - exact) < 0.5 * exact + 0.02
    sd = mon.state_dict()
    mon2 = ClassConditionalMonitor({"l": 64}, 10, ema_decay=0.8)
    mon2.load_state_dict(sd)
    assert mon2.kappa() == mon.kappa()
    d = mon.diag_mahalanobis("l", feats[:4], labels[:4])
    assert torch.isfinite(d).all()


def test_spectral_monitor_effective_rank():
    sm = SpectralMonitor({"h": 32}, ema_decay=0.5)
    low = torch.randn(200, 4) @ torch.randn(4, 32)
    for _ in range(5):
        sm.update({"h": low})
    assert sm.kappa()["h"] < 5.0
    iso = torch.randn(400, 32)
    sm2 = SpectralMonitor({"h": 32}, ema_decay=0.5)
    sm2.update({"h": iso})
    assert sm2.kappa()["h"] > 20.0


def test_statistics():
    a = np.array([51.2, 50.8, 51.9, 51.1, 50.5])
    b = a - np.array([1.0, 0.8, 1.2, 0.9, 1.1])
    res = compare_paired(a, b, n_boot=500, n_perm=500)
    assert res["boot_ci_low"] > 0 and res["p_ttest"] < 0.05 and res["p_perm"] == 0.0625 and res["perm_exact"]
    assert holm_bonferroni([0.01, 0.04, 0.03])[0] == pytest.approx(0.03)
    same = paired_permutation_test([1, 2, 3], [1, 2, 3])
    assert same["p_value"] == 1.0
    m = mcnemar_test(np.array([1, 1, 0, 0] * 25, bool), np.array([1, 0, 0, 1] * 25, bool))
    assert m["n_discordant"] == 50 and m["p_value"] == 1.0
    s = summarize([1, 2, 3, 4, 5])
    assert s["n"] == 5 and s["ci_low"] < 3 < s["ci_high"]


def test_control_loop_uses_task_error_mode_even_if_protocol_block_present():
    from caenl.core.control_loop import ControlLoop
    from caenl.utils.io import load_yaml
    from pathlib import Path

    proto = load_yaml(Path(__file__).resolve().parents[1] / "configs/protocol/revision_protocol.yaml")
    assert proto["macc_lite"]["error_mode"] == "log_ratio"  # vision default
    loop = ControlLoop({"h": 64}, "macc_lite", 0.03, proto, torch.device("cpu"), 0, 100, k=32, configured_rank=16, target_effective_rank=16, cadence=1, error_mode="absolute_log_ratio")
    assert loop.controller.error_mode == "absolute_log_ratio"
    # effective rank far below the target must *increase* lambda (rank target, not collapse target)
    low_rank = torch.randn(64, 2) @ torch.randn(2, 64)
    lam0 = loop.lambdas["h"]
    loop.update({"h": low_rank})
    assert loop.lambdas["h"] > lam0


def test_full_macc_explored_actions_do_not_update_policy():
    fm = FullMACC(["a"], {"a": 1.0}, state_dim=FullMACC.state_dim_for(1, 1), total_controller_steps=10, epsilon_start=1.0, epsilon_end=1.0)
    before = {k: v.clone() for k, v in fm.policy.state_dict().items()}
    for i in range(5):
        fm.observe_and_act({"a": 2.0}, {"a": 0.0}, i / 5, 0.1, extra_state=[0.0], step_index=i)
    assert all(torch.equal(before[k], v) for k, v in fm.policy.state_dict().items())


def test_full_macc_acquisition_credit_assignment_and_persistence():
    fm = FullMACC(["a", "b"], {"a": 1.0, "b": 0.5}, state_dim=FullMACC.state_dim_for(2, 1), total_controller_steps=50, epsilon_start=0.0, epsilon_end=0.0, reward_query_beta=0.01)
    # a few in-phase controller steps, then an acquisition governed by the last action's tau
    for i in range(3):
        fm.observe_and_act({"a": 2.0, "b": 0.7}, {"a": 0.1, "b": 0.0}, i / 10, 0.02, extra_state=[0.1], step_index=i)
    fm.mark_acquisition(value_before=-1.0, queries=500, step_index=3)
    assert fm.has_pending_acquisition()
    fm.new_phase()  # phase boundary drops the regularisation action but keeps the acquisition action
    assert fm._pending is None and fm.has_pending_acquisition()
    # survive a checkpoint/restore
    sd = fm.state_dict()
    fm2 = FullMACC(["a", "b"], {"a": 1.0, "b": 0.5}, state_dim=FullMACC.state_dim_for(2, 1), total_controller_steps=50, epsilon_start=0.0, epsilon_end=0.0, reward_query_beta=0.01)
    fm2.load_state_dict(sd)
    assert fm2.has_pending_acquisition()
    q_before = fm2.policy.quantile_head.weight.detach().clone()
    d_before = fm2.policy.delta_head.weight.detach().clone()
    info = fm2.reward_acquisition(value_after=-0.6, step_index=10)
    assert info["kind"] == "acquisition_update" and info["queries"] == 500
    assert abs(info["value_delta"] - 0.4) < 1e-9 and abs(info["reward"] - (10.0 * 0.4 - 0.01 * 500)) < 1e-9
    assert fm2.policy.quantile_head.weight.grad is not None and float(fm2.policy.quantile_head.weight.grad.abs().sum()) > 0
    assert fm2.policy.delta_head.weight.grad is None or float(fm2.policy.delta_head.weight.grad.abs().sum()) == 0  # only the quantile head is credited
    assert not fm2.has_pending_acquisition()
    assert any(h.get("kind") == "acquisition_update" for h in fm2.history)
    # the first acquisition reward has zero advantage (baseline initialised to the reward); the quantile head still
    # receives the entropy-gradient; a second acquisition with a larger gain must move the quantile weights
    fm2.observe_and_act({"a": 2.0, "b": 0.7}, {"a": 0.1, "b": 0.0}, 0.5, 0.02, extra_state=[0.1], step_index=11)
    fm2.mark_acquisition(value_before=-0.6, queries=500, step_index=12)
    fm2.new_phase()
    fm2.reward_acquisition(value_after=0.4, step_index=20)
    assert not torch.equal(q_before, fm2.policy.quantile_head.weight.detach())


def test_full_macc_quantile_head_disabled_for_non_acquisition_tasks():
    fm = FullMACC(["h"], {"h": 16.0}, state_dim=FullMACC.state_dim_for(1, 1), total_controller_steps=20, quantile_head_enabled=False, fixed_threshold_quantile=0.9, epsilon_start=0.0, epsilon_end=0.0)
    w = fm.policy.quantile_head.weight.detach().clone()
    for i in range(4):
        act, _ = fm.observe_and_act({"h": 20.0}, {"h": 0.0}, i / 4, 0.05, extra_state=[0.0], step_index=i)
        assert act.threshold_quantile == 0.9
    fm.mark_acquisition(-1.0, 10)
    assert not fm.has_pending_acquisition()
    assert torch.equal(w, fm.policy.quantile_head.weight.detach())  # head excluded from every update
