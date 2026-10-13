"""Small CPU contract tests; set PYTHONPATH to the original engine directory.

No ImageNet, CUDA, torchvision or scientific training run is required. When the
runtime's torchvision is unavailable, the test supplies only the original
support interfaces used by Control and the tiny-model diagnostic.
"""
from __future__ import annotations

import copy
import hashlib
import math
import sys
import types
import unittest
from contextlib import nullcontext

import torch
import torch.nn.functional as F


def _install_tiny_support():
    # The engine imports torchvision at support import; isolate the CPU tests
    # from that unrelated optional dependency without altering any source file.
    support = types.ModuleType('support')
    support.seed_for = lambda *args: int(hashlib.sha256('|'.join(map(str, args)).encode()).hexdigest()[:8], 16) % (2**31-1)
    support.loader = lambda *args, **kwargs: None
    support.BalancedBatches = object
    support.to_device = lambda batch, device: (batch[0].to(device).float().div(255), batch[1].to(device), batch[2])
    support.sync = lambda device: None
    support.amp = lambda device: nullcontext()
    def train_mode(model, initial):
        model.train()
        if not initial:
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
    support.train_mode = train_mode
    sys.modules['support'] = support


_install_tiny_support()
from aligned import aligned_loss
from control import Control
from cost import (CONDITIONS, aligned_tensors, condition_order, execute,
                  monitoring_records, _make_control, _summary)


CFG = dict(layers=['layer3', 'layer4'], lambda_fixed=.03, lambda_initial=.03,
    lambda_max=.1, lite_smoothing=.2, training_ema_beta=.05,
    control_interval=2, clip_grad_norm=5., eval_batch=4,
    alignment=dict(scatter_epsilon=1e-6, feature_norm_epsilon=1e-6,
                   huber_delta=.25, floor_weight=.01))
TARGETS = {layer: dict(log_kappa_reference=.5, log_kappa_target=.3,
            between_reference=.3, between_floor=.15) for layer in CFG['layers']}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = torch.nn.Linear(12, 8)
        self.bn = torch.nn.BatchNorm1d(8)
        self.head = torch.nn.Linear(8, 2)
    def forward_features(self, x):
        a = self.bn(self.feature(x.reshape(len(x), -1)))
        b = torch.tanh(a)
        return self.head(b), {'layer3': a, 'layer4': b}
    def forward(self, x):
        return self.forward_features(x)[0]


class CostContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_streamlined_matches_original_scalar_and_gradients(self):
        torch.manual_seed(1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        weights = {layer: .03 for layer in CFG['layers']}
        originals = {layer: torch.randn(6, 9, dtype=torch.float64, requires_grad=True)
                     for layer in CFG['layers']}
        simplified = {key: value.detach().clone().requires_grad_() for key, value in originals.items()}
        original, _ = aligned_loss(originals, labels, TARGETS, weights, CFG['alignment'])
        slim, stats = aligned_tensors(simplified, labels, TARGETS, weights, CFG['alignment'], False)
        torch.testing.assert_close(original, slim, rtol=0, atol=0)
        original.backward()
        slim.backward()
        for layer in originals:
            torch.testing.assert_close(originals[layer].grad, simplified[layer].grad, rtol=0, atol=0)
            self.assertNotIn('total', stats[layer])
            self.assertNotIn('kappa', stats[layer])

    def test_monitoring_omission_leaves_fixed_updates_identical(self):
        torch.manual_seed(2)
        initial = TinyModel()
        batch = (torch.randint(0, 256, (8, 3, 2, 2), dtype=torch.uint8),
                 torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]), torch.arange(8))
        finals = []
        for condition in ('streamlined_fixed', 'instrumented_fixed'):
            model = copy.deepcopy(initial)
            opt = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
            control = _make_control(condition, TARGETS, CFG, 'cpu', 2)
            result = execute(model, opt, control, condition, [batch], batch[:2],
                             TARGETS, CFG, 'cpu', 4)
            finals.append(model.state_dict())
            self.assertEqual(result['controller_windows'], 0 if control is None else 2)
            self.assertGreaterEqual(result['elapsed_seconds'], result['train_compute_seconds'])
        for key in finals[0]:
            torch.testing.assert_close(finals[0][key], finals[1][key], rtol=0, atol=0)

    def test_lite_retains_original_updates_with_same_training(self):
        torch.manual_seed(3)
        initial = TinyModel()
        batches = [(torch.randint(0, 256, (8, 3, 2, 2), dtype=torch.uint8),
                    torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]), torch.arange(8))
                   for _ in range(2)]
        states = []
        for condition in ('streamlined_lite', 'instrumented_lite'):
            model = copy.deepcopy(initial)
            opt = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
            control = _make_control(condition, TARGETS, CFG, 'cpu', 2)
            result = execute(model, opt, control, condition, batches, batches[0][:2],
                             TARGETS, CFG, 'cpu', 4)
            self.assertEqual(result['feedback_evaluations'], 3)
            states.append((model.state_dict(), control.state_dict()))
        for key in states[0][0]:
            torch.testing.assert_close(states[0][0][key], states[1][0][key], rtol=0, atol=0)
        self.assertEqual(states[0][1], states[1][1])

    def test_original_lite_formula_does_not_depend_on_ce(self):
        one = Control('aligned_lite', TARGETS, 62001, CFG, 'cpu', 2)
        two = Control('aligned_lite', TARGETS, 62001, CFG, 'cpu', 2)
        one.begin_phase(1., 0)
        two.begin_phase(9., 0)
        records = {layer: {'log_kappa': 1.1, 'between': .4} for layer in CFG['layers']}
        one.observe(records)
        two.observe(records)
        one.window(.9, 1, 2, 2, terminal=False)
        two.window(8.9, 1, 2, 2, terminal=False)
        self.assertEqual(one.weights, two.weights)
        self.assertEqual(one.ema, two.ema)
        self.assertNotEqual(one.prev_ce, two.prev_ce)

    def test_entropy_streamlined_has_no_geometry_or_controller(self):
        torch.manual_seed(4)
        model = TinyModel()
        batch = (torch.randint(0, 256, (8, 3, 2, 2), dtype=torch.uint8),
                 torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]), torch.arange(8))
        opt = torch.optim.SGD(model.parameters(), lr=.01)
        result = execute(model, opt, None, 'streamlined_entropy', [batch], batch[:2],
                         TARGETS, CFG, 'cpu', 2)
        self.assertEqual(result['feedback_evaluations'], 0)
        self.assertEqual(result['controller_state_json_bytes'], 0)
        self.assertEqual(result['controller_state'], None)

    def test_order_is_fixed_and_complete(self):
        for repetition in range(5):
            self.assertEqual(set(condition_order(repetition)), set(CONDITIONS))
            self.assertEqual(condition_order(repetition), condition_order(repetition))
        self.assertEqual(condition_order(0), list(reversed(condition_order(1))))

    def test_descriptive_pairs_are_by_repetition(self):
        rows = []
        for repetition in range(5):
            for index, condition in enumerate(CONDITIONS):
                rows.append(dict(condition=condition, repetition=repetition,
                    elapsed_seconds=10+index+repetition, train_compute_seconds=8.,
                    monitoring_seconds=1., transfer_seconds=.5,
                    peak_allocated_bytes=100, peak_reserved_bytes=200,
                    controller_state_json_bytes=10))
        summary = _summary(list(reversed(rows)))
        pair = summary['paired_descriptive_contrasts'][0]
        self.assertEqual(pair['paired_seconds_difference'], [1.] * 5)
        self.assertEqual(summary['hypothesis_tests'],
            'None: repeated timings on one checkpoint are descriptive')


if __name__ == '__main__':
    unittest.main()
