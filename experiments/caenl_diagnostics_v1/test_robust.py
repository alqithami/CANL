"""CPU contract checks, not validation of GPU AutoAttack efficacy."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import robust


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False))


def atomic_npz(path, **arrays):
    with Path(path).open('wb') as stream:
        np.savez_compressed(stream, **arrays)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[2., 1.], [-1., 2.], [.2, -.5]]))

    def forward_features(self, x):
        flat = x.reshape(len(x), 2)
        return self.linear(flat), {'layer3': flat, 'layer4': torch.cat((flat, flat.square()), 1)}

    def forward(self, x):
        return self.forward_features(x)[0]


class FakeAttack:
    calls = 0
    seeds = []

    def __init__(self, model, **kwargs):
        self.eps = kwargs['eps']
        self.attacks_to_run = list(robust.STANDARD_ATTACKS)
        type(self).seeds.append(kwargs['seed'])
        if kwargs['version'] != 'standard' or kwargs['norm'] != 'Linf':
            raise ValueError('Wrong constructor')

    def run_standard_evaluation(self, x, y, bs):
        type(self).calls += 1
        return (x + self.eps / 2).clamp(0, 1)


class RobustContracts(unittest.TestCase):
    def test_equal_class_geometry_matches_direct_unbalanced_definition(self):
        z = torch.tensor([[1., 0., 0.], [2., 1., 0.], [0., 1., 0.],
                          [0., 2., 1.], [1., 2., 1.], [1., 1., 2.], [2., 0., 2.]], dtype=torch.float64)
        y = np.array([0, 0, 1, 1, 1, 2, 2])
        whole = robust.class_moments(z, y, 3)
        stream = robust.merge_moments([robust.class_moments(z[:3], y[:3], 3),
                                      robust.class_moments(z[3:], y[3:], 3)])
        for key in whole:
            np.testing.assert_allclose(whole[key], stream[key], atol=1e-14)
        actual = robust.scatter_from_moments(stream)
        h = torch.nn.functional.normalize(z, dim=1)
        means = torch.stack([h[y == c].mean(0) for c in range(3)])
        within = torch.stack([(h[y == c] - means[c]).square().sum(1).mean() for c in range(3)]).mean()
        between = (means - means.mean(0)).square().sum(1).mean()
        self.assertAlmostEqual(actual['within'], float(within), places=13)
        self.assertAlmostEqual(actual['between'], float(between), places=13)
        self.assertAlmostEqual(actual['total'], float(within + between), places=13)
        scaled = robust.scatter_from_moments(robust.class_moments(19 * z, y, 3))
        self.assertAlmostEqual(actual['log_kappa'], scaled['log_kappa'], places=13)

    def test_moments_reject_missing_classes_and_nonfinite(self):
        with self.assertRaises(ValueError):
            robust.scatter_from_moments(robust.class_moments(torch.ones(4, 2), [0, 0, 1, 1], 3))
        with self.assertRaises(ValueError):
            robust.class_moments(torch.tensor([[float('nan'), 1.]]), [0], 2)

    def test_sparse_scalar_moments_preserve_exact_trace_geometry(self):
        z = torch.tensor([[1., 2.], [2., 3.], [4., 2.], [2., 1.]], dtype=torch.float64)
        labels = np.array([1, 1, 3, 3])
        features = {'layer3': z, 'layer4': torch.cat((z, z.square()), 1)}
        encoded = robust._moment_arrays(features, labels, 5, 1e-6, 'clean')
        for layer in robust.LAYERS:
            self.assertEqual(encoded[f'clean_{layer}_class_sums'].shape[0], 2)
            self.assertEqual(encoded[f'clean_{layer}_class_sumsq'].shape, (2,))
            decoded = robust._decode_moments(encoded, 'clean', layer)
            direct = robust.class_moments(features[layer], labels, 5)
            for key in direct:
                np.testing.assert_array_equal(decoded[key], direct[key])
        encoded['clean_layer3_class_ids'] = np.array([0, 3])
        with self.assertRaises(ValueError):
            robust._decode_moments(encoded, 'clean', 'layer3')

    def test_attack_outcome_groups_use_correct_masks_and_empty_groups_are_defined(self):
        arrays = {'labels': np.array([0, 1, 0, 1]), 'clean_predictions': np.array([0, 1, 1, 1]),
                  'adversarial_predictions': np.array([0, 0, 1, 0]),
                  'clean_margin': np.array([2., 3., -1., 4.]),
                  'adversarial_margin': np.array([1., -2., -1., -.5]),
                  'layer3_unit_feature_movement_l2': np.array([.1, .2, 0., .4]),
                  'layer4_unit_feature_movement_l2': np.array([.2, .3, 0., .5])}
        groups = robust.descriptive_attack_groups(arrays)
        self.assertEqual(groups['initially_clean_wrong']['n'], 1)
        self.assertEqual(groups['clean_correct_attack_success']['n'], 2)
        self.assertEqual(groups['clean_correct_survivor']['n'], 1)
        self.assertAlmostEqual(groups['clean_correct_attack_success']['feature_movement']['layer3']['mean_l2'], .3)
        arrays['adversarial_predictions'] = np.array([1, 0, 1, 0])
        empty = robust.descriptive_attack_groups(arrays)['clean_correct_survivor']
        self.assertEqual(empty['n'], 0)
        self.assertIsNone(empty['mean_adversarial_margin'])

    def test_margin_gradient_proxy_has_exact_linear_value_and_no_parameter_grad(self):
        model = Tiny()
        x = torch.tensor([[[[.4, .2]]], [[[.1, .5]]]])
        y = torch.tensor([0, 1])
        result = robust.margin_gradient_proxy(model, x, y)
        # The strongest competing class is class 1 for image 0 and class 0 for image 1.
        np.testing.assert_allclose(result['margin_gradient_l1'], [4., 4.], atol=1e-6)
        np.testing.assert_allclose(result['clean_margin'], [.99999994, .19999999], atol=1e-6)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        np.testing.assert_array_equal(robust.sensitivity_ids([2, 1, 2, 0, 1, 0, 1]), [0, 1, 2, 3, 4, 5])

    def test_suite_preserves_declared_contract_and_rejects_bad_attacks(self):
        model = Tiny()
        x = torch.tensor([[[[.4, .2]]], [[[.1, .5]]]])
        y = torch.tensor([0, 1])
        adv, norms = robust.suite_attack(model, x, y, 2 / 255, 918, 2, 'cpu', '/tmp/fake.log', FakeAttack)
        self.assertEqual(adv.dtype, torch.float32)
        self.assertTrue(torch.all(norms <= 2 / 255 + 1e-6))
        class WrongSuite(FakeAttack):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.attacks_to_run = ['apgd-ce']
        with self.assertRaises(RuntimeError):
            robust.suite_attack(model, x, y, 2 / 255, 918, 2, 'cpu', '/tmp/fake.log', WrongSuite)
        with self.assertRaises(ValueError):
            robust.check_threat_model(x, x + .1, 2 / 255)
        with self.assertRaises(ValueError):
            robust.check_threat_model(x.half(), adv.half(), 2 / 255)

    def test_verified_resume_rejects_wrong_binding_and_modified_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'chunk.npz'
            done = path.with_suffix('.json')
            atomic_npz(path, indices=np.arange(2))
            atomic_json(done, {'binding': 'b', 'sha256': sha(path)})
            cached = robust.load_verified_chunk(path, done, 'b', sha)
            np.testing.assert_array_equal(cached[0]['indices'], [0, 1])
            with self.assertRaises(ValueError):
                robust.load_verified_chunk(path, done, 'c', sha)
            atomic_npz(path, indices=np.arange(3))
            with self.assertRaises(ValueError):
                robust.load_verified_chunk(path, done, 'b', sha)

    def test_clean_reference_requires_predictions_logits_and_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'validation.npz'
            logits = np.array([[2., 1.], [.1, 3.]], dtype=np.float32)
            atomic_npz(path, indices=np.arange(2), labels=np.array([0, 1]), logits=logits,
                       predictions=logits.argmax(1))
            clean = {'indices': np.arange(2), 'labels': np.array([0, 1]),
                     'clean_logits': logits.copy(), 'clean_predictions': logits.argmax(1)}
            self.assertEqual(robust.check_clean_reference(clean, path)['status'], 'PASS')
            clean['clean_logits'][0, 1] += .01
            with self.assertRaises(ValueError):
                robust.check_clean_reference(clean, path)
            clean['clean_logits'] = logits.copy()
            clean['labels'] = np.array([1, 0])
            with self.assertRaises(ValueError):
                robust.check_clean_reference(clean, path)

    def test_same_clean_correct_intersection_uses_common_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = np.array([0, 1, 0, 1])
            clean_predictions = {'entropy': [0, 1, 1, 1], 'tuned_fixed': [0, 1, 0, 0],
                                 'lite': [0, 1, 0, 1]}
            adv_predictions = {'entropy': [0, 0, 0, 1], 'tuned_fixed': [0, 1, 0, 1],
                               'lite': [1, 1, 0, 1]}
            for method in robust.METHODS:
                out = root / 'seed62001' / method
                out.mkdir(parents=True)
                atomic_npz(out / 'clean.npz', labels=labels, clean_predictions=np.array(clean_predictions[method]))
                radius = out / 'linf_2p0_255'
                radius.mkdir()
                atomic_npz(radius / 'predictions.npz', labels=labels,
                           adversarial_predictions=np.array(adv_predictions[method]))
            rows = robust.common_clean_intersection(root, [62001], robust.METHODS, [2.])
            self.assertEqual(rows[0]['common_clean_correct_n'], 2)
            self.assertEqual(rows[0]['survival']['entropy']['accuracy'], .5)
            self.assertEqual(rows[0]['survival']['tuned_fixed']['accuracy'], 1.)
            self.assertEqual(rows[0]['survival']['lite']['accuracy'], .5)

    def test_chunked_model_diagnostic_resumes_without_repeating_suites(self):
        """Integration uses a synthetic attacker; it checks persistence only."""
        model = Tiny().eval()
        images = torch.tensor([[[[.2, .1]]], [[[.3, .4]]], [[[.1, .6]]],
                               [[[.2, .7]]], [[[.8, .1]]], [[[.7, .2]]]])
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        cat = types.SimpleNamespace(images=images, val_labels=labels.numpy())
        support = types.ModuleType('support')
        support.evaluation_batches = lambda ids, bs: [list(ids[j:j + bs]) for j in range(0, len(ids), bs)]
        support.loader = lambda cat, split, batches, seed, workers, augmented: ((cat.images[b], labels[b], torch.tensor(b)) for b in batches)
        support.to_device = lambda raw, device: raw
        support.seed_for = lambda *values: int(digest(values)[:8], 16) % (2**31 - 1)
        support.sync = lambda device: None
        autoattack = types.ModuleType('autoattack')
        autoattack.AutoAttack = FakeAttack
        FakeAttack.calls, FakeAttack.seeds = 0, []
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'support': support, 'autoattack': autoattack}):
            out = Path(tmp) / 'diagnostic'
            out.mkdir()
            reference = Path(tmp) / 'validation.npz'
            logits = model(images).detach().numpy()
            atomic_npz(reference, indices=np.arange(6), labels=labels.numpy(), logits=logits, predictions=logits.argmax(1))
            ctx = types.SimpleNamespace(cat=cat, device='cpu', binding='frozen', progress=lambda text: None,
                    cfg={'num_classes': 3, 'alignment': {'feature_norm_epsilon': 1e-6, 'scatter_epsilon': 1e-6}})
            record = {'seed': 62001, 'method': 'lite', 'sha256': 'ckpt', 'path': Path(tmp) / 'FINAL_MODEL.pt',
                      'validation_path': reference}
            opt = {'chunk': 4, 'attack_batch': 2, 'workers': 0}
            common = (atomic_json, atomic_npz, sha, digest)
            clean, clean_summary = robust._clean_model(ctx, model, record, out, opt, common)
            self.assertEqual(clean_summary['inference_replay']['status'], 'PASS')
            first = robust._radius(ctx, model, record, out, opt, 2., clean, common)
            self.assertEqual(FakeAttack.calls, 2)
            second = robust._radius(ctx, model, record, out, opt, 2., clean, common)
            self.assertEqual(FakeAttack.calls, 2)
            self.assertEqual(first['robust_accuracy'], second['robust_accuracy'])
            self.assertEqual(first['predictions_sha256'], second['predictions_sha256'])
            self.assertLessEqual(first['robust_correct'], first['clean_correct'])
            self.assertEqual(first['n'], 6)
            radius = out / 'linf_2p0_255'
            chunk = radius / 'chunk_00000.npz'
            with chunk.open('ab') as stream:
                stream.write(b'tampered')
            with self.assertRaises(ValueError):
                robust._radius(ctx, model, record, out, opt, 2., clean, common)

    def test_complete_frozen_matrix_and_statistics_integration(self):
        """All identities/endpoints with synthetic data/attacks, never a scientific result."""
        model = Tiny().eval()
        images = torch.tensor([[[[.2, .1]]], [[[.3, .4]]], [[[.1, .6]]],
                               [[[.2, .7]]], [[[.8, .1]]], [[[.7, .2]]]])
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        support = types.ModuleType('support')
        support.evaluation_batches = lambda ids, bs: [list(ids[j:j + bs]) for j in range(0, len(ids), bs)]
        support.loader = lambda cat, split, batches, seed, workers, augmented: ((images[b], labels[b], torch.tensor(b)) for b in batches)
        support.to_device = lambda raw, device: raw
        support.seed_for = lambda *values: int(digest(values)[:8], 16) % (2**31 - 1)
        support.sync = lambda device: None
        autoattack = types.ModuleType('autoattack')
        autoattack.AutoAttack = FakeAttack
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'support': support, 'autoattack': autoattack}):
            base = Path(tmp)
            reference = base/'validation.npz'
            logits = model(images).detach().numpy()
            atomic_npz(reference, indices=np.arange(6), labels=labels.numpy(), logits=logits, predictions=logits.argmax(1))
            checkpoints = []
            for seed in robust.SEEDS:
                for method in robust.METHODS:
                    p = base/f'{seed}_{method}.pt'
                    p.write_bytes(b'test fixture identity only')
                    checkpoints.append(dict(stage='confirmation', seed=seed, method=method, path=p,
                                            sha256=sha(p), validation_path=reference))
            ctx = types.SimpleNamespace(cat=types.SimpleNamespace(val_labels=labels.numpy()),
                    root=base/'results', device='cpu', binding='synthetic-matrix-contract',
                    progress=lambda message: None, records=checkpoints,
                    load_model=lambda record: Tiny().eval(),
                    protocol={'robustness':{'radii':[.5,1,2,4,8], 'primary':2,
                              'attack_batch':2, 'chunk':4, 'workers':0}},
                    cfg={'num_classes':3, 'alignment':{'feature_norm_epsilon':1e-6, 'scatter_epsilon':1e-6}})
            result = robust.run(ctx)
            self.assertEqual(result['models'], 30)
            self.assertEqual(result['standard_suites'], 150)
            self.assertEqual(len(result['primary_contrasts']), 3)
            self.assertEqual(len(result['same_clean_correct_intersection']), 50)
            for contrast in result['primary_contrasts']:
                self.assertEqual(contrast['n_pairs'], 10)
                self.assertEqual(contrast['paired_t_p_holm'], 1.)
                self.assertEqual(contrast['exact_signflip_p_holm'], 1.)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
