"""Frozen-checkpoint robustness and representation diagnostics.

This module does no training and writes exclusively under the diagnostic root.
It evaluates standard AutoAttack at five predeclared pixel radii on all cleaned
validation images. Clean and perturbed geometry are descriptive measurements;
input-margin gradients are local sensitivity proxies, not certificates.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


STANDARD_ATTACKS = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
LAYERS = ('layer3', 'layer4')
METHODS = ('entropy', 'tuned_fixed', 'lite')
SEEDS = tuple(range(62001, 62011))


def true_margins(logits, labels):
    """True-class logit minus the largest competing logit, for every image."""
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError('Expected logits with at least two classes')
    labels = labels.to(device=logits.device, dtype=torch.long)
    wrong = logits.clone()
    wrong.scatter_(1, labels[:, None], float('-inf'))
    return logits.gather(1, labels[:, None]).squeeze(1) - wrong.max(1).values


def class_moments(features, labels, classes, norm_eps=1e-6):
    """Compact float64 sufficient statistics of per-image unit features."""
    h = F.normalize(features.detach().double(), dim=1, eps=float(norm_eps)).cpu().numpy()
    y = np.asarray(labels, dtype=np.int64)
    if h.ndim != 2 or h.shape[0] != len(y) or not np.isfinite(h).all():
        raise ValueError('Invalid feature payload')
    if not len(y) or np.min(y) < 0 or np.max(y) >= classes:
        raise ValueError('Invalid class labels')
    counts = np.bincount(y, minlength=classes).astype(np.int64)
    sums = np.zeros((classes, h.shape[1]), dtype=np.float64)
    # Trace scatter requires only sum ||h||^2 per class, rather than a full
    # coordinate-wise squared-sum vector. Keep sums in float64 for the means.
    sumsq = np.zeros(classes, dtype=np.float64)
    # CPU float64 accumulation avoids device-dependent scatter-add ordering.
    for c in np.unique(y):
        hc = h[y == c]
        sums[c] = hc.sum(0)
        sumsq[c] = np.square(hc).sum()
    return {'counts': counts, 'sums': sums, 'sumsq': sumsq}


def merge_moments(blocks):
    if not blocks:
        raise ValueError('No moments to combine')
    return {key: np.sum([b[key] for b in blocks], axis=0) for key in ('counts', 'sums', 'sumsq')}


def scatter_from_moments(moments, eps=1e-6):
    """Equal-class scatter, including when validation classes differ in size."""
    counts = np.asarray(moments['counts'], dtype=np.int64)
    sums = np.asarray(moments['sums'], dtype=np.float64)
    sumsq = np.asarray(moments['sumsq'], dtype=np.float64)
    if len(counts) < 2 or np.min(counts) < 2:
        raise ValueError('Full geometry requires at least two images per class')
    if sums.ndim != 2 or sums.shape[0] != len(counts) or sumsq.shape != counts.shape:
        raise ValueError('Inconsistent class moments')
    means = sums / counts[:, None]
    per_class = sumsq / counts - np.square(means).sum(1)
    if per_class.min() < -1e-8 or not np.isfinite(per_class).all():
        raise ValueError('Invalid class variance')
    within = float(np.maximum(per_class, 0).mean())
    between = float(np.square(means - means.mean(0)).sum(1).mean())
    log_kappa = math.log(within + eps) - math.log(between + eps)
    return {'within': within, 'between': between, 'total': within + between,
            'log_kappa': log_kappa, 'kappa': math.exp(log_kappa),
            'weighting': 'equal weight for each true-label class',
            'min_class_count': int(counts.min()), 'max_class_count': int(counts.max())}


def sensitivity_ids(labels, per_class=2):
    """First validation indices per true class; no outcome-dependent selection."""
    y = np.asarray(labels, dtype=np.int64)
    return np.sort(np.concatenate([np.flatnonzero(y == c)[:per_class] for c in np.unique(y)]))


def margin_gradient_proxy(model, x, y):
    """Local d(true-class margin)/d(pixel), measured separately per image.

    The margin can change its competing class and its gradient over a finite
    perturbation. Its gradient norm is therefore not a robustness certificate.
    Parameter gradients are never accumulated or written to checkpoints.
    """
    model.eval()
    margins, norms = [], []
    for i in range(len(x)):
        xi = x[i:i + 1].detach().float().requires_grad_(True)
        margin = true_margins(model(xi).float(), y[i:i + 1])[0]
        grad, = torch.autograd.grad(margin, xi, only_inputs=True)
        if not torch.isfinite(grad).all() or not torch.isfinite(margin):
            raise FloatingPointError('Nonfinite local margin gradient')
        margins.append(float(margin.detach().cpu()))
        norms.append(float(grad.detach().abs().sum().cpu()))
    return {'clean_margin': np.asarray(margins, dtype=np.float32),
            'margin_gradient_l1': np.asarray(norms, dtype=np.float32)}


def check_threat_model(x, adversarial, epsilon):
    if x.shape != adversarial.shape or x.dtype != torch.float32 or adversarial.dtype != torch.float32:
        raise ValueError('Attack input/output must be matching full-precision tensors')
    if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError('Clean attack inputs must be in pixel space [0,1]')
    norms = (adversarial - x).abs().flatten(1).amax(1)
    if (not torch.isfinite(adversarial).all() or adversarial.min() < -1e-6
            or adversarial.max() > 1 + 1e-6 or norms.max() > epsilon + 2e-6):
        raise ValueError('Adversarial output violates the Linf pixel threat model')
    return norms


def suite_attack(model, x, y, epsilon, seed, batch_size, device, log_path,
                 attacker_factory=None):
    """Run the unchanged standard suite; factory injection is only for tests."""
    if attacker_factory is None:
        from autoattack import AutoAttack
        attacker_factory = AutoAttack
    check_threat_model(x, x, epsilon)
    model.eval()
    attacker = attacker_factory(model, norm='Linf', eps=float(epsilon), version='standard',
                                device=str(device), seed=int(seed), verbose=False,
                                log_path=str(log_path))
    if list(attacker.attacks_to_run) != STANDARD_ATTACKS:
        raise RuntimeError('Unexpected standard AutoAttack suite')
    # No autocast, custom attacks, reduced iterations, or changes to the suite.
    adv = attacker.run_standard_evaluation(x, y, bs=int(batch_size)).detach()
    norms = check_threat_model(x, adv, epsilon)
    return adv, norms


def load_verified_chunk(path, done, expected_binding, sha):
    """A committed chunk is reusable only when its hash and binding match."""
    import json
    path, done = Path(path), Path(done)
    if not done.exists():
        return None
    record = json.loads(done.read_text())
    if (record.get('binding') != expected_binding or not path.is_file()
            or sha(path) != record.get('sha256')):
        raise ValueError(f'Stale or damaged diagnostic chunk: {path}')
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k].copy() for k in saved.files}, record


def check_clean_reference(arrays, original_path):
    with np.load(original_path, allow_pickle=False) as reference:
        for key in ('indices', 'labels'):
            if not np.array_equal(arrays[key], reference[key]):
                raise ValueError(f'Clean replay {key} differs from completed run')
        old_logits = reference['logits']
        old_pred = reference['predictions'] if 'predictions' in reference.files else old_logits.argmax(1)
        if not np.array_equal(arrays['clean_predictions'], old_pred):
            raise ValueError('Clean replay predictions differ from completed run')
        if not np.allclose(arrays['clean_logits'], old_logits, atol=1e-4, rtol=1e-4):
            raise ValueError('Clean replay logits differ from completed run')
    delta = np.abs(arrays['clean_logits'].astype(np.float64) - old_logits.astype(np.float64))
    return {'status': 'PASS', 'prediction_equality': True, 'atol': 1e-4, 'rtol': 1e-4,
            'max_absolute_logit_difference': float(delta.max())}


def _options(protocol):
    opt = protocol.get('robustness', protocol)
    radii = tuple(float(r) for r in opt['radii'])
    if radii != (0.5, 1., 2., 4., 8.) or float(opt['primary']) != 2.:
        raise ValueError('Diagnostic attack radii/primary endpoint changed')
    return {**opt, 'radii': radii, 'primary': 2.,
            'attack_batch': int(opt['attack_batch']), 'chunk': int(opt['chunk']),
            'workers': int(opt['workers'])}


def _forward(model, x, batch_size, features=True):
    outputs, zs = [], {layer: [] for layer in LAYERS}
    with torch.no_grad():
        for j in range(0, len(x), batch_size):
            if features:
                logits, feat = model.forward_features(x[j:j + batch_size])
                for layer in LAYERS:
                    zs[layer].append(feat[layer].detach().float().cpu())
            else:
                logits = model(x[j:j + batch_size])
            outputs.append(logits.detach().float().cpu())
    result = torch.cat(outputs)
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite evaluation logits')
    return result, {layer: torch.cat(v) for layer, v in zs.items()} if features else {}


def _images(ctx, ids, opt):
    from support import loader, evaluation_batches, to_device
    batches, labels, indices = [], [], []
    for raw in loader(ctx.cat, 'val', evaluation_batches(ids, opt['attack_batch']),
                      0, opt['workers'], False):
        x, y, ix = to_device(raw, ctx.device)
        batches.append(x.detach().float())
        labels.append(y)
        indices.append(np.asarray(ix))
    if not np.array_equal(np.concatenate(indices), ids):
        raise ValueError('Validation loader index mapping differs')
    return torch.cat(batches), torch.cat(labels)


def _moment_arrays(features, labels, classes, norm_eps, prefix):
    result = {}
    for layer in LAYERS:
        m = class_moments(features[layer], labels, classes, norm_eps)
        active = np.flatnonzero(m['counts'])
        result[f'{prefix}_{layer}_class_ids'] = active
        result[f'{prefix}_{layer}_class_counts'] = m['counts']
        result[f'{prefix}_{layer}_class_sums'] = m['sums'][active]
        result[f'{prefix}_{layer}_class_sumsq'] = m['sumsq'][active]
    return result


def _decode_moments(block, prefix, layer):
    """Expand sparse active-class rows; chunks never store empty feature rows."""
    stem = f'{prefix}_{layer}_class_'
    counts = block[stem + 'counts']
    ids = block[stem + 'ids']
    sparse_sums, sparse_sumsq = block[stem + 'sums'], block[stem + 'sumsq']
    if (not np.array_equal(ids, np.flatnonzero(counts)) or sparse_sums.ndim != 2
            or sparse_sums.shape[0] != len(ids) or sparse_sumsq.shape != ids.shape):
        raise ValueError('Invalid sparse class-moment mapping')
    sums = np.zeros((len(counts), sparse_sums.shape[1]), dtype=np.float64)
    sumsq = np.zeros(len(counts), dtype=np.float64)
    sums[ids], sumsq[ids] = sparse_sums, sparse_sumsq
    return {'counts': counts, 'sums': sums, 'sumsq': sumsq}


def _geometry(blocks, prefix, scatter_eps):
    result, sufficient = {}, {}
    for layer in LAYERS:
        m = merge_moments([_decode_moments(block, prefix, layer) for block in blocks])
        result[layer] = scatter_from_moments(m, scatter_eps)
        sufficient.update({f'{prefix}_{layer}_class_{key}': value for key, value in m.items()})
    result['mean_log_kappa'] = float(np.mean([result[l]['log_kappa'] for l in LAYERS]))
    return result, sufficient


def descriptive_attack_groups(arrays):
    """Describe the heterogeneous standard-suite output without mediation claims.

    AutoAttack's returned tensor retains initially incorrect images and keeps
    successful perturbations from different component attacks. Therefore full
    returned-output geometry is a mixture, not a uniform perturbation treatment.
    Group labels use attack outcomes and are descriptive, not causal strata.
    """
    clean_correct = arrays['clean_predictions'] == arrays['labels']
    adv_correct = arrays['adversarial_predictions'] == arrays['labels']
    groups = {'initially_clean_wrong': ~clean_correct,
              'clean_correct_attack_success': clean_correct & ~adv_correct,
              'clean_correct_survivor': clean_correct & adv_correct}
    result = {}
    for name, selected in groups.items():
        n = int(selected.sum())
        result[name] = {'n': n, 'fraction_of_validation': n / len(selected),
                        'mean_clean_margin': float(arrays['clean_margin'][selected].mean()) if n else None,
                        'mean_adversarial_margin': float(arrays['adversarial_margin'][selected].mean()) if n else None,
                        'feature_movement': {layer: {
                            'mean_l2': float(arrays[f'{layer}_unit_feature_movement_l2'][selected].mean()) if n else None,
                            'median_l2': float(np.median(arrays[f'{layer}_unit_feature_movement_l2'][selected])) if n else None}
                            for layer in LAYERS}}
    return result


def _ordered(blocks, keys, labels):
    result = {k: np.concatenate([b[k] for b in blocks], axis=0) for k in keys}
    if (not np.array_equal(result['indices'], np.arange(len(labels)))
            or not np.array_equal(result['labels'], labels)):
        raise ValueError('Diagnostic full validation mapping differs')
    return result


def common_clean_intersection(root, seeds, methods, radii):
    """Secondary survival on the same clean-correct images within each seed.

    This removes changes in the clean-correct denominator when comparing the
    three methods. It remains descriptive and does not identify a mechanism.
    Seeds, rather than individual validation images, are the inference units.
    """
    records = []
    for seed in seeds:
        clean_masks, labels = [], None
        for method in methods:
            with np.load(Path(root) / f'seed{seed}' / method / 'clean.npz', allow_pickle=False) as clean:
                if labels is None:
                    labels = clean['labels'].copy()
                elif not np.array_equal(labels, clean['labels']):
                    raise ValueError('Intersection clean validation labels differ')
                clean_masks.append(clean['clean_predictions'] == labels)
        intersection = np.logical_and.reduce(clean_masks)
        count = int(intersection.sum())
        for pixels in radii:
            survival = {}
            for method in methods:
                filename = Path(root) / f'seed{seed}' / method / f"linf_{str(pixels).replace('.', 'p')}_255" / 'predictions.npz'
                with np.load(filename, allow_pickle=False) as adv:
                    if not np.array_equal(adv['labels'], labels):
                        raise ValueError('Intersection attack labels differ')
                    correct = (adv['adversarial_predictions'] == labels) & intersection
                    survival[method] = {'correct': int(correct.sum()),
                                        'accuracy': float(correct.sum() / count) if count else None}
            records.append({'seed': seed, 'epsilon_pixels': pixels, 'common_clean_correct_n': count,
                            'survival': survival})
    return records


def _clean_model(ctx, model, record, out, opt, common):
    atomic_json, atomic_npz, sha, digest = common
    labels = np.asarray(ctx.cat.val_labels, dtype=np.int64)
    n, blocks, receipts = len(labels), [], []
    norm_eps = float(ctx.cfg['alignment']['feature_norm_epsilon'])
    for start in range(0, n, opt['chunk']):
        ids = np.arange(start, min(n, start + opt['chunk']), dtype=np.int64)
        path = out / f'clean_{start:05d}.npz'
        done = path.with_suffix('.json')
        binding = digest({'diagnostic': ctx.binding, 'checkpoint': record['sha256'],
                          'kind': 'clean', 'indices': ids.tolist()})
        cached = load_verified_chunk(path, done, binding, sha)
        if cached is not None:
            arrays, receipt = cached
        else:
            ctx.progress(f"Clean replay seed{record['seed']} {record['method']}: {start + 1}-{ids[-1] + 1}/{n}")
            x, y = _images(ctx, ids, opt)
            logits, features = _forward(model, x, opt['attack_batch'])
            arrays = {'indices': ids, 'labels': y.cpu().numpy(), 'clean_logits': logits.numpy(),
                      'clean_predictions': logits.argmax(1).numpy(),
                      'clean_margin': true_margins(logits, y.cpu()).numpy()}
            arrays.update(_moment_arrays(features, arrays['labels'], ctx.cfg['num_classes'], norm_eps, 'clean'))
            atomic_npz(path, **arrays)
            receipt = {'binding': binding, 'sha256': sha(path), 'n': len(ids)}
            atomic_json(done, receipt)
            del x, y, features
        blocks.append(arrays)
        receipts.append(receipt)
    combined = _ordered(blocks, ('indices', 'labels', 'clean_logits', 'clean_predictions', 'clean_margin'), labels)
    reference = Path(record.get('validation_path', Path(record['path']).parent / 'rounds' / 'round5' / 'validation.npz'))
    replay = check_clean_reference(combined, reference)
    geometry, moments = _geometry(blocks, 'clean', float(ctx.cfg['alignment']['scatter_epsilon']))
    atomic_npz(out / 'clean.npz', **combined, **moments)
    # The predefined class-stratified gradient subset is run once per model.
    proxy_ids = sensitivity_ids(labels, per_class=2)
    proxy_path, proxy_done = out / 'margin_gradient_proxy.npz', out / 'margin_gradient_proxy.json'
    proxy_binding = digest({'diagnostic': ctx.binding, 'checkpoint': record['sha256'],
                            'kind': 'margin_gradient_proxy', 'indices': proxy_ids.tolist()})
    cached = load_verified_chunk(proxy_path, proxy_done, proxy_binding, sha)
    if cached is None:
        x, y = _images(ctx, proxy_ids, opt)
        proxy = margin_gradient_proxy(model, x, y)
        atomic_npz(proxy_path, indices=proxy_ids, labels=y.cpu().numpy(), **proxy)
        atomic_json(proxy_done, {'binding': proxy_binding, 'sha256': sha(proxy_path),
                                'n': len(proxy_ids), 'selection': 'first two cleaned validation indices per true class',
                                'interpretation': 'local pixel-margin gradient L1; not a robustness certificate'})
        del x, y
    else:
        proxy = cached[0]
    summary = {'n': n, 'accuracy': float(np.mean(combined['clean_predictions'] == labels)),
               'geometry': geometry, 'inference_replay': replay,
               'clean_sha256': sha(out / 'clean.npz'),
               'margin_gradient_proxy': {'n': len(proxy_ids), 'mean_l1': float(proxy['margin_gradient_l1'].mean()),
                    'median_l1': float(np.median(proxy['margin_gradient_l1'])),
                    'selection': 'first two cleaned validation indices per true class',
                    'interpretation': 'descriptive local sensitivity; not a certificate'}}
    atomic_json(out / 'clean_summary.json', summary)
    return combined, summary


def _radius(ctx, model, record, out, opt, pixels, clean, common):
    from support import seed_for, sync
    atomic_json, atomic_npz, sha, digest = common
    labels, n = np.asarray(ctx.cat.val_labels, dtype=np.int64), len(ctx.cat.val_labels)
    epsilon = pixels / 255.
    norm_eps = float(ctx.cfg['alignment']['feature_norm_epsilon'])
    radius_dir = out / f"linf_{str(pixels).replace('.', 'p')}_255"
    radius_dir.mkdir(parents=True, exist_ok=True)
    blocks, receipts = [], []
    for start in range(0, n, opt['chunk']):
        ids = np.arange(start, min(n, start + opt['chunk']), dtype=np.int64)
        path = radius_dir / f'chunk_{start:05d}.npz'
        done = path.with_suffix('.json')
        # Method deliberately absent: corresponding method suites share attack seeds.
        attack_seed = seed_for('caenl_diagnostic_standard_aa', record['seed'], pixels, start)
        binding = digest({'diagnostic': ctx.binding, 'checkpoint': record['sha256'],
                          'pixels': pixels, 'indices': ids.tolist(), 'attack_seed': attack_seed,
                          'suite': STANDARD_ATTACKS, 'batch': opt['attack_batch']})
        cached = load_verified_chunk(path, done, binding, sha)
        if cached is not None:
            arrays, receipt = cached
        else:
            ctx.progress(f"AutoAttack {pixels:g}/255 seed{record['seed']} {record['method']}: {start + 1}-{ids[-1] + 1}/{n}")
            x, y = _images(ctx, ids, opt)
            sync(ctx.device)
            started = time.perf_counter()
            adv, norms = suite_attack(model, x, y, epsilon, attack_seed, opt['attack_batch'],
                                      ctx.device, radius_dir / f'attack_{start:05d}.log')
            logits, adv_features = _forward(model, adv, opt['attack_batch'])
            clean_logits, clean_features = _forward(model, x, opt['attack_batch'])
            if (not np.array_equal(clean_logits.argmax(1).numpy(), clean['clean_predictions'][ids])
                    or not np.allclose(clean_logits.numpy(), clean['clean_logits'][ids], atol=1e-4, rtol=1e-4)):
                raise ValueError('Attack clean forward differs from verified clean replay')
            cp, ap, yn = clean['clean_predictions'][ids], logits.argmax(1).numpy(), y.cpu().numpy()
            arrays = {'indices': ids, 'labels': yn, 'clean_logits': clean['clean_logits'][ids],
                      'adversarial_logits': logits.numpy(), 'clean_predictions': cp,
                      'adversarial_predictions': ap, 'robust_mask': (cp == yn) & (ap == yn),
                      'clean_margin': clean['clean_margin'][ids],
                      'adversarial_margin': true_margins(logits, y.cpu()).numpy(),
                      'linf': norms.cpu().numpy()}
            for layer in LAYERS:
                hc = F.normalize(clean_features[layer].double(), dim=1, eps=norm_eps)
                ha = F.normalize(adv_features[layer].double(), dim=1, eps=norm_eps)
                arrays[f'{layer}_unit_feature_movement_l2'] = (ha - hc).norm(dim=1).float().numpy()
            arrays.update(_moment_arrays(adv_features, yn, ctx.cfg['num_classes'], norm_eps, 'adversarial'))
            sync(ctx.device)
            elapsed = time.perf_counter() - started
            atomic_npz(path, **arrays)
            receipt = {'binding': binding, 'sha256': sha(path), 'n': len(ids),
                       'attack_seed': attack_seed, 'elapsed_seconds': elapsed,
                       'robust_correct': int(arrays['robust_mask'].sum())}
            atomic_json(done, receipt)
            del x, y, adv, clean_features, adv_features
        blocks.append(arrays)
        receipts.append(receipt)
    keys = ('indices', 'labels', 'clean_logits', 'adversarial_logits', 'clean_predictions',
            'adversarial_predictions', 'robust_mask', 'clean_margin', 'adversarial_margin', 'linf',
            *[f'{layer}_unit_feature_movement_l2' for layer in LAYERS])
    arrays = _ordered(blocks, keys, labels)
    geometry, moments = _geometry(blocks, 'adversarial', float(ctx.cfg['alignment']['scatter_epsilon']))
    atomic_npz(radius_dir / 'predictions.npz', **arrays, **moments)
    clean_correct = int(np.sum(arrays['clean_predictions'] == labels))
    robust_correct = int(arrays['robust_mask'].sum())
    summary = {'binding': ctx.binding, 'checkpoint_sha256': record['sha256'],
               'seed': record['seed'], 'method': record['method'], 'n': n,
               'epsilon_pixels': pixels, 'epsilon': epsilon, 'norm': 'Linf',
               'suite': 'standard', 'attacks_to_run': STANDARD_ATTACKS,
               'attack_batch_size': opt['attack_batch'], 'chunk_size': opt['chunk'],
               'clean_accuracy': clean_correct / n, 'robust_accuracy': robust_correct / n,
               'robust_correct': robust_correct, 'clean_correct': clean_correct,
               'clean_correct_survival': robust_correct / clean_correct if clean_correct else None,
               'adversarial_prediction_accuracy': float(np.mean(arrays['adversarial_predictions'] == labels)),
               'adversarial_geometry': geometry,
               'feature_movement': {layer: {'mean_l2': float(arrays[f'{layer}_unit_feature_movement_l2'].mean()),
                    'median_l2': float(np.median(arrays[f'{layer}_unit_feature_movement_l2']))} for layer in LAYERS},
               'mean_adversarial_margin': float(arrays['adversarial_margin'].mean()),
               'descriptive_outcome_groups': descriptive_attack_groups(arrays),
               'max_observed_linf': float(arrays['linf'].max()),
               'elapsed_seconds': float(sum(r['elapsed_seconds'] for r in receipts)),
               'predictions_sha256': sha(radius_dir / 'predictions.npz'),
               'returned_output_geometry_interpretation': 'AutoAttack output mixes initially incorrect unchanged inputs, survivors, and successes retained from different component attacks; geometry is not a uniform perturbation treatment',
               'interpretation': 'outcome groups and returned-output geometry are descriptive; not causal mediation or certified robustness'}
    atomic_json(radius_dir / 'DONE.json', summary)
    return summary


def run(ctx):
    """Run the predeclared 30-model/150-suite confirmation-only diagnostic."""
    from common import atomic_json, atomic_npz, sha, digest, paired, holm
    common = (atomic_json, atomic_npz, sha, digest)
    opt = _options(ctx.protocol)
    if opt['attack_batch'] <= 0 or opt['chunk'] <= 0 or opt['workers'] < 0:
        raise ValueError('Invalid attack loader options')
    expected = {('confirmation', seed, method) for seed in SEEDS for method in METHODS}
    records = list(ctx.records)
    actual = [(r['stage'], int(r['seed']), r['method']) for r in records]
    if len(actual) != 30 or len(set(actual)) != 30 or set(actual) != expected:
        raise ValueError('Exactly 30 declared ResNet-50 confirmation models are required')
    root = Path(ctx.root) / 'robustness'
    root.mkdir(parents=True, exist_ok=True)
    summaries, clean_summaries = [], []
    for record in sorted(records, key=lambda r: (r['seed'], METHODS.index(r['method']))):
        checkpoint = Path(record['path'])
        if sha(checkpoint) != record['sha256']:
            raise ValueError(f'Frozen checkpoint changed: {checkpoint}')
        model = ctx.load_model(record).to(ctx.device).float().eval()
        out = root / f"seed{record['seed']}" / record['method']
        out.mkdir(parents=True, exist_ok=True)
        clean, clean_summary = _clean_model(ctx, model, record, out, opt, common)
        clean_summaries.append({'seed': record['seed'], 'method': record['method'], **clean_summary})
        for pixels in opt['radii']:
            summaries.append(_radius(ctx, model, record, out, opt, pixels, clean, common))
            ctx.progress(f"Robustness suites complete: {len(summaries)}/150; models started: {len(clean_summaries)}/30")
            if len(summaries) == 1:
                measured = summaries[0]['elapsed_seconds']
                ctx.progress(f"First full-validation suite took {measured / 3600:.2f} hours. "
                             f"150 at that same cost would take {150 * measured / 86400:.2f} days; "
                             'radius/model costs vary and clean diagnostics add time.')
        if sha(checkpoint) != record['sha256']:
            raise ValueError('Frozen checkpoint changed during diagnostics')
        del model
        if str(ctx.device).startswith('cuda'):
            torch.cuda.empty_cache()
    if len(summaries) != 150:
        raise RuntimeError('Expected 150 full-validation standard suites')
    contrasts = []
    primary = {(s['seed'], s['method']): s for s in summaries if s['epsilon_pixels'] == opt['primary']}
    for target, reference in (('lite', 'entropy'), ('tuned_fixed', 'entropy'), ('lite', 'tuned_fixed')):
        target_values = [100 * primary[(seed, target)]['robust_accuracy'] for seed in SEEDS]
        reference_values = [100 * primary[(seed, reference)]['robust_accuracy'] for seed in SEEDS]
        result = paired(target_values, reference_values)
        contrasts.append({'target': target, 'reference': reference,
                          'endpoint': 'robust accuracy, full cleaned validation denominator',
                          'epsilon_pixels': 2., **result})
    for key in ('paired_t_p', 'exact_signflip_p'):
        adjusted = holm([c[key] for c in contrasts])
        for contrast, value in zip(contrasts, adjusted):
            contrast[key + '_holm'] = float(value)
    intersection = common_clean_intersection(root, SEEDS, METHODS, opt['radii'])
    descriptive = []
    for pixels in opt['radii']:
        selected = {row['seed']: row for row in intersection if row['epsilon_pixels'] == pixels}
        for target, reference in (('lite', 'entropy'), ('tuned_fixed', 'entropy'), ('lite', 'tuned_fixed')):
            if any(selected[seed]['common_clean_correct_n'] == 0 for seed in SEEDS):
                descriptive.append({'epsilon_pixels': pixels, 'target': target, 'reference': reference,
                                    'status': 'UNDEFINED: at least one seed has an empty common clean-correct set'})
                continue
            target_values = [100 * selected[seed]['survival'][target]['accuracy'] for seed in SEEDS]
            reference_values = [100 * selected[seed]['survival'][reference]['accuracy'] for seed in SEEDS]
            result = paired(target_values, reference_values)
            descriptive.append({'epsilon_pixels': pixels, 'target': target, 'reference': reference,
                                'endpoint': 'survival on the same three-method clean-correct intersection',
                                'interpretation': 'descriptive, unadjusted interval; seed-level unit',
                                **{key: result[key] for key in ('n_pairs', 'difference_pp', 'sd_difference_pp',
                                                               'ci95_low_pp', 'ci95_high_pp', 'degenerate_note')}})
    summary = {'binding': ctx.binding, 'status': 'COMPLETE', 'models': 30, 'standard_suites': 150,
               'backbone': 'resnet50', 'stage': 'confirmation', 'seeds': list(SEEDS),
               'methods': list(METHODS), 'radii_pixels': list(opt['radii']), 'primary_radius_pixels': 2.,
               'clean': clean_summaries, 'suites': summaries, 'primary_contrasts': contrasts,
               'same_clean_correct_intersection': intersection,
               'same_clean_correct_descriptive_contrasts': descriptive,
               'moment_format': 'float64 unit-feature class sums and scalar sums of squared Euclidean norms; sparse active-class rows in chunks and dense class rows in final aggregate files',
               'geometry_interpretation': 'Full-set AutoAttack returned-output geometry includes unchanged initially incorrect images and successful perturbations retained from heterogeneous component attacks. Outcome-conditioned displacement and margin summaries are descriptive and do not identify causal mediation.',
               'secondary_endpoints': 'other radii, clean-correct survival, geometry, movement, and gradient proxies are descriptive',
               'scope': 'frozen final ResNet-50 confirmation checkpoints; no retraining, ResNet-18 robustness, or mediation claim'}
    atomic_json(root / 'ROBUSTNESS_SUMMARY.json', summary)
    return summary
