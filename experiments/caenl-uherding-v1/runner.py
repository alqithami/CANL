#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CAENL recent-baseline run v1: UHerding-Margin, matched-protocol implementation.


Six NEW branches, seeds 801--806. Existing runs are read-only.
No installation, network download, dataset change, or C4 run.


Reference: Bae, Oliveira and Sutherland, Uncertainty Herding:
One Active Learning Method for All Label Budgets, ICLR 2025,
https://arxiv.org/abs/2412.20644 ; https://github.com/BorealisAI/uherding


This independently implemented comparator uses the published uncertainty-
weighted RBF coverage objective, margin uncertainty, temperature selection
on held-out acquired labels, and adaptive minimum-positive-distance radius.
Explicit matching choices: fixed, normalized layer4 features from each seed's
SHARED INITIAL ResNet-50 instead of a separately pretrained SSL extractor;
the existing 10,000-candidate / 10%-per-pass acquisition convention;
existing warm-started task training rather than the paper's cold starts.
It is NOT an authors-code benchmark reproduction and uses no extra pretraining.
Lazy greedy is tested against exhaustive greedy before training.


Authoring runtime could not execute this script. IBM-side CPU, integration,
CUDA and data checks must pass before scientific training starts.
"""
from __future__ import annotations
import argparse
import ast
import contextlib
import copy
import gc
import hashlib
import heapq
import importlib.util
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone


sys.dont_write_bytecode = True
BASE = Path('/mnt/caenl/active/results/caenl-imagenet100-aligned-confirmation-v1')
FOCUS = Path('/mnt/caenl/active/results/caenl-focused-completion-v1')
ROOT = Path('/mnt/caenl/active/results/caenl-uherding-v1')
SESSION = 'caenl-uherding-v1'
NAME = 'uherding_margin_initial_r50'
METHOD = {'name': NAME, 'acquisition': 'uherding_margin', 'regularizer': 'none'}
SEEDS = list(range(801, 807))
OLD_BINDING = '159fe52901f774e7abeb40a3afe96de81f1084a1bde0f63d070c7d2ecda325e1'
FOCUS_BINDING = 'c6009897fdc2d87df4e257cd89e710bbb2465d6f80a98e4349e36ea01f2af058'
REPORT = 'caenl-uherding-v1.md'
np = None
torch = None




def utc():
    return datetime.now(timezone.utc).isoformat()




def require(ok, message):
    if not bool(ok):
        raise RuntimeError(message)




def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))




def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()




def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()




def inside(root, rel):
    rel = Path(rel)
    require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe relative path: ' + str(rel))
    p = root / rel
    require(p.resolve().is_relative_to(root.resolve()), 'Path escapes its root: ' + str(p))
    return p




def text_out(path, text):
    path = Path(path)
    require(path.resolve().is_relative_to(ROOT.resolve()), 'Write outside new result tree refused')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)




def put(path, obj):
    text_out(path, json.dumps(obj, indent=2, allow_nan=False) + '\n')




def progress(message):
    put(ROOT / 'STATUS.json', {'status': 'RUNNING', 'pid': os.getpid(), 'time': utc(), 'message': message})
    print(message, flush=True)




def pin(pins, path, expected=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing or symlinked input: ' + str(path))
    h = sha(path)
    if expected is not None:
        require(h == expected, 'Input checksum mismatch: ' + str(path))
    pins[str(path)] = h
    return h




def check_pins(pins):
    for path, h in pins.items():
        require(sha(path) == h, 'Previously bound input changed: ' + path)




def frozen_inputs():
    complete = read(BASE / 'COMPLETE.json')
    inp = read(BASE / 'INPUT_BINDING.json')
    require(complete.get('status') == 'COMPLETE' and complete.get('methods_complete') == 54, 'Original 54-run study is not complete')
    require(inp['binding'] == complete['binding'] == OLD_BINDING, 'Unexpected original campaign binding')
    cfg, files = inp['protocol'], inp['source_files']
    require(digest({'protocol': cfg, 'source_files': files}) == OLD_BINDING, 'Original binding cannot be recomputed')
    require(cfg['seeds'] == SEEDS and cfg['rounds'] == 5 and len(cfg['methods']) == 9, 'Unexpected original experimental matrix')
    require(cfg['acquisition_per_round'] == 12669 and cfg['num_classes'] == 100, 'Unexpected label budget/classes')
    require(cfg['acquisition']['candidate_size'] == 10000 and cfg['acquisition']['per_pass_fraction'] == .1, 'Unexpected candidate-pass convention')
    require(read(BASE / 'source/protocol.json') == cfg, 'Original protocol text and binding disagree')
    pins = {}
    for p in (BASE / 'COMPLETE.json', BASE / 'INPUT_BINDING.json', BASE / 'SHA256SUMS'):
        pin(pins, p)
    dest = ROOT / 'frozen_source'
    dest.mkdir(exist_ok=True)
    for rel, h in files.items():
        old, new = inside(BASE / 'source', rel), inside(dest, rel)
        pin(pins, old, h)
        new.parent.mkdir(parents=True, exist_ok=True)
        if new.exists():
            require(sha(new) == h, 'Copied frozen source changed: ' + rel)
        else:
            shutil.copyfile(old, new)
    fbind = read(FOCUS / 'ADDITIONAL_BINDING.json')
    require(fbind['binding'] == FOCUS_BINDING, 'Unexpected supplementary campaign')
    require(digest({'protocol': fbind['protocol'], 'plan': fbind['plan']}) == FOCUS_BINDING, 'Supplementary binding cannot be recomputed')
    require(read(FOCUS / 'COMPLETE.json').get('additional_method_runs') == 6, 'Six supplementary runs have not completed')
    # Reuse only the previously executed arithmetic-check functions, never its worker.
    audit_path = FOCUS / 'runner.py'
    pin(pins, audit_path, fbind['plan']['runner_sha256'])
    audit_copy = ROOT / 'saved_array_checks.py'
    if audit_copy.exists():
        require(sha(audit_copy) == sha(audit_path), 'Copied audit source differs')
    else:
        shutil.copyfile(audit_path, audit_copy)
    for p in (FOCUS / 'ADDITIONAL_BINDING.json', FOCUS / 'COMPLETE.json', FOCUS / 'AUTOATTACK_SOURCE.json'):
        pin(pins, p)
    for seed in SEEDS:
        sd = BASE / f'seed{seed}'
        initial = read(sd / 'shared_initial/summary.json')
        pin(pins, sd / 'shared_initial/state.pt', initial['checkpoint_sha256'])
        for rel in ('shared_initial/summary.json', 'splits.npz', 'targets.json'):
            pin(pins, sd / rel)
        for origin, method in ((BASE, 'entropy'), (FOCUS, 'entropy_aligned_lite')):
            out = origin / f'seed{seed}' / method
            done = read(out / 'DONE.json')
            pin(pins, out / 'DONE.json')
            pin(pins, out / 'summary.json', done['payload']['summary.json'])
            phase = out / 'rounds/round5'
            p_done = read(phase / 'DONE.json')
            for rel in ('DONE.json', 'summary.json'):
                pin(pins, phase / rel)
            pin(pins, phase / 'validation.npz', p_done['payload']['validation.npz'])
    if (ROOT / 'INPUT_FILES.json').exists():
        require(read(ROOT / 'INPUT_FILES.json') == pins, 'Inputs changed since this new run began')
    else:
        put(ROOT / 'INPUT_FILES.json', pins)
    sys.path.insert(0, str(dest))
    spec = importlib.util.spec_from_file_location('caenl_saved_array_checks', audit_copy)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    audit.np = np
    return copy.deepcopy(cfg), files, inp['environment'], pins, audit




def ece(logits, labels, temperature):
    x = np.asarray(logits, dtype=np.float64) / float(temperature)
    x -= x.max(axis=1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(axis=1, keepdims=True)
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == labels
    bins = np.minimum((confidence * 15).astype(np.int64), 14)
    value = 0.0
    for k in range(15):
        sel = bins == k
        if sel.any():
            value += float(sel.mean() * abs(correct[sel].mean() - confidence[sel].mean()))
    return value




def calibrate_temperature(logits, labels):
    require(len(labels) > 0 and np.isfinite(logits).all(), 'Invalid held-out calibration data')
    # Fixed in this implementation before any scientific run; no validation-set tuning.
    grid = np.logspace(-1, 1, 81, dtype=np.float64)
    values = np.asarray([ece(logits, labels, t) for t in grid])
    index = int(np.argmin(values))
    return float(grid[index]), {'temperatures': grid.tolist(), 'ece': values.tolist(), 'selected_index': index, 'selected_at_grid_boundary': index in (0, len(grid) - 1), 'labels_used': 'the existing acquired 633-image feedback holdout only'}




def margin_uncertainty(logits, temperature):
    x = np.asarray(logits, dtype=np.float64) / temperature
    require(x.ndim == 2 and x.shape[1] >= 2 and np.isfinite(x).all(), 'Invalid pool logits')
    x -= x.max(axis=1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(axis=1, keepdims=True)
    top = np.sort(p, axis=1)[:, -2:]
    u = 1.0 - (top[:, 1] - top[:, 0])
    require(np.isfinite(u).all() and u.min() >= -1e-12 and u.max() <= 1 + 1e-12, 'Invalid uncertainty weights')
    return np.clip(u, 0, 1)




def squared_dist(a, b):
    return ((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * (a @ b.T)).clamp_min(0)




def radius_squared(centers, report=None):
    require(len(centers) >= 2, 'At least two acquired feature vectors are needed')
    best = float('inf')
    size = 1024
    count = math.ceil(len(centers) / size)
    for ib, i in enumerate(range(0, len(centers), size)):
        a = centers[i:i + size]
        for j in range(i, len(centers), size):
            d = squared_dist(a, centers[j:j + size])
            # Double-precision numerical zero, including exact duplicate embeddings.
            local = d.masked_fill(d <= 1e-12, float('inf')).min()
            best = min(best, float(local))
        if report and (ib % 10 == 0 or ib + 1 == count):
            report(f'UHerding radius: labeled-feature block {ib + 1}/{count}')
    require(math.isfinite(best) and best > 0, 'No positive feature distance: cannot define the UHerding radius')
    return best




def nearest_squared(points, centers):
    require(len(centers) > 0, 'Empty coverage-center set')
    result = torch.full((len(points),), float('inf'), dtype=points.dtype, device=points.device)
    for i in range(0, len(points), 1024):
        best = torch.full((len(points[i:i + 1024]),), float('inf'), dtype=points.dtype, device=points.device)
        for j in range(0, len(centers), 4096):
            best = torch.minimum(best, squared_dist(points[i:i + 1024], centers[j:j + 4096]).min(1).values)
        result[i:i + len(best)] = best
    return result




def kernel_matrix(points, sigma2):
    k = torch.empty((len(points), len(points)), dtype=points.dtype, device=points.device)
    for i in range(0, len(points), 512):
        k[i:i + 512] = torch.exp(-squared_dist(points[i:i + 512], points) / sigma2)
    k.diagonal().fill_(1.0)
    return k




def greedy_positions(kernel, coverage, uncertainty, budget, lazy=True):
    """Greedy maximization of weighted additional coverage; no pool labels.


    Lazy heap values are decreasing marginal-gain upper bounds. The objective
    and tie rule are identical to exhaustive greedy on the same finite kernel.
    """
    n = len(coverage)
    require(kernel.shape == (n, n) and 0 < budget <= n, 'Invalid greedy selection shapes/budget')
    require(bool(torch.isfinite(kernel).all()) and bool(torch.isfinite(coverage).all()) and bool(torch.isfinite(uncertainty).all()), 'Nonfinite coverage input')
    require(float(kernel.min()) >= 0 and float(uncertainty.min()) >= 0, 'Coverage must be nonnegative')
    c = coverage.clone()
    chosen, gains = [], []
    used = torch.zeros(n, dtype=torch.bool, device=kernel.device)
    if not lazy:
        for _ in range(budget):
            gain = ((kernel - c[None, :]).clamp_min(0) * uncertainty[None, :]).sum(1)
            gain[used] = -float('inf')
            j = int(gain.argmax())
            chosen.append(j)
            gains.append(float(gain[j]))
            used[j] = True
            c = torch.maximum(c, kernel[j])
        return np.asarray(chosen, dtype=np.int64), np.asarray(gains)
    initial = []
    for i in range(0, n, 256):
        v = ((kernel[i:i + 256] - c[None, :]).clamp_min(0) * uncertainty[None, :]).sum(1)
        initial.extend(v.detach().cpu().tolist())
    heap = [(-float(v), j, 0) for j, v in enumerate(initial)]
    heapq.heapify(heap)
    for iteration in range(budget):
        while True:
            negative, j, evaluated_at = heapq.heappop(heap)
            if evaluated_at == iteration:
                chosen.append(j)
                gains.append(-negative)
                c = torch.maximum(c, kernel[j])
                break
            current = float(((kernel[j] - c).clamp_min(0) * uncertainty).sum())
            require(current <= -negative + 1e-8 * max(1.0, -negative), 'Numerical monotonicity check failed')
            heapq.heappush(heap, (-current, j, iteration))
    return np.asarray(chosen, dtype=np.int64), np.asarray(gains, dtype=np.float64)




def new_tests(device):
    checks = []
    generator = np.random.default_rng(418201)
    for trial in range(6):
        features = generator.normal(size=(29, 7))
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        z = torch.as_tensor(features, dtype=torch.float64, device=device)
        d = squared_dist(z, z).cpu().numpy()
        brute = ((features[:, None] - features[None, :]) ** 2).sum(2)
        require(np.allclose(d, brute, atol=2e-12), 'Distance implementation differs from direct calculation')
        sigma2 = radius_squared(z[:8])
        dref = brute[:8, :8]
        require(abs(sigma2 - dref[dref > 1e-12].min()) < 2e-12, 'Adaptive radius mismatch')
        k = kernel_matrix(z, sigma2)
        c = torch.exp(-nearest_squared(z, z[:3]) / sigma2)
        u = torch.as_tensor(generator.uniform(.02, 1, len(z)), dtype=torch.float64, device=device)
        a, ga = greedy_positions(k, c, u, 12, True)
        b, gb = greedy_positions(k, c, u, 12, False)
        require(np.array_equal(a, b) and np.allclose(ga, gb, rtol=1e-12, atol=1e-12), 'Lazy and exhaustive greedy disagree')
        require(len(np.unique(a)) == 12, 'Duplicate selected candidates')
        require(np.all(np.diff(ga) <= 1e-10), 'Marginal gains are not decreasing')
        checks.append('distance/radius/lazy-exhaustive trial ' + str(trial))
    k = torch.eye(6, dtype=torch.float64, device=device)
    c = torch.zeros(6, dtype=torch.float64, device=device)
    u = torch.tensor([.1, .8, .5, .9, .4, .2], dtype=torch.float64, device=device)
    ids, _ = greedy_positions(k, c, u, 6)
    require(ids.tolist() == [3, 1, 2, 4, 5, 0], 'Identity-kernel uncertainty limit failed')
    ids, _ = greedy_positions(k, c, c, 6)
    require(ids.tolist() == list(range(6)), 'Zero-gain deterministic selection failed')
    logits = generator.normal(size=(31, 4))
    labels = generator.integers(0, 4, 31)
    t, rec = calibrate_temperature(logits, labels)
    require(rec['ece'][rec['selected_index']] == min(rec['ece']), 'Temperature search did not minimize holdout ECE')
    u = margin_uncertainty(logits, t)
    require(np.allclose(u, margin_uncertainty(logits[:, ::-1], t)), 'Uncertainty changes under class permutation')
    require(np.allclose(margin_uncertainty(np.zeros((4, 3)), 1), 1), 'Uniform uncertainty limit failed')
    checks.extend(['uncertainty limit', 'zero-gain ties', 'temperature ECE search', 'class permutation', 'uniform probabilities'])
    return checks




def feature_cache(cfg, cat, seed, initial_cp, initial_hash, device, binding):
    from support import Model, atom_json
    from evaluation import infer
    out = ROOT / f'seed{seed}/features'
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'initial_layer4.npy'
    meta = out / 'COMPLETE.json'
    key = digest({'binding': binding, 'seed': seed, 'initial_checkpoint': initial_hash, 'manifest': cfg['expected_image_manifest_sha256'], 'layer': 'layer4', 'normalized': True})
    if meta.exists():
        m = read(meta)
        require(m['binding'] == key and sha(path) == m['sha256'], 'Frozen feature cache differs')
    else:
        progress(f'UHerding seed {seed}: caching fixed initial-ResNet features for the existing training pool')
        started = time.perf_counter()
        model = Model(cfg['num_classes']).to(device).float().to(memory_format=torch.channels_last)
        state = torch.load(initial_cp, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'], strict=True)
        del state
        ft_cfg = copy.deepcopy(cfg)
        ft_cfg['layers'] = ['layer4']
        rec = infer(model, cat, 'train', np.arange(len(cat.labels)), ft_cfg, device, features=True)
        # Discard labels and logits; g is frozen independently of every new acquisition.
        z = rec['features']['layer4'].float().numpy().copy()
        del rec, model
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        require(np.isfinite(z).all() and float(norms.min()) > 1e-12, 'Invalid initial features')
        z /= norms
        tmp = path.with_name(path.name + '.partial')
        with tmp.open('wb') as f:
            np.save(f, z, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        m = {'binding': key, 'sha256': sha(path), 'shape': list(z.shape), 'elapsed_seconds': time.perf_counter() - started, 'source': 'shared initial ResNet-50 layer4, frozen for all rounds', 'extra_pretrained_data': False}
        atom_json(meta, m)
        del z
        gc.collect()
        torch.cuda.empty_cache()
    features = np.load(path, allow_pickle=False)
    require(features.shape == (cfg['expected_train_count'], 2048) and np.isfinite(features).all(), 'Feature cache shape/values differ')
    return features, m




def query_round(model, cat, labeled, held, seed, phase, cfg, device, out, mb, cp_hash, features, progress_fn):
    from support import atom_json, atom_npz, seed_for, tensor_digest
    from evaluation import infer
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    selected_path, done_path = out / 'selection.npz', out / 'COMPLETE.json'
    labeled = np.sort(np.asarray(labeled, dtype=np.int64))
    held = np.asarray(held, dtype=np.int64)
    ann = np.sort(np.concatenate([labeled, held]))
    pool = np.setdiff1d(np.arange(len(cat.labels)), ann)
    desired = int(cfg['acquisition_per_round'])
    bound = digest({'binding': mb, 'checkpoint': cp_hash, 'labeled': labeled.tolist(), 'holdout': held.tolist(), 'seed': seed, 'phase': phase, 'method': METHOD})
    if done_path.exists():
        info = read(done_path)
        require(info['binding'] == bound and sha(selected_path) == info['selection_sha256'], 'Selection resume binding mismatch')
        with np.load(selected_path, allow_pickle=False) as f:
            chosen = f['selected']
        require(len(chosen) == desired and len(np.unique(chosen)) == desired and np.isin(chosen, pool).all(), 'Saved selection invalid')
        for p in info['passes']:
            require(sha(out / f"pass_{p['pass']:03d}.npz") == p['sha256'], 'Saved pass changed')
        return chosen, info
    require(len(pool) >= desired, 'Insufficient unlabeled pool')
    started = time.perf_counter()
    model_before = tensor_digest(model.state_dict())
    try:
        progress_fn(f'UHerding seed {seed} round {phase}/5: held-out temperature calibration and current pool predictions')
        calibration = infer(model, cat, 'train', held, cfg, device, features=False)
        temp, temp_record = calibrate_temperature(calibration['logits'], calibration['labels'])
        require(np.array_equal(calibration['labels'], cat.labels[held]), 'Holdout-label mapping differs')
        del calibration
        pool_rec = infer(model, cat, 'train', pool, cfg, device, features=False)
        uncertainty = margin_uncertainty(pool_rec['logits'], temp)
        # No ground-truth labels from the unlabeled inference output reach acquisition.
        del pool_rec
        require(tensor_digest(model.state_dict()) == model_before, 'Acquisition inference changed task-model weights')
        z = torch.as_tensor(features, dtype=torch.float64, device=device)
        pool_ids = torch.as_tensor(pool, dtype=torch.long, device=device)
        zp = z[pool_ids]
        centers = z[torch.as_tensor(ann, dtype=torch.long, device=device)]
        sigma2 = radius_squared(centers, progress_fn)
        progress_fn(f'UHerding seed {seed} round {phase}: computing coverage from existing annotations')
        min_d2 = nearest_squared(zp, centers)
        del centers
        uncertainty_t = torch.as_tensor(uncertainty, dtype=torch.float64, device=device)
        rng = np.random.default_rng(seed_for('candidate_passes', seed, phase))
        active = np.arange(len(pool), dtype=np.int64)
        chosen, passes = [], []
        pass_id = 0
        while len(chosen) < desired:
            n = min(len(active), int(cfg['acquisition']['candidate_size']))
            pos = rng.choice(active, n, replace=False)
            quota = min(desired - len(chosen), max(1, int(math.ceil(n * cfg['acquisition']['per_pass_fraction']))))
            pass_id += 1
            pp = out / f'pass_{pass_id:03d}.npz'
            pm = out / f'pass_{pass_id:03d}.json'
            pbind = digest({'round_binding': bound, 'pass': pass_id, 'temperature': temp, 'sigma_squared': sigma2, 'candidate_ids': pool[pos].tolist(), 'quota': quota})
            if pm.exists():
                r = read(pm)
                require(r['binding'] == pbind and sha(pp) == r['sha256'], 'Partial acquisition binding/hash differs')
                with np.load(pp, allow_pickle=False) as a:
                    require(np.array_equal(a['candidate_indices'], pool[pos]), 'Resumed candidate order differs')
                    local = a['selected_positions'].copy()
                require(len(local) == quota and len(np.unique(local)) == quota and np.all((local >= 0) & (local < n)), 'Invalid resumed pass')
            else:
                tpos = torch.as_tensor(pos, dtype=torch.long, device=device)
                kernel = kernel_matrix(zp[tpos], sigma2)
                coverage = torch.exp(-min_d2[tpos] / sigma2)
                local, gains = greedy_positions(kernel, coverage, uncertainty_t[tpos], quota, lazy=True)
                del kernel, coverage, tpos
                atom_npz(pp, candidate_indices=pool[pos], selected_indices=pool[pos[local]], selected_positions=local, uncertainty=uncertainty[pos], marginal_gain_sum=gains)
                r = {'binding': pbind, 'pass': pass_id, 'candidates': n, 'selected': quota, 'sha256': sha(pp)}
                atom_json(pm, r)
            take = pos[local]
            chosen.extend(pool[take].tolist())
            active = active[~np.isin(active, take)]
            passes.append({'pass': pass_id, 'candidates': n, 'selected': quota, 'sha256': sha(pp)})
            if len(chosen) < desired:
                min_d2 = torch.minimum(min_d2, nearest_squared(zp, zp[torch.as_tensor(take, dtype=torch.long, device=device)]))
            progress_fn(f'UHerding seed {seed} round {phase}: selected {len(chosen):,}/{desired:,}')
        chosen = np.asarray(chosen, dtype=np.int64)
        require(len(chosen) == desired and len(np.unique(chosen)) == desired and not np.intersect1d(chosen, ann).size, 'Acquisition budget/disjointness failed')
        atom_npz(selected_path, selected=chosen, previous_labeled=labeled, control_holdout=held)
        temp_record.update(binding=bound, temperature=temp, sigma=math.sqrt(sigma2), sigma_squared=sigma2, positive_distance_tolerance_squared=1e-12)
        atom_json(out / 'PARAMETERS.json', temp_record)
        info = {'binding': bound, 'method': METHOD, 'seed': seed, 'round': phase, 'requested': desired, 'selected': len(chosen), 'unique_pool_size': len(pool), 'passes': passes, 'candidate_presentations': sum(p['candidates'] for p in passes), 'temperature': temp, 'sigma': math.sqrt(sigma2), 'selection_sha256': sha(selected_path), 'original_checkpoint_sha256': cp_hash, 'pool_true_labels_used_for_selection': False, 'selected_class_counts': np.bincount(cat.labels[chosen], minlength=cfg['num_classes']).tolist(), 'feature_source': 'fixed normalized shared-initial ResNet-50 layer4', 'calibration_source': 'budgeted feedback holdout; official validation not used'}
        del z, zp, min_d2, pool_ids, uncertainty_t
        gc.collect()
        torch.cuda.empty_cache()
    except BaseException:
        with (out / 'attempts.jsonl').open('a') as f:
            f.write(json.dumps({'time': utc(), 'status': 'interrupted_or_failed', 'elapsed_seconds': time.perf_counter() - started}) + '\n')
        raise
    with (out / 'attempts.jsonl').open('a') as f:
        f.write(json.dumps({'time': utc(), 'status': 'completed', 'elapsed_seconds': time.perf_counter() - started}) + '\n')
    info['elapsed_seconds'] = sum(json.loads(line)['elapsed_seconds'] for line in (out / 'attempts.jsonl').read_text().splitlines())
    info['timing_scope'] = 'all recorded acquisition attempts, including calibration, pool inference and kernels; frozen feature extraction is separate'
    atom_json(done_path, info)
    return chosen, info




def integration_test(cfg, audit):
    from self_test import fixture, Tiny
    from support import stratified_sample, tensor_digest
    from training import run_phase
    from evaluation import infer
    from control import calibrate
    small = copy.deepcopy(cfg)
    small.update(num_classes=4, expected_train_count=144, expected_val_count=32, initial_labels=48, holdout_count=16, acquisition_per_round=16, rounds=1, initial_epochs=2, round_epochs=2, batch_classes=4, samples_per_class=2, workers=0, eval_workers=0, eval_batch=16, control_interval=2, calibration_batches=2, checkpoint_every_steps=2)
    small['acquisition'].update(candidate_size=32)
    with tempfile.TemporaryDirectory(prefix='selector_test_', dir=ROOT) as temp:
        root = Path(temp)
        cat = fixture(root / 'images', small)
        ann = stratified_sample(np.arange(144), cat.labels, 48, 4)
        held = stratified_sample(ann, cat.labels, 16, 5)
        labeled = np.setdiff1d(ann, held)
        initial_method = {'name': 'shared_initial', 'acquisition': 'none', 'regularizer': 'none'}
        _, source = run_phase(small, cat, 801, initial_method, 0, labeled, held, None, None, root / 'initial', 'uherding-test', 20, lambda x: None, 'cpu', model_factory=Tiny)
        model = Tiny(4)
        state = torch.load(source, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        targets, _ = calibrate(model, cat, labeled, 801, small, 'cpu')
        features = infer(model, cat, 'train', np.arange(144), small, 'cpu', features=True)['features']['layer4'].numpy()
        features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        before = tensor_digest(model.state_dict())
        out = root / 'query'
        chosen, info = query_round(model, cat, labeled, held, 801, 1, small, torch.device('cpu'), out, 'uherding-test', sha(source), features, lambda x: None)
        chosen2, _ = query_round(model, cat, labeled, held, 801, 1, small, torch.device('cpu'), out, 'uherding-test', sha(source), features, lambda x: None)
        require(np.array_equal(chosen, chosen2), 'End-to-end selector resume differs')
        require(len(np.unique(chosen)) == 16 and not np.intersect1d(chosen, ann).size, 'Toy selector annotation accounting failed')
        require(tensor_digest(model.state_dict()) == before, 'Toy acquisition modified task model')
        training_ids = np.sort(np.concatenate([labeled, chosen]))
        summary, checkpoint = run_phase(small, cat, 801, METHOD, 1, training_ids, held, source, targets, root / 'phase', 'uherding-test', 20, lambda x: None, 'cpu', model_factory=Tiny)
        audit.check_evaluation(root / 'phase/validation.npz', summary['validation'], cat.val_labels, np.arange(32), small)
        require(summary['steps'] == 12 and summary['annotation_count'] == 64, 'Toy new-method training budget differs')
        later = torch.load(checkpoint, map_location='cpu', weights_only=False)
        require(torch.equal(later['model']['bn.running_mean'], state['model']['bn.running_mean']), 'Toy new-method BatchNorm freeze changed')
    return ['actual toy query', 'query resume', 'label budget/disjointness', 'task-state preservation', 'actual new-method backward', 'independent saved-array arithmetic', 'BatchNorm freeze']




def references(cfg, cat, audit):
    rows = []
    for seed in SEEDS:
        for origin, method in ((BASE, 'entropy'), (FOCUS, 'entropy_aligned_lite')):
            out = origin / f'seed{seed}' / method
            summary = read(out / 'summary.json')
            final = summary['rounds'][-1]
            audit.check_evaluation(out / 'rounds/round5/validation.npz', final['validation'], cat.val_labels, np.arange(len(cat.val_labels)), cfg)
            v = final['validation']
            rows.append({'seed': seed, 'method': method, 'clean_accuracy': v['accuracy'], 'cross_entropy': v['ce'], 'ece': v['ece']})
    return rows




def report(status):
    lines = ['# CAENL UHerding recent-baseline run v1', '', 'Status: ' + status, 'UTC: ' + utc(), '',
             'This is a later matched-protocol comparator, not an expansion of the original prespecified family.',
             'UHerding-Margin uses fixed shared-initial ResNet-50 features, not a separately pretrained self-supervised encoder.',
             'The task backbone/training, label budgets, fixed holdout, seeds and final attack match the existing CAENL study.',
             'Candidate passes and warm-started training follow the CAENL protocol, not the authors original benchmark recipe.',
             'No original training run was restarted. C4 and other datasets were not run. Negative results count as valid outcomes.',
             'This study cannot claim reproduction of published UHerding benchmark numbers or universal state-of-the-art superiority.', '']
    for name in ('RUN_BINDING.json', 'SELF_TEST.json', 'CUDA_PREFLIGHT.json', 'DATA_VERIFIED.json', 'REFERENCE_METRICS.json', 'METRICS.json', 'COMPARISONS.json', 'PRESERVATION.json', 'FAILED.json'):
        p = ROOT / name
        if p.exists():
            lines += ['## ' + name, 'SHA-256: ' + sha(p), '````````', p.read_text(), '````````', '']
    for seed in SEEDS:
        p = ROOT / f'seed{seed}/features/COMPLETE.json'
        if p.exists():
            lines += ['## Feature extraction seed ' + str(seed), '```json', p.read_text(), '```']
        p = ROOT / f'seed{seed}' / NAME / 'summary.json'
        if p.exists():
            lines += ['## Method summary seed ' + str(seed), 'SHA-256: ' + sha(p), '````````', p.read_text(), '````````']
    lines += ['', '## Exact launcher source', '````````', (ROOT / 'runner.py').read_text(encoding='utf-8-sig'), '````````', '', 'END OF UHerding REPORT']
    target = ROOT / REPORT
    text_out(target, '\n'.join(lines) + '\n')
    text_out(ROOT / (REPORT + '.sha256'), sha(target) + '  ' + REPORT + '\n')
    return target




def run_study(cfg, cat, binding, device, audit, reference_rows):
    from support import Model, atom_json
    from training import run_phase, phase_steps
    from evaluation import autoattack
    from run_campaign import verify_done, verify_completed_phases
    rows = []
    for seed in SEEDS:
        sd = ROOT / f'seed{seed}'
        sd.mkdir(exist_ok=True)
        old = BASE / f'seed{seed}'
        for name in ('splits.npz', 'targets.json'):
            if (sd / name).exists():
                require(sha(sd / name) == sha(old / name), 'Copied seed metadata differs')
            else:
                shutil.copyfile(old / name, sd / name)
        with np.load(sd / 'splits.npz', allow_pickle=False) as a:
            initial, held = a['initial'].copy(), a['holdout'].copy()
        targets = read(sd / 'targets.json')['targets']
        init_summary = read(old / 'shared_initial/summary.json')
        initial_cp = old / 'shared_initial/state.pt'
        initial_hash = init_summary['checkpoint_sha256']
        require(sha(initial_cp) == initial_hash, 'Shared initialization changed')
        total_windows = sum(math.ceil(phase_steps(len(initial) + p * cfg['acquisition_per_round'], p, cfg)[0] / cfg['control_interval']) for p in range(1, 6))
        mb = digest({'binding': binding, 'seed': seed, 'method': METHOD, 'split': sha(sd / 'splits.npz'), 'targets': sha(sd / 'targets.json')})
        out = sd / NAME
        out.mkdir(exist_ok=True)
        if not verify_done(out, mb):
            features, feature_record = feature_cache(cfg, cat, seed, initial_cp, initial_hash, device, binding)
            source, labeled = initial_cp, initial.copy()
            rounds, acquisitions = [init_summary], []
            for phase in range(1, 6):
                progress(f'BASELINE RUN seed {seed}: acquisition {phase}/5, then matched task training')
                model = Model(cfg['num_classes']).to(device).float().to(memory_format=torch.channels_last)
                state = torch.load(source, map_location='cpu', weights_only=False)
                model.load_state_dict(state['model'], strict=True)
                del state
                source_hash = sha(source)
                chosen, info = query_round(model, cat, labeled, held, seed, phase, cfg, device, out / 'acquisitions' / f'round{phase}', mb, source_hash, features, progress)
                require(sha(source) == source_hash, 'Acquisition changed its source checkpoint')
                del model
                gc.collect()
                torch.cuda.empty_cache()
                labeled = np.sort(np.concatenate([labeled, chosen]))
                require(len(np.unique(labeled)) == len(initial) + phase * cfg['acquisition_per_round'] and not np.intersect1d(labeled, held).size, 'Annotation accounting failed')
                acquisitions.append(info)
                progress(f'SCIENTIFIC TRAINING: UHerding-Margin seed {seed}, phase {phase}/5')
                met, source = run_phase(cfg, cat, seed, METHOD, phase, labeled, held, source, targets, out / 'rounds' / f'round{phase}', mb, total_windows, progress, device)
                rounds.append(met)
            del features
            progress(f'FINAL EVALUATION: UHerding-Margin seed {seed}, full standard AutoAttack 8/255')
            model = Model(cfg['num_classes']).to(device).float().to(memory_format=torch.channels_last)
            state = torch.load(source, map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'], strict=True)
            del state
            model.requires_grad_(False)
            cp_hash = sha(source)
            robust = autoattack(model, cat, cfg, device, out / 'autoattack', mb, cp_hash, progress)
            require(sha(source) == cp_hash, 'Attack changed its checkpoint')
            del model
            gc.collect()
            torch.cuda.empty_cache()
            summary = {'binding': mb, 'seed': seed, 'method': NAME, 'role': 'supplementary_recent_baseline_matched_protocol_not_authors_recipe', 'rounds': rounds, 'acquisitions': acquisitions, 'autoattack': robust, 'final_checkpoint_sha256': cp_hash, 'per_pass_fraction': .1, 'learned_acquisition_head': False, 'regularizer': 'none', 'feature_extraction': feature_record, 'reused_initialization': str(initial_cp)}
            atom_json(out / 'summary.json', summary)
            atom_json(out / 'DONE.json', {'status': 'COMPLETE', 'binding': mb, 'payload': {'summary.json': sha(out / 'summary.json'), 'autoattack/predictions.npz': sha(out / 'autoattack/predictions.npz')}, 'time': utc()})
        require(verify_done(out, mb), 'Additional baseline did not complete')
        verify_completed_phases(out)
        summary = read(out / 'summary.json')
        for phase in range(1, 6):
            pd = out / 'rounds' / f'round{phase}'
            met = summary['rounds'][phase]
            audit.check_evaluation(pd / 'validation.npz', met['validation'], cat.val_labels, np.arange(len(cat.val_labels)), cfg)
            audit.check_evaluation(pd / 'feedback.npz', met['feedback'], cat.labels[held], held, cfg)
            require(met['learned_updates_cumulative'] == 0, 'Unexpected controller updates in an unregularized baseline')
        v = summary['rounds'][-1]['validation']
        r = audit.check_attack(out / 'autoattack', summary, summary['rounds'][-1], cat.val_labels, cfg, mb)
        rows.append({'seed': seed, 'method': NAME, 'clean_accuracy': v['accuracy'], 'robust_accuracy': r, 'cross_entropy': v['ce'], 'ece': v['ece'], 'mean_log_kappa': v['mean_log_kappa'], 'training_phase_seconds_excluding_shared_initial': sum(p['phase_training_elapsed_seconds'] for p in summary['rounds'][1:]), 'training_compute_seconds_excluding_shared_initial': sum(p['train_compute_seconds'] for p in summary['rounds'][1:]), 'feedback_seconds_excluding_shared_initial': sum(p['feedback_seconds'] for p in summary['rounds'][1:]), 'acquisition_seconds': sum(a['elapsed_seconds'] for a in summary['acquisitions']), 'fixed_feature_extraction_seconds': summary['feature_extraction']['elapsed_seconds'], 'autoattack_seconds': summary['autoattack']['elapsed_s']})
        put(ROOT / 'METRICS.json', rows)
        progress(f'BASELINE METHOD COMPLETE: {len(rows)}/6; seed {seed}')
        report('PARTIAL: ' + str(len(rows)) + '/6 baseline methods; no final inference')
    comparisons = audit.comparisons(reference_rows + rows, [[NAME, 'entropy'], ['entropy_aligned_lite', NAME]], ['clean_accuracy'])
    put(ROOT / 'COMPARISONS.json', comparisons)
    return rows




def worker():
    global np, torch
    import fcntl
    lock = (ROOT / 'RUN.lock').open('a')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('A worker already holds the lock; no duplicate started.')
        return 0
    try:
        import numpy as numpy
        import torch as torch_module
        np, torch = numpy, torch_module
        progress('PRECHECK: verify completed inputs and copy the frozen source; no training yet')
        cfg, source_files, env, pins, audit = frozen_inputs()
        require(torch.cuda.is_available() and 'L40S' in torch.cuda.get_device_name(0), 'Expected NVIDIA L40S is unavailable')
        require(str(torch.__version__).startswith('2.6.') and torch.version.cuda == '12.4', 'Use the existing PyTorch 2.6/CUDA 12.4 environment; do not reinstall')
        require(importlib.metadata.version('torchvision').startswith('0.21.'), 'Expected original torchvision 0.21')
        for key, value in (('torch', str(torch.__version__)), ('cuda', str(torch.version.cuda)), ('torchvision', importlib.metadata.version('torchvision'))):
            if env.get(key) is not None:
                require(str(env[key]) == value, 'Original environment differs: ' + key)
        from run_campaign import verify_autoattack, gpu_contract
        verify_autoattack(cfg['autoattack']['commit'])
        import autoattack as aa_package
        aa_dir = Path(aa_package.__file__).resolve().parent
        installed = {p.relative_to(aa_dir).as_posix(): sha(p) for p in sorted(aa_dir.rglob('*.py'))}
        require(installed == read(FOCUS / 'AUTOATTACK_SOURCE.json')['files'], 'Installed AutoAttack source differs from the completed supplementary run')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        from self_test import run as original_tests
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                originals = original_tests()
        finally:
            text_out(ROOT / 'original_cpu_tests.log', output.getvalue())
        own_cpu = new_tests(torch.device('cpu'))
        integration = integration_test(cfg, audit)
        device = torch.device('cuda:0')
        own_gpu = new_tests(device)
        put(ROOT / 'SELF_TEST.json', {'status': 'PASS', 'original_cpu_checks': len(originals), 'new_cpu_checks': own_cpu, 'new_integration_checks': integration, 'new_cuda_checks': own_gpu, 'scientific_training_counted': False, 'lazy_verified_against_exhaustive': True, 'authoring_runtime_execution': False})
        progress('CPU and new-selector CUDA checks passed. Running the original live GPU contract.')
        gpu_contract(cfg, ROOT, device)
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        from support import Catalog
        cat = Catalog(cfg)
        checked = cat.verify_images(progress)
        put(ROOT / 'DATA_VERIFIED.json', {'n': checked, 'manifest': cfg['expected_image_manifest_sha256'], 'time': utc()})
        refs = references(cfg, cat, audit)
        put(ROOT / 'REFERENCE_METRICS.json', refs)
        new_cfg = copy.deepcopy(cfg)
        new_cfg.update(campaign=SESSION, result_root=str(ROOT), archive_root=str(ROOT), methods=[METHOD], role='supplementary_recent_baseline_shared_initializations')
        new_cfg['statistics'] = {'alpha': .05, 'primary_endpoints': ['clean_accuracy'], 'comparisons': [[NAME, 'entropy'], ['entropy_aligned_lite', NAME]], 'test': 'paired t and exact sign flip; Holm within these TWO new comparisons only; earlier families unchanged'}
        plan = {'paper': 'Uncertainty Herding: One Active Learning Method for All Label Budgets, ICLR 2025', 'paper_url': 'https://arxiv.org/abs/2412.20644', 'author_code_reference': 'https://github.com/BorealisAI/uherding', 'implementation': 'independent implementation of weighted RBF-coverage greedy selection; no author code executed', 'replaces_tentative_candidate': 'AutoAL was only proposed and was not run', 'fixed_feature_map': 'per-seed shared-initial ResNet-50 normalized layer4; NO separately pretrained self-supervised model', 'acquisition_weights': '1 - (highest_probability - second_highest_probability), temperature-scaled', 'temperature': 'minimum 15-bin ECE on budgeted 633-image holdout; 81 log-spaced temperatures in [0.1,10]', 'radius': 'minimum positive Euclidean distance among all currently annotated fixed features; squared distances <=1e-12 treated as numerical zero', 'kernel': 'exp(-squared_distance/sigma_squared)', 'numerical_policy': 'float64 kernel, distance and gain computations; lazy greedy verified against exhaustive greedy', 'matching_choices': ['fixed initial task features instead of the paper benchmark SSL features', 'CAENL candidate-pass convention: at most 10000, 10 percent per pass until exact quota', 'CAENL warm-started ResNet-50 training with frozen BatchNorm after shared initialization, not authors cold-start benchmarks', 'same physical batches, optimization schedule, targets for logging, label totals, final validation and attacks', 'holdout labels used only for temperature selection; not optimizer updates', 'regularization OFF for this external comparator'], 'claims_excluded': ['reproduction of authors published benchmark scores', 'new independent seeds for previous conditions', 'universal state-of-the-art superiority', 'completed audio or diffusion experiments'], 'seeds': SEEDS, 'primary_endpoint': 'final clean_accuracy', 'new_contrasts_only': [[NAME, 'entropy'], ['entropy_aligned_lite', NAME]], 'statistics': 'paired t and all 64 exact sign flips; separate Holm corrections across these TWO new contrasts only; individual unadjusted intervals', 'design_timing': 'specified after earlier results, before any new baseline training; supplementary/post-hoc', 'robustness': 'unchanged full-set standard 8/255 endpoint; descriptive external-baseline result', 'original_binding': OLD_BINDING, 'previous_supplement_binding': FOCUS_BINDING, 'source_files': source_files, 'runner_sha256': sha(ROOT / 'runner.py'), 'input_files_digest': digest(pins)}
        binding = digest({'protocol': new_cfg, 'plan': plan})
        record = {'binding': binding, 'protocol': new_cfg, 'plan': plan}
        if (ROOT / 'RUN_BINDING.json').exists():
            require(read(ROOT / 'RUN_BINDING.json') == record, 'New protocol/source binding changed; refusing mixed results')
        else:
            put(ROOT / 'RUN_BINDING.json', record)
        report('CHECKS PASSED; baseline training not yet complete')
        rows = run_study(new_cfg, cat, binding, device, audit, refs)
        require(len(rows) == 6, 'Baseline study incomplete')
        progress('Final read-only preservation check of the bound original inputs')
        check_pins(pins)
        put(ROOT / 'PRESERVATION.json', {'status': 'PASS', 'bound_input_files_rechecked': len(pins), 'original_runs_repeated': 0, 'datasets_deleted': False, 'scope': 'hashes of frozen source, manifests, reused initial checkpoints/splits/targets and comparator final records; not a fresh audit of every historical binary'})
        target = report('COMPLETE: six new UHerding matched-protocol branches and 30 training phases')
        put(ROOT / 'COMPLETE.json', {'status': 'COMPLETE', 'time': utc(), 'binding': binding, 'method_runs': 6, 'training_phases': 30, 'original_runs_repeated': 0, 'report': str(target), 'report_sha256': sha(target), 'paper_ready': False})
        put(ROOT / 'STATUS.json', {'status': 'COMPLETE', 'time': utc(), 'message': 'UHerding BASELINE COMPLETE'})
        print('UHerding BASELINE COMPLETE\nShare: ' + str(target), flush=True)
        return 0
    except BaseException as exc:
        put(ROOT / 'FAILED.json', {'status': 'FAILED', 'time': utc(), 'error': repr(exc), 'traceback': traceback.format_exc(), 'original_results_deleted': False, 'instruction': 'Share --status output. Do not reinstall dependencies or restart original campaigns.'})
        try:
            report('STOPPED WITH ERROR; no complete-run claim')
        except Exception:
            pass
        traceback.print_exc()
        return 1
    finally:
        lock.close()




def show_status():
    print('SERVER:', os.uname().nodename, utc())
    if (ROOT / 'COMPLETE.json').exists():
        print('UHerding BASELINE COMPLETE')
        print(json.dumps(read(ROOT / 'COMPLETE.json'), indent=2))
    elif (ROOT / 'FAILED.json').exists():
        print('STOPPED WITH ERROR -- do not launch a duplicate or reinstall packages')
        print(json.dumps(read(ROOT / 'FAILED.json'), indent=2))
    elif (ROOT / 'STATUS.json').exists():
        alive = shutil.which('tmux') and subprocess.run(['tmux', 'has-session', '-t', SESSION], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        print('RUNNING' if alive else 'NO LIVE TMUX SESSION -- last recorded state follows')
        print(json.dumps(read(ROOT / 'STATUS.json'), indent=2))
    else:
        print('Not started yet.')
    print('Completed baseline method runs (target 6):', len(list(ROOT.glob('seed*/' + NAME + '/DONE.json'))))
    print('Completed baseline training phases (target 30):', len(list(ROOT.glob('seed*/' + NAME + '/rounds/round*/DONE.json'))))
    print('Report:', ROOT / REPORT)
    print('Log:', ROOT / 'run.log')
    if shutil.which('nvidia-smi'):
        subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', '--format=csv'], check=False)




def start(resume=False):
    require(sys.platform.startswith('linux'), 'Run --start on IBM, not directly on the Mac')
    require(Path('/mnt/caenl').is_mount(), 'Persistent /mnt/caenl volume is not mounted; no mounting or formatting attempted')
    require(shutil.which('tmux'), 'Existing tmux is not available')
    require((BASE / 'COMPLETE.json').is_file() and (FOCUS / 'COMPLETE.json').is_file(), 'Completed original and supplementary campaign folders are required')
    require(not ROOT.is_symlink(), 'Refusing a symlink result folder')
    ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    import fcntl
    with (ROOT / 'START.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if (ROOT / 'COMPLETE.json').exists():
            show_status()
            return
        if subprocess.run(['tmux', 'has-session', '-t', SESSION], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            print('ALREADY RUNNING. No duplicate was started.')
            show_status()
            return
        require(resume or not (ROOT / 'FAILED.json').exists(), 'A previous attempt failed. Share --status output before resuming.')
        candidates = [Path.home() / 'caenl-v5.4.3/caenl_revision_pipeline_v5.4.3_c4_confirmatory/.venv/bin/python', Path.home() / 'caenl-src/caenl_revision_pipeline_v5.3_ibm_rhel_l40s/.venv/bin/python']
        python = next((p for p in candidates if p.is_file()), None)
        require(python is not None, 'Original Python environment not found; no installation attempted')
        source = Path(__file__).read_bytes()
        ast.parse(source.decode('utf-8-sig'))
        target = ROOT / 'runner.py'
        if target.exists():
            require(target.read_bytes() == source, 'A different launcher already owns this result folder; no overwrite performed')
        else:
            target.write_bytes(source)
        require(shutil.disk_usage(ROOT).free >= 40 * 1024 ** 3, 'At least 40 GiB free space required; no cleanup performed')
        query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        require(not query.stdout.strip(), 'Another GPU process is active. It was not stopped; wait before starting this run')
        if resume and (ROOT / 'FAILED.json').exists():
            os.replace(ROOT / 'FAILED.json', ROOT / ('failure_before_resume_' + str(time.time_ns()) + '.json'))
        command = 'exec env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 ' + shlex.quote(str(python)) + ' ' + shlex.quote(str(target)) + ' --worker >> ' + shlex.quote(str(ROOT / 'run.log')) + ' 2>&1'
        subprocess.run(['tmux', 'new-session', '-d', '-s', SESSION, command], check=True)
        print('STARTED:', SESSION)
        print('First: integrity, CPU/CUDA and dataset checks. Then: six NEW baseline branches.')
        print('This is UHerding-Margin with fixed initial ResNet features and the existing CAENL training protocol, NOT an authors-recipe reproduction.')
        print('Status: python3 ~/caenl_uherding_v1.py --status')
        print('No original experiment is rerun and no package is installed.')




def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--start', action='store_true')
    group.add_argument('--resume', action='store_true')
    group.add_argument('--status', action='store_true')
    group.add_argument('--worker', action='store_true')
    args = p.parse_args()
    if args.worker:
        return worker()
    if args.status:
        show_status()
    else:
        start(resume=args.resume)
    return 0




if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print('STOPPED:', str(exc), file=sys.stderr)
        raise SystemExit(1)
# END OF CAENL UHerding LAUNCHER V1