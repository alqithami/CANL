#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CAENL entropy + fixed aligned DCR follow-up v1.
Read-only audit of the completed 54 method-seed runs; six additional entropy +
fixed aligned DCR branches in a DIFFERENT result tree. No installation,
download, deletion, modification of original source, or public upload.
Local numerical and orchestration tests accompany this script.
Real CPU integration, dataset and GPU checks run on IBM before training.
"""
from __future__ import annotations
import argparse
import ast
import copy
import csv
import gc
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import re
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
LITE_ROOT = Path('/mnt/caenl/active/results/caenl-focused-completion-v1')
ROOT = Path('/mnt/caenl/active/results/caenl-entropy-fixed-v1')
SESSION = 'caenl-entropy-fixed-v1'
REPORT_NAME = 'caenl-entropy-fixed-v1.md'
EXPECTED_BINDING = '159fe52901f774e7abeb40a3afe96de81f1084a1bde0f63d070c7d2ecda325e1'
EXPECTED_PROTOCOL = '9dfd0201a5d69d05dd8992ee9d642bb43c3a5d3b0b07b51c28ab03bd3c55e08e'
SEEDS = list(range(801, 807))
LITE_BINDING = 'c6009897fdc2d87df4e257cd89e710bbb2465d6f80a98e4349e36ea01f2af058'
METHOD = {'name': 'entropy_aligned_fixed', 'acquisition': 'entropy', 'regularizer': 'aligned_fixed'}
CONTRASTS = [['entropy_aligned_fixed', 'entropy'], ['entropy_aligned_lite', 'entropy_aligned_fixed']]
np = None




def utc():
    return datetime.now(timezone.utc).isoformat()




def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)




def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()




def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()




def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))




def under(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe relative path: ' + str(relative))
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()), 'Path escapes its source tree: ' + str(path))
    return path




def text_out(path, text):
    path = Path(path)
    require(path.resolve().is_relative_to(ROOT.resolve()), 'Refusing to write outside the NEW result tree: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)




def put(path, obj):
    text_out(path, json.dumps(obj, indent=2, allow_nan=False) + '\n')




def progress(message):
    put(ROOT / 'STATUS.json', {'status': 'RUNNING', 'pid': os.getpid(), 'time': utc(), 'message': message})
    print(message, flush=True)




def arrays(path):
    with np.load(path, allow_pickle=False) as payload:
        return {k: payload[k] for k in payload.files}




def close(actual, expected, label, tolerance=1e-8):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)), 'Nonfinite metric: ' + label)
    require(abs(float(actual) - float(expected)) <= tolerance,
            f'{label}: recomputed {actual!r}, recorded {expected!r}, tolerance {tolerance}')




def metric_numpy(logits, labels):
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    require(x.ndim == 2 and y.shape == (len(x),) and len(y) > 0, 'Invalid logit/label dimensions')
    require(np.isfinite(x).all() and (y >= 0).all() and (y < x.shape[1]).all(), 'Invalid logits/labels')
    shifted = x - x.max(axis=1, keepdims=True)
    log_denom = np.log(np.exp(shifted).sum(axis=1))
    probabilities = np.exp(shifted - log_denom[:, None])
    pred = x.argmax(axis=1)
    conf = probabilities[np.arange(len(y)), pred]
    good = pred == y
    bins = np.minimum((conf * 15).astype(np.int64), 14)
    ece = sum(abs(float(good[bins == b].sum()) - float(conf[bins == b].sum())) / len(y)
              for b in range(15) if np.any(bins == b))
    true_logits = x[np.arange(len(y)), y][:, None]
    greater = (x > true_logits).sum(axis=1)
    equal = (x == true_logits).sum(axis=1)
    k = min(5, x.shape[1])
    return {'n': len(y), 'accuracy': float(good.mean()),
            'ce': float((log_denom - shifted[np.arange(len(y)), y]).mean()), 'ece': float(ece),
            'top5_low': float((greater + equal <= k).mean()),
            'top5_high': float((greater < k).mean()),
            'top5_boundary_ties': int(((greater < k) & (greater + equal > k)).sum())}




def check_evaluation(path, summary, expected_labels, expected_ids, cfg):
    a = arrays(path)
    require(np.array_equal(a['indices'], expected_ids), 'Evaluation IDs/order differ: ' + str(path))
    require(np.array_equal(a['labels'], expected_labels), 'Evaluation labels differ: ' + str(path))
    require(a['logits'].dtype == np.float32, 'Expected stored float32 logits: ' + str(path))
    require(a['logits'].shape == (len(expected_ids), cfg['num_classes']), 'Evaluation logit shape differs')
    require(np.array_equal(a['predictions'], a['logits'].argmax(1)), 'Stored predictions differ from logits')
    m = metric_numpy(a['logits'], a['labels'])
    require(m['n'] == summary['n'], 'Evaluation count differs')
    for k in ('accuracy', 'ce', 'ece'):
        close(m[k], summary[k], str(path) + ':' + k)
    require(m['top5_low'] - 1e-12 <= summary['top5_accuracy'] <= m['top5_high'] + 1e-12,
            'Top-5 is incompatible with saved logits: ' + str(path))
    ratios = []
    for layer in cfg['layers']:
        counts = a[layer + '_class_counts']
        require(np.array_equal(counts, np.bincount(a['labels'], minlength=cfg['num_classes'])), 'Class-count mismatch')
        valid = counts > 0
        sums = np.asarray(a[layer + '_class_sums'], dtype=np.float64)[valid]
        sumsq = np.asarray(a[layer + '_class_sumsq'], dtype=np.float64)[valid]
        require(np.isfinite(sums).all() and np.isfinite(sumsq).all(), 'Nonfinite moment payload')
        count = counts[valid, None]
        means = sums / count
        per_class_w = (sumsq / count - means ** 2).sum(axis=1)
        require(per_class_w.min() >= -1e-9, 'Negative within-class scatter beyond rounding')
        w = float(per_class_w.mean())
        b = float(((means - means.mean(axis=0)) ** 2).sum(axis=1).mean())
        eps = cfg['alignment']['scatter_epsilon']
        require(w + eps > 0 and b + eps > 0, 'Invalid stabilized scatter ratio')
        q = math.log(w + eps) - math.log(b + eps)
        vals = {'within': w, 'between': b, 'total': w + b, 'log_kappa': q, 'kappa': math.exp(q)}
        for k, v in vals.items():
            close(v, summary[layer + '_' + k], str(path) + ':' + layer + '_' + k)
        ratios.append(q)
    m['mean_log_kappa'] = float(np.mean(ratios))
    close(m['mean_log_kappa'], summary['mean_log_kappa'], 'Mean log ratio')
    return m




def paired(a, b):
    from scipy import stats
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    require(len(d) == 6 and np.isfinite(d).all(), 'Six finite seed pairs are required')
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd < 1e-15:
        p = 1.0 if abs(mean) < 1e-15 else None
        lo = hi = 0.0 if abs(mean) < 1e-15 else None
    else:
        se = sd / math.sqrt(len(d))
        half = float(stats.t.ppf(.975, 5)) * se
        lo, hi = mean - half, mean + half
        p = float(2 * stats.t.sf(abs(mean / se), 5))
    null = [abs(float((d * np.asarray(s)).mean())) for s in itertools.product((-1.0, 1.0), repeat=6)]
    return {'n_pairs': 6, 'mean_difference': mean, 'ci95_low': lo, 'ci95_high': hi,
            'paired_t_p': p, 'exact_signflip_p': float(np.mean(np.asarray(null) >= abs(mean) - 1e-15)),
            'target_higher_count': int((d > 0).sum()), 'target_lower_count': int((d < 0).sum())}




def holm(values):
    order = sorted(range(len(values)), key=lambda i: 1.0 if values[i] is None else values[i])
    out = [None] * len(values)
    bound = 0.0
    for rank, index in enumerate(order):
        p = 1.0 if values[index] is None else values[index]
        bound = max(bound, min(1.0, (len(values) - rank) * p))
        if values[index] is not None:
            out[index] = bound
    return out




def comparisons(rows, contrasts, metrics):
    mapping = {(r['seed'], r['method']): r for r in rows}
    result = []
    for metric in metrics:
        group = []
        for target, ref in contrasts:
            row = {'metric': metric, 'target': target, 'reference': ref}
            row.update(paired([mapping[s, target][metric] for s in SEEDS], [mapping[s, ref][metric] for s in SEEDS]))
            group.append(row)
        for row, pt, pe in zip(group, holm([r['paired_t_p'] for r in group]), holm([r['exact_signflip_p'] for r in group])):
            row['paired_t_p_holm'] = pt
            row['exact_signflip_p_holm'] = pe
        result.extend(group)
    return result




def check_attack(directory, summary, final_summary, labels, cfg, method_binding, manifest=None):
    directory = Path(directory)
    path = directory / 'predictions.npz'
    aa = summary['autoattack']
    require(sha(path) == aa['predictions_sha256'], 'Final attack payload digest mismatch')
    a = arrays(path)
    n = len(labels)
    require(np.array_equal(a['indices'], np.arange(n)) and np.array_equal(a['labels'], labels), 'Attack ID/label mismatch')
    for logit_key, pred_key in (('clean_logits', 'clean_predictions'), ('adversarial_logits', 'adversarial_predictions')):
        require(a[logit_key].shape == (n, cfg['num_classes']) and np.isfinite(a[logit_key]).all(), 'Invalid attack logits')
        require(np.array_equal(a[logit_key].argmax(1), a[pred_key]), 'Attack logits/predictions disagree')
    robust = (a['clean_predictions'] == labels) & (a['adversarial_predictions'] == labels)
    require(np.array_equal(robust, a['robust_mask']), 'Robustness mask arithmetic differs')
    require(a['linf'].shape == (n,) and np.isfinite(a['linf']).all(), 'Invalid recorded perturbation norms')
    require(a['linf'].min() >= 0 and a['linf'].max() <= cfg['autoattack']['epsilon'] + 2e-6, 'Recorded norm bound violated')
    close(float(robust.mean()), aa['robust_accuracy'], 'Robust accuracy')
    close(float((a['clean_predictions'] == labels).mean()), aa['clean_accuracy'], 'Attack clean accuracy')
    close(aa['clean_accuracy'], final_summary['validation']['accuracy'], 'Clean evaluation/attack accuracy')
    require(int(robust.sum()) == aa['robust_correct'] and aa['n'] == n, 'Robust count mismatch')
    require(aa['suite'] == 'standard' and aa['norm'] == 'Linf', 'Unexpected attack suite/norm')
    close(aa['epsilon'], 8 / 255, 'Attack epsilon')
    require(aa['checkpoint_sha256'] == summary['final_checkpoint_sha256'] == final_summary['checkpoint_sha256'], 'Attack checkpoint linkage mismatch')
    chunk_size = cfg['autoattack']['chunk_size']
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        cp = directory / f'chunk_{start:05d}.npz'
        record = read(cp.with_suffix('.json'))
        expected = digest({'binding': method_binding, 'checkpoint': summary['final_checkpoint_sha256'], 'start': start, 'end': end, 'aa': cfg['autoattack']})
        require(record['binding'] == expected and sha(cp) == record['sha256'], 'Attack chunk binding/digest mismatch')
        block = arrays(cp)
        require(set(block) == set(a), 'Unexpected attack chunk fields')
        for key in a:
            require(np.array_equal(block[key], a[key][start:end]), 'Chunk/final concatenation differs: ' + key)
        require(record['n'] == end - start and record['robust_correct'] == int(robust[start:end].sum()), 'Attack chunk count differs')
    return float(robust.mean())




def verify_manifest():
    manifest = {}
    lines = (BASE / 'SHA256SUMS').read_text().splitlines()
    for index, line in enumerate(lines, 1):
        expected, rel = line.split('  ', 1)
        require(re.fullmatch('[0-9a-f]{64}', expected), 'Malformed archived hash')
        path = under(BASE, rel)
        require(rel not in manifest and path.is_file(), 'Duplicate/missing manifest member: ' + rel)
        require(sha(path) == expected, 'Original artifact checksum mismatch: ' + rel)
        manifest[rel] = expected
        if index % 250 == 0:
            progress(f'Audit: checked {index}/{len(lines)} original archived payload hashes')
    return manifest




def verify_payload(directory, payload, manifest=None, include_checkpoints=False):
    for rel, expected in payload.items():
        path = under(Path(directory), rel)
        if path.suffix in ('.pt', '.pth', '.ckpt') and not include_checkpoints:
            continue
        if manifest is not None:
            key = path.relative_to(BASE).as_posix()
            require(manifest.get(key) == expected, 'DONE/manifest hashes differ: ' + key)
        else:
            require(path.is_file() and sha(path) == expected, 'Payload mismatch: ' + str(path))




def prepare_source():
    require(BASE.is_dir(), 'Original result directory is missing: ' + str(BASE))
    complete = read(BASE / 'COMPLETE.json')
    require(complete.get('status') == 'COMPLETE' and complete.get('methods_complete') == 54, 'Original campaign is not marked complete')
    inp = read(BASE / 'INPUT_BINDING.json')
    cfg, source_files = inp['protocol'], inp['source_files']
    require(inp['binding'] == EXPECTED_BINDING == complete['binding'], 'Unexpected original protocol binding')
    require(digest({'protocol': cfg, 'source_files': source_files}) == EXPECTED_BINDING, 'Original binding cannot be recomputed')
    require(sha(BASE / 'source/protocol.json') == EXPECTED_PROTOCOL, 'Frozen protocol file differs from reviewed source')
    require(read(BASE / 'source/protocol.json') == cfg, 'Recorded/source protocol differs')
    require(cfg['seeds'] == SEEDS and len(cfg['methods']) == 9, 'Unexpected original matrix')
    destination = ROOT / 'frozen_source'
    destination.mkdir(parents=True, exist_ok=True)
    for rel, expected in source_files.items():
        old, new = under(BASE / 'source', rel), under(destination, rel)
        require(old.is_file() and sha(old) == expected, 'Frozen source mismatch: ' + rel)
        if new.exists():
            require(sha(new) == expected, 'The isolated source snapshot changed: ' + rel)
        else:
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(old, new)
    text_out(destination / 'SHA256SUMS', ''.join(f'{h}  {p}\n' for p, h in sorted(source_files.items())))
    return cfg, source_files, inp.get('environment', {})




def audit_original(cfg, manifest, cat):
    from support import stratified_sample, seed_for
    rows, curves, reusable = [], [], {}
    for seed in SEEDS:
        sd = BASE / f'seed{seed}'
        split = arrays(sd / 'splits.npz')
        selected = stratified_sample(np.arange(len(cat.labels)), cat.labels, cfg['initial_labels'], seed_for('initial_labels', seed))
        held = stratified_sample(selected, cat.labels, cfg['holdout_count'], seed_for('controller_holdout', seed))
        initial = np.setdiff1d(selected, held)
        for key, expected in (('initial', initial), ('holdout', held), ('annotated_initial', selected)):
            require(np.array_equal(split[key], expected), 'Seed split differs: ' + str(seed) + '/' + key)
        init_dir = sd / 'shared_initial'
        init_summary = read(init_dir / 'summary.json')
        verify_payload(init_dir, read(init_dir / 'DONE.json')['payload'], manifest)
        source = init_dir / 'state.pt'
        progress(f'Audit: verifying reusable initialization for seed {seed}')
        require(source.is_file() and sha(source) == init_summary['checkpoint_sha256'], 'Reusable initial checkpoint missing/changed: ' + str(source))
        targets = read(sd / 'targets.json')
        require(targets['initial_checkpoint_sha256'] == init_summary['checkpoint_sha256'], 'Target reference linkage differs')
        check_evaluation(init_dir / 'validation.npz', init_summary['validation'], cat.val_labels, np.arange(len(cat.val_labels)), cfg)
        check_evaluation(init_dir / 'feedback.npz', init_summary['feedback'], cat.labels[held], held, cfg)
        reusable[str(seed)] = {str(source): sha(source), str(sd / 'targets.json'): sha(sd / 'targets.json'), str(sd / 'splits.npz'): sha(sd / 'splits.npz')}
        for method in cfg['methods']:
            name = method['name']
            out = sd / name
            mb = digest({'binding': EXPECTED_BINDING, 'seed': seed, 'method': method, 'split': sha(sd / 'splits.npz'), 'targets': sha(sd / 'targets.json')})
            done, summary = read(out / 'DONE.json'), read(out / 'summary.json')
            require(done['binding'] == mb == summary['binding'] and done['status'] == 'COMPLETE', 'Method completion binding mismatch')
            verify_payload(out, done['payload'], manifest)
            require(summary['seed'] == seed and summary['method'] == name and len(summary['rounds']) == 6, 'Method summary identity differs')
            require(summary['rounds'][0] == init_summary, 'Shared initialization summary differs')
            labeled, previous_hash = initial.copy(), init_summary['checkpoint_sha256']
            for phase in range(1, 6):
                acq = out / 'acquisitions' / f'round{phase}'
                record = read(acq / 'COMPLETE.json')
                chosen = arrays(acq / 'selection.npz')
                verify_payload(acq, {'selection.npz': record['selection_sha256']}, manifest)
                ab = digest({'binding': mb, 'checkpoint': previous_hash, 'labeled': labeled.tolist(), 'holdout': held.tolist(), 'seed': seed, 'phase': phase, 'method': method})
                require(record['binding'] == ab, 'Acquisition binding mismatch')
                require(np.array_equal(chosen['previous_labeled'], labeled) and np.array_equal(chosen['control_holdout'], held), 'Acquisition previous-state mismatch')
                new = chosen['selected']
                require(len(new) == cfg['acquisition_per_round'] and len(np.unique(new)) == len(new), 'Acquisition count/duplicates')
                require((new >= 0).all() and (new < len(cat.labels)).all() and not np.intersect1d(new, np.concatenate([labeled, held])).size, 'Acquisition leakage or overlap')
                labeled = np.sort(np.concatenate([labeled, new]))
                pd = out / 'rounds' / f'round{phase}'
                ps, p_done = read(pd / 'summary.json'), read(pd / 'DONE.json')
                verify_payload(pd, p_done['payload'], manifest)
                pb = digest({'campaign': mb, 'seed': seed, 'method': name, 'phase': phase, 'source': previous_hash, 'labeled': labeled.tolist(), 'control': held.tolist(), 'targets': targets['targets']})
                require(ps['binding'] == pb == p_done['binding'], 'Phase binding mismatch')
                require(ps == summary['rounds'][phase], 'Phase/method summaries differ')
                expected_steps = math.ceil(len(labeled) / 128) * cfg['round_epochs']
                require(ps['phase'] == phase and ps['steps'] == expected_steps and ps['gradient_training_count'] == len(labeled), 'Training budget differs')
                require(ps['annotation_count'] == len(labeled) + len(held), 'Annotation accounting differs')
                m = check_evaluation(pd / 'validation.npz', ps['validation'], cat.val_labels, np.arange(len(cat.val_labels)), cfg)
                check_evaluation(pd / 'feedback.npz', ps['feedback'], cat.labels[held], held, cfg)
                curves.append({'seed': seed, 'method': name, 'phase': phase, 'annotations': ps['annotation_count'], 'accuracy': m['accuracy'], 'ce': m['ce'], 'mean_log_kappa': m['mean_log_kappa']})
                previous_hash = ps['checkpoint_sha256']
            ra = check_attack(out / 'autoattack', summary, ps, cat.val_labels, cfg, mb)
            rows.append({'seed': seed, 'method': name, 'clean_accuracy': m['accuracy'], 'robust_accuracy': ra, 'cross_entropy': m['ce'], 'ece': m['ece'], 'mean_log_kappa': m['mean_log_kappa']})
            progress(f'Audit: independently checked saved-array arithmetic for {len(rows)}/54 original method runs')
    require(len(rows) == 54 and len(curves) == 270, 'Incomplete original matrix')
    tests = comparisons(rows, cfg['statistics']['comparisons'], cfg['statistics']['primary_endpoints'])
    with (BASE / 'reports/paired_comparisons.csv').open() as f:
        recorded = {(r['metric'], r['target'], r['reference']): r for r in csv.DictReader(f)}
    require(len(recorded) == len(tests), 'Original statistical table length differs')
    for r in tests:
        old = recorded[r['metric'], r['target'], r['reference']]
        for key in ('mean_difference', 'ci95_low', 'ci95_high', 'paired_t_p', 'exact_signflip_p', 'paired_t_p_holm', 'exact_signflip_p_holm'):
            value = old.get(key)
            if r[key] is None:
                require(value in ('', None), 'Undefined test was reported as a number')
            else:
                close(r[key], float(value), 'Original statistical comparison ' + key)
    with (BASE / 'reports/per_seed_metrics.csv').open() as f:
        recorded_rows = {(int(r['seed']), r['method']): r for r in csv.DictReader(f)}
    require(len(recorded_rows) == 54, 'Original per-seed table count differs')
    for r in rows:
        for k in ('clean_accuracy', 'robust_accuracy', 'cross_entropy', 'ece', 'mean_log_kappa'):
            close(r[k], float(recorded_rows[r['seed'], r['method']][k]), 'Original per-seed CSV: ' + k)
    grouped = []
    with (BASE / 'reports/grouped_metrics.csv').open() as f:
        old_grouped = {r['method']: r for r in csv.DictReader(f)}
    require(len(old_grouped) == 9, 'Original grouped table count differs')
    for method in cfg['methods']:
        values = [r for r in rows if r['method'] == method['name']]
        require(len(values) == 6, 'Original grouping does not have six seeds')
        group = {'method': method['name'], 'n_seeds': 6}
        for metric in ('clean_accuracy', 'robust_accuracy', 'cross_entropy', 'ece', 'mean_log_kappa'):
            data = np.asarray([r[metric] for r in values], dtype=np.float64)
            for suffix, value in (('_mean', float(data.mean())), ('_std', float(data.std(ddof=1)))):
                group[metric + suffix] = value
                close(value, float(old_grouped[method['name']][metric + suffix]), 'Original grouped CSV: ' + metric + suffix)
        grouped.append(group)
    result = {'status': 'PASS', 'time': utc(), 'original_grouped_recomputed': grouped, 'original_binding': EXPECTED_BINDING,
              'manifest_files_checked': len(manifest), 'initializations_checked': 6,
              'method_runs_checked': 54, 'AL_phases_checked': 270,
              'validation_and_feedback_payloads_checked': 552, 'final_attack_payloads_checked': 54,
              'scope': 'Independent NumPy arithmetic from saved logits, predictions and moment sufficient statistics; archive-payload and input-linkage checks. No original checkpoint inference or original attack rerun. Stored perturbation norms were checked, not regenerated from unavailable adversarial images. Later-phase checkpoint binaries are not all rehashed.',
              'original_metrics': rows, 'original_tests_recomputed': tests, 'phase_metrics': curves, 'reused_input_digests': reusable}
    put(ROOT / 'original_audit.json', result)
    return result




def check_reused(audit):
    for files in audit['reused_input_digests'].values():
        for p, h in files.items():
            require(Path(p).is_file() and sha(p) == h, 'Original reused input changed: ' + p)




def self_tests(cfg):
    from self_test import run, fixture, Tiny
    from support import stratified_sample, sha256, tensor_digest
    from control import calibrate
    from acquisition import select_round
    from training import run_phase
    count = len(run())
    close(metric_numpy([[3., 0.], [0., 3.]], [0, 0])['accuracy'], .5, 'Audit test accuracy')
    close(paired([1.] * 6, [0.] * 6)['exact_signflip_p'], .03125, 'Audit test exact sign flips')
    require(holm([.01, .04, .5]) == [.03, .08, .5], 'Audit test Holm')
    import torch
    with tempfile.TemporaryDirectory(prefix='entropy-fixed-test-', dir=ROOT) as temp:
        root = Path(temp)
        small = copy.deepcopy(cfg)
        small.update(num_classes=4, expected_train_count=144, expected_val_count=32, initial_labels=48, holdout_count=16,
                     acquisition_per_round=16, rounds=2, initial_epochs=2, round_epochs=2, batch_classes=4, samples_per_class=2,
                     workers=0, eval_workers=0, eval_batch=16, control_interval=2, calibration_batches=2, checkpoint_every_steps=2)
        small['acquisition']['candidate_size'] = 32
        cat = fixture(root / 'images', small)
        ann = stratified_sample(np.arange(144), cat.labels, 48, 4)
        held = stratified_sample(ann, cat.labels, 16, 5)
        ids = np.setdiff1d(ann, held)
        init = {'name': 'shared_initial', 'regularizer': 'none', 'acquisition': 'none'}
        _, cp = run_phase(small, cat, 801, init, 0, ids, held, None, None, root / 'init', 'extension-test', 20, lambda m: None, 'cpu', model_factory=Tiny)
        before = sha256(cp)
        model = Tiny(4)
        model.load_state_dict(torch.load(cp, map_location='cpu', weights_only=False)['model'])
        targets, _ = calibrate(model, cat, ids, 801, small, 'cpu')
        selected, _ = select_round(model, cat, ids, held, METHOD, 801, 1, small, torch.device('cpu'), root / 'selection', 'extension-test', before, lambda m: None)
        plain_entropy = {'name': 'entropy', 'acquisition': 'entropy', 'regularizer': 'none'}
        selected_ref, _ = select_round(model, cat, ids, held, plain_entropy, 801, 1, small, torch.device('cpu'), root / 'selection-ref', 'extension-test', before, lambda m: None)
        require(np.array_equal(selected, selected_ref), 'Additional-method first acquisition does not match entropy')
        labeled = np.sort(np.concatenate([ids, selected]))
        met, new_cp = run_phase(small, cat, 801, METHOD, 1, labeled, held, cp, targets, root / 'phase', 'extension-test', 20, lambda m: None, 'cpu', model_factory=Tiny)
        state = torch.load(new_cp, map_location='cpu', weights_only=False)
        require(state['controller']['kind'] == 'aligned_fixed', 'Additional method did not activate fixed DCR')
        require(all(abs(r[l + '_lambda'] - small['lambda_fixed']) < 1e-12 for r in state['rows'] for l in small['layers']), 'Fixed DCR coefficient changed during training')
        require(state['controller']['controller'] is None, 'Fixed DCR activated an adaptive controller')
        require(sha256(cp) == before, 'Integration test changed its input checkpoint')
        check_evaluation(root / 'phase/validation.npz', met['validation'], cat.val_labels, np.arange(len(cat.val_labels)), small)
        check_evaluation(root / 'phase/feedback.npz', met['feedback'], cat.labels[held], held, small)
        resumed, same_cp = run_phase(small, cat, 801, METHOD, 1, labeled, held, cp, targets, root / 'phase', 'extension-test', 20, lambda m: None, 'cpu', model_factory=Tiny)
        require(resumed == met and sha256(same_cp) == sha256(new_cp), 'Completed additional phase did not reuse its saved result')
    progress(f'PRECHECK PASS: {count} inherited checks plus audit and entropy+fixed CPU integration checks')
    put(ROOT / 'SELF_TEST.json', {'status': 'PASS', 'time': utc(), 'inherited_checks': count,
                                  'additional_entropy_fixed_forward_backward': True, 'saved_array_arithmetic': True,
                                  'completed_phase_reuse': True, 'original_user_data_used': False})




def run_addition(cfg, cat, audit, source_files, original_env):
    import torch
    from support import Model
    from training import run_phase, phase_steps
    from acquisition import select_round
    from evaluation import autoattack
    from run_campaign import gpu_contract, verify_autoattack, verify_done, verify_completed_phases
    require(torch.cuda.is_available() and 'L40S' in torch.cuda.get_device_name(0), 'Expected NVIDIA L40S is not available')
    require(str(torch.__version__).startswith('2.6.') and torch.version.cuda == '12.4', 'Original PyTorch 2.6/CUDA 12.4 environment is required; nothing was installed')
    require(importlib.metadata.version('torchvision').startswith('0.21.'), 'Expected original torchvision 0.21 environment')
    verify_autoattack(cfg['autoattack']['commit'])
    for key, actual in (('torch', str(torch.__version__)), ('torchvision', importlib.metadata.version('torchvision')), ('cuda', str(torch.version.cuda))):
        if original_env.get(key) is not None:
            require(str(original_env[key]) == actual, 'Installed ' + key + ' differs from the original recorded environment')
    import autoattack as autoattack_package
    aa_dir = Path(autoattack_package.__file__).resolve().parent
    aa_sources = {p.relative_to(aa_dir).as_posix(): sha(p) for p in sorted(aa_dir.rglob('*.py'))}
    if (ROOT / 'AUTOATTACK_SOURCE.json').exists():
        require(read(ROOT / 'AUTOATTACK_SOURCE.json')['files'] == aa_sources, 'Installed attack source changed since this supplementary run began')
    else:
        put(ROOT / 'AUTOATTACK_SOURCE.json', {'commit_metadata': cfg['autoattack']['commit'], 'files': aa_sources, 'note': 'Includes the existing layout fix; this script makes no dependency edits.'})
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda:0')
    from autoattack.checks import check_zero_gradients
    check_zero_gradients(torch.randn(2, 3, 4, 5, device=device).transpose(2, 3), logger=None)
    gpu_contract(cfg, ROOT, device)
    progress('GPU checks passed. Verifying existing dataset image hashes before additional training.')
    verified_images = cat.verify_images(progress)
    put(ROOT / 'DATA_VERIFIED.json', {'n': verified_images, 'time': utc()})
    new_cfg = copy.deepcopy(cfg)
    new_cfg.update(campaign=SESSION, result_root=str(ROOT), archive_root=str(ROOT), methods=[METHOD],
                   role='post_hoc_matched_ablation_using_existing_seeds_and_initial_checkpoints', no_legacy_checkpoints_loaded=False)
    new_cfg['statistics'] = {'alpha': .05, 'primary_endpoints': ['clean_accuracy'],
                             'comparisons': CONTRASTS,
                             'test': 'paired t and exact sign flip; Holm over these two supplementary clean-accuracy contrasts only; original seven-contrast family unchanged'}
    plan = {'role': new_cfg['role'], 'decided_before_additional_training': True, 'seeds': SEEDS, 'method': METHOD,
            'comparisons': new_cfg['statistics'], 'reuse': audit['reused_input_digests'],
            'original_binding': EXPECTED_BINDING, 'source_files': source_files, 'runner_sha256': sha(ROOT / 'runner.py'),
            'original_statistics_not_overwritten': True, 'original_seeds_are_not_new_independent_replications': True,
            'no_new_hyperparameter_tuning': True, 'no_new_datasets': True,
            'robustness': 'same full-set standard 8/255 endpoint, supplementary descriptive result',
            'remaining_outside_this_script': ['tuned fixed/schedule and alignment-isolation controls if stronger mechanism claims are retained', 'fresh-seed or second-setting confirmation if broader claims are retained', 'full-label reference only for a label-saving claim', 'public deposit, declarations, and final manuscript audit'],
            'lite_input_digests': audit['lite_input_digests'],
            'accuracy_units': 'fraction; multiply differences and intervals by 100 for percentage points',
            'intervals': 'individual 95 percent paired-t intervals, not simultaneous intervals',
            'exact_test_resolution': 'minimum two-sided p=0.03125; two-contrast Holm cannot reject at 0.05 with six pairs'}
    binding = digest({'protocol': new_cfg, 'plan': plan})
    if (ROOT / 'ADDITIONAL_BINDING.json').exists():
        require(read(ROOT / 'ADDITIONAL_BINDING.json')['binding'] == binding, 'Additional protocol/source changed; refusing mixed results')
    else:
        put(ROOT / 'ADDITIONAL_BINDING.json', {'binding': binding, 'protocol': new_cfg, 'plan': plan, 'created_utc': utc(),
                                             'environment': {'python': sys.version, 'torch': torch.__version__, 'torchvision': importlib.metadata.version('torchvision'), 'numpy': np.__version__, 'scipy': importlib.metadata.version('scipy'), 'cuda': torch.version.cuda, 'original_recorded_environment': original_env}})
    new_rows = []
    for seed in SEEDS:
        sd, old = ROOT / f'seed{seed}', BASE / f'seed{seed}'
        sd.mkdir(exist_ok=True)
        for name in ('splits.npz', 'targets.json'):
            if (sd / name).exists():
                require(sha(sd / name) == sha(old / name), 'Copied seed metadata differs')
            else:
                shutil.copyfile(old / name, sd / name)
        split = arrays(sd / 'splits.npz')
        initial, held = split['initial'], split['holdout']
        targets = read(sd / 'targets.json')['targets']
        init_summary = read(old / 'shared_initial/summary.json')
        initial_cp = old / 'shared_initial/state.pt'
        require(sha(initial_cp) == init_summary['checkpoint_sha256'], 'Original initialization changed')
        total_windows = sum(math.ceil(phase_steps(len(initial) + p * cfg['acquisition_per_round'], p, cfg)[0] / cfg['control_interval']) for p in range(1, 6))
        mb = digest({'binding': binding, 'seed': seed, 'method': METHOD, 'split': sha(sd / 'splits.npz'), 'targets': sha(sd / 'targets.json')})
        out = sd / METHOD['name']
        out.mkdir(exist_ok=True)
        if not verify_done(out, mb):
            source, labeled = initial_cp, initial.copy()
            rounds, acquisitions = [init_summary], []
            for phase in range(1, 6):
                progress(f'ADDITIONAL TRAINING: entropy + fixed aligned DCR, seed {seed}, acquisition {phase}/5')
                model = Model(cfg['num_classes']).to(device).float().to(memory_format=torch.channels_last)
                state = torch.load(source, map_location='cpu', weights_only=False)
                model.load_state_dict(state['model'], strict=True)
                del state
                chosen, info = select_round(model, cat, labeled, held, METHOD, seed, phase, cfg, device,
                                             out / 'acquisitions' / f'round{phase}', mb, sha(source), progress)
                if phase == 1:
                    original_first = arrays(old / 'entropy/acquisitions/round1/selection.npz')['selected']
                    require(np.array_equal(chosen, original_first), 'First acquisition differs from original entropy despite identical starting inputs; stopping before training')
                del model
                gc.collect()
                torch.cuda.empty_cache()
                labeled = np.sort(np.concatenate([labeled, chosen]))
                acquisitions.append(info)
                require(len(np.unique(labeled)) == len(initial) + phase * cfg['acquisition_per_round'] and not np.intersect1d(labeled, held).size, 'Additional annotation accounting failed')
                met, source = run_phase(cfg, cat, seed, METHOD, phase, labeled, held, source, targets,
                                         out / 'rounds' / f'round{phase}', mb, total_windows, progress, device)
                rounds.append(met)
            progress(f'ADDITIONAL EVALUATION: standard AutoAttack, all 5,000 images, seed {seed}')
            model = Model(cfg['num_classes']).to(device).float().to(memory_format=torch.channels_last)
            state = torch.load(source, map_location='cpu', weights_only=False)
            model.load_state_dict(state['model'], strict=True)
            del state
            model.requires_grad_(False)
            checkpoint_hash = sha(source)
            robust = autoattack(model, cat, cfg, device, out / 'autoattack', mb, checkpoint_hash, progress)
            require(sha(source) == checkpoint_hash, 'Additional evaluation changed its checkpoint')
            del model
            gc.collect()
            torch.cuda.empty_cache()
            summary = {'binding': mb, 'seed': seed, 'method': METHOD['name'], 'role': new_cfg['role'],
                       'rounds': rounds, 'acquisitions': acquisitions, 'autoattack': robust,
                       'final_checkpoint_sha256': checkpoint_hash, 'fixed_acquisition_threshold': .9,
                       'learned_acquisition_head': False, 'reused_initialization': str(initial_cp)}
            put(out / 'summary.json', summary)
            put(out / 'DONE.json', {'status': 'COMPLETE', 'binding': mb, 'payload': {'summary.json': sha(out / 'summary.json'), 'autoattack/predictions.npz': sha(out / 'autoattack/predictions.npz')}, 'time': utc()})
        require(verify_done(out, mb), 'Additional method is incomplete')
        verify_completed_phases(out)
        summary = read(out / 'summary.json')
        for phase in range(1, 6):
            pd = out / 'rounds' / f'round{phase}'
            met = summary['rounds'][phase]
            verify_fixed_phase(pd, met, cfg)
            check_evaluation(pd / 'validation.npz', met['validation'], cat.val_labels, np.arange(len(cat.val_labels)), cfg)
            check_evaluation(pd / 'feedback.npz', met['feedback'], cat.labels[held], held, cfg)
        v = summary['rounds'][-1]['validation']
        r = check_attack(out / 'autoattack', summary, summary['rounds'][-1], cat.val_labels, cfg, mb)
        new_rows.append({'seed': seed, 'method': METHOD['name'], 'clean_accuracy': v['accuracy'], 'robust_accuracy': r,
                         'cross_entropy': v['ce'], 'ece': v['ece'], 'mean_log_kappa': v['mean_log_kappa'],
                         'training_compute_seconds_excluding_reused_initial': sum(t['train_compute_seconds'] for t in summary['rounds'][1:]),
                         'feedback_seconds_excluding_reused_initial': sum(t['feedback_seconds'] for t in summary['rounds'][1:]),
                         'acquisition_seconds': sum(a['elapsed_seconds'] for a in summary['acquisitions']),
                         'autoattack_seconds': summary['autoattack']['elapsed_s']})
        put(ROOT / 'additional_metrics.json', new_rows)
        progress(f'ADDITIONAL METHOD COMPLETE: {len(new_rows)}/6; seed {seed}; unfavorable results are retained')
    tests = comparisons(audit['original_metrics'] + audit['lite_metrics'] + new_rows, new_cfg['statistics']['comparisons'], ['clean_accuracy'])
    put(ROOT / 'supplementary_clean_comparisons.json', tests)
    export_curves(cfg, audit, include_fixed=True)
    check_reused(audit)
    check_lite_unchanged(audit)
    return new_rows




def write_share(status):
    lines = ['# CAENL entropy + fixed aligned DCR follow-up', '', 'Status: ' + status,
             'Generated UTC: ' + utc(), '',
             'This is a later matched follow-up on seeds 801–806. Existing entropy and Lite outcomes were already known when this control was planned.',
             'It is not fresh-seed confirmation. The two new clean-accuracy contrasts are fixed minus entropy and Lite minus fixed, with Holm correction within this new family.',
             'Intervals are individual paired-t 95% intervals. Accuracy JSON/CSV values are fractions; multiply by 100 for percent or percentage points.',
             'With six pairs, the smallest two-sided exact p-value is 0.03125. Neither test in this two-contrast Holm family can reject at 0.05.',
             'GPU training and standard AutoAttack are new only for the fixed-DCR condition. Existing conditions are checked using saved arrays and artifact linkage.',
             'The six reused initial checkpoint binaries are rehashed. Existing later-phase checkpoints and original attacks are not rerun.',
             'Exported learning curves contain measured phase points for entropy, entropy+Lite, and (after completion) entropy+fixed. No interpolation is performed.', '']
    for name in ('SELF_TEST.json', 'AUDIT_COMPLETE.json', 'original_audit.json', 'lite_audit.json',
                 'ADDITIONAL_BINDING.json', 'CUDA_PREFLIGHT.json', 'AUTOATTACK_SOURCE.json',
                 'DATA_VERIFIED.json', 'additional_metrics.json', 'supplementary_clean_comparisons.json', 'FAILED.json'):
        p = ROOT / name
        if p.exists():
            lines += ['', '## ' + name, 'SHA-256: ' + sha(p), '````````', p.read_text(), '````````']
    for name in ('accuracy_by_budget.csv', 'layer_geometry_by_budget.csv', 'grouped_accuracy_by_budget.csv'):
        p = ROOT / name
        if p.exists():
            lines += ['', '## ' + name, 'SHA-256: ' + sha(p), '```csv', p.read_text(), '```']
    lines += ['', '## Launcher source', '````````', (ROOT / 'runner.py').read_text(encoding='utf-8-sig'), '````````', '', 'END OF ENTROPY FIXED REPORT']
    target = ROOT / REPORT_NAME
    text_out(target, '\n'.join(lines) + '\n')
    text_out(ROOT / (REPORT_NAME + '.sha256'), sha(target) + '  ' + REPORT_NAME + '\n')
    return target


def show_status():
    print('SERVER:', os.uname().nodename, utc())
    if (ROOT / 'COMPLETE.json').exists():
        print('ENTROPY FIXED COMPLETE -- not a declaration that the manuscript is submission-ready')
        print(json.dumps(read(ROOT / 'COMPLETE.json'), indent=2))
    elif (ROOT / 'FAILED.json').exists():
        print('STOPPED: a check or operation failed. No original result was deleted. Do not launch a duplicate.')
        print(json.dumps(read(ROOT / 'FAILED.json'), indent=2))
    elif (ROOT / 'STATUS.json').exists():
        state = read(ROOT / 'STATUS.json')
        running = subprocess.run(['tmux', 'has-session', '-t', SESSION], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        print('RUNNING' if running else 'NO LIVE SESSION: the following status is the last recorded update')
        if state.get('status') == 'AUDIT_COMPLETE':
            print('AUDIT PASSED; use --start to launch the new fixed-DCR experiment.')
        print(json.dumps(state, indent=2))
    else:
        print('Not started yet.')
    audit = ROOT / 'original_audit.json'
    print('Original saved-array audit:', read(audit)['status'] if audit.exists() else 'not finished')
    print('Additional completed method runs (target 6):', len(list(ROOT.glob('seed*/entropy_aligned_fixed/DONE.json'))))
    print('Additional completed training phases (target 30):', len(list(ROOT.glob('seed*/entropy_aligned_fixed/rounds/round*/DONE.json'))))
    print('Report:', ROOT / REPORT_NAME)
    print('Current-run log:', ROOT / 'run.log')
    if shutil.which('nvidia-smi'):
        subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', '--format=csv'], check=False)




def start(resume=False):
    require(sys.platform.startswith('linux'), 'This launcher runs on IBM, not directly on the Mac.')
    require(Path('/mnt/caenl').is_mount(), 'The existing /mnt/caenl volume is not mounted. No formatting or mounting was attempted.')
    require(shutil.which('tmux'), 'tmux was not found; nothing was installed.')
    require(BASE.is_dir() and (BASE / 'COMPLETE.json').is_file(), 'The completed original campaign was not found.')
    require((LITE_ROOT / 'COMPLETE.json').is_file(), 'The completed entropy+Lite campaign was not found.')
    ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    import fcntl
    with (ROOT / 'RUN.lock').open('a') as probe:
        try:
            fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('An audit or training worker is already running. Use --status.')
            return
    if (ROOT / 'COMPLETE.json').exists():
        show_status()
        return
    if subprocess.run(['tmux', 'has-session', '-t', SESSION], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        print('ALREADY RUNNING. A second copy was not started.')
        show_status()
        return
    require(resume or not (ROOT / 'FAILED.json').exists(), 'A previous attempt failed. Run --status and share its report before resuming.')
    candidates = [Path.home() / 'caenl-v5.4.3/caenl_revision_pipeline_v5.4.3_c4_confirmatory/.venv/bin/python',
                  Path.home() / 'caenl-src/caenl_revision_pipeline_v5.3_ibm_rhel_l40s/.venv/bin/python']
    python = next((p for p in candidates if p.is_file()), None)
    require(python is not None, 'The existing CAENL Python environment was not found. Nothing was installed.')
    runner = ROOT / 'runner.py'
    incoming = Path(__file__).read_bytes()
    if runner.exists():
        require(runner.read_bytes() == incoming, 'A different launcher already owns this result tree. Refusing to mix versions.')
    else:
        runner.write_bytes(incoming)
    ast.parse(incoming.decode('utf-8-sig'), filename=str(runner))
    require(shutil.disk_usage(ROOT).free >= 30 * 1024 ** 3, 'At least 30 GiB free on the existing volume is required; no cleanup was performed.')
    query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    require(not query.stdout.strip(), 'Another GPU process is active. It was not stopped; wait for it before launching this addition.')
    if resume and (ROOT / 'FAILED.json').exists():
        history = ROOT / ('failure_before_resume_' + str(time.time_ns()) + '.json')
        shutil.copyfile(ROOT / 'FAILED.json', history)
        (ROOT / 'FAILED.json').unlink()
    command = 'exec env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 ' + shlex.quote(str(python)) + ' ' + shlex.quote(str(runner)) + ' --worker >> ' + shlex.quote(str(ROOT / 'run.log')) + ' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', SESSION, command], check=True)
    print('STARTED:', SESSION)
    print('First: saved-result audit and CPU/GPU checks. Then: six additional entropy + fixed aligned DCR branches.')
    print('Original 54 runs, original statistics, C4 results, and checkpoints are not overwritten.')
    print('Status: python3 ~/caenl_entropy_fixed_v1.py --status')




def worker(audit_only=False):
    global np
    import fcntl
    lock = (ROOT / 'RUN.lock').open('a')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        print('A worker already holds the lock; no duplicate was launched.')
        return 0
    try:
        if (ROOT / 'FAILED.json').exists():
            os.replace(ROOT / 'FAILED.json', ROOT / ('previous_failure_' + str(time.time_ns()) + '.json'))
        import numpy as numpy
        np = numpy
        progress('STAGE 1: validate frozen source; no original training will be repeated')
        cfg, source_files, original_env = prepare_source()
        sys.path.insert(0, str(ROOT / 'frozen_source'))
        import torch
        require(str(torch.__version__).startswith('2.6.') and str(torch.version.cuda) == '12.4', 'Expected original PyTorch/CUDA environment')
        self_tests(cfg)
        from support import Catalog
        cat = Catalog(cfg)
        manifest = verify_manifest()
        audit = audit_original(cfg, manifest, cat)
        audit_lite(cfg, cat, audit)
        export_curves(cfg, audit, include_fixed=False)
        write_share('ORIGINAL ARRAY AUDIT PASSED; ADDITIONAL TRAINING NOT YET COMPLETE')
        if audit_only:
            put(ROOT / 'AUDIT_COMPLETE.json', {'status': 'PASS', 'time': utc(), 'new_training_runs': 0, 'entropy_and_lite_curve_rows': 72})
            put(ROOT / 'STATUS.json', {'status': 'AUDIT_COMPLETE', 'time': utc(), 'message': 'Input audit and curve recovery passed; no new training was launched.'})
            write_share('INPUT AUDIT AND CURVE RECOVERY COMPLETE; NEW EXPERIMENT NOT RUN')
            print('AUDIT COMPLETE: 72 saved phase points recovered; no new training.', flush=True)
            return 0
        progress('ORIGINAL AND LITE INPUT AUDITS PASS. Preparing six matched fixed-DCR branches.')
        result = run_addition(cfg, cat, audit, source_files, original_env)
        require(len(result) == 6, 'The additional study is incomplete')
        progress('Final preservation check of original archived payloads')
        require(verify_manifest() == manifest, 'Original archived records changed during the addition')
        check_lite_unchanged(audit)
        target = write_share('COMPLETE: SIX MATCHED ENTROPY + FIXED-DCR RUNS')
        put(ROOT / 'COMPLETE.json', {'status': 'COMPLETE', 'time': utc(), 'additional_method_runs': 6,
                                     'additional_training_phases': 30, 'original_runs_repeated': 0,
                                     'report': str(target), 'report_sha256': sha(target), 'paper_submission_readiness': 'still requires the listed editorial and scope tasks'})
        put(ROOT / 'STATUS.json', {'status': 'COMPLETE', 'time': utc(), 'message': 'ENTROPY FIXED COMPLETE'})
        print('ENTROPY FIXED COMPLETE', flush=True)
        print('Share only this small text report:', target, flush=True)
        return 0
    except BaseException as exc:
        put(ROOT / 'FAILED.json', {'status': 'FAILED', 'time': utc(), 'error': repr(exc), 'traceback': traceback.format_exc(),
                                   'original_results_deleted': False, 'instruction': 'Do not reinstall or restart the original campaign. Share this report or --status output.'})
        try:
            write_share('STOPPED WITH AN ERROR; NO FULL-COMPLETION CLAIM')
        except Exception:
            pass
        traceback.print_exc()
        return 1
    finally:
        lock.close()




def verify_fixed_phase(directory, summary, cfg):
    require(summary['method'] == METHOD['name'] and summary['learned_updates_cumulative'] == 0,
            'The fixed-DCR phase used a different method or learned controller')
    with (Path(directory) / 'training.csv').open() as f:
        rows = list(csv.DictReader(f))
    require([int(r['step']) for r in rows] == list(range(1, summary['steps'] + 1)), 'Fixed-DCR step log is incomplete')
    for row in rows:
        for layer in cfg['layers']:
            close(float(row[layer + '_lambda']), cfg['lambda_fixed'], 'Fixed lambda: ' + layer, tolerance=1e-12)
    return True


def numerical_self_test():
    global np
    import numpy as numpy
    np = numpy
    close(paired([1.] * 6, [0.] * 6)['exact_signflip_p'], 2 / 64, 'Exact-test minimum')
    close(paired([0.] * 6, [0.] * 6)['exact_signflip_p'], 1., 'Zero differences')
    require(holm([.03125, .03125]) == [.0625, .0625], 'Two-contrast exact resolution')
    require(holm([.04, .01, .5]) == [.08, .03, .5], 'Holm ordering')
    a, b = [1, 2, 3, 4, 5, 7], [0, 1, 1, 3, 2, 4]
    forward, backward = paired(a, b), paired(b, a)
    close(forward['mean_difference'], -backward['mean_difference'], 'Contrast sign')
    close(forward['paired_t_p'], backward['paired_t_p'], 'Two-sided t symmetry')
    close(forward['ci95_low'], -backward['ci95_high'], 'Interval symmetry')
    close(metric_numpy([[3., 0.], [0., 3.]], [0, 0])['accuracy'], .5, 'Saved-logit accuracy')
    try:
        paired([1., float('nan'), 2., 3., 4., 5.], [0.] * 6)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Nonfinite pairs were accepted')
    print('PASS: numerical self-tests; no server files or GPU used.')


def audit_lite(cfg, cat, audit):
    """Verify existing Lite evidence and recover its phase curves without training."""
    require((LITE_ROOT / 'COMPLETE.json').is_file(), 'Entropy+Lite completion file is missing')
    completed = read(LITE_ROOT / 'COMPLETE.json')
    require(completed.get('status') == 'COMPLETE' and completed.get('additional_method_runs') == 6,
            'Entropy+Lite run is not complete')
    inp = read(LITE_ROOT / 'ADDITIONAL_BINDING.json')
    require(inp['binding'] == LITE_BINDING == digest({'protocol': inp['protocol'], 'plan': inp['plan']}),
            'Entropy+Lite binding differs from the reviewed experiment')
    require(inp['plan']['original_binding'] == EXPECTED_BINDING, 'Lite used a different original campaign')
    original = read(BASE / 'INPUT_BINDING.json')
    require(inp['plan']['source_files'] == original['source_files'], 'Lite training source differs')
    method = {'name': 'entropy_aligned_lite', 'acquisition': 'entropy', 'regularizer': 'aligned_lite'}
    require(inp['protocol']['methods'] == [method], 'Unexpected Lite method')
    for key, value in cfg.items():
        if key not in {'campaign', 'result_root', 'archive_root', 'methods', 'role', 'no_legacy_checkpoints_loaded', 'statistics'}:
            require(inp['protocol'].get(key) == value, 'Lite changed matched configuration: ' + key)
    files = {}

    def record(path):
        path = Path(path)
        require(path.is_file(), 'Missing Lite input: ' + str(path))
        files[str(path)] = sha(path)

    def payload(directory, values):
        verify_payload(directory, values)
        for rel in values:
            path = under(Path(directory), rel)
            if path.suffix not in ('.pt', '.pth', '.ckpt'):
                record(path)

    for p in ('COMPLETE.json', 'ADDITIONAL_BINDING.json'):
        record(LITE_ROOT / p)
    metrics, curves = [], []
    for seed in SEEDS:
        old, sd = BASE / f'seed{seed}', LITE_ROOT / f'seed{seed}'
        for rel in ('splits.npz', 'targets.json'):
            require(sha(sd / rel) == sha(old / rel), 'Lite split/target does not match original: ' + rel)
            record(sd / rel)
        split = arrays(sd / 'splits.npz')
        labeled, held = split['initial'].copy(), split['holdout']
        targets = read(sd / 'targets.json')['targets']
        initial = read(old / 'shared_initial/summary.json')
        out = sd / method['name']
        mb = digest({'binding': LITE_BINDING, 'seed': seed, 'method': method,
                     'split': sha(sd / 'splits.npz'), 'targets': sha(sd / 'targets.json')})
        done, summary = read(out / 'DONE.json'), read(out / 'summary.json')
        require(done.get('status') == 'COMPLETE' and done['binding'] == mb == summary['binding'], 'Lite completion binding mismatch')
        payload(out, done['payload'])
        record(out / 'DONE.json')
        require(summary['seed'] == seed and summary['method'] == method['name'] and len(summary['rounds']) == 6,
                'Lite summary identity mismatch')
        require(summary['rounds'][0] == initial, 'Lite reused a different initialization')
        previous_hash = initial['checkpoint_sha256']
        for phase in range(1, 6):
            acq = out / 'acquisitions' / f'round{phase}'
            selection = arrays(acq / 'selection.npz')
            meta = read(acq / 'COMPLETE.json')
            ab = digest({'binding': mb, 'checkpoint': previous_hash, 'labeled': labeled.tolist(),
                         'holdout': held.tolist(), 'seed': seed, 'phase': phase, 'method': method})
            require(meta['binding'] == ab, 'Lite acquisition binding mismatch')
            payload(acq, {'selection.npz': meta['selection_sha256']})
            record(acq / 'COMPLETE.json')
            for rec in meta['passes']:
                payload(acq, {f"pass_{int(rec['pass']):03d}.npz": rec['sha256']})
            require(np.array_equal(selection['previous_labeled'], labeled) and np.array_equal(selection['control_holdout'], held),
                    'Lite acquisition previous state differs')
            chosen = selection['selected']
            require(len(chosen) == cfg['acquisition_per_round'] and len(np.unique(chosen)) == len(chosen), 'Lite acquisition count mismatch')
            require((chosen >= 0).all() and (chosen < len(cat.labels)).all()
                    and not np.intersect1d(chosen, np.concatenate([labeled, held])).size, 'Lite acquisition overlap or invalid ID')
            if phase == 1:
                require(np.array_equal(chosen, arrays(old / 'entropy/acquisitions/round1/selection.npz')['selected']),
                        'Lite first acquisition differs from original entropy')
            labeled = np.sort(np.concatenate([labeled, chosen]))
            pd = out / 'rounds' / f'round{phase}'
            ps, phase_done = read(pd / 'summary.json'), read(pd / 'DONE.json')
            pb = digest({'campaign': mb, 'seed': seed, 'method': method['name'], 'phase': phase,
                         'source': previous_hash, 'labeled': labeled.tolist(), 'control': held.tolist(), 'targets': targets})
            require(ps['binding'] == pb == phase_done['binding'] and ps == summary['rounds'][phase], 'Lite phase linkage mismatch')
            require(ps['steps'] == math.ceil(len(labeled) / 128) * cfg['round_epochs']
                    and ps['annotation_count'] == len(labeled) + len(held), 'Lite budget mismatch')
            payload(pd, phase_done['payload'])
            record(pd / 'DONE.json')
            m = check_evaluation(pd / 'validation.npz', ps['validation'], cat.val_labels, np.arange(len(cat.val_labels)), cfg)
            check_evaluation(pd / 'feedback.npz', ps['feedback'], cat.labels[held], held, cfg)
            curves.append({'seed': seed, 'method': method['name'], 'phase': phase, 'annotations': ps['annotation_count'],
                           'accuracy': m['accuracy'], 'ce': m['ce'], 'mean_log_kappa': m['mean_log_kappa']})
            previous_hash = ps['checkpoint_sha256']
        robust = check_attack(out / 'autoattack', summary, ps, cat.val_labels, cfg, mb)
        for p in (out / 'autoattack').glob('*'):
            if p.is_file() and p.suffix in ('.npz', '.json'):
                record(p)
        metrics.append({'seed': seed, 'method': method['name'], 'clean_accuracy': m['accuracy'], 'robust_accuracy': robust,
                        'cross_entropy': m['ce'], 'ece': m['ece'], 'mean_log_kappa': m['mean_log_kappa']})
        progress(f'Lite saved-array audit: {len(metrics)}/6 seed runs')
    audit['lite_metrics'] = metrics
    audit['lite_phase_metrics'] = curves
    audit['lite_input_digests'] = files
    put(ROOT / 'lite_audit.json', {'status': 'PASS', 'binding': LITE_BINDING, 'metrics': metrics,
                                 'phase_metrics': curves, 'input_digests': files,
                                 'scope': 'Saved-array arithmetic and linkage; existing Lite training and attacks are not rerun.'})


def check_lite_unchanged(audit):
    for p, expected in audit['lite_input_digests'].items():
        require(sha(p) == expected, 'Existing Lite artifact changed: ' + p)


def export_curves(cfg, audit, include_fixed):
    """Only observed phase endpoints; original/Lite inputs were checked above."""
    import io
    rows, geometry = [], []
    methods = [(BASE, 'entropy'), (LITE_ROOT, 'entropy_aligned_lite')]
    if include_fixed:
        methods.append((ROOT, METHOD['name']))
    for base, method in methods:
        for seed in SEEDS:
            summary = read(base / f'seed{seed}' / method / 'summary.json')
            require(len(summary['rounds']) == 6, 'Cannot export a missing phase')
            for phase, ps in enumerate(summary['rounds']):
                annotations = cfg['initial_labels'] + phase * cfg['acquisition_per_round']
                require(ps['annotation_count'] == annotations, 'Curve budget mismatch')
                v = ps['validation']
                row = {'seed': seed, 'method': method, 'phase': phase, 'annotations': annotations,
                       'annotation_fraction': annotations / cfg['expected_train_count'],
                       'accuracy': v['accuracy'], 'cross_entropy': v['ce'], 'ece': v['ece']}
                rows.append(row)
                for layer in cfg['layers']:
                    geometry.append({'seed': seed, 'method': method, 'phase': phase, 'layer': layer,
                                     'annotations': annotations, 'within': v[layer + '_within'],
                                     'between': v[layer + '_between'], 'log_kappa': v[layer + '_log_kappa'],
                                     'measurement': 'full validation, 100 classes; not the 16-class calibration target'})
    grouped = []
    for _, method in methods:
        for phase in range(6):
            subset = [r for r in rows if r['method'] == method and r['phase'] == phase]
            require(sorted(r['seed'] for r in subset) == SEEDS, 'Curve grouping is missing seed pairs')
            values = np.asarray([r['accuracy'] for r in subset], dtype=np.float64)
            grouped.append({'method': method, 'phase': phase, 'n_seeds': 6, 'annotations': subset[0]['annotations'],
                            'annotation_fraction': subset[0]['annotation_fraction'],
                            'accuracy_mean': float(values.mean()), 'accuracy_sample_sd': float(values.std(ddof=1))})
    for name, records in [('accuracy_by_budget.csv', rows), ('layer_geometry_by_budget.csv', geometry),
                          ('grouped_accuracy_by_budget.csv', grouped)]:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
        text_out(ROOT / name, buf.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--start', action='store_true', help='Start the six fixed-DCR runs in tmux on IBM')
    group.add_argument('--resume', action='store_true', help='Resume a stopped run with identical source and protocol')
    group.add_argument('--status', action='store_true')
    group.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    group.add_argument('--audit-only', action='store_true', help='Existing-environment input checks and curve recovery; no new ImageNet training')
    group.add_argument('--self-test', action='store_true', help='Local NumPy/SciPy mathematical tests; no server access')
    args = parser.parse_args()
    if args.self_test:
        numerical_self_test()
        return 0
    if args.worker:
        return worker()
    if args.status:
        show_status()
    elif args.audit_only:
        require(sys.platform.startswith('linux') and Path('/mnt/caenl').is_mount(), 'Run --audit-only on the IBM server with the existing CAENL Python environment.')
        require(not (ROOT / 'COMPLETE.json').exists(), 'The experiment is already complete; use --status.')
        ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
        runner = ROOT / 'runner.py'
        incoming = Path(__file__).read_bytes()
        if runner.exists():
            require(runner.read_bytes() == incoming, 'A different launcher owns this result tree.')
        else:
            runner.write_bytes(incoming)
        return worker(audit_only=True)
    else:
        start(resume=args.resume)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
