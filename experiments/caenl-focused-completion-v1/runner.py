#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CAENL focused completion v1.
Read-only audit of the completed 54 method-seed runs; six additional entropy +
aligned MACC-Lite branches in a DIFFERENT result tree. No installation,
download, deletion, modification of original source, or public upload.
This orchestration script has not been executed in the authoring runtime.
It requires its local CPU/integration/GPU checks to pass on IBM before training.
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
ROOT = Path('/mnt/caenl/active/results/caenl-focused-completion-v1')
SESSION = 'caenl-focused-completion-v1'
REPORT_NAME = 'caenl-focused-completion-v1.md'
EXPECTED_BINDING = '159fe52901f774e7abeb40a3afe96de81f1084a1bde0f63d070c7d2ecda325e1'
EXPECTED_PROTOCOL = '9dfd0201a5d69d05dd8992ee9d642bb43c3a5d3b0b07b51c28ab03bd3c55e08e'
SEEDS = list(range(801, 807))
METHOD = {'name': 'entropy_aligned_lite', 'acquisition': 'entropy', 'regularizer': 'aligned_lite'}
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




def redact_config(obj):
    sensitive = {'password', 'secret', 'api_key', 'apikey', 'hf_token', 'access_token', 'auth_token', 'authorization', 'cookies', 'credentials'}
    if isinstance(obj, dict):
        return {k: ('[REDACTED]' if str(k).lower() in sensitive else redact_config(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_config(v) for v in obj]
    return obj




def collect_c4():
    c4 = Path('/mnt/caenl/active/results/caenl-c4-confirmatory-v2')
    lines = ['# C4 configuration recovery', '', 'Configuration/source evidence only; no C4 training or evaluation is executed.']
    if not c4.is_dir():
        lines += ['', 'The expected preserved C4 result directory was not found. Exact optimizer/scheduler recovery remains pending.']
    else:
        seen, total, selected, omitted = set(), 0, 0, []
        for p in sorted(c4.rglob('*')):
            if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(c4.resolve()):
                continue
            rel = p.relative_to(c4).as_posix()
            name = p.name.lower()
            use = (p.suffix.lower() in ('.json', '.yaml', '.yml') and ('config' in name or 'resolved' in name or name in ('job.json', 'plan.json', 'plan.yaml', 'plan.yml')))
            use = use or (p.suffix == '.py' and ('language' in name or name in ('c4.py', 'config.py')) and 'source' in rel)
            if not use:
                continue
            if p.stat().st_size > 512000 or total + p.stat().st_size > 4 * 1024 * 1024:
                omitted.append(rel)
                continue
            h = sha(p)
            if h in seen:
                lines.append('\nDuplicate-content configuration: ' + rel + ' (SHA-256 ' + h + ')')
                continue
            seen.add(h)
            content = p.read_text(encoding='utf-8-sig')
            if p.suffix == '.json':
                content = json.dumps(redact_config(json.loads(content)), indent=2)
            else:
                content = re.sub(r'(?im)^(\s*(?:hf_token|api_key|password|access_token|secret)\s*[:=]).*$', r'\1 [REDACTED]', content)
            lines += ['', '## ' + rel, 'Original-file SHA-256: ' + h, '````````', content, '````````']
            total += p.stat().st_size
            selected += 1
        lines += ['', f'Unique configuration/source files included: {selected}', 'Files omitted by size limit: ' + json.dumps(omitted),
                  'These include candidate resolved job configurations and implementation sources; the manuscript settings must be traced to the executed job, not merely a template.']
    text_out(ROOT / 'c4_configuration_evidence.md', '\n'.join(lines) + '\n')




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
    with tempfile.TemporaryDirectory(prefix='entropy-lite-test-', dir=ROOT) as temp:
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
        require(state['controller']['kind'] == 'aligned_lite', 'Additional method did not activate MACC-Lite')
        require(any(r['layer3_lambda'] > 0 or r['layer4_lambda'] > 0 for r in state['rows']), 'Additional method did not enable aligned regularization coefficients')
        require(sha256(cp) == before, 'Integration test changed its input checkpoint')
        check_evaluation(root / 'phase/validation.npz', met['validation'], cat.val_labels, np.arange(len(cat.val_labels)), small)
        check_evaluation(root / 'phase/feedback.npz', met['feedback'], cat.labels[held], held, small)
        resumed, same_cp = run_phase(small, cat, 801, METHOD, 1, labeled, held, cp, targets, root / 'phase', 'extension-test', 20, lambda m: None, 'cpu', model_factory=Tiny)
        require(resumed == met and sha256(same_cp) == sha256(new_cp), 'Completed additional phase did not reuse its saved result')
    progress(f'PRECHECK PASS: {count} inherited checks plus audit and entropy+Lite CPU integration checks')
    put(ROOT / 'SELF_TEST.json', {'status': 'PASS', 'time': utc(), 'inherited_checks': count,
                                  'additional_entropy_lite_forward_backward': True, 'saved_array_arithmetic': True,
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
                             'comparisons': [['entropy_aligned_lite', 'entropy'], ['caenl_aligned_lite', 'entropy_aligned_lite']],
                             'test': 'paired t and exact sign flip; Holm over these two supplementary clean-accuracy contrasts only; original seven-contrast family unchanged'}
    plan = {'role': new_cfg['role'], 'decided_before_additional_training': True, 'seeds': SEEDS, 'method': METHOD,
            'comparisons': new_cfg['statistics'], 'reuse': audit['reused_input_digests'],
            'original_binding': EXPECTED_BINDING, 'source_files': source_files, 'runner_sha256': sha(ROOT / 'runner.py'),
            'original_statistics_not_overwritten': True, 'original_seeds_are_not_new_independent_replications': True,
            'no_new_hyperparameter_tuning': True, 'no_new_datasets': True,
            'robustness': 'same full-set standard 8/255 endpoint, supplementary descriptive result',
            'remaining_outside_this_script': ['recent matched-baseline coverage decision', 'matched end-to-end overhead if claimed', 'manuscript/PDF and reviewer response', 'public deposit and author declarations', 'review recovered C4 production configuration']}
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
                progress(f'ADDITIONAL TRAINING: entropy + aligned MACC-Lite, seed {seed}, acquisition {phase}/5')
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
    tests = comparisons(audit['original_metrics'] + new_rows, new_cfg['statistics']['comparisons'], ['clean_accuracy'])
    put(ROOT / 'supplementary_clean_comparisons.json', tests)
    check_reused(audit)
    return new_rows




def write_share(status):
    lines = ['# CAENL focused completion v1', '', 'Status: ' + status, 'Generated UTC: ' + utc(), '',
             'Original 54 runs are unchanged. This is a later matched ablation, not an expansion of the originally prespecified hypothesis family.',
             'One additional condition is evaluated on the six original seed IDs for pairing. These are not new independent replications of the existing conditions.',
             'Verification distinguishes saved-array arithmetic from checkpoint inference, fresh attacks, image-level perturbation verification and independent reproduction.', '',
             'Remaining paper tasks: recent matched-baseline coverage decision; any stronger runtime/memory claim; full manuscript build; author declarations; public code/results release; reviewer response. No new C4 task training or new datasets were launched.']
    for name in ('SELF_TEST.json', 'original_audit.json', 'ADDITIONAL_BINDING.json', 'CUDA_PREFLIGHT.json', 'AUTOATTACK_SOURCE.json', 'DATA_VERIFIED.json', 'additional_metrics.json', 'supplementary_clean_comparisons.json', 'FAILED.json'):
        p = ROOT / name
        if p.exists():
            lines += ['', '## ' + name, 'SHA-256: ' + sha(p), '````````', p.read_text(), '````````']
    p = ROOT / 'c4_configuration_evidence.md'
    if p.exists():
        lines += ['', p.read_text()]
    lines += ['', '## Launcher source', '````````', (ROOT / 'runner.py').read_text(encoding='utf-8-sig'), '````````', '', 'END OF FOCUSED COMPLETION REPORT']
    target = ROOT / REPORT_NAME
    text_out(target, '\n'.join(lines) + '\n')
    text_out(ROOT / (REPORT_NAME + '.sha256'), sha(target) + '  ' + REPORT_NAME + '\n')
    return target




def show_status():
    print('SERVER:', os.uname().nodename, utc())
    if (ROOT / 'COMPLETE.json').exists():
        print('FOCUSED ADDITION COMPLETE -- not a declaration that the manuscript is submission-ready')
        print(json.dumps(read(ROOT / 'COMPLETE.json'), indent=2))
    elif (ROOT / 'FAILED.json').exists():
        print('STOPPED: a check or operation failed. No original result was deleted. Do not launch a duplicate.')
        print(json.dumps(read(ROOT / 'FAILED.json'), indent=2))
    elif (ROOT / 'STATUS.json').exists():
        state = read(ROOT / 'STATUS.json')
        running = subprocess.run(['tmux', 'has-session', '-t', SESSION], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        print('RUNNING' if running else 'NO LIVE SESSION: the following status is the last recorded update')
        print(json.dumps(state, indent=2))
    else:
        print('Not started yet.')
    audit = ROOT / 'original_audit.json'
    print('Original saved-array audit:', read(audit)['status'] if audit.exists() else 'not finished')
    print('Additional completed method runs (target 6):', len(list(ROOT.glob('seed*/entropy_aligned_lite/DONE.json'))))
    print('Additional completed training phases (target 30):', len(list(ROOT.glob('seed*/entropy_aligned_lite/rounds/round*/DONE.json'))))
    print('Report:', ROOT / REPORT_NAME)
    print('Current-run log:', ROOT / 'run.log')
    if shutil.which('nvidia-smi'):
        subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', '--format=csv'], check=False)




def start(resume=False):
    require(sys.platform.startswith('linux'), 'This launcher runs on IBM, not directly on the Mac.')
    require(Path('/mnt/caenl').is_mount(), 'The existing /mnt/caenl volume is not mounted. No formatting or mounting was attempted.')
    require(shutil.which('tmux'), 'tmux was not found; nothing was installed.')
    require(BASE.is_dir() and (BASE / 'COMPLETE.json').is_file(), 'The completed original campaign was not found.')
    ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
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
    print('First: saved-result audit and CPU/GPU checks. Then: six additional entropy + aligned MACC-Lite branches.')
    print('Original 54 runs, original statistics, C4 results, and checkpoints are not overwritten.')
    print('Status: python3 ~/caenl_focused_completion_v1.py --status')




def worker():
    global np
    import fcntl
    lock = (ROOT / 'RUN.lock').open('a')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('A worker already holds the lock; no duplicate was launched.')
        return 0
    try:
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
        collect_c4()
        write_share('ORIGINAL ARRAY AUDIT PASSED; ADDITIONAL TRAINING NOT YET COMPLETE')
        progress('ORIGINAL SAVED-ARRAY AUDIT PASS. Preparing the six additional matched branches.')
        result = run_addition(cfg, cat, audit, source_files, original_env)
        require(len(result) == 6, 'The additional study is incomplete')
        progress('Final preservation check of original archived payloads')
        require(verify_manifest() == manifest, 'Original archived records changed during the addition')
        target = write_share('COMPLETE: ORIGINAL ARRAY AUDIT + SIX ADDITIONAL MATCHED METHOD RUNS')
        put(ROOT / 'COMPLETE.json', {'status': 'COMPLETE', 'time': utc(), 'additional_method_runs': 6,
                                     'additional_training_phases': 30, 'original_runs_repeated': 0,
                                     'report': str(target), 'report_sha256': sha(target), 'paper_submission_readiness': 'still requires the listed editorial and scope tasks'})
        put(ROOT / 'STATUS.json', {'status': 'COMPLETE', 'time': utc(), 'message': 'FOCUSED ADDITION COMPLETE'})
        print('FOCUSED ADDITION COMPLETE', flush=True)
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




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--start', action='store_true')
    group.add_argument('--resume', action='store_true')
    group.add_argument('--status', action='store_true')
    group.add_argument('--worker', action='store_true')
    args = parser.parse_args()
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
    except Exception as error:
        print('STOPPED:', error, file=sys.stderr)
        raise SystemExit(1)