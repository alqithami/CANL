#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CAENL diffusion confirmation v1 -- diffusion ONLY, no audio downloads.


Six new seeds, four fine-tuning conditions, 10,000 updates and 50,000 generated
images each; six sampling-only evaluations of the same pretrained reference.
Reuses an integrity-checked COPY of the successfully executed pilot engine.
No completed experiment is resumed, deleted or overwritten. No packages installed.
This new orchestration was not executed in the authoring runtime. Local integration
and GPU checks run on IBM before scientific jobs. Null/harmful results are valid.
"""
from __future__ import annotations
import argparse
import ast
import copy
import gc
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone


sys.dont_write_bytecode = True
SESSION = 'caenl-diffusion-confirm-v1'
ROOT = Path('/mnt/caenl/active/results') / SESSION
PILOT = Path('/mnt/caenl/active/results/caenl-crossmodal-pilots-v1')
REPORT = SESSION + '.md'
SEEDS = (1101, 1102, 1103, 1104, 1105, 1106)
METHODS = ('baseline', 'fixed_dcr', 'macc_lite', 'full_macc')
ALL_METHODS = ('pretrained_reference',) + METHODS
PILOT_RUNNER_SHA = 'd3d01297c43357b9c34d81701b21f1b80c3a3a395bb65ef341026c61c918ac15'
PILOT_DATA_SHA = '2af95b63455f39d547b069ee3d6c867ab0a57824b5bc36e2134b81b3cb75cc57'
PILOT_SOURCE_SHA = 'e8a7a3039390e60491431376ec1d28aad07e674fbaf64ca89aaa18e52f6b9451'
MODEL_ID = 'google/ddpm-cifar10-32'
MODEL_REVISION = '267b167dc01f0e4e61923ea244e8b988f84deb80'
MODEL_FILES = {
 'config.json': '49579448cd5728fb83f14186de12464eee143d2f5dccb11b8e13744d712eae9d',
 'diffusion_pytorch_model.safetensors': '1558dbe3fb093b0857d473e376b711eeda1b186c557be5e85b3a56300cd49f0f',
 'scheduler_config.json': 'da5d9d26605d2aab854e8f2f2b26d3a407618de17b948bdb7d3846fb0561c7d3',
}
# Minimal, exact-count replacements in the copied pilot orchestrator only.
# The actual caenl diffusion, loss, controller, model and metric source is unchanged.
PATCHES = [
 ("'train_steps': 0 if method == 'pretrained_reference' else 2000", "'train_steps': 0 if method == 'pretrained_reference' else 10000"),
 ("'num_samples': 5000, 'save_samples': 5000", "'num_samples': 50000, 'save_samples': 50000"),
 ("result['steps'] == (0 if method == 'pretrained_reference' else 2000)", "result['steps'] == (0 if method == 'pretrained_reference' else 10000)"),
 ("result['final']['n_samples'] == 5000", "result['final']['n_samples'] == 50000"),
 ("final scheduled pilot step, never best test score", "final scheduled confirmation step, never best test score"),
 ("'evaluation_role': 'development; no official test outcomes'", "'evaluation_role': 'confirmation FID/KID/IS; internal holdout MSE remains descriptive; official-test EMA MSE evaluated separately'"),
]
PROTOCOL = {
 'study': SESSION, 'role': 'independent_seed_spectral_diffusion_confirmation_after_development_pilots',
 'seeds': list(SEEDS), 'excluded_development_seeds': [1001, 1002],
 'conditions': list(METHODS), 'sampling_reference': 'same unmodified pretrained model, separate matched noise stream per seed',
 'training_runs': 24, 'reference_evaluations': 6, 'training_updates_per_run': 10000,
 'batch_size': 128, 'training_images': 45000, 'fixed_internal_holdout_images': 5000,
 'pretrained_model': MODEL_ID, 'pretrained_model_revision': MODEL_REVISION,
 'pretraining_overlap': 'The fine-tuning holdout was in the original pretrained CIFAR10 training distribution; it is not pristine relative to pretraining.',
 'starting_point': 'original pretrained weights, never a pilot final checkpoint',
 'sampler': {'name': 'DDIM', 'steps': 100, 'eta': 0.0, 'batch_size': 128, 'samples': 50000,
             'same_seed_noise_across_conditions': True, 'model': 'EMA checkpoint'},
 'spectral_protocol': 'unchanged from the pilot: rank 16, retained k=64, fixed lambda .01, adaptive cap .05, interval 50, original spectral controllers with terminal reward handling',
 'not_evaluated': ['audio captioning', 'audio retrieval', 'aligned classification DCR on diffusion', 'active label acquisition'],
 'primary_endpoint': 'FID against original CIFAR10 training distribution using torch-fidelity 0.4.0',
 'contrasts_target_minus_reference': [['baseline', 'pretrained_reference'], ['fixed_dcr', 'baseline'], ['macc_lite', 'baseline'], ['full_macc', 'baseline']],
 'statistics': 'six paired seeds; paired t and all 64 exact sign flips; Holm across FOUR FID contrasts separately per test family; individual unadjusted 95-percent t intervals',
 'secondary_descriptive_endpoints': ['KID', 'Inception score', 'internal-holdout denoising MSE', 'full official-test EMA denoising MSE', 'representation ranks', 'controller histories', 'timing and peak allocation'],
 'official_test_ema_mse': {'images': 10000, 'noise_and_timestep_seed': 541211, 'batch_size': 128,
                         'model': 'EMA', 'precision': 'FP32 parameters with BF16 autocast, as sampling',
                         'selection': 'after final scheduled checkpoint only; not used for model selection or feedback'},
 'interpretation': 'fine-tuning and regularization study, not reproduction of DDPM published benchmark numbers or a robustness guarantee',
 'stopping': 'technical validity only, never a minimum gain; all completed conditions are reported',
 'authoring_runtime_execution': False,
}




def require(ok, message):
    if not bool(ok): raise RuntimeError(message)




def utc(): return datetime.now(timezone.utc).isoformat()




def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''): h.update(b)
    return h.hexdigest()




def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))




def sanitize(x):
    if isinstance(x, dict): return {str(k): sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [sanitize(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x): return None
    if isinstance(x, Path): return str(x)
    if hasattr(x, 'item'):
        try: return sanitize(x.item())
        except Exception: pass
    return x




def digest(x): return hashlib.sha256(json.dumps(sanitize(x), sort_keys=True, allow_nan=False).encode()).hexdigest()




def write(path, text):
    path = Path(path)
    require(path.resolve().is_relative_to(ROOT.resolve()), 'Refusing write outside the NEW study: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    with temp.open('w', encoding='utf-8') as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)




def put(path, obj): write(path, json.dumps(sanitize(obj), indent=2, allow_nan=False) + '\n')




def progress(message):
    put(ROOT / 'STATUS.json', {'status': 'RUNNING', 'pid': os.getpid(), 'time': utc(), 'message': message})
    print(message, flush=True)




def copy_checked(source, dest, expected=None):
    source, dest = Path(source), Path(dest)
    require(source.is_file(), 'Required preserved input is missing: ' + str(source))
    require(dest.resolve().is_relative_to(ROOT.resolve()), 'Invalid copy destination')
    actual = sha(source)
    require(expected is None or actual == expected, 'Input hash mismatch: ' + str(source))
    if dest.exists():
        require(dest.is_file() and not dest.is_symlink() and sha(dest) == actual, 'Copied input differs: ' + str(dest))
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_name(dest.name + '.copying')
        shutil.copyfile(source, temp)
        require(sha(temp) == actual, 'Copy hash mismatch')
        os.replace(temp, dest)
    return actual




def copy_tree(source, dest):
    source = Path(source)
    if not source.is_dir(): return
    for p in sorted(source.rglob('*')):
        if p.is_file() and not p.name.endswith(('.lock', '.tmp', '.partial')):
            copy_checked(p, Path(dest) / p.relative_to(source))




def prepare():
    progress('Verifying completed diffusion pilots and creating a separate confirmation workspace')
    require(sha(PILOT / 'runner.py') == PILOT_RUNNER_SHA, 'Pilot launcher does not match the reviewed version')
    require(sha(PILOT / 'DIFFUSION_DATA.json') == PILOT_DATA_SHA, 'Pilot split manifest differs')
    require(sha(PILOT / 'SOURCE.json') == PILOT_SOURCE_SHA, 'Pilot source manifest differs')
    complete = read(PILOT / 'DIFFUSION_COMPLETE.json')
    require(complete['training_runs'] == 8 and complete['reference_evaluations'] == 2 and complete['seeds'] == [1001, 1002], 'Expected completed two-seed diffusion pilot')
    pins = {}
    def pin(p, expected=None):
        h = sha(p); require(expected is None or h == expected, 'Input hash differs: ' + str(p)); pins[str(p)] = h
    for filename in ('runner.py', 'DIFFUSION_DATA.json', 'SOURCE.json', 'ENVIRONMENT.json', 'MODELS.json', 'DIFFUSION_COMPLETE.json'):
        pin(PILOT / filename)
    for seed in (1001, 1002):
        for method in ALL_METHODS:
            folder = PILOT / 'diffusion' / ('seed' + str(seed)) / method
            done = read(folder / 'DONE.json')
            require(done['status'] == 'COMPLETE', 'Pilot job is not complete')
            for rel, expected in done['files'].items():
                path = folder / rel
                require(path.resolve().is_relative_to(folder.resolve()), 'Unsafe pilot payload path')
                pin(path, expected)
            summary = read(folder / 'summary.json')
            require(summary['steps'] == (0 if method == 'pretrained_reference' else 2000) and summary['final']['n_samples'] == 5000, 'Unexpected pilot budget')
            copy_checked(folder / 'summary.json', ROOT / 'pilot_summaries' / ('seed' + str(seed)) / (method + '.json'))
    # Preserve the executed training and metric source, without importing editable site packages.
    src = read(PILOT / 'SOURCE.json')
    for rel, expected in src['files'].items():
        source = PILOT / 'source/caenl' / rel
        pin(source, expected)
        copy_checked(source, ROOT / 'source/caenl' / rel, expected)
    copy_checked(PILOT / 'SOURCE.json', ROOT / 'SOURCE.json', PILOT_SOURCE_SHA)
    copy_checked(PILOT / 'DIFFUSION_DATA.json', ROOT / 'DIFFUSION_DATA.json', PILOT_DATA_SHA)
    # Independent byte copies: no links that could overwrite the old dataset or metric cache.
    for rel in ('cache/vision/cifar10-r32', 'cache/torch_fidelity', 'data/torch_fidelity', 'assets/torch'):
        copy_tree(PILOT / rel, ROOT / rel)
    model_rec = read(PILOT / 'MODELS.json')[MODEL_ID]
    require(model_rec['snapshot_revision'] == MODEL_REVISION and model_rec['files'] == MODEL_FILES, 'Pretrained revision/hash set differs')
    model_dir = ROOT / 'assets/ddpm-cifar10-32' / MODEL_REVISION
    for rel, expected in MODEL_FILES.items():
        path = Path(model_rec['path']) / rel
        pin(path, expected); copy_checked(path, model_dir / rel, expected)
    new_model = dict(model_rec); new_model['path'] = str(model_dir)
    models = {MODEL_ID: new_model}
    if (ROOT / 'MODELS.json').exists(): require(read(ROOT / 'MODELS.json') == models, 'Confirmation model record differs')
    else: put(ROOT / 'MODELS.json', models)
    original = (PILOT / 'runner.py').read_text(encoding='utf-8-sig')
    patched = original
    for old, new in PATCHES:
        require(patched.count(old) == 1, 'Expected exactly one reviewed patch target: ' + old)
        patched = patched.replace(old, new)
    ast.parse(patched, filename='engine.py')
    if (ROOT / 'engine.py').exists(): require((ROOT / 'engine.py').read_text() == patched, 'Engine changed since start')
    else: write(ROOT / 'engine.py', patched)
    patch_record = {'original_sha256': PILOT_RUNNER_SHA, 'engine_sha256': sha(ROOT / 'engine.py'),
                    'exact_replacements': PATCHES, 'scientific_caenl_source_modified': False,
                    'runtime_overrides': 'ROOT, SEEDS and PLAN set by the versioned confirmation wrapper; audio worker is never invoked'}
    put(ROOT / 'ENGINE_PATCHES.json', patch_record)
    import numpy as np
    ds = read(ROOT / 'DIFFUSION_DATA.json')
    cache = ROOT / 'cache/vision/cifar10-r32'
    for filename, key in (('train_x.npy', 'train_images'), ('train_y.npy', 'train_labels'), ('test_x.npy', 'test_images'), ('test_y.npy', 'test_labels')):
        a = np.load(cache / filename, mmap_mode='r', allow_pickle=False)
        h = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
        require(h == ds['arrays_sha256'][key], 'Dataset array differs: ' + filename)
    train = np.asarray(ds['train_indices']); held = np.asarray(ds['heldout_indices'])
    require(len(train) == 45000 and len(held) == 5000 and len(np.unique(np.r_[train, held])) == 50000, 'Invalid preserved train/holdout split')
    if (ROOT / 'PRESERVED_INPUTS.json').exists(): require(read(ROOT / 'PRESERVED_INPUTS.json') == pins, 'Preserved inputs changed since study start')
    else: put(ROOT / 'PRESERVED_INPUTS.json', pins)
    frozen = {'protocol': PROTOCOL, 'runner_sha256': sha(ROOT / 'runner.py'), 'engine_sha256': sha(ROOT / 'engine.py'),
              'source_manifest_sha256': sha(ROOT / 'SOURCE.json'), 'data_manifest_sha256': sha(ROOT / 'DIFFUSION_DATA.json'), 'models': models,
              'pilot_inputs_digest': digest(pins)}
    frozen['binding'] = digest(frozen)
    if (ROOT / 'PROTOCOL.json').exists(): require(read(ROOT / 'PROTOCOL.json') == frozen, 'Confirmation protocol or inputs changed')
    else: put(ROOT / 'PROTOCOL.json', frozen)
    progress('Source, model, dataset and pilot preservation checks passed; protocol frozen')




def engine():
    spec = importlib.util.spec_from_file_location('caenl_confirm_engine', ROOT / 'engine.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.ROOT = ROOT; mod.SESSION = SESSION; mod.SEEDS = SEEDS
    mod.PLAN = copy.deepcopy(mod.PLAN)
    mod.PLAN['role'] = PROTOCOL['role']
    mod.PLAN.pop('pilot_seeds', None)
    mod.PLAN['confirmation_seeds'] = list(SEEDS)
    mod.PLAN['conditions'] = list(METHODS)
    mod.PLAN['diffusion'].update(train_steps=10000, samples_per_condition=50000,
                               official_test_outcomes_used=True,
                               official_test_use='EMA denoising MSE after the final scheduled checkpoint only; never feedback or selection')
    mod.PLAN['next_after_review_only'] = 'This wrapper executes the frozen six-seed diffusion confirmation; no audio work.'
    mod.PLAN['audio'] = {'status': 'not_run_by_this_launcher'}
    mod.PLAN['confirmation_protocol_binding'] = read(ROOT / 'PROTOCOL.json')['binding']
    mod.source_setup()
    return mod




def checks():
    e = engine(); e.runtime()
    previous_env = read(PILOT / 'ENVIRONMENT.json')
    env = e.environment_record()
    require(env == previous_env, 'Installed Python/package versions differ from the successful pilot; no packages were changed')
    put(ROOT / 'ENVIRONMENT.json', env)
    e.checks()
    # Configuration regression: assert every final job uses the intended budget and scope.
    for seed in SEEDS:
        for method in ALL_METHODS:
            cfg = e.configure('diffusion', method, seed)
            require(cfg['diffusion']['train_steps'] == (0 if method == 'pretrained_reference' else 10000), 'Final step budget mismatch')
            require(cfg['diffusion']['num_samples'] == 50000 and cfg['diffusion']['save_samples'] == 50000, 'Final sample budget mismatch')
            require('audio' not in cfg and cfg['role'] == PROTOCOL['role'], 'Wrong task/role in resolved configuration')
            require(str(cfg['paths']['data_root']).startswith(str(ROOT)), 'Data path not isolated')
    import numpy as np
    actual = paired_stats(np.array([1., 2., 3., 4., 5., 6.]), np.zeros(6))
    require(actual['exact_signflip_p'] == .03125, 'Exact sign-flip enumeration regression failed')
    require(paired_stats(np.ones(6), np.ones(6))['exact_signflip_p'] == 1., 'All-zero contrast regression failed')
    require(np.allclose(holm([.01, .04, .5]), [.03, .08, .5]), 'Holm correction regression failed')
    put(ROOT / 'CONFIRMATION_CHECKS.json', {'status': 'PASS', 'configuration_checks': 30, 'GPU_checks': 'in SELF_TEST.json',
          'statistics_checks': ['64 exact sign flips', 'all-zero effects', 'Holm example'],
          'new_orchestrator_local_authoring_execution': False, 'no_scientific_runs_counted': True})




def official_test_mse(e, folder):
    import numpy as np
    import torch
    from diffusers import UNet2DModel, DDPMScheduler
    path = folder / 'OFFICIAL_TEST_EMA_MSE.json'
    checkpoint = folder / 'checkpoints/state.pt'
    binding = {'checkpoint_sha256': sha(checkpoint), 'protocol_binding': read(ROOT / 'PROTOCOL.json')['binding']}
    if path.exists():
        result = read(path)
        require(result['binding'] == binding and sha(folder / 'official_test_mse.npy') == result['per_image_sha256'], 'Saved test payload differs')
        return result
    progress('Final EMA denoising assessment on all 10000 official CIFAR10 test images: ' + str(folder.relative_to(ROOT)))
    assets = read(ROOT / 'MODELS.json')[MODEL_ID]['path']
    model = UNet2DModel.from_config(UNet2DModel.load_config(assets)).to('cuda').float()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['ema'], strict=True); del state
    model.eval().requires_grad_(False)
    scheduler = DDPMScheduler.from_pretrained(assets, local_files_only=True)
    x = np.load(ROOT / 'cache/vision/cifar10-r32/test_x.npy', mmap_mode='r', allow_pickle=False)
    require(x.shape == (10000, 32, 32, 3) and x.dtype == np.uint8, 'Wrong official test array')
    g = torch.Generator(device='cpu').manual_seed(541211)
    values = np.empty(10000, dtype=np.float64)
    for i in range(0, 10000, 128):
        a = torch.from_numpy(np.array(x[i:i + 128], copy=True)).permute(0, 3, 1, 2).contiguous().to('cuda').float() / 127.5 - 1.
        noise = torch.randn(a.shape, generator=g, device='cpu').to('cuda')
        t = torch.randint(0, scheduler.config.num_train_timesteps, (len(a),), generator=g, device='cpu').to('cuda')
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            prediction = model(scheduler.add_noise(a, noise, t), t).sample
        errs = (prediction.float() - noise).square().flatten(1).mean(1)
        require(bool(torch.isfinite(errs).all()), 'Nonfinite official test MSE')
        values[i:i + len(a)] = errs.double().cpu().numpy()
    with (folder / 'official_test_mse.npy.partial').open('wb') as f:
        np.save(f, values, allow_pickle=False); f.flush(); os.fsync(f.fileno())
    os.replace(folder / 'official_test_mse.npy.partial', folder / 'official_test_mse.npy')
    result = {'binding': binding, 'n_images': 10000, 'mse': float(values.mean()), 'noise_seed': 541211,
              'weights': 'EMA', 'selected_using_test': False, 'feedback_used_test': False,
              'per_image_sha256': sha(folder / 'official_test_mse.npy')}
    put(path, result)
    require(sha(checkpoint) == binding['checkpoint_sha256'], 'Test assessment modified its checkpoint')
    del model, values; gc.collect(); torch.cuda.empty_cache()
    return result




def job(seed, method):
    e = engine(); e.runtime()
    e.execute_job('diffusion', method, seed)
    folder = ROOT / 'diffusion' / ('seed' + str(seed)) / method
    summary = read(folder / 'summary.json')
    require(summary['role'] == PROTOCOL['role'], 'Wrong final result role')
    if method in ('macc_lite', 'full_macc'):
        require(summary['controller_event_count'] == 200, 'Final controller event coverage is incomplete')
    if method == 'full_macc': require(summary['learned_policy_updates'] >= 100, 'Final Full MACC did not execute sufficient genuine updates')
    import numpy as np
    images = np.load(folder / 'artifacts/samples_uint8.npy', mmap_mode='r', allow_pickle=False)
    require(images.shape == (50000, 32, 32, 3) and images.dtype == np.uint8, 'Saved generation payload is incomplete')
    del images
    test = official_test_mse(e, folder)
    put(folder / 'CONFIRMED.json', {'status': 'COMPLETE', 'binding': read(ROOT / 'PROTOCOL.json')['binding'],
         'summary_sha256': sha(folder / 'summary.json'), 'test_sha256': sha(folder / 'OFFICIAL_TEST_EMA_MSE.json'),
         'generated_samples_sha256': sha(folder / 'artifacts/samples_uint8.npy'), 'n_samples': 50000, 'time': utc()})
    progress('CONFIRMED: ' + method + '/seed' + str(seed))




def paired_stats(target, reference):
    import numpy as np
    from scipy.stats import t as student
    d = np.asarray(target, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    require(d.shape == (6,) and np.isfinite(d).all(), 'Expected six finite paired differences')
    mean = float(d.mean()); sd = float(d.std(ddof=1)); se = sd / math.sqrt(6)
    if se == 0:
        t_p = 1. if mean == 0 else 0.; half = 0.
    else:
        t_p = float(2 * student.sf(abs(mean / se), 5)); half = float(student.ppf(.975, 5) * se)
    flips = np.asarray(list(itertools.product((-1., 1.), repeat=6)))
    exact = float(np.mean(np.abs((flips * d).mean(1)) >= abs(mean) - 1e-12))
    return {'n_pairs': 6, 'mean_difference': mean, 'sd_difference': sd, 'ci95_low': mean - half, 'ci95_high': mean + half,
            'paired_t_p': t_p, 'exact_signflip_p': exact, 'target_lower_count': int((d < 0).sum()), 'target_higher_count': int((d > 0).sum())}




def holm(values):
    ordered = sorted(range(len(values)), key=lambda i: values[i]); out = [0.] * len(values); high = 0.
    for rank, i in enumerate(ordered):
        high = max(high, (len(values) - rank) * float(values[i])); out[i] = min(1., high)
    return out




def aggregate():
    import numpy as np
    rows = []
    for seed in SEEDS:
        for method in ALL_METHODS:
            folder = ROOT / 'diffusion' / ('seed' + str(seed)) / method
            marker = read(folder / 'CONFIRMED.json')
            require(marker['summary_sha256'] == sha(folder / 'summary.json'), 'Summary changed')
            require(marker['test_sha256'] == sha(folder / 'OFFICIAL_TEST_EMA_MSE.json'), 'Test record changed')
            s = read(folder / 'summary.json'); test = read(folder / 'OFFICIAL_TEST_EMA_MSE.json')
            rows.append({'seed': seed, 'method': method, 'fid': s['final']['fid'], 'kid': s['final']['kid'],
                 'inception_score': s['final']['inception_score'], 'internal_holdout_mse': s['final']['val_loss'],
                 'official_test_ema_mse': test['mse'], 'effective_rank_mid': s['final'].get('effective_rank/mid'),
                 'final_lambda_mid': s['final']['lambda/mid'], 'steps': s['steps'], 'n_samples': s['final']['n_samples'],
                 'learned_updates': s['learned_policy_updates'], 'controller_events': s['controller_event_count'],
                 'job_wall_seconds': s['wall_time_s'], 'compute': s['compute'], 'telemetry': s['telemetry']})
    contrasts = []
    for target, reference in PROTOCOL['contrasts_target_minus_reference']:
        a = [next(r['fid'] for r in rows if r['seed'] == seed and r['method'] == target) for seed in SEEDS]
        b = [next(r['fid'] for r in rows if r['seed'] == seed and r['method'] == reference) for seed in SEEDS]
        contrasts.append({'metric': 'FID', 'target': target, 'reference': reference, 'negative_favors_target': True, **paired_stats(a, b)})
    for key in ('paired_t_p', 'exact_signflip_p'):
        adjusted = holm([r[key] for r in contrasts])
        for r, value in zip(contrasts, adjusted): r[key + '_holm'] = value
    groups = []
    for method in ALL_METHODS:
        group = {'method': method, 'n_seed_streams': 6}
        for key in ('fid', 'kid', 'inception_score', 'internal_holdout_mse', 'official_test_ema_mse', 'job_wall_seconds'):
            a = np.asarray([r[key] for r in rows if r['method'] == method])
            group[key] = {'mean': float(a.mean()), 'sample_sd': float(a.std(ddof=1))}
        groups.append(group)
    put(ROOT / 'PER_SEED.json', rows); put(ROOT / 'GROUPED.json', groups); put(ROOT / 'COMPARISONS.json', contrasts)




def report(label):
    lines = ['# CAENL diffusion confirmation v1', '', 'Status: ' + label, 'UTC: ' + utc(), '',
         'Diffusion-only six-seed study. Audio was not downloaded or trained.',
         'All trained models start from the original pretrained checkpoint; pilot final checkpoints are not extended.',
         'Pretrained-reference seed variability is sampling variability, not independent pretrained models.',
         'This is not a claim of successful efficacy, complete audio validation or a submission-ready manuscript.', '']
    for name in ('PROTOCOL.json', 'ENGINE_PATCHES.json', 'ENVIRONMENT.json', 'SELF_TEST.json', 'CONFIRMATION_CHECKS.json',
                 'PER_SEED.json', 'GROUPED.json', 'COMPARISONS.json', 'PRESERVATION.json', 'FAILED.json'):
        p = ROOT / name
        if p.exists(): lines += ['## ' + name, 'SHA-256: ' + sha(p), '```json', json.dumps(sanitize(read(p)), indent=2), '```', '']
    for p in sorted(ROOT.glob('diffusion/seed*/*/summary.json')):
        lines += ['## ' + p.relative_to(ROOT).as_posix(), 'SHA-256: ' + sha(p), '```json', json.dumps(sanitize(read(p)), indent=2), '```', '']
    for name in ('runner.py', 'engine.py'):
        p = ROOT / name
        if p.exists(): lines += ['## Exact code ' + name, 'SHA-256: ' + sha(p), '````````', p.read_text(encoding='utf-8-sig'), '````````', '']
    path = ROOT / REPORT
    write(path, '\n'.join(lines) + '\n'); write(ROOT / (REPORT + '.sha256'), sha(path) + '  ' + REPORT + '\n')
    return path




def worker():
    import fcntl
    with (ROOT / 'RUN.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            prepare()
            subprocess.run([sys.executable, str(ROOT / 'runner.py'), '--checks'], check=True, timeout=900)
            report('CHECKS PASSED; FINAL EXPERIMENTS NOT COMPLETE')
            idle = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
            require(not idle.stdout.strip(), 'Another GPU process started during preparation. It was not stopped.')
            for seed in SEEDS:
                for method in ALL_METHODS:
                    subprocess.run([sys.executable, str(ROOT / 'runner.py'), '--job', method, '--seed', str(seed)], check=True)
                    report('RUNNING: ' + str(len(list(ROOT.glob('diffusion/seed*/*/CONFIRMED.json')))) + '/30 completed evaluations')
            aggregate()
            for path, expected in read(ROOT / 'PRESERVED_INPUTS.json').items():
                require(sha(path) == expected, 'A preserved pilot input changed: ' + path)
            put(ROOT / 'PRESERVATION.json', {'status': 'PASS', 'bound_pilot_files_rechecked': len(read(ROOT / 'PRESERVED_INPUTS.json')),
                    'pilot_training_repeated': 0, 'old_checkpoints_deleted': False, 'audio_downloader_started': False,
                    'scope': 'bound pilot/source/model/data records, not a re-audit of every historical ImageNet or C4 artifact'})
            target = report('DIFFUSION CONFIRMATION COMPLETE: 24 training runs, 6 reference evaluations')
            put(ROOT / 'COMPLETE.json', {'status': 'COMPLETE', 'time': utc(), 'binding': read(ROOT / 'PROTOCOL.json')['binding'],
                   'training_runs': 24, 'reference_evaluations': 6, 'generated_images_per_evaluation': 50000,
                   'report': str(target), 'report_sha256': sha(target), 'audio_complete': False, 'paper_ready': False})
            put(ROOT / 'STATUS.json', {'status': 'COMPLETE', 'time': utc(), 'message': 'DIFFUSION CONFIRMATION COMPLETE'})
            print('DIFFUSION CONFIRMATION COMPLETE\nShare: ' + str(target), flush=True)
            return 0
        except BaseException as exc:
            log_tail = ''
            log = ROOT / 'run.log'
            if log.is_file():
                with log.open('rb') as f:
                    f.seek(max(0, log.stat().st_size - 16000)); log_tail = f.read().decode('utf-8', errors='replace')
            put(ROOT / 'FAILED.json', {'status': 'FAILED', 'time': utc(), 'error': repr(exc), 'traceback': traceback.format_exc(),
                    'recent_log': log_tail, 'instruction': 'Share --status or this report; do not delete checkpoints or reinstall packages.'})
            try: report('STOPPED; NOT COMPLETE')
            except Exception: pass
            traceback.print_exc(); return 1




def status():
    print('SERVER:', os.uname().nodename, utc())
    live = bool(shutil.which('tmux')) and subprocess.run(['tmux', 'has-session', '-t', SESSION], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    print('TMUX ACTIVE' if live else 'NO LIVE TMUX SESSION')
    for name in ('COMPLETE.json', 'FAILED.json', 'STATUS.json'):
        if (ROOT / name).exists(): print(json.dumps(read(ROOT / name), indent=2)); break
    else: print('Not started.')
    found = list(ROOT.glob('diffusion/seed*/*/CONFIRMED.json'))
    print('Completed training runs (target 24):', sum(p.parent.name != 'pretrained_reference' for p in found))
    print('Completed reference evaluations (target 6):', sum(p.parent.name == 'pretrained_reference' for p in found))
    print('Report:', ROOT / REPORT)
    if shutil.which('nvidia-smi'):
        subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', '--format=csv'], check=False)




def start(resume=False):
    require(sys.platform.startswith('linux'), 'Run the supplied command from Mac so it connects to IBM.')
    require(Path('/mnt/caenl').is_mount(), 'Persistent volume not mounted. No mounting or formatting attempted.')
    require(shutil.which('tmux'), 'Existing tmux is missing; no software installed.')
    require(not ROOT.is_symlink(), 'Output root cannot be a symlink')
    ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    import fcntl
    with (ROOT / 'START.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if (ROOT / 'COMPLETE.json').exists(): status(); return
        if subprocess.run(['tmux', 'has-session', '-t', SESSION], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            print('ALREADY RUNNING; no duplicate launched.'); status(); return
        require(resume or not (ROOT / 'FAILED.json').exists(), 'Previous attempt stopped. Share --status before resuming.')
        require(sha(PILOT / 'runner.py') == PILOT_RUNNER_SHA, 'Required completed pilot launcher missing or changed')
        require(shutil.disk_usage(ROOT).free >= 80 * 1024**3, '80 GiB free required. No files deleted.')
        query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        require(not query.stdout.strip(), 'Another GPU process is running. It was not stopped.')
        candidates = [Path.home() / 'caenl-v5.4.3/caenl_revision_pipeline_v5.4.3_c4_confirmatory/.venv/bin/python',
                      Path.home() / 'caenl-src/caenl_revision_pipeline_v5.3_ibm_rhel_l40s/.venv/bin/python']
        python = next((p for p in candidates if p.is_file()), None)
        require(python is not None, 'Existing CAENL Python environment missing; no installation attempted.')
        raw = Path(__file__).read_bytes(); ast.parse(raw.decode('utf-8-sig'))
        target = ROOT / 'runner.py'
        if target.exists(): require(target.read_bytes() == raw, 'A different launcher already owns this output root.')
        else: target.write_bytes(raw)
        if resume and (ROOT / 'FAILED.json').exists():
            os.replace(ROOT / 'FAILED.json', ROOT / ('failure_before_resume_' + str(time.time_ns()) + '.json'))
        env = 'exec env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 '
        env += 'TORCH_HOME=' + shlex.quote(str(ROOT / 'assets/torch')) + ' '
        env += 'HF_HOME=' + shlex.quote(str(ROOT / 'assets/hf_home')) + ' '
        env += 'XDG_CACHE_HOME=' + shlex.quote(str(ROOT / 'cache/xdg')) + ' '
        cmd = env + shlex.quote(str(python)) + ' ' + shlex.quote(str(target)) + ' --worker >> ' + shlex.quote(str(ROOT / 'run.log')) + ' 2>&1'
        subprocess.run(['tmux', 'new-session', '-d', '-s', SESSION, 'bash -c ' + shlex.quote(cmd)], check=True)
        print('STARTED:', SESSION)
        print('First: source/data integrity and GPU checks. Then: NEW six-seed diffusion confirmation.')
        print('No AudioCaps download or old campaign restart. No package installation.')
        print('Status: python3 ~/caenl_diffusion_confirm_v1.py --status')




def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--start', action='store_true'); g.add_argument('--resume', action='store_true')
    g.add_argument('--status', action='store_true'); g.add_argument('--worker', action='store_true')
    g.add_argument('--checks', action='store_true'); g.add_argument('--job', choices=ALL_METHODS)
    p.add_argument('--seed', type=int, choices=SEEDS)
    a = p.parse_args()
    if a.status: status()
    elif a.worker: return worker()
    elif a.checks: checks()
    elif a.job:
        require(a.seed is not None, 'Missing seed for internal job'); job(a.seed, a.job)
    else: start(a.resume)
    return 0




if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        print('STOPPED:', exc, file=sys.stderr)
        raise SystemExit(1)
# END OF CAENL DIFFUSION CONFIRMATION V1