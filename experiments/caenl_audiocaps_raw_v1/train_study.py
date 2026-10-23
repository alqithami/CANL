#!/usr/bin/env python3
"""Controlled raw-source AudioCaps follow-up. No downloads or historical campaign execution.

Only the unchanged reference/src audio/core/utils modules are imported. The historical
configuration and terminal-credit correction are ported here without importing its engine.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
AST_NAME = 'MIT/ast-finetuned-audioset-10-10-0.4593'
METHODS = ('baseline', 'tuned_fixed', 'macc_lite', 'full_macc')
TUNE_SEEDS = tuple(range(81001, 81004))
CONFIRM_SEEDS = tuple(range(82001, 82011))
LAMBDAS = (0., .001, .003, .01, .03, .05)
METRICS = ('cider_d', 'bleu1', 'bleu2', 'bleu3', 'bleu4', 'rouge_l', 'meteor', 'spice', 'spider')
ROOT: Path


def require(ok, message):
    if not bool(ok):
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def safe_child(directory, relative):
    rel = Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe relative path: ' + str(rel))
    path = Path(directory) / rel
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(Path(directory).resolve()), 'Missing/unsafe file: ' + str(path))
    return path


def put(path, value):
    path = Path(path)
    require(path.resolve().is_relative_to(ROOT.resolve()) and not path.is_symlink(), 'Unsafe output: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.writing_', delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded(value)); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def freeze(path, value):
    if Path(path).exists():
        require(read(path) == value, 'Frozen input changed: ' + str(path))
    else:
        put(path, value)


@contextmanager
def lock(path, *, blocking=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise RuntimeError('Already owned: ' + str(path)) from exc
        yield


def torch_save(path, value):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.writing_', delete=False) as handle:
        temporary = Path(handle.name)
        torch.save(value, handle); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def tensor_hash(model_or_state):
    import torch
    state = model_or_state.state_dict() if hasattr(model_or_state, 'state_dict') else model_or_state
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def source_setup():
    provenance = read(HERE / 'reference/SOURCE_PROVENANCE.json')
    require(provenance['commit'] == '8b64be063e92d9ec94c1249ff791da262f755e68', 'Reference revision differs')
    for row in provenance['files']:
        path = safe_child(HERE / 'reference', row['path'])
        require(sha(path) == row['sha256'], 'Reference source changed: ' + row['path'])
    sys.path.insert(0, str(HERE / 'reference/src'))
    sys.path.insert(0, str(ROOT / 'metric_vendor'))
    return {'provenance_sha256': sha(HERE / 'reference/SOURCE_PROVENANCE.json'), 'commit': provenance['commit'],
            'files': {row['path']: row['sha256'] for row in provenance['files']}}


def runtime(gpu=0, cpu=False):
    import torch
    require(sys.version_info[:2] == (3, 12), 'Expected Python 3.12')
    require(str(torch.__version__) == '2.6.0+cu124' and str(torch.version.cuda) == '12.4', 'Expected torch 2.6.0+cu124')
    require(importlib.metadata.version('transformers') == '5.15.1', 'Expected transformers 5.15.1')
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    if cpu:
        return torch.device('cpu')
    require(gpu == 0, '--gpu is logical 0; coordinator must set CUDA_VISIBLE_DEVICES')
    require(torch.cuda.is_available(), 'CUDA unavailable')
    torch.cuda.set_device(gpu)
    return torch.device('cuda', gpu)


def environment_record():
    names = ('torch', 'transformers', 'numpy', 'scipy', 'soundfile', 'huggingface-hub')
    return {'python': platform.python_version(), 'packages': {name: importlib.metadata.version(name) for name in names},
            'java_version': subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT, text=True).strip(),
            'JAVA_HOME': os.environ.get('JAVA_HOME'), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}


def verify_assets():
    models = read(ROOT / 'assets/MODELS_FROZEN.json')
    for repo, short in ((AST_NAME, 'ast'), ('gpt2', 'gpt2')):
        require(repo in models and models[repo]['files'], 'Missing frozen pretrained assets: ' + repo)
        for relative, expected in models[repo]['files'].items():
            require(sha(safe_child(ROOT / 'assets' / short, relative)) == expected, 'Pretrained asset changed: ' + relative)
    metric_manifest = ROOT / 'METRIC_COPY.json'
    require(metric_manifest.is_file(), 'Missing METRIC_COPY.json from exact historical metric copy')
    metric = read(metric_manifest)
    require(metric.get('files'), 'Empty metric binding')
    for relative, expected in metric['files'].items():
        require(sha(safe_child(ROOT / 'metric_vendor', relative)) == expected, 'Historical metric asset changed: ' + relative)
    java_home = ROOT / 'runtime/java8'
    require(os.environ.get('JAVA_HOME') and Path(os.environ['JAVA_HOME']).resolve() == java_home.resolve(), 'Set JAVA_HOME to root/runtime/java8')
    java = Path(shutil.which('java') or '/nonexistent')
    require(java.resolve() == (java_home / 'bin/java').resolve(), 'PATH must select the frozen Java 8 runtime')
    version = subprocess.check_output([str(java), '-version'], stderr=subprocess.STDOUT, text=True)
    require('1.8.' in version, 'Historical caption metrics require Java 8')
    java_files = {p.relative_to(java_home).as_posix(): sha(p) for p in sorted(java_home.rglob('*')) if p.is_file()}
    require(java_files, 'Empty Java runtime')
    return {'models_manifest_sha256': sha(ROOT / 'assets/MODELS_FROZEN.json'),
            'models': {repo: {'files': models[repo]['files'], 'snapshot_revision': models[repo].get('snapshot_revision')} for repo in (AST_NAME, 'gpt2')},
            'metric_copy_sha256': sha(metric_manifest), 'metric_files': metric['files'], 'java_runtime': java_files, 'java_version': version.strip()}


def verify_data():
    ready_path = ROOT / 'data/DATA_READY.json'
    ready = read(ready_path)
    require(ready.get('status') == 'DATA_READY', 'Raw-source data is not ready')
    require(not ready.get('blocking_issues', []), 'Dataset has blocking issues')
    manifest = ROOT / 'data/manifest.jsonl'
    require(sha(manifest) == ready['manifest_sha256'], 'Manifest differs from DATA_READY')
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    for name, key in (('manifest_meta.json', 'manifest_meta_sha256'), ('AUDIT.jsonl', 'audit_sha256'), ('EXCLUSIONS.jsonl', 'exclusions_sha256'), ('SELECTION_REPORT.json', 'selection_report_sha256')):
        if key in ready:
            require(sha(ROOT / 'data' / name) == ready[key], 'Readiness artifact changed: ' + name)
    require(ready.get('original_bytes_verified') is True and ready.get('eligibility_fixed_before_model_outcomes') is True, 'Unverified/outcome-dependent data readiness')
    require(ready.get('coverage') and all(float(value) >= .8 for value in ready['coverage'].values()), 'Coverage below fixed 80% gate')
    counts = dict(Counter(row['split'] for row in rows))
    require(set(counts) == {'train', 'valid', 'test'} and counts['train'] >= 32 and counts['valid'] > 64 and counts['test'] > 0, 'Invalid split sizes')
    require(ready['splits'] == counts and ready['n_clips'] == len(rows), 'Readiness split counts differ')
    require(len({r['id'] for r in rows}) == len(rows), 'Duplicate manifest IDs')
    videos, pcm = defaultdict(set), defaultdict(set)
    for row in rows:
        path = Path(row['path'])
        require(path.is_absolute() and path.is_file() and not path.is_symlink() and path.resolve().is_relative_to((ROOT / 'data').resolve()), 'Unsafe canonical audio path')
        require(path.stat().st_size == row['bytes'] and 8 <= row.get('duration_seconds', row.get('duration', -1)) <= 12, 'Canonical clip size/duration differs')
        require(len(row['captions']) == (1 if row['split'] == 'train' else 5), 'Caption/reference count differs')
        require(not row.get('waveform_conflict', False), 'Waveform conflict retained')
        videos[row['youtube_id']].add(row['split'])
        if not row.get('all_zero_waveform', row.get('canonical_all_zero', True)):
            pcm[row.get('pcm_float32_sha256', row.get('canonical_sha256'))].add(row['split'])
    require(all(len(s) == 1 for s in videos.values()) and all(len(s) == 1 for s in pcm.values()), 'Cross-split video/PCM leakage')
    preparation = read(ROOT / 'data/PREPARATION_BINDING.json')
    meta = read(ROOT / 'data/manifest_meta.json')
    require(preparation['binding_sha256'] == ready['binding_sha256'] == meta['binding_sha256'], 'Preparation source binding differs')
    require(preparation['archive_sha256'] == ready['archive_sha256'] and preparation['policy']['duration_seconds_inclusive'] == [8., 12.] and preparation['policy']['minimum_coverage_per_split'] == .8, 'Raw source/policy differs')
    require(preparation['implementation_sha256'] == sha(HERE / 'prepare_data.py'), 'Data preparation implementation differs')
    source_path = ROOT / 'incoming/SOURCE_READY.json'
    archive = ROOT / 'incoming/audiocaps_raw_audio.zip'
    require(sha(source_path) == meta['source']['source_ready_sha256'] and archive.is_file() and archive.stat().st_size == ready['archive_bytes'], 'Raw archive source receipt/size changed')
    recorded_stat = meta['source']['stat']
    require(archive.stat().st_mtime_ns == recorded_stat['mtime_ns'] and archive.stat().st_ino == recorded_stat['inode'], 'Raw archive changed after full audited hash')
    return rows, {'ready_sha256': sha(ready_path), 'ready': ready, 'manifest_sha256': sha(manifest), 'counts': counts}


def input_binding():
    source = source_setup()
    rows, data = verify_data()
    assets = verify_assets()
    code_files = ('train_study.py', 'study_features.py', 'study_stats.py', 'protocol.json', 'prepare_data.py')
    record = {'schema': 1, 'code': {name: sha(HERE / name) for name in code_files}, 'reference': source,
              'data': data, 'assets': assets, 'preprocessing': read(HERE / 'protocol.json')['features'],
              'environment': {key: value for key, value in environment_record().items() if key != 'cuda_visible_devices'}}
    record['sha256'] = digest(record)
    with lock(ROOT / '.input_binding.lock', blocking=True):
        freeze(ROOT / 'INPUT_BINDING.json', record)
    return rows, record


def configure(stage, method, seed, value=None):
    """Literal audio-side port of historical engine.configure; only study routing/budget changes."""
    require(stage in ('tuning', 'confirm', 'smoke'), 'Invalid stage')
    regularizer = 'fixed_dcr' if method in ('tuned_fixed', 'fixed_dcr') else ('none' if method == 'baseline' else method)
    cfg = {'seed': seed, 'job_id': stage + '_' + method + '_s' + str(seed), 'phase': stage,
       'paths': {'data_root': str(ROOT / 'data'), 'cache_root': str(ROOT / 'features')},
       'method': {'name': method, 'regularizer': regularizer, 'dcr_lambda': 0.0 if method == 'baseline' else (float(value) if regularizer == 'fixed_dcr' else .01)},
       'collapse': {'truncated_spectral_rank': 64, 'covariance_ema_decay': .99},
       'macc_lite': {'lambda_initial': .01, 'lambda_min': 0., 'lambda_max': .05, 'eta_lambda': .001, 'error_mode': 'absolute_log_ratio', 'deadband': .1},
       'full_macc': {'lambda_min': 0., 'lambda_max': .05, 'hidden_dimensions': [128, 128],
         'action_space': {'lambda_delta_choices': [-.005, -.0025, 0., .0025, .005], 'threshold_quantiles': [.9]},
         'policy_learning_rate': .0003, 'exploration_epsilon_start': .10, 'exploration_epsilon_end': .02,
         'reward_baseline_ema': .05, 'entropy_coefficient': .001,
         'reward': {'validation_delta_weight': 1., 'validation_delta_scale': 10., 'query_penalty_beta': 0., 'target_deviation_gamma': .1, 'value_delta_clip': 2.}},
       'reproducibility': {'save_pip_freeze': False, 'telemetry_interval_s': 10},
       'audio': {'dataset': 'audiocaps', 'manifest': str(ROOT / 'data/manifest.jsonl'),
          'manifest_min_coverage': .80, 'manifest_verify_hashes': 'all', 'backbone': str(ROOT / 'assets/ast'),
          'tokens_per_window': 64, 'max_windows': 1, 'configured_rank': 16, 'target_effective_rank': 16,
          'monitoring_cadence_steps': 50, 'controller_error_mode': 'absolute_log_ratio',
          'captioning': {'decoder': str(ROOT / 'assets/gpt2'), 'prefix_length': 8, 'encoder_layers': 2, 'heads': 8,
             'epochs': 10, 'batch_size': 32, 'lr': .0001, 'decoder_lr': .00005, 'weight_decay': .01,
             'warmup_steps': 100, 'max_caption_tokens': 48, 'ctrl_val_clips': 64, 'decode_batch_size': 32,
             'max_new_tokens': 30, 'beam_size': 3, 'java_metrics': True, 'spice': True, 'official_metrics': True,
             'log_every_steps': 50, 'checkpoint_every_steps': 250}}}
    cfg['input_binding'] = read(ROOT / 'INPUT_BINDING.json')['sha256'] if (ROOT / 'INPUT_BINDING.json').exists() else 'smoke-unbound'
    require(stage == 'smoke' or cfg['input_binding'] != 'smoke-unbound', 'Missing scientific INPUT_BINDING')
    if stage == 'confirm':
        cfg['selection_sha256'] = sha(ROOT / 'SELECTED_FIXED.json')
    cfg['binding'] = digest(cfg)
    return cfg


def terminal_loop_class():
    """Compile the unchanged historical TerminalSafeLoop class, without engine side effects."""
    import torch
    from caenl.core.control_loop import ControlLoop
    from caenl.core.controllers import FullMACC
    path = HERE / 'reference/experiments/caenl-audiocaps-confirm-v1/engine.py'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'install_window_loop')
    cls = next(node for node in function.body if isinstance(node, ast.ClassDef) and node.name == 'TerminalSafeLoop')
    namespace = {'torch': torch, 'ControlLoop': ControlLoop, 'FullMACC': FullMACC, 'require': require, 'math': math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace['TerminalSafeLoop']


def new_model(seed):
    import torch
    from transformers import AutoModelForCausalLM
    from caenl.audio.captioning import PrefixCaptioner
    from caenl.utils.seeding import derive_seed
    torch.manual_seed(derive_seed(seed, 'init'))
    decoder = AutoModelForCausalLM.from_pretrained(str(ROOT / 'assets/gpt2'), local_files_only=True).float().cpu()
    require(int(decoder.config.n_embd) == 768, 'Unexpected GPT-2 size')
    return PrefixCaptioner(768, decoder, 768, prefix_len=8, enc_layers=2, n_heads=8).float().cpu()


def initialize_seed(seed, binding, *, smoke_initial=False):
    import torch
    from caenl.utils.seeding import rng_state_dict
    folder = ROOT / 'initial' / (('smoke_s' if smoke_initial else 's') + str(seed)); folder.mkdir(parents=True, exist_ok=True)
    with lock(folder / '.lock'):
        path = folder / 'initial.pt'
        if path.exists():
            if (folder / 'INITIAL.json').exists():
                record = read(folder / 'INITIAL.json')
                require(record['input_binding'] == binding['sha256'] and sha(path) == record['file_sha256'], 'Immutable initial checkpoint changed')
                return record
            # Atomic checkpoint publication may precede the metadata rename at a
            # kill boundary. Recover metadata from that same binary, never rebuild.
            recovered = torch.load(path, map_location='cpu', weights_only=False)
            require(recovered['seed'] == seed and recovered['input_binding'] == binding['sha256'] and tensor_hash(recovered['model']) == recovered['model_weights_sha256'], 'Orphan initial checkpoint has wrong binding/state')
            record = {'seed': seed, 'input_binding': binding['sha256'], 'file_sha256': sha(path), 'model_weights_sha256': recovered['model_weights_sha256'],
                      'constructed_once_on': 'cpu', 'architecture': 'unchanged PrefixCaptioner(768,GPT2,768,prefix_len=8,enc_layers=2,n_heads=8)',
                      'prior_initialization_cause': 'unknown; historical AudioTokenEncoder.pos uses zeros followed by trunc_normal_'}
            freeze(folder / 'INITIAL.json', record)
            return record
        require(not (folder / 'INITIAL.json').exists(), 'Orphan initialization metadata')
        model = new_model(seed)
        weights = tensor_hash(model)
        torch_save(path, {'model': model.state_dict(), 'rng_after_construction': rng_state_dict(), 'seed': seed,
                          'input_binding': binding['sha256'], 'model_weights_sha256': weights})
        record = {'seed': seed, 'input_binding': binding['sha256'], 'file_sha256': sha(path), 'model_weights_sha256': weights,
                  'constructed_once_on': 'cpu', 'architecture': 'unchanged PrefixCaptioner(768,GPT2,768,prefix_len=8,enc_layers=2,n_heads=8)',
                  'prior_initialization_cause': 'unknown; historical AudioTokenEncoder.pos uses zeros followed by trunc_normal_'}
        freeze(folder / 'INITIAL.json', record)
        del model
        return record


def pooled_nll(records):
    require(records and all(r['target_tokens'] > 0 and math.isfinite(r['mean_token_nll']) for r in records), 'Invalid NLL records')
    count = sum(r['target_tokens'] for r in records)
    total = math.fsum(r['mean_token_nll'] * r['target_tokens'] for r in records)
    return total / count, count, total


def cider_only(candidates, references):
    from caenl.audio.metrics import _coco_inputs
    from pycocoevalcap.cider.cider import Cider
    with lock(ROOT / '.metric.lock', blocking=True):
        gts, res = _coco_inputs(candidates, references, ptb=True)
        score = float(Cider().compute_score(gts, res)[0])
    require(math.isfinite(score), 'Nonfinite official development CIDEr-D')
    return {'cider_d': score, 'metrics_source': 'pycocoevalcap', 'tokenizer': 'ptb'}


def full_metrics(candidates, references):
    from caenl.audio.metrics import caption_metrics
    with lock(ROOT / '.metric.lock', blocking=True):
        result = caption_metrics(candidates, references, java=True, spice=True, official=True)
    require(result['metrics_source'] == 'pycocoevalcap' and result['tokenizer'] == 'ptb', 'Metric fallback refused')
    require(all(math.isfinite(float(result[k])) for k in METRICS), 'Incomplete caption metrics')
    return result


def job_path(stage, method, seed, value):
    condition = ('lambda' + format(float(value), '.3g')) if stage == 'tuning' else method
    return ROOT / stage / condition / ('s' + str(seed))


def check_done(folder, binding):
    done = read(folder / 'DONE.json')
    require(done['status'] == 'COMPLETE' and done['binding'] == binding, 'Completed job binding differs')
    for rel, expected in done['files'].items():
        require(sha(safe_child(folder, rel)) == expected, 'Completed payload changed: ' + rel)
    return done


def training_class(fixture=None):
    import numpy as np
    import torch
    from caenl.audio.captioning import CaptioningJob
    from caenl.audio.common import HFTokenizer, split_indices, cosine_lr
    from caenl.utils.seeding import derive_seed, load_rng_state
    from study_features import verify_features
    Loop = terminal_loop_class()

    class FinalAudio(CaptioningJob):
        def setup(self):
            if fixture is None:
                rows, _ = verify_data()
                binding = read(ROOT / 'INPUT_BINDING.json')
                feat = verify_features(ROOT, rows, binding)
            else:
                rows, binding, feat = fixture['rows'], fixture['binding'], fixture['features']
            self.rows, self.features = feat['rows'], feat['features']
            self.n_tokens = np.load(Path(feat['cache_dir']) / 'n_tokens.npy', allow_pickle=False)
            self.splits = split_indices(self.rows)
            valid = sorted(map(int, self.splits['valid']), key=lambda index: self.rows[index]['id'])
            self.ctrl_val, self.report_val = (np.asarray(valid[:64]), np.asarray(valid[64:])) if self.cfg['phase'] != 'smoke' else (np.asarray([], dtype=np.int64), np.asarray(valid, dtype=np.int64))
            self.pairs = [(int(i), caption) for i in self.splits['train'] for caption in self.rows[int(i)]['captions']]
            self.bs, self.epochs, self.max_len = 32, 10, 48
            self.steps_per_epoch = len(self.pairs) // self.bs
            self.total_steps = self.steps_per_epoch * self.epochs
            self.amp = self.device.type == 'cuda'
            self.tok = HFTokenizer(str(ROOT / 'assets/gpt2'))
            self.model = new_model(self.ctx.seed)
            initial = ROOT / 'initial' / (('smoke_s' if fixture is not None else 's') + str(self.ctx.seed))
            rec = read(initial / 'INITIAL.json')
            require(rec['input_binding'] == self.cfg['input_binding'] and sha(initial / 'initial.pt') == rec['file_sha256'], 'Initial binary changed')
            payload = torch.load(initial / 'initial.pt', map_location='cpu', weights_only=False)
            require(payload['seed'] == self.ctx.seed and payload['input_binding'] == self.cfg['input_binding'], 'Initial identity differs')
            self.model.load_state_dict(payload['model'], strict=True)
            self.initial_hash = tensor_hash(self.model)
            require(self.initial_hash == rec['model_weights_sha256'] == payload['model_weights_sha256'], 'Paired initial tensor hash mismatch')
            load_rng_state(payload['rng_after_construction'])
            self.model = self.model.to(self.device).float(); self.decoder = self.model.decoder
            if self.cfg['phase'] == 'smoke':
                self.pairs = self.pairs[:2]; self.bs = 2; self.epochs = 2; self.steps_per_epoch = 1; self.total_steps = 2
            self.loop = Loop({'prefix': 768, 'audio_enc': 768}, self.method['regularizer'], self.method['dcr_lambda'], self.cfg,
                             self.device, derive_seed(self.ctx.seed, 'controller'), self.total_steps, k=64, configured_rank=16,
                             target_effective_rank=16, cadence=50, error_mode='absolute_log_ratio', ema_decay=.99)
            decoder_parameters = list(self.decoder.parameters()); decoder_ids = {id(p) for p in decoder_parameters}
            head_parameters = [p for p in self.model.parameters() if id(p) not in decoder_ids]
            self.optimizer = torch.optim.AdamW([{'params': head_parameters, 'lr': .0001}, {'params': decoder_parameters, 'lr': .00005}], weight_decay=.01)
            self.base_lrs = [.0001, .00005]
            self.rng = np.random.default_rng(derive_seed(self.ctx.seed, 'order')); self.step = 0
            sets = [set(map(int, values)) for values in (self.splits['train'], self.ctrl_val, self.report_val, self.splits['test'])]
            require(all(not sets[i].intersection(sets[j]) for i in range(4) for j in range(i + 1, 4)), 'Train/controller/development/test overlap')
            if self.cfg['phase'] != 'smoke':
                require(len(self.ctrl_val) == 64 and len(self.report_val) == len(self.splits['valid']) - 64, 'Evaluation routing differs')
            self.ctx.save_artifact_json('EVALUATION_SPLITS.json', {
                'controller_ids': [self.rows[int(i)]['id'] for i in self.ctrl_val],
                'development_ids': [self.rows[int(i)]['id'] for i in self.report_val],
                'official_test_ids': [self.rows[int(i)]['id'] for i in self.splits['test']],
                'controller_rule': 'first 64 lexicographically sorted valid IDs',
                'test_used_for_updates_or_checkpoint_selection': False})
            self.ctx.event('setup', steps=self.total_steps, training_pairs=len(self.pairs), initial_hash=self.initial_hash)

        def batch(self, pairs):
            if torch.is_grad_enabled() and self.model.training:
                seed = derive_seed(self.ctx.seed, 'audio_step_dropout', self.step)
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            return super().batch(pairs)

        def restore(self):
            result = super().restore()
            require(0 <= self.step <= self.total_steps and self.loop.step == self.step, 'Restored model/controller step mismatch')
            return result

        def assert_loss_route(self, clips):
            is_test = np.array_equal(clips, self.splits['test'])
            require(not is_test or (self.cfg['phase'] == 'confirm' and self.step == self.total_steps), 'Official test NLL forbidden before final confirmation')
            require(self.cfg['phase'] != 'smoke' or not is_test, 'Smoke cannot evaluate test')
            return 'test' if is_test else ('development' if np.array_equal(clips, self.report_val) else 'controller')

        @torch.no_grad()
        def val_loss(self, clips):
            label = self.assert_loss_route(clips)
            previous = self.model.training; self.model.eval(); records = []
            pairs = [(int(i), caption) for i in clips for caption in self.rows[int(i)]['captions']]
            try:
                for start in range(0, len(pairs), self.bs):
                    selected = pairs[start:start + self.bs]
                    x, mask, ids, am = self.batch(selected)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.amp):
                        value, _ = self.model(x, mask, ids, am)
                    records.append({'mean_token_nll': float(value), 'target_tokens': int(am[:, 1:].sum()), 'caption_count': len(selected)})
            finally:
                self.model.train(previous)
            value, tokens, total = pooled_nll(records)
            if self.cfg['phase'] == 'confirm' and self.step == self.total_steps and label in ('test', 'development'):
                self.ctx.save_artifact_json('token_nll_' + label + '.json', {'split': label, 'step': self.step, 'clip_count': len(clips),
                    'caption_count': len(pairs), 'target_tokens': tokens, 'sum_token_nll': total, 'pooled_token_nll': value, 'batches': records})
            return value

        @torch.no_grad()
        def decode_split(self, clips):
            expected = self.splits['test'] if self.cfg['phase'] == 'confirm' else self.report_val
            require(self.cfg['phase'] != 'smoke' and self.step == self.total_steps and np.array_equal(clips, expected), 'Caption decoding route forbidden')
            return super().decode_split(clips)

        @torch.no_grad()
        def shuffled_audio(self):
            require(self.cfg['phase'] == 'confirm' and self.step == self.total_steps, 'Shuffle diagnostic reserved for final confirmation')
            previous = self.model.training; self.model.eval()
            shift = 1 + derive_seed(self.ctx.seed, 'development_audio_shuffle') % (len(self.report_val) - 1)
            shuffled = np.roll(self.report_val, shift)
            pairs = [(int(j), caption) for i, j in zip(self.report_val, shuffled) for caption in self.rows[int(i)]['captions']]
            records = []
            try:
                for start in range(0, len(pairs), self.bs):
                    x, mask, ids, am = self.batch(pairs[start:start + self.bs])
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.amp):
                        value, _ = self.model(x, mask, ids, am)
                    records.append({'mean_token_nll': float(value), 'target_tokens': int(am[:, 1:].sum())})
            finally:
                self.model.train(previous)
            value, count, _ = pooled_nll(records)
            return {'split': 'development', 'cyclic_shift': int(shift), 'mismatched_audio_clips': len(shuffled),
                    'development_token_nll': value, 'target_tokens': count, 'training_updates': 0, 'favorable_direction_required': False}

        @torch.no_grad()
        def replay_probe(self):
            """Small deterministic development-only full-logit and pooled-loss checkpoint probe."""
            previous = self.model.training; self.model.eval()
            pairs = [(int(i), self.rows[int(i)]['captions'][0]) for i in self.report_val[:2]]
            try:
                x, mask, ids, am = self.batch(pairs)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.amp):
                    prefix, _ = self.model.prefix(x, mask)
                    emb = torch.cat([prefix, self.decoder.get_input_embeddings()(ids)], dim=1)
                    full_am = torch.cat([torch.ones(prefix.shape[:2], dtype=am.dtype, device=self.device), am], dim=1)
                    logits = self.decoder(inputs_embeds=emb, attention_mask=full_am).logits[:, self.model.prefix_len:-1]
                    loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction='none')
                    nll = (loss * am[:, 1:].reshape(-1).float()).sum() / am[:, 1:].sum()
                return {'ids': [self.rows[i]['id'] for i, _ in pairs], 'target_tokens': int(am[:, 1:].sum()),
                        'token_nll': float(nll), 'logits': logits.detach().cpu(), 'checkpoint_tensor_hash': tensor_hash(self.model)}
            finally:
                self.model.train(previous)

        def train_steps(self, stop_after=None):
            self.model.train(); start_step = self.step; started = time.perf_counter(); order = None; order_epoch = None
            while self.step < self.total_steps:
                epoch, batch_index = divmod(self.step, self.steps_per_epoch)
                if order_epoch != epoch:
                    order = np.random.default_rng(derive_seed(self.ctx.seed, 'epoch', epoch)).permutation(len(self.pairs)); order_epoch = epoch
                pairs = [self.pairs[int(j)] for j in order[batch_index * self.bs:(batch_index + 1) * self.bs]]
                x, mask, ids, am = self.batch(pairs)
                scale = cosine_lr(self.step, self.total_steps, 1., 100)
                for group, base in zip(self.optimizer.param_groups, self.base_lrs):
                    group['lr'] = base * scale
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.amp):
                    ce, features = self.model(x, mask, ids, am)
                regularization, per_layer = self.loop.dcr_loss(features)
                loss = ce + regularization
                require(torch.isfinite(loss), 'Nonfinite training loss')
                self.optimizer.zero_grad(set_to_none=True); loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
                require(float(norm) > 0, 'No caption-model gradient')
                if self.cfg['phase'] == 'smoke':
                    require(all(p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.norm()) > 0 for p in
                                (self.model.queries, self.model.encoder.proj.weight, self.decoder.get_input_embeddings().weight)), 'Missing/nonfinite smoke gradient')
                self.optimizer.step()
                info = self.loop.update({key: value.detach() for key, value in features.items()},
                    value_fn=lambda: -self.val_loss(self.ctrl_val), extra_state=(self.step / max(1, self.total_steps),))
                self.step += 1
                if info:
                    self.ctx.log_metrics({'kind': 'controller', 'step': self.step, **{k: v for k, v in info.items() if isinstance(v, (int, float)) or v is None}})
                if self.step % 50 == 0 or self.cfg['phase'] == 'smoke':
                    self.ctx.log_metrics({'kind': 'train', 'step': self.step, 'ce': float(ce.detach()), 'dcr': float(regularization.detach()), 'loss': float(loss.detach()), **self.loop.record()})
                    print(self.ctx.job_id + ': ' + str(self.step) + '/' + str(self.total_steps), flush=True)
                # Tuning is scored only once at its final checkpoint. Controller feedback stays on its separate 64 IDs.
                if self.cfg['phase'] == 'confirm' and self.step % self.steps_per_epoch == 0:
                    self.ctx.log_metrics({'kind': 'eval', 'step': self.step, 'development_token_nll': self.val_loss(self.report_val)})
                if self.step % 250 == 0 or self.step == self.total_steps or (stop_after and self.step >= stop_after):
                    self.save()
                if stop_after and self.step >= stop_after:
                    break
            return {'train_time_s': time.perf_counter() - started, 'steps_executed_this_attempt': self.step - start_step}

        def run(self):
            self.setup(); self.restore(); compute = self.train_steps()
            require(self.step == self.total_steps, 'Incomplete scheduled training')
            label = 'test' if self.cfg['phase'] == 'confirm' else 'development'
            clips = self.splits['test'] if label == 'test' else self.report_val
            with self.ctx.timer('decode'):
                candidates, references = self.decode_split(clips)
            metrics = full_metrics(candidates, references) if label == 'test' else cider_only(candidates, references)
            self.ctx.save_artifact_json('captions_' + label + '.json', [{'id': self.rows[int(i)]['id'], 'candidate': c, 'references': r} for i, c, r in zip(clips, candidates, references)])
            if label == 'test':
                metrics['test_token_nll'] = self.val_loss(self.splits['test'])
                metrics['development_token_nll'] = self.val_loss(self.report_val)
                shuffle = self.shuffled_audio()
            else:
                shuffle = None
            history = self.loop.summary()['history']
            learned = sum('policy_loss' in row.get('update', {}) for row in history)
            if self.cfg['method']['name'] == 'full_macc':
                require(learned >= 20, 'Fewer than 20 actual Full-MACC policy updates')
                require(self.loop.controller._pending is None and history[-1]['new_action_drawn'] is False, 'Terminal credit or unused-action guard failed')
            if self.cfg['method']['name'] in ('macc_lite', 'full_macc'):
                require(len(history) == math.ceil(self.total_steps / 50), 'Controller event count differs')
            self.ctx.save_artifact_json('controller_trajectory.json', {'regularizer': self.loop.regularizer, 'targets': self.loop.targets, 'history': history})
            final_hash = tensor_hash(self.model)
            torch_save(self.ctx.job_dir / 'final_state.pt', {'model': {k: v.detach().float().cpu() if v.is_floating_point() else v.detach().cpu() for k, v in self.model.state_dict().items()},
                'step': self.step, 'model_weights_sha256': final_hash, 'input_binding': self.cfg['input_binding'], 'job_binding': self.cfg['binding']})
            torch_save(self.ctx.artifact_dir / 'replay_probe.pt', self.replay_probe())
            compute['steps_per_s'] = compute['steps_executed_this_attempt'] / max(compute['train_time_s'], 1e-6)
            return {'phase': self.cfg['phase'], 'task': 'audio_captioning', 'method': self.cfg['method']['name'], 'seed': self.ctx.seed,
                    'steps': self.step, 'steps_expected': self.total_steps, 'final': metrics,
                    'official_test_caption_outcomes_computed': label == 'test', 'test_clips': len(self.splits['test']) if label == 'test' else 0,
                    'evaluation_clips': len(clips), 'initial_model_weights_sha256': self.initial_hash, 'final_model_weights_sha256': final_hash,
                    'initial_file_sha256': read(ROOT / 'initial' / ('s' + str(self.ctx.seed)) / 'INITIAL.json')['file_sha256'],
                    'learned_policy_updates': learned, 'controller_event_count': len(history), 'compute': compute,
                    'shuffled_audio_diagnostic': shuffle, 'checkpoint_selection': 'final scheduled step',
                    'caption_diagnostics': {'n': len(candidates), 'empty_count': sum(not c.strip() for c in candidates), 'unique_candidate_count': len(set(candidates)), 'all_outputs_retained': True}}
    return FinalAudio


def context_class():
    from caenl.utils.jobctx import JobContext

    class BoundContext(JobContext):
        def save_checkpoint(self, name, payload):
            return super().save_checkpoint(name, {**payload, 'input_binding': self.config['input_binding'], 'job_binding': self.config['binding']})

        def load_checkpoint(self, name, map_location='cpu'):
            payload = super().load_checkpoint(name, map_location)
            if payload is not None:
                require(payload.get('input_binding') == self.config['input_binding'] and payload.get('job_binding') == self.config['binding'], 'Checkpoint source/protocol binding differs')
            return payload
    return BoundContext


def audit_saved(folder, cfg, device):
    import torch
    from caenl.utils.seeding import seed_everything
    summary = read(folder / 'summary.json')
    rows, data = verify_data(); manifest = {row['id']: row for row in rows}
    expected_steps = (data['counts']['train'] // 32) * 10
    require(summary['steps'] == summary['steps_expected'] == expected_steps and summary['seed'] == cfg['seed'] and summary['method'] == cfg['method']['name'], 'Incomplete result identity/budget')
    label = 'test' if cfg['phase'] == 'confirm' else 'development'
    require(summary['official_test_caption_outcomes_computed'] == (label == 'test'), 'Evaluation scope changed')
    captions = read(folder / 'artifacts' / ('captions_' + label + '.json'))
    split = read(folder / 'artifacts/EVALUATION_SPLITS.json')
    expected_ids = split['official_test_ids'] if label == 'test' else split['development_ids']
    require([row['id'] for row in captions] == expected_ids and len(expected_ids) == len(set(expected_ids)), 'Prediction IDs/order differ')
    for row in captions:
        require(manifest[row['id']]['split'] == ('test' if label == 'test' else 'valid') and row['references'] == manifest[row['id']]['captions'] and len(row['references']) == 5, 'Original references differ')
        require(isinstance(row['candidate'], str), 'Invalid candidate type')
    actual = full_metrics([r['candidate'] for r in captions], [r['references'] for r in captions]) if label == 'test' else cider_only([r['candidate'] for r in captions], [r['references'] for r in captions])
    for metric in (METRICS if label == 'test' else ('cider_d',)):
        require(abs(float(actual[metric]) - float(summary['final'][metric])) <= 1e-10, 'Saved-caption recomputation differs: ' + metric)
    if label == 'test':
        for name in ('test', 'development'):
            record = read(folder / 'artifacts' / ('token_nll_' + name + '.json'))
            nll, count, _ = pooled_nll(record['batches'])
            require(abs(nll - summary['final'][name + '_token_nll']) <= 1e-12 and count == record['target_tokens'], 'Saved NLL arithmetic differs')
    # This process has never held the training model: load a clean CPU construction and final binary.
    seed_everything(cfg['seed'], deterministic=True, cudnn_benchmark=False)
    torch.use_deterministic_algorithms(True, warn_only=False)
    ctx = context_class()(folder, cfg); ctx.device = device
    model = training_class()(ctx); model.setup()
    final = torch.load(folder / 'final_state.pt', map_location='cpu', weights_only=False)
    require(final['input_binding'] == cfg['input_binding'] and final['job_binding'] == cfg['binding'] and final['step'] == expected_steps, 'Final-state binding differs')
    model.model.load_state_dict(final['model'], strict=True); model.step = expected_steps
    require(tensor_hash(model.model) == summary['final_model_weights_sha256'] == final['model_weights_sha256'], 'Clean final-state tensor hash differs')
    expected = torch.load(folder / 'artifacts/replay_probe.pt', map_location='cpu', weights_only=False)
    replay = model.replay_probe()
    require(replay['ids'] == expected['ids'] and replay['target_tokens'] == expected['target_tokens'] and replay['token_nll'] == expected['token_nll'], 'Clean-checkpoint NLL replay differs')
    require(torch.equal(replay['logits'], expected['logits']), 'Clean-checkpoint full-logit replay differs')
    put(folder / 'artifacts/RESULT_AUDIT.json', {'status': 'PASS', 'split': label, 'rows': len(captions),
        'saved_caption_metrics_recomputed': True, 'checkpoint_clean_replay': True, 'checkpoint_probe_ids': replay['ids'],
        'checkpoint_probe_token_nll': replay['token_nll'], 'checkpoint_full_probe_logits_bitwise_equal': True,
        'original_references_matched': True, 'pooled_NLL_recomputed': label == 'test', 'captions_regenerated_from_checkpoint': False,
        'independent_external_reproduction': False, 'job_binding': cfg['binding'], 'input_binding': cfg['input_binding'],
        'summary_sha256': sha(folder / 'summary.json'), 'final_state_sha256': sha(folder / 'final_state.pt')})
    files = {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and not p.name.startswith('.')
             and p.name not in ('DONE.json', 'status.json', 'system.csv') and '.tmp' not in p.name}
    put(folder / 'DONE.json', {'status': 'COMPLETE', 'seed': cfg['seed'], 'method': cfg['method']['name'], 'phase': cfg['phase'],
        'binding': cfg['binding'], 'input_binding': cfg['input_binding'], 'files': files, 'time': now(), 'scheduled_steps': expected_steps})


def _run_job_locked(stage, method, seed, value, gpu, audit_only=False):
    import torch
    from caenl.utils.seeding import seed_everything
    device = runtime(gpu)
    cfg = configure(stage, method, seed, value); folder = job_path(stage, method, seed, value)
    folder.mkdir(parents=True, exist_ok=True); freeze(folder / 'job.json', cfg)
    if (folder / 'DONE.json').exists():
        check_done(folder, cfg['binding']); print('ALREADY COMPLETE ' + str(folder), flush=True); return
    if audit_only:
        audit_saved(folder, cfg, device); return
    require((ROOT / 'initial' / ('s' + str(seed)) / 'INITIAL.json').is_file(), 'Run initialize before workers')
    seed_everything(seed, deterministic=True, cudnn_benchmark=False)
    torch.use_deterministic_algorithms(True, warn_only=False)
    ctx = context_class()(folder, cfg); ctx.device = device; ctx.start()
    try:
        result = training_class()(ctx).run()
        ctx.finish(result)
    except BaseException as exc:
        ctx.fail(exc); raise


def run_job(stage, method, seed, value, gpu, audit_only=False):
    # A child owns its own job lock. This survives a coordinator/worker parent
    # disappearing and prevents a restarted worker from creating a second writer.
    folder = job_path(stage, method, seed, value)
    with lock(folder / '.job.lock'):
        return _run_job_locked(stage, method, seed, value, gpu, audit_only)


def jobs(stage):
    if stage == 'tuning':
        return [('fixed_dcr', seed, value) for value in LAMBDAS for seed in TUNE_SEEDS]
    selected = read(ROOT / 'SELECTED_FIXED.json')
    require(selected['input_binding'] == read(ROOT / 'INPUT_BINDING.json')['sha256'], 'Fixed-selection input binding differs')
    # Rotate method order by seed so each method has five runs on each GPU,
    # avoiding a fixed method-to-device assignment while retaining modulo sharding.
    return [(method, seed, selected['lambda']) for index, seed in enumerate(CONFIRM_SEEDS)
            for method in METHODS[index % len(METHODS):] + METHODS[:index % len(METHODS)]]


def execution_workers():
    workers = read(HERE / 'protocol.json')['execution']['workers']
    require(type(workers) is int and workers in (1, 2), 'Protocol requires one or two deterministic workers')
    return workers


def worker_jobs(stage, workers, worker_index):
    require(type(workers) is int and workers in (1, 2) and
            type(worker_index) is int and 0 <= worker_index < workers, 'Invalid deterministic worker assignment')
    return [job for index, job in enumerate(jobs(stage)) if index % workers == worker_index]


def worker(args):
    require(args.workers == execution_workers(), 'Worker count differs from the frozen execution protocol')
    require(type(args.worker_index) is int and 0 <= args.worker_index < args.workers, 'Invalid worker index')
    require(args.gpu == 0, 'GPU index is logical 0')
    with lock(ROOT / 'locks' / ('worker' + str(args.worker_index) + '.lock')):
        for method, seed, value in worker_jobs(args.stage, args.workers, args.worker_index):
            cfg = configure(args.stage, method, seed, value)
            folder = job_path(args.stage, method, seed, value)
            if (folder / 'DONE.json').exists():
                check_done(folder, cfg['binding']); continue
            command = [sys.executable, str(HERE / 'train_study.py'), '--root', str(ROOT), '--phase', 'worker', '--stage', args.stage,
                       '--gpu', str(args.gpu), '--run-job', '--method', method, '--seed', str(seed), '--lambda-value', repr(value)]
            # Each training child exits before a fresh audit child is created, releasing its CUDA allocation.
            subprocess.run(command, check=True)
            subprocess.run(command + ['--audit-only'], check=True)
            check_done(folder, cfg['binding'])
        put(ROOT / (args.stage + '_WORKER' + str(args.worker_index) + '_DONE.json'),
            {'status': 'COMPLETE', 'stage': args.stage, 'worker_index': args.worker_index,
             'input_binding': read(ROOT / 'INPUT_BINDING.json')['sha256'], 'time': now()})


def select_fixed():
    rows = []
    for method, seed, value in jobs('tuning'):
        folder = job_path('tuning', method, seed, value)
        cfg = configure('tuning', method, seed, value); check_done(folder, cfg['binding'])
        summary = read(folder / 'summary.json')
        require(summary['phase'] == 'tuning' and summary['official_test_caption_outcomes_computed'] is False and summary['test_clips'] == 0, 'Tuning leaked official test outcomes')
        require(not (folder / 'artifacts/captions_test.json').exists() and not (folder / 'artifacts/token_nll_test.json').exists(), 'Forbidden tuning test output')
        require(summary['steps'] == summary['steps_expected'], 'Incomplete tuning budget')
        value_score = float(summary['final']['cider_d']); require(math.isfinite(value_score), 'Nonfinite development score')
        rows.append({'lambda': value, 'seed': seed, 'development_cider_d': value_score, 'done_sha256': sha(folder / 'DONE.json')})
    means = [{'lambda': value, 'mean_development_cider_d': math.fsum(row['development_cider_d'] for row in rows if row['lambda'] == value) / len(TUNE_SEEDS)} for value in LAMBDAS]
    chosen = min(means, key=lambda item: (-item['mean_development_cider_d'], item['lambda']))
    selection = {'status': 'FROZEN', 'lambda': chosen['lambda'], 'rule': 'highest mean final-step development CIDEr-D across three tuning seeds; exact tie: smaller lambda',
                 'input_binding': read(ROOT / 'INPUT_BINDING.json')['sha256'], 'tuning_seeds': list(TUNE_SEEDS), 'confirmation_seeds': list(CONFIRM_SEEDS),
                 'candidate_means': means, 'all_18_tuning_outcomes': rows, 'official_test_outcomes_used': False}
    freeze(ROOT / 'SELECTED_FIXED.json', selection)
    print('FROZEN tuned fixed lambda=' + repr(chosen['lambda']), flush=True)


def smoke(gpu, smoke_manifest=None):
    """Two real train examples, two optimizer steps, exact checkpoint resume; zero scientific runs."""
    import gc
    import torch
    from caenl.utils.seeding import seed_everything
    device = runtime(gpu); seed = TUNE_SEEDS[0]; fixture = None
    if smoke_manifest is not None:
        import numpy as np
        from study_features import extract_features, verify_features
        smoke_manifest = Path(smoke_manifest).resolve()
        rows = [json.loads(line) for line in smoke_manifest.read_text().splitlines() if line.strip()]
        require(len([r for r in rows if r['split'] == 'train']) == 2 and len([r for r in rows if r['split'] == 'valid']) >= 2 and not any(r['split'] == 'test' for r in rows), 'Smoke manifest needs exactly two real train clips, >=2 valid clips, and zero test clips')
        require(all(Path(r['path']).is_file() and sha(r['path']) == r['sha256'] for r in rows), 'Smoke source bytes differ')
        source, assets = source_setup(), verify_assets()
        binding = {'role': 'real_model_smoke_only', 'reference': source, 'assets': assets,
                   'code': {name: sha(HERE / name) for name in ('train_study.py', 'study_features.py', 'study_stats.py', 'protocol.json', 'prepare_data.py')},
                   'preprocessing': read(HERE / 'protocol.json')['features'], 'smoke_manifest_sha256': sha(smoke_manifest)}
        binding['sha256'] = digest(binding)
        feature_root = ROOT / 'smoke_cache'
        feature_root.mkdir(parents=True, exist_ok=True)
        # AST helper resolves model assets within its own root. This is a controlled local
        # directory link to the same frozen inventory, never a replacement source/model.
        asset_link = feature_root / 'assets'
        if not asset_link.exists():
            asset_link.symlink_to(ROOT / 'assets', target_is_directory=True)
        extract_features(feature_root, rows, binding, device)
        fixture = {'rows': rows, 'binding': binding, 'features': verify_features(feature_root, rows, binding)}
        freeze(ROOT / 'SMOKE_INPUT_BINDING.json', binding)
    else:
        binding = read(ROOT / 'INPUT_BINDING.json')
    initialize_seed(seed, binding, smoke_initial=fixture is not None)
    cls = training_class(fixture); Context = context_class()

    def make(name):
        cfg = configure('smoke', 'baseline', seed)
        cfg['input_binding'] = binding['sha256']
        cfg['job_id'] = 'smoke_' + name; cfg['binding'] = digest({k: v for k, v in cfg.items() if k != 'binding'})
        ctx = Context(ROOT / 'smoke' / name, cfg); ctx.device = device
        seed_everything(seed, deterministic=True, cudnn_benchmark=False)
        torch.use_deterministic_algorithms(True, warn_only=False)
        job = cls(ctx); job.setup()
        return job

    with lock(ROOT / 'smoke/.lock'):
        clean = make('uninterrupted'); initial_hash = clean.initial_hash
        clean.train_steps(); expected = tensor_hash(clean.model); probe = clean.replay_probe()
        del clean; gc.collect(); torch.cuda.empty_cache()
        partial = make('resumed'); require(partial.initial_hash == initial_hash, 'Smoke matched-initial assertion failed')
        partial.train_steps(stop_after=1)
        del partial; gc.collect(); torch.cuda.empty_cache()
        resumed = make('resumed'); require(resumed.restore() and resumed.step == 1, 'Smoke step-one checkpoint restore failed')
        resumed.train_steps(); actual = tensor_hash(resumed.model); replay = resumed.replay_probe()
        require(actual == expected and torch.equal(probe['logits'], replay['logits']) and probe['token_nll'] == replay['token_nll'], 'Smoke resumed training differs from clean run')
        # Exercise the actual frozen generation API before any long scientific
        # job. Only two development/validation rows are used, with no score or
        # output-quality threshold and no test examples or test outcomes.
        from caenl.audio.common import clip_batch
        import numpy as np
        generation_ids = resumed.report_val[:2]
        require(len(generation_ids) == 2 and all(resumed.rows[int(i)]['split'] == 'valid' for i in generation_ids), 'Generation smoke must use two validation rows')
        previous = resumed.model.training; resumed.model.eval()
        try:
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=resumed.amp):
                x, mask = clip_batch(resumed.features, resumed.n_tokens, np.asarray(generation_ids), device)
                generated = resumed.model.generate(x, mask, resumed.tok.bos_token_id, resumed.tok.eos_token_id,
                    resumed.tok.pad_token_id, max_new_tokens=30, num_beams=3)
            require(len(generated) == 2 and all(isinstance(tokens, list) and all(isinstance(token, int) and 0 <= token < resumed.tok.vocab_size for token in tokens) for tokens in generated), 'Invalid real-model generation output')
            smoke_captions = [resumed.tok.decode(tokens) for tokens in generated]
            require(all(isinstance(caption, str) for caption in smoke_captions), 'Generation tokenizer returned invalid output')
        finally:
            resumed.model.train(previous)
        put(ROOT / 'SMOKE_GENERATION.json', {'status': 'PASS', 'role': 'validation-only generation API fixture',
            'input_binding': binding['sha256'], 'ids': [resumed.rows[int(i)]['id'] for i in generation_ids],
            'candidates': smoke_captions, 'beam_size': 3, 'max_new_tokens': 30,
            'favorable_quality_required': False, 'scientific_runs_counted': 0, 'official_test_outcomes_computed': False})
        # Fixed synthetic text isolates Java/tokenizer/metric compatibility from
        # model quality. All nine required metrics must run using frozen local
        # assets; this fixture never selects a coefficient or evaluates a split.
        metric_candidates = ['A dog is barking outside.', 'A person is playing a piano.', 'Rain is falling on a roof.']
        metric_references = [[sentence] * 5 for sentence in metric_candidates]
        metric_fixture = full_metrics(metric_candidates, metric_references)
        put(ROOT / 'SMOKE_METRICS.json', {'status': 'PASS', 'role': 'fixed synthetic-text runtime fixture',
            'input_binding': binding['sha256'], 'candidates': metric_candidates, 'references': metric_references,
            'metrics': metric_fixture, 'required_metrics': list(METRICS),
            'scientific_runs_counted': 0, 'dataset_outcomes_computed': False, 'official_test_outcomes_computed': False})
        # Spectral objective has an actual finite nonzero gradient, independent of language-model CE.
        z = torch.randn(32, 64, device=device, requires_grad=True)
        from caenl.core.dcr import DCR
        regularizer = DCR.build({'probe': 64}, k=64, configured_rank=16, covariance='total', device=device)
        loss, _ = regularizer({'probe': z}, {'probe': .01})
        gradient = torch.autograd.grad(loss, z)[0]
        require(torch.isfinite(gradient).all() and float(gradient.norm()) > 0, 'Spectral gradient fixture failed')
        del resumed; gc.collect(); torch.cuda.empty_cache()
        # Full MACC state/terminal-credit exercise uses genuine policy updates, no audio outcome.
        loop = terminal_loop_class()({'probe': 64}, 'full_macc', .01, configure('smoke', 'full_macc', seed), device, 19, 100, k=64, configured_rank=16, cadence=2)
        for index in range(100):
            loop.update({'probe': z.detach()}, value_fn=lambda j=index: -2. + .002 * j, extra_state=(index / 100,))
            if index == 19:
                state = loop.state_dict()
                replacement = terminal_loop_class()({'probe': 64}, 'full_macc', .01, configure('smoke', 'full_macc', seed), device, 19, 100, k=64, configured_rank=16, cadence=2)
                replacement.load_state_dict(state); loop = replacement
        learned = sum('policy_loss' in row.get('update', {}) for row in loop.summary()['history'])
        require(learned >= 20 and loop.controller._pending is None and loop.summary()['history'][-1]['new_action_drawn'] is False, 'Full-MACC learned-update/terminal-credit fixture failed')
        put(ROOT / 'SMOKE.json', {'status': 'PASS', 'input_binding': binding['sha256'], 'real_pretrained_caption_model': True,
            'real_training_examples': 2, 'steps_in_clean_and_resumed_paths': 2, 'interruption_after_step': 1,
            'matched_initial_tensor_hash': initial_hash, 'final_tensor_hash_clean': expected, 'final_tensor_hash_resumed': actual,
            'full_probe_logits_bitwise_equal': True, 'query_encoder_decoder_gradients_finite_nonzero': True,
            'spectral_gradient_finite_nonzero': True, 'full_controller_fixture_learned_updates': learned,
            'real_validation_generation_api_passed': True, 'real_validation_generation_examples': 2,
            'fixed_synthetic_text_all_metrics_passed': True,
            'generation_fixture_sha256': sha(ROOT / 'SMOKE_GENERATION.json'),
            'metric_fixture_sha256': sha(ROOT / 'SMOKE_METRICS.json'),
            'scientific_runs_counted': 0, 'official_test_outcomes_computed': False,
            'smoke_fixture_manifest_sha256': sha(smoke_manifest) if smoke_manifest is not None else None,
            'source_assets_code_binding': digest({'reference': binding['reference'], 'assets': binding['assets'], 'code': binding['code']})})


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/srv/iga/caenl-audiocaps-raw-v1'))
    parser.add_argument('--phase', required=True, choices=('features', 'initialize', 'worker', 'analyze', 'smoke'))
    parser.add_argument('--stage', choices=('tuning', 'confirm'), default='tuning')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--smoke-manifest', type=Path, help='Optional real train/valid fixture manifest; no full data readiness or test outcomes')
    parser.add_argument('--worker-index', type=int, default=0)
    parser.add_argument('--workers', type=int, default=execution_workers())
    parser.add_argument('--run-job', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--audit-only', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--method', choices=METHODS + ('fixed_dcr',), help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--lambda-value', type=float, help=argparse.SUPPRESS)
    args = parser.parse_args(); ROOT = args.root.resolve(); ROOT.mkdir(parents=True, exist_ok=True)
    source_setup()
    if args.phase == 'smoke' and args.smoke_manifest is not None:
        smoke(args.gpu, args.smoke_manifest); return
    if args.run_job:
        require(args.phase == 'worker' and args.method is not None and args.seed is not None and args.lambda_value in LAMBDAS, 'Invalid internal job invocation')
        allowed = jobs(args.stage)
        require((args.method, args.seed, args.lambda_value) in allowed, 'Internal job is not in the frozen finite study')
        input_binding()  # Fail on changed source/data/assets even for interrupted jobs.
        run_job(args.stage, args.method, args.seed, args.lambda_value, args.gpu, args.audit_only)
        return
    if args.phase == 'analyze':
        rows, binding = input_binding()
        if args.stage == 'tuning':
            select_fixed()
        else:
            from study_stats import analyze
            analyze(ROOT, expected_input_binding=binding['sha256'], expected_steps=(binding['data']['counts']['train'] // 32) * 10, expected_test_clips=binding['data']['counts']['test'])
        return
    device = runtime(args.gpu, cpu=args.phase == 'initialize')
    rows, binding = input_binding()
    if args.phase == 'features':
        from study_features import extract_features
        extract_features(ROOT, rows, binding, device)
    elif args.phase == 'initialize':
        for seed in TUNE_SEEDS + CONFIRM_SEEDS:
            initialize_seed(seed, binding)
        put(ROOT / 'INITIALS_READY.json', {'status': 'READY', 'seeds': list(TUNE_SEEDS + CONFIRM_SEEDS), 'input_binding': binding['sha256'],
            'files': {('s' + str(seed) + '/initial.pt'): sha(ROOT / 'initial' / ('s' + str(seed)) / 'initial.pt') for seed in TUNE_SEEDS + CONFIRM_SEEDS}})
    elif args.phase == 'smoke':
        smoke(args.gpu)
    else:
        smoke_record = read(ROOT / 'SMOKE.json')
        expected_smoke_binding = digest({'reference': binding['reference'], 'assets': binding['assets'], 'code': binding['code']})
        require(smoke_record['status'] == 'PASS' and smoke_record['source_assets_code_binding'] == expected_smoke_binding, 'Run successful smoke with identical code/source/assets before workers')
        require(read(ROOT / 'INITIALS_READY.json')['input_binding'] == binding['sha256'], 'Initialize all 13 seeds before workers')
        worker(args)


if __name__ == '__main__':
    main()
