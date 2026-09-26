#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CAENL AudioCaps-only development training v1, 22 September 2026.


Use the completed, split-disjoint OpenSound/AudioCaps available subset.
Two development seeds, four methods, two epochs per method. No official-test
caption predictions and no final six-seed confirmation are launched here.
Reuses a pinned COPY of the earlier cross-modal engine and CAENL source.
The new wrapper has NOT been executed in the authoring runtime: Python/container
calls timed out. Offline/GPU/model/metric checks run on IBM before task training.
No pip/conda/system-package installations, dataset downloads, old job restarts,
source-data changes, or deletion of existing experiment artifacts.
Missing public pretrained-model assets may be downloaded to this new workspace.
Required caption-metric assets are checked in an isolated package copy; the
reference SPICE implementation may download its standard Stanford model assets.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
from contextlib import ExitStack
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback


sys.dont_write_bytecode = True
SESSION = 'caenl-audiocaps-train-pilot-v1'
ROOT = Path('/mnt/caenl/active/results') / SESSION
PILOT = Path('/mnt/caenl/active/results/caenl-crossmodal-pilots-v1')
DATA = Path('/mnt/caenl/persistent/datasets/audiocaps_opensound_v1/disjoint_v1')
DATA_BASE = DATA.parent
DATA_PARENT = DATA_BASE / 'quarantine_v1'
PROJECT = Path.home() / 'caenl-v5.4.3/caenl_revision_pipeline_v5.4.3_c4_confirmatory'
PYTHON = PROJECT / '.venv/bin/python'
REPORT = SESSION + '.md'
ENGINE_SHA = 'd3d01297c43357b9c34d81701b21f1b80c3a3a395bb65ef341026c61c918ac15'
SOURCE_SHA = 'e8a7a3039390e60491431376ec1d28aad07e674fbaf64ca89aaa18e52f6b9451'
PARENT_REPORT_SHA = '06f192148990312dcd31e77a8535f3a8cb83e47b34a96395d178d1de98c8ac45'
PARENT_MANIFEST_SHA = 'b36f2874140e6744b85207dfda3597fdb5d70634a0799a6438cdd256407aa19f'
SEEDS = (1201, 1202)
METHODS = ('baseline', 'fixed_dcr', 'macc_lite', 'full_macc')
COUNTS = {'train': 44466, 'valid': 440, 'test': 866}
AST_NAME = 'MIT/ast-finetuned-audioset-10-10-0.4593'
PROTOCOL = {
 'evidence_role': 'development_only_not_final_multiseed_confirmation',
 'dataset': 'original AudioCaps available subset via OpenSound mirror',
 'mirror_revision': 'b29b3243d6ce49c2cd0d48d4b5f0701ae7969ded',
 'official_caption_revision': 'd004db3ea1b01cf4fd0347dd8d27db90cadc8809',
 'retained_counts': COUNTS, 'earlier_exclusions_remain_excluded': True,
 'seeds': list(SEEDS), 'methods': list(METHODS), 'epochs': 2,
 'batch_size': 32, 'steps_per_epoch': 1389, 'steps_per_job': 2778,
 'epoch_definition': 'deterministic permutation of 44466 training pairs; last 18 pairs dropped per epoch, with a new permutation each epoch',
 'backbone': AST_NAME, 'backbone_frozen': True, 'decoder': 'gpt2',
 'model': 'retained 2-layer audio-token encoder, 8-token learned prefix and trainable GPT-2',
 'feature_extraction': {'target_sr': 16000, 'channel_mix': 'arithmetic mean as in retained loader',
   'max_windows': 1, 'window_seconds': 10.24, 'tokens_per_window': 64,
   'AST_forward_batch_size': 4, 'cache_dtype': 'float16', 'model_forward_dtype': 'float32',
   'clips_longer_than_window': 'first 10.24 seconds only; original bytes unmodified',
   'zero_waveforms': 'retain and report according to frozen dataset policy',
   'official_test_feature_extraction': 'frozen pretrained backbone only; no caption-model or task evaluation'},
 'optimizer': {'name': 'AdamW', 'encoder_prefix_lr': 0.0001, 'decoder_lr': 0.00005,
   'weight_decay': 0.01, 'warmup_steps': 100, 'schedule': 'retained cosine',
   'parameter_dtype': 'float32', 'training_autocast': 'bfloat16', 'grad_clip': 1.0},
 'spectral_control': {'not_supervised_aligned_classification_DCR': True, 'layers': ['prefix', 'audio_enc'],
   'retained_spectrum_size': 64, 'configured_target_rank': 16, 'monitor_ema_decay': 0.99,
   'cadence': 50, 'fixed_lambda': 0.01, 'initial_adaptive_lambda': 0.01,
   'lambda_bounds': [0.0, 0.05], 'lite_eta': 0.001, 'lite_deadband': 0.1,
   'full_terminal_credit': 'pending last action rewarded without drawing an unused next action',
   'acquisition_head': False, 'minimum_full_macc_learned_updates_for_technical_validation': 20},
 'validation': {'controller': 'first 64 official-validation IDs in the frozen manifest order',
   'development': 'remaining 376 official-validation IDs, all 5 reference captions',
   'caption_metrics': ['CIDEr-D', 'BLEU-1..4', 'ROUGE-L', 'METEOR', 'SPICE', 'SPIDEr'],
   'implementation': 'retained pycocoevalcap/PTB path; required metric failures stop, without fallback',
   'max_new_tokens': 30, 'beam_size': 3, 'decode_batch_size': 32,
   'nll': 'pooled next-token cross-entropy weighted by actual target-token count',
   'conditioning_diagnostic': 'deterministic mismatched-audio development NLL'},
 'official_test_caption_outcomes': False,
 'initialization': 'same pretrained GPT-2 and per-seed randomly initialized prefix/encoder across methods',
 'training_randomness': 'reset CPU/CUDA dropout stream from seed and optimizer step before each training batch',
 'checkpoint_selection': 'final scheduled pilot step; never best development/test performance',
 'completion_criterion': 'all scheduled steps and finite required outputs, not a minimum gain',
 'future_confirmation': 'independent six-seed protocol to be frozen after development report review; not launched here',
 'pretraining_overlap': 'AST AudioSet pretraining may include underlying AudioCaps recordings; not an untouched-pretraining claim',
 'data_rights': 'third-party mirror declares CC BY-NC 4.0; uploader rights not independently established; no redistribution',
 'authoring_runtime_execution': False,
}
CURRENT = {}
STATE_LOCK = threading.Lock()
E = None




def now(): return datetime.now(timezone.utc).isoformat()
def require(ok, msg):
    if not bool(ok): raise RuntimeError(msg)
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''): h.update(b)
    return h.hexdigest()
def encode(x): return (json.dumps(x, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
def digest(x): return hashlib.sha256(encode(x)).hexdigest()
def write(p, b):
    p = Path(p)
    require(p.resolve().is_relative_to(ROOT.resolve()) and not p.is_symlink(), 'Unsafe output: ' + str(p))
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(b, str): b = b.encode('utf-8')
    with tempfile.NamedTemporaryFile(dir=p.parent, prefix='.writing_', delete=False) as f:
        tmp = Path(f.name); f.write(b); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)
def put(p, x): write(p, encode(x))
def freeze(p, x):
    if Path(p).exists(): require(read(p) == x, 'Frozen input/protocol changed: ' + str(p))
    else: put(p, x)
def progress(message, **details):
    with STATE_LOCK:
        CURRENT.clear(); CURRENT.update(status='RUNNING', pid=os.getpid(), time=now(), message=message, **details)
        put(ROOT / 'STATUS.json', CURRENT)
    print(message, flush=True)
def beat(stop):
    while not stop.wait(30):
        with STATE_LOCK:
            if CURRENT: put(ROOT / 'STATUS.json', dict(CURRENT, heartbeat_utc=now()))
def copy_bound(source, target, expected):
    source, target = Path(source), Path(target)
    require(source.is_file() and not source.is_symlink() and sha(source) == expected, 'Retained input differs: ' + str(source))
    if target.exists(): require(sha(target) == expected, 'Copied input differs: ' + str(target))
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + '.copying')
        shutil.copyfile(source, tmp); require(sha(tmp) == expected, 'Copy digest mismatch'); os.replace(tmp, target)




def prepare_sources():
    copy_bound(PILOT / 'runner.py', ROOT / 'engine.py', ENGINE_SHA)
    copy_bound(PILOT / 'SOURCE.json', ROOT / 'SOURCE.json', SOURCE_SHA)
    for rel, expected in read(ROOT / 'SOURCE.json')['files'].items():
        require(not Path(rel).is_absolute() and '..' not in Path(rel).parts, 'Unsafe source entry')
        copy_bound(PILOT / 'source/caenl' / rel, ROOT / 'source/caenl' / rel, expected)
    freeze(ROOT / 'WRAPPER_PLAN.json', {'protocol': PROTOCOL, 'wrapper_sha256': sha(ROOT / 'runner.py'),
       'engine_sha256': ENGINE_SHA, 'source_manifest_sha256': SOURCE_SHA})




def get_engine():
    global E
    if E is None:
        require(sha(ROOT / 'engine.py') == ENGINE_SHA, 'Copied engine hash mismatch')
        spec = importlib.util.spec_from_file_location('caenl_audio_pilot_engine', ROOT / 'engine.py')
        E = importlib.util.module_from_spec(spec); sys.modules[spec.name] = E; spec.loader.exec_module(E)
        E.ROOT = ROOT; E.SEEDS = SEEDS; E.REPORT = REPORT
        E.PLAN = {'role': PROTOCOL['evidence_role'], 'audio': copy.deepcopy(E.PLAN['audio']), 'audio_only_protocol': PROTOCOL}
        E.progress = progress
        E.source_setup()
    return E




def verify_data():
    require(DATA.is_dir() and not DATA.is_symlink(), 'Finalized dataset missing')
    complete = read(DATA / 'COMPLETE.json')
    require(complete['status'] == 'DATA_READY' and complete['training_started'] is False, 'Dataset is not ready')
    for name, expected in complete['outputs'].items():
        require(Path(name).name == name and sha(DATA / name) == expected, 'Finalization output changed: ' + name)
    ready = read(DATA / 'AUDIOCAPS_READY.json')
    require(ready == read(DATA / 'SUMMARY.json'), 'Ready and summary records differ')
    require(ready['blocking_issues'] == [] and ready['retained_unique_clips'] == {'train': 44466, 'val': 440, 'test': 866}, 'Unexpected readiness/coverage')
    require(ready['parent_report_sha256'] == PARENT_REPORT_SHA and ready['parent_manifest_sha256'] == PARENT_MANIFEST_SHA, 'Different audited parent dataset')
    require(ready['cross_split_nonzero_pcm_groups_after'] == 0 and ready['cross_split_video_count'] == 0, 'Unresolved split leakage')
    require(ready['minimum_required_fraction'] == .8 and ready['duration_rule_seconds'] == [8, 12], 'Data rules changed')
    require(ready['previous_waveform_conflicts_remain_excluded'] and ready['previous_duration_exclusions_remain_excluded'], 'Quarantine rules changed')
    require(ready['revision'] == PROTOCOL['mirror_revision'] and ready['official_captions_revision'] == PROTOCOL['official_caption_revision'], 'Source revision changed')
    require(sha(DATA_PARENT / 'caenl-audiocaps-quarantine-v1.md') == PARENT_REPORT_SHA, 'Parent report changed')
    require(sha(DATA_PARENT / 'manifest.jsonl') == PARENT_MANIFEST_SHA, 'Parent manifest changed')
    require(sha(DATA / 'manifest.jsonl') == ready['manifest_sha256'], 'Final manifest changed')
    rows = [json.loads(s) for s in (DATA / 'manifest.jsonl').read_text().splitlines() if s.strip()]
    require(dict(Counter(r['split'] for r in rows)) == COUNTS and len({r['id'] for r in rows}) == len(rows), 'Actual split/ID counts differ')
    require(not {'test/473wBEwC35M_30', 'test/PLHXGDnig4M_3', 'train/xXZ_IpimXvs_3'}.intersection(r['id'] for r in rows), 'Excluded identity reintroduced')
    videos, pcm = defaultdict(set), defaultdict(set)
    for r in rows:
        p = Path(r['path'])
        require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(DATA_BASE.resolve()), 'Unsafe/missing recording')
        require(p.stat().st_size == r['bytes'] and 8 <= r['duration_seconds'] <= 12, 'Changed file size/duration')
        require(not r['waveform_conflict'], 'Conflicting waveform retained')
        require(len(r['captions']) == (1 if r['split'] == 'train' else 5), 'Unexpected reference-caption count')
        videos[r['youtube_id']].add(r['split'])
        if not r['all_zero_waveform']: pcm[r['pcm_float32_sha256']].add(r['split'])
    require(all(len(v) == 1 for v in videos.values()) and all(len(v) == 1 for v in pcm.values()), 'Cross-split identity/PCM duplicate')
    binding = {'complete_record_sha256': sha(DATA / 'COMPLETE.json'), 'manifest_sha256': ready['manifest_sha256'],
               'summary_sha256': sha(DATA / 'SUMMARY.json'), 'counts': COUNTS, 'directory': str(DATA)}
    freeze(ROOT / 'DATA_BINDING.json', binding)
    freeze(ROOT / 'AUDIO_READY.json', ready)
    return rows




def metric_vendor_setup():
    spec = importlib.util.find_spec('pycocoevalcap')
    require(spec is not None and spec.submodule_search_locations, 'Existing pycocoevalcap is missing; nothing installed')
    original = Path(next(iter(spec.submodule_search_locations)))
    target = ROOT / 'metric_vendor/pycocoevalcap'
    source_files = {}
    for p in sorted(original.rglob('*')):
        rel = p.relative_to(original)
        if any(part in ('cache', 'tmp', '__pycache__', '.git') for part in rel.parts): continue
        if p.is_file():
            require(not p.is_symlink(), 'Symlink inside installed metric package; review required')
            h = sha(p); source_files[rel.as_posix()] = h
            copy_bound(p, target / rel, h)
    if not (target / '__init__.py').exists(): write(target / '__init__.py', '# Isolated reference metric package copy.\n')
    freeze(ROOT / 'METRIC_VENDOR_SOURCE.json', {'original_directory': str(original), 'source_files': source_files,
        'installed_package_modified': False, 'cache_and_tmp_not_copied': True})




def activate_metric_vendor():
    require((ROOT / 'metric_vendor/pycocoevalcap').is_dir(), 'Isolated metric package missing')
    sys.path.insert(0, str(ROOT / 'metric_vendor'))




def record_metric_assets():
    d = ROOT / 'metric_vendor'
    files = {p.relative_to(d).as_posix(): sha(p) for p in sorted(d.rglob('*'))
             if p.is_file() and p.suffix in ('.py', '.jar') and not any(x in ('cache', 'tmp', '__pycache__') for x in p.relative_to(d).parts)}
    require(any(k.endswith('.jar') for k in files), 'Reference Java metric assets missing')
    freeze(ROOT / 'METRIC_ASSETS.json', {'files': files, 'java': subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT, text=True).strip()})




def metric_check():
    e = get_engine(); activate_metric_vendor()
    require(shutil.which('java'), 'Java metric runtime missing; no packages installed')
    from caenl.audio.metrics import caption_metrics
    scores = caption_metrics(['a dog barks outside', 'a person plays a guitar'],
       [['a dog is barking outside', 'a dog barks loudly outdoors'],
        ['someone is playing a guitar', 'a person plays music on a guitar']], java=True, spice=True, official=True)
    require(scores['metrics_source'] == 'pycocoevalcap' and scores['tokenizer'] == 'ptb', 'Unexpected metric fallback')
    require(all(k in scores and math.isfinite(float(scores[k])) for k in ('cider_d', 'bleu4', 'rouge_l', 'meteor', 'spice', 'spider')), 'Missing/nonfinite caption metric')
    record_metric_assets()
    put(ROOT / 'METRIC_TEST.json', {'status': 'PASS', 'toy_scores_not_scientific_results': scores})
    print('REFERENCE CAPTION METRICS: PASS', flush=True)




def prepare_models():
    e = get_engine()
    for repo, short in ((AST_NAME, 'ast'), ('gpt2', 'gpt2')):
        e.resolve_model(repo)
        models = read(ROOT / 'MODELS.json'); rec = models[repo]
        src = Path(rec['path']); target = ROOT / 'assets' / short
        for rel, h in rec['files'].items():
            require(not Path(rel).is_absolute() and '..' not in Path(rel).parts, 'Unsafe model filename')
            p = src / rel
            require(p.is_file() and sha(p) == h, 'Pretrained asset changed')
            if (target / rel).exists(): require(sha(target / rel) == h, 'Private model copy differs')
            else:
                dest = target / rel; dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name + '.copying'); shutil.copyfile(p, tmp)
                require(sha(tmp) == h, 'Model copy checksum mismatch'); os.replace(tmp, dest)
        rec['original_snapshot_path'] = rec.get('original_snapshot_path', str(src))
        rec['path'] = str(target); models[repo] = rec; put(ROOT / 'MODELS.json', models)
    freeze(ROOT / 'MODELS_FROZEN.json', read(ROOT / 'MODELS.json'))




def gpu_check():
    e = get_engine(); e.checks(); device = e.runtime()
    import numpy as np
    import torch
    from caenl.audio.data import ASTBackbone, load_audio_16k
    from caenl.audio.captioning import PrefixCaptioner
    from transformers import AutoModelForCausalLM
    rows = [json.loads(s) for s in (DATA / 'manifest.jsonl').read_text().splitlines()]
    r = next(r for r in rows if r['split'] == 'train' and not r['all_zero_waveform'])
    require(sha(r['path']) == r['sha256'], 'Preflight waveform changed')
    models = read(ROOT / 'MODELS.json')
    bb = ASTBackbone(models[AST_NAME]['path'], device=device)
    require(bb.dim == 768 and bb.window == 163840, 'AST dimensions differ from protocol')
    wav = torch.from_numpy(load_audio_16k(r['path'])[:bb.window]).to(device)
    with torch.no_grad(): z = bb(wav)
    require(z.ndim == 2 and z.shape[1] == 768 and torch.isfinite(z).all(), 'Real-audio AST feature check failed')
    del bb, wav, z; torch.cuda.empty_cache()
    torch.manual_seed(782012); torch.cuda.manual_seed_all(782012)
    decoder = AutoModelForCausalLM.from_pretrained(models['gpt2']['path'], local_files_only=True).float()
    cap = PrefixCaptioner(768, decoder, 768, prefix_len=8, enc_layers=2, n_heads=8).to(device)
    x = torch.randn(32, 64, 768, device=device)
    mask = torch.ones(32, 64, dtype=torch.bool, device=device)
    ids = torch.randint(0, 1000, (32, 48), device=device)
    am = torch.ones_like(ids)
    from caenl.core.dcr import DCR
    reg = DCR.build({'prefix': 768, 'audio_enc': 768}, k=64, configured_rank=16, covariance='total', device=device)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16): ce, feats = cap(x, mask, ids, am)
    regularization, _ = reg(feats, {'prefix': .01, 'audio_enc': .01})
    loss = ce + regularization; require(torch.isfinite(loss), 'Full-size caption training loss invalid')
    loss.backward(); norm = torch.nn.utils.clip_grad_norm_(cap.parameters(), 1.0, error_if_nonfinite=True)
    require(float(norm) > 0, 'No gradient in full-size caption fixture')
    cap.zero_grad(set_to_none=True); cap.eval()
    with torch.no_grad():
        prefix, _ = cap.prefix(x[:1], mask[:1])
        bos = torch.full((1, 1), 50256, device=device, dtype=torch.long)
        emb = torch.cat([prefix, cap.decoder.get_input_embeddings()(bos)], dim=1)
        pred = cap.decoder(inputs_embeds=emb, attention_mask=torch.ones(emb.shape[:2], device=device, dtype=torch.long)).logits[:, -1].argmax(-1)
        generated = cap.generate(x[:1], mask[:1], 50256, 50256, 50256, max_new_tokens=1, num_beams=1)
        require(len(generated) == 1 and len(generated[0]) == 1 and generated[0][0] == int(pred[0]), 'Generation API does not return the expected continuation-only tokens')
    put(ROOT / 'REAL_MODEL_TEST.json', {'status': 'PASS', 'real_AST_clip_id': r['id'],
       'full_caption_batch_size': 32, 'caption_fixture': 'synthetic token features; no scientific optimizer updates',
       'DCR_backward_finite': True, 'generation_matches_one_step_logits': True, 'scientific_runs_counted': 0})
    print('FULL-SIZE AUDIO MODEL GPU CHECK: PASS', flush=True)




def feature_binding():
    return digest({'manifest': read(ROOT / 'DATA_BINDING.json'), 'models': read(ROOT / 'MODELS_FROZEN.json'),
       'preprocessing': PROTOCOL['feature_extraction'], 'wrapper': sha(ROOT / 'runner.py'), 'engine': ENGINE_SHA})
def array_hash(a):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()




def extract_features():
    e = get_engine(); device = e.runtime()
    import numpy as np
    import torch
    from caenl.audio.data import ASTBackbone, load_audio_16k
    rows = [json.loads(s) for s in (DATA / 'manifest.jsonl').read_text().splitlines()]
    folder = ROOT / 'features'; folder.mkdir(exist_ok=True)
    binding = feature_binding(); shape = (len(rows), 64, 768)
    freeze(folder / 'CONFIG.json', {'binding': binding, 'shape': list(shape), 'dtype': 'float16'})
    if (ROOT / 'FEATURES_READY.json').exists(): verify_features(); print('USING COMPLETE FROZEN FEATURES', flush=True); return
    path = folder / 'features.npy'
    if path.exists():
        feats = np.load(path, mmap_mode='r+', allow_pickle=False)
        require(feats.shape == shape and feats.dtype == np.float16, 'Feature cache shape/type mismatch')
    else: feats = np.lib.format.open_memmap(path, mode='w+', dtype=np.float16, shape=shape)
    bb = None; records = []; t0 = time.monotonic(); chunk = 256
    for begin in range(0, len(rows), chunk):
        end = min(len(rows), begin + chunk); marker = folder / ('chunk_%06d.json' % begin)
        if marker.exists():
            rec = read(marker)
            require(rec['binding'] == binding and rec['start'] == begin and rec['end'] == end, 'Feature chunk binding mismatch')
            require(array_hash(feats[begin:end]) == rec['features_sha256'], 'Saved feature chunk changed')
            require([x['id'] for x in rec['inputs']] == [r['id'] for r in rows[begin:end]], 'Feature row ordering changed')
            records.extend(rec['inputs']); continue
        if bb is None:
            bb = ASTBackbone(read(ROOT / 'MODELS.json')[AST_NAME]['path'], device=device)
            bb.model.requires_grad_(False)
            require(bb.dim == 768 and bb.window == 163840, 'Unexpected AST shape')
        inputs = []
        for start in range(begin, end, 4):
            batch = rows[start:min(end, start + 4)]; waves = []
            for r in batch:
                require(sha(r['path']) == r['sha256'], 'Recording changed since finalization: ' + r['id'])
                wave = np.ascontiguousarray(load_audio_16k(r['path'])[:bb.window], dtype='<f4')
                require(wave.size >= 8 * 16000 and np.isfinite(wave).all(), 'Preprocessed audio invalid')
                h = hashlib.sha256(('16000:' + str(wave.shape) + ':float32le').encode() + wave.tobytes()).hexdigest()
                inputs.append({'id': r['id'], 'split': r['split'], 'file_sha256': r['sha256'],
                    'preprocessed_sha256': h, 'samples': len(wave), 'all_zero': not bool(np.any(wave))})
                waves.append(wave)
            with torch.no_grad():
                data = bb.extractor(waves, sampling_rate=16000, return_tensors='pt')
                tokens = bb.model(input_values=data['input_values'].to(device)).last_hidden_state[:, 2:]
                pooled = torch.nn.functional.adaptive_avg_pool1d(tokens.transpose(1, 2), 64).transpose(1, 2)
                values = pooled.float().cpu().numpy().astype(np.float16)
            require(values.shape == (len(batch), 64, 768) and np.isfinite(values).all(), 'Frozen feature output invalid')
            feats[start:start + len(batch)] = values
        feats.flush()
        with path.open('rb') as f: os.fsync(f.fileno())
        put(marker, {'binding': binding, 'start': begin, 'end': end, 'features_sha256': array_hash(feats[begin:end]), 'inputs': inputs})
        records.extend(inputs)
        progress('FROZEN AUDIO FEATURES: ' + str(end) + '/' + str(len(rows)), elapsed_seconds=round(time.monotonic()-t0, 1))
    groups = defaultdict(list)
    for r in records:
        if not r['all_zero']: groups[r['preprocessed_sha256']].append(r)
    crossing = {h: g for h, g in groups.items() if len({r['split'] for r in g}) > 1}
    put(ROOT / 'PREPROCESSED_INPUT_AUDIT.json', {'nonzero_exact_cross_split_groups': crossing,
        'recordings': len(records), 'all_zero_ids': [r['id'] for r in records if r['all_zero']],
        'test_features_are_fixed_backbone_only': True, 'near_duplicates_not_certified': True})
    require(not crossing, 'Nonzero audio inputs become exact cross-split duplicates after mono/resampling; no task training was started')
    npath = folder / 'n_tokens.npy'
    with npath.open('wb') as f: np.save(f, np.full(len(rows), 64, dtype=np.int64), allow_pickle=False)
    put(folder / 'meta.json', {'backbone': read(ROOT / 'MODELS.json')[AST_NAME]['path'], 'dim': 768,
        'tokens_per_window': 64, 'max_windows': 1, 'n_clips': len(rows), 'cache_dir': str(folder)})
    put(ROOT / 'FEATURES_READY.json', {'status': 'READY', 'binding': binding, 'n_clips': len(rows),
        'files': {p.name: sha(p) for p in (path, npath, folder / 'meta.json')},
        'preprocessing_audit_sha256': sha(ROOT / 'PREPROCESSED_INPUT_AUDIT.json'),
        'feature_extraction_wall_seconds_this_attempt': time.monotonic() - t0,
        'timing_includes_resume_verification_and_final_hashing': True})
    print('FROZEN AUDIO FEATURES: COMPLETE', flush=True)




def verify_features():
    r = read(ROOT / 'FEATURES_READY.json'); require(r['binding'] == feature_binding(), 'Frozen feature protocol differs')
    for name, h in r['files'].items():
        require(Path(name).name == name and sha(ROOT / 'features' / name) == h, 'Frozen feature payload changed: ' + name)
    require(sha(ROOT / 'PREPROCESSED_INPUT_AUDIT.json') == r['preprocessing_audit_sha256'], 'Preprocessing audit changed')




def install_audio_overrides(e):
    import numpy as np
    import torch
    import caenl.audio.captioning as captioning
    from caenl.utils.seeding import derive_seed
    verify_features(); activate_metric_vendor()
    for name, h in read(ROOT / 'METRIC_ASSETS.json')['files'].items():
        require(sha(ROOT / 'metric_vendor' / name) == h, 'Reference metric implementation changed')
    def frozen_cache(manifest, cache_root, backbone, tokens_per_window=64, max_windows=1, device=None, hf_cache=None):
        require(Path(manifest).resolve() == (DATA / 'manifest.jsonl').resolve() and tokens_per_window == 64 and max_windows == 1, 'Unexpected feature-cache request')
        info = read(ROOT / 'features/meta.json')
        require(info['backbone'] == backbone, 'Backbone identity mismatch')
        info['features'] = np.load(ROOT / 'features/features.npy', mmap_mode='r', allow_pickle=False)
        info['rows'] = [json.loads(s) for s in Path(manifest).read_text().splitlines()]
        return info
    captioning.build_feature_cache = frozen_cache
    # HF assets are local and byte-bound. The private feature cache is the only
    # feature source; the old dataset-download/extraction routes are not invoked.
    original_configure = e.configure
    def configure(task, method, seed):
        require(task == 'audio' and method in METHODS and seed in SEEDS, 'Only the frozen audio pilot is supported')
        cfg = original_configure(task, method, seed)
        cfg.pop('crossmodal_binding')
        cfg['audio']['manifest'] = str(DATA / 'manifest.jsonl')
        cfg['feature_binding'] = read(ROOT / 'FEATURES_READY.json')
        cfg['metric_assets_sha256'] = sha(ROOT / 'METRIC_ASSETS.json')
        cfg['audio_only_protocol_binding'] = read(ROOT / 'PROTOCOL.json')['binding']
        cfg['crossmodal_binding'] = digest(cfg)
        return cfg
    e.configure = configure
    original_install = e.install_task_classes
    def install(task):
        require(task == 'audio', 'No diffusion task can be launched')
        original_install(task)
        base = captioning.CaptioningJob
        class AudioPilot(base):
            def setup(self):
                super().setup()
                require(len(self.pairs) == 44466 and self.total_steps == 2778, 'Training budget mismatch')
                require(len(self.ctrl_val) == 64 and len(self.report_val) == 376 and len(self.official_test_ids) == 866, 'Feedback/development/test partition mismatch')
                require(set(self.splits['test']) == set(self.report_val), 'Pilot would evaluate the official test split')
                require(all(self.rows[int(i)]['split'] == 'valid' for i in self.report_val), 'Development split mismatch')
            def batch(self, pairs):
                if torch.is_grad_enabled() and self.model.training:
                    common_seed = derive_seed(self.ctx.seed, 'audio_step_dropout', self.step)
                    torch.manual_seed(common_seed); torch.cuda.manual_seed_all(common_seed)
                return super().batch(pairs)
            def restore(self):
                restored = super().restore()
                self.step_at_attempt_start = self.step
                return restored
            def run(self):
                result = super().run()
                executed = self.step - self.step_at_attempt_start
                result['compute']['steps_executed_this_attempt'] = executed
                result['compute']['steps_per_s'] = executed / max(result['compute']['train_time_s'], 1e-6)
                result['compute']['training_timer_scope'] = 'current attempt training loop, including feedback and in-loop checkpoints; excludes setup and final decoding/metrics'
                require(result['steps'] == 2778, 'Incomplete audio pilot training')
                require(result['caption_diagnostics']['n'] == 376, 'Incomplete development caption set')
                result['evidence_role'] = PROTOCOL['evidence_role']
                result['frozen_data_binding'] = read(ROOT / 'DATA_BINDING.json')
                result['audio_only_protocol'] = PROTOCOL
                result['actual_split_counts'] = COUNTS
                result['official_test_caption_outcomes_computed'] = False
                return result
        captioning.CaptioningJob = AudioPilot
        return captioning.run
    e.install_task_classes = install




def train_job(method, seed):
    verify_data(); e = get_engine(); e.runtime(); install_audio_overrides(e)
    e.execute_job('audio', method, seed)




def child(args, timeout=None, offline=False):
    env = dict(os.environ)
    if offline: env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    process = subprocess.Popen([sys.executable, str(ROOT / 'runner.py'), *map(str, args)], env=env, start_new_session=True)
    try:
        rc = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL); process.wait()
        raise RuntimeError('Stage timed out: ' + repr(args))
    if rc:
        log = ROOT / 'run.log'
        with log.open('rb') as f:
            f.seek(max(0, log.stat().st_size - 18000)); tail = f.read().decode('utf-8', errors='replace')
        put(ROOT / 'CHILD_FAILURE.json', {'arguments': list(args), 'exit_code': rc, 'log_tail': tail})
        raise RuntimeError('Stage stopped with exit ' + str(rc) + ': ' + repr(args))




def make_report(label):
    lines = ['# CAENL AudioCaps development-training pilot', '', 'Status: ' + label, 'UTC: ' + now(), '',
       'Two development seeds only. No final six-seed confirmation and no official-test caption predictions.',
       'All completed unfavorable, empty-caption or weak-conditioning outcomes remain reported.', '']
    for name in ('WRAPPER_PLAN.json', 'DATA_BINDING.json', 'AUDIO_READY.json', 'MODELS_FROZEN.json', 'ENVIRONMENT.json',
       'SELF_TEST.json', 'REAL_MODEL_TEST.json', 'METRIC_TEST.json', 'METRIC_ASSETS.json', 'PREPROCESSED_INPUT_AUDIT.json',
       'FEATURES_READY.json', 'PROTOCOL.json', 'PER_SEED.json', 'FAILED.json', 'CHILD_FAILURE.json', 'PRESERVATION.json'):
        p = ROOT / name
        if p.is_file(): lines += ['## ' + name, 'SHA-256: ' + sha(p), '```json', p.read_text().strip(), '```', '']
    for p in sorted(ROOT.glob('audio/seed*/*/summary.json')):
        lines += ['## ' + p.relative_to(ROOT).as_posix(), 'SHA-256: ' + sha(p), '```json', p.read_text().strip(), '```', '']
    for p in sorted(ROOT.glob('audio/seed*/*/artifacts/captions_development.json')):
        lines += ['## ' + p.relative_to(ROOT).as_posix(), 'SHA-256: ' + sha(p), '```json', p.read_text().strip(), '```', '']
    for name in ('runner.py', 'engine.py'):
        p = ROOT / name
        if p.is_file(): lines += ['## Exact source ' + name, 'SHA-256: ' + sha(p), '````````', p.read_text(encoding='utf-8-sig'), '````````', '']
    lines += ['END OF AUDIOCAPS TRAINING PILOT REPORT']
    write(ROOT / REPORT, '\n'.join(lines) + '\n')
    write(ROOT / (REPORT + '.sha256'), sha(ROOT / REPORT) + '  ' + REPORT + '\n')




def worker():
    import fcntl
    with ExitStack() as stack:
        lock = stack.enter_context((ROOT / 'RUN.lock').open('a')); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for directory, names in ((DATA_BASE, ('START.lock', 'RUN.lock')), (DATA_PARENT, ('START.lock', 'RUN.lock')), (DATA, ('RUN.lock',))):
            for name in names:
                h = stack.enter_context((directory / name).open('rb')); fcntl.flock(h.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        stop = threading.Event(); thread = threading.Thread(target=beat, args=(stop,), daemon=True); thread.start()
        try:
            progress('Checking finalized dataset and copying the retained engine/source; no old task is run')
            prepare_sources(); verify_data(); e = get_engine()
            freeze(ROOT / 'ENVIRONMENT.json', e.environment_record())
            metric_vendor_setup(); progress('Checking the reference caption metrics before feature extraction or training')
            child(['--metric-check'], timeout=1200)
            progress('Resolving public AST/GPT-2 assets; existing model and experiment files remain unchanged')
            prepare_models()
            child(['--gpu-check'], timeout=900, offline=True)
            progress('Building shared frozen AST features once; caption training follows automatically')
            child(['--features'], offline=True)
            frozen = {'protocol': PROTOCOL, 'wrapper_sha256': sha(ROOT / 'runner.py'), 'engine_sha256': ENGINE_SHA,
                'source_manifest_sha256': SOURCE_SHA, 'data': read(ROOT / 'DATA_BINDING.json'),
                'features_sha256': sha(ROOT / 'FEATURES_READY.json'), 'models': read(ROOT / 'MODELS_FROZEN.json'),
                'metric_assets_sha256': sha(ROOT / 'METRIC_ASSETS.json')}
            frozen['binding'] = digest(frozen); freeze(ROOT / 'PROTOCOL.json', frozen)
            for seed in SEEDS:
                for method in METHODS:
                    progress('AUDIO TRAINING: ' + method + '/seed' + str(seed) + ' (two-epoch development run)')
                    child(['--job', method, '--seed', seed], offline=True)
                    make_report('RUNNING; COMPLETED JOBS RETAINED')
            summaries = []
            for seed in SEEDS:
                for method in METHODS:
                    folder = ROOT / 'audio' / ('seed' + str(seed)) / method
                    done = read(folder / 'DONE.json')
                    for rel, h in done['files'].items(): require(sha(folder / rel) == h, 'Completed audio artifact changed')
                    s = read(folder / 'summary.json')
                    require(s['steps'] == 2778 and s['official_test_caption_outcomes_computed'] is False, 'Unexpected completed result')
                    summaries.append({'seed': seed, 'method': method, 'steps': s['steps'], 'development_metrics': s['final'],
                        'caption_diagnostics': s['caption_diagnostics'], 'shuffled_audio_diagnostic': s['shuffled_audio_diagnostic'],
                        'learned_policy_updates': s['learned_policy_updates'], 'controller_events': s['controller_event_count'],
                        'compute': s['compute'], 'wall_time_s': s['wall_time_s']})
            put(ROOT / 'PER_SEED.json', summaries); verify_data(); verify_features()
            require(sha(PILOT / 'runner.py') == ENGINE_SHA and sha(PILOT / 'SOURCE.json') == SOURCE_SHA, 'Retained engine/source manifest changed')
            put(ROOT / 'PRESERVATION.json', {'status': 'PASS', 'data_completion_and_manifest_hashes_rechecked': True,
                'retained_engine_and_source_manifest_rechecked': True, 'old_task_workspaces_not_opened_for_writing': True,
                'recordings_not_modified': True, 'scope': 'bound inputs and output isolation, not a new audit of all historical checkpoints'})
            make_report('AUDIOCAPS PILOT COMPLETE')
            stop.set(); thread.join(timeout=5)
            result = {'status': 'COMPLETE', 'completed_method_runs': 8, 'seeds': list(SEEDS), 'time': now(),
                'evidence_role': PROTOCOL['evidence_role'], 'official_test_caption_outcomes_computed': False,
                'report': str(ROOT / REPORT), 'report_sha256': sha(ROOT / REPORT)}
            put(ROOT / 'COMPLETE.json', result); put(ROOT / 'STATUS.json', result)
            print('AUDIOCAPS PILOT COMPLETE\nReport: ' + str(ROOT / REPORT), flush=True); return 0
        except BaseException as exc:
            stop.set(); thread.join(timeout=5)
            put(ROOT / 'FAILED.json', {'status': 'FAILED', 'time': now(), 'error': repr(exc), 'traceback': traceback.format_exc(),
                'completed_jobs_preserved': len(list(ROOT.glob('audio/seed*/*/DONE.json'))),
                'instruction': 'Share the training report or --status. Do not delete checkpoints, redownload AudioCaps or change dependencies.'})
            try: make_report('STOPPED; NOT COMPLETE')
            except Exception: pass
            traceback.print_exc(); return 1




def live():
    return shutil.which('tmux') is not None and subprocess.run(['tmux', 'has-session', '-t', SESSION], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def status():
    print('SERVER:', os.uname().nodename, now(), flush=True)
    print('TMUX ACTIVE' if live() else 'NO LIVE TMUX SESSION', flush=True)
    for name in ('FAILED.json', 'COMPLETE.json', 'STATUS.json'):
        if (ROOT / name).is_file(): print(json.dumps(read(ROOT / name), indent=2)); break
    else: print('Not started.')
    print('Completed audio pilot jobs (target 8):', len(list(ROOT.glob('audio/seed*/*/DONE.json'))))
    print('Completed feature chunks:', len(list((ROOT / 'features').glob('chunk_*.json'))), '(target 179)')
    if (ROOT / 'CHILD_FAILURE.json').exists(): print(json.dumps(read(ROOT / 'CHILD_FAILURE.json'), indent=2))
    print('Report:', ROOT / REPORT); print('Log:', ROOT / 'run.log')
    if shutil.which('nvidia-smi'): subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', '--format=csv'], check=False)




def start(resume=False):
    import fcntl
    require(sys.platform.startswith('linux') and Path('/mnt/caenl').is_mount(), 'Run on IBM with the persistent volume mounted')
    require(PYTHON.is_file() and shutil.which('tmux') and shutil.which('nvidia-smi'), 'Existing Python/tmux/NVIDIA tools missing; nothing installed')
    require((DATA / 'AUDIOCAPS_READY.json').is_file(), 'Finalized AudioCaps data is not ready')
    require(not ROOT.is_symlink(), 'Unsafe result root'); ROOT.mkdir(mode=0o700, exist_ok=True)
    with (ROOT / 'START.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if live(): print('ALREADY RUNNING: no duplicate started.'); status(); return
        if (ROOT / 'COMPLETE.json').exists(): status(); return
        require(resume or not (ROOT / 'FAILED.json').exists(), 'Previous attempt failed. Share --status before resuming.')
        gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
        require(not gpu.strip(), 'Another GPU process is active; it has not been stopped')
        require(shutil.disk_usage(ROOT).free >= 50 * 1024**3, 'Need 50 GiB free for features, model assets and new checkpoints; nothing deleted')
        raw = Path(__file__).read_bytes(); ast.parse(raw.decode('utf-8-sig'))
        target = ROOT / 'runner.py'
        if target.exists(): require(target.read_bytes() == raw, 'Different launcher already owns this study')
        else: write(target, raw)
        if resume:
            old = ROOT / ('resume_history_' + str(time.time_ns())); old.mkdir()
            for name in ('FAILED.json', 'CHILD_FAILURE.json'):
                if (ROOT / name).exists(): os.replace(ROOT / name, old / name)
        shell = 'set -e; if [ -f "$HOME/.caenl-java8-env" ]; then source "$HOME/.caenl-java8-env"; fi; '
        shell += 'unset CAENL_DEBUG_STOP_AT_STEP HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; '
        shell += 'export PATH=' + shlex.quote(str(PYTHON.parent)) + ':"$PATH"; '
        shell += 'export HF_HOME=' + shlex.quote(str(ROOT / 'assets/hf_home')) + '; '
        shell += 'export TORCH_HOME=' + shlex.quote(str(ROOT / 'assets/torch')) + '; '
        shell += 'export XDG_CACHE_HOME=' + shlex.quote(str(ROOT / 'cache/xdg')) + '; '
        shell += 'exec env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 '
        shell += 'CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DOWNLOAD_TIMEOUT=60 '
        shell += shlex.quote(str(PYTHON)) + ' ' + shlex.quote(str(target)) + ' --worker >> ' + shlex.quote(str(ROOT / 'run.log')) + ' 2>&1'
        subprocess.run(['tmux', 'new-session', '-d', '-s', SESSION, 'bash -c ' + shlex.quote(shell)], check=True)
        print('STARTED:', SESSION)
        print('Checks -> frozen audio features -> eight two-epoch audio DEVELOPMENT runs.')
        print('No dataset download, old task restart, package installation or official-test caption evaluation.')
        print('Status: python3 ~/caenl_audiocaps_train_pilot_v1.py --status')




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--start', action='store_true'); group.add_argument('--resume', action='store_true')
    group.add_argument('--status', action='store_true'); group.add_argument('--worker', action='store_true')
    group.add_argument('--metric-check', action='store_true'); group.add_argument('--gpu-check', action='store_true')
    group.add_argument('--features', action='store_true'); group.add_argument('--job', choices=METHODS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    a = parser.parse_args()
    if a.status: status()
    elif a.worker: return worker()
    elif a.metric_check: metric_check()
    elif a.gpu_check: gpu_check()
    elif a.features: extract_features()
    elif a.job:
        require(a.seed is not None, 'Seed required for internal job'); train_job(a.job, a.seed)
    else: start(a.resume)
    return 0


if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        print('STOPPED:', exc, file=sys.stderr); raise SystemExit(1)
# END OF CAENL AUDIOCAPS TRAINING PILOT V1