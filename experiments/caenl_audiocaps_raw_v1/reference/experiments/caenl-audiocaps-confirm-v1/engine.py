#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CAENL cross-modal DEVELOPMENT pilots v1, 20 September 2026.


Sequential real-data diffusion and audio-captioning pilots, not final multi-seed
confirmation. Reuses a hash-checked COPY of the retained v5.4.3 implementation.
Existing ImageNet/C4/UHerding source, results, environments and data are not edited.
No packages are installed. Public model/data downloads are permitted; inaccessible
AudioCaps clips are recorded, never replaced by synthetic audio or another corpus.
This launcher was source-reviewed but could not be executed in the authoring
runtime. The embedded CPU/GPU and metric checks run on IBM before task training.
"""
from __future__ import annotations
import argparse
import ast
import concurrent.futures
import copy
import gc
import hashlib
import importlib.metadata
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
import time
import traceback
from datetime import datetime, timezone


sys.dont_write_bytecode = True
SESSION = 'caenl-crossmodal-pilots-v1'
ROOT = Path('/mnt/caenl/active/results') / SESSION
PROJECT = Path.home() / 'caenl-v5.4.3/caenl_revision_pipeline_v5.4.3_c4_confirmatory'
REPORT = SESSION + '.md'
SEEDS = (1001, 1002)
METHODS = ('baseline', 'fixed_dcr', 'macc_lite', 'full_macc')
PINS = {
 'audio/audiocaps.py': 'e7aa1b9747dd1c66693b1619627028b2272ea0c1036c0241d7b1eb84d88dc424',
 'audio/captioning.py': '438c3284238702d31cebab8835d1f87242c5fae20d7eea969cb1011797ba7281',
 'audio/common.py': '432c7c003ebd598f34ff133e3ad9ae7ebd5ead490001612b3c8ba7b7fc9de5cc',
 'audio/data.py': '15587567bf11ddc5fe089c12cf23a7ace4ed44fb63c3965b758cbd499323ec20',
 'audio/metrics.py': '25380bd44ac9d41cdf7cca0cf85a0886bfb0038df474b998e8f8fd5f0fdf0ef5',
 'core/control_loop.py': '47d7c12c84463c2287a5ab2223bfa8049878382fc3457dfe807f8ba87c0e1aba',
 'core/controllers.py': '9e5fb0237e49643a9a2e8b24077a5b9f1cfd81487be298aa549d432a5345b88e',
 'core/dcr.py': '0283aac8e0407ddf15a19a8e947f689be61a7dbac5cad1e597df0d1423183644',
 'core/monitor.py': '598f2c326d664f66e1c73431e6ed78a73a1402affe12900007f3c3babdd1e2c9',
 'diffusion/task.py': '51eefd38caab642b38a102be0b201f21e824774110438bbd266abcd4efcbc820',
 'utils/jobctx.py': '2298f8c5ae73f918e2afe9d0d52101eb5bc02cb74d3b7f44fe35557f23b81336',
 'utils/seeding.py': '33b837c944f367e10f743f60b0d19ec290290d3ee776fc7861c15b8b8386e0a4',
 'vision/data.py': 'db19385befa56ff58e27c45d355c283e695be43627b6c03d60ffe4c1a8dbec1c',
}
PLAN = {
 'role': 'development_pilots_only_not_final_confirmatory_evidence',
 'pilot_seeds': list(SEEDS), 'conditions': list(METHODS),
 'diffusion': {'task': 'CIFAR10 pretrained DDPM fine-tuning',
   'model': 'google/ddpm-cifar10-32', 'train_steps': 2000, 'batch_size': 128,
   'lr': 2e-5, 'ema_decay': 0.9995, 'sampling_steps': 100, 'samples_per_condition': 5000,
   'sampling_batch_size': 128, 'train_images': 45000,
   'internal_holdout_from_original_training_split': 5000,
   'controller_images': 256, 'development_mse_images': 1024,
   'official_test_outcomes_used': False,
   'pretraining_overlap_note': 'Fine-tuning holdout was part of pretrained CIFAR10 data; not pristine relative to pretraining.',
   'fid_reference': 'full original CIFAR10 training distribution, 50000 images',
   'additional_reference': 'unmodified pretrained model evaluated with the same sampler/noise for each pilot seed'},
 'audio': {'dataset': 'AudioCaps, pinned official caption CSVs',
   'minimum_available_fraction_in_each_official_split': 0.80,
   'no_silent_subset_or_dataset_substitution': True,
   'backbone': 'MIT/ast-finetuned-audioset-10-10-0.4593', 'decoder': 'gpt2',
   'frozen_AST': True, 'trainable_AST_token_encoder_and_GPT2_decoder': True,
   'epochs': 2, 'batch_size': 32, 'encoder_lr': 1e-4, 'decoder_lr': 5e-5,
   'tokens_per_audio_window': 64, 'max_windows': 1,
   'controller_validation_clips': 64,
   'caption_evaluation': 'all available official validation clips excluding the controller subset',
   'official_test_caption_outcomes_used': False,
   'conditioning_diagnostic': 'development token NLL after deterministic mismatching of audio and captions; no extra training and no favorable-result gate',
   'metric_implementation': 'existing pycocoevalcap PTB, CIDEr-D, BLEU, ROUGE-L, METEOR, SPICE; no fallback',
   'pretraining_overlap_note': 'AST uses AudioSet pretraining, which may include the underlying AudioCaps recordings.'},
 'spectral_objective': {'version': 'retained spectral DCR, not aligned classification DCR',
   'target_rank': 16, 'retained_eigenvalues': 64, 'all_diffusion_timesteps': [0, 999],
   'fixed_lambda': 0.01, 'controller_initial_lambda': 0.01, 'controller_max_lambda': 0.05,
   'monitor_interval_steps': 50, 'covariance_ema_decay': 0.99,
   'lite_error': 'absolute_log_ratio minus 0.1 deadband', 'lite_step_size': 0.001,
   'full_macc_terminal_reward': 'reward the last pending action without drawing an unexecuted next action',
   'full_macc_query_head': False},
 'stopping_rule': 'all finite outcomes, including harmful or null effects, retained; no minimum performance gain',
 'next_after_review_only': 'freeze independent six-seed final protocols and task budgets using pilot behavior and timings; no final jobs launched by this script',
 'authoring_runtime_execution': False,
}
np = None
torch = None




def utc():
    return datetime.now(timezone.utc).isoformat()




def require(ok, message):
    if not bool(ok):
        raise RuntimeError(message)




def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()




def clean(obj):
    if isinstance(obj, dict): return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [clean(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj): return None
    if isinstance(obj, Path): return str(obj)
    if hasattr(obj, 'item') and not isinstance(obj, (str, bytes)):
        try: return clean(obj.item())
        except Exception: pass
    return obj




def digest(obj):
    return hashlib.sha256(json.dumps(clean(obj), sort_keys=True, allow_nan=False).encode()).hexdigest()




def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))




def write(path, text):
    p = Path(path)
    require(p.resolve().is_relative_to(ROOT.resolve()), 'Refusing output outside the NEW result tree: ' + str(p))
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_name(p.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, p)




def put(path, obj):
    write(path, json.dumps(clean(obj), indent=2, allow_nan=False) + '\n')




def progress(message):
    put(ROOT / 'STATUS.json', {'status': 'RUNNING', 'pid': os.getpid(), 'time': utc(), 'message': message})
    print(message, flush=True)




def source_setup(create=False):
    original = PROJECT / 'src/caenl'
    copied = ROOT / 'source/caenl'
    if create:
        require(original.is_dir(), 'The retained v5.4.3 source folder was not found.')
        for relative, expected in PINS.items():
            require((original / relative).is_file() and sha(original / relative) == expected,
                    'Reviewed source differs: ' + relative + '. No original source was changed.')
        files = {}
        for p in sorted(original.rglob('*.py')):
            require(not p.is_symlink(), 'Source symlink refused: ' + str(p))
            rel = p.relative_to(original).as_posix()
            raw = p.read_bytes(); ast.parse(raw.decode('utf-8-sig'), filename=rel)
            files[rel] = hashlib.sha256(raw).hexdigest()
            dest = copied / rel
            if dest.exists():
                require(sha(dest) == files[rel], 'Existing isolated source changed: ' + rel)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(raw)
        record = {'source': str(original), 'files': files, 'critical_files': PINS}
        if (ROOT / 'SOURCE.json').exists(): require(read(ROOT / 'SOURCE.json') == record, 'Source changed since pilot start')
        else: put(ROOT / 'SOURCE.json', record)
    record = read(ROOT / 'SOURCE.json')
    for rel, expected in record['files'].items():
        p = copied / rel
        require(not p.is_symlink() and p.is_file() and sha(p) == expected, 'Copied source integrity failure: ' + rel)
    sys.path.insert(0, str(ROOT / 'source'))
    return record




def runtime():
    global np, torch
    import numpy as numpy
    import torch as pytorch
    np, torch = numpy, pytorch
    require(str(torch.__version__).startswith('2.6.') and str(torch.version.cuda) == '12.4',
            'Expected existing PyTorch 2.6/CUDA 12.4. No dependency changes were attempted.')
    require(torch.cuda.is_available() and 'L40S' in torch.cuda.get_device_name(0), 'Expected accessible NVIDIA L40S')
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return torch.device('cuda:0')




def tensor_hash(model):
    h = hashlib.sha256()
    for key, val in sorted(model.state_dict().items()):
        t = val.detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()




def resolve_model(repo):
    from huggingface_hub import snapshot_download
    db = ROOT / 'MODELS.json'
    records = read(db) if db.exists() else {}
    if repo in records:
        rec = records[repo]
        for rel, expected in rec['files'].items():
            require(sha(Path(rec['path']) / rel) == expected, 'Pinned model asset changed: ' + repo + '/' + rel)
        return rec['path']
    patterns = ['config.json', 'scheduler_config.json', 'model_index.json', 'generation_config.json',
                'preprocessor_config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                'vocab.json', 'merges.txt', '*.safetensors', 'pytorch_model.bin', 'diffusion_pytorch_model.bin']
    caches = [ROOT / 'assets/hf', Path('/mnt/caenl/active/data/hf'), Path('/mnt/caenl/active/cache/huggingface/hub'),
              Path('/dev/shm/data/hf'), Path.home() / '.cache/huggingface/hub']
    found = None
    for cache in caches:
        if not cache.exists(): continue
        try:
            candidate = Path(snapshot_download(repo, cache_dir=str(cache), local_files_only=True, allow_patterns=patterns))
            if (candidate / 'config.json').is_file() and (list(candidate.glob('*.safetensors')) or list(candidate.glob('*.bin'))):
                if repo == 'gpt2' and not ((candidate / 'vocab.json').is_file() and (candidate / 'merges.txt').is_file()): continue
                if 'ast-' in repo and not (candidate / 'preprocessor_config.json').is_file(): continue
                if 'ddpm-' in repo and not (candidate / 'scheduler_config.json').is_file(): continue
                found = candidate; break
        except Exception: pass
    if found is None:
        progress('Downloading public pretrained assets: ' + repo)
        found = Path(snapshot_download(repo, cache_dir=str(ROOT / 'assets/hf'), allow_patterns=patterns))
    files = {p.relative_to(found).as_posix(): sha(p) for p in sorted(found.rglob('*')) if p.is_file() and p.suffix in ('.json', '.bin', '.safetensors', '.txt')}
    require(bool(files), 'Empty pretrained asset snapshot: ' + repo)
    records[repo] = {'path': str(found), 'snapshot_revision': found.name, 'files': files,
                     'trust_remote_code': False, 'resolved_before_task_training': True}
    put(db, records)
    return str(found)




def configure(task, method, seed):
    all_models = read(ROOT / 'MODELS.json')
    needed = [PLAN['diffusion']['model']] if task == 'diffusion' else [PLAN['audio']['backbone'], 'gpt2']
    models = {name: all_models[name] for name in needed}
    cfg = {'seed': seed, 'job_id': task + '_' + method + '_seed' + str(seed),
       'role': PLAN['role'], 'paths': {'data_root': str(ROOT / 'data'), 'cache_root': str(ROOT / 'cache')},
       'method': {'name': method, 'regularizer': 'none' if method in ('baseline', 'pretrained_reference') else method,
                  'dcr_lambda': 0.0 if method in ('baseline', 'pretrained_reference') else 0.01},
       'collapse': {'truncated_spectral_rank': 64, 'covariance_ema_decay': 0.99},
       'macc_lite': {'lambda_initial': 0.01, 'lambda_min': 0.0, 'lambda_max': 0.05,
                     'eta_lambda': 0.001, 'error_mode': 'absolute_log_ratio', 'deadband': 0.1},
       'full_macc': {'lambda_min': 0.0, 'lambda_max': 0.05, 'hidden_dimensions': [128, 128],
         'action_space': {'lambda_delta_choices': [-0.005, -0.0025, 0.0, 0.0025, 0.005], 'threshold_quantiles': [0.9]},
         'policy_learning_rate': 0.0003, 'exploration_epsilon_start': 0.10, 'exploration_epsilon_end': 0.02,
         'reward_baseline_ema': 0.05, 'entropy_coefficient': 0.001,
         'reward': {'validation_delta_weight': 1.0, 'validation_delta_scale': 10.0,
                    'query_penalty_beta': 0.0, 'target_deviation_gamma': 0.1, 'value_delta_clip': 2.0}},
       'reproducibility': {'save_pip_freeze': False, 'telemetry_interval_s': 10}}
    if task == 'diffusion':
        cfg['diffusion'] = {'dataset': 'cifar10', 'model': models[PLAN['diffusion']['model']]['path'],
          'train_steps': 0 if method == 'pretrained_reference' else 2000, 'batch_size': 128,
          'lr': 2e-5, 'weight_decay': 0.0, 'ema_decay': 0.9995, 'grad_clip': 1.0,
          'configured_rank': 16, 'target_effective_rank': 16, 'monitoring_cadence_steps': 50,
          'controller_error_mode': 'absolute_log_ratio', 'dcr_timestep_range': [0, 999],
          'val_examples': 256, 'report_val_examples': 1024, 'num_samples': 5000, 'save_samples': 5000,
          'sampling_steps': 100, 'sampling_batch_size': 128, 'amp': True,
          'log_every_steps': 50, 'eval_every_steps': 500, 'checkpoint_every_steps': 250,
          'fid_reference': 'cifar10-train', 'evaluation_role': 'development; no official test outcomes'}
        cfg['data_binding'] = {'manifest': str(ROOT / 'DIFFUSION_DATA.json'),
                               'sha256': sha(ROOT / 'DIFFUSION_DATA.json'),
                               'training_images': 45000, 'heldout_images': 5000}
    else:
        cfg['audio'] = {'dataset': 'audiocaps', 'manifest': str(ROOT / 'audio_corpus/manifest.jsonl'),
          'manifest_min_coverage': 0.80, 'manifest_verify_hashes': 'all',
          'backbone': models[PLAN['audio']['backbone']]['path'], 'tokens_per_window': 64, 'max_windows': 1,
          'configured_rank': 16, 'target_effective_rank': 16, 'monitoring_cadence_steps': 50,
          'controller_error_mode': 'absolute_log_ratio',
          'captioning': {'decoder': models['gpt2']['path'], 'prefix_length': 8, 'encoder_layers': 2, 'heads': 8,
             'epochs': 2, 'batch_size': 32, 'lr': 0.0001, 'decoder_lr': 0.00005, 'weight_decay': 0.01,
             'warmup_steps': 100, 'max_caption_tokens': 48, 'ctrl_val_clips': 64,
             'decode_batch_size': 32, 'max_new_tokens': 30, 'beam_size': 3,
             'java_metrics': True, 'spice': True, 'official_metrics': True,
             'log_every_steps': 50, 'checkpoint_every_steps': 250}}
        cfg['data_binding'] = read(ROOT / 'AUDIO_READY.json')
    cfg['crossmodal_binding'] = digest({'config': cfg, 'plan': PLAN, 'source': read(ROOT / 'SOURCE.json'),
                                       'models': models, 'runner': sha(ROOT / 'runner.py')})
    return cfg




def install_window_loop():
    from caenl.core.control_loop import ControlLoop
    from caenl.core.controllers import FullMACC
    class TerminalSafeLoop(ControlLoop):
        def update(self, feats, value_fn=None, extra_state=()):
            self.step += 1
            terminal = self.step == self.total_steps
            event = self.step % self.cadence == 0 or terminal
            with torch.no_grad():
                self.monitor.update({l: f.detach() for l, f in feats.items() if l in self.monitor.layers},
                                    compute_kappa=event or self.step <= 2)
            info = {}
            if event and self.regularizer in ('macc_lite', 'full_macc'):
                kappa = self.monitor.kappa()
                require(all(math.isfinite(float(v)) and float(v) > 0 for v in kappa.values()), 'Nonfinite spectral feedback')
                if isinstance(self.controller, FullMACC):
                    value = float(value_fn())
                    require(math.isfinite(value), 'Nonfinite held-out controller value')
                    delta = None if self.prev_value is None else value - self.prev_value
                    self.prev_value = value
                    if terminal:
                        upd = {}
                        if self.controller._pending is not None and delta is not None:
                            components = self.controller.reward_components(delta, 0, kappa)
                            upd = self.controller.update(components['reward']); upd.update(components)
                        require(self.controller._pending is None, 'Uncredited terminal policy action')
                        self.controller.history.append({'kind': 'terminal', 'step': self.step, 'kappa': kappa,
                             'lambdas': dict(self.lambdas), 'update': upd, 'new_action_drawn': False})
                    else:
                        self.action, upd = self.controller.observe_and_act(kappa, self.monitor.instability(),
                            self.step / max(1, self.total_steps), delta, 0, extra_state, step_index=self.step)
                    info.update(ctrl_value=value, ctrl_delta=delta)
                    info.update({k: v for k, v in upd.items() if isinstance(v, (int, float))})
                else:
                    self.action = self.controller.step(kappa, step_index=self.step)
                require(all(math.isfinite(float(v)) and 0 <= float(v) <= 0.050001 for v in self.lambdas.values()), 'Coefficient bound violated')
                info.update({f'lambda/{l}': v for l, v in self.lambdas.items()})
                info.update(self.monitor.record())
            return info
    import caenl.diffusion.task as d
    import caenl.audio.captioning as a
    d.ControlLoop = TerminalSafeLoop
    a.ControlLoop = TerminalSafeLoop
    return TerminalSafeLoop




def checks():
    source_setup(); device = runtime()
    from caenl.core.dcr import DCR
    from caenl.audio.captioning import PrefixCaptioner
    from caenl.audio.common import CharTokenizer, pad_batch
    from transformers import GPT2Config, GPT2LMHeadModel
    from diffusers import UNet2DModel
    loop_class = install_window_loop()
    checks_done = []
    torch.manual_seed(5519)
    z = torch.randn(32, 64, device=device, requires_grad=True)
    reg = DCR.build({'mid': 64}, k=32, configured_rank=16, covariance='total', device=device)
    loss, _ = reg({'mid': z}, {'mid': .01})
    gradient = torch.autograd.grad(loss, z)[0]
    require(torch.isfinite(gradient).all() and float(gradient.norm()) > 0, 'Spectral DCR gradient test failed')
    checks_done.append('finite nonzero spectral DCR gradient')
    test_cfg = {'macc_lite': {'lambda_max': .05, 'eta_lambda': .001},
                'full_macc': {'lambda_max': .05, 'hidden_dimensions': [16, 16],
                   'exploration_epsilon_start': 0., 'exploration_epsilon_end': 0.,
                   'action_space': {'lambda_delta_choices': [-.005, 0, .005]}}}
    for kind in ('none', 'fixed_dcr', 'macc_lite', 'full_macc'):
        torch.manual_seed(42)
        loop = loop_class({'mid': 64}, kind, 0 if kind == 'none' else .01, test_cfg, device, 13, 20,
                          k=32, configured_rank=16, cadence=2)
        for i in range(20):
            loop.update({'mid': z.detach()}, lambda j=i: -1.0 + j * .002, (i / 20,))
            if i == 9:
                st = loop.state_dict()
                new = loop_class({'mid': 64}, kind, .01, test_cfg, device, 13, 20, k=32, configured_rank=16, cadence=2)
                new.load_state_dict(st); loop = new
        require(loop.step == 20, 'Controller restore step mismatch')
        if kind == 'full_macc':
            require(loop.controller._pending is None, 'Terminal credit test failed')
            require(sum('policy_loss' in r.get('update', {}) for r in loop.controller.history) >= 5, 'Full MACC is not learning')
        checks_done.append(kind + ': real updates, state restoration, terminal processing')
    model = UNet2DModel(sample_size=16, in_channels=3, out_channels=3, layers_per_block=1,
                       block_out_channels=(32, 64), down_block_types=('DownBlock2D', 'AttnDownBlock2D'),
                       up_block_types=('AttnUpBlock2D', 'UpBlock2D'), norm_num_groups=8).to(device)
    x = torch.randn(4, 3, 16, 16, device=device)
    mse = model(x, torch.tensor([1, 5, 12, 20], device=device)).sample.square().mean()
    mse.backward()
    require(torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)), 'U-Net backward failed')
    checks_done.append('real tiny Diffusers U-Net GPU backward')
    del model, x, mse
    conf = GPT2Config(vocab_size=259, n_positions=80, n_embd=32, n_layer=2, n_head=2,
                      bos_token_id=256, eos_token_id=257, pad_token_id=258)
    cap = PrefixCaptioner(16, GPT2LMHeadModel(conf), 32, prefix_len=4, enc_layers=1, n_heads=2).to(device)
    tok = CharTokenizer(); ids, mask = pad_batch([tok.encode('a bell'), tok.encode('wind')], 258)
    tokens = torch.randn(2, 8, 16, device=device); tm = torch.ones(2, 8, dtype=torch.bool, device=device)
    ce, features = cap(tokens, tm, ids.to(device), mask.long().to(device))
    ce.backward()
    require(torch.isfinite(ce) and float(torch.nn.utils.clip_grad_norm_(cap.parameters(), 1., error_if_nonfinite=True)) > 0, 'Caption-model backward failed')
    cap.eval()
    generated = cap.generate(tokens, tm, 256, 257, 258, max_new_tokens=4, num_beams=2)
    require(len(generated) == 2 and all(isinstance(r, list) and len(r) > 0 for r in generated), 'Caption generation API failed')
    checks_done.append('real tiny prefix-captioner GPU backward and beam-generation API')
    put(ROOT / 'SELF_TEST.json', {'status': 'PASS', 'checks': checks_done, 'scientific_runs_counted': 0,
                                  'tested_on': 'IBM host', 'authoring_runtime_execution': False})
    print('CROSS-MODAL IMPLEMENTATION CHECKS: PASS', flush=True)




def metric_test():
    source_setup(); runtime()
    from caenl.audio.metrics import official_metrics
    m = official_metrics(['a dog barks outside', 'a person plays a guitar'],
       [['a dog is barking outside', 'a dog barks loudly outdoors'],
        ['someone is playing a guitar', 'a person plays music on a guitar']], java=True, spice=True)
    require(all(k in m and math.isfinite(float(m[k])) for k in ('cider_d', 'bleu4', 'meteor', 'spice')), 'Required caption metric self-test failed')
    put(ROOT / 'AUDIO_METRIC_TEST.json', {'status': 'PASS', 'scores': m, 'toy_not_scientific_data': True})




def prepare_diffusion():
    progress('Preparing the real CIFAR10 diffusion pilot and pinning pretrained weights')
    resolve_model(PLAN['diffusion']['model'])
    import numpy as np_local
    from caenl.vision.data import load_vision_dataset
    ds = load_vision_dataset('cifar10', ROOT / 'data', ROOT / 'cache', {})
    require(len(ds.train_y) == 50000 and len(ds.test_y) == 10000 and tuple(ds.train_x.shape[1:]) == (32, 32, 3), 'Expected full native CIFAR10 cache')
    rng = np_local.random.default_rng(781302)
    order = rng.permutation(50000)
    held, train = order[:5000], order[5000:]
    hashes = {}
    for name, array in [('train_images', ds.train_x), ('train_labels', ds.train_y), ('test_images', ds.test_x), ('test_labels', ds.test_y)]:
        hashes[name] = hashlib.sha256(np_local.ascontiguousarray(array).tobytes()).hexdigest()
    rec = {'dataset': 'cifar10', 'arrays_sha256': hashes, 'train_indices': train.tolist(), 'heldout_indices': held.tolist(),
           'official_test_evaluated': False, 'purpose': 'pilot calibration only; internal heldout belongs to the pretrained model original training distribution'}
    if (ROOT / 'DIFFUSION_DATA.json').exists(): require(read(ROOT / 'DIFFUSION_DATA.json') == rec, 'CIFAR10 data changed')
    else: put(ROOT / 'DIFFUSION_DATA.json', rec)




def inspect_audio(path):
    try:
        import soundfile as sf
        info = sf.info(str(path)); duration = info.frames / info.samplerate
        return path.is_file() and path.stat().st_size > 1000 and 8 <= duration <= 12 and info.channels > 0
    except Exception:
        return False




def get_public_clip(item):
    split, yt, start = item
    require(re.fullmatch(r'[A-Za-z0-9_-]{11}', yt) is not None and isinstance(start, int) and start >= 0, 'Invalid official clip ID')
    dest = ROOT / 'audio_corpus/audio' / split / (yt + '_' + str(start) + '.wav')
    if inspect_audio(dest): return item, str(dest), None
    dest.parent.mkdir(parents=True, exist_ok=True)
    work = ROOT / 'audio_corpus/download_temp'; work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='clip_', dir=work) as tmp:
        template = str(Path(tmp) / 'segment.%(ext)s')
        cmd = [shutil.which('yt-dlp'), '--ignore-config', '--no-playlist', '--quiet', '--no-warnings',
               '--retries', '1', '--fragment-retries', '1', '--socket-timeout', '15', '-f', 'bestaudio/best',
               '--download-sections', '*' + str(start) + '-' + str(start + 10),
               '--extract-audio', '--audio-format', 'wav', '--postprocessor-args', 'ffmpeg:-ar 16000 -ac 1 -t 10',
               '-o', template, 'https://www.youtube.com/watch?v=' + yt]
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
            try:
                stdout, stderr = process.communicate(timeout=120)
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                process.communicate()
                raise
            result = subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
            wav = Path(tmp) / 'segment.wav'
            if result.returncode == 0 and inspect_audio(wav):
                require(not dest.exists(), 'Destination appeared during download')
                shutil.copyfile(wav, dest)
                return item, str(dest), None
            # Do not export raw downloader errors, cookies, headers or proxy configuration.
            return item, None, 'unavailable_or_invalid_segment_rc_' + str(result.returncode)
        except subprocess.TimeoutExpired:
            return item, None, 'download_timeout'
        except Exception as exc:
            return item, None, type(exc).__name__




def prepare_audio():
    from caenl.audio.audiocaps import fetch_caption_csvs, read_official_clips, OFFICIAL_CLIPS, AUDIOCAPS_REPO_COMMIT, AUDIOCAPS_CSV_SHA256, write_manifest, validate_audio_manifest
    corpus = ROOT / 'audio_corpus'; corpus.mkdir(exist_ok=True)
    ready = ROOT / 'AUDIO_READY.json'
    if ready.exists():
        rec = read(ready)
        require(sha(corpus / 'manifest.jsonl') == rec['manifest_sha256'], 'Frozen AudioCaps manifest changed')
        rep = validate_audio_manifest(corpus / 'manifest.jsonl', min_coverage=.8, verify_hashes='all')
        require(rep['valid'], 'Frozen AudioCaps files changed: ' + str(rep['errors']))
        return
    progress('AudioCaps: verifying official caption files and looking for existing local clips')
    csvs = fetch_caption_csvs(corpus)
    clips = read_official_clips(csvs)
    require({k: len(v) for k, v in clips.items()} == OFFICIAL_CLIPS, 'Official caption inventory differs')
    roots = [corpus, Path('/mnt/caenl/persistent/datasets/audiocaps'), Path('/mnt/caenl/active/data/audiocaps'),
             Path('/mnt/caenl/data/audiocaps'), Path('/dev/shm/data/audiocaps')]
    existing = {}
    for root in roots:
        manifest = root / 'manifest.jsonl'
        if manifest.is_file():
            for line in manifest.read_text().splitlines():
                try:
                    r = json.loads(line); sp = 'val' if r.get('split') == 'valid' else r.get('split')
                    key = (r.get('youtube_id'), int(r.get('start_time', -1)))
                    if key in clips.get(sp, {}) and inspect_audio(Path(r['path'])):
                        existing[(sp, key[0], key[1])] = Path(r['path'])
                except (ValueError, KeyError, TypeError): pass
    found = {k: {} for k in clips}; missing = []
    for split, entries in clips.items():
        directories = []
        for root in roots:
            directories += [root / 'audio' / split, root / split]
            if split == 'val': directories += [root / 'audio/valid', root / 'valid']
        directories = [d for d in directories if d.is_dir()]
        for yt, st in sorted(entries):
            p = existing.get((split, yt, st))
            if p is None:
                for directory in directories:
                    candidate = directory / (yt + '_' + str(st) + '.wav')
                    if inspect_audio(candidate): p = candidate; break
            if p is not None: found[split][(yt, st)] = p
            else: missing.append((split, yt, st))
    failures = {}; attempted = 0; source_blocked = False
    def accept(result):
        nonlocal attempted
        item, path, error = result; attempted += 1
        split, yt, st = item
        if path is not None: found[split][(yt, st)] = Path(path)
        else: failures[split + '/' + yt + '_' + str(st)] = error
    if missing:
        if not shutil.which('yt-dlp') or not shutil.which('ffmpeg'):
            source_blocked = True
        else:
            # A small distributed access probe prevents an overnight queue of blocked requests.
            import random
            g = random.Random(41708); probe = []
            for split in clips:
                group = [r for r in missing if r[0] == split]
                probe += g.sample(group, min(4, len(group)))
            progress('AudioCaps: testing public access to ' + str(len(probe)) + ' missing clips; no cookies or access bypass')
            before = sum(map(len, found.values()))
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                for result in ex.map(get_public_clip, probe): accept(result)
            gained = sum(map(len, found.values())) - before
            source_blocked = gained < math.ceil(len(probe) / 2)
            if not source_blocked:
                todo = [r for r in missing if r not in set(probe)]
                progress('AudioCaps: acquiring remaining public clips directly on IBM; unavailable clips are logged')
                deadline = time.monotonic() + 12 * 3600
                iterator = iter(todo)
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                    pending = {}
                    for _ in range(4):
                        item = next(iterator, None)
                        if item is not None: pending[ex.submit(get_public_clip, item)] = item
                    while pending:
                        done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                        for future in done:
                            pending.pop(future); accept(future.result())
                            if attempted % 200 == 0:
                                progress('AudioCaps downloads: ' + str(attempted) + ' attempted; available ' + str({k: len(v) for k, v in found.items()}))
                            if time.monotonic() < deadline:
                                item = next(iterator, None)
                                if item is not None: pending[ex.submit(get_public_clip, item)] = item
    coverage = {k: len(found[k]) / len(clips[k]) for k in clips}
    put(corpus / 'availability.json', {'time': utc(), 'official': OFFICIAL_CLIPS, 'available': {k: len(v) for k, v in found.items()},
            'coverage': coverage, 'download_attempts': attempted, 'access_probe_blocked': source_blocked,
            'download_failure_count': len(failures), 'download_policy': 'public URLs only; no credentials, restriction bypass or automatic retries beyond one per clip'})
    put(corpus / 'unavailable_clips.json', failures)
    if not all(v >= .80 for v in coverage.values()):
        put(ROOT / 'AUDIO_BLOCKED.json', {'status': 'BLOCKED_DATA', 'coverage': coverage,
            'available': {k: len(v) for k, v in found.items()}, 'required_minimum': .80,
            'reason': 'Insufficient verified AudioCaps coverage or public access unavailable. No audio training occurred; no smaller or synthetic dataset substituted.'})
        return
    for split, group in found.items():
        require(all(inspect_audio(p) for p in group.values()), 'Invalid audio segment in ' + split)
    info = write_manifest(clips, found, corpus / 'manifest.jsonl', hash_audio=True)
    put(corpus / 'manifest_meta.json', {'dataset': 'audiocaps', 'captions_source': {'repository': 'cdjkim/audiocaps', 'commit': AUDIOCAPS_REPO_COMMIT, 'csv_sha256': AUDIOCAPS_CSV_SHA256},
         'official_clips': OFFICIAL_CLIPS, 'coverage': coverage, **info})
    rep = validate_audio_manifest(corpus / 'manifest.jsonl', min_coverage=.80, verify_hashes='all')
    require(rep['valid'], 'AudioCaps manifest failed validation: ' + str(rep['errors']))
    put(ready, {'status': 'READY', 'manifest_sha256': sha(corpus / 'manifest.jsonl'), 'n_clips': rep['n_clips'],
                'splits': rep['splits'], 'coverage': coverage, 'captions_commit': AUDIOCAPS_REPO_COMMIT,
                'every_retained_file_hashed': True, 'subset_is_available_official_clips_not_full_coverage_claim': True})




def install_task_classes(task):
    from caenl.utils.seeding import derive_seed
    install_window_loop()
    if task == 'diffusion':
        import dataclasses
        import caenl.diffusion.task as m
        original_loader = m.load_vision_dataset
        def pilot_loader(name, data_root, cache_root, options):
            ds = original_loader(name, data_root, cache_root, options)
            rec = read(ROOT / 'DIFFUSION_DATA.json')
            train = np.asarray(rec['train_indices']); held = np.asarray(rec['heldout_indices'])
            return dataclasses.replace(ds, train_x=np.asarray(ds.train_x[train]), train_y=np.asarray(ds.train_y[train]),
                         test_x=np.asarray(ds.train_x[held]), test_y=np.asarray(ds.train_y[held]))
        m.load_vision_dataset = pilot_loader
        base = m.DiffusionJob
        class Pilot(base):
            def setup(self):
                super().setup()
                initial = tensor_hash(self.unet)
                key = ROOT / ('DIFFUSION_INITIAL_seed' + str(self.ctx.seed) + '.json')
                if key.exists(): require(read(key)['model_sha256'] == initial, 'Diffusion methods did not share pretrained initialization')
                else: put(key, {'model_sha256': initial, 'seed': self.ctx.seed})
                torch.manual_seed(derive_seed(self.ctx.seed, 'common_training_rng'))
                torch.cuda.manual_seed_all(derive_seed(self.ctx.seed, 'common_training_rng'))
            def fidelity(self, samples):
                from caenl.utils.seeding import derive_seed
                # KID subset randomness is fixed across conditions as well as the generation noise.
                np.random.seed(derive_seed(0, 'pilot_fidelity'))
                torch.manual_seed(derive_seed(0, 'pilot_fidelity'))
                return super().fidelity(samples)
        m.DiffusionJob = Pilot
        return m.run
    import caenl.audio.captioning as m
    base = m.CaptioningJob
    class Pilot(base):
        def setup(self):
            super().setup()
            self.model.float()
            require(len(self.pairs) >= 39800 and len(self.report_val) >= 300, 'Audio pilot would be too small; not launched')
            self.official_test_ids = [self.rows[int(i)]['id'] for i in self.splits['test']]
            # The old task driver decodes the key named test. Route it only to the
            # development portion of official validation and explicitly relabel outputs.
            self.splits['test'] = self.report_val.copy()
            put(self.ctx.job_dir / 'EVALUATION_SPLIT.json', {'caption_split': 'official_validation_minus_controller',
                'controller_clip_ids': [self.rows[int(i)]['id'] for i in self.ctrl_val],
                'development_clip_ids': [self.rows[int(i)]['id'] for i in self.report_val],
                'official_test_outcomes_computed': False, 'official_test_clip_count_reserved': len(self.official_test_ids)})
            initial = tensor_hash(self.model)
            key = ROOT / ('AUDIO_INITIAL_seed' + str(self.ctx.seed) + '.json')
            if key.exists(): require(read(key)['model_sha256'] == initial, 'Audio conditions have different initial models')
            else: put(key, {'model_sha256': initial, 'seed': self.ctx.seed})
            torch.manual_seed(derive_seed(self.ctx.seed, 'common_training_rng'))
            torch.cuda.manual_seed_all(derive_seed(self.ctx.seed, 'common_training_rng'))
        @torch.no_grad()
        def val_loss(self, clips):
            # Correct corpus token weighting rather than averaging batch means by clip count.
            self.model.eval(); total = 0.; count = 0
            pairs = [(int(i), c) for i in clips for c in self.rows[int(i)]['captions']]
            for pos in range(0, len(pairs), self.bs):
                x, mask, ids, am = self.batch(pairs[pos:pos + self.bs])
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.amp):
                    loss, _ = self.model(x, mask, ids, am)
                n = int(am[:, 1:].sum()); total += float(loss) * n; count += n
            self.model.train()
            require(count > 0 and math.isfinite(total), 'Invalid token-pooled audio loss')
            return total / count
        @torch.no_grad()
        def shuffled_audio_nll(self):
            self.model.eval()
            ids = np.asarray(self.report_val)
            shift = 1 + derive_seed(self.ctx.seed, 'development_audio_shuffle') % (len(ids) - 1)
            shuffled = np.roll(ids, shift)
            pairs = [(int(j), caption) for i, j in zip(ids, shuffled) for caption in self.rows[int(i)]['captions']]
            total = 0.; count = 0
            for pos in range(0, len(pairs), self.bs):
                x, mask, tokens, attention = self.batch(pairs[pos:pos + self.bs])
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.amp):
                    loss, _ = self.model(x, mask, tokens, attention)
                n = int(attention[:, 1:].sum()); total += float(loss) * n; count += n
            self.model.train()
            require(count > 0 and math.isfinite(total), 'Invalid shuffled-audio diagnostic')
            return {'development_token_nll': total / count, 'mismatched_audio_clips': len(ids),
                    'cyclic_shift': int(shift), 'training_updates': 0,
                    'interpretation': 'development conditioning diagnostic; no required favorable direction'}
        def run(self):
            result = super().run()
            result['shuffled_audio_diagnostic'] = self.shuffled_audio_nll()
            result['evaluation_split'] = 'official_validation_minus_controller; NOT official test'
            result['official_test_outcomes_computed'] = False
            result['final']['development_token_nll'] = result['final'].pop('test_loss')
            p = self.ctx.artifact_dir / 'captions_test.json'
            captions = read(p)
            dest = self.ctx.artifact_dir / 'captions_development.json'
            os.replace(p, dest)
            result['caption_diagnostics'] = {'n': len(captions), 'empty_count': sum(not r['candidate'].strip() for r in captions),
                    'unique_candidate_count': len({r['candidate'] for r in captions}), 'unfavorable_or_collapsed_predictions_are_retained': True}
            return result
    m.CaptioningJob = Pilot
    return m.run




def execute_job(task, method, seed):
    source_setup(); runtime()
    from caenl.utils.jobctx import JobContext
    from caenl.utils.seeding import seed_everything
    config = configure(task, method, seed)
    out = ROOT / task / ('seed' + str(seed)) / method
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'job.json').exists(): require(read(out / 'job.json') == config, 'Job protocol/source/assets changed; refusing mixed results')
    else: put(out / 'job.json', config)
    if (out / 'DONE.json').exists():
        record = read(out / 'DONE.json')
        require(record['binding'] == config['crossmodal_binding'], 'Completed job binding differs')
        for rel, expected in record['files'].items(): require(sha(out / rel) == expected, 'Completed payload changed: ' + rel)
        print('ALREADY COMPLETE:', task, seed, method, flush=True); return
    class Context(JobContext):
        def save_checkpoint(self, name, payload):
            payload = dict(payload); payload['crossmodal_binding'] = config['crossmodal_binding']
            return super().save_checkpoint(name, payload)
        def load_checkpoint(self, name, map_location='cpu'):
            payload = super().load_checkpoint(name, map_location)
            if payload is not None: require(payload.get('crossmodal_binding') == config['crossmodal_binding'], 'Checkpoint belongs to another protocol')
            return payload
        def log_metrics(self, record):
            super().log_metrics(record)
            if record.get('kind') == 'train':
                progress(task.upper() + ' PILOT: seed ' + str(seed) + ', ' + method + ', step ' + str(record.get('step')))
    # Numerical faults must be fatal, not silently converted into result rows.
    original_clip = torch.nn.utils.clip_grad_norm_
    def finite_clip(parameters, max_norm, *args, **kwargs):
        kwargs['error_if_nonfinite'] = True
        return original_clip(parameters, max_norm, *args, **kwargs)
    torch.nn.utils.clip_grad_norm_ = finite_clip
    # Parent task checkpoints already store CPU/CUDA RNG and sampler positions.
    seed_everything(seed, deterministic=True, cudnn_benchmark=False)
    # The retained seeding helper enables warn-only mode; this new study fails
    # rather than silently accepting nondeterministic operations.
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    runner = install_task_classes(task)
    ctx = Context(out, config); ctx.start()
    try:
        progress('Starting ' + task + ' DEVELOPMENT pilot: ' + method + '/seed' + str(seed))
        result = runner(ctx)
        result['role'] = PLAN['role']; result['checkpoint_selection'] = 'final scheduled pilot step, never best test score'
        result['expected_improvement_required'] = False
        result['spectral_not_aligned_classification_variant'] = True
        if task == 'diffusion':
            require(result['steps'] == (0 if method == 'pretrained_reference' else 2000), 'Incomplete diffusion training')
            require(result['final']['n_samples'] == 5000, 'Incomplete generated sample evaluation')
            for k in ('fid', 'kid', 'inception_score', 'val_loss'):
                require(math.isfinite(float(result['final'][k])), 'Invalid diffusion metric ' + k)
        else:
            for k in ('cider_d', 'bleu4', 'rouge_l', 'meteor', 'spice', 'development_token_nll'):
                require(math.isfinite(float(result['final'][k])), 'Invalid caption metric ' + k)
            require(result['metrics_source'] == 'pycocoevalcap', 'Unexpected metric fallback')
        history = read(ctx.artifact_dir / 'controller_trajectory.json')['history']
        learned = sum('policy_loss' in r.get('update', {}) for r in history)
        if method == 'full_macc': require(learned >= 20, 'Too few real learned updates; pilot is not validated')
        result['learned_policy_updates'] = learned
        result['controller_event_count'] = len(history)
        ctx.finish(result)
        payload = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*'))
                   if p.is_file() and p.name not in ('DONE.json', 'status.json', 'system.csv') and '.tmp' not in p.name}
        put(out / 'DONE.json', {'status': 'COMPLETE', 'binding': config['crossmodal_binding'], 'files': payload, 'time': utc()})
    except BaseException as exc:
        ctx.fail(exc); raise




def environment_record():
    names = ('torch', 'torchvision', 'diffusers', 'transformers', 'huggingface-hub', 'torch-fidelity', 'numpy', 'scipy', 'soundfile', 'pycocoevalcap')
    out = {'python': sys.version, 'packages': {}}
    for name in names:
        try: out['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: out['packages'][name] = None
    return out




def make_report(status):
    lines = ['# CAENL cross-modal development pilots v1', '', 'Status: ' + status, 'UTC: ' + utc(), '',
       'These are development pilots, not the final replicated audio/diffusion results.',
       'Completed ImageNet, C4 and UHerding studies are preserved. No favorable outcome is required.',
       'Official CIFAR10 test outcomes and AudioCaps test-caption metrics are not used in pilot decisions.',
       'The pretrained models may have seen underlying data during pretraining; local holdouts are not claimed pristine relative to that pretraining.', '']
    names = ['PLAN.json', 'SOURCE.json', 'ENVIRONMENT.json', 'SELF_TEST.json', 'AUDIO_METRIC_TEST.json',
             'AUDIO_READY.json', 'AUDIO_BLOCKED.json', 'AUDIO_ERROR.json', 'FAILED.json', 'CHILD_FAILURE.json', 'PRESERVATION.json',
             'DIFFUSION_COMPLETE.json', 'AUDIO_COMPLETE.json', 'MODELS.json', 'audio_corpus/availability.json']
    for name in names:
        p = ROOT / name
        if p.is_file(): lines += ['## ' + name, 'SHA-256: ' + sha(p), '```json', json.dumps(clean(read(p)), indent=2), '```', '']
    for p in sorted(ROOT.glob('*/seed*/*/summary.json')):
        summary = read(p)
        lines += ['## RESULT ' + p.relative_to(ROOT).as_posix(), 'SHA-256: ' + sha(p), '```json', json.dumps(clean(summary), indent=2), '```', '']
    for p in sorted(ROOT.glob('audio/seed*/*/artifacts/captions_development.json')):
        lines += ['## DEVELOPMENT CAPTIONS ' + p.relative_to(ROOT).as_posix(), '```json', json.dumps(read(p), ensure_ascii=False, indent=2), '```', '']
    lines += ['## Exact new orchestration source', '````````', (ROOT / 'runner.py').read_text(encoding='utf-8-sig'), '````````', '', 'END OF CROSS-MODAL PILOT REPORT']
    target = ROOT / REPORT
    write(target, '\n'.join(lines) + '\n'); write(ROOT / (REPORT + '.sha256'), sha(target) + '  ' + REPORT + '\n')
    return target




def child(*args, timeout=None):
    try:
        subprocess.run([sys.executable, str(ROOT / 'runner.py'), *map(str, args)], check=True, timeout=timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        log = ROOT / 'run.log'
        tail = ''
        if log.is_file():
            with log.open('rb') as f:
                f.seek(max(0, log.stat().st_size - 24000))
                tail = f.read().decode('utf-8', errors='replace')
        put(ROOT / 'CHILD_FAILURE.json', {'time': utc(), 'arguments': list(map(str, args)),
              'error': repr(exc), 'log_tail': tail})
        raise




def worker():
    import fcntl
    with (ROOT / 'RUN.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            source_setup(create=True)
            env = environment_record()
            if (ROOT / 'ENVIRONMENT.json').exists(): require(read(ROOT / 'ENVIRONMENT.json') == env, 'Installed environment changed since pilot start')
            else: put(ROOT / 'ENVIRONMENT.json', env)
            bound = {'plan': PLAN, 'runner_sha256': sha(ROOT / 'runner.py')}
            if (ROOT / 'PLAN.json').exists(): require(read(ROOT / 'PLAN.json') == bound, 'Pilot plan changed')
            else: put(ROOT / 'PLAN.json', bound)
            require(env['packages']['torch-fidelity'] is not None, 'Existing torch-fidelity package missing. No installation attempted.')
            progress('Running real CPU/GPU implementation checks before any task training')
            child('--checks', timeout=600)
            prepare_diffusion()
            for seed in SEEDS:
                for method in ('pretrained_reference',) + METHODS:
                    child('--job', 'diffusion', '--method', method, '--seed', seed)
                    make_report('DIFFUSION DEVELOPMENT RUNS IN PROGRESS; AUDIO NOT YET COMPLETE')
            put(ROOT / 'DIFFUSION_COMPLETE.json', {'status': 'PILOT_COMPLETE', 'training_runs': 8, 'reference_evaluations': 2, 'seeds': list(SEEDS), 'final_confirmation': False})
            progress('DIFFUSION PILOT COMPLETE. Preparing AudioCaps without changing the previous results.')
            make_report('DIFFUSION PILOT COMPLETE; AUDIO PREPARATION NEXT')
            try:
                prepare_audio()
                if (ROOT / 'AUDIO_READY.json').exists():
                    progress('Checking Java caption metrics before feature extraction or audio training')
                    child('--metric-test', timeout=300)
                    resolve_model(PLAN['audio']['backbone']); resolve_model('gpt2')
                    for seed in SEEDS:
                        for method in METHODS:
                            child('--job', 'audio', '--method', method, '--seed', seed)
                            make_report('DIFFUSION COMPLETE; AUDIO DEVELOPMENT RUNS IN PROGRESS')
                    put(ROOT / 'AUDIO_COMPLETE.json', {'status': 'PILOT_COMPLETE', 'training_runs': 8, 'seeds': list(SEEDS), 'final_confirmation': False})
            except Exception as exc:
                put(ROOT / 'AUDIO_ERROR.json', {'status': 'STOPPED_AUDIO', 'error': repr(exc), 'traceback': traceback.format_exc(),
                     'diffusion_results_preserved': True, 'no_fallback_metrics_or_dataset': True})
                traceback.print_exc()
            original = PROJECT / 'src/caenl'
            for rel, expected in read(ROOT / 'SOURCE.json')['files'].items():
                require(sha(original / rel) == expected, 'Original source changed externally during pilot: ' + rel)
            put(ROOT / 'PRESERVATION.json', {'status': 'PASS', 'original_source_hashes_rechecked': True,
                'old_results_not_opened_for_writing': True, 'old_checkpoints_not_deleted': True,
                'scope': 'source hash recheck and output-path isolation; not a re-audit of every previous experiment'})
            complete = (ROOT / 'AUDIO_COMPLETE.json').exists()
            label = 'CROSS-MODAL PILOTS COMPLETE' if complete else 'DIFFUSION PILOT COMPLETE; AUDIO BLOCKED OR STOPPED'
            target = make_report(label)
            put(ROOT / 'FINISHED.json', {'status': label, 'time': utc(), 'diffusion_training_runs': 8,
                  'audio_training_runs_completed': len(list(ROOT.glob('audio/seed*/*/DONE.json'))),
                  'report': str(target), 'report_sha256': sha(target), 'final_scientific_confirmation': False})
            put(ROOT / 'STATUS.json', {'status': label, 'time': utc(), 'message': 'Download the pilot report for protocol and runtime review; no final multi-seed study has been launched.'})
            print(label + '\nShare only: ' + str(target), flush=True)
            return 0 if complete else 2
        except BaseException as exc:
            put(ROOT / 'FAILED.json', {'status': 'FAILED', 'time': utc(), 'error': repr(exc), 'traceback': traceback.format_exc(), 'original_results_changed': False})
            try: make_report('STOPPED; NOT COMPLETE')
            except Exception: pass
            traceback.print_exc(); return 1




def status():
    print('SERVER:', os.uname().nodename, utc())
    alive = bool(shutil.which('tmux')) and subprocess.run(['tmux', 'has-session', '-t', SESSION], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    print('TMUX SESSION ACTIVE' if alive else 'NO LIVE TMUX SESSION; records below may show the last saved state')
    for name in ('FAILED.json', 'STATUS.json'):
        if (ROOT / name).exists():
            print(json.dumps(read(ROOT / name), indent=2)); break
    else: print('Not started.')
    print('Diffusion pilot training jobs (target 8):', len([p for p in ROOT.glob('diffusion/seed*/*/DONE.json') if p.parent.name != 'pretrained_reference']))
    print('Pretrained reference evaluations (target 2):', len(list(ROOT.glob('diffusion/seed*/pretrained_reference/DONE.json'))))
    print('Audio pilot training jobs (target 8, subject to verified data):', len(list(ROOT.glob('audio/seed*/*/DONE.json'))))
    for name in ('AUDIO_BLOCKED.json', 'AUDIO_ERROR.json'):
        if (ROOT / name).exists(): print(json.dumps(read(ROOT / name), indent=2))
    print('Report:', ROOT / REPORT)
    print('Current log:', ROOT / 'run.log')
    if shutil.which('nvidia-smi'):
        subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', '--format=csv'], check=False)




def start(resume=False):
    require(sys.platform.startswith('linux'), 'Start on IBM through the supplied Mac command.')
    require(Path('/mnt/caenl').is_mount(), 'Persistent volume not mounted; no mount/format attempt made.')
    require(shutil.which('tmux'), 'tmux is not available. Nothing installed.')
    require(not ROOT.is_symlink(), 'Refusing symlink output root')
    ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    import fcntl
    with (ROOT / 'START.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if subprocess.run(['tmux', 'has-session', '-t', SESSION], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            print('ALREADY RUNNING. No second copy started.'); status(); return
        if (ROOT / 'FINISHED.json').exists() and not resume:
            status(); return
        require(resume or not (ROOT / 'FAILED.json').exists(), 'Previous attempt stopped; share --status before resuming.')
        require(shutil.disk_usage(ROOT).free >= 80 * 1024 ** 3, 'At least 80 GiB free required. No files deleted.')
        pythons = [PROJECT / '.venv/bin/python', Path.home() / 'caenl-src/caenl_revision_pipeline_v5.3_ibm_rhel_l40s/.venv/bin/python']
        python = next((p for p in pythons if p.is_file()), None)
        require(python is not None, 'Existing CAENL environment missing; no installation attempted.')
        query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        require(not query.stdout.strip(), 'Another GPU process is active. It has not been stopped.')
        raw = Path(__file__).read_bytes(); ast.parse(raw.decode('utf-8-sig'))
        target = ROOT / 'runner.py'
        if target.exists(): require(target.read_bytes() == raw, 'A different launcher already owns this result folder.')
        else: target.write_bytes(raw)
        if resume:
            hist = ROOT / ('resume_history_' + str(time.time_ns())); hist.mkdir()
            for name in ('FAILED.json', 'FINISHED.json', 'AUDIO_BLOCKED.json', 'AUDIO_ERROR.json', 'CHILD_FAILURE.json'):
                if (ROOT / name).exists(): os.replace(ROOT / name, hist / name)
        # Reuse only the user's existing Java/media executable paths, never export credentials.
        shell = 'set -e; if [ -f "$HOME/.caenl-java8-env" ]; then source "$HOME/.caenl-java8-env"; fi; '
        shell += 'if [ -f "$HOME/.caenl-media-env" ]; then source "$HOME/.caenl-media-env"; fi; '
        shell += 'export PATH=' + shlex.quote(str(python.parent)) + ':"$PATH"; '
        shell += 'export HF_HOME=' + shlex.quote(str(ROOT / 'assets/hf_home')) + '; '
        shell += 'export TORCH_HOME=' + shlex.quote(str(ROOT / 'assets/torch')) + '; '
        shell += 'export XDG_CACHE_HOME=' + shlex.quote(str(ROOT / 'cache/xdg')) + '; '
        shell += 'exec env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 '
        shell += shlex.quote(str(python)) + ' ' + shlex.quote(str(target)) + ' --worker >> ' + shlex.quote(str(ROOT / 'run.log')) + ' 2>&1'
        subprocess.run(['tmux', 'new-session', '-d', '-s', SESSION, 'bash -c ' + shlex.quote(shell)], check=True)
        print('STARTED:', SESSION)
        print('Development pilots: diffusion first, then verified AudioCaps. Not final confirmation.')
        print('Only NEW pilot jobs are started. Original campaigns are not rerun; dependencies are not installed.')
        print('Status: python3 ~/caenl_crossmodal_pilots_v1.py --status')




def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--start', action='store_true'); g.add_argument('--resume', action='store_true')
    g.add_argument('--status', action='store_true'); g.add_argument('--worker', action='store_true')
    g.add_argument('--checks', action='store_true'); g.add_argument('--metric-test', action='store_true')
    g.add_argument('--job', choices=('diffusion', 'audio'))
    p.add_argument('--method', choices=('pretrained_reference',) + METHODS)
    p.add_argument('--seed', type=int, choices=SEEDS)
    args = p.parse_args()
    if args.status: status()
    elif args.checks: checks()
    elif args.metric_test: metric_test()
    elif args.job:
        require(args.method is not None and args.seed is not None, 'Internal job arguments are incomplete')
        execute_job(args.job, args.method, args.seed)
    elif args.worker: return worker()
    else: start(args.resume)
    return 0




if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception as error:
        print('STOPPED:', error, file=sys.stderr)
        raise SystemExit(1)
# END OF CAENL CROSSMODAL PILOTS V1