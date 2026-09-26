#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CAENL AudioCaps six-seed confirmation v1, 22 September 2026.


NEW experiment: four methods, seeds 1301--1306, ten scheduled epochs.
Reuses verified pretrained assets and READ-ONLY frozen pilot audio features.
Final caption evaluation uses the 866 retained official-test recordings.
The two pilot seeds and their development results are not pooled into inference.
No dataset download, installation, feature extraction, prior-run extension,
recording modification, or deletion. Every outcome, including harm, is retained.


This new orchestration has not been executed in the authoring environment:
container and Python execution returned tool errors. Its numerical, live model,
metric, source, split and artifact checks execute on IBM before final training.
It uses the hash-bound source and working model/metric path of the completed
pilot. Technical success is not a guarantee of a favorable scientific result.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
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
NAME = 'caenl-audiocaps-confirm-v1'
ROOT = Path('/mnt/caenl/active/results') / NAME
PILOT = Path('/mnt/caenl/active/results/caenl-audiocaps-train-pilot-v1')
DATA = Path('/mnt/caenl/persistent/datasets/audiocaps_opensound_v1/disjoint_v1')
PYTHON = Path.home() / 'caenl-v5.4.3/caenl_revision_pipeline_v5.4.3_c4_confirmatory/.venv/bin/python'
REPORT = NAME + '.md'
SEEDS = (1301, 1302, 1303, 1304, 1305, 1306)
METHODS = ('baseline', 'fixed_dcr', 'macc_lite', 'full_macc')
STEPS = 13890
PILOT_PINS = {
 'caenl-audiocaps-train-pilot-v1.md': '3127b78b4a4f130c1c145565a03c04ab0d187df531f43b95b715efeca5347f0e',
 'runner.py': '20acc1c0c585041ff5356d22c021d29e270d56f8f82d020d1c1b83714c3af879',
 'engine.py': 'd3d01297c43357b9c34d81701b21f1b80c3a3a395bb65ef341026c61c918ac15',
 'SOURCE.json': 'e8a7a3039390e60491431376ec1d28aad07e674fbaf64ca89aaa18e52f6b9451',
 'WRAPPER_PLAN.json': 'e493f681c1379386dca82887b12b5d419294d8ed503cc33bd443951e7f2d1044',
 'PER_SEED.json': '4f575d0beed95a175f55398eee0d0dc30abb217fb9321266a5d06926768345d3',
 'PROTOCOL.json': '92f9adada5b3a35591e0b31730076adac2451e780614b97ad1c4d0f9af44c868',
 'DATA_BINDING.json': 'dc65b35ddd71b1b65593448b663e130ee2cf6e875193f7b6be783bb2ef977c78',
 'AUDIO_READY.json': 'c01dcea0e4e3176da301176b7f34bbf1c0da533a74878133a8d3c088a725ac99',
 'MODELS_FROZEN.json': 'd8e8383e9921a4eb1c11515ccad86c25f0e7c37e39f41cce748f8f8c3578f2ef',
 'FEATURES_READY.json': 'f757c0e35c86235b8d0878aa7a5c003e9dc10816984fcbbacdf45ad88129aba3',
 'METRIC_ASSETS.json': '0c9b7fd53ca26eb246aaffb2228f598bca0a205f842a255659ee274aa2ef7ba7',
 'PREPROCESSED_INPUT_AUDIT.json': 'b0761e485b215e926bda82c7836a93e93c34585a7c05c7f356adbe06114b7d88',
 'ENVIRONMENT.json': '6590e2b9f565c59ea618ef657254edc6d7274929245e9fcf8ad56c5eb46f32aa',
}
PRIMARY = [('fixed_dcr', 'baseline'), ('macc_lite', 'baseline'), ('full_macc', 'baseline')]
METRICS = ('cider_d', 'bleu1', 'bleu2', 'bleu3', 'bleu4', 'rouge_l', 'meteor', 'spice', 'spider')
PROTOCOL = {
 'name': NAME, 'evidence_role': 'independent_six_training_seed_confirmation_conditional_on_fixed_pretrained_models_and_dataset',
 'seeds': list(SEEDS), 'methods': list(METHODS), 'epochs': 10, 'steps_per_epoch': 1389,
 'steps_per_job': STEPS, 'training_jobs': 24, 'batch_size': 32,
 'epoch_definition': 'new deterministic permutation of 44466 training pairs each epoch; drop final 18 pairs, unchanged from pilot',
 'initialization': 'original pretrained GPT-2, frozen AST features and new per-seed random audio encoder/prefix; no pilot-final weights',
 'within_seed_matching': 'same initial task weights, epoch permutations, per-step dropout seed and scheduled optimizer budget across methods',
 'dataset_counts': {'train': 44466, 'valid': 440, 'test': 866},
 'dataset': 'original AudioCaps available subset from pinned OpenSound mirror, with existing conflict/duration/split exclusions',
 'controller_validation_count': 64, 'development_validation_count': 376,
 'controller_and_development_split': 'unchanged pilot manifest-order partition of official validation',
 'final_evaluation': 'all 866 retained official-test recordings with five original captions each; no test feedback or checkpoint selection',
 'primary_endpoint': 'CIDEr-D from retained pycocoevalcap/PTB implementation; higher is better',
 'primary_contrasts': [{'target': a, 'reference': b} for a, b in PRIMARY],
 'statistics': 'paired-t and all 64 exact sign flips; separate Holm correction across THREE primary contrasts; individual unadjusted 95% paired-t intervals',
 'secondary_endpoints': list(METRICS[1:]) + ['test token NLL', 'development token NLL', 'development mismatched-audio NLL', 'controller diagnostics', 'recorded costs'],
 'secondary_inference': 'descriptive only; do not replace the primary endpoint with a more favorable secondary metric',
 'unchanged_pilot_components': ['spectral objective and targets', 'coefficient/control settings', 'AdamW rates and decay', '100-step warmup', 'batch size', 'feature representation', 'beam decoding', 'Java metric implementation'],
 'changed_after_pilot_before_confirmation': ['six new seeds', '10 rather than 2 epochs and corresponding cosine horizon', 'final official-test caption evaluation'],
 'checkpoint_selection': 'final scheduled step, regardless of validation trend or result; no early stopping or best-score selection',
 'no_new_feature_extraction': True, 'no_dataset_or_model_downloads': True,
 'pretraining_overlap': 'AST AudioSet pretraining may include underlying AudioCaps recordings; no verified absence of pretraining overlap',
 'source_boundary': 'third-party available-recording subset, not the full original AudioCaps dataset or an official acoustic-authenticity certificate',
 'data_rights': 'mirror declares CC BY-NC 4.0; uploader rights not independently established; no raw-audio redistribution',
 'negative_results_retained': True, 'authoring_runtime_executed': False,
}
W = None
STATE = {}
STATE_LOCK = threading.Lock()




def require(ok, message):
    if not bool(ok): raise RuntimeError(message)
def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''): h.update(b)
    return h.hexdigest()
def encoded(x): return (json.dumps(x, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
def digest(x): return hashlib.sha256(encoded(x)).hexdigest()
def write(p, value):
    p = Path(p)
    require(p.resolve().is_relative_to(ROOT.resolve()) and not p.is_symlink(), 'Unsafe output path: ' + str(p))
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str): value = value.encode('utf-8')
    with tempfile.NamedTemporaryFile(dir=p.parent, prefix='.writing_', delete=False) as f:
        temp = Path(f.name); f.write(value); f.flush(); os.fsync(f.fileno())
    os.replace(temp, p)
def put(p, value): write(p, encoded(value))
def freeze(p, value):
    if Path(p).exists(): require(read(p) == value, 'Previously frozen value changed: ' + str(p))
    else: put(p, value)
def progress(message, **kw):
    with STATE_LOCK:
        STATE.clear(); STATE.update(status='RUNNING', pid=os.getpid(), time=now(), message=message, **kw)
        put(ROOT / 'STATUS.json', STATE)
    print(message, flush=True)
def heartbeat(stop):
    while not stop.wait(30):
        with STATE_LOCK:
            if STATE: put(ROOT / 'STATUS.json', dict(STATE, heartbeat_utc=now()))
def safe_child(root, rel):
    p = Path(rel)
    require(not p.is_absolute() and '..' not in p.parts, 'Unsafe relative input: ' + str(p))
    out = Path(root) / p
    require(out.is_file() and not out.is_symlink() and out.resolve().is_relative_to(Path(root).resolve()), 'Unsafe/missing bound input: ' + str(out))
    return out


def copy_checked(src, dst, expected):
    src, dst = Path(src), Path(dst)
    require(src.is_file() and not src.is_symlink() and sha(src) == expected, 'Input checksum mismatch: ' + str(src))
    require(dst.resolve().is_relative_to(ROOT.resolve()) and not dst.is_symlink(), 'Unsafe copy destination')
    if dst.exists(): require(sha(dst) == expected, 'Existing copied file differs: ' + str(dst)); return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + '.copying')
    shutil.copyfile(src, tmp); require(sha(tmp) == expected, 'Copy verification failed')
    os.replace(tmp, dst)


def wrapper():
    global W
    if W is None:
        path = ROOT / 'pilot_wrapper.py'
        require(sha(path) == PILOT_PINS['runner.py'], 'Stored pilot wrapper changed')
        spec = importlib.util.spec_from_file_location('caenl_audio_confirmation_helpers', path)
        W = importlib.util.module_from_spec(spec); sys.modules[spec.name] = W; spec.loader.exec_module(W)
        W.ROOT = ROOT; W.SESSION = NAME; W.SEEDS = SEEDS; W.REPORT = REPORT
        # Retain the original detailed settings and explicitly update the changed fields.
        cfg = copy.deepcopy(W.PROTOCOL)
        cfg.update(evidence_role=PROTOCOL['evidence_role'], seeds=list(SEEDS), epochs=10,
                   steps_per_job=STEPS, official_test_caption_outcomes=True,
                   checkpoint_selection=PROTOCOL['checkpoint_selection'])
        cfg.pop('future_confirmation', None)
        cfg['validation']['final_evaluation'] = PROTOCOL['final_evaluation']
        cfg['confirmation'] = PROTOCOL
        W.PROTOCOL = cfg; W.progress = progress
    return W


def engine():
    w = wrapper(); e = w.get_engine()
    e.PLAN['audio']['epochs'] = 10
    return e


def verify_pilot(artifacts=True):
    pins = {}
    for rel, expected in PILOT_PINS.items():
        p = safe_child(PILOT, rel); require(sha(p) == expected, 'Reviewed pilot file differs: ' + rel)
        pins[str(p)] = expected
    done = read(PILOT / 'COMPLETE.json')
    require(done['status'] == 'COMPLETE' and done['completed_method_runs'] == 8 and done['seeds'] == [1201,1202], 'Unexpected pilot completion')
    require(done['report_sha256'] == PILOT_PINS['caenl-audiocaps-train-pilot-v1.md'], 'Pilot report binding differs')
    require(done['official_test_caption_outcomes_computed'] is False, 'Pilot test-evaluation boundary changed')
    if artifacts:
        progress('Verifying eight completed pilot artifact sets; no pilot is rerun')
        expected = {(s,m) for s in (1201,1202) for m in METHODS}
        per_seed = read(PILOT / 'PER_SEED.json')
        require({(r['seed'],r['method']) for r in per_seed} == expected and len(per_seed) == 8, 'Pilot seed table incomplete')
        for entry in per_seed:
            folder = PILOT / 'audio' / ('seed' + str(entry['seed'])) / entry['method']
            record = read(folder / 'DONE.json')
            for rel, h in record['files'].items():
                path = safe_child(folder,rel); require(sha(path) == h, 'Pilot artifact differs: ' + str(path)); pins[str(path)] = h
            summary = read(folder / 'summary.json')
            require(summary['steps'] == 2778 and summary['official_test_caption_outcomes_computed'] is False, 'Pilot result scope differs')
            require(summary['final'] == entry['development_metrics'], 'Pilot summary/table mismatch')
            caps = read(folder / 'artifacts/captions_development.json')
            require(len(caps) == 376 and len({r['id'] for r in caps}) == 376, 'Incomplete pilot captions')
            require(summary['caption_diagnostics']['empty_count'] == sum(not r['candidate'].strip() for r in caps), 'Pilot caption diagnostic differs')
    return pins


def verify_features():
    import numpy as np
    require(sha(PILOT / 'FEATURES_READY.json') == PILOT_PINS['FEATURES_READY.json'], 'Pilot feature binding changed')
    record = read(PILOT / 'FEATURES_READY.json')
    require(record['status'] == 'READY' and record['n_clips'] == 45772, 'Incomplete retained features')
    for rel,h in record['files'].items():
        require(sha(safe_child(PILOT / 'features',rel)) == h, 'Retained feature payload changed: '+rel)
    require(sha(PILOT / 'PREPROCESSED_INPUT_AUDIT.json') == record['preprocessing_audit_sha256'], 'Preprocessing audit changed')
    require(not read(PILOT / 'PREPROCESSED_INPUT_AUDIT.json')['nonzero_exact_cross_split_groups'], 'Preprocessed split leakage')
    feats = np.load(PILOT / 'features/features.npy', mmap_mode='r', allow_pickle=False)
    nt = np.load(PILOT / 'features/n_tokens.npy', allow_pickle=False)
    require(feats.shape == (45772,64,768) and feats.dtype == np.float16 and nt.shape == (45772,) and (nt==64).all(), 'Frozen feature shape/count mismatch')
    return record


def prepare():
    progress('Binding the completed audio pilot; new confirmation will not reuse its trained checkpoints')
    pins = verify_pilot(artifacts=True)
    copy_checked(PILOT / 'runner.py', ROOT / 'pilot_wrapper.py', PILOT_PINS['runner.py'])
    copy_checked(PILOT / 'engine.py', ROOT / 'engine.py', PILOT_PINS['engine.py'])
    copy_checked(PILOT / 'SOURCE.json', ROOT / 'SOURCE.json', PILOT_PINS['SOURCE.json'])
    for rel,h in read(PILOT / 'SOURCE.json')['files'].items():
        src = safe_child(PILOT / 'source/caenl',rel)
        copy_checked(src, ROOT / 'source/caenl' / rel,h); pins[str(src)] = h
    w = wrapper(); rows = w.verify_data()
    require(sha(DATA / 'manifest.jsonl') == '6a19d868feb14ab49ea2a0fba29e2ffa2e6c4b26f0335831db13a8ab7da915b1', 'Finalized data manifest differs')
    models = read(PILOT / 'MODELS_FROZEN.json'); copied = copy.deepcopy(models)
    for repo,rec in models.items():
        origin = Path(rec['path'])
        require(origin.resolve().is_relative_to(PILOT.resolve()), 'Model path is outside pilot assets')
        dest = ROOT / 'assets' / ('ast' if repo == w.AST_NAME else 'gpt2')
        for rel,h in rec['files'].items():
            src = safe_child(origin,rel); copy_checked(src,dest/rel,h); pins[str(src)]=h
        copied[repo]['path']=str(dest)
    freeze(ROOT / 'MODELS.json',copied); freeze(ROOT / 'MODELS_FROZEN.json',copied)
    progress('Reusing local caption-metric assets and frozen AST features; no downloads')
    metric_files={}
    for src in sorted((PILOT / 'metric_vendor').rglob('*')):
        rel=src.relative_to(PILOT / 'metric_vendor')
        if any(v in ('cache','tmp','__pycache__','.git') for v in rel.parts) or not src.is_file(): continue
        require(not src.is_symlink(), 'Metric asset symlink refused')
        h=sha(src);copy_checked(src,ROOT/'metric_vendor'/rel,h);pins[str(src)]=h;metric_files[rel.as_posix()]=h
    for rel,h in read(PILOT/'METRIC_ASSETS.json')['files'].items():
        require(metric_files.get(rel)==h, 'Bound metric code/jar differs: '+rel)
    freeze(ROOT/'METRIC_COPY.json',{'files':metric_files,'source':str(PILOT/'metric_vendor'),'no_packages_installed':True})
    rec=verify_features()
    for rel,h in rec['files'].items(): pins[str(PILOT/'features'/rel)] = h
    freeze(ROOT/'FEATURES_REUSE.json',{'parent':str(PILOT),'parent_ready_sha256':PILOT_PINS['FEATURES_READY.json'],'record':rec,
         'read_only':True,'new_extraction':False,'row_order':'unchanged finalized manifest'})
    e=engine(); env=e.environment_record()
    require(env==read(PILOT/'ENVIRONMENT.json'), 'Installed environment differs from the completed pilot; no package changes attempted')
    freeze(ROOT/'ENVIRONMENT.json',env)
    freeze(ROOT/'PRESERVED_INPUTS.json',pins)
    # This is committed before any new optimizer update, not selected using test scores.
    study={'protocol':w.PROTOCOL,'runner_sha256':sha(ROOT/'runner.py'),'helper_sha256':PILOT_PINS['runner.py'],
        'engine_sha256':PILOT_PINS['engine.py'],'source_manifest_sha256':PILOT_PINS['SOURCE.json'],
        'data':read(ROOT/'DATA_BINDING.json'),'models':copied,'features':read(ROOT/'FEATURES_REUSE.json'),
        'metric_copy_sha256':sha(ROOT/'METRIC_COPY.json'),'preserved_inputs_sha256':sha(ROOT/'PRESERVED_INPUTS.json')}
    study['binding']=digest(study);freeze(ROOT/'PROTOCOL.json',study)
    progress('CONFIRMATION PROTOCOL FROZEN: 24 new ten-epoch runs; final test evaluation reserved for final checkpoints')




def configure(method,seed):
    require(method in METHODS and seed in SEEDS, 'Unexpected method or confirmation seed')
    e=engine();cfg=e.configure('audio',method,seed);cfg.pop('crossmodal_binding',None)
    cfg['audio']['manifest']=str(DATA/'manifest.jsonl')
    cfg['audio']['captioning']['epochs']=10
    cfg['data_binding']=read(ROOT/'DATA_BINDING.json')
    cfg['features_reuse']=read(ROOT/'FEATURES_REUSE.json')
    cfg['confirmation_binding']=read(ROOT/'PROTOCOL.json')['binding']
    cfg['metric_copy_sha256']=sha(ROOT/'METRIC_COPY.json')
    cfg['crossmodal_binding']=digest(cfg)
    return cfg




def pooled_nll(batches):
    require(batches and all(r['target_tokens']>0 and math.isfinite(r['mean_token_nll']) for r in batches), 'Invalid token-loss records')
    count=sum(r['target_tokens'] for r in batches)
    total=math.fsum(r['mean_token_nll']*r['target_tokens'] for r in batches)
    return total/count,count,total


def holm(values):
    require(values and all(math.isfinite(v) and 0<=v<=1 for v in values), 'Invalid p-values')
    result=[0.]*len(values);previous=0.
    for rank,index in enumerate(sorted(range(len(values)),key=lambda i:values[i])):
        previous=max(previous,min(1.,(len(values)-rank)*values[index]));result[index]=previous
    return result


def paired(a,b):
    import numpy as np
    from scipy.stats import t as student, ttest_rel
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    require(a.shape==b.shape==(6,) and np.isfinite(a).all() and np.isfinite(b).all(), 'Exactly six finite paired results required')
    d=a-b;mean=float(d.mean());sd=float(d.std(ddof=1));se=sd/math.sqrt(6)
    if sd==0: p=1. if mean==0 else 0.;interval=[mean,mean]
    else:
        res=ttest_rel(a,b);p=float(res.pvalue);radius=float(student.ppf(.975,5))*se;interval=[mean-radius,mean+radius]
    signs=np.asarray(list(itertools.product((-1.,1.),repeat=6)))
    perm=np.abs((signs*d).mean(axis=1));threshold=abs(mean)
    tol=16*np.finfo(np.float64).eps*max(threshold,float(np.max(np.abs(d))),np.finfo(np.float64).tiny)
    exact=float(np.count_nonzero(perm>=threshold-tol)/64)
    return {'mean_difference':mean,'difference_sd':sd,'individual_95pct_paired_t_interval':interval,
            'paired_t_p_raw':p,'exact_sign_flip_p_raw':exact,'differences':d.tolist(),'n_paired_seeds':6}




def numerical_tests():
    import numpy as np
    from scipy.stats import ttest_rel
    require(np.allclose(holm([.01,.04,.03]),[.03,.06,.06],rtol=0,atol=1e-15),'Holm fixture failed')
    require(paired([1]*6,[0]*6)['exact_sign_flip_p_raw']==2/64,'Exact resolution fixture failed')
    require(paired([0]*6,[0]*6)['exact_sign_flip_p_raw']==1,'Zero-difference fixture failed')
    a=np.asarray([.1,.3,.2,.4,.31,.5]);b=np.asarray([.09,.11,.19,.39,.21,.3]);r=paired(a,b)
    ref=ttest_rel(a,b)
    require(abs(r['paired_t_p_raw']-float(ref.pvalue))<1e-14,'Paired-t check failed')
    ci=ref.confidence_interval(.95)
    require(np.allclose(r['individual_95pct_paired_t_interval'],[ci.low,ci.high]),'Interval fixture failed')
    p,n,total=pooled_nll([{'mean_token_nll':1.,'target_tokens':2},{'mean_token_nll':3.,'target_tokens':6}])
    require(p==2.5 and n==8 and total==20,'Token-weighting fixture failed')
    for method in METHODS:
        cfg=configure(method,SEEDS[0])
        require(cfg['audio']['captioning']['epochs']==10 and cfg['seed']==1301,'Final training configuration not applied')
        require(cfg['audio']['manifest']==str(DATA/'manifest.jsonl'),'Finalized manifest not selected')
    result={'status':'PASS','checks':['Holm reference fixture','64 sign flips and two-sided resolution','zero differences',
       'paired-t and interval against SciPy','target-token rather than clip weighting','all four final configs'],
       'GPU_scientific_runs_counted':0,'execution_host':os.uname().nodename}
    put(ROOT/'NUMERICAL_TESTS.json',result);print('CONFIRMATION NUMERICAL/CONFIGURATION CHECKS: PASS',flush=True)




def model_class():
    import numpy as np
    import torch
    w=wrapper();e=engine();w.activate_metric_vendor();e.install_window_loop()
    import caenl.audio.captioning as mod
    from caenl.utils.seeding import derive_seed
    # Frozen features are accessed read-only; the old extractor is never called.
    def cache(manifest,cache_root,backbone,tokens_per_window=64,max_windows=1,device=None,hf_cache=None):
        require(Path(manifest).resolve()==(DATA/'manifest.jsonl').resolve(),'Unexpected data manifest')
        require(tokens_per_window==64 and max_windows==1 and backbone==read(ROOT/'MODELS.json')[w.AST_NAME]['path'],'Unexpected feature request')
        rows=[json.loads(line) for line in (DATA/'manifest.jsonl').read_text().splitlines() if line.strip()]
        return {'features':np.load(PILOT/'features/features.npy',mmap_mode='r',allow_pickle=False),
                'rows':rows,'cache_dir':str(PILOT/'features'),'dim':768,'n_clips':len(rows),
                'backbone':backbone,'tokens_per_window':64,'max_windows':1}
    mod.build_feature_cache=cache
    base=mod.CaptioningJob
    class FinalAudio(base):
        def setup(self):
            super().setup();self.model.float()
            require(len(self.pairs)==44466 and self.total_steps==STEPS,'Final optimizer budget differs')
            require(len(self.ctrl_val)==64 and len(self.report_val)==376 and len(self.splits['test'])==866,'Split count mismatch')
            sets=[set(map(int,x)) for x in (self.splits['train'],self.ctrl_val,self.report_val,self.splits['test'])]
            require(all(not sets[i].intersection(sets[j]) for i in range(4) for j in range(i+1,4)),'Gradient/feedback/development/test overlap')
            require(all(self.rows[int(i)]['split']=='test' for i in self.splits['test']),'Final test routing is incorrect')
            require(all(self.rows[int(i)]['split']=='valid' for i in np.r_[self.ctrl_val,self.report_val]),'Validation routing incorrect')
            put(self.ctx.artifact_dir/'EVALUATION_SPLITS.json',{
                'controller_ids':[self.rows[int(i)]['id'] for i in self.ctrl_val],
                'development_ids':[self.rows[int(i)]['id'] for i in self.report_val],
                'official_test_ids':[self.rows[int(i)]['id'] for i in self.splits['test']],
                'test_used_for_updates_or_checkpoint_selection':False})
            h=e.tensor_hash(self.model)
            freeze(ROOT/('INITIALIZATION_seed'+str(self.ctx.seed)+'.json'),{'seed':self.ctx.seed,'model_weights_sha256':h,
                'source':'original pretrained GPT-2 plus new seed-specific encoder/prefix; not pilot-final weights'})
            self.initial_hash=h
        def batch(self,pairs):
            if torch.is_grad_enabled() and self.model.training:
                seed=derive_seed(self.ctx.seed,'audio_step_dropout',self.step)
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            return super().batch(pairs)
        def restore(self):
            result=super().restore();self.step_at_attempt_start=self.step
            require(0<=self.step<=STEPS and self.loop.step==self.step,'Restored training/controller steps disagree')
            return result
        @torch.no_grad()
        def val_loss(self,clips):
            previous=self.model.training;self.model.eval();records=[]
            pairs=[(int(i),c) for i in clips for c in self.rows[int(i)]['captions']]
            try:
                for pos in range(0,len(pairs),self.bs):
                    x,mask,ids,am=self.batch(pairs[pos:pos+self.bs])
                    with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=self.amp):loss,_=self.model(x,mask,ids,am)
                    records.append({'mean_token_nll':float(loss),'target_tokens':int(am[:,1:].sum()),'caption_count':len(pairs[pos:pos+self.bs])})
            finally:self.model.train(previous)
            value,n,total=pooled_nll(records)
            label='test' if np.array_equal(clips,self.splits['test']) else ('development' if np.array_equal(clips,self.report_val) else 'controller')
            require(label!='test' or self.step==STEPS,'Official test loss requested before final scheduled checkpoint')
            if self.step==STEPS and label in ('test','development'):
                put(self.ctx.artifact_dir/('token_nll_'+label+'.json'),{'split':label,'step':self.step,'clip_count':len(clips),
                    'caption_count':len(pairs),'target_tokens':n,'sum_nll_from_token_weighted_batch_losses':total,
                    'pooled_token_nll':value,'batches':records,'forward_autocast':'bfloat16; cross-entropy computed in float32'})
            return value
        @torch.no_grad()
        def decode_split(self,clips):
            require(self.step==STEPS and np.array_equal(clips,self.splits['test']),'Caption evaluation is allowed only for final official-test checkpoint')
            return super().decode_split(clips)
        @torch.no_grad()
        def shuffled_audio(self):
            previous=self.model.training;self.model.eval();records=[];ids=np.asarray(self.report_val)
            shift=1+derive_seed(self.ctx.seed,'development_audio_shuffle')%(len(ids)-1)
            shuffled=np.roll(ids,shift)
            pairs=[(int(j),caption) for i,j in zip(ids,shuffled) for caption in self.rows[int(i)]['captions']]
            try:
                for pos in range(0,len(pairs),self.bs):
                    x,mask,tokens,am=self.batch(pairs[pos:pos+self.bs])
                    with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=self.amp):loss,_=self.model(x,mask,tokens,am)
                    records.append({'mean_token_nll':float(loss),'target_tokens':int(am[:,1:].sum())})
            finally:self.model.train(previous)
            value,n,total=pooled_nll(records)
            return {'split':'development','cyclic_shift':int(shift),'mismatched_audio_clips':len(ids),
                'development_token_nll':value,'target_tokens':n,'training_updates':0,'favorable_direction_required':False}
        def run(self):
            result=super().run()
            result['final']['test_token_nll']=result['final'].pop('test_loss')
            result['final']['development_token_nll']=result['final'].pop('val_loss')
            result['shuffled_audio_diagnostic']=self.shuffled_audio()
            captions=read(self.ctx.artifact_dir/'captions_test.json')
            require(len(captions)==866 and len({r['id'] for r in captions})==866,'Incomplete official-test captions')
            result['caption_diagnostics']={'n':866,'empty_count':sum(not r['candidate'].strip() for r in captions),
                'unique_candidate_count':len({r['candidate'] for r in captions}),'empty_or_unfavorable_outputs_are_not_dropped':True}
            executed=self.step-self.step_at_attempt_start
            result['compute']['steps_executed_this_attempt']=executed
            result['compute']['steps_per_s']=executed/max(result['compute']['train_time_s'],1e-6)
            result['compute']['training_timer_scope']='current attempt training loop including controller feedback and checkpoints; setup, final decoding, metric and audit work are separate'
            result['initial_model_weights_sha256']=self.initial_hash
            result['final_model_weights_sha256']=e.tensor_hash(self.model)
            result['role']=PROTOCOL['evidence_role'];result['confirmation_protocol']=PROTOCOL
            result['official_test_caption_outcomes_computed']=True
            result['test_clips']=866;result['steps_expected']=STEPS
            return result
    return FinalAudio




def audit_saved(job,summary,recompute_metrics=True):
    require(summary['steps']==STEPS and summary['test_clips']==866 and summary['official_test_caption_outcomes_computed'] is True,'Incomplete confirmation job')
    require(all(math.isfinite(float(summary['final'][k])) for k in METRICS+('test_token_nll','development_token_nll')),'Missing/nonfinite result')
    captions=read(job/'artifacts/captions_test.json');splits=read(job/'artifacts/EVALUATION_SPLITS.json')
    require([r['id'] for r in captions]==splits['official_test_ids'] and len(captions)==866,'Test prediction IDs or count differ')
    manifest={r['id']:r for r in (json.loads(x) for x in (DATA/'manifest.jsonl').read_text().splitlines() if x.strip())}
    for row in captions:
        require(manifest[row['id']]['split']=='test' and row['references']==manifest[row['id']]['captions'],'Test caption references differ from the frozen original annotations')
        require(isinstance(row['candidate'],str),'Invalid predicted caption')
    for label,key in [('test','test_token_nll'),('development','development_token_nll')]:
        payload=read(job/('artifacts/token_nll_'+label+'.json'))
        value,n,total=pooled_nll(payload['batches'])
        require(abs(value-summary['final'][key])<=1e-12 and n==payload['target_tokens'],'Saved NLL arithmetic mismatch')
    metric_check=False
    if recompute_metrics:
        from caenl.audio.metrics import caption_metrics
        actual=caption_metrics([r['candidate'] for r in captions],[r['references'] for r in captions],java=True,spice=True,official=True)
        for key in METRICS: require(abs(actual[key]-summary['final'][key])<=1e-10,'Saved-caption metric recomputation differs: '+key)
        metric_check=True
    put(job/'artifacts/RESULT_AUDIT.json',{'status':'PASS','test_rows':len(captions),
        'saved_caption_metrics_recomputed':metric_check,'original_references_matched':True,'pooled_NLL_recomputed':True,
        'predictions_regenerated_from_checkpoint_by_this_audit':False,'independent_external_reproduction':False})




def run_job(method,seed):
    w=wrapper();w.verify_data();verify_features();e=engine();device=e.runtime();w.activate_metric_vendor()
    for rel,h in read(ROOT/'METRIC_COPY.json')['files'].items():require(sha(safe_child(ROOT/'metric_vendor',rel))==h,'Metric asset changed')
    import torch
    from caenl.utils.jobctx import JobContext
    from caenl.utils.seeding import seed_everything
    cfg=configure(method,seed);job=ROOT/'audio'/('seed'+str(seed))/method;job.mkdir(parents=True,exist_ok=True)
    freeze(job/'job.json',cfg)
    if (job/'DONE.json').exists():
        done=read(job/'DONE.json');require(done['binding']==cfg['crossmodal_binding'],'Completed job binding changed')
        for rel,h in done['files'].items():require(sha(safe_child(job,rel))==h,'Completed output changed')
        print('ALREADY COMPLETE:',method,seed,flush=True);return
    class Context(JobContext):
        def save_checkpoint(self,name,payload):
            return super().save_checkpoint(name,dict(payload,confirmation_binding=cfg['crossmodal_binding']))
        def load_checkpoint(self,name,map_location='cpu'):
            payload=super().load_checkpoint(name,map_location)
            if payload is not None:require(payload.get('confirmation_binding')==cfg['crossmodal_binding'],'Checkpoint belongs to another source/protocol')
            return payload
        def log_metrics(self,record):
            super().log_metrics(record)
            if record.get('kind')=='train':progress('AUDIO CONFIRMATION '+method+'/seed'+str(seed)+': step '+str(record['step'])+'/'+str(STEPS))
    seed_everything(seed,deterministic=True,cudnn_benchmark=False)
    torch.use_deterministic_algorithms(True,warn_only=False)
    original_clip=torch.nn.utils.clip_grad_norm_
    def checked_clip(parameters,max_norm,*a,**kw):
        kw['error_if_nonfinite']=True;return original_clip(parameters,max_norm,*a,**kw)
    torch.nn.utils.clip_grad_norm_=checked_clip
    cls=model_class();ctx=Context(job,cfg);ctx.start()
    try:
        result=cls(ctx).run()
        require(result['steps']==STEPS,'Incomplete scheduled training')
        history=read(ctx.artifact_dir/'controller_trajectory.json')['history']
        learned=sum('policy_loss' in row.get('update',{}) for row in history)
        if method=='full_macc':require(learned>=20,'Full MACC did not execute sufficient actual policy updates')
        if method in ('macc_lite','full_macc'):require(len(history)==math.ceil(STEPS/50),'Controller window count differs')
        result['learned_policy_updates']=learned;result['controller_event_count']=len(history)
        result['checkpoint_selection']=PROTOCOL['checkpoint_selection']
        result['training_started']=True
        progress('Auditing saved test captions and metrics: '+method+'/seed'+str(seed))
        audit_saved(job,result,recompute_metrics=True)
        ctx.finish(result)
        payload={p.relative_to(job).as_posix():sha(p) for p in sorted(job.rglob('*')) if p.is_file()
            and p.name not in ('DONE.json','status.json','system.csv') and '.tmp' not in p.name and not p.name.startswith('.writing_')}
        put(job/'DONE.json',{'status':'COMPLETE','seed':seed,'method':method,'binding':cfg['crossmodal_binding'],
                            'files':payload,'time':now(),'scheduled_steps':STEPS,'test_clips':866})
    except BaseException as exc:ctx.fail(exc);raise




def aggregate():
    import numpy as np
    rows=[];by={m:[] for m in METHODS}
    for seed in SEEDS:
        for method in METHODS:
            folder=ROOT/'audio'/('seed'+str(seed))/method;done=read(folder/'DONE.json')
            for rel,h in done['files'].items():require(sha(safe_child(folder,rel))==h,'Final payload checksum differs')
            s=read(folder/'summary.json')
            require(s['steps']==STEPS and s['seed']==seed and s['method']==method and s['test_clips']==866,'Final result identity mismatch')
            require(read(folder/'artifacts/RESULT_AUDIT.json')['status']=='PASS','Missing saved-result audit')
            values={k:float(s['final'][k]) for k in METRICS+('test_token_nll','development_token_nll')}
            by[method].append(values)
            rows.append({'seed':seed,'method':method,'final_metrics':values,'learned_policy_updates':s['learned_policy_updates'],
                'controller_events':s['controller_event_count'],'compute':s['compute'],'wall_time_s':s['wall_time_s'],
                'caption_diagnostics':s['caption_diagnostics'],'shuffled_audio_diagnostic':s['shuffled_audio_diagnostic']})
    grouped={m:{k:{'mean':float(np.mean([r[k] for r in results])),'sample_sd':float(np.std([r[k] for r in results],ddof=1))}
                for k in METRICS+('test_token_nll','development_token_nll')} for m,results in by.items()}
    contrasts=[]
    for target,reference in PRIMARY:
        contrasts.append(dict(paired([r['cider_d'] for r in by[target]],[r['cider_d'] for r in by[reference]]),
                            target=target,reference=reference,endpoint='CIDEr-D',positive_favors_target=True))
    pt=holm([r['paired_t_p_raw'] for r in contrasts]);pe=holm([r['exact_sign_flip_p_raw'] for r in contrasts])
    for i,r in enumerate(contrasts):r.update(paired_t_p_holm=pt[i],exact_sign_flip_p_holm=pe[i],family_size=3)
    put(ROOT/'PER_SEED.json',rows);put(ROOT/'GROUPED_METRICS.json',grouped);put(ROOT/'PRIMARY_CONTRASTS.json',contrasts)




def report(label):
    lines=['# CAENL AudioCaps six-seed confirmation','', 'Status: '+label,'UTC: '+now(),'',
       'Six new training seeds, ten epochs, all 866 retained official-test recordings; fixed available subset.',
       'Primary endpoint CIDEr-D; three declared paired contrasts. Pilot results are separate development evidence.',
       'No favorable-score stopping rule. Fixed-backbone pretraining overlap and third-party provenance limitations remain.', '']
    for name in ('PROTOCOL.json','DATA_BINDING.json','FEATURES_REUSE.json','ENVIRONMENT.json','NUMERICAL_TESTS.json',
                 'SELF_TEST.json','REAL_MODEL_TEST.json','METRIC_TEST.json','METRIC_ASSETS.json','PER_SEED.json','GROUPED_METRICS.json',
                 'PRIMARY_CONTRASTS.json','PRESERVATION.json','FAILED.json','CHILD_FAILURE.json'):
        p=ROOT/name
        if p.is_file():lines+=['## '+name,'SHA-256: '+sha(p),'```json',p.read_text().strip(),'```','']
    for p in sorted(ROOT.glob('audio/seed*/*/summary.json')):
        lines+=['## '+p.relative_to(ROOT).as_posix(),'SHA-256: '+sha(p),'```json',p.read_text().strip(),'```','']
    for p in sorted(ROOT.glob('audio/seed*/*/artifacts/*.json')):
        if p.name not in ('captions_test.json','RESULT_AUDIT.json','token_nll_test.json','token_nll_development.json','EVALUATION_SPLITS.json'):continue
        lines+=['## '+p.relative_to(ROOT).as_posix(),'SHA-256: '+sha(p),'```json',p.read_text().strip(),'```','']
    for name in ('runner.py','pilot_wrapper.py','engine.py'):
        p=ROOT/name
        if p.is_file():lines+=['## Exact source '+name,'SHA-256: '+sha(p),'````````',p.read_text(encoding='utf-8-sig'),'````````','']
    lines+=['END OF AUDIOCAPS CONFIRMATION REPORT']
    write(ROOT/REPORT,'\n'.join(lines)+'\n');write(ROOT/(REPORT+'.sha256'),sha(ROOT/REPORT)+'  '+REPORT+'\n')




def child(args,timeout=None):
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    process=subprocess.Popen([sys.executable,str(ROOT/'runner.py'),*map(str,args)],env=env,start_new_session=True)
    try:rc=process.wait(timeout=timeout)
    except BaseException:
        if process.poll() is None:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait()
        raise
    if rc:
        p=ROOT/'run.log';tail=''
        if p.exists():
            with p.open('rb') as f:f.seek(max(0,p.stat().st_size-24000));tail=f.read().decode('utf-8',errors='replace')
        put(ROOT/'CHILD_FAILURE.json',{'arguments':args,'exit_code':rc,'recent_log':tail})
        raise RuntimeError('Child stage failed: '+repr(args)+'; see CHILD_FAILURE.json')




def worker():
    import fcntl
    stop=threading.Event();thread=None
    try:
        with ExitStack() as stack:
            lock=stack.enter_context((ROOT/'RUN.lock').open('a'));fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            for directory,names in ((PILOT,('START.lock','RUN.lock')),(DATA,('RUN.lock',)),(DATA.parent,('START.lock','RUN.lock')),(DATA.parent/'quarantine_v1',('START.lock','RUN.lock'))):
                for name in names:
                    handle=stack.enter_context((directory/name).open('rb'));fcntl.flock(handle.fileno(),fcntl.LOCK_SH|fcntl.LOCK_NB)
            thread=threading.Thread(target=heartbeat,args=(stop,),daemon=True);thread.start()
            prepare();child(['--numerical-check'],timeout=180)
            progress('Testing local reference caption metrics; no dataset or model downloads')
            child(['--metric-check'],timeout=1200)
            progress('Testing the real AST/GPT-2 model and spectral gradients on the existing GPU')
            child(['--gpu-check'],timeout=900)
            report('PREFLIGHT PASSED; TRAINING NEXT')
            for seed in SEEDS:
                for method in METHODS:
                    progress('Starting/resuming NEW audio confirmation: '+method+'/seed'+str(seed))
                    child(['--job',method,'--seed',seed]);report('RUNNING; COMPLETED RESULTS RETAINED')
            progress('All 24 method runs finished; checking artifacts and calculating the declared paired comparisons')
            aggregate()
            for path,h in read(ROOT/'PRESERVED_INPUTS.json').items():require(sha(path)==h,'Preserved input changed: '+path)
            wrapper().verify_data();verify_features()
            put(ROOT/'PRESERVATION.json',{'status':'PASS','bound_pilot_inputs_rehashed':True,'raw_recordings_not_opened_for_writing':True,
                'feature_cache_read_only':True,'pilot_checkpoints_not_used_for_initialization':True,'other_completed_campaigns_not_rerun':True,
                'boundary':'bound-file verification and saved-caption arithmetic, not independent third-party model training'})
            stop.set();thread.join(timeout=5);report('AUDIOCAPS CONFIRMATION COMPLETE')
            complete={'status':'COMPLETE','completed_method_runs':24,'seeds':list(SEEDS),'epochs':10,'steps_per_job':STEPS,
                'official_test_clips_per_run':866,'binding':read(ROOT/'PROTOCOL.json')['binding'],'time':now(),
                'report':str(ROOT/REPORT),'report_sha256':sha(ROOT/REPORT),'pilot_seeds_pooled_into_inference':False}
            put(ROOT/'COMPLETE.json',complete);put(ROOT/'STATUS.json',complete)
            print('AUDIOCAPS CONFIRMATION COMPLETE\nShare: '+str(ROOT/REPORT),flush=True);return 0
    except BaseException as exc:
        stop.set()
        if thread is not None:thread.join(timeout=5)
        put(ROOT/'FAILED.json',{'status':'FAILED','time':now(),'error':repr(exc),'traceback':traceback.format_exc(),
             'completed_jobs_preserved':len(list(ROOT.glob('audio/seed*/*/DONE.json'))),
             'instruction':'Share --status. Do not reinstall, redownload, delete checkpoints or restart older campaigns.'})
        try:report('STOPPED; NOT COMPLETE')
        except Exception:pass
        traceback.print_exc();return 1




def live():
    return shutil.which('tmux') is not None and subprocess.run(['tmux','has-session','-t',NAME],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0


def status():
    print('SERVER:',os.uname().nodename,now(),flush=True)
    print('TMUX ACTIVE' if live() else 'NO LIVE TMUX SESSION',flush=True)
    for name in ('FAILED.json','COMPLETE.json','STATUS.json'):
        if (ROOT/name).exists():print(json.dumps(read(ROOT/name),indent=2));break
    else:print('Not started.')
    print('Completed audio confirmation runs (target 24):',len(list(ROOT.glob('audio/seed*/*/DONE.json'))))
    if (ROOT/'CHILD_FAILURE.json').exists():print(json.dumps(read(ROOT/'CHILD_FAILURE.json'),indent=2))
    print('Report:',ROOT/REPORT);print('Log:',ROOT/'run.log')
    if shutil.which('nvidia-smi'):subprocess.run(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total','--format=csv'],check=False)




def start(resume=False):
    import ast,fcntl
    require(sys.platform.startswith('linux') and Path('/mnt/caenl').is_mount(),'Start on IBM through the supplied Mac command')
    require(PYTHON.is_file() and shutil.which('tmux') and shutil.which('nvidia-smi'),'Existing interpreter/tmux/GPU tools missing; no installation performed')
    require(not ROOT.is_symlink(),'Result root is a symlink');ROOT.mkdir(mode=0o700,exist_ok=True)
    with (ROOT/'START.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        if live():print('ALREADY RUNNING; no second copy started');status();return
        if (ROOT/'COMPLETE.json').exists():status();return
        require(resume or not (ROOT/'FAILED.json').exists(),'Previous attempt failed; share status before resuming')
        require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'Another GPU process is active; it has not been stopped')
        require(shutil.disk_usage(ROOT).free>=70*1024**3,'At least 70 GiB free required for 24 new checkpoints; no files deleted')
        for rel in ('runner.py','caenl-audiocaps-train-pilot-v1.md'):
            require(sha(safe_child(PILOT,rel))==PILOT_PINS[rel],'Reviewed pilot source/report differs')
        raw=Path(__file__).read_bytes();ast.parse(raw.decode('utf-8-sig'))
        if (ROOT/'runner.py').exists():require((ROOT/'runner.py').read_bytes()==raw,'Different launcher owns this study folder')
        else:write(ROOT/'runner.py',raw)
        if resume:
            history=ROOT/('resume_history_'+str(time.time_ns()));history.mkdir()
            for name in ('FAILED.json','CHILD_FAILURE.json'):
                if (ROOT/name).exists():os.replace(ROOT/name,history/name)
        shell='set -e; if [ -f "$HOME/.caenl-java8-env" ]; then source "$HOME/.caenl-java8-env"; fi; '
        shell+='unset CAENL_DEBUG_STOP_AT_STEP; export PATH='+shlex.quote(str(PYTHON.parent))+':"$PATH"; '
        shell+='export HF_HOME='+shlex.quote(str(ROOT/'assets/hf_home'))+'; '
        shell+='export TORCH_HOME='+shlex.quote(str(ROOT/'assets/torch'))+'; '
        shell+='export XDG_CACHE_HOME='+shlex.quote(str(ROOT/'cache/xdg'))+'; '
        shell+='exec env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 '
        shell+='CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 '
        shell+=shlex.quote(str(PYTHON))+' '+shlex.quote(str(ROOT/'runner.py'))+' --worker >> '+shlex.quote(str(ROOT/'run.log'))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',NAME,'bash -c '+shlex.quote(shell)],check=True)
        print('STARTED:',NAME)
        print('Input/metric/model checks -> 24 NEW ten-epoch runs -> all 866 test clips -> paired statistics.')
        print('Existing audio features reused read-only. No downloads, installations, or old training restarts.')
        print('Status: python3 ~/caenl_audiocaps_confirm_v1.py --status')




def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--start',action='store_true');g.add_argument('--resume',action='store_true')
    g.add_argument('--worker',action='store_true');g.add_argument('--status',action='store_true')
    g.add_argument('--numerical-check',action='store_true');g.add_argument('--metric-check',action='store_true')
    g.add_argument('--gpu-check',action='store_true');g.add_argument('--job',choices=METHODS)
    p.add_argument('--seed',type=int,choices=SEEDS);a=p.parse_args()
    if a.status:status()
    elif a.worker:return worker()
    elif a.numerical_check:numerical_tests()
    elif a.metric_check:wrapper().metric_check()
    elif a.gpu_check:wrapper().gpu_check()
    elif a.job:
        require(a.seed is not None,'Confirmation seed missing');run_job(a.job,a.seed)
    else:start(a.resume)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print('STOPPED:',exc,file=sys.stderr,flush=True);raise SystemExit(1)
# END OF CAENL AUDIOCAPS CONFIRMATION V1