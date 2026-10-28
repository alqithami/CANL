"""Synthetic orchestration/contract tests: never opens real data or GPU."""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import tarfile
import pickle
import io
import types
from unittest.mock import patch
import numpy as np

BASE = Path(__file__).resolve().parent
REPORT_DIR = Path(os.environ.get('CAENL_TEST_REPORT_DIR', str(BASE / 'validation')))
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FROZEN = BASE.parent / 'caenl_cifar100_focused_confirmation_v1/source'
sys.path.insert(0, str(FROZEN))
import analyze as actual_analyze
import confirmation as actual_confirmation
spec = importlib.util.spec_from_file_location('release', BASE / 'source/release_and_evaluate_v2.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

checks = []
template = json.loads((FROZEN / 'ANALYSIS_SPEC.json').read_text())
template['source_binding_sha256'] = '1' * 64
locked = m.locked_protocol(template, '2' * 64)
assert locked['locked'] and not template['locked']
assert {k for k in locked if locked[k] != template[k]} == {'locked', 'test_release_receipt_sha256'}
actual_analyze.validate_protocol(locked)
checks.append('Only explicit release fields change; unchanged analysis contract validates')

class Tensor:
    def __init__(self, a): self.a = np.asarray(a)
    def to(self, *a): return self
    def cpu(self): return self
    def numpy(self): return self.a
    def argmax(self, dim): return Tensor(self.a.argmax(axis=dim))
    def all(self): return self.a.all()

class Net:
    def load_state_dict(self, state, strict=True): assert strict; self.state = state
    def state_dict(self): return self.state
    def to(self, *a): return self
    def eval(self): return self

with tempfile.TemporaryDirectory() as td:
    root = Path(td); train = root / 'training'; source = train / 'source'; out = train / 'output'; target = root / 'release/output'; pilot = root / 'pilot'; tmp = root / 'tmp'
    source.mkdir(parents=True); (out / 'jobs').mkdir(parents=True); pilot.mkdir(); tmp.mkdir()
    for name in ['SOURCE_MANIFEST.json', 'ANALYSIS_SPEC.json', 'analyze.py', 'data_core.py']:
        (source / name).write_bytes((FROZEN / name).read_bytes())
    def write(path, value):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    archive = pilot / 'archive.tar.gz'; archive.write_bytes(b'not accessed synthetic archive')
    write(out / 'FREEZE.json', {})
    jobs = actual_confirmation.job_list(actual_confirmation.protocol())
    write(out / 'SHARD_JOB_LIST.json', jobs); write(out / 'ANALYSIS_FROZEN_TEMPLATE.json', template)
    write(out / 'PID.json', {})
    write(out / 'STATUS.json', {'state': 'CONFIRMATION_TRAINING_VERIFIED_TEST_NOT_RELEASED'})
    write(out / 'RESULTS.json', {'status': 'CONFIRMATION_TRAINING_VERIFIED_TEST_NOT_RELEASED', 'actual_optimizer_updates': 4674150})
    cp_index = {}
    for j in jobs:
        d = out / 'jobs' / j['id']; d.mkdir()
        cp = d / 'CHECKPOINT.pt'; cp.write_bytes(('mock checkpoint ' + j['id']).encode())
        updates = j['steps'] if j['kind'] == 'shared_dcr_prefix' else j['treatment_steps']
        r = {'job_id': j['id'], 'kind': j['kind'], 'seed': j['seed'], 'pair_id': j['pair_id'], 'training': {'updates': updates, 'image_views': updates * 100, 'seconds': 1.}, 'total_seconds': 2., 'checkpoint_sha256': m.sha(cp)}
        if j['kind'] == 'treatment': r['method'] = j['method']
        write(d / 'RECEIPT.json', r); cp_index[str(cp)] = j
    labels = np.repeat(np.arange(100), 100)
    test_accesses = []
    def synthetic_test(path):
        assert path == archive
        assert (target / 'TEST_RELEASE_RECEIPT.json').exists() and (target / 'LOCKED_ANALYSIS_SPEC.json').exists()
        assert json.loads((target / 'LOCKED_ANALYSIS_SPEC.json').read_text())['locked'] is True
        test_accesses.append('after explicit release')
        return np.zeros((10000, 3, 32, 32), dtype=np.uint8), labels, ['mock' + str(i) for i in range(10000)], '3' * 64
    def forward(net, x, torch):
        a = np.full((len(x.a), 100), -1., dtype=np.float32)
        a[np.arange(len(x.a)), labels[x.a]] = 1.
        return Tensor(a), Tensor(np.zeros((len(x.a), 1)))
    def load_checkpoint(path, **kw):
        j = cp_index[str(path)]
        return {'model': {'mock': 1}, 'method': j['method'], 'config': j['config'], 'freeze_sha256': m.FREEZE_SHA}
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True, mem_get_info=lambda: (16 * 1024**3, 16 * 1024**3), synchronize=lambda: None, empty_cache=lambda: None), load=load_checkpoint, from_numpy=Tensor, inference_mode=contextlib.nullcontext, isfinite=lambda t: Tensor(np.isfinite(t.a)), nn=types.SimpleNamespace(functional=types.SimpleNamespace(cross_entropy=lambda logits, y, reduction: Tensor(np.full(len(y.a), .5, dtype=np.float32)))))
    core = types.SimpleNamespace(write=write, load_torch=lambda: fake_torch, model_new=lambda torch: Net(), tensor=lambda images, ids, device, torch, seed=None: Tensor(ids), forward=forward, state_sha=lambda state: '5' * 64, MEAN=actual_confirmation.core.MEAN, STD=actual_confirmation.core.STD)
    fake_c = types.SimpleNamespace(__file__=str(source / 'confirmation.py'), core=core, identity=lambda pid: {'pid': pid, 'start_ticks': '0', 'boot_id': 'mock'}, validate_freeze=lambda *a, **kw: {'source_binding': {'sha256': '1' * 64}}, verify_job=lambda o, r: r, source_binding=lambda: {'sha256': '1' * 64}, gpu_idle=lambda: None)
    raw_sha = m.sha
    def bound_sha(path):
        path = Path(path)
        if path == source / 'SOURCE_MANIFEST.json': return m.MANIFEST_SHA
        if path == out / 'FREEZE.json': return m.FREEZE_SHA
        if path == archive: return m.ARCHIVE_SHA
        return raw_sha(path)
    prior = root / 'prior_release/output'; prior.mkdir(parents=True)
    prior_started = time.time() - 120
    prior_request = {'started_utc': 'synthetic_original_start', 'started_epoch_seconds': prior_started, 'deadline_epoch_seconds': prior_started + m.GLOBAL_SECONDS, 'identity': {}, 'source_sha256': m.PRIOR_EVALUATOR_SHA, 'finite_models': 160, 'global_max_seconds': m.GLOBAL_SECONDS, 'per_model_max_seconds': m.MODEL_SECONDS, 'training_enabled': False, 'automatic_retry': False}
    write(prior / 'RUN_REQUEST.json', prior_request)
    write(prior / 'STATUS.json', {'state': 'BLOCKED', 'error': "RuntimeError('Official CIFAR100 test shape/labels/identity mismatch')", 'official_test_accessed': True})
    prior_models = {j['id']: {'checkpoint_sha256': m.sha(out / 'jobs' / j['id'] / 'CHECKPOINT.pt'), 'training_receipt_sha256': m.sha(out / 'jobs' / j['id'] / 'RECEIPT.json'), 'job': j} for j in jobs if j['kind'] == 'treatment'}
    write(prior / 'TEST_RELEASE_RECEIPT.json', {'evaluator_sha256': m.PRIOR_EVALUATOR_SHA, 'prior_official_test_accessed': False, 'models': prior_models, 'source_binding_sha256': '1' * 64, 'archive_sha256': m.ARCHIVE_SHA})
    write(prior / 'LOCKED_ANALYSIS_SPEC.json', m.locked_protocol(template, m.sha(prior / 'TEST_RELEASE_RECEIPT.json')))
    old_evidence = {str(p.relative_to(prior)): m.sha(p) for p in prior.iterdir()}
    args = argparse.Namespace(training_root=train, pilot_output=pilot, output=target, archive=archive, archive_sha256=m.ARCHIVE_SHA, prior_release_output=prior)
    with patch.multiple(m, TRAIN_ROOT=train, PILOT=pilot), patch.object(m, 'sha', side_effect=bound_sha), patch.object(m, 'official_test', side_effect=synthetic_test), patch.object(actual_analyze, '__file__', str(source / 'analyze.py')), patch.dict(sys.modules, {'confirmation': fake_c}), patch.dict(os.environ, {'TMPDIR': str(tmp)}), contextlib.redirect_stdout(open(os.devnull, 'w')):
        m.run(args)
    assert test_accesses == ['after explicit release']
    current_request = json.loads((target / 'RUN_REQUEST.json').read_text())
    assert current_request['started_epoch_seconds'] == prior_request['started_epoch_seconds']
    assert current_request['deadline_epoch_seconds'] == prior_request['deadline_epoch_seconds']
    assert all(m.sha(prior / name) == value for name, value in old_evidence.items())
    assert json.loads((target / 'TEST_RELEASE_RECEIPT.json').read_text())['prior_official_test_accessed'] is True
    checks.append('Original two-hour start/deadline and prior accessed release bytes preserved exactly')
    checks.append('New release binds prior access and proves zero model attempts/receipts/scores')
    (prior / 'models/partial').mkdir(parents=True)
    try: m.prior_input_validation_failure(prior)
    except RuntimeError: pass
    else: raise AssertionError('Prior model evidence accepted for recovery')
    (prior / 'models/partial').rmdir(); (prior / 'models').rmdir()
    checks.append('Any prior model attempt evidence blocks input-validation recovery')
    assert json.loads((target / 'RESULTS.json').read_text())['model_count'] == 160
    analysis = json.loads((target / 'ANALYSIS.json').read_text())
    assert analysis['n_paired_seeds'] == 10 and analysis['primary_family_size'] == 3
    assert analysis['methods']['ce']['mean_AULC'] == 100.0
    checks.append('Full160-model mocked orchestration reaches frozen three-contrast analysis')
    checks.append('First and only synthetic test access occurs after immutable release and locked protocol')
    checks.append('Complete 40-record descriptive cost family validates')
    first = target / 'models' / jobs[1]['id']
    receipt = json.loads((first / 'RECEIPT.json').read_text())
    m.verify_prediction(first / 'PREDICTIONS.npz', labels, receipt)
    receipt['correct'] -= 1
    try: m.verify_prediction(first / 'PREDICTIONS.npz', labels, receipt)
    except RuntimeError: pass
    else: raise AssertionError('Score tampering accepted')
    checks.append('Independent per-example accuracy/CE recomputation rejects score tampering')
    try:
        with patch.multiple(m, TRAIN_ROOT=train, PILOT=pilot): m.run(args)
    except RuntimeError as error: assert 'Existing test-release attempt' in str(error)
    else: raise AssertionError('Replay accepted')
    checks.append('Existing release evidence refuses any replay')
    import zipfile
    review = target / 'CAENL_CIFAR100_OFFICIAL_TEST_REVIEW.zip'
    with zipfile.ZipFile(review) as z:
        payloads = json.loads(z.read('PAYLOAD_SHA256.json'))
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in payloads.items())
    checks.append('Compact review CRC and every payload hash verified')

with tempfile.TemporaryDirectory() as td:
    archive = Path(td) / 'synthetic.tar.gz'
    names = [('mock' + str(i)).encode() for i in range(10000)]; names[-1] = names[0]
    payload = pickle.dumps({b'data': np.zeros((10000, 3072), dtype=np.uint8), b'fine_labels': np.repeat(np.arange(100), 100).tolist(), b'filenames': names}, protocol=2)
    with tarfile.open(archive, 'w:gz') as tf:
        member = tarfile.TarInfo(m.TEST_MEMBER); member.size = len(payload); tf.addfile(member, io.BytesIO(payload))
    images, labels, recovered_names, payload_sha = m.official_test(archive)
    assert images.shape == (10000, 3, 32, 32) and len(labels) == 10000 and len(recovered_names) == 10000 and len(set(recovered_names)) == 9999
    assert recovered_names[0] == recovered_names[-1] and payload_sha == hashlib.sha256(payload).hexdigest()
    checks.append('Duplicate filename regression retains all10000 rows and original order')

report = {'status': 'PASSED', 'synthetic_only': True, 'real_dataset_accessed': False, 'real_gpu_accessed': False, 'checks': checks, 'evaluator_sha256': m.sha(BASE / 'source/release_and_evaluate_v2.py')}
(REPORT_DIR / 'RELEASE_V2_TESTS.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
