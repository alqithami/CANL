#!/usr/bin/env python3
"""One-shot, finite official-test release and evaluation of160 frozen models.

Training source/output are read-only. Existing evaluation evidence blocks reuse.
No optimizer, checkpoint choice, configuration selection, model-score retry or
experiment launch is provided. v2 corrects only a test-filename validation
assumption after v1 stopped before any model attempt; original deadline stays. This program is separate from frozen confirmation training.
"""
from __future__ import annotations
import argparse
import collections
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import zipfile
import numpy as np

TRAIN_ROOT = Path('/mnt/caenl/active/caenl-cifar100-focused-confirmation-v1')
PILOT = Path('/mnt/caenl/active/caenl-cifar100-v1/pilot_output')
MANIFEST_SHA = '2d920804e8ccd7b5e7d133afe5db88407e1783ead5c0e69c9d4c984a3ce81aba'
FREEZE_SHA = '0c330280a7dd84d0b208f646003e498bcfbf3b1d231056bd2023220cc0d03d3a'
ARCHIVE_SHA = '85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7'
TEST_MEMBER = 'cifar-100-python/test'
PRIOR_EVALUATOR_SHA = 'a56a0be8d7ad2dcb87c530dc33868159bd9e0ba7802d46b0974d9c378ae31a2d'
GLOBAL_SECONDS = 7200
MODEL_SECONDS = 180
BATCH_SIZE = 256


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    """Create immutable evidence; an existing name is never overwritten."""
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def alive(value):
    if not value:
        return False
    try:
        fields = Path('/proc', str(value['pid']), 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == value['start_ticks'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == value['boot_id']
    except (OSError, KeyError, IndexError):
        return False


def prior_input_validation_failure(prior):
    """Versioned recovery is confined to v1's zero-model input validation block."""
    prior = Path(prior).resolve()
    request = read(prior / 'RUN_REQUEST.json')
    status = read(prior / 'STATUS.json')
    release = read(prior / 'TEST_RELEASE_RECEIPT.json')
    release_sha = sha(prior / 'TEST_RELEASE_RECEIPT.json')
    protocol = read(prior / 'LOCKED_ANALYSIS_SPEC.json')
    if status.get('state') != 'BLOCKED' or status.get('error') != "RuntimeError('Official CIFAR100 test shape/labels/identity mismatch')" or status.get('official_test_accessed') is not True:
        raise RuntimeError('Prior failure is not the reviewed input-validation block')
    if any((prior / 'models').glob('*')) or any((prior / name).exists() for name in ['RESULTS_INPUT.json', 'ANALYSIS.json', 'RESULTS.json']):
        raise RuntimeError('Prior model-attempt/score evidence preserved; v2 recovery refused')
    if alive(request.get('identity')):
        raise RuntimeError('Prior evaluator is still live')
    if request.get('source_sha256') != PRIOR_EVALUATOR_SHA or release.get('evaluator_sha256') != PRIOR_EVALUATOR_SHA or release.get('prior_official_test_accessed') is not False:
        raise RuntimeError('Wrong prior official-test release/evaluator binding')
    if protocol.get('locked') is not True or protocol.get('test_release_receipt_sha256') != release_sha:
        raise RuntimeError('Prior locked-analysis release binding mismatch')
    if request.get('finite_models') != 160 or request.get('global_max_seconds') != GLOBAL_SECONDS or request.get('per_model_max_seconds') != MODEL_SECONDS or request.get('training_enabled') is not False or request.get('automatic_retry') is not False:
        raise RuntimeError('Original finite evaluation scope changed')
    if request['deadline_epoch_seconds'] - request['started_epoch_seconds'] != GLOBAL_SECONDS or request['deadline_epoch_seconds'] <= time.time():
        raise RuntimeError('Original evaluation deadline exhausted or altered; no reset')
    evidence = {'prior_output': str(prior), 'prior_release_receipt_sha256': release_sha,
                'prior_run_request_sha256': sha(prior / 'RUN_REQUEST.json'),
                'prior_status_sha256': sha(prior / 'STATUS.json'),
                'prior_locked_analysis_spec_sha256': sha(prior / 'LOCKED_ANALYSIS_SPEC.json'),
                'prior_official_test_accessed': True, 'prior_model_attempts': 0,
                'prior_model_receipts': 0, 'prior_model_scores': 0,
                'validation_amendment': 'Filename uniqueness is not a CIFAR100 row-identity requirement. The pinned test member has10000 rows and9999 unique filenames. Keep every original row; use row indices0..9999 as identities. No duplicate-image inference.',
                'statistical_decisions_changed': False, 'models_or_configurations_changed': False,
                'model_score_retries': 0, 'deadline_extended': False}
    return request, release, protocol, evidence


def locked_protocol(template, release_sha):
    protocol = json.loads(json.dumps(template))
    protocol['locked'] = True
    protocol['test_release_receipt_sha256'] = release_sha
    changed = {k for k in set(template) | set(protocol) if template.get(k) != protocol.get(k)}
    if changed != {'locked', 'test_release_receipt_sha256'}:
        raise RuntimeError('Only the explicit release fields may change')
    return protocol


def official_test(archive):
    """Called ONLY after immutable explicit release and analysis locking."""
    with tarfile.open(archive, 'r:gz') as tf:
        member = tf.getmember(TEST_MEMBER)
        if not member.isfile() or member.size > 40 * 1024**2:
            raise RuntimeError('Unexpected official-test member')
        stream = tf.extractfile(member)
        if stream is None:
            raise RuntimeError('Missing official-test payload')
        payload = stream.read(40 * 1024**2 + 1)
    if len(payload) != member.size:
        raise RuntimeError('Official-test payload length mismatch')
    # The exact official archive SHA is pinned before deserializing its payload.
    batch = pickle.loads(payload, encoding='bytes')
    images = np.asarray(batch[b'data'])
    labels = np.asarray(batch[b'fine_labels'], dtype=np.int64)
    names = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in batch[b'filenames']]
    if images.shape != (10000, 3072) or images.dtype != np.uint8 or labels.shape != (10000,) or labels.min() != 0 or labels.max() != 99 or not np.array_equal(np.bincount(labels, minlength=100), np.full(100, 100)) or len(names) != 10000:
        raise RuntimeError('Official CIFAR100 test shape/labels/identity mismatch')
    return images.reshape(10000, 3, 32, 32), labels, names, hashlib.sha256(payload).hexdigest()


def verify_prediction(path, labels, receipt):
    if sha(path) != receipt['predictions_sha256']:
        raise RuntimeError('Prediction bytes changed')
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {'example_ids', 'predicted_class', 'cross_entropy'}:
            raise RuntimeError('Prediction fields mismatch')
        ids = data['example_ids']; predicted = data['predicted_class']; losses = data['cross_entropy']
        if not np.array_equal(ids, np.arange(10000)) or predicted.shape != labels.shape or losses.shape != labels.shape or not np.isfinite(losses).all() or (losses < 0).any() or predicted.min() < 0 or predicted.max() > 99:
            raise RuntimeError('Invalid per-example prediction records')
        correct = int(np.count_nonzero(predicted == labels))
        if correct != receipt['correct'] or 100 * correct / len(labels) != receipt['accuracy_percent'] or float(losses.astype(np.float64).mean()) != receipt['cross_entropy']:
            raise RuntimeError('Stored prediction/score recomputation mismatch')
    return correct


def run(args):
    training = args.training_root.resolve(); source = training / 'source'; train_out = training / 'output'
    output = args.output.resolve(); archive = args.archive.resolve()
    if training != TRAIN_ROOT or args.pilot_output.resolve() != PILOT or args.archive_sha256 != ARCHIVE_SHA:
        raise RuntimeError('Pinned original campaign/cache/archive required')
    if output == training or training in output.parents or output in training.parents or output == PILOT or PILOT in output.parents or output in PILOT.parents:
        raise RuntimeError('Release output must be separate from original source/models/data')
    prior_request, prior_release, prior_protocol, recovery = prior_input_validation_failure(args.prior_release_output)
    if output == args.prior_release_output.resolve() or args.prior_release_output.resolve() in output.parents or output in args.prior_release_output.resolve().parents:
        raise RuntimeError('Version2 output must preserve prior release evidence separately')
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Existing test-release attempt preserved; no replay/retry')
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source))
    import confirmation as c
    import analyze
    from scipy.stats import t as _required_prespecified_t
    if Path(c.__file__).resolve() != source / 'confirmation.py' or Path(analyze.__file__).resolve() != source / 'analyze.py':
        raise RuntimeError('Wrong frozen module resolution')
    tmp = Path(os.environ.get('TMPDIR', '')).resolve()
    if not os.environ.get('TMPDIR') or not tmp.is_dir() or min(shutil.disk_usage(output).free, shutil.disk_usage(tmp).free) < 8 * 1024**3:
        raise RuntimeError('Spacious persistent TMPDIR/output required')
    start = prior_request['started_epoch_seconds']; deadline = prior_request['deadline_epoch_seconds']; script_sha = sha(Path(__file__))
    def timeout(_sig, _frame):
        raise TimeoutError('Finite official-test evaluation time allowance exhausted')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(max(1, int(deadline - time.time())))
    identity = c.identity(os.getpid())
    write_new(output / 'RUN_REQUEST.json', {'started_utc': prior_request['started_utc'], 'continuation_started_utc': now(), 'prior_release_evidence': recovery, 'started_epoch_seconds': start, 'deadline_epoch_seconds': deadline, 'identity': identity, 'global_max_seconds': GLOBAL_SECONDS, 'per_model_max_seconds': MODEL_SECONDS, 'finite_models': 160, 'automatic_retry': False, 'training_enabled': False, 'source_sha256': script_sha})
    c.core.write(output / 'STATUS.json', {'state': 'VERIFYING_ALL_TRAINING_BEFORE_TEST_RELEASE', 'updated_utc': now(), 'official_test_accessed': True, 'prior_official_test_accessed': True, 'identity': identity})
    def guard(model_started=None):
        if time.time() > deadline or (model_started is not None and time.monotonic() - model_started > MODEL_SECONDS):
            raise TimeoutError('Frozen global/per-model evaluation ceiling reached')
        if sha(Path(__file__)) != script_sha or min(shutil.disk_usage(output).free, shutil.disk_usage(tmp).free) < 2 * 1024**3:
            raise RuntimeError('Evaluator source changed or insufficient runtime/output space')
    with (output / 'RUN.lock').open('a+') as own_lock, (train_out / 'RUN.lock').open('a+') as train_lock, (training.parent / 'CAENL_GPU_CONFIRMATION.lock').open('a+') as gpu_lock:
        for lock in (own_lock, train_lock, gpu_lock):
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        accessed = True
        try:
            if sha(source / 'SOURCE_MANIFEST.json') != MANIFEST_SHA or sha(train_out / 'FREEZE.json') != FREEZE_SHA:
                raise RuntimeError('Original source/freeze binding mismatch')
            freeze = c.validate_freeze(train_out, PILOT, full_inputs=True)
            state = read(train_out / 'STATUS.json')
            if state['state'] != 'CONFIRMATION_TRAINING_VERIFIED_TEST_NOT_RELEASED' or alive(read(train_out / 'PID.json').get('identity')):
                raise RuntimeError('Original confirmation not complete/stopped')
            c.gpu_idle()
            jobs = read(train_out / 'SHARD_JOB_LIST.json')
            receipts = [c.verify_job(train_out, read(train_out / 'jobs' / j['id'] / 'RECEIPT.json')) for j in jobs]
            if len(jobs) != 200 or collections.Counter(r['kind'] for r in receipts) != {'treatment': 160, 'shared_dcr_prefix': 40} or sum(r['training']['updates'] for r in receipts) != 4674150 or sum(r['training']['image_views'] for r in receipts) != 467415000:
                raise RuntimeError('Incomplete or altered finite training matrix')
            result = read(train_out / 'RESULTS.json')
            if result['status'] != 'CONFIRMATION_TRAINING_VERIFIED_TEST_NOT_RELEASED' or result['actual_optimizer_updates'] != 4674150:
                raise RuntimeError('Wrong completed training results')
            treatment_jobs = [j for j in jobs if j['kind'] == 'treatment']
            by_id = {r['job_id']: r for r in receipts}
            model_bindings = {j['id']: {'checkpoint_sha256': by_id[j['id']]['checkpoint_sha256'], 'training_receipt_sha256': sha(train_out / 'jobs' / j['id'] / 'RECEIPT.json'), 'job': j} for j in treatment_jobs}
            if sha(archive) != args.archive_sha256:
                raise RuntimeError('Official source archive bytes changed')
            template = read(train_out / 'ANALYSIS_FROZEN_TEMPLATE.json')
            if template['locked'] is not False or template['test_release_receipt_sha256'] is not None or template['source_binding_sha256'] != freeze['source_binding']['sha256'] or template['analysis'] != analyze.ANALYSIS_CONTRACT:
                raise RuntimeError('Prespecified analysis template changed')
            if prior_release['models'] != model_bindings or prior_release['source_binding_sha256'] != freeze['source_binding']['sha256'] or prior_release['archive_sha256'] != args.archive_sha256:
                raise RuntimeError('Prior released models/source/archive differ; recovery refused')
            unchanged_prior = dict(prior_protocol); unchanged_prior['locked'] = False; unchanged_prior['test_release_receipt_sha256'] = None
            if unchanged_prior != template:
                raise RuntimeError('Prior analysis decisions differ from unchanged frozen template')
            guard()
            write_new(output / 'INPUT_VALIDATION_RECOVERY.json', recovery)
            # This immutable record exists before opening/deserializing the test member.
            release = {'schema_version': 1, 'status': 'EXPLICIT_OFFICIAL_TEST_RELEASE_AUTHORIZED', 'created_utc': now(), 'authorization': 'User instructed proceed after completion/verification; evaluate all frozen models with unchanged analysis.', 'prior_official_test_accessed': True, 'prior_release_receipt_sha256': recovery['prior_release_receipt_sha256'], 'input_validation_recovery_sha256': sha(output / 'INPUT_VALIDATION_RECOVERY.json'), 'original_training_root': str(training), 'source_binding_sha256': freeze['source_binding']['sha256'], 'source_manifest_sha256': MANIFEST_SHA, 'freeze_sha256': FREEZE_SHA, 'training_results_sha256': sha(train_out / 'RESULTS.json'), 'analysis_template_sha256': sha(train_out / 'ANALYSIS_FROZEN_TEMPLATE.json'), 'evaluator_sha256': script_sha, 'archive_path': str(archive), 'archive_sha256': args.archive_sha256, 'accessed_member_after_release': TEST_MEMBER, 'models': model_bindings, 'evaluation': {'partition': 'official_test', 'count': 10000, 'batch_size': BATCH_SIZE, 'normalization_mean': list(c.core.MEAN), 'normalization_std': list(c.core.STD), 'augmentation': None, 'selection': 'all160_final_epoch_checkpoints', 'precision': 'float32_no_AMP_no_TF32', 'network_mode': 'eval'}, 'training_enabled': False, 'configuration_changes': False, 'automatic_retry': False, 'analysis_decisions_changed': False}
            write_new(output / 'TEST_RELEASE_RECEIPT.json', release)
            release_sha = sha(output / 'TEST_RELEASE_RECEIPT.json')
            protocol = locked_protocol(template, release_sha)
            analyze.validate_protocol(protocol)
            write_new(output / 'LOCKED_ANALYSIS_SPEC.json', protocol)
            protocol_sha = sha(output / 'LOCKED_ANALYSIS_SPEC.json')
            torch = c.core.load_torch()
            if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 8 * 1024**3:
                raise RuntimeError('CUDA with8GiBfree required')
            c.gpu_idle(); guard()
            accessed = True
            c.core.write(output / 'STATUS.json', {'state': 'OFFICIAL_TEST_RELEASED_LOADING_EXACT_MEMBER', 'updated_utc': now(), 'official_test_accessed': True, 'identity': identity, 'test_release_receipt_sha256': release_sha})
            images, labels, names, member_sha = official_test(archive)
            np.save(output / 'official_test_labels.npy', labels, allow_pickle=False)
            write_new(output / 'TEST_DATA_MANIFEST.json', {'archive_sha256': args.archive_sha256, 'member': TEST_MEMBER, 'member_sha256': member_sha, 'count': 10000, 'fine_label_counts': np.bincount(labels, minlength=100).tolist(), 'labels_sha256': sha(output / 'official_test_labels.npy'), 'filename_count': len(names), 'unique_filename_count': len(set(names)), 'row_identity': 'original_archive_row_index_0_to9999; filenames are metadata and may repeat', 'filename_order_sha256': hashlib.sha256(json.dumps(names).encode()).hexdigest(), 'images_uint8_content_sha256': hashlib.sha256(images.tobytes()).hexdigest(), 'evaluation_order': 'original_archive_test_order_0_to9999', 'test_release_receipt_sha256': release_sha})
            records = []; test_receipt_hashes = {}; evaluation_times = {}
            for index, job in enumerate(treatment_jobs):
                guard(); c.gpu_idle()
                directory = output / 'models' / job['id']; directory.mkdir(parents=True, exist_ok=False)
                cp_path = train_out / 'jobs' / job['id'] / 'CHECKPOINT.pt'
                if sha(cp_path) != model_bindings[job['id']]['checkpoint_sha256']:
                    raise RuntimeError('Bound model changed before official evaluation')
                model_started = time.monotonic()
                signal.alarm(max(1, int(min(MODEL_SECONDS, deadline - time.time()))))
                write_new(directory / 'ATTEMPT.json', {'started_utc': now(), 'model': job['id'], 'checkpoint_sha256': sha(cp_path), 'test_release_receipt_sha256': release_sha, 'identity': identity})
                c.core.write(output / 'STATUS.json', {'state': 'EVALUATING_OFFICIAL_TEST', 'updated_utc': now(), 'current_model': job['id'], 'model_index': index + 1, 'finite_models': 160, 'completed_models': len(records), 'official_test_accessed': True, 'identity': identity})
                checkpoint = torch.load(cp_path, map_location='cpu', weights_only=False)
                if checkpoint['method'] != job['method'] or checkpoint['config'] != job['config'] or checkpoint['freeze_sha256'] != FREEZE_SHA:
                    raise RuntimeError('Checkpoint method/config binding mismatch')
                net = c.core.model_new(torch); net.load_state_dict(checkpoint['model'], strict=True)
                net = net.to('cuda:0').eval()
                before_state = c.core.state_sha(net.state_dict()); predicted = []; per_loss = []
                torch.cuda.synchronize()
                with torch.inference_mode():
                    for begin in range(0, len(labels), BATCH_SIZE):
                        guard(model_started)
                        ids = np.arange(begin, min(begin + BATCH_SIZE, len(labels)))
                        x = c.core.tensor(images, ids, 'cuda:0', torch, seed=None)
                        y = torch.from_numpy(np.array(labels[ids], copy=True)).to('cuda:0')
                        logits = c.core.forward(net, x, torch)[0]
                        loss = torch.nn.functional.cross_entropy(logits, y, reduction='none')
                        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss).all()):
                            raise FloatingPointError('Nonfinite official-test prediction')
                        predicted.append(logits.argmax(1).cpu().numpy().astype(np.uint8)); per_loss.append(loss.cpu().numpy())
                torch.cuda.synchronize(); guard(model_started); c.gpu_idle()
                if c.core.state_sha(net.state_dict()) != before_state or sha(cp_path) != model_bindings[job['id']]['checkpoint_sha256']:
                    raise RuntimeError('Model state/checkpoint mutated during evaluation')
                predictions = np.concatenate(predicted); losses = np.concatenate(per_loss).astype(np.float32)
                pp = directory / 'PREDICTIONS.npz'
                with pp.open('xb') as f:
                    np.savez_compressed(f, example_ids=np.arange(10000, dtype=np.int64), predicted_class=predictions, cross_entropy=losses)
                correct = int(np.count_nonzero(predictions == labels))
                ev = {'schema_version': 1, 'stage': 'confirmation', 'status': 'COMPLETED_VERIFIED', 'job_id': job['id'], 'method': job['method'], 'seed': job['seed'], 'label_count': job['label_count'], 'label_fraction': job['label_fraction'], 'evaluation_partition': 'official_test', 'count': 10000, 'correct': correct, 'accuracy_percent': 100 * correct / 10000, 'cross_entropy': float(losses.astype(np.float64).mean()), 'checkpoint_sha256': model_bindings[job['id']]['checkpoint_sha256'], 'training_receipt_sha256': model_bindings[job['id']]['training_receipt_sha256'], 'test_release_receipt_sha256': release_sha, 'protocol_sha256': protocol_sha, 'source_binding_sha256': freeze['source_binding']['sha256'], 'test_member_sha256': member_sha, 'predictions_sha256': sha(pp), 'model_state_unchanged': True, 'evaluation_seconds': time.monotonic() - model_started, 'completed_utc': now()}
                verify_prediction(pp, labels, ev)
                write_new(directory / 'RECEIPT.json', ev)
                receipt_sha = sha(directory / 'RECEIPT.json'); test_receipt_hashes[job['id']] = receipt_sha
                records.append({**{k: ev[k] for k in ['stage', 'status', 'method', 'seed', 'label_count', 'label_fraction', 'accuracy_percent', 'evaluation_partition', 'protocol_sha256', 'source_binding_sha256']}, 'receipt_sha256': receipt_sha})
                evaluation_times[job['id']] = ev['evaluation_seconds']
                del net, checkpoint, predicted, per_loss, predictions, losses
                torch.cuda.empty_cache()
                signal.alarm(max(1, int(deadline - time.time())))
                print(json.dumps({'completed_model': job['id'], 'count': index + 1, 'total': 160}), flush=True)
            cost_records = []
            for method in analyze.METHODS:
                for seed in analyze.CONFIRMATION_SEEDS:
                    selected = [r for r in receipts if r.get('method') == method and r['seed'] == seed]
                    prefixes = [by_id[r['pair_id'] + '_shared_dcr_prefix'] for r in selected] if method in ('ratio_dcr', 'full_dcr') else []
                    cost_records.append({'method': method, 'seed': seed, 'training_seconds': sum(r['training']['seconds'] for r in selected + prefixes), 'end_to_end_seconds': sum(r['total_seconds'] for r in selected + prefixes)})
            data = {'schema_version': 1, 'stage': 'confirmation', 'status': 'COMPLETED_VERIFIED', 'protocol_id': analyze.STUDY_ID, 'protocol_sha256': protocol_sha, 'source_binding_sha256': freeze['source_binding']['sha256'], 'evaluation_partition': 'official_test', 'test_release_verified': True, 'test_release_receipt_sha256': release_sha, 'records': records, 'costs': {'scope': 'Per method-seed sum over four budgets of recorded model training and job times; DCR includes its attributed shared prefix. Prefix attribution appears in both DCR methods; not physical total. Excludes pipeline downtime, recovery, data/preparation, test inference and transfer.', 'records': cost_records}}
            if len(records) != 160:
                raise RuntimeError('Incomplete official-test matrix')
            for job in treatment_jobs:
                directory = output / 'models' / job['id']; ev = read(directory / 'RECEIPT.json')
                if sha(directory / 'RECEIPT.json') != test_receipt_hashes[job['id']]:
                    raise RuntimeError('Test receipt changed')
                verify_prediction(directory / 'PREDICTIONS.npz', labels, ev)
            c.source_binding()
            if sha(train_out / 'FREEZE.json') != FREEZE_SHA or sha(output / 'TEST_RELEASE_RECEIPT.json') != release_sha or sha(output / 'LOCKED_ANALYSIS_SPEC.json') != protocol_sha:
                raise RuntimeError('Release/analysis/training binding changed')
            write_new(output / 'RESULTS_INPUT.json', data)
            analysis = analyze.read_and_analyze(output / 'RESULTS_INPUT.json', output / 'LOCKED_ANALYSIS_SPEC.json')
            write_new(output / 'ANALYSIS.json', analysis)
            summary = {'status': 'OFFICIAL_TEST_EVALUATION_AND_FROZEN_ANALYSIS_VERIFIED', 'completed_utc': now(), 'model_count': 160, 'test_examples_per_model': 10000, 'official_test_accessed': True, 'test_release_receipt_sha256': release_sha, 'locked_analysis_spec_sha256': protocol_sha, 'results_input_sha256': sha(output / 'RESULTS_INPUT.json'), 'analysis_sha256': sha(output / 'ANALYSIS.json'), 'receipt_hashes': test_receipt_hashes, 'all_prediction_scores_recomputed': True, 'all_model_states_unchanged': True, 'training_enabled': False, 'evaluation_seconds': sum(evaluation_times.values()), 'elapsed_seconds': time.time() - start, 'physical_training_updates': 4674150, 'physical_training_image_views': 467415000, 'limits': ['Four-method focused confirmation only; SupCon/NONL comparisons remain development-only.', 'No active-acquisition, annotation-saving, equivalence, robustness or mechanism claim is enabled.', 'Sign-flip symmetry and independent seed assumptions; n10 has limited sensitivity to small effects.', 'Potential prior training GPU contention and operational interruption retained in separate recovery record.']}
            write_new(output / 'RESULTS.json', summary)
            review = output / 'CAENL_CIFAR100_OFFICIAL_TEST_REVIEW.zip'; payloads = {}
            with zipfile.ZipFile(review, 'x', zipfile.ZIP_DEFLATED) as z:
                for p in sorted(output.rglob('*')):
                    if p.is_file() and p != review and p.name not in ('RUN.lock', 'STATUS.json'):
                        name = str(p.relative_to(output)); payloads[name] = sha(p); z.write(p, name)
                z.write(Path(__file__), 'source/release_and_evaluate_v2.py'); payloads['source/release_and_evaluate_v2.py'] = script_sha
                for name in ['SOURCE_MANIFEST.json', 'ANALYSIS_SPEC.json', 'analyze.py', 'data_core.py']:
                    p = source / name; name = 'frozen_training_source/' + name; payloads[name] = sha(p); z.write(p, name)
                z.writestr('PAYLOAD_SHA256.json', json.dumps(payloads, indent=2, sort_keys=True) + '\n')
            with zipfile.ZipFile(review) as z:
                if z.testzip() is not None or any(hashlib.sha256(z.read(n)).hexdigest() != h for n, h in payloads.items()):
                    raise RuntimeError('Review ZIP CRC/payload hash mismatch')
            (output / (review.name + '.sha256')).write_text(sha(review) + '  ' + review.name + '\n')
            c.core.write(output / 'STATUS.json', {'state': summary['status'], 'updated_utc': now(), 'official_test_accessed': True, 'completed_models': 160, 'review_sha256': sha(review), 'identity': identity})
            signal.alarm(0)
            print(json.dumps({'status': summary['status'], 'review': str(review), 'sha256': sha(review)}), flush=True)
        except BaseException as error:
            signal.alarm(0)
            c.core.write(output / 'STATUS.json', {'state': 'BLOCKED', 'updated_utc': now(), 'error': repr(error), 'official_test_accessed': accessed, 'identity': identity, 'automatic_retry': False})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root', type=Path, default=TRAIN_ROOT)
    parser.add_argument('--pilot-output', type=Path, default=PILOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--archive-sha256', required=True)
    parser.add_argument('--prior-release-output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', required=True)
    run(parser.parse_args())
