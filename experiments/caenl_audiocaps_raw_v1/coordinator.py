#!/usr/bin/env python3
"""Finite, restartable coordinator for the separate 58-job AudioCaps study.

This program never provisions a server, downloads data, deletes evidence, changes
the protocol, or starts work on another campaign. See COORDINATOR_README.md.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = Path('/srv/iga/caenl-audiocaps-raw-v1')
DEFAULT_PYTHON = Path('/srv/iga/venv/bin/python')
STUDY = 'caenl-audiocaps-raw-source-controlled-followup-v1'
ARCHIVE = 'CAENL_AUDIOCAPS_RAW_REVIEW.zip'
CORE = ('coordinator.py', 'prepare_data.py', 'train_study.py', 'study_features.py',
        'study_stats.py', 'protocol.json', 'reference/SOURCE_PROVENANCE.json',
        'EXECUTION_AMENDMENT_20261018.json')


class Blocked(RuntimeError):
    pass


class AlreadyOwned(Blocked):
    pass


def require(value, message):
    if not value:
        raise Blocked(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + '\n').encode()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def put(path, value):
    path = Path(path)
    require(not path.is_symlink(), 'Refusing symlink output: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.coordinator-', delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded(value)); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def freeze(path, value):
    if Path(path).exists():
        require(read(path) == value, 'Frozen coordinator contract changed: ' + str(path))
    else:
        put(path, value)


@contextmanager
def lock(path, blocking=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), 'Refusing symlink lock: ' + str(path))
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise AlreadyOwned('Coordinator already owns ' + str(path)) from exc
        yield


def identity(pid):
    """Linux PID identity includes boot ID and kernel start tick, not PID alone."""
    try:
        pid = int(pid)
        text = Path(f'/proc/{pid}/stat').read_text()
        rest = text[text.rfind(')') + 2:].split()
        if rest[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': rest[19],
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except (OSError, ValueError, IndexError, TypeError):
        return None


def alive(record):
    return (isinstance(record, dict) and isinstance(record.get('identity'), dict)
            and identity(record.get('pid')) == record['identity'])


def counts(root):
    output = {}
    for stage, expected in (('tuning', 18), ('confirm', 40)):
        completed = []
        for path in sorted((root / stage).rglob('DONE.json')):
            try:
                item = read(path)
                if item.get('status') == 'COMPLETE' and item.get('phase') == stage:
                    completed.append(str(path.relative_to(root)))
            except (OSError, ValueError):
                continue
        output[stage] = {'receipt_count': len(completed), 'expected': expected,
                         'note': 'Progress receipt count only; scientific completion requires full hash and analysis verification.'}
    return output


def status(root):
    path = root / 'STATUS.json'
    result = read(path) if path.exists() else {'status': 'NOT_STARTED'}
    pid_path = root / 'PID.json'
    pid = read(pid_path) if pid_path.exists() else None
    result['worker_active'] = alive(pid)
    result['progress'] = counts(root)
    children = root / 'CHILDREN.json'
    result['live_children'] = [row for row in read(children).get('children', []) if alive(row)] if children.exists() else []
    result['status_path'] = str(path)
    if result.get('status') not in ('COMPLETE', 'BLOCKED', 'NOT_STARTED') and not result['worker_active']:
        result['access_note'] = 'Coordinator is not live. Inspect logs and live_children before --start; do not duplicate a live child.'
    return result


def execution_workers(protocol=None):
    protocol = read(HERE / 'protocol.json') if protocol is None else protocol
    workers = protocol['execution']['workers']
    require(type(workers) is int and workers in (1, 2), 'Execution requires one or two deterministic workers')
    physical = protocol['execution'].get('physical_gpu_indices', list(range(workers)))
    require(physical == list(range(workers)), 'Physical GPU mapping must be contiguous from zero and match worker count')
    return workers


def contract(python):
    protocol = read(HERE / 'protocol.json')
    require(protocol['study'] == STUDY and protocol['tuning']['runs'] == 18 and
            protocol['confirmation']['runs'] == 40,
            'Protocol is not the frozen finite 18+40 job study')
    workers = execution_workers(protocol)
    files = {name: sha(HERE / name) for name in CORE}
    provenance = read(HERE / 'reference/SOURCE_PROVENANCE.json')
    for row in provenance['files']:
        relative = Path(row['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe reference provenance path')
        path = HERE / 'reference' / relative
        require(path.is_file() and not path.is_symlink() and sha(path) == row['sha256'],
                'Reference source differs: ' + str(relative))
        files['reference/' + relative.as_posix()] = row['sha256']
    require(python.is_file(), 'Python is unavailable: ' + str(python))
    return {'schema': 1, 'study': STUDY, 'method_runs': 58, 'tuning_runs': 18,
            'confirmation_runs': 40, 'workers': workers, 'physical_gpu_indices': list(range(workers)),
            'python': str(python.absolute()), 'python_resolved_binary': str(python.resolve()), 'code_sha256': files,
            'historical_campaigns_modified': False, 'no_outcome_stopping': True}


def child_environment(root, gpu=None):
    env = dict(os.environ)
    overlay = [str(root / 'runtime/pydeps'), str(root / 'metric_vendor')]
    if env.get('PYTHONPATH'):
        overlay.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(overlay)
    env['JAVA_HOME'] = str(root / 'runtime/java8')
    env['PATH'] = str(root / 'runtime/java8/bin') + os.pathsep + env.get('PATH', '')
    env['TMPDIR'] = str(root / 'runtime/tmp')
    env['HF_HUB_OFFLINE'] = '1'; env['TRANSFORMERS_OFFLINE'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'; env['PYTHONUNBUFFERED'] = '1'
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    env['CUDA_VISIBLE_DEVICES'] = '' if gpu is None else str(gpu)
    Path(env['TMPDIR']).mkdir(parents=True, exist_ok=True)
    return env


class Coordinator:
    def __init__(self, args):
        self.args = args; self.root = args.root.resolve()
        # Preserve the venv invocation path: resolving its executable symlink
        # would bypass pyvenv.cfg and discard the installed base dependencies.
        self.python = args.python.absolute()
        self.workers = execution_workers()
        self.state = {'schema': 1, 'study': STUDY, 'pid': os.getpid(), 'started_at': now(),
                      'method_runs_expected': 58, 'historical_results_pooled': False}
        self.children = []

    def update(self, state, **fields):
        self.state.update(status=state, updated_at=now(), **fields)
        self.state['progress'] = counts(self.root)
        put(self.root / 'STATUS.json', self.state)
        print(json.dumps({k: self.state[k] for k in ('status', 'updated_at', 'progress')}, sort_keys=True), flush=True)

    def record_children(self):
        put(self.root / 'CHILDREN.json', {'updated_at': now(), 'children': self.children})

    def wait_readiness(self):
        paths = {'source': ('incoming/SOURCE_READY.json', {'VERIFIED', 'SOURCE_READY', 'READY'}),
                 'assets': ('assets/ASSETS_READY.json', {'VERIFIED', 'READY'}),
                 'runtime': ('runtime/RUNTIME_READY.json', {'READY'}),
                 'smoke': ('SMOKE.json', {'PASS'})}
        deadline_path = self.root / 'READINESS_DEADLINE.json'
        if not deadline_path.exists():
            put(deadline_path, {'created_at': now(), 'deadline_epoch': time.time() + self.args.wait_hours * 3600,
                                'wait_hours': self.args.wait_hours})
        deadline = float(read(deadline_path)['deadline_epoch'])
        while True:
            missing = []; records = {}
            for name, (relative, acceptable) in paths.items():
                path = self.root / relative
                if not path.exists():
                    missing.append(relative); continue
                row = read(path)
                value = row.get('status', row.get('state'))
                require(value in acceptable, 'Readiness receipt is not successful: ' + relative + ' (' + str(value) + ')')
                records[name] = {'path': relative, 'sha256': sha(path)}
            if not missing:
                source = read(self.root / paths['source'][0])
                archive = self.root / 'incoming/audiocaps_raw_audio.zip'
                require(archive.is_file() and archive.stat().st_size == source['archive_bytes'], 'Verified source archive is missing or has changed size')
                require(len(source['archive_sha256']) == 64, 'Missing source SHA-256')
                smoke = read(self.root / 'SMOKE.json')
                smoke_binding_path = self.root / 'SMOKE_INPUT_BINDING.json'
                require(smoke_binding_path.is_file(), 'Missing immutable real-model smoke source/assets/code binding')
                smoke_binding = read(smoke_binding_path)
                scientific_code = ('train_study.py', 'study_features.py', 'study_stats.py', 'protocol.json', 'prepare_data.py')
                require(smoke_binding.get('code') == {name: sha(HERE / name) for name in scientific_code},
                        'Smoke was run with different scientific source/protocol files')
                smoke_contract = {key: smoke_binding[key] for key in ('reference', 'assets', 'code')}
                require(smoke.get('source_assets_code_binding') == hashlib.sha256(encoded(smoke_contract)).hexdigest(),
                        'Smoke receipt differs from its source/assets/code binding')
                require(smoke.get('real_validation_generation_api_passed') is True and
                        smoke.get('real_validation_generation_examples') == 2 and
                        smoke.get('fixed_synthetic_text_all_metrics_passed') is True,
                        'Smoke must pass actual validation generation and every frozen caption metric')
                for name, key in (('SMOKE_GENERATION.json', 'generation_fixture_sha256'),
                                  ('SMOKE_METRICS.json', 'metric_fixture_sha256')):
                    fixture = self.root / name
                    require(fixture.is_file() and sha(fixture) == smoke.get(key), 'Smoke fixture receipt changed: ' + name)
                    row = read(fixture)
                    require(row.get('status') == 'PASS' and row.get('input_binding') == smoke['input_binding'] and
                            row.get('scientific_runs_counted') == 0 and row.get('official_test_outcomes_computed') is False,
                            'Smoke fixture is not bound or computed test outcomes: ' + name)
                # prepare_data independently verifies every source byte before DATA_READY.
                records['smoke_binding'] = {'path': 'SMOKE_INPUT_BINDING.json', 'sha256': sha(smoke_binding_path)}
                freeze(self.root / 'COORDINATOR_READINESS.json', records)
                return
            require(time.time() < deadline, 'Finite readiness deadline expired; missing: ' + ', '.join(missing) + '. Finish staging, inspect receipts, then explicitly extend READINESS_DEADLINE.json before resuming.')
            self.update('WAITING_FOR_VERIFIED_INPUTS', waiting_for=missing, readiness_deadline_epoch=deadline)
            time.sleep(min(30, max(.1, deadline - time.time())))

    def run_phase(self, label, specs):
        """Wait all assigned children; a failed sibling never causes evidence deletion."""
        if any(gpu is not None for _, gpu in specs):
            self.gpu_preflight()
        self.update('RUNNING', phase=label, waiting_for=[])
        started = time.monotonic(); active = []; self.children = []
        for index, (arguments, gpu) in enumerate(specs):
            log = self.root / 'logs' / (label + '_' + str(index) + '.log')
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [str(self.python), '-B', *arguments]
            with log.open('ab', buffering=0) as output:
                child = subprocess.Popen(command, cwd=HERE, env=child_environment(self.root, gpu),
                                         stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            ident = identity(child.pid)
            row = {'pid': child.pid, 'identity': ident, 'phase': label, 'physical_gpu': gpu,
                   'command': command, 'log': str(log), 'started_at': now()}
            self.children.append(row); self.record_children(); active.append((child, row))
        last = 0.; failed = []
        while active:
            pending = []
            for child, row in active:
                code = child.poll()
                if code is None:
                    pending.append((child, row)); continue
                row.update(exit_code=code, stopped_at=now())
                if code != 0:
                    failed.append({'exit_code': code, 'log': row['log'], 'phase': label})
                self.record_children()
            active = pending
            if time.monotonic() - last >= 15:
                self.update('BLOCKED_WAITING_FOR_EXISTING_WORKER' if failed else 'RUNNING',
                            phase=label, child_failures=failed,
                            live_child_pids=[child.pid for child, _ in active])
                last = time.monotonic()
            if active:
                time.sleep(2)
        elapsed = time.monotonic() - started
        with (self.root / 'PHASE_ATTEMPTS.jsonl').open('ab') as handle:
            event = {'phase': label, 'finished_at': now(), 'wall_seconds': elapsed,
                     'children': self.children, 'successful': not failed}
            handle.write((json.dumps(event, sort_keys=True, allow_nan=False) + '\n').encode())
            handle.flush(); os.fsync(handle.fileno())
        require(not failed, 'Child phase failed: ' + json.dumps(failed) + '; other assigned workers were allowed to finish. Correct the infrastructure/runtime issue, preserve all evidence, then --start to resume.')

    def gpu_preflight(self):
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,memory.total',
                                             '--format=csv,noheader,nounits'], text=True)
        rows = []
        for line in inventory.splitlines():
            fields = [field.strip() for field in line.split(',')]
            require(len(fields) == 4, 'Cannot parse GPU inventory')
            if fields[0] in {str(index) for index in range(self.workers)}:
                rows.append({'index': int(fields[0]), 'name': fields[1], 'uuid': fields[2], 'memory_mib': int(fields[3])})
        require(len(rows) == self.workers and {r['index'] for r in rows} == set(range(self.workers)),
                'Study requires selected physical GPU indices ' + str(list(range(self.workers))))
        freeze(self.root / 'GPU_INVENTORY.json', sorted(rows, key=lambda row: row['index']))
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                             '--format=csv,noheader,nounits'], text=True)
        occupied = [line for line in processes.splitlines() if any(row['uuid'] in line for row in rows)]
        require(not occupied, 'Selected GPU has an existing compute process; no competing job was started: ' + '; '.join(occupied))

    def train_args(self, phase, *extra):
        return [str(HERE / 'train_study.py'), '--root', str(self.root), '--phase', phase, *extra]

    def execute(self):
        root = self.root
        root.mkdir(parents=True, exist_ok=True)
        require(not root.is_symlink(), 'Refusing symlink root')
        with lock(root / '.coordinator.lock'):
            prior = root / 'CHILDREN.json'
            orphan = [row for row in read(prior).get('children', []) if alive(row)] if prior.exists() else []
            require(not orphan, 'Existing child processes are still live; no duplicate was started: ' + str([r['pid'] for r in orphan]) + '. Wait for them to stop and rerun --start.')
            put(root / 'PID.json', {'pid': os.getpid(), 'identity': identity(os.getpid()), 'started_at': now(), 'command': sys.argv})
            freeze(root / 'COORDINATOR_CONTRACT.json', contract(self.python))
            if (root / 'review/ARCHIVE_RECEIPT.json').exists():
                receipt = verify_review(root / 'review')
                self.update('COMPLETE', review=receipt, scientific_success_not_guaranteed=True)
                return
            self.wait_readiness()
            self.run_phase('prepare', [([str(HERE / 'prepare_data.py'), '--root', str(root)], None)])
            self.run_phase('features', [(self.train_args('features', '--gpu', '0'), 0)])
            self.run_phase('initialize', [(self.train_args('initialize'), None)])
            for stage in ('tuning', 'confirm'):
                self.run_phase(stage, [(self.train_args('worker', '--stage', stage, '--worker-index', str(gpu), '--workers', str(self.workers), '--gpu', '0'), gpu) for gpu in range(self.workers)])
                self.run_phase('analyze_' + stage, [(self.train_args('analyze', '--stage', stage), None)])
            self.update('PACKAGING_VERIFIED_REVIEW', phase='package')
            receipt = package_review(root)
            self.update('COMPLETE', review=receipt, scientific_success_not_guaranteed=True)


def practical_cost(root):
    rows = []
    for stage in ('tuning', 'confirm'):
        for path in sorted((root / stage).rglob('summary.json')):
            row = read(path)
            rows.append({'path': str(path.relative_to(root)), 'stage': stage, 'method': row['method'],
                         'seed': row['seed'], 'wall_time_s': row['wall_time_s'],
                         'timings_s': row.get('timings_s'), 'attempt_count': row.get('attempt_count'),
                         'compute_last_attempt': row.get('compute'), 'telemetry_last_attempt': row.get('telemetry')})
    require(len(rows) == 58, 'Practical-cost report requires all 58 summaries')
    grouped = {}
    for stage in ('tuning', 'confirm'):
        for method in sorted({row['method'] for row in rows if row['stage'] == stage}):
            subset = [row for row in rows if row['stage'] == stage and row['method'] == method]
            walls = [row['wall_time_s'] for row in subset]
            grouped[stage + '/' + method] = {'n': len(walls), 'sum_job_wall_seconds': sum(walls),
                                           'mean_job_wall_seconds': statistics.mean(walls)}
    return {'rows': rows, 'grouped': grouped,
            'search_cost_included': True, 'phase_attempts_path': 'PHASE_ATTEMPTS.jsonl',
            'limits': ['Job wall time includes the retained training/evaluation attempts recorded by JobContext.',
                       'Fresh saved-result audit subprocess time is recorded at coordinator phase level, not job wall time.',
                       'The compute train_time_s and allocator peaks in each summary describe its last attempt; cumulative job wall includes prior attempts.',
                       'Wall time is not a hardware-independent compute measure. Worker count and physical GPU mapping are recorded in the frozen coordinator contract.',
                       'Hyperparameter search, feature extraction, initialization, controller-state maintenance and engineering complexity remain explicit costs.']}


def review_payloads(root):
    paths = []
    fixed = ['RESULTS.json', 'analysis.json', 'PRACTICAL_COST.json', 'SELECTED_FIXED.json',
             'INPUT_BINDING.json', 'COORDINATOR_CONTRACT.json', 'COORDINATOR_READINESS.json',
             'PHASE_ATTEMPTS.jsonl', 'SMOKE.json', 'INITIALS_READY.json', 'FEATURES_READY.json',
             'SMOKE_GENERATION.json', 'SMOKE_METRICS.json', 'SMOKE_INPUT_BINDING.json',
             'features/EXTRACTION_BINDING.json', 'features/EXTRACTION_REPORT.json',
             'features/PREPROCESSED_INPUT_AUDIT.json', 'features/meta.json',
             'GPU_INVENTORY.json',
             'incoming/SOURCE_READY.json', 'assets/MODELS_FROZEN.json', 'assets/ASSETS_READY.json',
             'METRIC_COPY.json', 'runtime/RUNTIME_READY.json', 'runtime/JAVA8_MANIFEST.json',
             'data/DATA_READY.json', 'data/PREPARATION_BINDING.json', 'data/manifest.jsonl',
             'data/manifest_meta.json', 'data/AUDIT.jsonl', 'data/EXCLUSIONS.jsonl', 'data/SELECTION_REPORT.json']
    for relative in fixed:
        path = root / relative
        require(path.is_file() and not path.is_symlink(), 'Missing review evidence: ' + relative)
        paths.append((relative, path))
    for split in ('train', 'val', 'test'):
        path = root / 'incoming/official_captions' / (split + '.csv')
        require(path.is_file(), 'Missing original caption file: ' + str(path))
        paths.append((str(path.relative_to(root)), path))
    for stage in ('tuning', 'confirm'):
        for folder in sorted({p.parent for p in (root / stage).rglob('DONE.json')}):
            for path in sorted(folder.rglob('*')):
                if path.is_file() and not path.is_symlink() and path.suffix in ('.json', '.jsonl', '.csv') and not any(p.startswith('.') for p in path.relative_to(folder).parts):
                    paths.append((str(path.relative_to(root)), path))
    for relative, expected in read(root / 'COORDINATOR_CONTRACT.json')['code_sha256'].items():
        require(sha(HERE / relative) == expected, 'Frozen source changed before review packaging: ' + relative)
        paths.append(('source/' + relative, HERE / relative))
    for name in ('COORDINATOR_README.md', 'TRAINING_README.md'):
        if (HERE / name).is_file():
            paths.append(('source/' + name, HERE / name))
    for path in sorted((root / 'initial').glob('*/INITIAL.json')):
        paths.append((str(path.relative_to(root)), path))
    unique = dict(paths)
    require(len(unique) == len(paths), 'Repeated review member path')
    require(sum(p.stat().st_size for p in unique.values()) <= 1 << 30, 'Text review exceeds 1 GiB; preserve all evidence and inspect packaging before transfer')
    return unique


def verify_review(directory):
    receipt = read(directory / 'ARCHIVE_RECEIPT.json')
    archive = directory / ARCHIVE
    require(archive.is_file() and archive.stat().st_size == receipt['archive_bytes'] and sha(archive) == receipt['archive_sha256'], 'Review ZIP differs from authoritative receipt')
    require((directory / (ARCHIVE + '.sha256')).read_text() == receipt['archive_sha256'] + '  ' + ARCHIVE + '\n', 'Review checksum file differs')
    with zipfile.ZipFile(archive) as zipped:
        require(zipped.testzip() is None, 'Review ZIP CRC failed')
        manifest_bytes = zipped.read('PAYLOAD_MANIFEST.json')
        require(hashlib.sha256(manifest_bytes).hexdigest() == receipt['payload_manifest_sha256'], 'Payload manifest differs')
        manifest = json.loads(manifest_bytes)
        require(len(zipped.namelist()) == len(set(zipped.namelist())) and
                set(zipped.namelist()) == set(manifest['files']) | {'PAYLOAD_MANIFEST.json'}, 'ZIP payload inventory differs')
        for name, row in manifest['files'].items():
            rel = Path(name)
            require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe ZIP member')
            digest = hashlib.sha256(); size = 0
            with zipped.open(name) as handle:
                for block in iter(lambda: handle.read(4 << 20), b''):
                    digest.update(block); size += len(block)
            require(size == row['bytes'] and digest.hexdigest() == row['sha256'], 'ZIP payload hash failed: ' + name)
    return receipt


def package_review(root):
    """Only called after full training-side receipt/hash/contrast verification."""
    directory = root / 'review'; directory.mkdir(exist_ok=True)
    if (directory / 'ARCHIVE_RECEIPT.json').exists():
        return verify_review(directory)
    analysis = read(root / 'analysis.json'); selection = read(root / 'SELECTED_FIXED.json')
    require(analysis['status'] == 'COMPLETE' and analysis['completed_method_runs'] == 40 and
            len(analysis['primary_contrasts']) == 5 and len(selection['all_18_tuning_outcomes']) == 18 and
            selection['official_test_outcomes_used'] is False and analysis['input_binding'] == selection['input_binding'],
            'Complete locked 18+40 outcomes and five planned contrasts are required')
    cost = practical_cost(root); freeze(root / 'PRACTICAL_COST.json', cost)
    result = {'schema': 1, 'status': 'COMPLETE', 'study': STUDY,
              'method_runs': {'tuning': 18, 'confirmation': 40, 'total': 58},
              'input_binding': analysis['input_binding'], 'data': read(root / 'data/DATA_READY.json'),
              'fixed_selection': selection, 'confirmation': analysis, 'practical_cost': cost,
              'evidence_scope': {'historical_24_audio_runs': 'separate; not pooled or overwritten',
                                 'adaptive_vs_tuned_fixed': 'two prespecified direct paired contrasts; preserve every estimate, interval and adjusted p-value',
                                 'component_attribution': 'spectral audio regularization on a common fixed labeled training set; not a classification acquisition ablation',
                                 'robustness_and_geometry': 'not measured in this AudioCaps study; smaller-radius classification diagnostics remain a separate campaign',
                                 'external_generalization': 'same previously studied AudioCaps benchmark; not an untouched or new external dataset'},
              'scientific_success_not_guaranteed': True,
              'compact_review_exclusions': ['raw audio archive/WAVs', 'model weights and resume checkpoints', 'AST feature arrays', 'large pretrained and metric runtime assets'],
              'retained_binary_evidence': 'Kept at the campaign root; original DONE and asset/data manifests retain their authoritative hashes.'}
    freeze(root / 'RESULTS.json', result)
    payloads = review_payloads(root)
    manifest = {'schema': 1, 'study': STUDY, 'files': {name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for name, p in sorted(payloads.items())}}
    manifest_bytes = encoded(manifest)
    temporary = directory / (ARCHIVE + '.part'); archive = directory / ARCHIVE
    require(not temporary.exists() and not archive.exists(), 'Unreceipted review ZIP/partial exists; preserve it and inspect before retrying packaging')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zipped:
        for name, path in sorted(payloads.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with path.open('rb') as source, zipped.open(info, 'w', force_zip64=True) as target:
                for block in iter(lambda: source.read(4 << 20), b''):
                    target.write(block)
        zipped.writestr('PAYLOAD_MANIFEST.json', manifest_bytes)
    os.replace(temporary, archive)
    receipt = {'schema': 1, 'status': 'VERIFIED', 'archive': ARCHIVE, 'archive_bytes': archive.stat().st_size,
               'archive_sha256': sha(archive), 'payload_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
               'input_binding': analysis['input_binding'], 'method_runs': 58}
    (directory / (ARCHIVE + '.sha256')).write_text(receipt['archive_sha256'] + '  ' + ARCHIVE + '\n')
    put(directory / 'ARCHIVE_RECEIPT.json', receipt)
    return verify_review(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--python', type=Path, default=DEFAULT_PYTHON)
    parser.add_argument('--wait-hours', type=float, default=72., help='Initial readiness deadline only; capped at 72 hours, never an outcome stopping rule')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--start', action='store_true'); modes.add_argument('--status', action='store_true')
    modes.add_argument('--worker', action='store_true')
    args = parser.parse_args(); args.root = args.root.expanduser().absolute(); args.python = args.python.expanduser().absolute()
    require(0 < args.wait_hours <= 72, '--wait-hours must be >0 and <=72')
    require(not args.root.is_symlink(), 'Refusing symlink root')
    if args.status:
        print(json.dumps(status(args.root), indent=2, sort_keys=True)); return 0
    args.root.mkdir(parents=True, exist_ok=True)
    if args.start:
        with lock(args.root / '.coordinator.start.lock', blocking=True):
            current = status(args.root)
            if current['worker_active'] or current['live_children']:
                print(json.dumps(current, indent=2, sort_keys=True)); return 0
            if current.get('status') == 'COMPLETE':
                verify_review(args.root / 'review')
                print(json.dumps(current, indent=2, sort_keys=True)); return 0
            command = [str(args.python), '-B', str(HERE / 'coordinator.py'), '--root', str(args.root),
                       '--python', str(args.python), '--wait-hours', str(args.wait_hours), '--worker']
            log = args.root / 'logs/coordinator.log'; log.parent.mkdir(exist_ok=True)
            with log.open('ab', buffering=0) as output:
                process = subprocess.Popen(command, cwd=HERE, env=child_environment(args.root),
                                           stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            ident = identity(process.pid)
            require(ident is not None, 'Detached worker did not stay alive; inspect ' + str(log))
            put(args.root / 'PID.json', {'pid': process.pid, 'identity': ident, 'started_at': now(), 'command': command})
            print(json.dumps({'status': 'STARTED', 'pid': process.pid, 'log': str(log), 'finite_method_runs': 58}, indent=2)); return 0
    coordinator = Coordinator(args)
    try:
        coordinator.execute(); return 0
    except AlreadyOwned:
        print(json.dumps(status(args.root), indent=2, sort_keys=True)); return 0
    except BaseException as exc:
        coordinator.update('BLOCKED', reason=type(exc).__name__ + ': ' + str(exc),
                           live_child_pids=[row['pid'] for row in coordinator.children if alive(row)],
                           evidence_deleted=False)
        traceback.print_exc(); return 2


if __name__ == '__main__':
    raise SystemExit(main())
