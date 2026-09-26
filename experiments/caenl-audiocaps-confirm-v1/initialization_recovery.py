#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded recovery for CAENL AudioCaps initialization, 23 September 2026.


This is an OUTER SCHEDULING wrapper, not a model or hash-check patch.
The original confirmation runner, frozen protocol, initialization records,
source, weights and finished jobs stay unchanged. Original job subprocesses
are retried only when their exact initialization guard fails during setup,
with no checkpoint, no training metrics, and no summary for that job.
All retries use the original command and same seed. At most eight attempts
per unstarted job per recovery launch. Never retry a training/metric failure.
A run can reach training only by passing the ORIGINAL whole-model hash check.
No tensor replacement, numeric tolerance, dependency installation or download.
The lower-level reason for the observed encoder.pos variation is not proved.


The wrapper is source-reviewed but NOT executed in the authoring runtime.
Safety-classifier tests and existing-artifact verification run on IBM.
"""
from __future__ import annotations
import argparse
import ast
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback


sys.dont_write_bytecode = True
NAME = 'caenl-audiocaps-confirm-v1'
ROOT = Path('/mnt/caenl/active/results') / NAME
REC = ROOT / 'initialization_recovery_v1'
PYTHON = Path.home() / 'caenl-v5.4.3/caenl_revision_pipeline_v5.4.3_c4_confirmatory/.venv/bin/python'
METHODS = ('baseline', 'fixed_dcr', 'macc_lite', 'full_macc')
SEEDS = tuple(range(1301, 1307))
MAX_ATTEMPTS = 8
EXPECTED_1303 = '985f69ff2d7a47053af46f7b12791050d8f0abf30f3a2714bbb445bec468702e'
CODE_PINS = {
    'engine.py': 'd3d01297c43357b9c34d81701b21f1b80c3a3a395bb65ef341026c61c918ac15',
    'pilot_wrapper.py': '20acc1c0c585041ff5356d22c021d29e270d56f8f82d020d1c1b83714c3af879',
    'SOURCE.json': 'e8a7a3039390e60491431376ec1d28aad07e674fbaf64ca89aaa18e52f6b9451',
}
POLICY = {
    'version': 'bounded_original_setup_retry_v1',
    'max_attempts_per_unstarted_job_per_launch': MAX_ATTEMPTS,
    'retry_condition': 'exact original initialization mismatch in setup, before training or checkpoint creation',
    'same_seed_and_original_job_command': True,
    'original_whole_model_hash_guard_retained': True,
    'original_runner_and_protocol_not_edited': True,
    'initialization_records_not_replaced': True,
    'tensor_values_not_replaced_by_recovery': True,
    'finished_jobs_skipped_by_original_runner': True,
    'checkpoint_or_training_or_evaluation_failure_retries': False,
    'outcome_based_selection': False,
    'scope': 'retry failed unstarted construction, not rerun an unfavorable experiment',
    'lower_level_initialization_nondeterminism_resolved': False,
    'locally_executed_in_authoring_environment': False,
}




def require(ok, message):
    if not bool(ok):
        raise RuntimeError(message)




def now():
    return datetime.now(timezone.utc).isoformat()




def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))




def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()




def checked_path(base, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe relative file path')
    p = Path(base) / relative
    require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(Path(base).resolve()),
            'Unsafe or missing input: ' + str(p))
    return p




def write(path, value):
    path = Path(path)
    require(path.resolve().is_relative_to(REC.resolve()) and not path.is_symlink(), 'Unsafe recovery output')
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        value = value.encode('utf-8')
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.writing_', delete=False) as stream:
        tmp = Path(stream.name)
        stream.write(value); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)




def put(path, value):
    write(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')




def event(kind, **fields):
    record = dict(time=now(), kind=kind, **fields)
    p = REC / 'events.jsonl'
    require(not p.is_symlink(), 'Unsafe recovery event path')
    with p.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())




def live():
    return shutil.which('tmux') is not None and subprocess.run(
        ['tmux', 'has-session', '-t', NAME], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL).returncode == 0




def verify_original():
    protocol = read(checked_path(ROOT, 'PROTOCOL.json'))
    require(sha(checked_path(ROOT, 'runner.py')) == protocol['runner_sha256'],
            'Original runner differs from its frozen protocol')
    for name, expected in CODE_PINS.items():
        require(sha(checked_path(ROOT, name)) == expected, 'Bound helper/source changed: ' + name)
    require(read(ROOT / 'INITIALIZATION_seed1303.json')['model_weights_sha256'] == EXPECTED_1303,
            'Seed 1303 initialization no longer matches the supplied diagnostic')
    return protocol




def load_original():
    verify_original()
    spec = importlib.util.spec_from_file_location('caenl_original_audio_confirmation', ROOT / 'runner.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    require(module.ROOT == ROOT and module.NAME == NAME and module.STEPS == 13890,
            'Unexpected original experiment')
    require(tuple(module.SEEDS) == SEEDS and tuple(module.METHODS) == METHODS, 'Study matrix differs')
    return module




def completed_snapshot():
    """Verify original DONE payloads and same-seed starting hashes, not final-score desirability."""
    files = {}; jobs = []
    for marker in sorted(ROOT.glob('audio/seed*/*/DONE.json')):
        folder = marker.parent
        done = read(marker)
        method, seed = folder.name, int(folder.parent.name.removeprefix('seed'))
        require(method in METHODS and seed in SEEDS and done['status'] == 'COMPLETE', 'Unexpected completed job')
        for relative, expected in done['files'].items():
            p = checked_path(folder, relative)
            require(sha(p) == expected, 'Completed payload checksum differs: ' + str(p))
            files[str(p)] = expected
        files[str(marker)] = sha(marker)
        s = read(folder / 'summary.json')
        initial = ROOT / ('INITIALIZATION_seed' + str(seed) + '.json')
        require(s['seed'] == seed and s['method'] == method and s['steps'] == 13890, 'Completed identity/budget differs')
        require(s['initial_model_weights_sha256'] == read(initial)['model_weights_sha256'],
                'Completed run has inconsistent initialization: ' + str(folder))
        files[str(initial)] = sha(initial)
        jobs.append({'seed': seed, 'method': method})
    require(len(jobs) >= 10, 'Fewer than the ten reported completed jobs were found; stop for review')
    for name in ('runner.py', 'PROTOCOL.json', *CODE_PINS):
        files[str(ROOT / name)] = sha(ROOT / name)
    return {'time': now(), 'completed_jobs': jobs, 'completed_count': len(jobs), 'files': files}




def verify_preserved():
    rec = read(REC / 'PRESERVED_AT_FIRST_LAUNCH.json')
    for name, expected in rec['files'].items():
        p = Path(name)
        require(p.resolve().is_relative_to(ROOT.resolve()) and not p.is_symlink(), 'Unsafe preserved path')
        require(sha(p) == expected, 'A protected pre-recovery file changed: ' + name)
    return rec['completed_count']




def attempt_number(job):
    p = job / 'attempts.jsonl'
    if not p.exists():
        return 0
    rows = [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
    require(all(r.get('attempt') == i + 1 for i, r in enumerate(rows)), 'Unexpected job attempt history')
    return len(rows)




def eligibility(expected_message, state, prior_attempt, has_checkpoint, has_metrics,
                has_result, initial_unchanged, matching_child):
    return (state.get('state') == 'failed'
            and state.get('error') == repr(RuntimeError(expected_message))
            and 'in setup' in str(state.get('traceback', ''))
            and 'in freeze' in str(state.get('traceback', ''))
            and state.get('attempt') == prior_attempt + 1
            and not has_checkpoint and not has_metrics and not has_result
            and initial_unchanged and matching_child)




def safety_tests():
    expected = 'Previously frozen value changed: /example/INITIALIZATION_seed1303.json'
    state = {'state': 'failed', 'error': repr(RuntimeError(expected)),
             'traceback': 'in setup\nin freeze', 'attempt': 2}
    args = [expected, state, 1, False, False, False, True, True]
    require(eligibility(*args), 'Exact unstarted setup-failure fixture rejected')
    for index in (3, 4, 5):
        modified = list(args); modified[index] = True
        require(not eligibility(*modified), 'Checkpoint/training/result fixture was incorrectly retryable')
    for index in (6, 7):
        modified = list(args); modified[index] = False
        require(not eligibility(*modified), 'Changed initialization or unrelated child fixture was retryable')
    for key, value in (('error', "RuntimeError('CUDA out of memory')"), ('attempt', 1), ('traceback', 'in run')):
        modified = list(args); modified[1] = dict(state, **{key: value})
        require(not eligibility(*modified), 'Unrelated failure fixture was retryable')
    return {'status': 'PASS', 'tests': ['exact setup mismatch accepted', 'checkpoint blocks retry',
        'training metrics block retry', 'completed output blocks retry', 'changed frozen hash blocks retry',
        'unrelated child blocks retry', 'other error blocks retry', 'stale attempt blocks retry',
        'wrong failure location blocks retry'], 'training_performed': False, 'execution_host': os.uname().nodename}




def qualified_failure(args, prior_attempt, before_initial_sha):
    if len(args) != 4 or args[0] != '--job' or args[2] != '--seed':
        return None
    method, seed = str(args[1]), int(args[3])
    if method not in METHODS or seed not in SEEDS or before_initial_sha is None:
        return None
    job = ROOT / 'audio' / ('seed' + str(seed)) / method
    init = ROOT / ('INITIALIZATION_seed' + str(seed) + '.json')
    state_path, child_path = job / 'status.json', ROOT / 'CHILD_FAILURE.json'
    if not state_path.is_file() or not child_path.is_file() or not init.is_file():
        return None
    state, child = read(state_path), read(child_path)
    ckpts = job / 'checkpoints'; metrics = job / 'metrics.jsonl'
    has_checkpoint = ckpts.exists() and any(p.is_file() for p in ckpts.rglob('*'))
    has_metrics = metrics.exists() and metrics.stat().st_size > 0
    message = 'Previously frozen value changed: ' + str(init)
    unchanged = sha(init) == before_initial_sha
    matching = child.get('exit_code') == 1 and list(map(str, child.get('arguments', []))) == list(map(str, args))
    allowed = eligibility(message, state, prior_attempt, has_checkpoint, has_metrics,
                          (job / 'DONE.json').exists() or (job / 'summary.json').exists(), unchanged, matching)
    if not allowed:
        return None
    require(attempt_number(job) == prior_attempt + 1, 'Failure attempt history does not match job status')
    return {'arguments': list(map(str, args)), 'job_status': state, 'child_failure': child,
        'frozen_initialization_sha256': before_initial_sha,
        'expected_model_weights_sha256': read(init)['model_weights_sha256'],
        'no_checkpoints': True, 'no_training_metrics': True, 'no_completed_result': True}




def archive_root_failure():
    history = REC / ('history_' + str(time.time_ns()))
    history.mkdir()
    recovery_failure = REC / 'RECOVERY_FAILURE.json'
    if recovery_failure.is_file():
        require(not recovery_failure.is_symlink(), 'Recovery marker symlink refused')
        os.replace(recovery_failure, history / 'RECOVERY_FAILURE.json')
    for name in ('FAILED.json', 'CHILD_FAILURE.json'):
        p = ROOT / name
        if p.is_file():
            require(not p.is_symlink(), 'Failure marker symlink refused')
            os.replace(p, history / name)
    return history




def recovery_worker():
    module = load_original()
    require(sha(REC / 'supervisor.py') == read(REC / 'OWNER.json')['supervisor_sha256'], 'Recovery source changed')
    require(read(REC / 'POLICY.json') == POLICY, 'Recovery policy changed')
    original_child, original_report, original_aggregate = module.child, module.report, module.aggregate
    protected = read(REC / 'PRESERVED_AT_FIRST_LAUNCH.json')
    # Fast immutable checks between attempts; full completed-payload verification occurs at start and final aggregation.
    control_pins = {p: h for p, h in protected['files'].items() if Path(p).parent == ROOT}


    def guarded_child(args, timeout=None):
        if not args or args[0] != '--job':
            return original_child(args, timeout=timeout)
        method, seed = str(args[1]), int(args[3])
        folder = ROOT / 'audio' / ('seed' + str(seed)) / method
        for trial in range(1, MAX_ATTEMPTS + 1):
            verify_original()
            for p, h in control_pins.items():
                require(sha(p) == h, 'Frozen control/initialization record changed')
            init = ROOT / ('INITIALIZATION_seed' + str(seed) + '.json')
            initial_before = sha(init) if init.is_file() else None
            prior = attempt_number(folder)
            try:
                original_child(args, timeout=timeout)
            except RuntimeError:
                reason = qualified_failure(args, prior, initial_before)
                if reason is None:
                    raise
                event('INITIALIZATION_REJECTED', trial=trial, **reason)
                if trial == MAX_ATTEMPTS:
                    print('INITIALIZATION RETRY LIMIT REACHED; original check still enforced.', flush=True)
                    raise
                archive_root_failure()
                print('EXACT-INITIALIZATION RETRY: ' + method + '/seed' + str(seed)
                      + '; same seed; no training occurred in the rejected attempt; attempt '
                      + str(trial + 1) + '/' + str(MAX_ATTEMPTS), flush=True)
                time.sleep(1)
                continue
            done = read(folder / 'DONE.json')
            s = read(folder / 'summary.json')
            require(sha(folder / 'summary.json') == done['files']['summary.json'], 'Successful summary differs from DONE record')
            require(s['initial_model_weights_sha256'] == read(init)['model_weights_sha256'],
                    'Successful run did not retain the authoritative starting hash')
            if initial_before is not None:
                require(sha(init) == initial_before, 'A frozen initialization was replaced')
            event('ORIGINAL_JOB_COMPLETE_OR_ALREADY_COMPLETE', method=method, seed=seed,
                  supervisor_trial=trial, initialization_hash=s['initial_model_weights_sha256'])
            return


    def augmented_report(label):
        original_report(label)
        report_path = ROOT / module.REPORT
        extra = ['\n\n## Initialization recovery provenance\n',
            'An outer scheduler retried only pre-training initialization-guard failures. ',
            'The original runner and full-model equality check were not edited. ',
            'This records a workaround, not a proof that lower-level nondeterminism was eliminated.\n']
        for name in ('POLICY.json', 'SAFETY_TESTS.json', 'LAUNCH.json', 'FINAL_PRESERVATION.json'):
            p = REC / name
            if p.is_file():
                extra += ['\n### ' + name + '\nSHA-256: ' + sha(p) + '\n```json\n', p.read_text(), '\n```\n']
        p = REC / 'events.jsonl'
        if p.is_file():
            extra += ['\n### Recorded setup retries and accepted original jobs\n```jsonl\n', p.read_text(), '\n```\n']
        source = REC / 'supervisor.py'
        extra += ['\n### Exact outer recovery scheduler\nSHA-256: ' + sha(source) + '\n````````\n',
                  source.read_text(encoding='utf-8-sig'), '\n````````\n']
        module.write(report_path, report_path.read_text() + ''.join(extra))
        module.write(ROOT / (module.REPORT + '.sha256'), module.sha(report_path) + '  ' + module.REPORT + '\n')


    def checked_aggregate():
        original_aggregate()
        count = verify_preserved()
        put(REC / 'FINAL_PRESERVATION.json', {'status': 'PASS',
            'pre_recovery_completed_jobs_reverified_unchanged': count,
            'original_runner_protocol_initialization_and_bound_outputs_unchanged': True,
            'model_tensor_values_modified_by_recovery': False, 'time': now()})


    module.child, module.report, module.aggregate = guarded_child, augmented_report, checked_aggregate
    event('RECOVERY_WORKER_STARTED', original_runner_sha256=sha(ROOT / 'runner.py'))
    return module.worker()




def start():
    import fcntl
    require(sys.platform.startswith('linux') and Path('/mnt/caenl').is_mount(), 'Run on IBM with the existing persistent volume')
    require(ROOT.is_dir() and not ROOT.is_symlink() and PYTHON.is_file() and shutil.which('tmux'), 'Expected existing study/Python/tmux missing')
    with (ROOT / 'START.lock').open('rb') as start_lock:
        fcntl.flock(start_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if live():
            print('ALREADY ACTIVE: nothing restarted.'); load_original().status(); return
        if (ROOT / 'COMPLETE.json').is_file():
            print('ALREADY COMPLETE: no training restarted.'); load_original().status(); return
        with (ROOT / 'RUN.lock').open('rb') as run_lock:
            fcntl.flock(run_lock.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            require(not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip(),
                    'Another GPU process is active; it was not stopped')
            verify_original()
            child = read(ROOT / 'CHILD_FAILURE.json')
            args = child.get('arguments', [])
            require(len(args) == 4 and args[0] == '--job' and args[2] == '--seed', 'Current failure is not a supported setup failure')
            method, seed = str(args[1]), int(args[3])
            require(method in METHODS and seed in SEEDS, 'Unexpected failed job identity')
            job = ROOT / 'audio' / ('seed' + str(seed)) / method
            init = ROOT / ('INITIALIZATION_seed' + str(seed) + '.json')
            require(qualified_failure(args, attempt_number(job) - 1, sha(init)) is not None,
                    'Current error is not the exact unstarted initialization mismatch; no retry launched')
            require(not REC.is_symlink(), 'Unsafe recovery folder')
            REC.mkdir(exist_ok=True, mode=0o700)
            raw = Path(__file__).read_bytes(); ast.parse(raw.decode('utf-8-sig'))
            owner = {'supervisor_sha256': hashlib.sha256(raw).hexdigest(), 'policy': POLICY}
            if (REC / 'OWNER.json').is_file():
                require(read(REC / 'OWNER.json') == owner and sha(REC / 'supervisor.py') == owner['supervisor_sha256'],
                        'Different recovery program owns this directory')
            else:
                require(not any(REC.iterdir()), 'Unowned nonempty recovery folder')
                write(REC / 'supervisor.py', raw); put(REC / 'OWNER.json', owner); put(REC / 'POLICY.json', POLICY)
            put(REC / 'SAFETY_TESTS.json', safety_tests())
            print('RECOVERY SAFETY TESTS: PASS', flush=True)
            print('Verifying preserved completed outputs before resuming...', flush=True)
            snapshot = completed_snapshot()
            if (REC / 'PRESERVED_AT_FIRST_LAUNCH.json').is_file():
                verify_preserved()
            else:
                put(REC / 'PRESERVED_AT_FIRST_LAUNCH.json', snapshot)
            launch = {'time': now(), 'completed_jobs_before_launch': snapshot['completed_count'],
                      'original_runner_sha256': sha(ROOT / 'runner.py'),
                      'original_protocol_sha256': sha(ROOT / 'PROTOCOL.json'),
                      'source_reviewed_not_locally_executed': True,
                      'initial_failed_job': list(map(str, args))}
            put(REC / 'LAUNCH.json', launch)
            event('ORIGINAL_FAILURE_PRESERVED', state=read(ROOT / 'FAILED.json'), child=child)
            archive_root_failure()
            print('VERIFIED COMPLETED JOBS:', snapshot['completed_count'], flush=True)
        # Release RUN.lock before the original worker acquires its exclusive lock.
        shell = 'set -e; if [ -f "$HOME/.caenl-java8-env" ]; then source "$HOME/.caenl-java8-env"; fi; '
        shell += 'unset CAENL_DEBUG_STOP_AT_STEP; export PATH=' + shlex.quote(str(PYTHON.parent)) + ':"$PATH"; '
        shell += 'export HF_HOME=' + shlex.quote(str(ROOT / 'assets/hf_home')) + '; '
        shell += 'export TORCH_HOME=' + shlex.quote(str(ROOT / 'assets/torch')) + '; '
        shell += 'export XDG_CACHE_HOME=' + shlex.quote(str(ROOT / 'cache/xdg')) + '; '
        shell += 'exec env PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 '
        shell += 'CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 '
        shell += shlex.quote(str(PYTHON)) + ' ' + shlex.quote(str(REC / 'supervisor.py')) + ' --worker >> '
        shell += shlex.quote(str(ROOT / 'run.log')) + ' 2>&1'
        subprocess.run(['tmux', 'new-session', '-d', '-s', NAME, 'bash -c ' + shlex.quote(shell)], check=True)
        print('STARTED: ' + NAME + ' (guarded initialization recovery)', flush=True)
        print('Original hash checks and completed runs retained. No old pilot/download is restarted.', flush=True)
        print('Status: python3 ~/caenl_audiocaps_init_recovery_v1.py --status', flush=True)




def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--start', action='store_true')
    group.add_argument('--status', action='store_true')
    group.add_argument('--worker', action='store_true')
    args = p.parse_args()
    if args.status:
        load_original().status()
        failure = REC / 'RECOVERY_FAILURE.json'
        if failure.is_file():
            print('RECOVERY SUPERVISOR ERROR:')
            print(json.dumps(read(failure), indent=2))
        return 0
    if args.worker:
        try:
            return recovery_worker()
        except Exception as exc:
            put(REC / 'RECOVERY_FAILURE.json', {'status': 'FAILED', 'time': now(),
                'error': repr(exc), 'traceback': traceback.format_exc(),
                'original_source_and_initialization_records_not_edited': True})
            raise
    start(); return 0




if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print('RECOVERY STOPPED: ' + str(exc), file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(1)
# END OF CAENL AUDIOCAPS INITIALIZATION RECOVERY V1