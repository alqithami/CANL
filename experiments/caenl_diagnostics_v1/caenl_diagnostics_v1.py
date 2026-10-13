#!/usr/bin/env python3
"""Finite, separately recorded checkpoint and deployment diagnostics for CÆNL."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import zipfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
DEFAULT_CAMPAIGN = Path('/mnt/caenl/active/results/caenl-solidification-v1')
DEFAULT_OUTPUT = Path('/mnt/caenl/active/results/caenl-diagnostics-v1')
SOURCE_DEFAULT = Path.home() / 'CAENL_solidification_v1'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def verify_files(directory, manifest):
    directory = Path(directory)
    for rel, expected in manifest.items():
        p = directory / rel
        require(not Path(rel).is_absolute() and '..' not in Path(rel).parts,
                'Unsafe source manifest entry')
        require(p.is_file() and not p.is_symlink() and sha(p) == expected,
                'Source mismatch: ' + str(p))


def verify_source():
    manifest = read(HERE / 'SOURCE_MANIFEST.json')
    verify_files(HERE, manifest)
    return manifest


def process_active(root, name):
    p = Path(root) / 'PID.json'
    if not p.exists():
        return False
    try:
        value = (Path('/proc') / str(read(p)['pid']) / 'cmdline').read_bytes()
        return b'--worker' in value and name.encode() in value
    except (OSError, ValueError, KeyError):
        return False


def resolve_locations(args):
    saved = read(HERE / 'RUN_LOCATION.json') if (HERE / 'RUN_LOCATION.json').exists() else {}
    source = (args.source or Path(saved.get('source', SOURCE_DEFAULT))).resolve()
    source_record = source / 'RUN_LOCATION.json'
    remembered_campaign = read(source_record)['root'] if source_record.exists() else DEFAULT_CAMPAIGN
    campaign = (args.campaign or Path(saved.get('campaign', remembered_campaign))).resolve()
    output = (args.output or Path(saved.get('output', DEFAULT_OUTPUT))).resolve()
    for protected in [source, campaign, HERE, Path('/mnt/caenl/active/results/caenl-feedback-study-v1')]:
        require(output != protected and output not in protected.parents and protected not in output.parents,
                'Diagnostics output must be separate from source and existing experiment trees')
    current = dict(source=str(source), campaign=str(campaign), output=str(output))
    if saved:
        require(saved == current, 'This installed release already uses different locations')
    return source, campaign, output


def choose_python(campaign):
    identity = Path(campaign) / 'INPUT_BINDING.json'
    require(identity.is_file(), 'Current campaign INPUT_BINDING.json is unavailable')
    info = read(identity)['environment']
    expected = info['original_environment']['versions']
    candidates = [Path.home() / 'caenl-v5.4.3/caenl_revision_pipeline_v5.4.3_c4_confirmatory/.venv/bin/python',
                  Path.home() / 'caenl-src/caenl_revision_pipeline_v5.3_ibm_rhel_l40s/.venv/bin/python',
                  Path(sys.executable)]
    probe = "import json,importlib.metadata,importlib.util;print(json.dumps({'versions':{n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','scipy','Pillow')},'autoattack_available':importlib.util.find_spec('autoattack') is not None}))"
    for p in candidates:
        if not p.is_file():
            continue
        q = subprocess.run([str(p), '-B', '-c', probe], capture_output=True, text=True)
        if q.returncode == 0:
            found = json.loads(q.stdout)
            if found['versions'] == expected and found['autoattack_available']:
                return p
    raise RuntimeError('Original package environment with AutoAttack unavailable. No package is installed or changed.')


def readiness(campaign):
    """Pure read-only completion gate; the queued worker never launches training."""
    campaign = Path(campaign)
    state = read(campaign / 'STATUS.json') if (campaign / 'STATUS.json').exists() else {}
    active = process_active(campaign, 'caenl_solidification_v1.py')
    if state.get('status') == 'BLOCKED':
        raise RuntimeError('Solidification is BLOCKED: ' + str(state.get('message', 'inspect its log')))
    if active:
        return False, 'WAITING_FOR_SOLIDIFICATION'
    if (campaign / 'COMPLETE.json').exists():
        c = read(campaign / 'COMPLETE.json')
        require(c.get('status') == 'COMPLETE' and c.get('method_runs') == 206,
                'Unexpected completed campaign counts')
        require((campaign / 'RESULTS.json').exists(), 'Completed campaign RESULTS.json missing')
        return True, 'READY'
    if state.get('status') == 'COMPLETE':
        return False, 'WAITING_FOR_COMPLETION_RECORD'
    raise RuntimeError('Solidification has no active worker and no valid COMPLETE.json; inspect it before resuming')


def gpu_pids():
    q = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, check=True)
    return {int(line.strip()) for line in q.stdout.splitlines() if line.strip().isdigit()}


def start(args, queue):
    verify_source()
    source, campaign, output = resolve_locations(args)
    require((source / 'SOURCE_MANIFEST.json').is_file(), 'Frozen solidification source is missing')
    require(read(source / 'SOURCE_MANIFEST.json') == read(HERE / 'CAMPAIGN_SOURCE_MANIFEST.json'),
            'Solidification source differs from the source this diagnostic release was tested against')
    verify_files(source, read(HERE / 'CAMPAIGN_SOURCE_MANIFEST.json'))
    if not queue:
        ready, message = readiness(campaign)
        require(ready, message + '; use --queue to wait without taking GPU memory')
    python = choose_python(campaign)
    output.mkdir(parents=True, exist_ok=True)
    if process_active(output, Path(__file__).name):
        print('Diagnostics worker already active. Use --status.')
        return 0
    if (output / 'COMPLETE.json').exists():
        c = read(output / 'COMPLETE.json')
        archive = output / 'review/CAENL_DIAGNOSTICS_REVIEW.zip'
        require(archive.is_file() and sha(archive) == c['review_sha256'], 'Completed review archive changed')
        print('Already complete. Share: ' + str(archive))
        return 0
    locations = dict(source=str(source), campaign=str(campaign), output=str(output))
    atomic(HERE / 'RUN_LOCATION.json', locations)
    if not (output / 'QUEUE.json').exists():
        days = read(HERE / 'protocol.json')['queue_deadline_days']
        atomic(output / 'QUEUE.json', dict(created=time.time(), deadline=time.time() + days * 86400,
                                         queue_requested=queue))
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    command = [str(python), '-B', str(Path(__file__).resolve()), '--worker',
               '--source', str(source), '--campaign', str(campaign), '--output', str(output)]
    atomic(output / 'STATUS.json', dict(status='QUEUED' if queue else 'STARTING', time=time.time()))
    with (output / 'run.log').open('a', buffering=1) as log:
        p = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True, env=env)
    atomic(output / 'PID.json', dict(pid=p.pid))
    print('QUEUED: waits for verified solidification completion and an idle GPU.' if queue else 'STARTED: completed-model diagnostics.')
    print('No new scientific training campaign. 30 saved ResNet-50 models, 150 standard attack suites, short cost benchmark.')
    print('You may disconnect. Check: python3 -B ' + str(HERE / Path(__file__).name) + ' --status')
    print('Log: ' + str(output / 'run.log'))
    return 0


def make_context(source, campaign, output, progress):
    """Import CUDA dependencies only after both completion and idle-GPU gates."""
    import numpy as np
    import torch
    from common import digest
    sys.path.insert(0, str(source / 'engine'))
    from support import Catalog, Model, tensor_digest
    from runner import verify_method
    cfg = read(campaign / 'DERIVED_PROTOCOL.json')
    protocol = read(HERE / 'protocol.json')
    manifest = read(source / 'SOURCE_MANIFEST.json')
    verify_files(source, manifest)
    require(manifest == read(campaign / 'SOURCE_MANIFEST.json') == read(HERE / 'CAMPAIGN_SOURCE_MANIFEST.json'),
            'Frozen campaign/source identity differs')
    complete = read(campaign / 'COMPLETE.json')
    review_path = campaign / 'review/CAENL_SOLIDIFICATION_REVIEW.zip'
    require(sha(review_path) == complete['review_sha256'],
            'Solidification review archive hash mismatch')
    with zipfile.ZipFile(review_path) as saved_review:
        for rel in ['SOURCE_MANIFEST.json', 'DERIVED_PROTOCOL.json', 'INPUT_BINDING.json', 'SELECTION.json', 'RESULTS.json']:
            require(hashlib.sha256(saved_review.read(rel)).hexdigest() == sha(campaign / rel),
                    'Current campaign record differs from completed review: ' + rel)
    results = read(campaign / 'RESULTS.json')
    require(results['status'] == 'COMPLETE' and results['completed_methods'] == 206 and
            results['analysis']['status'] == 'COMPLETE', 'Solidification results are incomplete')
    require(torch.cuda.is_available(), 'CUDA unavailable')
    expected = read(campaign / 'INPUT_BINDING.json')['environment']
    require(torch.cuda.get_device_name(0) == expected['gpu'] and torch.version.cuda == expected['cuda'],
            'GPU/CUDA identity changed')
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    raw = Catalog(read(source / 'engine/protocol.json'))
    audit = campaign / 'data_audit'
    for rel, h in read(audit / 'DONE.json')['payload'].items():
        require(sha(audit / rel) == h, 'Cleaned manifest changed: ' + rel)
    retained = read(audit / 'retained.json')
    rows = {'train': [], 'val': []}
    for r in retained:
        original = raw.rows[r['split']][r['original_index']]
        require(original['path'] == r['path'] and original['class_id'] == r['class_id'] and
                original['sha256'] == r['file_sha256'], 'Retained mapping changed')
        rows[r['split']].append(dict(original))
    cat = SimpleNamespace(root=raw.root, rows=rows,
                          labels=np.asarray([r['class_id'] for r in rows['train']], dtype=np.int64),
                          val_labels=np.asarray([r['class_id'] for r in rows['val']], dtype=np.int64))
    require(len(cat.labels) == cfg['expected_train_count'] and len(cat.val_labels) == cfg['expected_val_count'],
            'Derived cleaned counts differ')
    progress('Preflight: verifying bytes of all retained validation images')
    for i, row in enumerate(cat.rows['val']):
        require(sha(cat.root / row['path']) == row['sha256'], 'Validation image changed: ' + row['path'])
        if (i + 1) % 1000 == 0:
            progress('Preflight: verified validation images ' + str(i + 1))
    records = []
    expected_rows = {(r['stage'], r['seed'], r['method']): r for r in results['results']}
    for seed in protocol['seeds']:
        with zipfile.ZipFile(review_path) as saved_review:
            for rel in ['confirmation/seed' + str(seed) + '/targets.json',
                        'confirmation/seed' + str(seed) + '/splits.npz']:
                require(hashlib.sha256(saved_review.read(rel)).hexdigest() == sha(campaign / rel),
                        'Seed input differs from completed review: ' + rel)
        for method in protocol['methods']:
            p = campaign / protocol['stage'] / ('seed' + str(seed)) / method
            require(verify_method(p), 'Unverified method: ' + str(p))
            summary = read(p / 'summary.json')
            require(summary == expected_rows[('confirmation', seed, method)],
                    'Method summary differs from the completed numerical record')
            require(summary['seed'] == seed and summary['method'] == method and summary['backbone'] == 'resnet50',
                    'Checkpoint identity mismatch')
            records.append(dict(stage=protocol['stage'], seed=seed, method=method,
                                path=p / 'FINAL_MODEL.pt', sha256=sha(p / 'FINAL_MODEL.pt'),
                                tensor_sha256=summary['rounds'][-1]['final_model_weights_sha256']))
    require(len(records) == 30, 'Expected exactly 30 specified checkpoints')
    from autoattack import AutoAttack
    import autoattack
    module_dir = Path(autoattack.__file__).resolve().parent
    aa_source = {p.relative_to(module_dir).as_posix(): sha(p) for p in sorted(module_dir.rglob('*.py'))}
    identity = dict(campaign_binding=complete['binding'], campaign_complete_sha256=sha(campaign / 'COMPLETE.json'),
                    protocol=protocol, source=read(HERE / 'SOURCE_MANIFEST.json'),
                    retained_manifest_sha256=sha(audit / 'retained.json'),
                    checkpoint_hashes=[dict(seed=r['seed'], method=r['method'], sha256=r['sha256']) for r in records],
                    runtime=dict(python=sys.version, torch=str(torch.__version__), cuda=torch.version.cuda,
                                 gpu=torch.cuda.get_device_name(0), autoattack_source=aa_source))
    binding = digest(identity)
    identity['binding'] = binding
    if (output / 'INPUT_BINDING.json').exists():
        require(read(output / 'INPUT_BINDING.json') == identity, 'Follow-up input/source/runtime changed on resume')
    else:
        atomic(output / 'INPUT_BINDING.json', identity)
    def load_model(record):
        require(not (gpu_pids() - {os.getpid()}), 'Another GPU process appeared; no other job was stopped')
        require(sha(record['path']) == record['sha256'], 'Checkpoint changed before load')
        saved = torch.load(record['path'], map_location='cpu', weights_only=False)
        require(saved['seed'] == record['seed'] and saved['method'] == record['method'] and
                tensor_digest(saved['model']) == record['tensor_sha256'], 'Saved model tensors differ')
        model = Model(cfg['num_classes'], backbone='resnet50').to('cuda:0').float().to(memory_format=torch.channels_last)
        model.load_state_dict(saved['model'], strict=True)
        model.eval()
        return model
    return SimpleNamespace(cfg=cfg, protocol=protocol, cat=cat, root=output, source_root=source,
                           campaign_root=campaign, binding=binding, device='cuda:0', records=records,
                           progress=progress, load_model=load_model)


def gpu_contract(ctx):
    """Real model/image forward, input gradient and loss-gradient parity before long attacks."""
    import numpy as np
    import torch
    from support import loader, evaluation_batches, to_device
    from aligned import aligned_loss
    from cost import aligned_tensors
    p = ctx.root / 'GPU_CONTRACT.json'
    if p.exists():
        require(read(p)['binding'] == ctx.binding and read(p)['status'] == 'PASS', 'GPU contract binding changed')
        return
    model = ctx.load_model(ctx.records[0])
    batch = next(iter(loader(ctx.cat, 'val', evaluation_batches(np.arange(2), 2), 0, 0, False)))
    x, y, _ = to_device(batch, ctx.device)
    x.requires_grad_(True)
    logits, features = model.forward_features(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    g = torch.autograd.grad(loss, x)[0]
    require(torch.isfinite(logits).all() and torch.isfinite(g).all(), 'Nonfinite real-model contract')
    # A balanced synthetic feature fixture tests the optimized fixed-loss gradient
    # without training or altering a saved model.
    labels = torch.arange(4, device=ctx.device).repeat_interleave(3)
    z = {l: torch.randn(12, 16, device=ctx.device, requires_grad=True) for l in ctx.cfg['layers']}
    targets = {l: dict(log_kappa_target=-2., between_floor=.25) for l in z}
    weights = {l: .03 for l in z}
    a = aligned_loss(z, labels, targets, weights, ctx.cfg['alignment'])[0]
    b = aligned_tensors(z, labels, targets, weights, ctx.cfg['alignment'], instrumented=False)[0]
    ga = torch.autograd.grad(a, tuple(z.values()), retain_graph=True)
    gb = torch.autograd.grad(b, tuple(z.values()))
    require(torch.allclose(a, b, atol=1e-6, rtol=1e-5) and
            all(torch.allclose(u, v, atol=1e-6, rtol=1e-5) for u, v in zip(ga, gb)),
            'Streamlined loss/gradient differs from original')
    atomic(p, dict(status='PASS', binding=ctx.binding, real_model_input_gradient_finite=True,
                   original_and_streamlined_loss_gradient_match=True,
                   scope='Contract only; full standard attack validation occurs per bounded chunk'))
    del model, x, y, logits, features, g, a, b, ga, gb, z
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def package(output):
    review = output / 'review'
    review.mkdir(exist_ok=True)
    archive = review / 'CAENL_DIAGNOSTICS_REVIEW.zip'
    metadata = [p for p in output.rglob('*') if p.is_file() and review not in p.parents and
                p.name not in ['PID.json', 'RUN.lock', 'COMPLETE.json'] and
                p.suffix in ['.json', '.txt', '.md', '.csv', '.log']]
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        payload = {}
        for p in sorted(metadata):
            rel = p.relative_to(output).as_posix()
            z.write(p, rel)
            payload[rel] = sha(p)
        for name in ['PROTOCOL.md', 'protocol.json', 'SOURCE_MANIFEST.json', 'LOCAL_VALIDATION.json']:
            p = HERE / name
            if p.exists():
                z.write(p, 'release/' + name)
                payload['release/' + name] = sha(p)
        z.writestr('REVIEW_MANIFEST.json', json.dumps(payload, indent=2) + '\n')
    archive.with_suffix('.zip.sha256').write_text(sha(archive) + '  ' + archive.name + '\n')
    files = sorted(p for p in output.rglob('*.npz') if review not in p.parents)
    groups, group, size = [], [], 0
    for p in files:
        if group and size + p.stat().st_size > 24 * 2 ** 20:
            groups.append(group)
            group, size = [], 0
        require(p.stat().st_size < 29 * 2 ** 20, 'Individual numeric file too large to package')
        group.append(p)
        size += p.stat().st_size
    if group:
        groups.append(group)
    parts = []
    for i, group in enumerate(groups, 1):
        pz = review / ('CAENL_DIAGNOSTICS_EVIDENCE_part_%03d.zip' % i)
        with zipfile.ZipFile(pz, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            payload = {}
            for p in group:
                rel = p.relative_to(output).as_posix()
                z.write(p, rel)
                payload[rel] = sha(p)
            z.writestr('PART_MANIFEST.json', json.dumps(payload, indent=2) + '\n')
        require(pz.stat().st_size < 30 * 2 ** 20, 'Evidence part exceeds upload boundary')
        parts.append(dict(name=pz.name, bytes=pz.stat().st_size, sha256=sha(pz)))
    atomic(review / 'EVIDENCE_INDEX.json', dict(parts=parts))
    (review / 'EVIDENCE_SHA256SUMS').write_text(''.join(r['sha256'] + '  ' + r['name'] + '\n' for r in parts))
    return archive, parts


def worker(args):
    source, campaign, output = resolve_locations(args)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'RUN.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Another diagnostics worker holds the lock.')
        return 1
    started = time.time()
    def progress(message, status='RUNNING'):
        atomic(output / 'STATUS.json', dict(status=status, pid=os.getpid(), message=message, time=time.time()))
        print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), message, flush=True)
    try:
        verify_source()
        deadline = read(output / 'QUEUE.json')['deadline']
        while True:
            require(time.time() < deadline, 'Finite queue deadline reached. Existing experiments and outputs preserved.')
            ready, state = readiness(campaign)
            if ready:
                others = gpu_pids() - {os.getpid()}
                if not others:
                    break
                state = 'WAITING_FOR_IDLE_GPU'
            progress(state + ': no CUDA allocation or model loading while waiting', status='QUEUED')
            time.sleep(30)
        require(shutil.disk_usage(output).free >= 8 * 2 ** 30, 'Need 8 GiB free for separately retained diagnostic records')
        progress('Preflight: CPU contracts in the original environment')
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ''
        subprocess.run([sys.executable, '-B', str(HERE / Path(__file__).name), '--self-test', '--source', str(source)],
                       env=env, check=True)
        require(not (gpu_pids() - {os.getpid()}), 'GPU became busy during CPU preflight; use --queue after inspecting status')
        ctx = make_context(source, campaign, output, progress)
        progress('GPU contract: original model input gradients and streamlined loss parity')
        gpu_contract(ctx)
        from cost import run as cost_run
        from robust import run as robust_run
        progress('Practical-cost microbenchmark: transient updates only, saved models stay unchanged')
        cost = cost_run(ctx)
        progress('Robustness/geometry: 30 checkpoints x 5 radii, full standard AutoAttack')
        robustness = robust_run(ctx)
        for record in ctx.records:
            require(sha(record['path']) == record['sha256'], 'Original checkpoint changed during diagnostics')
        atomic(output / 'RESULTS.json', dict(status='COMPLETE', binding=ctx.binding,
                                           cost=cost, robustness=robustness))
        (output / 'REPORT.txt').write_text(
            'CAENL CHECKPOINT AND PRACTICAL-COST DIAGNOSTICS\n'
            'Completed 30 saved ResNet-50 models at five specified attack radii.\n'
            'Primary endpoint: robust top-1 at 2/255; three separate Holm-adjusted contrasts.\n'
            'Cost data are short cached-data microbenchmarks, not whole-job estimates.\n'
            'Geometry/sensitivity measurements are diagnostics, not certificates or causal mediation.\n'
            'No original trained model, protocol or result was modified.\n'
            'Inspect RESULTS.json and robustness/cost summaries before changing scientific claims.\n')
        progress('All diagnostics completed; packaging compact review and numeric evidence')
        archive, parts = package(output)
        atomic(output / 'COMPLETE.json', dict(status='COMPLETE', binding=ctx.binding,
                                            models=30, attack_suites=150, evidence_parts=len(parts),
                                            seconds_this_invocation=time.time() - started,
                                            review_sha256=sha(archive)))
        atomic(output / 'STATUS.json', dict(status='COMPLETE', worker_active=False,
                                          models=30, attack_suites=150, time=time.time()))
        print('CAENL_DIAGNOSTICS_COMPLETE\nShare: ' + str(archive), flush=True)
        return 0
    except Exception as exc:
        atomic(output / 'STATUS.json', dict(status='BLOCKED', message=str(exc), time=time.time(), worker_active=False))
        print(traceback.format_exc(), flush=True)
        return 2


def status(args):
    _, campaign, output = resolve_locations(args)
    state = read(output / 'STATUS.json') if (output / 'STATUS.json').exists() else dict(status='NOT_STARTED')
    state['worker_active'] = process_active(output, Path(__file__).name)
    state['solidification_worker_active'] = process_active(campaign, 'caenl_solidification_v1.py')
    print(json.dumps(state, indent=2))
    print('Diagnostics endpoint: 30 models; 150 bounded attack suites; short cost benchmark.')
    cost_count = len(list((output / 'cost').glob('repetition_*.json')))
    radius_count = len(list((output / 'robustness').glob('seed*/*/linf_*/DONE.json')))
    print('Cost repetitions: ' + str(cost_count) + '/30; attack suites: ' + str(radius_count) + '/150')
    print('Review: ' + str(output / 'review/CAENL_DIAGNOSTICS_REVIEW.zip'))
    print('Log: ' + str(output / 'run.log'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path)
    p.add_argument('--campaign', type=Path)
    p.add_argument('--output', type=Path)
    g = p.add_mutually_exclusive_group(required=True)
    for name in ['queue', 'start', 'status', 'worker', 'self-test']:
        g.add_argument('--' + name, action='store_true')
    args = p.parse_args()
    if args.queue or args.start:
        return start(args, args.queue)
    if args.worker:
        return worker(args)
    if args.status:
        status(args)
        return 0
    # No runtime/source state is required for these bounded CPU fixtures.
    source = args.source or SOURCE_DEFAULT
    require((source / 'engine/aligned.py').is_file(), 'Self-test requires the frozen source engine via --source')
    sys.path.insert(0, str(source / 'engine'))
    import unittest
    suite = unittest.defaultTestLoader.discover(str(HERE), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 2


if __name__ == '__main__':
    raise SystemExit(main())
