#!/usr/bin/env python3
"""Verify the public CIFAR-100 release and reconstruct its frozen statistics.

This is CPU-only released-record arithmetic. It never opens raw datasets,
prediction arrays or checkpoints and never launches training/inference.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_RESULTS = REPO / 'results/caenl-cifar100-focused-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(actual, expected, path='analysis'):
    if isinstance(actual, dict):
        require(isinstance(expected, dict) and set(actual) == set(expected), 'Keys differ: ' + path)
        for key in actual:
            compare(actual[key], expected[key], path + '.' + key)
    elif isinstance(actual, list):
        require(isinstance(expected, list) and len(actual) == len(expected), 'Lengths differ: ' + path)
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, path + '[' + str(index) + ']')
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool):
        require(isinstance(expected, (int, float)) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), 'Numeric value differs: ' + path)
    else:
        require(actual == expected, 'Value differs: ' + path)


def run(args):
    result = args.results.resolve()
    output = args.output.resolve()
    require(not output.exists(), 'Output must be a new directory')
    checks = []
    sums = load(result / 'SHA256SUMS.json')
    require(isinstance(sums, dict) and sums, 'Missing public checksum map')
    expected_paths = set(sums)
    actual_paths = {str(p.relative_to(result)) for p in result.rglob('*') if p.is_file() and p.name != 'SHA256SUMS.json'}
    require(expected_paths == actual_paths, 'Public numeric inventory changed')
    for name, expected in sums.items():
        path = (result / name).resolve()
        require(result in path.parents and sha(path) == expected, 'Public file hash mismatch: ' + name)
    checks.append({'public_numeric_payload_hashes': len(sums)})

    frozen = REPO / 'experiments/caenl_cifar100_focused_confirmation_v1/source'
    require(sha(frozen / 'SOURCE_MANIFEST.json') == '2d920804e8ccd7b5e7d133afe5db88407e1783ead5c0e69c9d4c984a3ce81aba', 'Original source manifest changed')
    manifest = load(frozen / 'SOURCE_MANIFEST.json')['files']
    files = {str(p.relative_to(frozen)) for p in frozen.rglob('*') if p.is_file() and p.name != 'SOURCE_MANIFEST.json' and '__pycache__' not in p.parts}
    require(set(manifest) == files and len(files) == 20, 'Frozen source inventory changed')
    for name, expected in manifest.items():
        require(sha(frozen / name) == expected, 'Original source bytes changed: ' + name)
    checks.append({'exact_frozen_source_files': len(files)})

    import analyze
    import independent_stats_check
    data = load(result / 'RESULTS_INPUT.json')
    protocol_raw = (result / 'LOCKED_ANALYSIS_SPEC.json').read_bytes()
    protocol = json.loads(protocol_raw)
    protocol_sha = hashlib.sha256(protocol_raw).hexdigest()
    release = load(result / 'TEST_RELEASE_RECEIPT.json')
    require(sha(result / 'TEST_RELEASE_RECEIPT.json') == protocol['test_release_receipt_sha256'], 'Explicit release receipt binding changed')
    require(release['prior_official_test_accessed'] is True, 'Prior metadata-only test access must be retained')
    require(len(release['models']) == 160, 'Release model matrix incomplete')
    for record in data['records']:
        job_id = 's%d_n%d_%s' % (record['seed'], record['label_count'], record['method'])
        receipt_path = result / 'models' / job_id / 'RECEIPT.json'
        receipt = load(receipt_path)
        bound = release['models'][job_id]
        require(sha(receipt_path) == record['receipt_sha256'], 'Original test receipt changed: ' + job_id)
        require(receipt['checkpoint_sha256'] == bound['checkpoint_sha256'] and receipt['training_receipt_sha256'] == bound['training_receipt_sha256'], 'Checkpoint/training metadata binding changed: ' + job_id)
        require(receipt['count'] == 10000 and receipt['accuracy_percent'] == 100 * receipt['correct'] / 10000 == record['accuracy_percent'], 'Saved numerator/score mismatch: ' + job_id)
        require(receipt['test_release_receipt_sha256'] == protocol['test_release_receipt_sha256'] and receipt['protocol_sha256'] == protocol_sha, 'Test receipt release/analysis binding mismatch: ' + job_id)
    checks.append({'complete_bound_test_receipts_and_saved_numerators': 160, 'prediction_arrays_replayed': False})

    development = load(result / 'development/RESULTS.json')
    development_rows = 0
    for method, candidates in development['all_candidate_scores'].items():
        for candidate in candidates:
            scores = list(candidate['seed_scores'].values())
            require(len(scores) == 3, 'Development seed coverage changed')
            require(math.isclose(sum(scores) / 3, candidate['mean_final_development_accuracy_percent'], abs_tol=1e-10), 'Development mean changed')
            development_rows += len(scores)
        selected = sorted(candidates, key=lambda c: (-c['mean_final_development_accuracy_percent'], c['total_seconds'], c['candidate_index']))[0]
        require(selected['candidate_index'] == development['selected_configurations'][method]['candidate_index'], 'Development selection changed: ' + method)
    require(development_rows == development['completed_treatments'] == 54, 'Incomplete development scores')
    checks.append({'retained_development_scores': development_rows})

    training = load(result / 'TRAINING_RECORDS.json')
    receipts = training['records']
    require(len(receipts) == 200, 'Training stage coverage changed')
    require(sum(r['kind'] == 'treatment' for r in receipts) == 160 and sum(r['kind'] == 'shared_dcr_prefix' for r in receipts) == 40, 'Training stage family changed')
    require(sum(r['training']['updates'] for r in receipts) == 4674150 and sum(r['training']['image_views'] for r in receipts) == 467415000, 'Physical training totals changed')
    by_job = {r['job_id']: r for r in receipts}
    for job_id, binding in release['models'].items():
        require(binding['training_receipt_sha256'] == training['original_receipt_sha256'][job_id] and binding['checkpoint_sha256'] == by_job[job_id]['checkpoint_sha256'], 'Normalized training record binding mismatch: ' + job_id)
    checks.append({'training_stage_objects': 200, 'physical_optimizer_updates': 4674150, 'original_training_receipt_bytes_distributed': False})

    actual = analyze.read_and_analyze(result / 'RESULTS_INPUT.json', result / 'LOCKED_ANALYSIS_SPEC.json')
    expected = load(result / 'ANALYSIS.json')
    compare(actual, expected)
    independent = independent_stats_check.check(data, protocol, expected, protocol_raw)
    checks.append(independent)
    output.mkdir(parents=True)
    (output / 'ANALYSIS.json').write_text(json.dumps(actual, indent=2, sort_keys=True, allow_nan=False) + '\n')
    (output / 'INDEPENDENT_STATS.json').write_text(json.dumps(independent, indent=2) + '\n')
    command = [sys.executable, '-B', str(HERE / 'plot_and_tables.py'), '--input', str(result / 'RESULTS_INPUT.json'), '--analysis', str(result / 'ANALYSIS.json'), '--output', str(output / 'exports')]
    subprocess.run(command, check=True)
    report = {'status': 'PUBLIC_CIFAR100_RECONSTRUCTION_VERIFIED', 'checks': checks,
              'primary_family_size': 3, 'paired_seeds': 10, 'models': 160,
              'scope': 'Released numeric/hash declarations and frozen score-level statistics; no raw data, checkpoint or prediction regeneration.',
              'source_sha256': sha(__file__)}
    (output / 'VERIFICATION.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
