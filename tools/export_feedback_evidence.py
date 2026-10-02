#!/usr/bin/env python3
"""Export saved feedback-study evidence without importing training code or PyTorch.

Original payload bytes are retained in ordinary ZIP volumes. No experiment file
is written, no checkpoint is unpickled, and no GPU is used. Python >= 3.8.
"""
import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile
import zlib


DEFAULT_ROOT = '/mnt/caenl/active/results/caenl-feedback-study-v1'
DEFAULT_OUTPUT = '/mnt/caenl/active/exports/caenl-feedback-study-v1-evidence'
MIB = 1024 * 1024


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(MIB), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(path):
    s = Path(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def safe_file(base, rel):
    rel = Path(rel)
    require(not rel.is_absolute() and '..' not in rel.parts,
            'Unsafe manifest path: ' + str(rel))
    p = base / rel
    require(p.is_file() and not p.is_symlink(), 'Missing or linked file: ' + str(p))
    require(base.resolve() in p.resolve().parents, 'File outside evidence root: ' + str(p))
    return p


def json_write(path, obj):
    with Path(path).open('w', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def walk_files(root):
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs.sort()
        for name in dirs:
            require(not (Path(directory) / name).is_symlink(),
                    'Linked directory in evidence tree: ' + str(Path(directory) / name))
        for name in sorted(names):
            p = Path(directory) / name
            require(not p.is_symlink(), 'Linked file in evidence tree: ' + str(p))
            if p.is_file():
                yield p


def compressed_size(path, stored):
    if stored:
        return path.stat().st_size
    compressor = zlib.compressobj(6, zlib.DEFLATED, -15)
    size = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(MIB), b''):
            size += len(compressor.compress(block))
    return size + len(compressor.flush())


def verify_export(output, require_complete=True):
    manifest_path = output / 'EXPORT_MANIFEST.json'
    if require_complete:
        complete = read_json(output / 'EXPORT_COMPLETE.json')
        require(complete['manifest_sha256'] == sha256(manifest_path), 'Export manifest changed')
    manifest = read_json(manifest_path)
    seen = set()
    expected = {item['archive_path']: item for item in manifest['exported_files']}
    for archive in manifest['archives']:
        p = safe_file(output, archive['name'])
        require(sha256(p) == archive['sha256'], 'Archive checksum differs: ' + p.name)
        with zipfile.ZipFile(p) as zf:
            for entry in zf.infolist():
                require(entry.filename not in seen, 'Duplicate archived path: ' + entry.filename)
                require(entry.filename in expected, 'Unlisted archived file: ' + entry.filename)
                seen.add(entry.filename)
                h = hashlib.sha256()
                with zf.open(entry) as stream:
                    for block in iter(lambda: stream.read(MIB), b''):
                        h.update(block)
                item = expected[entry.filename]
                require(h.hexdigest() == item['sha256'] and entry.file_size == item['size_bytes'],
                        'Archived payload differs: ' + entry.filename)
        print('Verified ' + p.name, flush=True)
    require(seen == set(expected), 'Some exported files are missing from ZIP volumes')
    print('EVIDENCE_EXPORT_VERIFIED', flush=True)
    return manifest


def export(root, output, part_mib=28, all_arrays=False, verify_checkpoints=False):
    root = root.resolve()
    output = output.resolve()
    require(root.is_dir(), 'Result directory not found: ' + str(root))
    require(output != root and root not in output.parents and output not in root.parents,
            'Export directory must be separate from the result directory')
    require(not output.exists(), 'Output already exists; use --verify-export or choose a new --output')
    require(8 <= part_mib <= 512, '--part-mib must be between 8 and 512')
    limit = int(part_mib * MIB)
    complete = read_json(safe_file(root, 'COMPLETE.json'))
    binding = read_json(safe_file(root, 'INPUT_BINDING.json'))
    results = read_json(safe_file(root, 'results.json'))
    require(complete['status'] == 'COMPLETE' and results['status'] == 'COMPLETE',
            'Only a completed study can be exported')
    require(complete['binding'] == binding['binding'], 'Completion binding mismatch')
    require(sha256(safe_file(root, 'caenl-feedback-study-v1.md')) == complete['report_sha256'],
            'Report differs from the completion record')
    payload = {key: binding[key] for key in ('protocol', 'source_files', 'environment')}
    calculated_binding = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    require(calculated_binding == binding['binding'], 'Input binding content mismatch')
    for rel, digest in binding['source_files'].items():
        require(sha256(safe_file(root / 'source', rel)) == digest, 'Source changed: ' + rel)
    rows = results['per_seed']
    counts = collections.Counter(row['stage'] for row in rows)
    require(counts == {'tuning': 24, 'confirmation': 40, 'transfer': 40},
            'Unexpected or incomplete method matrix')
    require(len({(row['stage'], row['seed'], row['method']) for row in rows}) == 104,
            'Duplicate or missing method records')
    require(complete['completed_method_runs'] == 104, 'Completion count mismatch')
    expected = {}

    def expect(path, digest):
        rel = path.relative_to(root).as_posix()
        require(rel not in expected or expected[rel] == digest, 'Conflicting hash records: ' + rel)
        expected[rel] = digest

    for p in root.rglob('DONE.json'):
        record = read_json(p)
        for rel, digest in record.get('payload', {}).items():
            expect(safe_file(p.parent, rel), digest)
    for p in root.glob('*/seed*/**/acquisitions/round*/COMPLETE.json'):
        record = read_json(p)
        expect(safe_file(p.parent, 'selection.npz'), record['selection_sha256'])
        for item in record['passes']:
            expect(safe_file(p.parent, 'pass_%03d.npz' % item['pass']), item['sha256'])
    for row in rows:
        method = root / row['stage'] / ('seed%d' % row['seed']) / row['method']
        require((method / 'DONE.json').is_file(), 'Missing method completion: ' + str(method))
        require(len(list((method / 'rounds').glob('round*/summary.json'))) == 5,
                'Missing phase summaries: ' + str(method))
        expect(safe_file(method, 'rounds/round5/state.pt'), row['final_checkpoint_sha256'])
        expect(safe_file(method.parent, 'shared_initial/state.pt'), row['initial_checkpoint_sha256'])
        require((method / 'rounds/round5/feedback.npz').is_file(), 'Missing final holdout array')
        if row['stage'] != 'tuning':
            require((method / 'rounds/round5/validation.npz').is_file(), 'Missing final validation array')

    selected = []
    inventory = []
    unknown = []
    for p in walk_files(root):
        rel = p.relative_to(root).as_posix()
        record = {'path': rel, 'size_bytes': p.stat().st_size,
                  'recorded_sha256': expected.get(rel), 'sha256_verified_in_this_export': False}
        group = None
        if p.suffix in ('.pt', '.pth', '.ckpt'):
            record['category'] = 'checkpoint_retained_on_server'
            if verify_checkpoints:
                require(record['recorded_sha256'] is not None, 'No recorded checkpoint hash: ' + rel)
                print('Verifying checkpoint ' + rel, flush=True)
                require(sha256(p) == record['recorded_sha256'], 'Checkpoint hash differs: ' + rel)
                record['sha256_verified_in_this_export'] = True
        elif p.name == 'training.csv':
            group = 'traces'
        elif p.suffix == '.npz':
            if '/rounds/round5/' in rel and p.name in ('validation.npz', 'feedback.npz'):
                group = 'predictions'
            elif p.name in ('splits.npz', 'selection.npz'):
                group = 'review'
            elif all_arrays:
                group = 'additional_arrays'
            else:
                record['category'] = 'earlier_evaluation_or_candidate_array_retained_on_server'
        elif p.suffix in ('.json', '.jsonl', '.md', '.txt', '.py', '.log', '.sha256', '.csv'):
            group = 'review'
        else:
            record['category'] = 'unselected_runtime_file'
            unknown.append(rel)
        if group:
            record['category'] = group
            selected.append((p, 'results/' + rel, group, expected.get(rel)))
        inventory.append(record)

    # Include the exact catalog used to interpret image indices, never dataset images.
    dataset = Path(binding['protocol']['dataset_root'])
    for rel in ('IMAGENET100_READY.json', 'class_counts.csv', 'images_manifest.jsonl'):
        p = safe_file(dataset, rel)
        expected_hash = (binding['protocol']['expected_image_manifest_sha256']
                         if rel == 'images_manifest.jsonl' else None)
        selected.append((p, 'dataset_metadata/' + rel, 'review', expected_hash))

    print('Indexing %d files for export; %d checkpoints remain on the server.' %
          (len(selected), sum(i['category'] == 'checkpoint_retained_on_server' for i in inventory)), flush=True)
    # Precompute deflated sizes so every ZIP volume has a checked size bound.
    entries = []
    for index, (p, arc, group, recorded_hash) in enumerate(sorted(selected, key=lambda x: (x[2], x[1])), 1):
        before = fingerprint(p)
        digest = sha256(p)
        require(recorded_hash is None or digest == recorded_hash, 'Recorded payload hash differs: ' + str(p))
        stored = p.suffix == '.npz'
        size = compressed_size(p, stored)
        require(fingerprint(p) == before, 'Source file changed while indexing: ' + str(p))
        require(size + 2 * len(arc.encode('utf-8')) + 4096 < limit,
                'One file exceeds the volume limit: %s; use a larger --part-mib' % p)
        entries.append({'source': p, 'archive_path': arc, 'group': group, 'size_bytes': before[2],
                        'sha256': digest, 'recorded_sha256': recorded_hash,
                        'compressed_size_estimate': size, 'stored': stored, 'fingerprint': before})
        if index % 100 == 0:
            print('Indexed %d/%d files' % (index, len(selected)), flush=True)
    estimated = sum(e['compressed_size_estimate'] + 2 * len(e['archive_path'].encode('utf-8')) + 200 for e in entries)
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    require(shutil.disk_usage(ancestor).free > estimated * 1.1 + 64 * MIB,
            'Insufficient free disk for the export; no source files were changed')
    output.mkdir(parents=True)
    archives = []
    for group in ('review', 'predictions', 'traces', 'additional_arrays'):
        batch = []
        size = 4096
        batches = []
        for e in (e for e in entries if e['group'] == group):
            cost = e['compressed_size_estimate'] + 2 * len(e['archive_path'].encode('utf-8')) + 200
            if batch and size + cost > limit:
                batches.append(batch)
                batch = []
                size = 4096
            batch.append(e)
            size += cost
        if batch:
            batches.append(batch)
        for number, batch in enumerate(batches, 1):
            name = 'CAENL_feedback_%s_%03d.zip' % (group, number)
            path = output / name
            print('Writing ' + name, flush=True)
            with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
                for e in batch:
                    require(fingerprint(e['source']) == e['fingerprint'], 'Source changed before packaging: ' + str(e['source']))
                    zf.write(e['source'], e['archive_path'], compress_type=zipfile.ZIP_STORED if e['stored'] else zipfile.ZIP_DEFLATED)
                    require(fingerprint(e['source']) == e['fingerprint'], 'Source changed during packaging: ' + str(e['source']))
            require(path.stat().st_size <= limit, 'Archive size limit exceeded: ' + name)
            for e in batch:
                e['archive'] = name
            archives.append({'name': name, 'group': group, 'size_bytes': path.stat().st_size,
                             'sha256': sha256(path), 'files': len(batch)})

    exported = [{k: v for k, v in e.items() if k not in ('source', 'fingerprint', 'compressed_size_estimate', 'stored')}
                for e in entries]
    verified_paths = {e['archive_path'][8:] for e in exported if e['archive_path'].startswith('results/')}
    for item in inventory:
        if item['path'] in verified_paths:
            item['sha256_verified_in_this_export'] = True
    manifest = {
        'format': 'caenl-feedback-evidence-v1',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'campaign_binding': binding['binding'],
        'original_report_sha256': complete['report_sha256'],
        'exporter_sha256': sha256(Path(__file__)),
        'all_arrays_included': all_arrays,
        'checkpoint_bytes_included': False,
        'checkpoint_hashes_verified_in_this_export': verify_checkpoints,
        'source_mutation': False,
        'gpu_used': False,
        'archives': archives,
        'exported_files': exported,
        'source_inventory': inventory,
        'unselected_runtime_files': unknown,
        'scope': ('Original bytes of all selected files; final holdout and validation arrays, all phase summaries, '
                  'all controller histories, all step CSVs, acquisition selections, splits, source and data catalog. '
                  'Earlier evaluation arrays and candidate-pass arrays are included only with --all-arrays. '
                  'Checkpoint paths and recorded hashes are listed, but model weights and dataset images are not copied. '
                  'Export integrity is not independent reproduction of model inference or proof of a scientific claim.')
    }
    json_write(output / 'EXPORT_MANIFEST.json', manifest)
    verify_export(output, require_complete=False)
    json_write(output / 'EXPORT_COMPLETE.json', {'status': 'COMPLETE', 'archives': len(archives),
               'files': len(exported), 'manifest_sha256': sha256(output / 'EXPORT_MANIFEST.json')})
    with (output / 'SHA256SUMS').open('w', encoding='utf-8') as stream:
        for name in [a['name'] for a in archives] + ['EXPORT_MANIFEST.json', 'EXPORT_COMPLETE.json']:
            stream.write('%s  %s\n' % (sha256(output / name), name))
    print('EVIDENCE_EXPORT_COMPLETE', flush=True)
    print('Output: ' + str(output), flush=True)
    print('Upload review ZIPs and EXPORT_MANIFEST.json first; prediction and trace ZIPs provide the native numerical evidence.', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument('--output', type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument('--part-mib', type=int, default=28)
    parser.add_argument('--all-arrays', action='store_true', help='Also export every earlier evaluation and candidate-pass NPZ')
    parser.add_argument('--verify-checkpoints', action='store_true', help='Read and hash all checkpoint bytes; does not load tensors or use a GPU')
    parser.add_argument('--verify-export', action='store_true', help='Recheck an existing export without reading the experiment tree')
    args = parser.parse_args()
    try:
        if args.verify_export:
            verify_export(args.output.resolve())
        else:
            export(args.root, args.output, args.part_mib, args.all_arrays, args.verify_checkpoints)
    except (OSError, ValueError, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
        print('EXPORT FAILED: ' + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
