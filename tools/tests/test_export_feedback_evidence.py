import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / 'export_feedback_evidence.py'
spec = importlib.util.spec_from_file_location('exporter', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
checks = []


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        p.write_bytes(value)
    else:
        m.json_write(p, value)
    return m.sha256(p)


def assert_rejects(fn, expected):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
    except (RuntimeError, zipfile.BadZipFile) as exc:
        assert expected in str(exc), str(exc)
        checks.append(expected)
    else:
        raise AssertionError('Expected rejection: ' + expected)


with tempfile.TemporaryDirectory() as temporary:
    base = Path(temporary)
    root = base / 'results'
    dataset = base / 'dataset'
    root.mkdir()
    dataset.mkdir()
    write(dataset/'IMAGENET100_READY.json', {'status':'READY'})
    write(dataset/'class_counts.csv', b'class_id,wnid\n0,class0\n')
    image_hash = write(dataset/'images_manifest.jsonl', b'{"path":"fixture","class_id":0,"sha256":"fixture"}\n')
    code_hash = write(root/'source/fixture.py', b'# Synthetic exporter fixture; not experiment output.\n')
    payload = {'protocol': {'dataset_root':str(dataset), 'expected_image_manifest_sha256':image_hash},
               'source_files': {'fixture.py':code_hash}, 'environment': {'fixture':True}}
    binding_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    write(root/'INPUT_BINDING.json', dict(payload, binding=binding_hash))
    report_hash = write(root/'caenl-feedback-study-v1.md', b'# Synthetic test fixture\n')
    # ZIP-shaped payloads are sufficient here: the exporter must preserve opaque bytes.
    small = io.BytesIO()
    with zipfile.ZipFile(small, 'w') as z:
        z.writestr('fixture.npy', b'fixture-npy-member-for-byte-preservation-only')
    array = small.getvalue()
    rows = []
    for stage, seeds, names in (
        ('tuning',range(41001,41004),['candidate%d'%i for i in range(8)]),
        ('confirmation',range(42001,42011),['entropy','lite','tuned_fixed','tuned_schedule']),
        ('transfer',range(43001,43011),['entropy','lite','tuned_fixed','tuned_schedule'])):
        for seed in seeds:
            sd = root/stage/('seed%d'%seed)
            ih = write(sd/'shared_initial/state.pt', ('initial%d'%seed).encode())
            initial_files={'state.pt':ih}
            for name, value in [('summary.json',{'phase':0}),('training.csv',b'step,train_ce\n1,1.0\n'),('feedback.npz',array)]:
                initial_files[name]=write(sd/'shared_initial'/name,value)
            write(sd/'shared_initial/DONE.json',{'payload':initial_files})
            write(sd/'splits.npz',array)
            write(sd/'targets.json',{'fixture':True})
            for method in names:
                md=sd/method
                method_payload={}
                for phase in range(1,6):
                    pd=md/'rounds'/('round%d'%phase)
                    files={}
                    files['state.pt']=write(pd/'state.pt', ('%s%d%s%d'%(stage,seed,method,phase)).encode())
                    for name,value in [('summary.json',{'phase':phase}),('training.csv',b'step,train_ce\n1,1.0\n'),('controller_trajectory.json',[{'phase':phase}]),('feedback.npz',array)]:
                        files[name]=write(pd/name,value)
                    if stage!='tuning': files['validation.npz']=write(pd/'validation.npz',array)
                    method_payload['rounds/round%d/DONE.json'%phase]=write(pd/'DONE.json',{'payload':files})
                    ad=md/'acquisitions'/('round%d'%phase)
                    ah=write(ad/'selection.npz',array)
                    ph=write(ad/'pass_001.npz',array)
                    method_payload['acquisitions/round%d/COMPLETE.json'%phase]=write(ad/'COMPLETE.json',{'selection_sha256':ah,'passes':[{'pass':1,'sha256':ph}]})
                method_payload['summary.json']=write(md/'summary.json',{'fixture':True})
                write(md/'DONE.json',{'payload':method_payload})
                rows.append({'stage':stage,'seed':seed,'method':method,'initial_checkpoint_sha256':ih,'final_checkpoint_sha256':files['state.pt']})
    write(root/'results.json',{'status':'COMPLETE','per_seed':rows})
    write(root/'COMPLETE.json',{'status':'COMPLETE','binding':binding_hash,'report_sha256':report_hash,'completed_method_runs':104})
    # Incompressible text fixtures force genuine multi-volume output.
    write(root/'size_fixture_a.txt',os.urandom(5*m.MIB))
    write(root/'size_fixture_b.txt',os.urandom(5*m.MIB))
    before={str(p.relative_to(root)):m.sha256(p) for p in m.walk_files(root)}
    out=base/'export'
    with contextlib.redirect_stdout(io.StringIO()):
        manifest=m.export(root,out,part_mib=8,all_arrays=True,verify_checkpoints=True)
    after={str(p.relative_to(root)):m.sha256(p) for p in m.walk_files(root)}
    assert before==after
    checks.append('source bytes preserved')
    assert all(x['size_bytes']<=8*m.MIB for x in manifest['archives'])
    assert sum(a['group']=='review' for a in manifest['archives'])>=2
    checks.append('multi-volume size bound')
    archived={e['archive_path']:e for e in manifest['exported_files']}
    assert len([e for e in archived.values() if e['group']=='predictions'])==184
    assert len([e for e in archived.values() if e['group']=='traces'])==543
    assert len([e for e in archived.values() if e['group']=='additional_arrays'])==1279
    checks.append('all expected predictions, traces and optional arrays included')
    cp=[x for x in manifest['source_inventory'] if x['category']=='checkpoint_retained_on_server']
    assert len(cp)==543 and all(x['sha256_verified_in_this_export'] for x in cp)
    assert not any(e['archive_path'].endswith('.pt') for e in archived.values())
    checks.append('all 543 checkpoint hashes verified without export')
    root.rename(base/'temporarily_unavailable')
    with contextlib.redirect_stdout(io.StringIO()): m.verify_export(out)
    (base/'temporarily_unavailable').rename(root)
    checks.append('export verification independent of source tree')
    out2=base/'default_export'
    with contextlib.redirect_stdout(io.StringIO()):
        default=m.export(root,out2,part_mib=8)
    assert not any(e['group']=='additional_arrays' for e in default['exported_files'])
    assert all(not x['sha256_verified_in_this_export'] for x in default['source_inventory'] if x['category']=='checkpoint_retained_on_server')
    checks.append('default omissions explicitly recorded')
    assert_rejects(lambda:m.export(root,out),'Output already exists')
    assert_rejects(lambda:m.export(root,root/'export'),'separate from the result')
    original=(root/'caenl-feedback-study-v1.md').read_bytes()
    (root/'caenl-feedback-study-v1.md').write_bytes(b'changed')
    assert_rejects(lambda:m.export(root,base/'bad_report'),'Report differs')
    (root/'caenl-feedback-study-v1.md').write_bytes(original)
    selected=root/'confirmation/seed42001/entropy/acquisitions/round1/selection.npz'
    original=selected.read_bytes();selected.write_bytes(b'changed')
    assert_rejects(lambda:m.export(root,base/'bad_selection'),'Recorded payload hash differs')
    selected.write_bytes(original)
    (root/'unexpected_link').symlink_to(dataset/'class_counts.csv')
    assert_rejects(lambda:m.export(root,base/'bad_link'),'Linked file in evidence tree')
    (root/'unexpected_link').unlink()
    p=out/manifest['archives'][0]['name'];data=p.read_bytes();p.write_bytes(data[:-4]+b'xxxx')
    assert_rejects(lambda:m.verify_export(out),'Archive checksum differs')
    p.write_bytes(data)
    original=(out/'EXPORT_MANIFEST.json').read_bytes();(out/'EXPORT_MANIFEST.json').write_bytes(original+b' ')
    assert_rejects(lambda:m.verify_export(out),'Export manifest changed')
    (out/'EXPORT_MANIFEST.json').write_bytes(original)
    print(json.dumps({'status':'PASS','checks':checks,'exported_files_full_fixture':len(manifest['exported_files']),
                      'archives_full_fixture':len(manifest['archives']),'production_GPU_artifacts_accessed':False},indent=2))
