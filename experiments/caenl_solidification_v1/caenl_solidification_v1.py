#!/usr/bin/env python3
"""Frozen, detached CÆNL integrity and classification follow-up campaign."""
import argparse,fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time,traceback,zipfile
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ORIGINAL=Path('/mnt/caenl/active/results/caenl-feedback-study-v1')
DEFAULT_ROOT=Path('/mnt/caenl/active/results/caenl-solidification-v1')
BINDING='ae404ae3af4e41283f26da1fa354f788bb68c9f543f833f8e4e74a68fc5ea76c'

def require(ok,message):
    if not ok:raise RuntimeError(message)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def atomic(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def verify_source():
    manifest=read(HERE/'SOURCE_MANIFEST.json')
    for rel,d in manifest.items():
        p=HERE/rel
        require(not Path(rel).is_absolute() and '..' not in Path(rel).parts and p.is_file() and not p.is_symlink(),'Unsafe/missing source file')
        require(sha(p)==d,'Source checksum mismatch: '+rel)
    return manifest

def alive(root):
    p=root/'PID.json'
    if not p.exists():return False
    try:
        cmd=Path('/proc')/str(read(p)['pid'])/'cmdline';value=cmd.read_bytes()
        return b'--worker' in value and Path(__file__).name.encode() in value
    except (OSError,KeyError):return False

def choose_python():
    expected=read(ORIGINAL/'INPUT_BINDING.json')['environment']['versions']
    paths=[Path.home()/'caenl-v5.4.3/caenl_revision_pipeline_v5.4.3_c4_confirmatory/.venv/bin/python',Path.home()/'caenl-src/caenl_revision_pipeline_v5.3_ibm_rhel_l40s/.venv/bin/python',Path(sys.executable)]
    probe="import json,importlib.metadata;print(json.dumps({n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','scipy','Pillow')}))"
    for p in paths:
        if not p.is_file():continue
        q=subprocess.run([str(p),'-B','-c',probe],capture_output=True,text=True)
        if q.returncode==0 and json.loads(q.stdout)==expected:return p
    raise RuntimeError('Original Python environment unavailable. No packages installed or changed.')

def resolved_root(args):
    record=HERE/'RUN_LOCATION.json'
    root=(args.root or (Path(read(record)['root']) if record.exists() else DEFAULT_ROOT)).resolve()
    require(root!=ORIGINAL and ORIGINAL not in root.parents and root not in ORIGINAL.parents,'Output must be separate from original experiment')
    require(root!=HERE and HERE not in root.parents and root not in HERE.parents,'Results must be separate from source directory')
    if record.exists():require(str(root)==read(record)['root'],'Existing run uses a different result directory')
    return root

def start(args):
    manifest=verify_source();root=resolved_root(args);root.mkdir(parents=True,exist_ok=True)
    if alive(root):print('Worker already active. Use --status.');return
    if (root/'SOURCE_MANIFEST.json').exists():require(read(root/'SOURCE_MANIFEST.json')==manifest,'Existing campaign uses another source manifest')
    if (root/'COMPLETE.json').exists():
        record=read(root/'COMPLETE.json');archive=root/'review/CAENL_SOLIDIFICATION_REVIEW.zip'
        require(archive.is_file() and sha(archive)==record['review_sha256'],'Completed review archive is missing or changed')
        print('Already complete. Share: '+str(archive));return
    python=choose_python();env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    atomic(HERE/'RUN_LOCATION.json',dict(root=str(root)))
    atomic(root/'STATUS.json',dict(status='STARTING',time=time.time()))
    with (root/'run.log').open('a',buffering=1) as log:
        p=subprocess.Popen([str(python),'-B',str(Path(__file__).resolve()),'--worker','--root',str(root)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    atomic(root/'PID.json',dict(pid=p.pid))
    print('STARTED: source/runtime checks, original checkpoint replay, content audit, then 206 new method runs.')
    print('Detached worker: you may disconnect. Root: '+str(root));print('Use --status; progress: '+str(root/'run.log'))

def audio_inventory(root):
    base=Path('/mnt/caenl/active/results/caenl-audiocaps-confirm-v1');items=[];records=[]
    if base.exists():
        for p in sorted(base.rglob('*')):
            if p.is_symlink() or not p.is_file():continue
            name=p.name.lower()
            if p.suffix.lower() in ('.pt','.pth','.safetensors') and ('initial' in str(p.relative_to(base)).lower() or 'init_' in name):
                items.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
            if p.suffix.lower()=='.json' and 'initial' in name and p.stat().st_size<2**20:
                records.append(dict(path=str(p),sha256=sha(p),record=read(p)))
    atomic(root/'AUDIO_INITIALIZATION_INVENTORY.json',dict(root=str(base),root_exists=base.exists(),candidate_binaries=items,initialization_records=records,
        scope='Read-only inventory of named initial-state candidates; not proof that binaries match all accepted tensor digests or that seed-only initialization is deterministic. No audio retraining.'))

def package_review(root):
    dest=root/'review';dest.mkdir(exist_ok=True);path=dest/'CAENL_SOLIDIFICATION_REVIEW.zip'
    files=[p for p in root.glob('*.json') if p.name!='PID.json']+[p for p in [root/'REPORT.txt',root/'run.log'] if p.exists()]
    files += [p for p in (root/'data_audit').glob('*') if p.name in ['summary.json','excluded.json','duplicates.json','DONE.json','original_indices.npz']]
    files += list(root.glob('*/seed*/targets.json'))+list(root.glob('*/seed*/splits.npz'))+list(root.glob('*/seed*/*/INFERENCE_REPLAY.json'))
    files += [p for p in (root/'checkpoint_replay').glob('*.json') if p.name in ['RESULT.json','PROTOCOL.json','STATUS.json']]
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(set(files)):z.write(p,p.relative_to(root).as_posix())
        for p in [HERE/'SOURCE_MANIFEST.json',HERE/'PROTOCOL.txt',HERE/'LAUNCH_AND_SHARE.txt']:z.write(p,'release/'+p.name)
    path.with_suffix('.zip.sha256').write_text(sha(path)+'  '+path.name+'\n')
    return path

def export_evidence(root,progress):
    """Independently readable ZIP parts below 30 MiB; excludes images/model binaries."""
    dest=root/'review';dest.mkdir(exist_ok=True)
    cfg=read(root/'DERIVED_PROTOCOL.json');last='round'+str(cfg['rounds'])
    files=list(root.glob('*/seed*/*/rounds/*/training.csv'))+list(root.glob('*/seed*/*/rounds/*/controller_trajectory.json'))
    files+=list(root.glob(f'*/seed*/*/rounds/{last}/*.npz'))+list(root.glob('*/seed*/*/acquisitions/*/selection.npz'))
    files+=list(root.glob('*/seed*/*/rounds/*/summary.json'))+list(root.glob('*/seed*/*/rounds/*/DONE.json'))+list(root.glob('*/seed*/*/rounds/*/PRUNED_STATE.json'))
    # Candidate scores remain on the server. Their digests accompany every acquisition record.
    groups=[];current=[];size=0
    for p in sorted(set(files)):
        n=p.stat().st_size
        if current and size+n>26*2**20:groups.append(current);current=[];size=0
        current.append(p);size+=n
    if current:groups.append(current)
    index=[]
    for i,group in enumerate(groups,1):
        path=dest/f'CAENL_EVIDENCE_part_{i:03d}.zip';progress(f'Packaging numeric evidence {i}/{len(groups)}')
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            manifest={}
            for p in group:
                rel=p.relative_to(root).as_posix();z.write(p,rel);manifest[rel]=sha(p)
            z.writestr('PART_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
        require(path.stat().st_size<30*2**20,'A numeric evidence part exceeds 30 MiB; preserve results and inspect packaging')
        index.append(dict(name=path.name,sha256=sha(path),bytes=path.stat().st_size,files=len(group)))
    atomic(dest/'EVIDENCE_INDEX.json',dict(parts=index,scope='All new training traces, final evaluation arrays, acquisition selections and phase summaries; no raw images, model weights or candidate-score arrays.'))
    (dest/'EVIDENCE_SHA256SUMS').write_text(''.join(r['sha256']+'  '+r['name']+'\n' for r in index))
    return index

def worker(args):
    root=resolved_root(args);root.mkdir(parents=True,exist_ok=True);lock=(root/'RUN.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print('Another worker holds the campaign lock.');return 1
    start_time=time.time()
    def progress(message):
        atomic(root/'STATUS.json',dict(status='RUNNING',pid=os.getpid(),message=message,time=time.time()))
        print(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),message,flush=True)
    try:
        manifest=verify_source()
        if (root/'SOURCE_MANIFEST.json').exists():require(read(root/'SOURCE_MANIFEST.json')==manifest,'Existing campaign source changed')
        else:atomic(root/'SOURCE_MANIFEST.json',manifest)
        require(shutil.disk_usage(root).free>2**30,'Need at least 1 GiB free for audits; no existing file was removed.')
        q=read(ORIGINAL/'COMPLETE.json');require(q['status']=='COMPLETE' and q['binding']==BINDING,'Unexpected original completed study')
        progress('Preflight: CPU integration tests in an isolated process')
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
        checked=subprocess.run([sys.executable,'-B',str(HERE/'engine/self_test.py')],env=env,check=True)
        atomic(root/'CPU_CHECK.json',dict(status='PASS',source_manifest_sha256=sha(HERE/'SOURCE_MANIFEST.json'),scientific_runs_counted=0))
        progress('Replaying original 104 final checkpoints; no training yet')
        subprocess.run([sys.executable,'-B',str(HERE/'engine/checkpoint_replay.py'),'--worker','--root',str(ORIGINAL),'--output',str(root/'checkpoint_replay')],check=True)
        require(read(root/'checkpoint_replay/RESULT.json')['status']=='PASS','Original checkpoint replay is not PASS; training not started')
        sys.path.insert(0,str(HERE/'engine'))
        import torch,numpy as np
        from support import Catalog,Model,read_json,digest,seed_all,tensor_digest,atom_torch
        from runner import freeze,run_study,read_rows
        from data_audit import audit_catalog
        from study import planned_counts
        require(torch.cuda.is_available(),'CUDA GPU unavailable')
        previous=read(ORIGINAL/'INPUT_BINDING.json')['environment']
        require(torch.cuda.get_device_name(0)==previous['gpu'] and torch.version.cuda==previous['cuda'],'Original GPU/CUDA environment required')
        other=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        require(not {int(x) for x in other.stdout.splitlines() if x.strip().isdigit()}-{os.getpid()},'GPU is busy; no other job stopped')
        torch.set_num_threads(8);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        cfg=read_json(HERE/'engine/protocol.json');original=Catalog(cfg)
        progress('Decoding dataset images to build a content-disjoint manifest')
        cat,audit=audit_catalog(original,cfg,root/'data_audit',progress)
        cfg.update(expected_train_count=len(cat.labels),expected_val_count=len(cat.val_labels),initial_labels=len(cat.labels)//10,acquisition_per_round=len(cat.labels)//10)
        cfg['holdout_count']=cfg['initial_labels']//20;cfg['result_root']=str(root)
        freeze(root/'DERIVED_PROTOCOL.json',cfg)
        environment=dict(python=sys.version,torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),original_environment=previous)
        binding=digest(dict(source=manifest,protocol=cfg,data_audit=audit['binding'],retained_manifest=sha(root/'data_audit/retained.json'),environment=environment))
        freeze(root/'INPUT_BINDING.json',dict(binding=binding,source=manifest,protocol=cfg,retained_manifest_sha256=sha(root/'data_audit/retained.json'),environment=environment))
        audio_inventory(root)
        counts=planned_counts(cfg);require(counts['method_runs']==206 and counts['initializations']==23,'Production matrix differs')
        finished=len(read_rows(root));need=(10+54*(206-finished)/206)*2**30
        require(shutil.disk_usage(root).free>=need,f'Training requires about {need/2**30:.1f} GiB free at this stage. Audits are preserved; no old data deleted.')
        progress('GPU contract: real ResNet-50 and ResNet-18 steps, new contrastive objective, portable checkpoint reload')
        from contrastive import contrastive_penalty
        from aligned import class_scatter,aligned_loss
        gpu_checks=[]
        for backbone in ['resnet50','resnet18']:
            seed_all(913901);m=Model(100,backbone).cuda().to(memory_format=torch.channels_last);opt=torch.optim.SGD(m.parameters(),lr=.01,momentum=.9)
            x=torch.rand(128,3,224,224,device='cuda').contiguous(memory_format=torch.channels_last);y=torch.arange(16,device='cuda').repeat_interleave(8)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits,z=m.forward_features(x)
            ce=torch.nn.functional.cross_entropy(logits.float(),y);targets={}
            for l,v in z.items():
                g=class_scatter(v,y);targets[l]=dict(log_kappa_target=float(g['log_kappa'].detach())-.2,between_floor=max(float(g['between'].detach())*.5,1e-5))
            reg,_=aligned_loss(z,y,targets,{l:.03 for l in z},cfg['alignment'])
            loss=ce+reg+contrastive_penalty(z,y,.03);loss.backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()),'Nonfinite GPU contract gradient');opt.step()
            import io
            m.eval()
            with torch.no_grad():reference=m(x[:8]).float().cpu()
            buf=io.BytesIO();torch.save({'model':m.state_dict()},buf);buf.seek(0)
            recovered=torch.load(buf,map_location='cpu',weights_only=False)
            check_model=Model(100,backbone).cuda().to(memory_format=torch.channels_last);check_model.load_state_dict(recovered['model'],strict=True);check_model.eval()
            with torch.no_grad():replayed=check_model(x[:8]).float().cpu()
            require(torch.equal(reference.argmax(1),replayed.argmax(1)) and torch.allclose(reference,replayed,atol=1e-5,rtol=1e-5),'GPU checkpoint reload mismatch')
            gpu_checks.append(dict(backbone=backbone,loss=float(loss.detach()),model_weights_sha256=tensor_digest(m.state_dict()),portable_checkpoint_reloaded=True))
            del m,opt,x,y,logits,z,ce,reg,loss,check_model,recovered,buf,reference,replayed
            import gc;gc.collect();torch.cuda.empty_cache()
        atomic(root/'GPU_CHECK.json',dict(status='PASS',checks=gpu_checks,scientific_runs_counted=0))
        result=run_study(root,cfg,cat,binding,progress,'cuda:0')
        progress('All training and per-method checkpoint reloads complete; packaging evidence')
        parts=export_evidence(root,progress)
        atomic(root/'STATUS.json',dict(status='COMPLETE',method_runs=206,worker_active=False))
        archive=package_review(root)
        atomic(root/'COMPLETE.json',dict(status='COMPLETE',binding=binding,method_runs=result['completed_methods'],planned=counts,evidence_parts=len(parts),review_sha256=sha(archive),evidence_index_sha256=sha(root/'review/EVIDENCE_INDEX.json'),seconds_this_invocation=time.time()-start_time,scientific_success_not_guaranteed=True))
        print('CAENL_SOLIDIFICATION_COMPLETE\nShare: '+str(root/'review/CAENL_SOLIDIFICATION_REVIEW.zip'),flush=True)
        return 0
    except Exception as exc:
        atomic(root/'STATUS.json',dict(status='BLOCKED',message=str(exc),time=time.time(),worker_active=False))
        print(traceback.format_exc(),flush=True)
        try:package_review(root)
        except Exception:print('Diagnostic packaging also failed; retain run.log',flush=True)
        return 2

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path)
    g=p.add_mutually_exclusive_group(required=True)
    for flag in ['start','status','worker','self-test','export']:g.add_argument('--'+flag,action='store_true')
    a=p.parse_args()
    if a.start:start(a)
    elif a.worker:return worker(a)
    elif a.self_test:
        verify_source();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
        return subprocess.run([sys.executable,'-B',str(HERE/'engine/self_test.py')],env=env).returncode
    elif a.export:
        root=resolved_root(a);require(not alive(root),'Do not export a moving result tree');package_review(root);print(root/'review/CAENL_SOLIDIFICATION_REVIEW.zip')
    else:
        root=resolved_root(a);state=read(root/'STATUS.json') if (root/'STATUS.json').exists() else {'status':'NOT_STARTED'}
        state['worker_active']=alive(root);print(json.dumps(state,indent=2))
        total=0
        for stage,target in [('tuning',36),('confirmation',110),('transfer',60)]:
            count=sum(p.parent.name!='shared_initial' for p in (root/stage).glob('seed*/*/DONE.json'));total+=count;print(f'{stage}: {count}/{target}')
        print(f'Total: {total}/206; 23 shared initializations.');print('Log: '+str(root/'run.log'));print('Review: '+str(root/'review/CAENL_SOLIDIFICATION_REVIEW.zip'))
    return 0
if __name__=='__main__':raise SystemExit(main())
