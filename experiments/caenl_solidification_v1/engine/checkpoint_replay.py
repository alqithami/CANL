#!/usr/bin/env python3
"""Replay final CÆNL feedback-study checkpoints without training or modifying inputs.

Scope: 104 final checkpoints, 104 training holdouts and 80 official validation
sets. Original archived inference code is reused; metrics and feature moments
are checked with NumPy. This is checkpoint re-execution, not a new independent
model implementation, acquisition audit, adversarial test or training run.
"""
import argparse, hashlib, importlib.metadata, json, os, platform, shutil
import subprocess, sys, time, traceback, zipfile
from collections import Counter
from pathlib import Path
sys.dont_write_bytecode = True
DEFAULT_ROOT = Path('/mnt/caenl/active/results/caenl-feedback-study-v1')
DEFAULT_OUT = Path.home()/'caenl-checkpoint-verify-v1'
EXPECTED_BINDING = 'ae404ae3af4e41283f26da1fa354f788bb68c9f543f833f8e4e74a68fc5ea76c'
SCOPE = __doc__.strip()

def require(ok, message):
    if not ok: raise RuntimeError(message)

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    return h.hexdigest()

def atomic(p, value):
    p=Path(p);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temp,p)

def contained(path, parent):
    return path==parent or parent in path.parents

def safe_file(parent, rel):
    r=Path(rel);p=parent/r
    require(not r.is_absolute() and '..' not in r.parts,'Unsafe relative input path')
    require(p.is_file() and not p.is_symlink() and contained(p.resolve(),parent.resolve()),'Missing or unsafe input: '+str(p))
    return p

def verify_payload(parent, names=None):
    record=read(parent/'DONE.json')['payload']
    for rel in (record if names is None else names):
        require(rel in record,'Missing manifest entry: '+rel)
        require(sha(safe_file(parent,rel))==record[rel],'Payload checksum mismatch: '+str(parent/rel))
    return record

def np_metrics(logits, labels):
    import numpy as np
    x=np.asarray(logits,dtype=np.float64);y=np.asarray(labels,dtype=np.int64)
    require(x.ndim==2 and x.shape[0]==len(y) and len(y)>0 and np.isfinite(x).all(),'Invalid logits')
    require(y.min()>=0 and y.max()<x.shape[1],'Invalid label')
    centered=x-x.max(1,keepdims=True);exp=np.exp(centered);prob=exp/exp.sum(1,keepdims=True)
    logz=np.log(exp.sum(1));ce=float(np.mean(logz-centered[np.arange(len(y)),y]))
    pred=x.argmax(1);correct=pred==y;confidence=prob.max(1)
    bins=np.minimum((confidence*15).astype(int),14);ece=0.
    for k in range(15):
        sel=bins==k
        if sel.any():ece+=float(sel.mean()*abs(correct[sel].mean()-confidence[sel].mean()))
    top5=np.argsort(-x,axis=1,kind='stable')[:,:min(5,x.shape[1])]
    return dict(n=len(y),accuracy=float(correct.mean()),top5_accuracy=float(np.any(top5==y[:,None],axis=1).mean()),ce=ce,ece=ece)

def compare_logits(fresh, saved):
    import numpy as np
    require(fresh.shape==saved.shape and np.isfinite(fresh).all() and np.isfinite(saved).all(),'Logit shape/nonfinite mismatch')
    return dict(max_abs_error=float(np.max(np.abs(fresh.astype(float)-saved.astype(float)))),
                within_tolerance=bool(np.allclose(fresh,saved,atol=1e-5,rtol=1e-5)),
                predictions_equal=bool(np.array_equal(fresh.argmax(1),saved.argmax(1))),
                bitwise_equal=bool(np.array_equal(fresh,saved)))

def np_geometry(features, labels, classes, norm_eps, eps):
    import numpy as np
    z=np.asarray(features,dtype=np.float64);y=np.asarray(labels,dtype=np.int64)
    require(np.isfinite(z).all(),'Nonfinite features')
    h=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),norm_eps)
    counts=np.bincount(y,minlength=classes)
    require(np.all(counts>=2),'Insufficient class support')
    sums=np.stack([h[y==c].sum(0) for c in range(classes)])
    sumsq=np.stack([(h[y==c]**2).sum(0) for c in range(classes)])
    means=sums/counts[:,None];grand=means.mean(0)
    within=float(np.mean([np.mean(np.sum((h[y==c]-means[c])**2,axis=1)) for c in range(classes)]))
    between=float(np.mean(np.sum((means-grand)**2,axis=1)))
    total=float(np.mean([np.mean(np.sum((h[y==c]-grand)**2,axis=1)) for c in range(classes)]))
    q=float(np.log(within+eps)-np.log(between+eps))
    return dict(within=within,between=between,total=total,log_kappa=q,kappa=float(np.exp(q))),counts,sums,sumsq

def evaluate(model, cat, cfg, source_infer, split, ids, saved_path, summary, device):
    import numpy as np
    fresh=source_infer(model,cat,split,ids,cfg,device,features=True)
    expected=(cat.labels if split=='train' else cat.val_labels)[ids]
    with np.load(saved_path,allow_pickle=False) as saved:
        for field,values in [('indices',ids),('labels',expected)]:
            require(np.array_equal(saved[field],values) and np.array_equal(fresh[field],values),'Index/label mismatch: '+field)
        require(np.array_equal(saved['predictions'],saved['logits'].argmax(1)),'Saved prediction mismatch')
        checks=compare_logits(fresh['logits'],saved['logits']);issues=[]
        if not checks['within_tolerance']:issues.append('logit_tolerance')
        if not checks['predictions_equal']:issues.append('top1_predictions')
        metrics=np_metrics(fresh['logits'],expected)
        for key,value in metrics.items():
            if abs(value-summary[key])>1e-6:issues.append('metric_'+key)
        geometry={}
        for layer in cfg['layers']:
            g,counts,sums,sumsq=np_geometry(fresh['features'][layer].numpy(),expected,cfg['num_classes'],cfg['alignment']['feature_norm_epsilon'],cfg['alignment']['scatter_epsilon'])
            require(np.array_equal(counts,saved[layer+'_class_counts']),'Class-count mismatch')
            for suffix,values in [('class_sums',sums),('class_sumsq',sumsq)]:
                if not np.allclose(values,saved[layer+'_'+suffix],atol=1e-5,rtol=1e-5):issues.append(layer+'_'+suffix)
            geometry[layer]={'metrics':g,'maximum_summary_error':max(abs(v-summary[layer+'_'+k]) for k,v in g.items())}
            if geometry[layer]['maximum_summary_error']>1e-5:issues.append(layer+'_geometry')
    return dict(status='PASS' if not issues else 'REVIEW_REQUIRED',issues=issues,logits=checks,metrics=metrics,geometry=geometry,
                replay_logits_sha256=hashlib.sha256(fresh['logits'].tobytes()).hexdigest())

def jobs_for(root,cfg):
    jobs=[]
    for stage,backbone in [('tuning','resnet50'),('confirmation','resnet50'),('transfer','resnet18')]:
        found=set()
        for p in sorted((root/stage).glob('seed*/*/DONE.json')):
            if p.parent.name=='shared_initial':continue
            require(read(p).get('record_type')=='method_complete','Unexpected completion marker')
            verify_payload(p.parent);summary=read(p.parent/'summary.json')
            seed=int(summary['seed']);found.add(seed)
            require(summary['stage']==stage and summary['backbone']==backbone,'Stage/backbone mismatch')
            require(summary['method']==p.parent.name and p.parent.parent.name=='seed'+str(seed),'Method path mismatch')
            jobs.append((stage,backbone,seed,p.parent,summary))
        require(found==set(cfg['study'][stage+'_seeds']),'Seed inventory mismatch for '+stage)
    require(Counter(j[0] for j in jobs)=={'tuning':24,'confirmation':40,'transfer':40},'Expected all 104 completed method runs')
    require(len({(j[0],j[2],j[3].name) for j in jobs})==104,'Duplicate method job')
    return jobs

def package(out):
    path=out/'CAENL_CHECKPOINT_REPLAY.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.glob('*.json')):z.write(p,p.name)
        for p in sorted((out/'records').glob('*.json')):z.write(p,'records/'+p.name)
        if (out/'run.log').exists():z.write(out/'run.log','run.log')
    path.with_suffix('.zip.sha256').write_text(sha(path)+'  '+path.name+'\n')
    return path

def worker(args):
    import fcntl, gc
    root=args.root.resolve();out=args.output.resolve();source=root/'source'
    require(not contained(out,root) and not contained(root,out),'Output must be separate from the original result tree')
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'worker.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print('A verification worker is already active.');return 1
    started=time.time();rows=[]
    def progress(message):
        print(message,flush=True)
        atomic(out/'STATUS.json',dict(status='RUNNING',pid=os.getpid(),message=message,completed=len(rows),time=time.time()))
    try:
        progress('Checking original runtime and source before replay')
        import numpy as np, torch
        require(shutil.disk_usage(out).free>=64*1024**2,'Need 64 MiB free in the selected output filesystem; no cleanup attempted')
        complete=read(root/'COMPLETE.json');bound=read(root/'INPUT_BINDING.json')
        require(complete['status']=='COMPLETE' and complete['binding']==bound['binding']==EXPECTED_BINDING,'Unexpected completed study binding')
        calculated=hashlib.sha256(json.dumps({k:bound[k] for k in ('protocol','source_files','environment')},sort_keys=True,allow_nan=False).encode()).hexdigest()
        require(calculated==EXPECTED_BINDING,'Input binding content failed its recorded hash')
        manifest=read(source/'SOURCE_MANIFEST.json');cfg=read(source/'protocol.json')
        require(manifest==bound['source_files'] and cfg==bound['protocol'],'Source/protocol binding differs')
        for rel,digest in manifest.items():require(sha(safe_file(source,rel))==digest,'Source checksum mismatch: '+rel)
        versions={n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','scipy','Pillow')}
        require(versions==bound['environment']['versions'],'Use the original recorded Python environment; package versions differ')
        require(torch.cuda.is_available() and torch.version.cuda==bound['environment']['cuda'],'Original CUDA environment is required')
        require(torch.cuda.get_device_name(0)==bound['environment']['gpu'],'GPU differs from the recorded environment')
        q=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        others={int(x.strip()) for x in q.stdout.splitlines() if x.strip().isdigit()}-{os.getpid()}
        require(not others,'GPU is busy with another process; no process was stopped. Retry when it is free.')
        torch.set_num_threads(8);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        sys.path.insert(0,str(source))
        from support import Catalog,Model,stratified_sample,seed_for,tensor_digest
        from evaluation import infer
        catalog=Catalog(cfg);progress('Checking every dataset image hash before checkpoint replay')
        catalog.verify_images(progress)
        jobs=jobs_for(root,cfg);(out/'records').mkdir(exist_ok=True)
        atomic(out/'PROTOCOL.json',dict(scope=SCOPE,study_binding=bound['binding'],script_sha256=sha(__file__),source_manifest=manifest,
             versions=versions,gpu=torch.cuda.get_device_name(0),tolerances={'logits_atol':1e-5,'logits_rtol':1e-5,'metrics_atol':1e-6,'geometry_summary_atol':1e-5},
             planned_final_checkpoints=104,planned_holdout_evaluations=104,planned_official_evaluations=80))
        for stage,backbone,seed,method_dir,method in jobs:
            name=f'{stage}_seed{seed}_{method_dir.name}';record_path=out/'records'/(name+'.json')
            progress(f'Checkpoint replay {len(rows)+1}/104: {name}')
            model=None
            try:
                phase=method_dir/'rounds'/('round'+str(cfg['rounds']))
                required=['state.pt','summary.json','feedback.npz']+([] if stage=='tuning' else ['validation.npz'])
                payload=verify_payload(phase,required)
                checkpoint=phase/'state.pt';summary=read(phase/'summary.json')
                require(payload['state.pt']==method['final_checkpoint_sha256']==summary['checkpoint_sha256'],'Final checkpoint hash mismatch')
                signature=hashlib.sha256(json.dumps({'files':{k:payload[k] for k in required},'script':sha(__file__),'binding':bound['binding']},sort_keys=True).encode()).hexdigest()
                if record_path.exists():
                    old=read(record_path)
                    if old.get('signature')==signature and old.get('status')=='PASS':rows.append(old);print('Retaining completed replay with matching inputs',flush=True);continue
                selected=stratified_sample(np.arange(len(catalog.labels)),catalog.labels,cfg['initial_labels'],seed_for('initial_labels',seed))
                holdout=stratified_sample(selected,catalog.labels,cfg['holdout_count'],seed_for('controller_holdout',seed))
                with np.load(method_dir.parent/'splits.npz',allow_pickle=False) as split:require(np.array_equal(split['holdout'],holdout),'Holdout reconstruction mismatch')
                state=torch.load(checkpoint,map_location='cpu',weights_only=False)
                require(state['seed']==seed and state['method']==method_dir.name and state['phase']==cfg['rounds'] and state['step']==state['total_steps'],'Checkpoint is not the final scheduled state')
                require(np.array_equal(state['control_indices'],holdout),'Checkpoint holdout differs')
                require(not np.intersect1d(state['labeled'],holdout).size,'Index overlap between gradient training and holdout')
                require(tensor_digest(state['model'])==summary['final_model_weights_sha256'],'Model tensor digest differs')
                model=Model(cfg['num_classes'],backbone).to('cuda:0').float().to(memory_format=torch.channels_last)
                model.load_state_dict(state['model'],strict=True);model.eval();del state
                assessments={'holdout':evaluate(model,catalog,cfg,infer,'train',holdout,phase/'feedback.npz',summary['feedback'],'cuda:0')}
                if stage!='tuning':assessments['official_validation']=evaluate(model,catalog,cfg,infer,'val',np.arange(len(catalog.val_labels)),phase/'validation.npz',summary['validation'],'cuda:0')
                row=dict(name=name,signature=signature,checkpoint_sha256=payload['state.pt'],stage=stage,seed=seed,method=method_dir.name,
                         status='PASS' if all(v['status']=='PASS' for v in assessments.values()) else 'REVIEW_REQUIRED',evaluations=assessments)
                del model;model=None;gc.collect();torch.cuda.empty_cache()
            except Exception as exc:
                row=dict(name=name,status='ERROR',error=str(exc));print(traceback.format_exc(),flush=True)
                if model is not None:del model;model=None
                gc.collect();torch.cuda.empty_cache()
            atomic(record_path,row);rows.append(row)
        passed=sum(r['status']=='PASS' for r in rows)
        status='PASS' if len(rows)==passed==104 else 'REVIEW_REQUIRED'
        result=dict(status=status,completed_final_checkpoints=len(rows),passed_final_checkpoints=passed,
                    replayed_or_resumed_holdouts=sum('holdout' in r.get('evaluations',{}) for r in rows),
                    replayed_or_resumed_official_sets=sum('official_validation' in r.get('evaluations',{}) for r in rows),
                    scope=SCOPE,seconds_this_invocation=time.time()-started,records=rows)
        atomic(out/'RESULT.json',result);atomic(out/'STATUS.json',dict(status=status,completed=len(rows),passed=passed,worker_active=False))
        archive=package(out);print('CHECKPOINT_REPLAY_COMPLETE_'+status,flush=True);print('Share: '+str(archive),flush=True)
        return 0 if status=='PASS' else 2
    except Exception as exc:
        atomic(out/'STATUS.json',dict(status='BLOCKED',error=str(exc),completed=len(rows),worker_active=False))
        print(traceback.format_exc(),flush=True);package(out);return 2

def self_test():
    import numpy as np
    import tempfile
    x=np.array([[4.,1.,0.],[0.,3.,1.],[-2.,-1.,5.]])
    y=np.arange(3);m=np_metrics(x,y)
    require(m['accuracy']==1 and m['ce']>0 and np.isfinite(m['ece']),'Metric fixture')
    require(compare_logits(x,x.copy())['bitwise_equal'],'Identity fixture')
    z=x.copy();z[0,1]=8
    require(not compare_logits(z,x)['predictions_equal'],'Changed prediction must fail')
    z=x+0.1;require(not compare_logits(z,x)['within_tolerance'],'Logit drift must fail despite equal predictions')
    features=np.array([[1.,0.],[1.,0.],[1.,0.],[1.,0.],[-1.,0.],[-1.,0.]])
    g,counts,sums,sumsq=np_geometry(features,np.repeat(np.arange(3),2),3,1e-6,1e-6)
    require(g['within']==0 and abs(g['between']-8/9)<1e-12 and abs(g['total']-8/9)<1e-12,'Geometry counterexample/decomposition fixture')
    require(np.array_equal(counts,[2,2,2]),'Class counts')
    for fn in [lambda:np_metrics(np.full((3,3),np.nan),y),lambda:compare_logits(x,np.ones((2,2)))]:
        failed=False
        try:fn()
        except RuntimeError:failed=True
        require(failed,'Invalid inputs must fail')
    with tempfile.TemporaryDirectory() as tmp:
        d=Path(tmp);(d/'sample.txt').write_text('original')
        atomic(d/'DONE.json',{'payload':{'sample.txt':sha(d/'sample.txt')}})
        verify_payload(d);(d/'sample.txt').write_text('changed')
        failed=False
        try:verify_payload(d)
        except RuntimeError:failed=True
        require(failed,'Altered payload must fail')
    print('CPU_SELF_TEST_PASS: metrics, comparison failures, geometry, invalid inputs and altered-payload rejection. No GPU execution tested.')

def start(args):
    out=args.output.resolve();root=args.root.resolve()
    require(not contained(out,root) and not contained(root,out),'Output must be separate from original result tree')
    out.mkdir(parents=True,exist_ok=True)
    old=read(out/'STATUS.json') if (out/'STATUS.json').exists() else {}
    pid=old.get('pid') or (read(out/'PID.json').get('pid') if (out/'PID.json').exists() else None)
    if pid and Path('/proc') .exists():
        try:
            cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
            if b'--worker' in cmd and Path(__file__).name.encode() in cmd:print('Verification worker is already active.');return
        except OSError:pass
    candidates=[Path.home()/'caenl-v5.4.3/caenl_revision_pipeline_v5.4.3_c4_confirmatory/.venv/bin/python',Path.home()/'caenl-src/caenl_revision_pipeline_v5.3_ibm_rhel_l40s/.venv/bin/python',Path(sys.executable)]
    python=None;expected=read(root/'INPUT_BINDING.json')['environment']['versions']
    probe="import importlib.metadata,json;print(json.dumps({n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','scipy','Pillow')}))"
    for candidate in candidates:
        if not candidate.is_file():continue
        p=subprocess.run([str(candidate),'-B','-c',probe],capture_output=True,text=True)
        if p.returncode==0 and json.loads(p.stdout)==expected:python=candidate;break
    require(python is not None,'Original runtime not found; no packages or data were downloaded or changed')
    env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    atomic(out/'STATUS.json',dict(status='STARTING',time=time.time()))
    with (out/'run.log').open('a',buffering=1) as log:
        p=subprocess.Popen([str(python),'-B',str(Path(__file__).resolve()),'--worker','--root',str(root),'--output',str(out)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    atomic(out/'PID.json',dict(pid=p.pid,time=time.time()))
    print('STARTED checkpoint verification only; no training. You may disconnect.');print('Output: '+str(out))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT);p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    g=p.add_mutually_exclusive_group(required=True)
    for flag in ['start','status','self-test','worker']:g.add_argument('--'+flag,action='store_true')
    args=p.parse_args()
    if args.self_test:self_test()
    elif args.start:start(args)
    elif args.worker:return worker(args)
    else:
        path=args.output/'STATUS.json';state=read(path) if path.exists() else {'status':'NOT_STARTED'}
        pid=state.get('pid') or (read(args.output/'PID.json').get('pid') if (args.output/'PID.json').exists() else None)
        active=False
        if pid:
            try:
                cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
                active=b'--worker' in cmd and Path(__file__).name.encode() in cmd
            except OSError:pass
        state['worker_active']=active
        if not active and state['status'] in ('RUNNING','STARTING'):state['note']='No active verifier found; inspect the log and restart to resume completed checks.'
        print(json.dumps(state,indent=2))
        print('Log: '+str(args.output/'run.log'));print('Result archive: '+str(args.output/'CAENL_CHECKPOINT_REPLAY.zip'))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
