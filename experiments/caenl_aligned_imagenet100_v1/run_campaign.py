#!/usr/bin/env python3
"""One local, frozen, independent-seed campaign. Reads existing data; writes a NEW result tree."""
from __future__ import annotations
import argparse,copy,gc,hashlib,importlib.metadata,json,math,os,platform,shutil,sys,tarfile,time,traceback
from pathlib import Path
import numpy as np
import torch
from support import (Catalog,Model,read_json,atom_json,atom_npz,sha256,digest,seed_for,seed_all,stratified_sample,utc)
from training import run_phase,phase_steps
from control import calibrate
from acquisition import select_round
from evaluation import autoattack
from reporting import make_report

HERE=Path(__file__).resolve().parent

def load_cfg():return read_json(HERE/'protocol.json')
def verify_package():
    files={}
    for line in (HERE/'SHA256SUMS').read_text().splitlines():
        h,rel=line.split('  ',1);p=HERE/rel
        if '..' in Path(rel).parts or not p.is_file() or sha256(p)!=h:raise RuntimeError('Package integrity failure: '+rel)
        files[rel]=h
    return files

def verify_autoattack(expected):
    from autoattack import AutoAttack
    for dist in importlib.metadata.distributions():
        if dist.metadata.get('Name','').lower().replace('_','-') in ('autoattack','auto-attack'):
            data=dist.read_text('direct_url.json')
            if data:
                commit=json.loads(data).get('vcs_info',{}).get('commit_id')
                if commit==expected:return {'distribution':dist.metadata['Name'],'commit':commit}
                if commit:raise RuntimeError('Installed AutoAttack commit differs from frozen protocol: '+commit)
    raise RuntimeError('Cannot verify installed pinned AutoAttack direct_url metadata; do not silently substitute another attack')

def verify_done(out,binding):
    p=Path(out)/'DONE.json'
    if not p.exists():return False
    obj=read_json(p)
    if obj['binding']!=binding:raise RuntimeError('Completed method binding mismatch')
    for name,h in obj['payload'].items():
        f=Path(out)/name
        if not f.is_file() or sha256(f)!=h:raise RuntimeError('Completed method artifact changed: '+str(f))
    return True

def verify_completed_phases(out):
    for d in sorted((Path(out)/'acquisitions').glob('round*/COMPLETE.json')):
        r=read_json(d)
        if sha256(d.parent/'selection.npz')!=r['selection_sha256']:
            raise RuntimeError('Acquisition selection changed: '+str(d.parent))
        for rec in r['passes']:
            p=d.parent/f"pass_{int(rec['pass']):03d}.npz"
            if not p.is_file() or sha256(p)!=rec['sha256']:
                raise RuntimeError('Acquisition pass changed: '+str(p))
    for d in sorted((Path(out)/'rounds').glob('round*/DONE.json')):
        r=read_json(d)
        for p,h in r['payload'].items():
            if sha256(d.parent/p)!=h:raise RuntimeError('Phase artifact changed: '+str(d.parent/p))

def export(root,cfg,binding):
    root=Path(root);ar=Path(cfg['archive_root']);ar.mkdir(parents=True,exist_ok=True)
    snap=root/'source';snap.mkdir(exist_ok=True)
    for rel in verify_package():
        p=snap/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(HERE/rel,p)
    inventory=[];paths=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.name in ('RUNNING.json','FAILED.json','COMPLETE.json','SHA256SUMS') or p.suffix=='.tmp':continue
        include=p.suffix not in ('.pt','.pth','.ckpt','.safetensors')
        inventory.append({'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'included':include})
        if include:paths.append(p)
    atom_json(root/'reports/full_inventory.json',inventory);paths.append(root/'reports/full_inventory.json');paths=sorted(set(paths))
    manifest=''.join(f'{sha256(p)}  {p.relative_to(root).as_posix()}\n' for p in paths);(root/'SHA256SUMS').write_text(manifest);paths.append(root/'SHA256SUMS')
    target=ar/'imagenet100-aligned-results.tar.gz';part=target.with_name(target.name+'.partial')
    with tarfile.open(part,'w:gz') as tf:
        for p in paths:tf.add(p,arcname=cfg['campaign']+'/'+p.relative_to(root).as_posix(),recursive=False)
    # Read every archived payload and verify its recorded digest before publication of the filename.
    expected={p.relative_to(root).as_posix():sha256(p) for p in paths}
    with tarfile.open(part,'r:gz') as tf:
        seen=set()
        for m in tf:
            rel=m.name.split('/',1)[1]
            if not m.isfile() or rel not in expected:raise RuntimeError('Archive contains an unexpected member')
            h=hashlib.sha256();f=tf.extractfile(m)
            for b in iter(lambda:f.read(4<<20),b''):h.update(b)
            if h.hexdigest()!=expected[rel]:raise RuntimeError('Archive verification failure: '+rel)
            seen.add(rel)
        if seen!=set(expected):raise RuntimeError('Archive missing payload files')
    os.replace(part,target);h=sha256(target);target.with_name(target.name+'.sha256').write_text(f'{h}  {target.name}\n')
    (ar/'LATEST_caenl-imagenet100-aligned-confirmation-v1.txt').write_text(str(target)+'\n')
    atom_json(root/'COMPLETE.json',{'status':'COMPLETE','campaign':cfg['campaign'],'binding':binding,'methods_complete':len(cfg['methods'])*len(cfg['seeds']),'archive':str(target),'sha256':h,'completed_utc':utc(),'checkpoint_deletion':False})
    print('EXPORT PASS\nArchive: '+str(target)+'\nALIGNED IMAGENET100 COMPLETE',flush=True)

def gpu_contract(cfg,root,device):
    """One real ResNet-50 aligned backward plus a real official attack on a tiny toy.
    No checkpoint/data from earlier runs is touched; no scientific result is created.
    """
    from aligned import class_scatter,aligned_loss
    from autoattack import AutoAttack
    from support import amp,optimizer_for
    seed_all(991071);m=Model(cfg['num_classes']).to(device).to(memory_format=torch.channels_last)
    n=cfg['batch_classes']*cfg['samples_per_class'];x=torch.rand(n,3,224,224,device=device).contiguous(memory_format=torch.channels_last)
    y=torch.arange(cfg['batch_classes'],device=device).repeat_interleave(cfg['samples_per_class'])
    opt=optimizer_for(m,cfg);opt.zero_grad(set_to_none=True)
    with amp(device):lo,z=m.forward_features(x);ce=torch.nn.functional.cross_entropy(lo.float(),y)
    targets={}
    for l in cfg['layers']:
        q=class_scatter(z[l].detach(),y);targets[l]={'log_kappa_target':float(q['log_kappa'])-.1,'between_floor':max(1e-6,float(q['between'])*.5)}
    reg,_=aligned_loss(z,y,targets,{l:cfg['lambda_fixed'] for l in cfg['layers']},cfg['alignment'])
    (ce+reg).backward();torch.nn.utils.clip_grad_norm_(m.parameters(),cfg['clip_grad_norm'],error_if_nonfinite=True);opt.step()
    if not torch.isfinite(ce+reg):raise RuntimeError('CUDA training preflight failed')
    peak=torch.cuda.max_memory_allocated();del m,opt,x,y,z,lo,ce,reg;gc.collect();torch.cuda.empty_cache()
    class Toy(torch.nn.Module):
        def forward(self,x):return x.flatten(1)[:,:10]
    toy=Toy().to(device).eval();inputs=torch.full((2,3,4,4),.5,device=device);inputs[:,0,0,0]=.51;labels=toy(inputs).argmax(1)
    adv=AutoAttack(toy,norm='Linf',eps=8/255,version='standard',device=str(device),seed=42,verbose=False)
    attacked=adv.run_standard_evaluation(inputs,labels,bs=2)
    if (attacked-inputs).abs().max()>8/255+2e-6 or torch.equal(toy(attacked).argmax(1),labels):raise RuntimeError('Live AutoAttack sanity check failed')
    atom_json(root/'CUDA_PREFLIGHT.json',{'status':'PASS','batch':n,'resnet50_aligned_backward':True,'standard_autoattack_live_toy':True,'peak_allocated_bytes':peak,'scientific_training':False})
    del toy,inputs,attacked,adv;gc.collect();torch.cuda.empty_cache()
    print('CUDA PREFLIGHT PASS: batch-128 ResNet-50 aligned backward and live standard AutoAttack',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:
        from self_test import run
        import contextlib,io
        buf=io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):checks=run()
        except BaseException:
            print(buf.getvalue());raise
        print(f'PRECHECK PASS: {len(checks)} local CPU checks. No scientific run was counted.');return 0
    cfg=load_cfg();root=Path(cfg['result_root']);root.mkdir(parents=True,exist_ok=True);phase='preflight'
    def progress(msg):
        nonlocal phase
        phase=msg;atom_json(root/'RUNNING.json',{'pid':os.getpid(),'time':utc(),'message':msg,'campaign':cfg['campaign']})
    try:
        
        if (root/'FAILED.json').exists():
            from support import append_json
            append_json(root/'failure_history.jsonl',read_json(root/'FAILED.json'));(root/'FAILED.json').unlink()
        progress('Verifying frozen protocol, CUDA, and mounted storage')
        package=verify_package()
        if not Path('/mnt/caenl').is_mount():raise RuntimeError('Persistent volume not mounted')
        if shutil.disk_usage(root).free<150*1024**3:raise RuntimeError('At least 150 GiB free required for saved checkpoints; no deletion will be attempted')
        if not torch.cuda.is_available() or 'L40S' not in torch.cuda.get_device_name(0):raise RuntimeError('Expected accessible NVIDIA L40S')
        if not str(torch.__version__).startswith('2.6.') or str(torch.version.cuda)!='12.4':raise RuntimeError('Expected tested PyTorch 2.6 / CUDA 12.4 environment; no installation attempted')
        aa_info=verify_autoattack(cfg['autoattack']['commit']);device=torch.device('cuda:0')
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        cat=Catalog(cfg);binding=digest({'protocol':cfg,'source_files':package})
        if (root/'INPUT_BINDING.json').exists() and read_json(root/'INPUT_BINDING.json')['binding']!=binding:raise RuntimeError('Source/protocol changed: refusing to mix runs')
        atom_json(root/'INPUT_BINDING.json',{'binding':binding,'protocol':cfg,'source_files':package,'environment':{'python':sys.version,'torch':torch.__version__,'torchvision':importlib.metadata.version('torchvision'),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'platform':platform.platform(),'autoattack':aa_info}})
        if (root/'COMPLETE.json').exists():
            info=read_json(root/'COMPLETE.json')
            if sha256(info['archive'])!=info['sha256']:raise RuntimeError('Completed archive checksum mismatch')
            print('ALREADY COMPLETE: '+info['archive']);return 0
        gpu_contract(cfg,root,device)
        checked=cat.verify_images(progress);atom_json(root/'DATA_VERIFIED.json',{'count':checked,'manifest':cfg['expected_image_manifest_sha256'],'time':utc()})
        for old in ['FAILED.json']:
            if (root/old).exists():(root/old).unlink()
        for seed in cfg['seeds']:
            sd=root/f'seed{seed}';sd.mkdir(exist_ok=True);split=sd/'splits.npz'
            selected=stratified_sample(np.arange(len(cat.labels)),cat.labels,cfg['initial_labels'],seed_for('initial_labels',seed))
            holdout=stratified_sample(selected,cat.labels,cfg['holdout_count'],seed_for('controller_holdout',seed));initial=np.setdiff1d(selected,holdout)
            if split.exists():
                a=dict(np.load(split,allow_pickle=False))
                if not np.array_equal(a['initial'],initial) or not np.array_equal(a['holdout'],holdout):raise RuntimeError('Stored seed split mismatch')
            else:atom_npz(split,initial=initial,holdout=holdout,annotated_initial=selected)
            total_windows=sum(math.ceil(phase_steps(len(initial)+p*cfg['acquisition_per_round'],p,cfg)[0]/cfg['control_interval']) for p in range(1,cfg['rounds']+1))
            init_method={'name':'shared_initial','acquisition':'none','regularizer':'none'}
            initial_summary,initial_ckpt=run_phase(cfg,cat,seed,init_method,0,initial,holdout,None,None,sd/'shared_initial',binding,total_windows,progress,device)
            targets_file=sd/'targets.json'
            if targets_file.exists():
                tr=read_json(targets_file)
                if tr['initial_checkpoint_sha256']!=sha256(initial_ckpt):raise RuntimeError('Calibration reference changed')
                targets=tr['targets']
            else:
                progress(f'Calibrating fixed per-seed targets for seed{seed}')
                model=Model(cfg['num_classes']).to(device).float().to(memory_format=torch.channels_last);ck=torch.load(initial_ckpt,map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);del ck
                targets,values=calibrate(model,cat,initial,seed,cfg,device);atom_json(targets_file,{'targets':targets,'anchor_statistics':values,'initial_checkpoint_sha256':sha256(initial_ckpt),'validation_used':False,'frozen_for_all_rounds':True})
                del model;gc.collect();torch.cuda.empty_cache()
            for method in cfg['methods']:
                out=sd/method['name'];out.mkdir(exist_ok=True)
                mb=digest({'binding':binding,'seed':seed,'method':method,'split':sha256(split),'targets':sha256(targets_file)})
                if verify_done(out,mb):verify_completed_phases(out);continue
                print(f'\n[method] seed{seed} {method["name"]}; five acquisitions, 10% to 60%',flush=True)
                source=initial_ckpt;labeled=initial.copy();rounds=[initial_summary];acquisitions=[]
                for phase_id in range(1,cfg['rounds']+1):
                    progress(f'{method["name"]} seed{seed}: acquisition {phase_id}/5')
                    model=Model(cfg['num_classes']).to(device).float().to(memory_format=torch.channels_last);ck=torch.load(source,map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);del ck
                    chosen,ai=select_round(model,cat,labeled,holdout,method,seed,phase_id,cfg,device,out/'acquisitions'/f'round{phase_id}',mb,sha256(source),progress)
                    del model;gc.collect();torch.cuda.empty_cache();labeled=np.sort(np.concatenate([labeled,chosen]));acquisitions.append(ai)
                    expected=len(initial)+cfg['acquisition_per_round']*phase_id
                    if len(labeled)!=expected or len(np.unique(labeled))!=expected or np.intersect1d(labeled,holdout).size:raise RuntimeError('Acquisition integrity failure')
                    met,source=run_phase(cfg,cat,seed,method,phase_id,labeled,holdout,source,targets,out/'rounds'/f'round{phase_id}',mb,total_windows,progress,device);rounds.append(met)
                if method['regularizer']=='aligned_full' and rounds[-1]['learned_updates_cumulative']<cfg['minimum_policy_updates_per_completed_run']:raise RuntimeError('Missing genuine Full MACC regularization updates')
                progress(f'{method["name"]} seed{seed}: final standard AutoAttack on all 5,000 images')
                model=Model(cfg['num_classes']).to(device).float().to(memory_format=torch.channels_last);ck=torch.load(source,map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);del ck
                model.requires_grad_(False)
                before=sha256(source);rob=autoattack(model,cat,cfg,device,out/'autoattack',mb,before,progress)
                if sha256(source)!=before:raise RuntimeError('Evaluation changed source checkpoint')
                del model;gc.collect();torch.cuda.empty_cache()
                summary={'binding':mb,'seed':seed,'method':method['name'],'role':cfg['role'],'rounds':rounds,'acquisitions':acquisitions,'autoattack':rob,'final_checkpoint_sha256':before,'fixed_acquisition_threshold':.9,'learned_acquisition_head':False}
                atom_json(out/'summary.json',summary)
                payload={'summary.json':sha256(out/'summary.json'),'autoattack/predictions.npz':sha256(out/'autoattack/predictions.npz')}
                atom_json(out/'DONE.json',{'status':'COMPLETE','binding':mb,'payload':payload,'time':utc()})
                rows=make_report(root,cfg,complete=False);print(f'[method complete] {len(rows)}/{len(cfg["seeds"])*len(cfg["methods"])}; {method["name"]}/seed{seed}',flush=True)
            print(f'[seed complete] {seed}',flush=True)
        progress('Final full-matrix reporting and archive verification')
        for seed in cfg['seeds']:
            for method in cfg['methods']:
                out=root/f'seed{seed}'/method['name']
                expected=digest({'binding':binding,'seed':seed,'method':method,'split':sha256(root/f'seed{seed}'/'splits.npz'),'targets':sha256(root/f'seed{seed}'/'targets.json')})
                if not verify_done(out,expected):raise RuntimeError('Incomplete expected method: '+str(out))
                verify_completed_phases(out)
        make_report(root,cfg,complete=True)
        atom_json(root/'INTEGRITY.json',{'status':'COMPLETE','binding':binding,'independent_seeds':cfg['seeds'],'expected_method_runs':54,'negative_outcomes_retained':True,'trained_from_scratch':True,'input_checkpoints_from_development':False,'dataset':cfg['dataset_root']})
        export(root,cfg,binding);return 0
    except BaseException as exc:
        atom_json(root/'FAILED.json',{'status':'FAILED','time':utc(),'pid':os.getpid(),'operation':phase,'error':repr(exc),'traceback':traceback.format_exc()})
        raise
    finally:
        p=root/'RUNNING.json'
        if p.exists():p.unlink()

if __name__=='__main__':raise SystemExit(main())
