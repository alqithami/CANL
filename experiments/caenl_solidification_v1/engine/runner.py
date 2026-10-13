"""Resume-safe orchestration for content-disjoint replication and common-label ablations."""
import copy,gc,hashlib,json,math,os,shutil,time,zipfile
from pathlib import Path
import numpy as np
import torch
from support import Model,atom_json,atom_npz,atom_torch,read_json,sha256,digest,stratified_sample,seed_for,tensor_digest
from study import candidates,evaluated_methods,replay_methods,stage_specs,method_config,planned_counts,choose_controls
from training import run_phase,phase_steps
from control import calibrate
from acquisition import select_round
from analysis import primary_analysis
from checkpoint_replay import evaluate
from evaluation import infer

def require(ok,msg):
    if not ok:raise RuntimeError(msg)

def freeze(p,value):
    p=Path(p)
    if p.exists():require(read_json(p)==value,'Frozen record differs: '+str(p))
    else:atom_json(p,value)

def verify_phase(out,allow_pruned=True):
    out=Path(out);d=read_json(out/'DONE.json')
    for rel,h in d['payload'].items():
        p=out/rel
        if p.exists():require(sha256(p)==h,'Phase payload changed: '+str(p))
        else:
            require(allow_pruned and rel=='state.pt' and (out/'PRUNED_STATE.json').exists(),'Missing phase payload: '+str(p))
            rec=read_json(out/'PRUNED_STATE.json');require(rec['sha256']==h and rec['phase_binding']==d['binding'],'Prune receipt differs')
    return d

def prune_state(out,root):
    out=Path(out).resolve();root=Path(root).resolve();p=out/'state.pt'
    require(root in out.parents and out.name.startswith('round'),'Only new per-round states can be pruned')
    if not p.exists():return
    d=verify_phase(out,False)
    freeze(out/'PRUNED_STATE.json',dict(sha256=d['payload']['state.pt'],phase_binding=d['binding'],reason='Superseded optimizer state; traces and evaluation arrays retained'))
    p.unlink()

def verify_method(out):
    out=Path(out)
    if not (out/'DONE.json').exists():return False
    d=read_json(out/'DONE.json')
    for rel,h in d['payload'].items():require((out/rel).is_file() and sha256(out/rel)==h,'Method payload changed: '+str(out/rel))
    for p in (out/'rounds').glob('round*/DONE.json'):verify_phase(p.parent)
    for p in (out/'acquisitions').glob('round*/COMPLETE.json'):
        a=read_json(p);require(sha256(p.parent/'selection.npz')==a['selection_sha256'],'Acquisition changed')
        for x in a.get('passes',[]):require(sha256(p.parent/f"pass_{x['pass']:03d}.npz")==x['sha256'],'Candidate scores changed')
    return True

def load_model(factory,cfg,path,device):
    model=factory(cfg['num_classes']).to(device).float()
    if str(device).startswith('cuda'):model=model.to(memory_format=torch.channels_last)
    state=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(state['model'],strict=True)
    return model

def release():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()

def read_rows(root):
    result=[]
    for stage in ('tuning','confirmation','transfer'):
        for p in sorted((Path(root)/stage).glob('seed*/*/DONE.json')):
            if p.parent.name=='shared_initial':continue
            d=read_json(p);summary=p.parent/'summary.json'
            require(sha256(summary)==d['payload']['summary.json'],'Result summary changed')
            result.append(read_json(summary))
    return result

def export(root,cfg,status):
    rows=read_rows(root);record=dict(status=status,planned=planned_counts(cfg),completed_methods=len(rows),results=rows,analysis=primary_analysis(rows,cfg),scope=cfg['study']['scope'])
    atom_json(root/'RESULTS.json',record)
    lines=['CAENL solidification v1',f'Status: {status}',f'Method runs: {len(rows)}/{planned_counts(cfg)["method_runs"]}',cfg['study']['scope'],'']
    for c in record['analysis']['contrasts']:
        lines.append(f"{c['stage']} {c['target']} - {c['reference']}: {c['difference_pp']:+.4f} pp; 95% CI [{c['ci95_low_pp']:+.4f}, {c['ci95_high_pp']:+.4f}]; Holm t={c['holm_t_p']:.6g}, exact={c['holm_exact_p']:.6g}")
    lines+=['','Completion is an execution result, not confirmation of a hypothesis.']
    p=root/'REPORT.txt';tmp=p.with_suffix('.tmp');tmp.write_text('\n'.join(lines)+'\n');os.replace(tmp,p)
    return record

def make_split(cat,cfg,seed,sd):
    chosen=stratified_sample(np.arange(len(cat.labels)),cat.labels,cfg['initial_labels'],seed_for('initial_labels',seed))
    hold=stratified_sample(chosen,cat.labels,cfg['holdout_count'],seed_for('controller_holdout',seed));initial=np.setdiff1d(chosen,hold)
    require(np.bincount(cat.labels[initial],minlength=cfg['num_classes']).min()>=cfg['samples_per_class'],'Initial per-class support is insufficient for balanced batches')
    require(np.bincount(cat.labels[hold],minlength=cfg['num_classes']).min()>=2,'Feedback holdout lacks at least two examples per class')
    path=sd/'splits.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as a:
            require(np.array_equal(a['initial'],initial) and np.array_equal(a['holdout'],hold),'Seed split differs')
    else:atom_npz(path,initial=initial,holdout=hold,annotated_initial=chosen)
    return initial,hold

def run_seed(root,cfg,cat,stage,backbone,seed,methods,binding,progress,device,factory=None,prune=True):
    import functools
    root=Path(root);sd=root/stage/f'seed{seed}';sd.mkdir(parents=True,exist_ok=True)
    factory=factory or functools.partial(Model,backbone=backbone)
    initial,hold=make_split(cat,cfg,seed,sd)
    total_windows=sum(math.ceil(phase_steps(len(initial)+r*cfg['acquisition_per_round'],r,cfg)[0]/cfg['control_interval']) for r in range(1,cfg['rounds']+1))
    im=dict(name='shared_initial',regularizer='none',acquisition='none')
    ic=method_config(cfg,im,0,stage,backbone)
    _,initial_ckpt=run_phase(ic,cat,seed,im,0,initial,hold,None,None,sd/'shared_initial',binding,total_windows,progress,device,model_factory=factory)
    initial_hash=sha256(initial_ckpt);target_path=sd/'targets.json'
    if target_path.exists():
        tr=read_json(target_path);require(tr['initial_checkpoint_sha256']==initial_hash,'Calibration source differs');targets=tr['targets']
    else:
        model=load_model(factory,cfg,initial_ckpt,device);targets,values=calibrate(model,cat,initial,seed,cfg,device)
        atom_json(target_path,dict(targets=targets,anchor_statistics=values,initial_checkpoint_sha256=initial_hash));del model;release()
    # The entropy donor must finish before replay treatments. Remaining order is seed-specific.
    methods=sorted(methods,key=lambda m:(m['name']!='entropy',seed_for('solidification_order',stage,seed,m['name'])))
    for method in methods:
        out=sd/method['name'];out.mkdir(exist_ok=True)
        mb=digest(dict(campaign=binding,stage=stage,seed=seed,backbone=backbone,method=method,initial=initial_hash,split=sha256(sd/'splits.npz'),targets=sha256(target_path)))
        if verify_method(out):
            require(read_json(out/'DONE.json')['binding']==mb,'Completed method binding differs')
            if prune:
                for p in (out/'rounds').glob('round*/state.pt'):prune_state(p.parent,root)
            continue
        source=initial_ckpt;labeled=initial.copy();rounds=[];acquisitions=[]
        # Inspect committed rounds without needing superseded optimizer states.
        for phase in range(1,cfg['rounds']+1):
            pd=out/'rounds'/f'round{phase}';ap=out/'acquisitions'/f'round{phase}'
            if not (pd/'DONE.json').exists():break
            verify_phase(pd);result=read_json(pd/'summary.json');info=read_json(ap/'COMPLETE.json')
            require(sha256(ap/'selection.npz')==info['selection_sha256'],'Committed selection differs')
            with np.load(ap/'selection.npz') as a:
                require(np.array_equal(a['previous_labeled'],labeled),'Acquisition sequence differs');labeled=np.sort(np.r_[labeled,a['selected']])
            rounds.append(result);acquisitions.append(info);source=pd/'state.pt'
        first_unfinished=len(rounds)+1
        if first_unfinished<=cfg['rounds']:require(source.is_file(),'The latest resumable state is missing')
        for phase in range(first_unfinished,cfg['rounds']+1):
            c=method_config(cfg,method,phase,stage,backbone);ap=out/'acquisitions'/f'round{phase}';ap.mkdir(parents=True,exist_ok=True)
            if method['acquisition'] in ('replay','full'):
                if method['acquisition']=='replay':
                    donor=sd/'entropy'/'acquisitions'/f'round{phase}'
                    require(verify_method(sd/'entropy'),'Entropy replay donor incomplete')
                    with np.load(donor/'selection.npz') as a:
                        require(np.array_equal(a['previous_labeled'],labeled) and np.array_equal(a['control_holdout'],hold),'Replay labels differ');chosen=a['selected'].copy()
                    origin=sha256(donor/'selection.npz')
                else:chosen=np.setdiff1d(np.arange(len(cat.labels)),np.r_[labeled,hold]);origin=None
                if (ap/'selection.npz').exists():
                    with np.load(ap/'selection.npz') as a:require(np.array_equal(a['selected'],chosen) and np.array_equal(a['previous_labeled'],labeled),'Stored replay/full selection differs')
                else:atom_npz(ap/'selection.npz',selected=chosen,previous_labeled=labeled,control_holdout=hold)
                info=dict(selection_sha256=sha256(ap/'selection.npz'),selected=len(chosen),requested=len(chosen),passes=[],elapsed_seconds=0.,kind=method['acquisition'],donor_selection_sha256=origin,pool_true_labels_used_for_selection=False)
                freeze(ap/'COMPLETE.json',info)
            else:
                model=load_model(factory,c,source,device)
                chosen,info=select_round(model,cat,labeled,hold,method,seed,phase,c,device,ap,mb,sha256(source),progress)
                del model;release()
            if phase==1 and method['acquisition']=='entropy':freeze(sd/'FIRST_ENTROPY_QUERY.json',dict(selected_digest=digest(chosen.tolist()),initial_checkpoint_sha256=initial_hash))
            labeled=np.sort(np.r_[labeled,chosen]);require(len(labeled)==len(set(labeled)) and not np.intersect1d(labeled,hold).size,'Label budget overlap')
            expected=len(cat.labels)-len(hold) if method['acquisition']=='full' else len(initial)+phase*cfg['acquisition_per_round']
            require(len(labeled)==expected,'Label budget incorrect')
            previous=source
            result,source=run_phase(c,cat,seed,method,phase,labeled,hold,source,targets,out/'rounds'/f'round{phase}',mb,total_windows,progress,device,model_factory=factory)
            verify_phase(source.parent,False)
            rounds.append(result);acquisitions.append(info)
            if prune and phase>1:prune_state(previous.parent,root)
        final=out/'rounds'/f"round{cfg['rounds']}";summary=rounds[-1]
        # Retain portable FP32 final weights and verify by reloading before marking complete.
        final_model=out/'FINAL_MODEL.pt'
        if not final_model.exists():
            require(source.is_file(),'Final state unavailable before model extraction')
            state=torch.load(source,map_location='cpu',weights_only=False)
            require(tensor_digest(state['model'])==summary['final_model_weights_sha256'],'Final tensor digest mismatch')
            atom_torch(final_model,dict(model=state['model'],seed=seed,method=method['name'],binding=mb,source_checkpoint_sha256=summary['checkpoint_sha256']));del state
        saved=torch.load(final_model,map_location='cpu',weights_only=False)
        require(saved['binding']==mb and tensor_digest(saved['model'])==summary['final_model_weights_sha256'],'Portable model differs');del saved
        model=load_model(factory,cfg,final_model,device);model.eval()
        replay={'holdout':evaluate(model,cat,cfg,infer,'train',hold,final/'feedback.npz',summary['feedback'],device)}
        if stage!='tuning':replay['official_validation']=evaluate(model,cat,cfg,infer,'val',np.arange(len(cat.val_labels)),final/'validation.npz',summary['validation'],device)
        require(all(r['status']=='PASS' for r in replay.values()),'Reloaded final checkpoint differs from saved evaluation')
        atom_json(out/'INFERENCE_REPLAY.json',replay);del model;release()
        # Shared acquisitions, sampler and augmentation streams must really match for mechanism inference.
        if method['acquisition']=='replay':
            for phase,r in enumerate(rounds,1):
                donor=read_json(sd/'entropy'/'rounds'/f'round{phase}'/'summary.json')
                require(r['batch_chain_sha256']==donor['batch_chain_sha256'],'Replay did not preserve minibatch IDs')
        result=dict(stage=stage,backbone=backbone,seed=seed,method=method['name'],method_spec=method,binding=mb,
          holdout=summary['feedback'],official_validation=summary['validation'],initial_checkpoint_sha256=initial_hash,
          final_model_sha256=sha256(final_model),rounds=rounds,acquisitions=acquisitions,
          training_seconds=sum(r['phase_training_elapsed_seconds'] for r in rounds),acquisition_seconds=sum(r['elapsed_seconds'] for r in acquisitions),
          annotation_count=summary['annotation_count'],training_steps=sum(r['steps'] for r in rounds),final_checkpoint_reloaded=True)
        atom_json(out/'summary.json',result)
        files=[out/'summary.json',final_model,out/'INFERENCE_REPLAY.json']+list((out/'rounds').glob('round*/DONE.json'))+list((out/'acquisitions').glob('round*/COMPLETE.json'))
        atom_json(out/'DONE.json',dict(binding=mb,payload={p.relative_to(out).as_posix():sha256(p) for p in files}))
        if prune:prune_state(final,root)
        export(root,cfg,'RUNNING');progress(f'METHOD COMPLETE: {stage} seed{seed} {method["name"]}')

def run_study(root,cfg,cat,binding,progress,device,factory=None,prune=True):
    root=Path(root)
    for stage,backbone,seeds in stage_specs(cfg):
        if stage=='tuning':methods=candidates(cfg)
        else:
            selection=read_json(root/'SELECTION.json');methods=evaluated_methods(selection)
            if stage=='confirmation':methods+=replay_methods(selection)
        for seed in seeds:
            progress(f'STAGE {stage}: {backbone} seed{seed}')
            run_seed(root,cfg,cat,stage,backbone,seed,methods,binding,progress,device,factory,prune)
        if stage=='tuning':
            result=choose_controls([r for r in read_rows(root) if r['stage']=='tuning'],cfg);result['campaign_binding']=binding
            freeze(root/'SELECTION.json',result)
    result=export(root,cfg,'COMPLETE')
    require(result['completed_methods']==planned_counts(cfg)['method_runs'] and result['analysis']['status']=='COMPLETE','Incomplete campaign')
    for p in root.glob('*/seed*/*/DONE.json'):
        if p.parent.name!='shared_initial':require(verify_method(p.parent),'Method verification failed')
    return result
