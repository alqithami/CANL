#!/usr/bin/env python3
"""One frozen, resumable campaign; all writes stay in its new result tree."""
from __future__ import annotations
import argparse, contextlib, copy, fcntl, functools, gc, hashlib, importlib.metadata
import io, json, math, os, platform, shutil, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from support import (Catalog, Model, atom_json, atom_npz, read_json, sha256, digest, seed_for,
                     seed_all, stratified_sample, utc, tensor_digest)
from training import run_phase, phase_steps
from control import calibrate
from acquisition import select_round
from evaluation import metrics_from_logits
from study import candidates, evaluated_methods, stage_specs, method_config, choose_controls, planned_counts
from analysis import primary_analysis

HERE=Path(__file__).resolve().parent


def require(value,message):
    if not value:raise RuntimeError(message)


def atomic_text(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w') as f:f.write(text);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def freeze(path,value):
    path=Path(path)
    if path.exists():require(read_json(path)==value,'Frozen record differs: '+str(path))
    else:atom_json(path,value)


def verify_source():
    manifest=read_json(HERE/'SOURCE_MANIFEST.json')
    for rel,h in manifest.items():
        p=Path(rel)
        require(not p.is_absolute() and '..' not in p.parts,'Unsafe source manifest path')
        require((HERE/p).is_file() and sha256(HERE/p)==h,'Source integrity failure: '+rel)
    return manifest


def verify_payload(directory,record):
    directory=Path(directory)
    for rel,h in record['payload'].items():
        p=Path(rel)
        require(not p.is_absolute() and '..' not in p.parts,'Unsafe payload path')
        require((directory/p).is_file() and sha256(directory/p)==h,'Artifact changed: '+str(directory/p))


def verify_method(out,binding=None):
    out=Path(out)
    if not (out/'DONE.json').exists():return False
    done=read_json(out/'DONE.json')
    if binding is not None:require(done['binding']==binding,'Method binding differs: '+str(out))
    verify_payload(out,done)
    for p in sorted((out/'rounds').glob('round*/DONE.json')):verify_payload(p.parent,read_json(p))
    for p in sorted((out/'acquisitions').glob('round*/COMPLETE.json')):
        r=read_json(p)
        require(sha256(p.parent/'selection.npz')==r['selection_sha256'],'Selection digest differs')
        for rec in r['passes']:
            require(sha256(p.parent/f"pass_{rec['pass']:03d}.npz")==rec['sha256'],'Candidate pass changed')
    return True


def audit_metric(path,record,labels,ids):
    with np.load(path,allow_pickle=False) as a:
        require(np.array_equal(a['indices'],ids) and np.array_equal(a['labels'],labels),'Evaluation ID/label mismatch')
        require(np.array_equal(a['predictions'],a['logits'].argmax(1)),'Predictions do not match logits')
        m=metrics_from_logits(a['logits'],a['labels'])
        for k in ('n','accuracy','ce','ece','top5_accuracy'):
            require(abs(m[k]-record[k])<=1e-10,'Evaluation arithmetic mismatch: '+k)
    return True


def compact_result(summary):
    return {k:summary[k] for k in ('stage','backbone','seed','method','method_spec','holdout',
            'official_validation','final_checkpoint_sha256','initial_checkpoint_sha256',
            'first_acquisition_digest','phase_duration_seconds','train_compute_seconds','acquisition_seconds')}


def read_rows(root):
    rows=[]
    for stage in ('tuning','confirmation','transfer'):
        for p in sorted((Path(root)/stage).glob('seed*/*/DONE.json')):
            if p.parent.name=='shared_initial':continue
            done=read_json(p)
            if done.get('record_type')!='method_complete':continue
            summary_path=p.parent/'summary.json'
            require(sha256(summary_path)==done['payload']['summary.json'],'Completed summary digest mismatch')
            rows.append(compact_result(read_json(summary_path)))
    return rows


def export_report(root,cfg,status):
    root=Path(root);rows=read_rows(root);counts=planned_counts(cfg)
    selection=read_json(root/'SELECTION.json') if (root/'SELECTION.json').exists() else None
    inference=primary_analysis(rows,cfg)
    record=dict(campaign=cfg['campaign'],status=status,counts=counts,completed_method_runs=len(rows),
                seed_sets={k:cfg['study'][k] for k in ('tuning_seeds','confirmation_seeds','transfer_seeds')},
                selection=selection,primary_analysis=inference,per_seed=rows,
                protocol_sha256=sha256(HERE/'protocol.json'),
                input_binding=read_json(root/'INPUT_BINDING.json') if (root/'INPUT_BINDING.json').exists() else None,
                verification=read_json(root/'VERIFICATION.json') if (root/'VERIFICATION.json').exists() else None,
                scientific_scope=cfg['study']['claim_scope'])
    atom_json(root/'results.json',record)
    lines=['# CAENL feedback study v1', '', f'Status: **{status}**',
           f'Completed method runs: {len(rows)}/{counts["method_runs"]}. No original runs are repeated or overwritten.', '',
           'The study compares unchanged MACC-Lite with separately tuned fixed and preset-schedule controls,',
           'then transfers those frozen settings from ResNet-50 to ResNet-18 on ImageNet-100.',
           'Tuning uses the budgeted training holdout; official validation is excluded from model selection.', '']
    if selection:
        lines+=['Selected fixed control: `'+selection['fixed']['name']+'`.',
                'Selected open-loop control: `'+selection['schedule']['name']+'`.',
                'Selection was frozen before confirmation; it does not change during backbone transfer.', '']
    lines+=['| Stage | Method | Completed seeds | Mean top-1 (%) | Seed SD (pp) |',
            '|---|---|---:|---:|---:|']
    for stage,_,_ in stage_specs(cfg):
        for method in sorted({r['method'] for r in rows if r['stage']==stage}):
            rr=[r for r in rows if r['stage']==stage and r['method']==method]
            endpoint='holdout' if stage=='tuning' else 'official_validation'
            vals=[100*r[endpoint]['accuracy'] for r in rr]
            sd=f'{np.std(vals,ddof=1):.4f}' if len(vals)>1 else '—'
            lines.append(f'| {stage} | {method} | {len(vals)} | {np.mean(vals):.4f} | {sd} |')
    lines+=['', 'Tuning rows above are selection scores, not independent performance estimates.', '',
            '| Backbone stage | Target minus reference | Gain (pp) | Individual 95% CI | Holm t p | Holm exact p |',
            '|---|---|---:|---|---:|---:|']
    for r in inference['contrasts']:
        lines.append(f"| {r['stage']} | {r['target']} − {r['reference']} | {r['difference_pp']:+.4f} | "
                     f"[{r['ci95_low_pp']:+.4f}, {r['ci95_high_pp']:+.4f}] | {r['holm_t_p']:.6g} | {r['holm_exact_p']:.6g} |")
    if inference['status']!='COMPLETE':lines+=['', 'Inferential tests withheld until all six planned contrasts are available.']
    lines+=['', 'The six contrasts form one Holm family across both backbones, separately for each test type.',
            'Intervals are individual, unadjusted paired-t intervals. Small-sample and sign-exchangeability assumptions remain.',
            'Fresh seeds do not make this an untouched dataset. There is no cross-dataset, robustness, or isolated-alignment claim.',
            'A negative or uncertain result is retained; completion does not imply publication readiness.', '',
            '## Complete compact numerical record', '', '```json', json.dumps(record,indent=2,allow_nan=False), '```','']
    path=root/'caenl-feedback-study-v1.md';atomic_text(path,'\n'.join(lines))
    atomic_text(path.with_suffix('.md.sha256'),f'{sha256(path)}  {path.name}\n')
    return record


def make_split(cat,cfg,seed,path):
    selected=stratified_sample(np.arange(len(cat.labels)),cat.labels,cfg['initial_labels'],seed_for('initial_labels',seed))
    holdout=stratified_sample(selected,cat.labels,cfg['holdout_count'],seed_for('controller_holdout',seed))
    initial=np.setdiff1d(selected,holdout)
    if path.exists():
        with np.load(path,allow_pickle=False) as a:
            for k,v in [('initial',initial),('holdout',holdout),('annotated_initial',selected)]:
                require(np.array_equal(a[k],v),'Stored split differs: '+k)
    else:atom_npz(path,initial=initial,holdout=holdout,annotated_initial=selected)
    return initial,holdout


def load_model(factory,cfg,checkpoint,device):
    model=factory(cfg['num_classes']).to(device).float()
    if str(device).startswith('cuda'):model=model.to(memory_format=torch.channels_last)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'],strict=True);del state
    return model


def release():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def run_seed(root,cfg,cat,stage,backbone,seed,methods,binding,progress,device,model_factory=None):
    sd=Path(root)/stage/f'seed{seed}';sd.mkdir(parents=True,exist_ok=True)
    initial,holdout=make_split(cat,cfg,seed,sd/'splits.npz')
    factory=model_factory or functools.partial(Model,backbone=backbone)
    total_windows=sum(math.ceil(phase_steps(len(initial)+p*cfg['acquisition_per_round'],p,cfg)[0]/cfg['control_interval'])
                      for p in range(1,cfg['rounds']+1))
    im=dict(name='shared_initial',acquisition='none',regularizer='none')
    initial_cfg=method_config(cfg,im,0,stage,backbone)
    initial_summary,initial_ckpt=run_phase(initial_cfg,cat,seed,im,0,initial,holdout,None,None,
                         sd/'shared_initial',binding,total_windows,progress,device,model_factory=factory)
    initial_hash=sha256(initial_ckpt)
    target_path=sd/'targets.json'
    if target_path.exists():
        tr=read_json(target_path)
        require(tr['initial_checkpoint_sha256']==initial_hash,'Calibration reference changed')
        targets=tr['targets']
    else:
        progress(f'{stage} {backbone} seed{seed}: calibrating shared training anchors')
        model=load_model(factory,cfg,initial_ckpt,device)
        targets,values=calibrate(model,cat,initial,seed,cfg,device)
        atom_json(target_path,dict(targets=targets,anchor_statistics=values,
                  initial_checkpoint_sha256=initial_hash,official_validation_used=False))
        del model;release()
    ordered=sorted(methods,key=lambda m:seed_for('feedback_study_order',stage,seed,m['name']))
    for method in ordered:
        out=sd/method['name'];out.mkdir(exist_ok=True)
        mb=digest(dict(campaign=binding,stage=stage,backbone=backbone,seed=seed,method=method,
                       split=sha256(sd/'splits.npz'),targets=sha256(target_path),initial=initial_hash))
        if verify_method(out,mb):continue
        source=initial_ckpt;labeled=initial.copy();rounds=[];acquisitions=[];first=None
        print(f'\n[method] {stage} {backbone} seed{seed} {method["name"]}',flush=True)
        for phase in range(1,cfg['rounds']+1):
            c=method_config(cfg,method,phase,stage,backbone)
            model=load_model(factory,c,source,device)
            chosen,info=select_round(model,cat,labeled,holdout,method,seed,phase,c,device,
                                    out/'acquisitions'/f'round{phase}',mb,sha256(source),progress)
            del model;release()
            if phase==1:
                first=digest(chosen.tolist())
                freeze(sd/'FIRST_ACQUISITION.json',dict(selected_indices_digest=first,initial_checkpoint_sha256=initial_hash))
            labeled=np.sort(np.concatenate([labeled,chosen]));acquisitions.append(info)
            require(len(labeled)==len(initial)+phase*c['acquisition_per_round'] and
                    len(np.unique(labeled))==len(labeled) and not np.intersect1d(labeled,holdout).size,
                    'Acquisition budget or holdout exclusion failed')
            result,source=run_phase(c,cat,seed,method,phase,labeled,holdout,source,targets,
                         out/'rounds'/f'round{phase}',mb,total_windows,progress,device,model_factory=factory)
            audit_metric(out/'rounds'/f'round{phase}'/'feedback.npz',result['feedback'],cat.labels[holdout],holdout)
            if result['validation'] is not None:
                audit_metric(out/'rounds'/f'round{phase}'/'validation.npz',result['validation'],cat.val_labels,np.arange(len(cat.val_labels)))
            require(stage!='tuning' or result['validation'] is None,'Official validation leaked into tuning')
            rounds.append(result)
        summary=dict(binding=mb,stage=stage,backbone=backbone,seed=seed,method=method['name'],method_spec=method,
                     initial_checkpoint_sha256=initial_hash,first_acquisition_digest=first,
                     holdout=rounds[-1]['feedback'],official_validation=rounds[-1]['validation'],
                     final_checkpoint_sha256=sha256(source),rounds=rounds,acquisitions=acquisitions,
                     phase_duration_seconds=sum(r['phase_training_elapsed_seconds'] for r in rounds),
                     train_compute_seconds=sum(r['train_compute_seconds'] for r in rounds),
                     acquisition_seconds=sum(r['elapsed_seconds'] for r in acquisitions))
        atom_json(out/'summary.json',summary)
        paths=[out/'summary.json']+list((out/'rounds').glob('round*/DONE.json'))+list((out/'acquisitions').glob('round*/COMPLETE.json'))
        atom_json(out/'DONE.json',dict(record_type='method_complete',binding=mb,time=utc(),
                 payload={p.relative_to(out).as_posix():sha256(p) for p in sorted(paths)}))
        export_report(root,cfg,'RUNNING')
        print(f'[method complete] {stage} seed{seed} {method["name"]}',flush=True)


def run_study(root,cfg,cat,binding,progress,device,model_factory=None):
    root=Path(root)
    # Never select from an incomplete grid, even if partial results appear favorable.
    for stage,backbone,seeds in stage_specs(cfg):
        if stage=='tuning':methods=candidates(cfg)
        else:
            require((root/'SELECTION.json').is_file(),'Control selection is not frozen')
            selection=read_json(root/'SELECTION.json')
            require(selection['campaign_binding']==binding,'Selection binding changed')
            methods=evaluated_methods(selection)
        for seed in seeds:
            progress(f'STAGE {stage}: {backbone} seed{seed}')
            run_seed(root,cfg,cat,stage,backbone,seed,methods,binding,progress,device,model_factory)
        if stage=='tuning':
            rows=[r for r in read_rows(root) if r['stage']=='tuning']
            result=choose_controls(rows,cfg)
            result.update(campaign_binding=binding,tuning_records_digest=digest(rows),
                          planned_confirmation_seeds=cfg['study']['confirmation_seeds'],
                          planned_transfer_seeds=cfg['study']['transfer_seeds'])
            freeze(root/'SELECTION.json',result)
            print('TUNING COMPLETE; both controls frozen before official-validation evaluation.',flush=True)
        export_report(root,cfg,'STAGE COMPLETE: '+stage)
    rows=read_rows(root);require(len(rows)==planned_counts(cfg)['method_runs'],'Incomplete final matrix')
    for stage,_,seeds in stage_specs(cfg):
        methods=candidates(cfg) if stage=='tuning' else evaluated_methods(read_json(root/'SELECTION.json'))
        for seed in seeds:
            for m in methods:require(verify_method(root/stage/f'seed{seed}'/m['name']),'Final artifact verification failed')
    require(primary_analysis(rows,cfg)['status']=='COMPLETE','Primary family incomplete')
    atom_json(root/'VERIFICATION.json',dict(status='PASS',method_runs=len(rows),
              saved_logit_metrics_recomputed=True,phase_and_acquisition_hashes_checked=True,
              first_acquisition_identical_within_seed=True,old_results_modified=False,
              official_validation_used_for_selection=False,source_and_protocol_binding=binding))
    record=export_report(root,cfg,'COMPLETE')
    return record


def environment_info():
    versions={n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','scipy','Pillow')}
    return dict(python=sys.version,versions=versions,cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0),platform=platform.platform())


def gpu_contract(cfg,root):
    """Real batch-128 BF16 training step for both backbones, with no dataset outcomes."""
    from aligned import class_scatter,aligned_loss
    from support import optimizer_for,amp
    checks=[];device=torch.device('cuda:0');batch=cfg['batch_classes']*cfg['samples_per_class']
    for backbone in ('resnet50','resnet18'):
        seed_all(912433);model=Model(cfg['num_classes'],backbone).to(device).to(memory_format=torch.channels_last)
        opt=optimizer_for(model,cfg);x=torch.rand(batch,3,224,224,device=device).contiguous(memory_format=torch.channels_last)
        y=torch.arange(cfg['batch_classes'],device=device).repeat_interleave(cfg['samples_per_class'])
        with amp(device):logits,z=model.forward_features(x);ce=torch.nn.functional.cross_entropy(logits.float(),y)
        target={}
        for l in cfg['layers']:
            s=class_scatter(z[l].detach(),y);target[l]={'log_kappa_target':float(s['log_kappa'])-.1,
                  'between_floor':max(float(s['between'])*.5,1e-5)}
        reg,_=aligned_loss(z,y,target,{l:.03 for l in cfg['layers']},cfg['alignment'])
        (ce+reg).backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_grad_norm'],error_if_nonfinite=True)
        opt.step();torch.cuda.synchronize();require(torch.isfinite(ce+reg),'Nonfinite CUDA smoke loss')
        checks.append(dict(backbone=backbone,batch=batch,loss=float(ce+reg),passed=True))
        del model,opt,x,y,logits,z,ce,reg;release()
    atom_json(root/'CUDA_CHECK.json',dict(status='PASS',checks=checks,scientific_runs_counted=0))


def required_free_gib(root,cfg):
    rows=read_rows(root);c=planned_counts(cfg)
    r50=sum(r['backbone']=='resnet50' for r in rows);r18=sum(r['backbone']=='resnet18' for r in rows)
    # Conservative allowance for all phase checkpoints, Adam-free SGD momentum, arrays and logs.
    return max(12., 1.8*(c['tuning_runs']+c['confirmation_runs']-r50)+.85*(c['transfer_runs']-r18)+10.)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args();raw_root=Path(args.root);root=raw_root.resolve();cfg=read_json(HERE/'protocol.json')
    require(root==Path(cfg['result_root']).resolve(),'Unexpected campaign result root')
    require(Path('/mnt/caenl').is_mount(),'Existing /mnt/caenl volume is not mounted')
    require(not raw_root.is_symlink(),'Result root must not be a symlink')
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'RUN.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('A campaign process already holds the lock.');return 0
    started=time.time();current='preflight'
    def progress(message):
        nonlocal current
        current=message;atom_json(root/'STATUS.json',dict(status='RUNNING',time=utc(),pid=os.getpid(),message=message))
        print(message,flush=True)
    try:
        if (root/'FAILED.json').exists():
            os.replace(root/'FAILED.json',root/f'failure_{time.time_ns()}.json')
        progress('Preflight: source, runtime, and available GPU')
        files=verify_source()
        require(torch.cuda.is_available() and 'L40S' in torch.cuda.get_device_name(0),'Expected accessible L40S GPU')
        require(str(torch.__version__).startswith('2.6.') and torch.version.cuda=='12.4','Expected existing PyTorch 2.6 / CUDA 12.4 environment')
        require(importlib.metadata.version('torchvision').startswith('0.21.'),'Expected torchvision 0.21 environment')
        query=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        others={int(x.strip()) for x in query.stdout.splitlines() if x.strip().isdigit()}-{os.getpid()}
        require(not others,'Another GPU process is active; it has not been stopped')
        torch.set_num_threads(8);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        env=environment_info();binding=digest(dict(protocol=cfg,source_files=files,environment=env))
        freeze(root/'INPUT_BINDING.json',dict(binding=binding,protocol=cfg,source_files=files,environment=env))
        if (root/'COMPLETE.json').exists():
            info=read_json(root/'COMPLETE.json');require(info['binding']==binding,'Completion binding changed')
            require(sha256(root/'caenl-feedback-study-v1.md')==info['report_sha256'],'Completed report changed')
            print('ALREADY COMPLETE: '+str(root/'caenl-feedback-study-v1.md'));return 0
        available=shutil.disk_usage(root).free/1024**3;need=required_free_gib(root,cfg)
        require(available>=need,f'Need approximately {need:.1f} GiB free; available {available:.1f} GiB. No cleanup attempted.')
        from self_test import run
        buf=io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):checks=run()
        except BaseException:print(buf.getvalue());raise
        torch.set_num_threads(8)
        atom_json(root/'CPU_CHECK.json',dict(status='PASS',checks=checks,scientific_runs_counted=0))
        print(f'CPU integration checks passed: {len(checks)} checks.',flush=True)
        cat=Catalog(cfg);progress('Preflight: verifying every existing dataset image hash')
        count=cat.verify_images(progress)
        atom_json(root/'DATA_CHECK.json',dict(status='PASS',images=count,manifest=cfg['expected_image_manifest_sha256'],time=utc()))
        progress('Preflight: live CUDA training steps for ResNet-50 and ResNet-18')
        gpu_contract(cfg,root)
        if args.check_only:
            atom_json(root/'STATUS.json',dict(status='CHECK_COMPLETE',time=utc(),message='Preflight passed; no scientific training runs started.'))
            print('CHECK COMPLETE; start the campaign with --start.');return 0
        record=run_study(root,cfg,cat,binding,progress,torch.device('cuda:0'))
        atom_json(root/'COMPLETE.json',dict(status='COMPLETE',binding=binding,time=utc(),
                   completed_method_runs=record['completed_method_runs'],expected_counts=planned_counts(cfg),
                   report=str(root/'caenl-feedback-study-v1.md'),report_sha256=sha256(root/'caenl-feedback-study-v1.md'),
                   wall_seconds_this_invocation=time.time()-started,scientific_success_not_guaranteed=True))
        atom_json(root/'STATUS.json',dict(status='COMPLETE',time=utc(),message='CAENL FEEDBACK STUDY COMPLETE'))
        print('CAENL FEEDBACK STUDY COMPLETE\nShare: '+str(root/'caenl-feedback-study-v1.md'),flush=True)
        return 0
    except BaseException as exc:
        atom_json(root/'FAILED.json',dict(status='FAILED',time=utc(),operation=current,error=repr(exc),traceback=traceback.format_exc()))
        atom_json(root/'STATUS.json',dict(status='FAILED',time=utc(),message=str(exc)))
        try:export_report(root,cfg,'FAILED: '+str(exc))
        except Exception:pass
        traceback.print_exc();return 1
    finally:lock.close()


if __name__=='__main__':raise SystemExit(main())
