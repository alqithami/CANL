"""Independent from-scratch initialization and matched, resumable active-learning phases."""
from __future__ import annotations
import copy,gc,hashlib,math,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from aligned import aligned_loss,class_scatter
from control import Control
from evaluation import assess
from support import (Model,seed_for,seed_all,read_json,atom_json,atom_torch,write_csv,sha256,digest,
    tensor_digest,loader,BalancedBatches,to_device,optimizer_for,train_mode,amp,sync,rng_state,restore_rng)


def phase_steps(n,phase,cfg):
    epochs=cfg['initial_epochs'] if phase==0 else cfg['round_epochs']
    per_epoch=math.ceil(n/(cfg['batch_classes']*cfg['samples_per_class']))
    return per_epoch*epochs,per_epoch

def cosine_lr(step,total,warmup,base):
    if step<warmup:return base*(step+1)/max(1,warmup)
    p=min(1.,(step-warmup)/max(1,total-warmup))
    return base*.5*(1+math.cos(math.pi*p))

def phase_complete(out,binding):
    out=Path(out);done=out/'DONE.json'
    if not done.exists():return False
    d=read_json(done)
    if d['binding']!=binding:raise ValueError('Completed phase protocol binding changed')
    for p,h in d['payload'].items():
        if not (out/p).is_file() or sha256(out/p)!=h:raise ValueError('Completed phase checksum mismatch: '+str(out/p))
    return True

@torch.no_grad()
def prepare_feedback(cat,control_ids,cfg):
    # The fixed 633-image holdout is decoded once per phase, not in fresh worker
    # processes every 100 steps. Native bytes and deterministic crop are unchanged.
    from support import evaluation_batches
    xs=[];ys=[]
    for x,y,_ in loader(cat,'train',evaluation_batches(control_ids,cfg['eval_batch']),0,0,False):
        xs.append(x);ys.append(y)
    return torch.cat(xs),torch.cat(ys)

@torch.no_grad()
def feedback_ce(model,cached,cfg,device):
    model.eval();images,labels=cached;total=0.
    for i in range(0,len(labels),cfg['eval_batch']):
        x=images[i:i+cfg['eval_batch']].to(device).float().div_(255).contiguous(memory_format=torch.channels_last)
        y=labels[i:i+cfg['eval_batch']].to(device);logits=model(x).float()
        total+=float(F.cross_entropy(logits,y,reduction='sum'))
    result=total/len(labels)
    if not math.isfinite(result):raise FloatingPointError('Nonfinite control CE')
    return result

def run_phase(cfg,cat,seed,method,phase,labeled,control_ids,source_ckpt,targets,out,binding,total_windows,progress,device,model_factory=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);initial=phase==0;kind='none' if initial else method['regularizer']
    source_hash=sha256(source_ckpt) if source_ckpt else None
    bnd=digest({'campaign':binding,'seed':seed,'method':method['name'],'phase':phase,'source':source_hash,'labeled':np.asarray(labeled).tolist(),'control':np.asarray(control_ids).tolist(),'targets':targets})
    if phase_complete(out,bnd):return read_json(out/'summary.json'),out/'state.pt'
    steps,per_epoch=phase_steps(len(labeled),phase,cfg);ckpt=out/'state.pt';rows=[];step0=0;elapsed_before=0.;compute_before=0.;feedback_before=0.;peak_before=0
    bs=cfg['batch_classes']*cfg['samples_per_class'];stream_seed=seed_for('phase_stream',seed,phase)
    seed_all(seed_for('initial_model',seed));model=(model_factory or Model)(cfg['num_classes']).to(device).float()
    if str(device).startswith('cuda'):model=model.to(memory_format=torch.channels_last)
    opt=optimizer_for(model,cfg);ctrl=Control(kind,targets,seed,cfg,device,total_windows) if targets else None
    if source_ckpt:
        st=torch.load(source_ckpt,map_location='cpu',weights_only=False);model.load_state_dict(st['model'],strict=True);opt.load_state_dict(st['optimizer'])
        if ctrl and st.get('controller') is not None:ctrl.load_state_dict(st['controller'])
        del st
    initial_weights_hash=tensor_digest(model.state_dict());chain=hashlib.sha256(b'BATCH_CHAIN_V1').hexdigest()
    cached_feedback=prepare_feedback(cat,control_ids,cfg) if ctrl else None
    if ckpt.exists():
        st=torch.load(ckpt,map_location='cpu',weights_only=False)
        if st['binding']!=bnd or st['total_steps']!=steps:raise ValueError('Checkpoint input/protocol mismatch')
        model.load_state_dict(st['model'],strict=True);opt.load_state_dict(st['optimizer'])
        if ctrl:ctrl.load_state_dict(st['controller'])
        step0=st['step'];rows=st['rows'];elapsed_before=st['elapsed_s'];compute_before=st['compute_s'];feedback_before=st['feedback_s'];peak_before=st['peak_allocated_bytes'];chain=st['batch_chain'];initial_weights_hash=st['initial_weights_hash'];restore_rng(st['rng']);del st
        print(f'[resume] {method["name"]}/seed{seed} round{phase}: {step0}/{steps}',flush=True)
    elif ctrl:
        t=time.perf_counter();control_value=feedback_ce(model,cached_feedback,cfg,device);feedback_before+=time.perf_counter()-t
        ctrl.begin_phase(control_value,0)
    base_lr=cfg['lr_initial'] if initial else cfg['lr_round'];warmup=per_epoch*(cfg['warmup_initial_epochs'] if initial else cfg['warmup_round_epochs'])
    sampler=BalancedBatches(labeled,cat.labels,stream_seed,steps,cfg['batch_classes'],cfg['samples_per_class'],step0)
    dl=loader(cat,'train',sampler,stream_seed,cfg['workers'],True)
    train_mode(model,initial);sync(device);begin=time.perf_counter();compute_s=compute_before;feedback_s=feedback_before
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    for step,batch in enumerate(dl,step0+1):
        x,y,idx=to_device(batch,device);lr=cosine_lr(step-1,steps,warmup,base_lr)
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True);sync(device);t=time.perf_counter()
        with amp(device):logits,features=model.forward_features(x);ce=F.cross_entropy(logits.float(),y)
        if kind.startswith('aligned_'):
            reg,recs=aligned_loss(features,y,targets,ctrl.weights,cfg['alignment'])
        else:
            reg=ce.new_zeros(())
            with torch.no_grad():
                recs={l:{k:float(v) for k,v in class_scatter(z.detach(),y,eps=cfg['alignment']['scatter_epsilon'],norm_eps=cfg['alignment']['feature_norm_epsilon']).items()} for l,z in features.items()}
        loss=ce+reg
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss {method["name"]}/seed{seed}/round{phase}/step{step}')
        loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_grad_norm'],error_if_nonfinite=True));opt.step();sync(device);compute_s+=time.perf_counter()-t
        if ctrl:ctrl.observe(recs)
        chain=hashlib.sha256(bytes.fromhex(chain)+np.asarray(idx,dtype=np.int64).tobytes()).hexdigest()
        row={'step':step,'epoch':(step-1)//per_epoch+1,'train_ce':float(ce.detach()),'regularizer':float(reg.detach()),'lr':lr,'preclip_grad_norm':gn,'train_accuracy':float((logits.detach().argmax(1)==y).float().mean())}
        for l in cfg['layers']:
            row.update({f'{l}_log_kappa':recs[l]['log_kappa'],f'{l}_between':recs[l]['between'],f'{l}_lambda':ctrl.weights[l] if ctrl else 0.})
        rows.append(row)
        if ctrl and (step%cfg['control_interval']==0 or step==steps):
            t=time.perf_counter();control_value=feedback_ce(model,cached_feedback,cfg,device);feedback_s+=time.perf_counter()-t
            ctrl.window(control_value,phase,step,step,terminal=step==steps);train_mode(model,initial)
        if step%50==0 or step==steps:
            progress(f'{method["name"]} seed{seed} round{phase}: step {step}/{steps}, epoch {(step-1)//per_epoch+1}, CE={float(ce.detach()):.4f}')
        if step%per_epoch==0:
            erows=rows[-per_epoch:];avg=float(np.mean([r['train_ce'] for r in erows]))
            print(f'[epoch] {method["name"]}/seed{seed} round{phase} {step//per_epoch}/{steps//per_epoch} CE={avg:.4f} step={step}/{steps}',flush=True)
        save=step==steps or step%cfg['checkpoint_every_steps']==0 or (cfg['checkpoint_each_epoch'] and step%per_epoch==0)
        if save:
            sync(device);peak=max(peak_before,torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0)
            st={'binding':bnd,'method':method['name'],'seed':seed,'phase':phase,'step':step,'total_steps':steps,'model':model.state_dict(),'optimizer':opt.state_dict(),'controller':ctrl.state_dict() if ctrl else None,'rows':rows,'rng':rng_state(),'elapsed_s':elapsed_before+time.perf_counter()-begin,'compute_s':compute_s,'feedback_s':feedback_s,'peak_allocated_bytes':peak,'batch_chain':chain,'initial_weights_hash':initial_weights_hash,'labeled':np.asarray(labeled),'control_indices':np.asarray(control_ids),'targets':targets}
            atom_torch(ckpt,st);write_csv(out/'training.csv',rows)
            if ctrl:atom_json(out/'controller_trajectory.json',ctrl.history)
            del st
    # Full-precision validation is reporting only; never used by any controller.
    progress(f'{method["name"]} seed{seed} round{phase}: full clean validation')
    met=assess(model,cat,'val',np.arange(len(cat.val_labels)),cfg,device,path=out/'validation.npz',geometry=True)
    feedback=assess(model,cat,'train',control_ids,cfg,device,path=out/'feedback.npz',geometry=True)
    if len(rows)!=steps or [r['step'] for r in rows]!=list(range(1,steps+1)):raise ValueError('Training step history incomplete')
    controller_state=ctrl.state_dict() if ctrl else None
    if kind=='aligned_full' and ctrl.ctrl._pending is not None:raise RuntimeError('Terminal action credit lost')
    st=torch.load(ckpt,map_location='cpu',weights_only=False)
    # Summary timing reflects the saved last training step, not repeated evaluation after resume.
    summary={'binding':bnd,'method':method['name'],'seed':seed,'phase':phase,'steps':steps,'examples_processed':steps*bs,'nominal_epochs':steps//per_epoch,'epoch_definition':'ceil(labeled_count/128) class-balanced sampled minibatches; not a permutation pass',
        'gradient_training_count':len(labeled),'annotation_count':len(labeled)+len(control_ids),'controller_holdout_count':len(control_ids),'validation':met,'feedback':feedback,
        'train_compute_seconds':st['compute_s'],'feedback_seconds':st['feedback_s'],'phase_training_elapsed_seconds':st['elapsed_s'],'peak_allocated_bytes':st['peak_allocated_bytes'],'initial_model_weights_sha256':initial_weights_hash,'final_model_weights_sha256':tensor_digest(model.state_dict()),'batch_chain_sha256':st['batch_chain'],'checkpoint_sha256':sha256(ckpt),
        'controller_windows_cumulative':ctrl.windows if ctrl else 0,'learned_updates_cumulative':sum('policy_loss' in r.get('update',{}) for r in ctrl.history) if ctrl else 0,'unfavorable_metrics_are_valid':True}
    del st
    atom_json(out/'summary.json',summary)
    files=['state.pt','training.csv','validation.npz','feedback.npz','summary.json']+(['controller_trajectory.json'] if ctrl else [])
    atom_json(out/'DONE.json',{'status':'COMPLETE','binding':bnd,'payload':{p:sha256(out/p) for p in files}})
    del model,opt,ctrl,dl;gc.collect()
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    print(f'[phase complete] {method["name"]}/seed{seed} round{phase}: clean={met["accuracy"]*100:.2f}%',flush=True)
    return summary,ckpt
