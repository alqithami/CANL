"""Label-blind pool selection; exact BADGE kernel; explicitly named independent Noise Stability implementation."""
from __future__ import annotations
import math,time
from pathlib import Path
import numpy as np
import torch
from evaluation import infer
from support import atom_json,atom_npz,read_json,sha256,digest,seed_for,tensor_digest


def top_indices(scores,n):
    scores=np.asarray(scores,dtype=np.float64)
    if not np.isfinite(scores).all():raise FloatingPointError('Invalid acquisition scores')
    return np.lexsort((np.arange(len(scores)),-scores))[:n].astype(np.int64)

def entropy(logits):
    p=torch.as_tensor(logits).float().softmax(1)
    return (-(p*p.clamp_min(1e-12).log()).sum(1)).numpy()

def badge_positions(logits,features,b,seed,device):
    u=torch.as_tensor(logits,device=device).float().softmax(1);h=torch.as_tensor(features,device=device).float()
    pred=u.argmax(1);u[torch.arange(len(u),device=device),pred]-=1
    norm=(u*u).sum(1)*(h*h).sum(1)
    first=int(norm.argmax());chosen=[first];used=torch.zeros(len(u),dtype=torch.bool,device=device);used[first]=True
    dist=(norm+norm[first]-2*(u@u[first])*(h@h[first])).clamp_min(0);dist[first]=0
    gen=torch.Generator(device=device).manual_seed(seed)
    for _ in range(1,b):
        mass=dist.masked_fill(used,0.).sum()
        if not torch.isfinite(mass):raise FloatingPointError('BADGE distance mass is nonfinite')
        if float(mass)<=0:
            remain=torch.nonzero(~used).flatten();chosen.extend(remain[:b-len(chosen)].cpu().tolist());break
        j=int(torch.multinomial(dist.masked_fill(used,0.)/mass,1,generator=gen));chosen.append(j);used[j]=True
        dn=(norm+norm[j]-2*(u@u[j])*(h@h[j])).clamp_min(0);dist=torch.minimum(dist,dn);dist[used]=0
    return np.asarray(chosen,dtype=np.int64)

def kcenter_positions(features,b,device):
    x=torch.as_tensor(features,device=device).float();sq=(x*x).sum(1);j=int(sq.argmax());ds=torch.full_like(sq,float('inf'));chosen=[]
    for _ in range(b):
        chosen.append(j);dn=(sq+sq[j]-2*x@x[j]).clamp_min(0);ds=torch.minimum(ds,dn);ds[torch.tensor(chosen,device=device)]=-1;j=int(ds.argmax())
    return np.asarray(chosen,dtype=np.int64)

def class_statistics(features,labels,classes,eps):
    """Labels here are restricted to currently acquired gradient-training examples."""
    z=torch.as_tensor(features).double();y=torch.as_tensor(labels);means=[];var=[]
    for c in range(classes):
        v=z[y==c]
        if len(v)<2:raise ValueError(f'Insufficient labeled support for class {c}')
        means.append(v.mean(0));var.append(v.var(0,unbiased=False)+eps)
    return torch.stack(means).float(),torch.stack(var).float()

def mahalanobis(logits,features,stats,layer_weights):
    pred=np.asarray(logits).argmax(1);total=np.zeros(len(pred),dtype=np.float64)
    for l,(means,var) in stats.items():
        z=torch.as_tensor(features[l]).float();mu=means[pred];vv=var[pred]
        dist=((z-mu).square()/vv).sum(1).numpy()
        total+=float(layer_weights[l])*dist
    return total

@torch.no_grad()
def noise_features(model,cat,pool,base_logits,seed,phase,cfg,device,progress):
    """Five globally normalized Gaussian parameter perturbations; bitwise restoration."""
    params=[p for p in model.parameters() if p.requires_grad];original=[p.detach().clone() for p in params]
    base=torch.from_numpy(base_logits).float().softmax(1);norm=math.sqrt(sum(float(p.double().square().sum()) for p in params))
    scale=float(cfg['acquisition']['noise_scale']);k=int(cfg['acquisition']['noise_samples']);zs=[]
    gen=torch.Generator(device=device).manual_seed(seed_for('noise',seed,phase));before=tensor_digest(model.state_dict())
    try:
        for draw in range(k):
            progress(f'Noise Stability: forward pass {draw+1}/{k}, {len(pool):,} candidate images')
            noise=[torch.randn(p.shape,device=p.device,dtype=p.dtype,generator=gen) for p in params]
            nnorm=math.sqrt(sum(float(v.double().square().sum()) for v in noise));alpha=scale*norm/max(nnorm,1e-12)
            for p,o,v in zip(params,original,noise):p.copy_(o+alpha*v)
            del noise
            rec=infer(model,cat,'train',pool,cfg,device,features=False)
            noisy=torch.from_numpy(rec['logits']).float().softmax(1)
            zs.append(((noisy-base)/max(scale*norm,1e-12)).numpy());del noisy,rec
            for p,o in zip(params,original):p.copy_(o)
    finally:
        for p,o in zip(params,original):p.copy_(o)
    if tensor_digest(model.state_dict())!=before:raise RuntimeError('Noise Stability altered model parameters/buffers')
    return np.concatenate(zs,axis=1)


def select_round(model,cat,labeled,holdout,method,seed,phase,cfg,device,out,binding,checkpoint_hash,progress):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);selected_file=out/'selection.npz';done=out/'COMPLETE.json'
    desired=int(cfg['acquisition_per_round']);labeled=np.sort(labeled);pool=np.setdiff1d(np.arange(len(cat.labels)),np.concatenate([labeled,holdout]))
    bound=digest({'binding':binding,'checkpoint':checkpoint_hash,'labeled':labeled.tolist(),'holdout':np.asarray(holdout).tolist(),'seed':seed,'phase':phase,'method':method})
    if done.exists():
        r=read_json(done)
        if r['binding']!=bound or sha256(selected_file)!=r['selection_sha256']:raise ValueError('Acquisition resume binding mismatch')
        for record in r['passes']:
            pass_path=out/f"pass_{int(record['pass']):03d}.npz"
            if not pass_path.is_file() or sha256(pass_path)!=record['sha256']:
                raise ValueError('Saved acquisition pass checksum mismatch: '+str(pass_path))
        arr=dict(np.load(selected_file,allow_pickle=False));chosen=arr['selected']
        if len(chosen)!=desired or len(np.unique(chosen))!=desired or not np.isin(chosen,pool).all():raise ValueError('Bad saved acquisition')
        return chosen,r
    if len(pool)<desired:raise ValueError('Insufficient unlabeled pool')
    t=time.perf_counter();acq=method['acquisition'];forward_seconds=0.;stats={};logits=None;features={};noise=None
    if acq!='random':
        progress(f"{method['name']} seed{seed} round{phase}: extracting {len(pool):,} pool predictions")
        ft=time.perf_counter();rec=infer(model,cat,'train',pool,cfg,device,features=acq in ('mahalanobis','badge'))
        # Inference also returns ground-truth labels for auditing elsewhere. Never expose them to selection.
        logits=rec['logits'];features=rec['features'];del rec;forward_seconds+=time.perf_counter()-ft
        if acq=='mahalanobis':
            progress(f'Class statistics from {len(labeled):,} acquired training examples only')
            ft=time.perf_counter();rec=infer(model,cat,'train',labeled,cfg,device,features=True)
            for l in cfg['layers']:stats[l]=class_statistics(rec['features'][l],rec['labels'],cfg['num_classes'],cfg['acquisition']['mahalanobis_shrinkage'])
            del rec;forward_seconds+=time.perf_counter()-ft
        if acq=='noise_stability':noise=noise_features(model,cat,pool,logits,seed,phase,cfg,device,progress)
    rng=np.random.default_rng(seed_for('candidate_passes',seed,phase));active=np.arange(len(pool));chosen=[];passes=[];pass_id=0
    while len(chosen)<desired:
        n=min(len(active),cfg['acquisition']['candidate_size']);pos=rng.choice(active,n,replace=False)
        quota=min(desired-len(chosen),max(1,int(math.ceil(n*cfg['acquisition']['per_pass_fraction']))));pass_id+=1
        scores=None
        if acq=='random':indices=rng.permutation(n)[:quota]
        elif acq=='entropy':scores=entropy(logits[pos]);indices=top_indices(scores,quota)
        elif acq=='badge':indices=badge_positions(logits[pos],features['layer4'][pos],quota,seed_for('badge',seed,phase,pass_id),device)
        elif acq=='noise_stability':indices=kcenter_positions(noise[pos],quota,device)
        elif acq=='mahalanobis':
            scores=mahalanobis(logits[pos],{l:features[l][pos] for l in cfg['layers']},stats,cfg['acquisition']['layer_weights']);indices=top_indices(scores,quota)
        else:raise ValueError('Unknown acquisition: '+acq)
        take=pos[indices]
        if len(np.unique(take))!=quota:raise RuntimeError('Acquisition returned duplicate indices')
        chosen.extend(pool[take].tolist());active=active[~np.isin(active,take)]
        payload={'candidate_indices':pool[pos],'selected_indices':pool[take]}
        if scores is not None:payload['scores']=scores
        if logits is not None:payload['pseudo_labels']=logits[pos].argmax(1)
        path=out/f'pass_{pass_id:03d}.npz';atom_npz(path,**payload)
        passes.append({'pass':pass_id,'candidates':n,'selected':quota,'sha256':sha256(path)})
        progress(f"{method['name']} seed{seed} round{phase}: selected {len(chosen):,}/{desired:,}")
    chosen=np.asarray(chosen,dtype=np.int64)
    if len(np.unique(chosen))!=desired or np.intersect1d(chosen,np.concatenate([labeled,holdout])).size:raise ValueError('Label-budget/integrity failure')
    atom_npz(selected_file,selected=chosen,previous_labeled=labeled,control_holdout=np.asarray(holdout))
    info={'binding':bound,'method':method,'seed':seed,'round':phase,'requested':desired,'selected':len(chosen),'candidate_presentations':sum(r['candidates'] for r in passes),'unique_pool_size':len(pool),'forward_seconds':forward_seconds,'elapsed_seconds':time.perf_counter()-t,'passes':passes,'selected_class_counts':np.bincount(cat.labels[chosen],minlength=cfg['num_classes']).tolist(),'selection_sha256':sha256(selected_file),'original_checkpoint_sha256':checkpoint_hash,'pool_true_labels_used_for_selection':False,'covariance_mode':cfg['acquisition']['covariance']}
    atom_json(done,info);return chosen,info
