"""Full-precision reporting; unchanged original standard AutoAttack endpoint."""
from __future__ import annotations
import math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from aligned import class_scatter
from support import (loader,evaluation_batches,to_device,atom_npz,atom_json,read_json,sha256,seed_for,sync,digest)


def metrics_from_logits(logits,labels):
    x=torch.as_tensor(logits,dtype=torch.float64);y=torch.as_tensor(labels,dtype=torch.long)
    if x.ndim!=2 or x.shape[0]!=len(y) or not len(y) or not torch.isfinite(x).all():raise ValueError('Invalid evaluation payload')
    p=x.softmax(1);conf,pred=p.max(1);correct=pred.eq(y);ece=0.
    bins=(conf*15).long().clamp(max=14)
    for i in range(15):
        sel=bins==i
        if sel.any():ece+=float(sel.double().mean()*abs(correct[sel].double().mean()-conf[sel].mean()))
    return {'n':len(y),'ce':float(F.cross_entropy(x,y)), 'accuracy':float(correct.double().mean()),
            'top5_accuracy':float(x.topk(min(5,x.shape[1]),1).indices.eq(y[:,None]).any(1).double().mean()),'ece':ece}

@torch.no_grad()
def infer(model,cat,split,ids,cfg,device,features=False):
    model.eval();logits=[];ys=[];ixes=[];zs={l:[] for l in cfg['layers']} if features else {}
    for b in loader(cat,split,evaluation_batches(ids,cfg['eval_batch']),0,cfg['eval_workers'],False):
        x,y,ix=to_device(b,device);zout,z=model.forward_features(x)
        if not torch.isfinite(zout).all():raise FloatingPointError('Nonfinite logits')
        logits.append(zout.float().cpu());ys.append(y.cpu());ixes.append(ix)
        if features:
            for l in zs:zs[l].append(z[l].float().cpu())
    if not logits:raise ValueError('Empty inference set')
    result={'indices':torch.cat(ixes).numpy(),'labels':torch.cat(ys).numpy(),'logits':torch.cat(logits).numpy()}
    if not np.array_equal(result['indices'],np.asarray(ids)):raise ValueError('Evaluation index order mismatch')
    result['features']={l:torch.cat(v) for l,v in zs.items()}
    return result

def assess(model,cat,split,ids,cfg,device,path=None,geometry=True):
    r=infer(model,cat,split,ids,cfg,device,features=geometry);m=metrics_from_logits(r['logits'],r['labels'])
    if geometry:
        for l,z in r['features'].items():
            s=class_scatter(z.double(),torch.from_numpy(r['labels']),eps=cfg['alignment']['scatter_epsilon'],norm_eps=cfg['alignment']['feature_norm_epsilon'],require_balanced=False)
            for k,v in s.items():m[l+'_'+k]=float(v)
        m['mean_log_kappa']=float(np.mean([m[l+'_log_kappa'] for l in cfg['layers']]))
    if path is not None:
        payload={k:r[k] for k in ('indices','labels','logits')};payload['predictions']=r['logits'].argmax(1)
        # Store compact class-scatter sufficient statistics for arithmetic checks.
        if geometry:
            for l,z in r['features'].items():
                h=F.normalize(z.double(),dim=1,eps=cfg['alignment']['feature_norm_epsilon']);y=torch.from_numpy(r['labels'])
                payload[l+'_class_counts']=np.bincount(r['labels'],minlength=cfg['num_classes'])
                payload[l+'_class_sums']=np.stack([h[y==c].sum(0).numpy() for c in range(cfg['num_classes'])])
                payload[l+'_class_sumsq']=np.stack([(h[y==c]**2).sum(0).numpy() for c in range(cfg['num_classes'])])
        atom_npz(path,**payload)
    return m


def autoattack(model,cat,cfg,device,out,binding,checkpoint_hash,progress):
    """Resume between independent standard-suite chunks; no weakening/no custom iteration count."""
    from autoattack import AutoAttack
    out=Path(out);out.mkdir(parents=True,exist_ok=True);aa=cfg['autoattack'];n=len(cat.rows['val'])
    if not aa['all_validation_images']:raise ValueError('Full official validation attack required')
    model.eval();all_rows=[];blocks=[];eps=float(aa['epsilon']);bs=int(aa['batch_size']);chunk=int(aa['chunk_size'])
    for start in range(0,n,chunk):
        end=min(n,start+chunk);ids=np.arange(start,end,dtype=np.int64)
        path=out/f'chunk_{start:05d}.npz';done=out/f'chunk_{start:05d}.json'
        cb=digest({'binding':binding,'checkpoint':checkpoint_hash,'start':start,'end':end,'aa':aa})
        if done.exists():
            rec=read_json(done)
            if rec['binding']!=cb or sha256(path)!=rec['sha256']:raise ValueError('Attack chunk binding/digest mismatch')
            arr=dict(np.load(path,allow_pickle=False));blocks.append(arr);all_rows.append(rec);continue
        progress(f'AutoAttack 8/255: images {start+1}-{end}/{n}')
        raw=[];yraw=[]
        for b in loader(cat,'val',evaluation_batches(ids,bs),0,cfg['eval_workers'],False):
            xb,yb,_=to_device(b,device);raw.append(xb);yraw.append(yb)
        x=torch.cat(raw);y=torch.cat(yraw)
        if x.min()<0 or x.max()>1:raise ValueError('Attack input not in pixel space')
        with torch.no_grad():
            clean=torch.cat([model(x[j:j+bs]).float() for j in range(0,len(x),bs)]);cp=clean.argmax(1)
        adv=AutoAttack(model,norm='Linf',eps=eps,version='standard',device=str(device),seed=seed_for('aa',start),verbose=False,log_path=str(out/f'attack_{start:05d}.log'))
        if list(adv.attacks_to_run)!=['apgd-ce','apgd-t','fab-t','square']:raise RuntimeError('Unexpected standard attack list')
        sync(device);t=time.perf_counter()
        xa=adv.run_standard_evaluation(x,y,bs=bs)
        with torch.no_grad():
            al=torch.cat([model(xa[j:j+bs]).float() for j in range(0,len(xa),bs)]);ap=al.argmax(1)
        norms=(xa-x).abs().flatten(1).amax(1)
        if not torch.isfinite(xa).all() or xa.min() < -1e-6 or xa.max()>1+1e-6 or norms.max()>eps+2e-6:raise ValueError('Attack violates threat model')
        good=ap.eq(y)&cp.eq(y)
        arr=dict(indices=ids,labels=y.cpu().numpy(),clean_logits=clean.cpu().numpy(),adversarial_logits=al.cpu().numpy(),clean_predictions=cp.cpu().numpy(),adversarial_predictions=ap.cpu().numpy(),robust_mask=good.cpu().numpy(),linf=norms.cpu().numpy())
        atom_npz(path,**arr);sync(device)
        rec={'binding':cb,'n':len(y),'robust_correct':int(good.sum()),'clean_correct':int(cp.eq(y).sum()),'elapsed_s':time.perf_counter()-t,'sha256':sha256(path)}
        atom_json(done,rec);blocks.append(arr);all_rows.append(rec)
        del x,y,xa,raw,yraw,clean,al,adv
    arrays={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    if not np.array_equal(arrays['indices'],np.arange(n)) or not np.array_equal(arrays['labels'],cat.val_labels):raise RuntimeError('Attack full-set mapping error')
    atom_npz(out/'predictions.npz',**arrays)
    result={'n':n,'clean_accuracy':float(np.mean(arrays['clean_predictions']==arrays['labels'])),'robust_accuracy':float(arrays['robust_mask'].mean()),'robust_correct':int(arrays['robust_mask'].sum()),'epsilon':eps,'norm':'Linf','suite':'standard','checkpoint_sha256':checkpoint_hash,'elapsed_s':sum(r['elapsed_s'] for r in all_rows),'predictions_sha256':sha256(out/'predictions.npz')}
    atom_json(out/'summary.json',result);return result
