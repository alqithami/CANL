"""CPU contract/integration tests. No user data or old checkpoints are read or modified."""
from __future__ import annotations
import copy, hashlib, json, math, tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from PIL import Image
from aligned import class_scatter,aligned_loss,positive_huber,LeakyMACCLite
from support import (BalancedBatches,seed_for,stratified_sample,atom_npz,atom_json,read_json,tensor_digest,Images,loader)
from control import Control,calibrate
from acquisition import (badge_positions,kcenter_positions,mahalanobis,class_statistics,select_round,noise_features)
from evaluation import metrics_from_logits,assess
from reporting import paired_stats,holm
import training

class Tiny(nn.Module):
    def __init__(self,classes=4):
        super().__init__();self.l1=nn.Linear(3,32);self.bn=nn.BatchNorm1d(32);self.l2=nn.Linear(32,24);self.fc=nn.Linear(24,classes)
    def forward_features(self,x):
        z3=torch.tanh(self.bn(self.l1(x.mean((2,3)))));z4=torch.tanh(self.l2(z3));return self.fc(z4),{'layer3':z3,'layer4':z4}
    def forward(self,x):return self.forward_features(x)[0]


def fixture(root,cfg):
    rows={'train':[],'val':[]}
    for split,count in [('train',36),('val',8)]:
        for c in range(4):
            for j in range(count):
                p=f'{split}/class{c}/{j}.png';(root/p).parent.mkdir(parents=True,exist_ok=True)
                a=np.random.default_rng(seed_for(split,c,j)).integers(20+c*45,50+c*45,size=(18,22,3),dtype=np.uint8)
                a[:,:,c%3]=min(240,40+c*50);Image.fromarray(a).save(root/p);rows[split].append({'path':p,'class_id':c})
        rows[split].sort(key=lambda r:r['path'])
    return SimpleNamespace(root=root,rows=rows,labels=np.array([r['class_id'] for r in rows['train']]),val_labels=np.array([r['class_id'] for r in rows['val']]))


def run():
    torch.set_num_threads(2);torch.manual_seed(771);checks=[]
    def ck(x,label):
        if not bool(x):raise AssertionError(label)
        checks.append(label)
    cfg=read_json(Path(__file__).resolve().parent/'protocol.json')
    # Source-to-protocol fidelity for the frozen regularizer/controller settings.
    ck(cfg['alignment']=={'target_kappa_factor':.9,'between_floor_factor':.5,'scatter_epsilon':1e-6,'feature_norm_epsilon':1e-6,'huber_delta':.25,'floor_weight':.01},'audited aligned objective frozen')
    ck(cfg['lambda_fixed']==.03 and cfg['lambda_max']==.1 and cfg['lite_smoothing']==.2,'audited controller coefficients frozen')
    ck(len(cfg['seeds'])==6 and not set(cfg['seeds'])&{601,602,701,702},'independent seed identifiers')
    ck(len(cfg['methods'])==9 and cfg['acquisition']['full_macc_acquisition_head'] is False,'nine method families; no unvalidated acquisition policy')
    ck(cfg['autoattack']['epsilon']==8/255 and cfg['autoattack']['all_validation_images'],'original attack endpoint preserved')
    ck(cfg['initial_labels']+5*cfg['acquisition_per_round']==76014,'annotation budget exact')
    z=torch.randn(24,16,dtype=torch.float64,requires_grad=True);y=torch.arange(4).repeat_interleave(6)
    s=class_scatter(z,y);ck(torch.allclose(s['within']+s['between'],s['total'],atol=1e-12),'scatter decomposition')
    orth,_=torch.linalg.qr(torch.randn(16,16,dtype=torch.float64));s2=class_scatter(z@orth,y)
    ck(torch.allclose(s['log_kappa'],s2['log_kappa'],atol=1e-10),'orthogonal invariance')
    target={'a':{'log_kappa_target':float(s['log_kappa'].detach())-.2,'between_floor':float(s['between'].detach())*.5}}
    loss,_=aligned_loss({'a':z},y,target,{'a':.03},cfg['alignment']);grad=torch.autograd.grad(loss,z)[0]
    ck(torch.isfinite(grad).all() and grad.norm()>0,'finite nonzero aligned gradient')
    lower,_=aligned_loss({'a':z-.01*grad},y,target,{'a':.03},cfg['alignment']);ck(lower<loss,'direct feature-space descent')
    lite=LeakyMACCLite(['a']);
    for t in range(20):lite.step({'a':.2},t)
    ck(0<=lite.lambdas['a']<=.1,'bounded leaky response')
    bad=False
    try:class_scatter(z[:-1],y[:-1])
    except ValueError:bad=True
    ck(bad,'unbalanced objective batch rejected')
    scores=metrics_from_logits(np.array([[3.,1.,0.],[0.,2.,1.],[1.,2.,3.]]),[0,1,1]);ck(abs(scores['accuracy']-2/3)<1e-12,'pooled exact accuracy')
    ck(holm([.01,.04,.5])==[.03,.08,.5],'Holm correction')
    p=paired_stats([1.]*6,[1.]*6);ck(p['exact_signflip_p']==1 and p['paired_t_p']==1,'all-zero robustness differences valid')
    p=paired_stats([1.]*6,[0.]*6);ck(p['exact_signflip_p']==.03125,'six-seed exact sign-flip resolution')
    # Exact factorized BADGE distances equal explicit Kronecker embedding distances.
    l=torch.randn(12,4);h=torch.randn(12,7);u=l.softmax(1);u[torch.arange(12),u.argmax(1)]-=1
    G=torch.einsum('nc,nd->ncd',u,h).flatten(1)
    n=(u*u).sum(1)*(h*h).sum(1);df=n+n[3]-2*(u@u[3])*(h@h[3]);de=(G-G[3]).square().sum(1)
    ck(torch.allclose(df,de,atol=1e-5),'BADGE factorization exact')
    chosen=badge_positions(l,h,7,123,torch.device('cpu'));ck(len(set(chosen))==7,'BADGE selects unique candidates')
    ck(len(set(kcenter_positions(h,7,torch.device('cpu'))))==7,'noise-diversity k-center unique')
    mu,var=class_statistics(z.detach().float(),y,4,1e-5)
    log=np.zeros((24,4));log[np.arange(24),y.numpy()]=1
    ms=mahalanobis(log,{'a':z.detach().float()},{'a':(mu,var)},{'a':1.})
    ck(np.allclose(ms,((z.detach().float()-mu[y]).square()/var[y]).sum(1).numpy()),'Mahalanobis label-blind distance formula')
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);small=copy.deepcopy(cfg);small.update(num_classes=4,expected_train_count=144,expected_val_count=32,initial_labels=48,holdout_count=16,acquisition_per_round=16,rounds=2,initial_epochs=2,round_epochs=2,batch_classes=4,samples_per_class=2,workers=0,eval_workers=0,eval_batch=16,control_interval=2,calibration_batches=2,checkpoint_every_steps=2,minimum_policy_updates_per_completed_run=2)
        small['acquisition'].update(candidate_size=32,noise_samples=2);small['full_macc'].update(hidden_dims=[8,8],epsilon_start=0.,epsilon_end=0.)
        cat=fixture(root/'images',small);ann=stratified_sample(np.arange(144),cat.labels,48,4);hold=stratified_sample(ann,cat.labels,16,5);ids=np.setdiff1d(ann,hold)
        ck(len(ids)==32 and np.intersect1d(ids,hold).size==0,'budgeted holdout disjoint')
        sampler=BalancedBatches(ids,cat.labels,16,8,4,2);a=list(sampler);b=list(BalancedBatches(ids,cat.labels,16,8,4,2,start_step=4))
        ck(a[4:]==b,'step-addressable resume sampling')
        for batch in a:
            inds=[i for _,i in batch];ck(len(set(inds))==8 and np.all(np.bincount(cat.labels[inds])==2),'balanced per-class batch')
        ds=Images(cat,'train',44,True);ck(torch.equal(ds[(3,int(ids[0]))][0],ds[(3,int(ids[0]))][0]),'augmentation reproducible by sample key')
        base_method={'name':'shared_initial','regularizer':'none','acquisition':'none'}
        init,source=training.run_phase(small,cat,801,base_method,0,ids,hold,None,None,root/'initial','test-binding',20,lambda x:None,'cpu',model_factory=Tiny)
        ck(init['steps']==8,'real CPU initialization loop')
        model=Tiny(4);model.load_state_dict(torch.load(source,weights_only=False,map_location='cpu')['model'])
        targets,_=calibrate(model,cat,ids,801,small,'cpu');ck(len(targets)==2,'common target calibration')
        before=tensor_digest(model.state_dict());pool=np.setdiff1d(np.arange(144),ann)
        base=infer_local(model,cat,pool,small)
        nf=noise_features(model,cat,pool,base,801,1,small,torch.device('cpu'),lambda x:None)
        ck(tensor_digest(model.state_dict())==before,'noise perturbations restore exact weights and buffers')
        ck(nf.shape==(len(pool),8),'noise features exact, not sketched')
        for method in small['methods']:
            out=root/method['name']
            chosen,info=select_round(model,cat,ids,hold,method,801,1,small,torch.device('cpu'),out/'selection','test-binding',sha256_local(source),lambda x:None)
            ck(len(chosen)==16 and np.intersect1d(chosen,ann).size==0,'selection '+method['name'])
            chosen2,_=select_round(model,cat,ids,hold,method,801,1,small,torch.device('cpu'),out/'selection','test-binding',sha256_local(source),lambda x:None)
            ck(np.array_equal(chosen,chosen2),'selection resume '+method['name'])
            labeled=np.sort(np.concatenate([ids,chosen]));summary,ckpt=training.run_phase(small,cat,801,method,1,labeled,hold,source,targets,out/'phase1','test-binding',20,lambda x:None,'cpu',model_factory=Tiny)
            state=torch.load(ckpt,weights_only=False,map_location='cpu')
            ck(summary['steps']==12,'AL phase real backward '+method['name'])
            orig=torch.load(source,weights_only=False,map_location='cpu')['model']
            ck(torch.equal(orig['bn.running_mean'],state['model']['bn.running_mean']),'BN freeze matched '+method['name'])
            if method['regularizer']=='aligned_full':
                ck(summary['learned_updates_cumulative']>=5 and state['controller']['controller']['pending'] is None,'Full MACC terminal credit preserved')
                assert state['controller']['controller']['pending_acq'] is None
        # Full MACC interrupted after committing a pending-action checkpoint.
        method=next(m for m in small['methods'] if m['name']=='caenl_aligned_full')
        reference_path=root/method['name']/'phase1/state.pt';reference=torch.load(reference_path,weights_only=False,map_location='cpu')
        labeled=reference['labeled'];original=training.atom_torch
        def interrupt(path,state):
            original(path,state)
            if state.get('step')==4:raise InterruptedError('synthetic interrupted after a checkpoint commit')
        training.atom_torch=interrupt
        try:
            training.run_phase(small,cat,801,method,1,labeled,hold,source,targets,root/'resume','test-binding',20,lambda x:None,'cpu',model_factory=Tiny)
        except InterruptedError:pass
        else:raise AssertionError('Interruption not triggered')
        finally:training.atom_torch=original
        _,afterpath=training.run_phase(small,cat,801,method,1,labeled,hold,source,targets,root/'resume','test-binding',20,lambda x:None,'cpu',model_factory=Tiny)
        after=torch.load(afterpath,weights_only=False,map_location='cpu')
        ck(tensor_digest(reference['model'])==tensor_digest(after['model']),'resume model bitwise identical on CPU')
        ck(tensor_digest(reference['controller']['controller']['policy'])==tensor_digest(after['controller']['controller']['policy']),'resume learned policy identical on CPU')
        ck(reference['controller']['history']==after['controller']['history'],'resume rewards/actions identical on CPU')
        ck(reference['batch_chain']==after['batch_chain'],'resume batch stream identical')
        phase2ids=np.sort(np.concatenate([labeled,np.setdiff1d(pool,labeled)[:16]]))
        second,_=training.run_phase(small,cat,801,method,2,phase2ids,hold,afterpath,targets,root/'round2','test-binding',20,lambda x:None,'cpu',model_factory=Tiny)
        ck(second['learned_updates_cumulative']>reference['controller']['windows'],'controller survives actual phase boundary')
        ck(second['annotation_count']==80,'second AL budget exact')
    print('SELF-TEST PASS:',len(checks),'checks; all nine real CPU branches and interrupted/resumed Full MACC')
    return checks

def infer_local(model,cat,pool,cfg):
    from evaluation import infer
    return infer(model,cat,'train',pool,cfg,'cpu')['logits']
def sha256_local(p):
    from support import sha256
    return sha256(p)

if __name__=='__main__':run()
