"""Real CPU training, interruption/resume and scientific-design contracts on tiny fixtures."""
import contextlib,copy,io,json,math,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from PIL import Image
from support import read_json,seed_for,tensor_digest,Model,sha256
from study import candidates,choose_controls,planned_counts,method_config,evaluated_methods,replay_methods
from analysis import paired,holm,primary_analysis
from aligned import class_scatter,aligned_loss
from contrastive import supcon
from data_audit import group_policy,decoded_hash,audit_catalog
import training

class Tiny(nn.Module):
    def __init__(self,classes=4):
        super().__init__();self.a=nn.Linear(3,16);self.bn=nn.BatchNorm1d(16);self.b=nn.Linear(16,12);self.fc=nn.Linear(12,classes)
    def forward_features(self,x):
        z3=torch.tanh(self.bn(self.a(x.mean((2,3)))));z4=torch.tanh(self.b(z3));return self.fc(z4),{'layer3':z3,'layer4':z4}
    def forward(self,x):return self.forward_features(x)[0]

def fixture(root,train_count=20):
    rows={'train':[],'val':[]}
    for split,n in [('train',train_count),('val',4)]:
        for c in range(4):
            for j in range(n):
                rel=f'{split}/class{c}/{j:03d}.png';p=root/rel;p.parent.mkdir(parents=True,exist_ok=True)
                a=np.random.default_rng(seed_for(split,c,j)).integers(20+c*40,50+c*40,size=(18,22,3),dtype=np.uint8);a[:,:,c%3]=40+c*45;Image.fromarray(a).save(p)
                rows[split].append(dict(path=rel,class_id=c,sha256=sha256(p)))
        rows[split].sort(key=lambda r:r['path'])
    return SimpleNamespace(root=root,rows=rows,labels=np.array([r['class_id'] for r in rows['train']]),val_labels=np.array([r['class_id'] for r in rows['val']]))

def run():
    from runner import run_study,verify_method,freeze,read_rows,prune_state
    torch.set_num_threads(2);checks=[]
    def ck(ok,name):
        if not bool(ok):raise AssertionError(name)
        checks.append(name)
    def rejects(fn,name):
        try:fn()
        except (ValueError,RuntimeError):checks.append(name)
        else:raise AssertionError(name)
    cfg=read_json(Path(__file__).with_name('protocol.json'))
    ck(planned_counts(cfg)['method_runs']==206 and planned_counts(cfg)['initializations']==23,'206 method runs / 23 initializations')
    ck(len(candidates(cfg))==12,'equal four-candidate grids for fixed, schedule and contrastive controls')
    sets=[set(cfg['study'][k]) for k in ['tuning_seeds','confirmation_seeds','transfer_seeds']]
    ck(not any(sets[i]&sets[j] for i in range(3) for j in range(i)),'disjoint selection and evaluation seeds')
    ck(not set.union(*sets)&set(range(41000,44000)),'fresh seeds relative to historical feedback study')
    rows=[dict(stage='tuning',seed=s,method=m['name'],holdout=dict(accuracy=.5,ce=1.),official_validation=None) for s in cfg['study']['tuning_seeds'] for m in candidates(cfg)]
    selected=choose_controls(rows,cfg);ck(set(selected)>={'fixed','schedule','supcon'},'all three selections completed')
    c=method_config(cfg,{'name':'sc','regularizer':'supcon','coefficient':.003,'acquisition':'entropy'},1,'tuning','resnet50')
    ck(c['lambda_fixed']==c['supcon_coefficient']==.003,'contrastive coefficient logging matches applied value')
    rejects(lambda:choose_controls(rows[:-1],cfg),'partial tuning rejected')
    bad=copy.deepcopy(rows);bad[0]['official_validation']={};rejects(lambda:choose_controls(bad,cfg),'official validation in selection rejected')
    ck(primary_analysis([],cfg)['status']=='INCOMPLETE','inference withheld on partial family')
    ck(paired([1.]*10,[0.]*10)['exact_signflip_p']==2/1024,'ten-seed exact resolution')
    ck(holm([.01,.04,.5])==[.03,.08,.5],'Holm implementation')
    # Label conflicts are removed in entirety; within-split identical labels retain one record.
    records=[dict(path='train/a',split='train',class_id=0,pixel_sha256='x'),dict(path='val/b',split='val',class_id=0,pixel_sha256='x'),dict(path='train/c',split='train',class_id=0,pixel_sha256='y'),dict(path='train/d',split='train',class_id=1,pixel_sha256='y'),dict(path='train/f',split='train',class_id=2,pixel_sha256='z'),dict(path='train/e',split='train',class_id=2,pixel_sha256='z')]
    kept,excluded,_=group_policy(records);ck([r['path'] for r in kept]==['train/e'] and len(excluded)==5,'conflict and content grouping policy')
    ck(group_policy(list(reversed(records)))==group_policy(records),'data policy order invariant')
    torch.manual_seed(3);z=torch.randn(8,7,requires_grad=True);y=torch.arange(4).repeat_interleave(2)
    loss=supcon(z,y);h=z.detach().numpy();h=h/np.linalg.norm(h,axis=1,keepdims=True);scores=h@h.T/.1
    expected=np.mean([-np.mean([scores[i,j]-np.log(np.exp(np.delete(scores[i],i)).sum()) for j in range(8) if j!=i and y[i]==y[j]]) for i in range(8)])
    ck(abs(float(loss.detach())-expected)<2e-6,'contrastive loss agrees with independent NumPy equation')
    loss.backward();ck(torch.isfinite(z.grad).all() and z.grad.norm()>0,'contrastive gradients finite and nonzero')
    rejects(lambda:supcon(z,torch.arange(8)),'anchors without positives rejected')
    g=class_scatter(z.detach().double(),y);t={'a':dict(log_kappa_target=float(g['log_kappa'])-.2,between_floor=float(g['between'])*2)}
    a=cfg['alignment'];l0,_=aligned_loss({'a':z},y,t,{'a':0.},{**a,'floor_weight':0.});lr,_=aligned_loss({'a':z},y,t,{'a':.1},{**a,'floor_weight':0.});ls,_=aligned_loss({'a':z},y,t,{'a':0.},a);lb,_=aligned_loss({'a':z},y,t,{'a':.1},a)
    ck(float(l0)==0 and torch.allclose(lb,lr+ls),'2x2 penalty decomposition')
    for backbone in ['resnet50','resnet18']:
        m=Model(4,backbone);logits,features=m.forward_features(torch.rand(8,3,32,32));v=logits.square().mean()+.01*supcon(features['layer4'],y);v.backward()
        ck(torch.isfinite(m.model.net.fc.weight.grad).all(),'real '+backbone+' CPU forward/backward with new penalty');del m,logits,features,v
    with tempfile.TemporaryDirectory(prefix='caenl-solid-cpu-') as temp:
        base=Path(temp);cat=fixture(base/'images');small=copy.deepcopy(cfg)
        small.update(num_classes=4,initial_labels=24,holdout_count=8,acquisition_per_round=8,rounds=2,initial_epochs=1,round_epochs=1,batch_classes=4,samples_per_class=2,workers=0,eval_workers=0,eval_batch=16,control_interval=2,calibration_batches=2,checkpoint_every_steps=1,warmup_initial_epochs=0,warmup_round_epochs=0)
        small['acquisition'].update(candidate_size=16,per_pass_fraction=.25)
        small['study'].update(tuning_seeds=[97001],confirmation_seeds=[97002,97003],transfer_seeds=[97004,97005],fixed_coefficients=[.03],schedule_shapes=['linear_up'],schedule_peaks=[.03],supcon_coefficients=[.03])
        audited=fixture(base/'audit_images',110);audit_cfg=copy.deepcopy(cfg);audit_cfg['num_classes']=4
        cleaned,report=audit_catalog(audited,audit_cfg,base/'audit',lambda x:None)
        cleaned2,report2=audit_catalog(audited,audit_cfg,base/'audit',lambda x:None)
        ck(len(cleaned.labels)==440 and len(cleaned.val_labels)==16 and report==report2,'full pixel audit and cached policy resume')
        root=base/'run';root.mkdir()
        rec=run_study(root,small,cat,'cpu-fixture',lambda x:None,'cpu',factory=Tiny,prune=True)
        ck(rec['completed_methods']==37 and rec['analysis']['status']=='COMPLETE' and len(rec['analysis']['contrasts'])==17,'complete 37-run tiny campaign and 17-contrast family')
        ck(not list((root/'tuning').rglob('validation.npz')),'no official validation inference during tuning')
        ck(len(list(root.glob('*/seed*/*/FINAL_MODEL.pt')))==37,'all final model weights retained')
        ck(not list(root.glob('*/seed*/*/rounds/round*/state.pt')),'only new superseded optimizer states pruned')
        oldbackward=torch.Tensor.backward
        def forbidden(*a,**k):raise AssertionError('Completed campaign retrained')
        torch.Tensor.backward=forbidden
        try:run_study(root,small,cat,'cpu-fixture',lambda x:None,'cpu',factory=Tiny,prune=True)
        finally:torch.Tensor.backward=oldbackward
        ck(True,'complete resume without training after pruning')
        sd=root/'confirmation/seed97002';baseline=read_json(sd/'entropy/summary.json');full=read_json(sd/'full_labels/summary.json')
        ck(baseline['training_steps']==full['training_steps'] and full['annotation_count']==len(cat.labels),'100% annotation / matched optimizer step budget')
        for name in ['replay_ratio','replay_separation','replay_fixed','replay_lite','replay_schedule']:
            rr=read_json(sd/name/'summary.json')
            ck([r['batch_chain_sha256'] for r in rr['rounds']]==[r['batch_chain_sha256'] for r in baseline['rounds']],'actual minibatch identity: '+name)
        # A phase interrupted after saving must match a clean training execution, including momentum.
        with np.load(sd/'splits.npz') as a:hold=a['holdout'];initial=a['initial']
        targets=read_json(sd/'targets.json')['targets'];source=sd/'shared_initial/state.pt'
        method=dict(name='resume_supcon',regularizer='supcon',coefficient=.03,acquisition='entropy');c=method_config(small,method,1,'confirmation','resnet50')
        with np.load(sd/'entropy/acquisitions/round1/selection.npz') as a:labeled=np.sort(np.r_[initial,a['selected']])
        _,reference=training.run_phase(c,cat,97002,method,1,labeled,hold,source,targets,base/'reference','resume-test',4,lambda x:None,'cpu',model_factory=Tiny)
        save=training.atom_torch
        def interrupt(path,state):
            save(path,state)
            if state['step']==2:raise InterruptedError('Deliberate checkpoint interruption')
        training.atom_torch=interrupt
        try:training.run_phase(c,cat,97002,method,1,labeled,hold,source,targets,base/'resumed','resume-test',4,lambda x:None,'cpu',model_factory=Tiny)
        except InterruptedError:pass
        else:raise AssertionError('Interruption fixture failed')
        finally:training.atom_torch=save
        _,resumed=training.run_phase(c,cat,97002,method,1,labeled,hold,source,targets,base/'resumed','resume-test',4,lambda x:None,'cpu',model_factory=Tiny)
        aa=torch.load(reference,map_location='cpu',weights_only=False);bb=torch.load(resumed,map_location='cpu',weights_only=False)
        ck(tensor_digest(aa['model'])==tensor_digest(bb['model']) and aa['batch_chain']==bb['batch_chain'],'bitwise interrupted/resumed model and sampled batch sequence')
        ck(all(torch.equal(v['momentum_buffer'],bb['optimizer']['state'][k]['momentum_buffer']) for k,v in aa['optimizer']['state'].items()),'optimizer momentum survives interruption')
        bad=sd/'lite/rounds/round1/training.csv';b=bad.read_bytes();bad.write_bytes(b+b'\n');rejects(lambda:verify_method(sd/'lite'),'altered trace rejected');bad.write_bytes(b)
        rejects(lambda:freeze(root/'SELECTION.json',{}),'selection overwrite rejected')
        p1=base/'same_a.png';p2=base/'same_b.png';im=Image.new('RGB',(10,12),(1,2,3));im.save(p1,compress_level=0);im.save(p2,compress_level=9)
        ck(sha256(p1)!=sha256(p2) and decoded_hash(p1)==decoded_hash(p2),'decoded identity detects byte-distinct copies')
    print('CPU_INTEGRATION_PASS:',len(checks),'checks; real CPU training, reloads, pruning and interruption recovery. No scientific GPU result.')
    return checks

if __name__=='__main__':
    checks=run();print(json.dumps(checks,indent=2))
