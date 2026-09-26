"""CPU contracts plus real training, complete tiny-study orchestration and resume tests."""
from __future__ import annotations
import copy, json, math, tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from PIL import Image
from support import read_json, seed_for, tensor_digest, Model, sha256
from aligned import class_scatter, aligned_loss, LeakyMACCLite
from study import candidates, scheduled_strength, choose_controls, method_config, planned_counts
from analysis import paired, holm, primary_analysis
from control import Control
import training


class Tiny(nn.Module):
    def __init__(self,classes=4):
        super().__init__();self.l1=nn.Linear(3,16);self.bn=nn.BatchNorm1d(16)
        self.l2=nn.Linear(16,12);self.fc=nn.Linear(12,classes)
    def forward_features(self,x):
        z3=torch.tanh(self.bn(self.l1(x.mean((2,3)))));z4=torch.tanh(self.l2(z3))
        return self.fc(z4),{'layer3':z3,'layer4':z4}
    def forward(self,x):return self.forward_features(x)[0]


def fixture(root):
    rows={'train':[],'val':[]}
    for split,n in [('train',20),('val',4)]:
        for c in range(4):
            for j in range(n):
                rel=f'{split}/class{c}/{j:03d}.png';path=root/rel;path.parent.mkdir(parents=True,exist_ok=True)
                a=np.random.default_rng(seed_for(split,c,j)).integers(20+c*40,50+c*40,size=(18,22,3),dtype=np.uint8)
                a[:,:,c%3]=40+c*45;Image.fromarray(a).save(path)
                rows[split].append({'path':rel,'class_id':c})
        rows[split].sort(key=lambda x:x['path'])
    return SimpleNamespace(root=root,rows=rows,
            labels=np.asarray([r['class_id'] for r in rows['train']]),
            val_labels=np.asarray([r['class_id'] for r in rows['val']]))


def run():
    from campaign import run_study, verify_method, freeze, read_rows
    from acquisition import select_round
    torch.set_num_threads(2);torch.manual_seed(92117);checks=[]
    def ck(value,label):
        if not bool(value):raise AssertionError(label)
        checks.append(label)
    def rejects(fn,label):
        try:fn()
        except (ValueError,RuntimeError):checks.append(label)
        else:raise AssertionError(label)
    cfg=read_json(Path(__file__).with_name('protocol.json'));grid=candidates(cfg)
    ck(len(grid)==8 and sum(c['family']=='fixed' for c in grid)==4 and sum(c['family']=='schedule' for c in grid)==4,
       'equal four-candidate control grids')
    counts=planned_counts(cfg)
    ck(counts==dict(tuning_runs=24,confirmation_runs=40,transfer_runs=40,method_runs=104,
                   initializations=23,postinitial_phases=520,optimizer_steps=4155880),'frozen workload accounting')
    sets=[set(cfg['study'][k]) for k in ('tuning_seeds','confirmation_seeds','transfer_seeds')]
    ck(not any(sets[i]&sets[j] for i in range(3) for j in range(i)),'tuning and evaluation seeds disjoint')
    ck(not set.union(*sets)&set(range(801,807)),'no reuse of published classification seeds')
    ck(cfg['lambda_initial']==.03 and cfg['lambda_max']==.1 and cfg['lite_smoothing']==.2,'published Lite settings preserved')
    for shape in ('linear_up','linear_down'):
        v=[scheduled_strength(shape,.1,t/100) for t in range(101)]
        ck(min(v)>=0 and max(v)<=.1 and abs(np.mean(v)-.05)<1e-12,'bounded '+shape+' schedule')
    rejects(lambda:scheduled_strength('linear_up',.1,1.1),'invalid schedule progress rejected')
    target={l:dict(log_kappa_reference=1.,log_kappa_target=.9,between_reference=.2,between_floor=.1) for l in cfg['layers']}
    spec=next(x for x in grid if x['regularizer']=='aligned_schedule')
    c1=method_config(cfg,spec,1,'confirmation','resnet50');c2=method_config(cfg,spec,2,'confirmation','resnet50')
    a=Control('aligned_schedule',target,1,c1,'cpu',400);b=Control('aligned_schedule',target,1,c1,'cpu',400)
    a.begin_phase(1.,0);b.begin_phase(9.,0)
    b.ema={l:dict(log_kappa=-100.,between=100.) for l in cfg['layers']}
    a.window(1.,1,100,100,False);b.window(9.,1,100,100,False)
    ck(a.weights==b.weights,'open-loop strengths do not depend on losses or geometry')
    b2=Control('aligned_schedule',target,1,c2,'cpu',400);b2.load_state_dict(a.state_dict());b2.begin_phase(2.,0)
    expected=scheduled_strength(spec['shape'],spec['peak'],3880/39120)
    ck(all(abs(v-expected)<1e-12 for v in b2.weights.values()),'schedule uses global progress across phase boundaries')
    control=Control('aligned_lite',target,1,cfg,'cpu',400)
    ref=LeakyMACCLite(cfg['layers'],initial=.03,maximum=.1,smoothing=.2,scale=.25)
    control.begin_phase(1.,0);ref.step({l:.1 for l in cfg['layers']},0)
    ck(control.weights==ref.lambdas,'unchanged initial Lite response')
    ck(holm([.01,.04,.5])==[.03,.08,.5],'Holm arithmetic')
    p=paired([1.]*10,[0.]*10)
    ck(p['exact_signflip_p']==2/1024 and p['degenerate_note'] is not None,'ten-seed exact resolution and degenerate t handling')
    ck(paired([0.]*10,[0.]*10)['paired_t_p']==1.,'zero differences accepted')
    from scipy import stats
    x=np.array([.2,.4,-.1,.8,.5,.3,.7,.1,.4,.2]);p=paired(x,np.zeros(10))
    ck(abs(p['paired_t_p']-float(stats.ttest_1samp(x,0).pvalue))<1e-12,'paired t agrees with independent scipy calculation')
    duplicate=dict(stage='confirmation',seed=42001,method='lite')
    rejects(lambda:primary_analysis([duplicate,duplicate],cfg),'duplicate paired records rejected')
    rows=[dict(stage='tuning',seed=seed,method=c['name'],holdout=dict(accuracy=.5,ce=1.),official_validation=None)
          for seed in cfg['study']['tuning_seeds'] for c in grid]
    selection=choose_controls(rows,cfg)
    ck(selection['fixed']['name']==min(c['name'] for c in grid if c['family']=='fixed'),'deterministic selection tie break')
    rejects(lambda:choose_controls(rows[:-1],cfg),'partial tuning grid rejected')
    bad=copy.deepcopy(rows);bad[0]['official_validation']={'accuracy':.9}
    rejects(lambda:choose_controls(bad,cfg),'official-validation selection leakage rejected')
    ck(primary_analysis([],cfg)['status']=='INCOMPLETE','partial inference family withheld')
    z=torch.randn(16,12,dtype=torch.float64,requires_grad=True);y=torch.arange(4).repeat_interleave(4)
    s=class_scatter(z,y);ck(torch.allclose(s['within']+s['between'],s['total'],atol=1e-12),'scatter decomposition')
    t={'a':{'log_kappa_target':float(s['log_kappa'].detach())-.2,'between_floor':float(s['between'].detach())*.5}}
    loss,_=aligned_loss({'a':z},y,t,{'a':.03},cfg['alignment']);g=torch.autograd.grad(loss,z)[0]
    ck(torch.isfinite(g).all() and g.norm()>0,'nonzero aligned gradient')
    loss2,_=aligned_loss({'a':z-.01*g},y,t,{'a':.03},cfg['alignment']);ck(loss2<loss,'isolated regularizer descent')
    # Actual TorchVision architectures are exercised on small CPU tensors as well.
    for backbone in ('resnet50','resnet18'):
        m=Model(4,backbone);out,features=m.forward_features(torch.rand(8,3,32,32))
        q=class_scatter(features['layer4'],torch.arange(4).repeat_interleave(2))
        (out.square().mean()+.01*q['log_kappa']).backward()
        ck(m.model.net.fc.weight.grad is not None and torch.isfinite(m.model.net.fc.weight.grad).all(),
           'real '+backbone+' CPU forward/backward')
        del m,out,features,q
    with tempfile.TemporaryDirectory(prefix='caenl-feedback-cpu-') as temp:
        root=Path(temp);small=copy.deepcopy(cfg)
        small.update(num_classes=4,initial_labels=24,holdout_count=8,acquisition_per_round=8,rounds=2,
                     initial_epochs=1,round_epochs=1,batch_classes=4,samples_per_class=2,
                     workers=0,eval_workers=0,eval_batch=16,control_interval=2,calibration_batches=2,
                     checkpoint_every_steps=1,warmup_initial_epochs=0,warmup_round_epochs=0)
        small['acquisition'].update(candidate_size=16,per_pass_fraction=.25)
        small['study'].update(tuning_seeds=[97001],confirmation_seeds=[97002,97003],transfer_seeds=[97004,97005],
                             fixed_coefficients=[.03],schedule_shapes=['linear_up'],schedule_peaks=[.03])
        cat=fixture(root/'images');study_root=root/'results'
        record=run_study(study_root,small,cat,'cpu-fixture',lambda x:None,'cpu',model_factory=Tiny)
        ck(record['completed_method_runs']==18,'complete tiny study: tuning, selection, both fresh-seed matrices')
        ck(record['primary_analysis']['status']=='COMPLETE' and len(record['primary_analysis']['contrasts'])==6,
           'complete family of six contrasts')
        ck(not list((study_root/'tuning').rglob('validation.npz')),'no official validation files generated while tuning')
        frozen=(study_root/'SELECTION.json').read_bytes()
        ck(read_json(study_root/'SELECTION.json')['official_validation_used'] is False,'selection explicitly excludes official validation')
        # Prove that rerunning a completed study does not invoke a single training update.
        original_backward=torch.Tensor.backward
        def forbidden(*a,**kw):raise AssertionError('Completed study retrained')
        torch.Tensor.backward=forbidden
        try:run_study(study_root,small,cat,'cpu-fixture',lambda x:None,'cpu',model_factory=Tiny)
        finally:torch.Tensor.backward=original_backward
        ck((study_root/'SELECTION.json').read_bytes()==frozen,'complete-study resume reuses results and frozen selection')
        sd=study_root/'confirmation'/'seed97002';split=dict(np.load(sd/'splits.npz',allow_pickle=False))
        initial_ckpt=sd/'shared_initial'/'state.pt';targets=read_json(sd/'targets.json')['targets']
        for name in ('lite','tuned_schedule'):
            summary=read_json(sd/name/'summary.json');method=summary['method_spec']
            c=method_config(small,method,1,'confirmation','resnet50')
            refstate=torch.load(sd/name/'rounds'/'round1'/'state.pt',map_location='cpu',weights_only=False)
            original_save=training.atom_torch
            def interrupt(path,state):
                original_save(path,state)
                if state['step']==2:raise InterruptedError('Deliberate CPU interruption after atomic checkpoint')
            training.atom_torch=interrupt
            try:
                training.run_phase(c,cat,97002,method,1,refstate['labeled'],split['holdout'],initial_ckpt,
                    targets,root/('resume_'+name),summary['binding'],4,lambda x:None,'cpu',model_factory=Tiny)
            except InterruptedError:pass
            else:raise AssertionError('Test interruption did not occur')
            finally:training.atom_torch=original_save
            _,path=training.run_phase(c,cat,97002,method,1,refstate['labeled'],split['holdout'],initial_ckpt,
                    targets,root/('resume_'+name),summary['binding'],4,lambda x:None,'cpu',model_factory=Tiny)
            after=torch.load(path,map_location='cpu',weights_only=False)
            ck(tensor_digest(after['model'])==tensor_digest(refstate['model']),'bitwise interrupted/resumed model: '+name)
            ck(after['controller']==refstate['controller'],'identical feedback/schedule history after resume: '+name)
            ck(after['batch_chain']==refstate['batch_chain'],'identical sampled batches after resume: '+name)
            for key,value in refstate['optimizer']['state'].items():
                ck(torch.equal(value['momentum_buffer'],after['optimizer']['state'][key]['momentum_buffer']),
                   'optimizer momentum preserved: '+name+':'+str(key))
        # Entropy selection must be unchanged when hidden pool labels are permuted.
        model=Tiny(4);model.load_state_dict(torch.load(initial_ckpt,map_location='cpu',weights_only=False)['model'])
        method={'name':'blindness_test','acquisition':'entropy','regularizer':'none'}
        chosen,_=select_round(model,cat,split['initial'],split['holdout'],method,97002,1,small,'cpu',
                             root/'blind_a','blind',sha256(initial_ckpt),lambda x:None)
        altered=copy.deepcopy(cat);pool=np.setdiff1d(np.arange(len(cat.labels)),split['annotated_initial'])
        for ix in pool:altered.labels[ix]=(altered.labels[ix]+1)%4;altered.rows['train'][ix]['class_id']=int(altered.labels[ix])
        chosen2,_=select_round(model,altered,split['initial'],split['holdout'],method,97002,1,small,'cpu',
                              root/'blind_b','blind',sha256(initial_ckpt),lambda x:None)
        ck(np.array_equal(chosen,chosen2),'entropy acquisition invariant to hidden pool-label permutation')
        path=sd/'lite'/'rounds'/'round1'/'training.csv';data=path.read_bytes();path.write_bytes(data+b'\n')
        rejects(lambda:verify_method(sd/'lite'),'completed payload tampering rejected');path.write_bytes(data)
        rejects(lambda:freeze(study_root/'SELECTION.json',{'changed':True}),'frozen selection overwrite rejected')
    print(f'SELF-TEST PASS: {len(checks)} checks; real CPU training and interruption recovery.')
    return checks


if __name__=='__main__':run()
