"""Audited aligned-DCR parameters; Full MACC controls regularization, NOT query thresholds."""
from __future__ import annotations
import math,copy
import numpy as np
import torch
from aligned import class_scatter,positive_huber,LeakyMACCLite
from vendor.controllers import FullMACC
from support import seed_for,loader,to_device,BalancedBatches

@torch.no_grad()
def calibrate(model,cat,ids,seed,cfg,device):
    model.eval();values={l:[] for l in cfg['layers']}
    batches=BalancedBatches(ids,cat.labels,seed_for('phase0_anchors',seed),cfg['calibration_batches'],cfg['batch_classes'],cfg['samples_per_class'])
    for batch in loader(cat,'train',batches,seed,0,False):
        x,y,_=to_device(batch,device);_,z=model.forward_features(x)
        for l in values:
            s=class_scatter(z[l],y,eps=cfg['alignment']['scatter_epsilon'],norm_eps=cfg['alignment']['feature_norm_epsilon'])
            values[l].append({k:float(v) for k,v in s.items()})
    targets={}
    for l,vs in values.items():
        q=float(np.mean([r['log_kappa'] for r in vs]));b=float(np.mean([r['between'] for r in vs]))
        if b<=10*cfg['alignment']['scatter_epsilon']:raise ValueError('Insufficient class separation in initial model to define aligned targets')
        targets[l]={'log_kappa_reference':q,'log_kappa_target':q+math.log(cfg['alignment']['target_kappa_factor']),'between_reference':b,'between_floor':b*cfg['alignment']['between_floor_factor']}
    return targets,values

class Control:
    def __init__(self,kind,targets,seed,cfg,device,total_windows):
        self.kind=kind;self.targets=copy.deepcopy(targets);self.cfg=cfg;self.layers=cfg['layers'];self.ctrl=None
        self.ema={l:{'log_kappa':targets[l]['log_kappa_reference'],'between':targets[l]['between_reference']} for l in self.layers}
        self.weights={l:(cfg['lambda_fixed'] if kind!='none' else 0.) for l in self.layers}
        self.prev_q={l:self.ema[l]['log_kappa'] for l in self.layers};self.history=[];self.windows=0;self.total_windows=total_windows
        self.prev_ce=None;self.ce_scale=None;self.prev_potential=None
        if kind=='aligned_lite':self.ctrl=LeakyMACCLite(self.layers,initial=cfg['lambda_initial'],maximum=cfg['lambda_max'],smoothing=cfg['lite_smoothing'],scale=cfg['alignment']['huber_delta'])
        if kind=='aligned_full':
            f=cfg['full_macc']
            self.ctrl=FullMACC(self.layers,{l:math.exp(targets[l]['log_kappa_target']) for l in self.layers},
                state_dim=FullMACC.state_dim_for(len(self.layers),1),lambda_initial=cfg['lambda_initial'],lambda_min=0.,lambda_max=cfg['lambda_max'],hidden_dims=f['hidden_dims'],lambda_delta_choices=f['lambda_delta_choices'],policy_lr=f['policy_lr'],epsilon_start=f['epsilon_start'],epsilon_end=f['epsilon_end'],total_controller_steps=total_windows,reward_baseline_ema=f['reward_baseline_ema'],entropy_coefficient=f['entropy_coefficient'],device=device,seed=seed_for('aligned_policy',seed),quantile_head_enabled=False)
    def gaps(self):return {l:self.ema[l]['log_kappa']-self.targets[l]['log_kappa_target'] for l in self.layers}
    def potential(self):
        a=self.cfg['alignment'];v=0.
        for l,g in self.gaps().items():
            bg=math.log(self.targets[l]['between_floor'])-math.log(self.ema[l]['between']+a['scatter_epsilon'])
            v+=float(positive_huber(torch.tensor(g,dtype=torch.float64),a['huber_delta']))+a['floor_weight']*float(positive_huber(torch.tensor(bg,dtype=torch.float64),a['huber_delta']))
        return v
    def observe(self,records):
        b=self.cfg['training_ema_beta']
        for l in self.layers:
            for k in ('log_kappa','between'):self.ema[l][k]=(1-b)*self.ema[l][k]+b*records[l][k]
    def act(self,ce,global_step):
        if self.kind=='aligned_lite':self.weights=self.ctrl.step(self.gaps(),global_step)
        elif self.kind=='aligned_full':
            k={l:math.exp(self.ema[l]['log_kappa']) for l in self.layers}
            instability={l:abs(self.ema[l]['log_kappa']-self.prev_q[l])*k[l] for l in self.layers}
            state=self.ctrl.build_state(k,instability,min(1.,self.windows/max(1,self.total_windows)),(ce,))
            action=self.ctrl.act(state);self.weights=action.effective_lambdas()
            if action.threshold_quantile!=.9:raise ValueError('Acquisition threshold must remain fixed')
            return action.extra
        return {}
    def begin_phase(self,ce,global_step):
        if self.kind=='aligned_full' and self.ctrl._pending is not None:raise RuntimeError('Previous-phase reward not consumed')
        self.prev_ce=ce;self.ce_scale=max(.1,ce);self.prev_potential=self.potential()
        action=self.act(ce,global_step)
        self.history.append({'kind':'phase_initial_action','global_step':global_step,'control_ce':ce,'lambdas':dict(self.weights),'action':action})
    def window(self,ce,phase,step,global_step,terminal):
        potential=self.potential();record={'kind':'control_window','phase':phase,'step':step,'global_step':global_step,'control_ce':ce,'gaps':self.gaps(),'applied_lambdas':dict(self.weights),'official_validation_used':False,'terminal':terminal}
        if self.kind=='aligned_full':
            f=self.cfg['full_macc'];vt=max(-1.,min(1.,(self.prev_ce-ce)/self.ce_scale));gt=f['reward_geometry_weight']*max(-1.,min(1.,self.prev_potential-potential));et=-f['reward_effort_weight']*np.mean(list(self.weights.values()))/self.cfg['lambda_max']
            reward=float(vt+gt+et);record['update']=self.ctrl.update(reward);record['reward_components']={'task':vt,'geometry':gt,'effort':float(et),'total':reward}
        self.windows+=1
        if not terminal:record['next_action']=self.act(ce,global_step)
        self.prev_q={l:self.ema[l]['log_kappa'] for l in self.layers};self.prev_ce=ce;self.prev_potential=potential
        record['next_lambdas']=dict(self.weights);self.history.append(record)
        return record
    def state_dict(self):
        return {'kind':self.kind,'ema':copy.deepcopy(self.ema),'weights':dict(self.weights),'prev_q':dict(self.prev_q),'history':copy.deepcopy(self.history),'windows':self.windows,'prev_ce':self.prev_ce,'ce_scale':self.ce_scale,'prev_potential':self.prev_potential,'controller':self.ctrl.state_dict() if self.ctrl else None}
    def load_state_dict(self,state):
        if state['kind']!=self.kind:raise ValueError('Controller kind mismatch')
        for k in ('ema','weights','prev_q','history','windows','prev_ce','ce_scale','prev_potential'):setattr(self,k,copy.deepcopy(state[k]))
        if self.ctrl:self.ctrl.load_state_dict(state['controller'])
