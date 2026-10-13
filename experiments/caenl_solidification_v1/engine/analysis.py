"""Paired-seed inference for the frozen seventeen-contrast family; no partial-family tests."""
from __future__ import annotations
import itertools, math
import numpy as np
from scipy import stats


def holm(pvalues):
    if any(not math.isfinite(p) or not 0<=p<=1 for p in pvalues):
        raise ValueError('Invalid p-value')
    order=sorted(range(len(pvalues)),key=lambda i:pvalues[i]);out=[None]*len(pvalues);last=0.
    for j,i in enumerate(order):
        last=max(last,min(1.,(len(order)-j)*pvalues[i]));out[i]=last
    return out


def paired(target, reference):
    d=np.asarray(target,dtype=np.float64)-np.asarray(reference,dtype=np.float64)
    if d.ndim!=1 or len(d)<2 or not np.isfinite(d).all():raise ValueError('Invalid paired data')
    n=len(d);mean=float(d.mean());sd=float(d.std(ddof=1))
    if sd==0:
        lo=hi=mean;p=1. if mean==0 else 0.
        note='Degenerate paired differences; t result is a limiting value, not independent precision evidence.'
    else:
        se=sd/math.sqrt(n);half=float(stats.t.ppf(.975,n-1))*se
        lo,hi=mean-half,mean+half;p=float(2*stats.t.sf(abs(mean/se),n-1));note=None
    # Symmetry/sign-exchangeability is an assumption; enumeration alone is not assumption-free.
    absolute=abs(mean);extreme=sum(abs(float(np.dot(d,signs)/n))>=absolute-1e-14
                                 for signs in itertools.product((-1.,1.),repeat=n))
    return dict(n_pairs=n,difference_pp=mean,sd_difference_pp=sd,ci95_low_pp=lo,ci95_high_pp=hi,
                paired_t_p=p,exact_signflip_p=extreme/(2**n),positive_seeds=int((d>0).sum()),
                negative_seeds=int((d<0).sum()),zero_seeds=int((d==0).sum()),degenerate_note=note)


def primary_analysis(rows,cfg):
    from study import candidates,evaluated_methods,replay_methods,stage_specs
    # Method identities are fixed; coefficient selection does not alter the family.
    evaluated=[r for r in rows if r['stage']!='tuning']
    data={(r['stage'],r['seed'],r['method']):r for r in evaluated}
    expected=set()
    for stage,key in [('confirmation','confirmation_seeds'),('transfer','transfer_seeds')]:
        methods=['entropy','tuned_fixed','tuned_schedule','lite','tuned_supcon','full_labels']
        if stage=='confirmation':methods+=['replay_ratio','replay_separation','replay_fixed','replay_lite','replay_schedule']
        expected.update((stage,seed,m) for seed in cfg['study'][key] for m in methods)
    if len(data)!=len(evaluated):raise ValueError('Duplicate result identity')
    if set(data)!=expected:return dict(status='INCOMPLETE',family_size=17,contrasts=[])
    contrasts=[]
    def values(stage,seeds,m):return np.array([100*data[(stage,s,m)]['official_validation']['accuracy'] for s in seeds])
    for stage,key in [('confirmation','confirmation_seeds'),('transfer','transfer_seeds')]:
        seeds=cfg['study'][key]
        for ref in ['entropy','tuned_fixed','tuned_schedule','tuned_supcon','full_labels']:
            contrasts.append(dict(stage=stage,target='lite',reference=ref,**paired(values(stage,seeds,'lite'),values(stage,seeds,ref))))
    seeds=cfg['study']['confirmation_seeds'];stage='confirmation'
    for target,ref in [('replay_ratio','entropy'),('replay_separation','entropy'),('replay_fixed','replay_ratio'),('replay_fixed','replay_separation'),('replay_lite','replay_fixed'),('replay_schedule','replay_fixed')]:
        contrasts.append(dict(stage='mechanism',target=target,reference=ref,**paired(values(stage,seeds,target),values(stage,seeds,ref))))
    d=values(stage,seeds,'replay_fixed')-values(stage,seeds,'replay_ratio')-values(stage,seeds,'replay_separation')+values(stage,seeds,'entropy')
    contrasts.append(dict(stage='mechanism',target='factorial_interaction',reference='zero',**paired(d,np.zeros(len(d)))))
    if len(contrasts)!=17:raise ValueError('Contrast family mismatch')
    t=holm([r['paired_t_p'] for r in contrasts]);e=holm([r['exact_signflip_p'] for r in contrasts])
    for r,tp,ep in zip(contrasts,t,e):r.update(holm_t_p=tp,holm_exact_p=ep)
    return dict(status='COMPLETE',family_size=17,contrasts=contrasts,intervals='Individual, unadjusted paired-t 95% intervals.',
        scope='Seeds are paired units. Fixed dataset and selection procedure. Separate from historical families; no programme-wide error guarantee.',
        label_savings='No automatic equivalence or label-savings declaration; full-label control is matched in optimizer steps, not optimized to convergence.')
