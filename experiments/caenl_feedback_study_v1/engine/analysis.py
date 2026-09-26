"""Paired-seed inference for the frozen six-contrast family; no partial-family tests."""
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
    expected=[]
    for stage,key in [('confirmation','confirmation_seeds'),('transfer','transfer_seeds')]:
        expected.extend((stage,seed,method) for seed in cfg['study'][key]
                        for method in ('entropy','tuned_fixed','tuned_schedule','lite'))
    evaluated=[r for r in rows if r['stage']!='tuning']
    data={(r['stage'],r['seed'],r['method']):r for r in evaluated}
    if len(data)!=len(evaluated):raise ValueError('Duplicate stage/seed/method record')
    if len(data)!=len(expected) or set(data)!=set(expected):
        return dict(status='INCOMPLETE',planned_family_size=6,contrasts=[],
                    reason='No adjusted tests until both complete backbone matrices are available.')
    out=[]
    for stage,key in [('confirmation','confirmation_seeds'),('transfer','transfer_seeds')]:
        seeds=cfg['study'][key]
        for target,ref in cfg['study']['primary_contrasts_per_backbone']:
            a=[100*data[(stage,x,target)]['official_validation']['accuracy'] for x in seeds]
            b=[100*data[(stage,x,ref)]['official_validation']['accuracy'] for x in seeds]
            out.append(dict(stage=stage,target=target,reference=ref,**paired(a,b)))
    t=holm([r['paired_t_p'] for r in out]);exact=holm([r['exact_signflip_p'] for r in out])
    for r,tp,ep in zip(out,t,exact):r.update(holm_t_p=tp,holm_exact_p=ep)
    return dict(status='COMPLETE',planned_family_size=6,contrasts=out,
                intervals='Individual unadjusted 95% paired-t intervals, not simultaneous intervals.',
                assumptions='Paired-t approximation and sign-exchangeability for exact sign flips; seeds are units.',
                scope='Conditional on the fixed dataset and frozen tuning procedure; no programme-wide error guarantee.')
