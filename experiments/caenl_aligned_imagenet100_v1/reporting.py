"""Descriptive interim reporting; independent-seed inference only on the frozen full matrix."""
from __future__ import annotations
import itertools,math
from pathlib import Path
import numpy as np
from scipy import stats
from support import atom_json,write_csv,read_json


def paired_stats(a,b):
    d=np.asarray(a,dtype=float)-np.asarray(b,dtype=float);n=len(d);mu=float(d.mean());sd=float(d.std(ddof=1))
    if n<3:raise ValueError('Inferential report requires >=3 paired independent seeds')
    if not np.isfinite(d).all():raise ValueError('Nonfinite metric difference')
    if sd<1e-15:
        rawp=1. if abs(mu)<1e-15 else None;lo=hi=(0. if abs(mu)<1e-15 else None)
    else:
        se=sd/math.sqrt(n);q=float(stats.t.ppf(.975,n-1));lo=mu-q*se;hi=mu+q*se;rawp=float(2*stats.t.sf(abs(mu/se),n-1))
    observed=abs(mu);null=np.asarray([abs(np.mean(d*np.array(s))) for s in itertools.product((-1.,1.),repeat=n)])
    return dict(n_pairs=n,mean_difference=mu,ci95_low=lo,ci95_high=hi,paired_t_p=rawp,exact_signflip_p=float(np.mean(null>=observed-1e-15)),target_higher_count=int((d>0).sum()),target_lower_count=int((d<0).sum()),zero_variance=sd<1e-15)

def holm(p):
    missing=[x is None for x in p];p=np.asarray([1. if x is None else x for x in p],dtype=float);order=np.argsort(p);v=np.maximum.accumulate(np.minimum(1.,p[order]*(len(p)-np.arange(len(p)))))
    out=np.empty_like(v);out[order]=v;return [None if m else float(x) for m,x in zip(missing,out)]

def make_report(root,cfg,complete=False):
    root=Path(root);out=root/'reports';out.mkdir(exist_ok=True);rows=[];curves=[];all_summaries={}
    for seed in cfg['seeds']:
        for method in cfg['methods']:
            p=root/f'seed{seed}'/method['name']/'summary.json'
            if not p.exists():continue
            s=read_json(p);all_summaries[(seed,method['name'])]=s;v=s['rounds'][-1]['validation'];aa=s['autoattack']
            rows.append({'seed':seed,'method':method['name'],'clean_accuracy':v['accuracy'],'robust_accuracy':aa['robust_accuracy'],'cross_entropy':v['ce'],'ece':v['ece'],'top5_accuracy':v['top5_accuracy'],'mean_log_kappa':v['mean_log_kappa'],'train_compute_seconds_including_shared_init':sum(r['train_compute_seconds'] for r in s['rounds']),'feedback_seconds_including_shared_init':sum(r['feedback_seconds'] for r in s['rounds']),'acquisition_seconds':sum(r['elapsed_seconds'] for r in s['acquisitions']),'autoattack_seconds':aa['elapsed_s'],'max_allocated_bytes':max(r['peak_allocated_bytes'] for r in s['rounds']),'learned_policy_updates':s['rounds'][-1]['learned_updates_cumulative']})
            for r in s['rounds']:
                curves.append({'seed':seed,'method':method['name'],'round':r['phase'],'annotation_count':r['annotation_count'],'label_fraction':r['annotation_count']/cfg['expected_train_count'],'accuracy':r['validation']['accuracy'],'ce':r['validation']['ce'],'mean_log_kappa':r['validation']['mean_log_kappa']})
    write_csv(out/'per_seed_metrics.csv',rows);write_csv(out/'label_budget_curves.csv',curves)
    if complete and len(rows)!=len(cfg['seeds'])*len(cfg['methods']):raise ValueError('Incomplete scientific matrix')
    grouped=[]
    for method in cfg['methods']:
        vals=[r for r in rows if r['method']==method['name']]
        if not vals:continue
        g={'method':method['name'],'n_seeds':len(vals)}
        for k in rows[0]:
            if k in ('seed','method'):continue
            v=[r[k] for r in vals];g[k+'_mean']=float(np.mean(v));g[k+'_std']=float(np.std(v,ddof=1)) if len(v)>1 else None
        grouped.append(g)
    write_csv(out/'grouped_metrics.csv',grouped)
    if complete:
        pairs=[]
        for metric in cfg['statistics']['primary_endpoints']:
            current=[]
            for target,reference in cfg['statistics']['comparisons']:
                a=[next(r[metric] for r in rows if r['method']==target and r['seed']==seed) for seed in cfg['seeds']]
                b=[next(r[metric] for r in rows if r['method']==reference and r['seed']==seed) for seed in cfg['seeds']]
                current.append(dict(metric=metric,target=target,reference=reference,**paired_stats(a,b)))
            ht=holm([r['paired_t_p'] for r in current]);he=holm([r['exact_signflip_p'] for r in current])
            for r,t,e in zip(current,ht,he):r['paired_t_p_holm']=t;r['exact_signflip_p_holm']=e
            pairs+=current
        write_csv(out/'paired_comparisons.csv',pairs)
    lines=['# ImageNet-100 aligned independent-seed active learning','',f"Status: {'COMPLETE' if complete else 'PARTIAL — no final inference'}. {len(rows)}/{len(cfg['seeds'])*len(cfg['methods'])} method runs.",
       'Original 8/255 AutoAttack endpoint on all 5,000 validation images. These are empirical results, not certificates.',
       'Full MACC controls regularization only; query thresholds remain fixed. All methods share a 40-epoch initial CE checkpoint per seed and frozen BatchNorm running statistics thereafter.',
       'Noise Stability is an independently implemented output-deviation/k-center baseline, not an authors-repository reproduction. BADGE uses exact factorized last-weight gradients excluding bias.',
       'Six seeds, not validation images, are the independent units. No method is required to win to complete. All task and controller failures remain visible.','',
       '| Method | n seeds | Clean accuracy (%) | AA robust accuracy (%) | CE |','|---|---:|---:|---:|---:|']
    for g in grouped:lines.append(f"| {g['method']} | {g['n_seeds']} | {100*g['clean_accuracy_mean']:.3f} | {100*g['robust_accuracy_mean']:.3f} | {g['cross_entropy_mean']:.6f} |")
    lines+=['','Geometry is measured on all represented classes; it is not compared directly to 16-class training-anchor targets.',
       'Training compute includes the shared initial compute once per conceptual method run. It is relative to monitored baselines, not a plain uninstrumented network.',
       'This experiment is ImageNet-100 only; it does not establish ImageNet-1K or cross-modal performance.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    if complete:
        tex=['\\begin{tabular}{lrrr}','\\hline','Method & Clean top-1 (\\%) & AutoAttack (\\%) & CE \\\\','\\hline']
        for g in grouped:
            name=g['method'].replace('_',r'\_')
            tex.append(f"{name} & {100*g['clean_accuracy_mean']:.3f} $\\pm$ {100*g['clean_accuracy_std']:.3f} & {100*g['robust_accuracy_mean']:.3f} $\\pm$ {100*g['robust_accuracy_std']:.3f} & {g['cross_entropy_mean']:.6f} $\\pm$ {g['cross_entropy_std']:.6f} \\\\")
        tex+=['\\hline','\\end{tabular}'];(out/'results_table.tex').write_text('\n'.join(tex)+'\n')
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9,6))
        for method in cfg['methods']:
            sets=[r for r in curves if r['method']==method['name']];xs=sorted({r['round'] for r in sets});ys=[];sd=[]
            for x in xs:
                y=[100*r['accuracy'] for r in sets if r['round']==x];ys.append(np.mean(y));sd.append(np.std(y,ddof=1))
            ax.errorbar([10*(x+1) for x in xs],ys,yerr=sd,marker='o',capsize=2,label=method['name'])
        ax.set(xlabel='Annotation budget (%)',ylabel='Clean top-1 accuracy (%)',title='ImageNet-100: mean and seed SD');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=180);fig.savefig(out/'learning_curves.svg');plt.close(fig)
    return rows
