"""Frozen candidate grid, stage definitions and predetermined coefficient schedules."""
from __future__ import annotations
import copy, math


def scheduled_strength(shape, peak, progress):
    if shape not in ('linear_up', 'linear_down'):
        raise ValueError('Unknown open-loop schedule')
    if not math.isfinite(peak) or not 0 < peak <= .1:
        raise ValueError('Invalid schedule peak')
    if not math.isfinite(progress) or not 0 <= progress <= 1:
        raise ValueError('Invalid schedule progress')
    return float(peak * (progress if shape == 'linear_up' else 1-progress))


def candidates(cfg):
    result=[]
    for v in cfg['study']['fixed_coefficients']:
        result.append(dict(name='fixed_'+format(v,'.3f').replace('.','p'), family='fixed',
                           acquisition='entropy', regularizer='aligned_fixed', coefficient=v))
    for shape in cfg['study']['schedule_shapes']:
        for peak in cfg['study']['schedule_peaks']:
            result.append(dict(name=shape+'_'+format(peak,'.2f').replace('.','p'), family='schedule',
                               acquisition='entropy', regularizer='aligned_schedule', shape=shape, peak=peak))
    return result


def evaluated_methods(selection):
    return [dict(name='entropy', acquisition='entropy', regularizer='none'),
            {**copy.deepcopy(selection['fixed']), 'name':'tuned_fixed'},
            {**copy.deepcopy(selection['schedule']), 'name':'tuned_schedule'},
            dict(name='lite', acquisition='entropy', regularizer='aligned_lite')]


def stage_specs(cfg):
    s=cfg['study']
    return [('tuning','resnet50',s['tuning_seeds']),
            ('confirmation','resnet50',s['confirmation_seeds']),
            ('transfer','resnet18',s['transfer_seeds'])]


def method_config(cfg, method, phase, stage, backbone):
    from training import phase_steps
    c=copy.deepcopy(cfg)
    c.update(study_stage=stage, backbone=backbone, official_reporting=stage!='tuning' and phase>0)
    if method['regularizer']=='aligned_fixed':c['lambda_fixed']=method['coefficient']
    if method['regularizer']=='aligned_schedule':
        n=c['initial_labels']-c['holdout_count']
        lengths=[phase_steps(n+r*c['acquisition_per_round'],r,c)[0] for r in range(1,c['rounds']+1)]
        c.update(schedule_shape=method['shape'],schedule_peak=method['peak'],
                 schedule_offset=sum(lengths[:max(0,phase-1)]),schedule_total_steps=sum(lengths))
    return c


def choose_controls(rows, cfg):
    """Select solely from COMPLETE tuning rows; never consume official outcomes."""
    expected={(seed,c['name']) for seed in cfg['study']['tuning_seeds'] for c in candidates(cfg)}
    if len(rows)!=len(expected) or {(r['seed'],r['method']) for r in rows}!=expected:
        raise ValueError('All planned tuning runs are required before selection')
    if any(r['stage']!='tuning' or r.get('official_validation') is not None for r in rows):
        raise ValueError('Selection must not receive official validation outcomes')
    ranking=[]
    for c in candidates(cfg):
        rr=[r for r in rows if r['method']==c['name']]
        acc=sum(float(r['holdout']['accuracy']) for r in rr)/len(rr)
        ce=sum(float(r['holdout']['ce']) for r in rr)/len(rr)
        if not math.isfinite(acc+ce):raise ValueError('Nonfinite tuning outcome')
        ranking.append(dict(candidate=c,mean_accuracy=acc,mean_ce=ce,n_seeds=len(rr)))
    selected={}
    for family in ('fixed','schedule'):
        rr=[r for r in ranking if r['candidate']['family']==family]
        best=max(r['mean_accuracy'] for r in rr)
        tied=[r for r in rr if best-r['mean_accuracy']<=1e-12]
        selected[family]=min(tied,key=lambda r:(r['mean_ce'],r['candidate']['name']))['candidate']
    return dict(**selected, ranking=ranking,
                criterion='Mean final budgeted training-holdout top-1; CE then candidate ID for ties',
                official_validation_used=False)


def planned_counts(cfg):
    from training import phase_steps
    s=cfg['study'];nt=len(s['tuning_seeds'])*len(candidates(cfg))
    nc=4*len(s['confirmation_seeds']);nx=4*len(s['transfer_seeds'])
    ni=sum(len(seeds) for _,_,seeds in stage_specs(cfg))
    n0=cfg['initial_labels']-cfg['holdout_count']
    per_run=sum(phase_steps(n0+r*cfg['acquisition_per_round'],r,cfg)[0] for r in range(1,cfg['rounds']+1))
    return dict(tuning_runs=nt,confirmation_runs=nc,transfer_runs=nx,method_runs=nt+nc+nx,
                initializations=ni,postinitial_phases=(nt+nc+nx)*cfg['rounds'],
                optimizer_steps=(nt+nc+nx)*per_run+ni*phase_steps(n0,0,cfg)[0])
