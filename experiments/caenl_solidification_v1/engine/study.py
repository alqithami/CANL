"""Frozen 206-run design; selections use only three training holdout seeds."""
import copy,math

def scheduled_strength(shape,peak,progress):
    if shape not in ('linear_up','linear_down') or not 0<peak<=.1 or not 0<=progress<=1:raise ValueError('Invalid schedule')
    return peak*(progress if shape=='linear_up' else 1-progress)

def candidates(cfg):
    s=cfg['study'];m=[]
    for v in s['fixed_coefficients']:m.append(dict(name='fixed_'+str(v),family='fixed',regularizer='aligned_fixed',coefficient=v,acquisition='entropy'))
    for shape in s['schedule_shapes']:
        for peak in s['schedule_peaks']:m.append(dict(name=shape+'_'+str(peak),family='schedule',regularizer='aligned_schedule',shape=shape,peak=peak,acquisition='entropy'))
    for v in s['supcon_coefficients']:m.append(dict(name='supcon_'+str(v),family='supcon',regularizer='supcon',coefficient=v,acquisition='entropy'))
    return m

def evaluated_methods(selection):
    return [dict(name='entropy',regularizer='none',acquisition='entropy'),
      dict(**{**selection['fixed'],'name':'tuned_fixed'}),dict(**{**selection['schedule'],'name':'tuned_schedule'}),
      dict(name='lite',regularizer='aligned_lite',acquisition='entropy'),
      dict(**{**selection['supcon'],'name':'tuned_supcon'}),dict(name='full_labels',regularizer='none',acquisition='full')]

def replay_methods(selection):
    fixed=selection['fixed'];sched=selection['schedule']
    return [dict(**{**fixed,'name':'replay_ratio','floor_weight':0.,'acquisition':'replay'}),
      dict(name='replay_separation',regularizer='aligned_fixed',coefficient=0.,acquisition='replay'),
      dict(**{**fixed,'name':'replay_fixed','acquisition':'replay'}),
      dict(name='replay_lite',regularizer='aligned_lite',acquisition='replay'),
      dict(**{**sched,'name':'replay_schedule','acquisition':'replay'})]

def stage_specs(cfg):
    return [('tuning','resnet50',cfg['study']['tuning_seeds']),('confirmation','resnet50',cfg['study']['confirmation_seeds']),('transfer','resnet18',cfg['study']['transfer_seeds'])]

def method_config(cfg,method,phase,stage,backbone):
    from training import phase_steps
    c=copy.deepcopy(cfg);c.update(backbone=backbone,study_stage=stage,official_reporting=stage!='tuning' and phase>0)
    c['alignment']['floor_weight']=method.get('floor_weight',cfg['alignment']['floor_weight'])
    if method['regularizer']=='aligned_fixed':c['lambda_fixed']=method['coefficient']
    if method['regularizer']=='supcon':c['supcon_coefficient']=method['coefficient'];c['lambda_fixed']=method['coefficient'];c['supcon_temperature']=.1
    if method['regularizer']=='aligned_schedule':
        n=c['initial_labels']-c['holdout_count'];lens=[phase_steps(n+r*c['acquisition_per_round'],r,c)[0] for r in range(1,c['rounds']+1)]
        c.update(schedule_shape=method['shape'],schedule_peak=method['peak'],schedule_offset=sum(lens[:max(0,phase-1)]),schedule_total_steps=sum(lens))
    if method['acquisition']=='full':c['step_budget_counts']={str(r):c['initial_labels']-c['holdout_count']+r*c['acquisition_per_round'] for r in range(1,c['rounds']+1)}
    return c

def choose_controls(rows,cfg):
    expected={(seed,m['name']) for seed in cfg['study']['tuning_seeds'] for m in candidates(cfg)}
    if len(rows)!=len(expected) or {(r['seed'],r['method']) for r in rows}!=expected:raise ValueError('Incomplete tuning grid')
    if any(r['stage']!='tuning' or r['official_validation'] is not None for r in rows):raise ValueError('Official validation cannot enter selection')
    ranked=[]
    for m in candidates(cfg):
        rr=[r for r in rows if r['method']==m['name']]
        a=sum(r['holdout']['accuracy'] for r in rr)/len(rr);ce=sum(r['holdout']['ce'] for r in rr)/len(rr)
        if not math.isfinite(a+ce):raise ValueError('Nonfinite selection metric')
        ranked.append(dict(candidate=m,mean_accuracy=a,mean_ce=ce))
    selected={}
    for family in ('fixed','schedule','supcon'):
        rr=[r for r in ranked if r['candidate']['family']==family];best=max(r['mean_accuracy'] for r in rr)
        selected[family]=min([r for r in rr if best-r['mean_accuracy']<=1e-12],key=lambda r:(r['mean_ce'],r['candidate']['name']))['candidate']
    return dict(**selected,ranking=ranked,official_validation_used=False)

def planned_counts(cfg):
    from training import phase_steps
    nt=len(candidates(cfg))*len(cfg['study']['tuning_seeds']);nc=11*len(cfg['study']['confirmation_seeds']);nx=6*len(cfg['study']['transfer_seeds'])
    ni=sum(len(seeds) for _,_,seeds in stage_specs(cfg));n=cfg['initial_labels']-cfg['holdout_count']
    steps=sum(phase_steps(n+r*cfg['acquisition_per_round'],r,cfg)[0] for r in range(1,cfg['rounds']+1))
    return dict(tuning=nt,confirmation=nc,transfer=nx,method_runs=nt+nc+nx,initializations=ni,postinitial_phases=(nt+nc+nx)*cfg['rounds'],optimizer_steps=(nt+nc+nx)*steps+ni*phase_steps(n,0,cfg)[0])
