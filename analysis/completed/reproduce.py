#!/usr/bin/env python3
"""CPU reconstruction of completed checkpoint diagnostics and raw-source audio.

Requires NumPy/SciPy. No training, inference, asset access, or model downloads.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path
from collections import Counter

import numpy as np
from scipy import stats

ROOT=Path(__file__).resolve().parents[2]

def check(x,y):
    if isinstance(x,(int,float)) and isinstance(y,(int,float)):
        assert abs(x-y)<1e-9,(x,y)
    else:
        assert x==y,(x,y)

def holm(values):
    order=np.argsort(values,kind='stable');out=np.empty(len(values));running=0.
    for rank,i in enumerate(order):
        running=max(running,min(1.,values[i]*(len(values)-rank)));out[i]=running
    return out

def paired(values):
    d=np.asarray(values,dtype=float);n=len(d);mu=float(d.mean());sd=float(d.std(ddof=1))
    h=float(stats.t.ppf(.975,n-1)*sd/math.sqrt(n))
    signed=np.abs(np.array(list(itertools.product([-1.,1.],repeat=n)))@d/n)
    return {'mean':mu,'sd':sd,'low':mu-h,'high':mu+h,
            't':float(stats.ttest_1samp(d,0).pvalue),
            'exact':float(np.mean(signed>=abs(mu)-1e-12))}

def load(slug):
    directory=ROOT/'results'/slug
    for line in (directory/'SHA256SUMS.txt').read_text().splitlines():
        digest,name=line.split('  ',1)
        check(hashlib.sha256((directory/name).read_bytes()).hexdigest(),digest)
    return json.loads((directory/'RESULTS.json').read_text())

def main():
    d=load('caenl-diagnostics-v1');r=d['robustness'];cost=d['cost']
    check(len(r['clean']),30);check(len(r['suites']),150)
    clean={(x['seed'],x['method']):x for x in r['clean']}
    suites={(x['seed'],x['method'],x['epsilon_pixels']):x for x in r['suites']}
    check(len(clean),30);check(len(suites),150)
    for key,row in clean.items():
        check(row['n'],4995);check(row['inference_replay']['status'],'PASS')
        check(row['inference_replay']['prediction_equality'],True)
        for radius in r['radii_pixels']:
            q=suites[key+(radius,)]
            check(q['n'],4995);check(q['robust_correct']/4995,q['robust_accuracy'])
            assert 0<=q['robust_correct']<=q['clean_correct']<=4995
            check(q['attacks_to_run'],['apgd-ce','apgd-t','fab-t','square'])
    calculated=[]
    for c in r['primary_contrasts']:
        values=[100*(suites[(seed,c['target'],2)]['robust_accuracy']-suites[(seed,c['reference'],2)]['robust_accuracy']) for seed in r['seeds']]
        z=paired(values);calculated.append(z)
        for k,k2 in [('mean','difference_pp'),('sd','sd_difference_pp'),('low','ci95_low_pp'),('high','ci95_high_pp'),('t','paired_t_p'),('exact','exact_signflip_p')]:check(z[k],c[k2])
    for k,k2 in [('t','paired_t_p_holm'),('exact','exact_signflip_p_holm')]:
        for value,c in zip(holm([z[k] for z in calculated]),r['primary_contrasts']):check(float(value),c[k2])
    intersection={(x['seed'],x['epsilon_pixels']):x for x in r['same_clean_correct_intersection']}
    check(len(intersection),50)
    for c in r['same_clean_correct_descriptive_contrasts']:
        values=[100*(intersection[(seed,c['epsilon_pixels'])]['survival'][c['target']]['accuracy']-intersection[(seed,c['epsilon_pixels'])]['survival'][c['reference']]['accuracy']) for seed in r['seeds']]
        z=paired(values)
        for k,k2 in [('mean','difference_pp'),('sd','sd_difference_pp'),('low','ci95_low_pp'),('high','ci95_high_pp')]:check(z[k],c[k2])
    check(len(cost['rows']),30);check(cost['original_inputs_unchanged'],True)
    for condition,summary in cost['summary']['conditions'].items():
        rows=[x for x in cost['rows'] if x['condition']==condition]
        check(len(rows),5);check(sorted(x['repetition'] for x in rows),list(range(5)))
        for metric,s in summary.items():
            values=np.array([x[metric] for x in rows]);check(float(values.mean()),s['mean']);check(float(values.std(ddof=1)),s['sd'])
    print('PASS diagnostics: 30 clean replays, 150 suites, 3 primary and 15 descriptive contrasts, 30 cost rows.')

    a=load('caenl-audiocaps-raw-v1');c=a['confirmation'];rows=c['per_seed']
    check(a['method_runs'],{'confirmation':40,'total':58,'tuning':18})
    check(len(rows),40);check(len({(x['seed'],x['method']) for x in rows}),40)
    by={(x['seed'],x['method']):x['final_metrics'] for x in rows}
    check(c['seeds'],list(range(82001,82011)))
    for method,g in c['grouped_metrics'].items():
        for metric,s in g.items():
            values=np.array([by[(seed,method)][metric] for seed in c['seeds']])
            assert np.isfinite(values).all()
            check(float(values.mean()),s['mean']);check(float(values.std(ddof=1)),s['sample_sd'])
    calculated=[]
    for p in c['primary_contrasts']:
        differences=[by[(seed,p['target'])]['cider_d']-by[(seed,p['reference'])]['cider_d'] for seed in c['seeds']]
        check(differences,p['paired_differences']);z=paired(differences);calculated.append(z)
        for k,k2 in [('mean','difference_mean'),('sd','difference_sample_sd'),('t','paired_t_p_raw'),('exact','exact_sign_flip_p_raw')]:check(z[k],p[k2])
        check(z['low'],p['confidence_interval_95_unadjusted'][0]);check(z['high'],p['confidence_interval_95_unadjusted'][1])
    for k,k2 in [('t','paired_t_p_holm'),('exact','exact_sign_flip_p_holm')]:
        for value,p in zip(holm([z[k] for z in calculated]),c['primary_contrasts']):check(float(value),p[k2])
    selection=a['fixed_selection'];tuning=selection['all_18_tuning_outcomes']
    check(len(tuning),18);check(len({(x['seed'],x['lambda']) for x in tuning}),18)
    means=[]
    for x in selection['candidate_means']:
        group=[q for q in tuning if q['lambda']==x['lambda']]
        check(sorted(q['seed'] for q in group),[81001,81002,81003])
        mean=float(np.mean([q['development_cider_d'] for q in group]));check(mean,x['mean_development_cider_d']);means.append((mean,-x['lambda']))
    check(-max(means)[1],selection['lambda']);check(selection['official_test_outcomes_used'],False)
    print('PASS AudioCaps: all 18 tuning and 40 confirmation endpoints, secondary means/SDs, all 5 primary effects and both separate Holm families.')
    print('Scope: saved-endpoint arithmetic only; no independent model training, adversarial generation or caption regeneration.')

if __name__=='__main__':main()
