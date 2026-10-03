#!/usr/bin/env python3
"""Reconstruct the frozen six-contrast feedback study from released seed records.

Requires NumPy and SciPy. No GPU, licensed images or model weights are needed.
This reconstructs record arithmetic; it does not repeat training or inference.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy import stats


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def holm(values):
    order=np.argsort(values,kind='stable'); result=np.empty(len(values)); running=0.0
    for rank,index in enumerate(order):
        running=max(running,min(1.,(len(values)-rank)*values[index]));result[index]=running
    return result


def family(values):
    rows=[]
    for stage,start in [('confirmation',42001),('transfer',43001)]:
        for control in ['entropy','tuned_fixed','tuned_schedule']:
            d=np.array([100*(values[stage,seed,'lite']-values[stage,seed,control]) for seed in range(start,start+10)])
            mean=float(d.mean());sd=float(d.std(ddof=1));margin=float(stats.t.ppf(.975,9)*sd/np.sqrt(10))
            signs=np.asarray(list(itertools.product([-1.,1.],repeat=10)))
            exact=float(np.mean(np.abs(signs@d/10)>=abs(mean)-1e-12))
            rows.append(dict(stage=stage,reference=control,difference_pp=mean,sd_difference_pp=sd,ci95_low_pp=mean-margin,ci95_high_pp=mean+margin,paired_t_p=float(stats.ttest_1samp(d,0).pvalue),exact_signflip_p=exact))
    for key,new in [('paired_t_p','holm_t_p'),('exact_signflip_p','holm_exact_p')]:
        for row,p in zip(rows,holm([r[key] for r in rows])):row[new]=float(p)
    return rows


def check_reference(rows,path):
    with path.open() as stream: refs=list(csv.DictReader(stream))
    assert len(refs)==len(rows)==6
    aliases={'low95':'ci95_low_pp','high95':'ci95_high_pp','t_p':'paired_t_p','exact_p':'exact_signflip_p'}
    refs=[{aliases.get(k,k):v for k,v in r.items()} for r in refs]
    assert all(all(k in r for k in ['difference_pp','ci95_low_pp','ci95_high_pp','paired_t_p','exact_signflip_p','holm_t_p','holm_exact_p']) for r in refs)
    max_error=0.
    for r,ref in zip(rows,refs):
        assert (r['stage'],r['reference'])==(ref['stage'],ref['reference'])
        for k,v in r.items():
            if k not in ('stage','reference') and k in ref:
                error=abs(v-float(ref[k]));max_error=max(error,max_error)
                if error>1e-9:raise ValueError(f'Mismatch {r["stage"]}/{r["reference"]}/{k}: {error}')
    return max_error


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',type=Path,default=Path(__file__).resolve().parents[2]/'results/2026-10-02-feedback')
    parser.add_argument('--output',type=Path,required=True,help='New output directory; existing paths are never overwritten.')
    args=parser.parse_args();root=args.records.resolve()
    if args.output.exists():raise SystemExit('Choose a new output directory.')
    manifest=json.loads((root/'SHA256SUMS.json').read_text())
    for name,expected in manifest.items():
        if Path(name).name!=name:raise ValueError('Unsafe manifest path')
        actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
        if actual!=expected:raise ValueError('Input checksum mismatch: '+name)
    records=json.loads((root/'reconstructed_evaluations.json').read_text())
    assert len(records)==104
    assert len({(r['stage'],r['seed'],r['method']) for r in records})==104
    values={(r['stage'],r['seed'],r['method']):r['validation']['accuracy'] for r in records if r['validation'] is not None}
    assert len(values)==80
    primary=family(values)
    with (root/'duplicate_exclusion_per_seed.csv').open() as stream: duplicate=list(csv.DictReader(stream))
    assert len(duplicate)==80
    clean={(r['stage'],int(r['seed']),r['method']):float(r['deduplicated_accuracy']) for r in duplicate}
    sensitivity=family(clean)
    errors={'primary':check_reference(primary,root/'primary_contrasts.csv'),'sensitivity':check_reference(sensitivity,root/'sensitivity_contrasts.csv')}
    groups=[]
    for stage in ['confirmation','transfer']:
        for method in ['entropy','lite','tuned_fixed','tuned_schedule']:
            selected=[r['validation'] for r in records if r['stage']==stage and r['method']==method]
            assert len(selected)==10
            row={'stage':stage,'method':method,'n_seeds':len(selected)}
            for metric,scale in [('accuracy',100),('ce',1),('ece',1),('mean_log_kappa',1)]:
                vals=np.asarray([r[metric]*scale for r in selected]);row[metric+'_mean']=float(vals.mean());row[metric+'_sd']=float(vals.std(ddof=1))
            groups.append(row)
    ranking=[]
    for method in sorted({r['method'] for r in records if r['stage']=='tuning'}):
        selected=[r['holdout'] for r in records if r['stage']=='tuning' and r['method']==method];assert len(selected)==3
        ranking.append({'method':method,'holdout_accuracy_mean':float(np.mean([r['accuracy'] for r in selected])),'holdout_ce_mean':float(np.mean([r['ce'] for r in selected]))})
    ranking.sort(key=lambda r:(-r['holdout_accuracy_mean'],r['holdout_ce_mean'],r['method']))
    fixed=next(r['method'] for r in ranking if r['method'].startswith('fixed_'))
    schedule=next(r['method'] for r in ranking if r['method'].startswith('linear_'))
    assert fixed=='fixed_0p100' and schedule=='linear_down_0p10'
    unchanged=all((a[k]<.05)==(b[k]<.05) for a,b in zip(primary,sensitivity) for k in ['holm_t_p','holm_exact_p'])
    args.output.mkdir(parents=True)
    for name,rows in [('primary_contrasts.csv',primary),('sensitivity_contrasts.csv',sensitivity),('group_outcomes.csv',groups),('tuning_ranking.csv',ranking)]:write_csv(args.output/name,rows)
    verification={'status':'PASS','verified_input_files':len(manifest),'completed_records':104,'official_validation_records':80,'selected_fixed':fixed,'selected_schedule':schedule,'primary_contrasts':6,'exact_sign_patterns_per_contrast':1024,'family_definition':'Lite minus entropy, selected fixed and selected schedule across both backbones; separate Holm adjustment for each test type.','interval_definition':'Individual 95% paired-t intervals; not simultaneous.','sensitivity_decisions_unchanged':unchanged,'maximum_reference_discrepancy':errors,'scope':'Reconstruction of released numerical records; no training or checkpoint inference.'}
    (args.output/'VERIFICATION.json').write_text(json.dumps(verification,indent=2)+'\n');print(json.dumps(verification,indent=2))

if __name__=='__main__':main()
