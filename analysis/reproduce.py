#!/usr/bin/env python3
"""Reconstruct published aggregate results from public per-seed records; no GPU."""
import argparse,csv,hashlib,itertools,json
from pathlib import Path
import numpy as np
from scipy import stats
ROOT=Path(__file__).resolve().parents[1]
def require(ok,message):
    if not ok:raise ValueError(message)
def write_json(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def holm(values):
    order = np.argsort(values); result = np.empty(len(values)); previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,min(1.,(len(values)-rank)*values[index]))
        result[index]=previous
    return result.tolist()

def paired(rows, target, reference, metric):
    a={r['seed']:r[metric] for r in rows if r['method']==target}
    b={r['seed']:r[metric] for r in rows if r['method']==reference}
    require(set(a)==set(b) and len(a)==6, f'Seed pairing failed: {metric} {target} {reference}')
    seeds=sorted(a); d=np.array([a[s]-b[s] for s in seeds], dtype=float)
    require(np.isfinite(d).all(),'Nonfinite differences')
    sd=float(d.std(ddof=1)); mean=float(d.mean())
    half=float(stats.t.ppf(.975,5)*sd/np.sqrt(6))
    p=1. if np.all(d==0) else (0. if sd==0 else float(stats.ttest_1samp(d,0).pvalue))
    exact=float(np.mean([abs(np.mean(d*signs))>=abs(mean)-1e-15 for signs in itertools.product([-1,1],repeat=6)]))
    return dict(metric=metric,target=target,reference=reference,n_pairs=6,seeds=seeds,
                mean_difference=mean,sd_difference=sd,ci95_low=mean-half,ci95_high=mean+half,
                paired_t_p=p,exact_signflip_p=exact,target_higher_count=int((d>0).sum()),
                target_lower_count=int((d<0).sum()),differences=d.tolist())

def compare_number(a,b,context):
    # Some historical summaries encode undefined all-zero t statistics as null.
    if b is None:
        return
    require(np.isclose(float(a),float(b),rtol=1e-8,atol=2e-12),f'Recomputed statistic differs: {context}: {a} vs {b}')

def family(name, rows, old, aliases=None, metric_alias=None):
    aliases=aliases or {}; metric_alias=metric_alias or {}
    output=[]
    for rec in old:
        metric=rec.get('metric',rec.get('endpoint'))
        calc=paired(rows,rec['target'],rec['reference'],metric_alias.get(metric,metric))
        calc['metric']=metric
        output.append(calc)
    for calc, pt, pe, rec in zip(output,holm([r['paired_t_p'] for r in output]),holm([r['exact_signflip_p'] for r in output]),old):
        calc.update(family=name,family_size=len(old),paired_t_p_holm=pt,exact_signflip_p_holm=pe)
        for key in ['n_pairs','mean_difference','sd_difference','ci95_low','ci95_high','paired_t_p','exact_signflip_p','paired_t_p_holm','exact_signflip_p_holm','target_higher_count','target_lower_count']:
            source=aliases.get(key,key)
            if source in rec: compare_number(calc[key],rec[source],f'{name}/{key}')
        if 'individual_95pct_paired_t_interval' in rec:
            for k,v in zip(['ci95_low','ci95_high'],rec['individual_95pct_paired_t_interval']):compare_number(calc[k],v,name+k)
    return output

def csv_write(path, rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve()
    require(not out.exists(),'Use a new output directory')
    data=ROOT/'results/2026-09-26'
    checks=json.loads((data/'SHA256.json').read_text())
    for name,h in checks.items():require(hashlib.sha256((data/name).read_bytes()).hexdigest()==h,'Input hash differs: '+name)
    rows={name:json.loads((data/(name+'_per_seed.json')).read_text()) for name in ['classification','c4','diffusion','audio']}
    contrasts=[]
    for f in json.loads((data/'reported_contrasts.json').read_text()):
        contrasts+=family(f['name'],rows[f['dataset']],f['contrasts'],f['aliases'],f['metric_alias'])
    require(len(contrasts)==39,'Expected 39 comparisons')
    grouped=[]
    for dataset,records in rows.items():
        for method in sorted({r['method'] for r in records}):
            group=[r for r in records if r['method']==method]
            require(len(group)==6 and len({r['seed'] for r in group})==6,'Expected six unique seeds')
            for metric in group[0]:
                if metric in ['seed','method']:continue
                values=[r.get(metric) for r in group]
                if all(isinstance(v,(int,float)) for v in values):
                    a=np.array(values,dtype=float)
                    grouped.append(dict(dataset=dataset,method=method,metric=metric,n=6,mean=float(a.mean()),sample_sd=float(a.std(ddof=1))))
    with (data/'accuracy_by_budget.csv').open() as f:curves=list(csv.DictReader(f))
    with (data/'layer_geometry_by_budget.csv').open() as f:geometry=list(csv.DictReader(f))
    require(len(curves)==108 and len(geometry)==216,'Curve/geometry row count differs')
    require(len({(r['method'],r['seed'],r['phase']) for r in curves})==108,'Duplicate curve rows')
    for r in curves:
        require(int(r['annotations'])==12669*(int(r['phase'])+1),'Annotation count differs')
        if int(r['phase'])==5:
            final=next(q for q in rows['classification'] if q['method']==r['method'] and q['seed']==int(r['seed']))
            compare_number(r['accuracy'],final['clean_accuracy'],'Final curve endpoint')
    for r in geometry:compare_number(float(r['log_kappa']),np.log(float(r['within'])+1e-6)-np.log(float(r['between'])+1e-6),'Geometry identity')
    out.mkdir(parents=True)
    csv_write(out/'paired_comparisons.csv',contrasts);csv_write(out/'grouped_metrics.csv',grouped)
    write_json(out/'paired_comparisons.json',contrasts)
    status=dict(status='PASS',paired_contrasts=39,per_seed_rows={k:len(v) for k,v in rows.items()},curve_rows=108,geometry_rows=216,
                scope='Statistical reconstruction from exported numerical records; no training, attack rerun, or regeneration of raw-data metrics.')
    write_json(out/'VERIFICATION.json',status);print(json.dumps(status,indent=2))
if __name__=='__main__':main()
