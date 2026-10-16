#!/usr/bin/env python3
"""Audit ONLY current-extension output; read, never rerun, frozen baselines.
Usage: summarize.py CURRENT_ARTIFACT FROZEN_RESULTS HISTORICAL_SUMMARY OUT
This never imports sklearn, starts a database, or invokes a fitted estimator.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, statistics
from pathlib import Path
import numpy as np


def load(path): return json.loads(path.read_text())
def save(path, obj): path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
def csvfile(path, rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def independently_score(y, raw, meta):
    if meta['task']=='regression':
        pred=raw*meta['y_scale']+meta['y_mean']; diff=y-pred
        return {'rmse':float(np.linalg.norm(diff)/np.sqrt(len(y))),
                'mae':float(np.abs(diff).sum()/len(y)),
                'r2':float(1-np.dot(diff,diff)/np.dot(y-y.mean(),y-y.mean()))}
    classes=np.unique(y)
    matrix=np.array([[np.count_nonzero((y==a)&(raw==b)) for b in classes] for a in classes])
    tp=matrix.diagonal();support=matrix.sum(axis=1);predicted=matrix.sum(axis=0)
    f1=np.divide(2*tp,support+predicted,out=np.zeros(len(classes)),where=(support+predicted)>0)
    return {'accuracy':float(tp.sum()/len(y)), 'balanced_accuracy':float((tp/support).mean()),
            'macro_f1':float(f1.mean())}


def audit(root, frozen, historical, out):
    out.mkdir(parents=True,exist_ok=True)
    with historical.open() as f: old=list(csv.DictReader(f))
    allrows=[];flat=[];predmatches=[];environments=[]
    seen=set()
    for seed in (42,2024,7):
        base=root/f'seed-{seed}'; complete=load(base/'completion.json')
        assert complete['complete'] and complete['external_baselines_executed']==0
        rows=load(base/'results.json'); assert len(rows)==(42 if seed==42 else 40)
        environment=load(base/'environment.json');environments.append(environment)
        assert environment['source_revision']=='e3da873258b0942d73b0166cf836599ec4895aad'
        for r in rows:
            key=(r['dataset'],r['method'],r['variant'],seed);assert key not in seen;seen.add(key)
            assert r['seed']==seed
            case=frozen/f"{r['dataset']}-{seed}";m=load(case/'meta.json');z=np.load(case/'data.npz')
            assert hashlib.sha256(np.ascontiguousarray(z['X_train']).tobytes()).hexdigest()==m['train_X_sha256']
            assert not set(z['train_ids']).intersection(z['test_ids'])
            current=base/case.name;im=load(current/'input-meta.json')
            for field in ['train_ids_sha256','test_ids_sha256','train_X_sha256','y_mean','y_scale']:
                assert m[field]==im[field]
            a=np.loadtxt(current/f"{r['method']}-{r['variant']}-predictions.csv",delimiter=',',ndmin=2)
            assert np.array_equal(a[:,0],np.arange(m['n_test'])) and np.isfinite(a).all()
            computed=independently_score(z['y_test'],a[:,1],m)
            for metric,value in computed.items():
                assert np.isclose(value,r['quality'][metric],atol=1e-12,rtol=1e-12),(key,metric,value,r['quality'])
            assert len(r['fit_ms'])==3 and len(r['predict_ms'])==3
            assert all(v>0 for v in r['fit_ms']+r['predict_ms'])
            assert r['fit_median_ms']==statistics.median(r['fit_ms'])
            assert r['metrics']['iterations']==200
            if r['variant']=='batch':
                previous=np.loadtxt(case/f"pg_{r['method']}-predictions.csv",delimiter=',',ndmin=2)
                assert np.array_equal(a[:,0],previous[:,0])
                delta=float(np.max(np.abs(a[:,1]-previous[:,1])))
                assert delta==r['frozen_old_pg_prediction_max_abs_diff']
                predmatches.append({'dataset':r['dataset'],'method':r['method'],'seed':seed,'max_abs_diff':delta})
            entry={k:r[k] for k in ('dataset','method','variant','seed','n_train','n_test','fit_median_ms','predict_median_ms')}
            entry.update(r['quality']);flat.append(entry);allrows.append(r)
    grouped=[];threshold=[]
    for dataset,method,variant in sorted(set((r['dataset'],r['method'],r['variant']) for r in allrows)):
        rr=[r for r in allrows if (r['dataset'],r['method'],r['variant'])==(dataset,method,variant)]
        agg={'dataset':dataset,'method':method,'variant':variant,'splits':len(rr),
             'fit_median_ms':statistics.median(r['fit_median_ms'] for r in rr),
             'predict_median_ms':statistics.median(r['predict_median_ms'] for r in rr)}
        for metric in rr[0]['quality']:
            values=[r['quality'][metric] for r in rr]
            agg[metric+'_mean']=statistics.mean(values)
            agg[metric+'_sd']=statistics.stdev(values) if len(values)>1 else None
        grouped.append(agg)
        if variant=='batch':
            previous={r['engine']:float(r['fit_ms_median']) for r in old if r['dataset']==dataset and r['method']==method and r['engine'] in ('pg','og')}
            threshold.append({'dataset':dataset,'method':method,'old_pg_ms':previous['pg'],
                              'frozen_og_ms':previous['og'],'current_pg_ms':agg['fit_median_ms'],
                              'historical_og_over_current':previous['og']/agg['fit_median_ms'],
                              'timing_comparability':'historical threshold, not same-host causal comparison'})
    ablations=[]
    for dataset in ('breast_cancer','banknote'):
        chosen={r['variant']:r for r in allrows if r['dataset']==dataset and r['method']=='svm_gaussian' and r['seed']==42}
        cache=chosen['batch'];uncached=chosen['uncached'];assert cache['model_md5']==uncached['model_md5']
        ablations.append({'dataset':dataset,'cached_ms':cache['fit_median_ms'],'uncached_ms':uncached['fit_median_ms'],
                          'uncached_over_cached':uncached['fit_median_ms']/cache['fit_median_ms'],
                          'model_md5':cache['model_md5'],
                          'cached_transform_rows':cache['metrics']['kernel_transformed_rows'],
                          'uncached_transform_rows':uncached['metrics']['kernel_transformed_rows']})
    stress=load(root/'ordered-stress/results.json');assert len(stress)==9
    meta=load(frozen/'banknote-42/meta.json');z=np.load(frozen/'banknote-42/data.npz')
    stressflat=[]
    for r in stress:
        a=np.loadtxt(root/'ordered-stress'/f"stress_{r['mode']}_{r['epoch_budget']}-predictions.csv",delimiter=',',ndmin=2)
        assert np.array_equal(a[:,0],np.arange(meta['n_test']))
        scores=independently_score(z['y_test'],a[:,1],meta)
        for metric,val in scores.items():assert np.isclose(val,r['quality'][metric],atol=1e-12,rtol=1e-12)
        assert r['metrics']['iterations']==r['epoch_budget']
        stressflat.append({'mode':r['mode'],'epochs':r['epoch_budget'],'fit_median_ms':r['median_fit_ms'],
                           'training_loss':r['metrics']['training_loss'],**scores})
    limits=load(root/'container.json')[0]['HostConfig']
    assert limits['NanoCpus']==10**9 and limits['Memory']==4*1024**3 and limits['NetworkMode']=='none'
    corgiratios=[]
    for r in allrows:
        if r['variant']=='corgipile':
            batch=next(s for s in allrows if (s['dataset'],s['method'],s['seed'],s['variant'])==(r['dataset'],r['method'],r['seed'],'batch'))
            corgiratios.append(r['fit_median_ms']/batch['fit_median_ms'])
    validation={'complete':True,'current_cases':len(allrows),'current_stress_cases':len(stress),
                'metric_checks':len(allrows)+len(stress),'frozen_prediction_checks':len(predmatches),
                'frozen_prediction_exact_matches':sum(r['max_abs_diff']==0 for r in predmatches),
                'feature_cache_model_identity_checks':len(ablations),'external_baselines_executed':0,
                'historical_thresholds_met':sum(r['current_pg_ms']<r['frozen_og_ms'] for r in threshold),
                'historical_threshold_cases':len(threshold),
                'current_corgipile_over_batch_median_ratio':statistics.median(corgiratios),
                'limitations':['historical timing comparisons are not same-machine ablations',
                               'stress epoch budgets are independent fits, not continuation checkpoints',
                               'no async producer-consumer or physical disk I/O throughput claim']}
    csvfile(out/'per-split.csv',flat);csvfile(out/'summary.csv',grouped);csvfile(out/'historical-thresholds.csv',threshold)
    csvfile(out/'stress.csv',sorted(stressflat,key=lambda r:(r['epochs'],r['mode'])))
    save(out/'validation.json',validation);save(out/'kernel-ablation.json',ablations)
    save(out/'prediction-parity.json',predmatches);save(out/'environments.json',environments)
    save(out/'raw-current-results.json',allrows);save(out/'raw-stress-results.json',stress)
    print(json.dumps(validation,indent=2));return validation

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('current',type=Path);p.add_argument('frozen',type=Path)
    p.add_argument('historical_summary',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();audit(a.current,a.frozen,a.historical_summary,a.out)
