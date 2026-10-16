#!/usr/bin/env python3
"""Aggregate measured results without treating timing repeats as independent folds."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import dump

QUALITY=['accuracy','balanced_accuracy','macro_f1','roc_auc','rmse','mae','r2']
TIMING=['fit_ms','serialize_ms','fit_persist_ms','load_ms','predict_ms']


def summarize(root: Path, out: Path):
    rows=json.loads((root/'results.json').read_text())
    df=pd.DataFrame(rows); out.mkdir(parents=True,exist_ok=True)
    failures=df[df.status!='ok'];failures.to_json(out/'failures.json',orient='records',indent=2)
    good=df[df.status=='ok'].copy();group=['dataset','engine','method','seed']
    folds=[]
    for key,frame in good.groupby(group,dropna=False):
        row=dict(zip(group,key));row['timing_repeats']=int(frame['repeat'].notna().sum()) if 'repeat' in frame else 0
        for m in QUALITY:
            if m in frame and frame[m].notna().any():
                values=frame[m].dropna().to_numpy()
                if not np.allclose(values,values[0],rtol=0,atol=1e-12):
                    raise AssertionError(f'{key}: timing repeats unexpectedly have different quality: {m}')
                row[m]=float(values[0])
        for m in TIMING:
            if m in frame and frame[m].notna().any(): row[m]=float(frame[m].median())
        folds.append(row)
    fold=pd.DataFrame(folds);fold.to_csv(out/'per-split.csv',index=False)
    aggregates=[]
    for key,frame in fold.groupby(['dataset','engine','method']):
        row=dict(zip(['dataset','engine','method'],key));row['n_splits']=len(frame)
        for m in QUALITY:
            if m in frame and frame[m].notna().any():
                row[m+'_mean']=float(frame[m].mean())
                row[m+'_std']=float(frame[m].std(ddof=1)) if len(frame)>1 else None
        for m in TIMING:
            if m in frame and frame[m].notna().any():
                row[m+'_median']=float(frame[m].median())
                row[m+'_split_min']=float(frame[m].min());row[m+'_split_max']=float(frame[m].max())
        aggregates.append(row)
    dump(out/'summary.json',aggregates);pd.DataFrame(aggregates).to_csv(out/'summary.csv',index=False)
    paired=[]
    for (dataset,method,seed),frame in fold[fold.engine.isin(['pg','og'])].groupby(['dataset','method','seed']):
        if set(frame.engine)!= {'pg','og'}: continue
        pg=frame[frame.engine=='pg'].iloc[0];og=frame[frame.engine=='og'].iloc[0]
        row=dict(dataset=dataset,method=method,seed=int(seed))
        for m in ['fit_ms','predict_ms']:
            row[m+'_og_over_pg']=float(og[m]/pg[m])
        for m in QUALITY:
            if m in pg and pd.notna(pg[m]) and pd.notna(og[m]):row[m+'_pg_minus_og']=float(pg[m]-og[m])
        paired.append(row)
    dump(out/'paired-pg-og.json',paired);pd.DataFrame(paired).to_csv(out/'paired-pg-og.csv',index=False)
    datasets=[]; checks=[]
    for case_name in json.loads((root/'datasets.json').read_text())['cases']:
        case=root/case_name;meta=json.loads((case/'meta.json').read_text());datasets.append(meta)
        z=np.load(case/'data.npz')
        if set(z['train_ids']).intersection(z['test_ids']): raise AssertionError('row leakage')
        if (case/'pg_xgboost-predictions.csv').exists() and (case/'python_xgboost-predictions.npy').exists():
            pg=np.loadtxt(case/'pg_xgboost-predictions.csv',delimiter=',',ndmin=2)[:,1]
            py=np.load(case/'python_xgboost-predictions.npy')
            compared=(pg>=.5).astype(int) if meta['task']=='binary' else pg
            entry=dict(dataset=meta['dataset'],seed=meta['seed'],
                       independent_training_max_abs_diff=float(np.max(np.abs(compared-py))),
                       independent_training_scope='class labels' if meta['task']=='binary' else 'scaled regression values',
                       sentinel_minus_one_train=int((z['X_train'].astype('f4')==-1).sum()),
                       sentinel_minus_one_test=int((z['X_test'].astype('f4')==-1).sum()))
            if (case/'xgboost_roundtrip.json').exists():
                entry['export_reload_raw_prediction_max_abs_diff']=json.loads((case/'xgboost_roundtrip.json').read_text())['max_abs_difference']
            checks.append(entry)
    dump(out/'dataset-manifest.json',datasets);dump(out/'xgboost-parity.json',checks)
    coverage=df.groupby(['engine','status']).size().reset_index(name='records').to_dict(orient='records')
    dump(out/'validation.json',dict(records=len(df),successful_fold_methods=len(fold),failures=len(failures),coverage=coverage,
                                   summary_rule='quality: mean/sample SD over splits once each; timing: median of within-split medians',
                                   xgboost_checked_cases=len(checks)))
    print('Summary written:',out,'failure cases:',len(failures))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();summarize(a.results,a.out)
