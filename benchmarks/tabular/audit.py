#!/usr/bin/env python3
"""Independent checks of exported predictions, metrics and Docker resource limits.
Does not call common.metrics or sklearn.metrics; checks raw exported outputs.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def audit(root: Path):
    rows=json.loads((root/'results.json').read_text())
    cases=json.loads((root/'datasets.json').read_text())['cases']
    errors=[r for r in rows if r['status']!='ok']; checked=0; findings=[]
    for case_name in cases:
        case=root/case_name; meta=json.loads((case/'meta.json').read_text()); z=np.load(case/'data.npz')
        train_ids=set(z['train_ids']);test_ids=set(z['test_ids'])
        assert not train_ids.intersection(test_ids)
        assert len(train_ids)==meta['n_train'] and len(test_ids)==meta['n_test']
        y=z['y_test']
        for row in rows:
            if row['status']!='ok' or row['engine'] not in ('pg','og') or row.get('repeat')!=0:
                continue
            if row['dataset']!=meta['dataset'] or row['seed']!=meta['seed']:continue
            a=np.loadtxt(case/(row['engine']+'_'+row['method']+'-predictions.csv'),delimiter=',',ndmin=2)
            assert np.array_equal(a[:,0],np.arange(len(y)))
            pred=a[:,1]
            assert np.isfinite(pred).all()
            if meta['task']=='regression':
                pred=pred*meta['y_scale']+meta['y_mean'];diff=y-pred
                expected={'rmse':float(np.sqrt(np.dot(diff,diff)/len(y))),
                          'mae':float(np.abs(diff).mean()),
                          'r2':float(1-np.dot(diff,diff)/np.dot(y-y.mean(),y-y.mean()))}
            else:
                if row['method']=='xgboost':pred=(pred>=.5).astype(int)
                recalls=[];f1s=[]
                for c in np.unique(y):
                    tp=np.count_nonzero((pred==c)&(y==c));fp=np.count_nonzero((pred==c)&(y!=c));fn=np.count_nonzero((pred!=c)&(y==c))
                    recalls.append(tp/(tp+fn));f1s.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
                expected={'accuracy':float((pred==y).mean()),'balanced_accuracy':float(np.mean(recalls)),'macro_f1':float(np.mean(f1s))}
            for key,value in expected.items():
                if not np.isclose(row[key],value,rtol=1e-11,atol=1e-12):findings.append(f'{case_name} {row["engine"]} {row["method"]} {key} mismatch')
            checked+=1
    docker=[]
    for engine in ['pg','og']:
        item=json.loads((root/(engine+'-container.json')).read_text())[0];cfg=item['HostConfig']
        assert cfg['Memory']==4*1024**3 and cfg['NanoCpus']==10**9
        docker.append({'engine':engine,'memory_bytes':cfg['Memory'],'cpuset':cfg['CpusetCpus'],'nano_cpus':cfg['NanoCpus'],'network':cfg['NetworkMode']})
    assert docker[0]['cpuset']==docker[1]['cpuset']
    result={'independently_verified_db_fold_methods':checked,'failed_method_cases':len(errors),
            'metric_discrepancies':findings,'docker_limits':docker,
            'scope':'classification accuracy/balanced accuracy/macro F1 and regression RMSE/MAE/R2 from raw DB predictions; not AUC',
            'ready_with_stated_methodology_caveats':not errors and not findings}
    (root/'independent-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if findings:raise SystemExit('metric discrepancies found')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);a=p.parse_args();audit(a.results)
