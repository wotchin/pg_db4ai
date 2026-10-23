#!/usr/bin/env python3
"""Read-only audit of AutoML evidence, using independent stdlib metric formulas.
Does not run PostgreSQL, train models, invoke sklearn or choose hyperparameters.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, statistics
from collections import Counter
from pathlib import Path


def audit(root: Path) -> dict:
    rows=json.loads((root/'results.json').read_text()); report_count=0; prediction_rows=0; native_rows=0
    for record in rows:
        if record['status']!='ok':
            continue  # Explicitly retained failures, not fabricated zero metrics.
        case=root/f"{record['dataset']}-{record['seed']}"
        preds=json.loads((case/'predictions.json').read_text())
        with (case/'test.csv').open() as f:
            raw=list(csv.reader(f))
        assert len(raw)==len(preds)==record['outer_test_rows']
        assert [int(v[0]) for v in raw]==list(range(len(raw)))
        y=[float(v[1]) for v in raw]; prediction=[float(p['prediction']) for p in preds]
        n=len(y); classification='accuracy' in record
        if classification:
            correct=Counter(); actual=Counter(y); estimated=Counter(prediction)
            for a,b in zip(y,prediction):
                if a==b:correct[a]+=1
            values={'accuracy':sum(correct.values())/n,
                    'balanced_accuracy':statistics.mean(correct[k]/actual[k] for k in actual),
                    'macro_f1':statistics.mean(2*correct[k]/(actual[k]+estimated[k]) for k in actual)}
            loss=[]
            for a,p in zip(y,preds):
                probabilities=p['probabilities']; classes=p['classes']
                assert all(math.isfinite(x) and 0<=x<=1 for x in probabilities)
                assert math.isclose(math.fsum(probabilities),1,abs_tol=1e-12)
                assert p['prediction']==classes[p['class_index']]
                assert p['class_index']==max(range(len(classes)),key=lambda i:probabilities[i])
                loss.append(-math.log(max(1e-15,probabilities[classes.index(str(int(a)))])))
            values['log_loss']=math.fsum(loss)/n
        else:
            squared=math.fsum((a-b)**2 for a,b in zip(y,prediction));mean=math.fsum(y)/n
            variance=math.fsum((a-mean)**2 for a in y)
            values={'rmse':math.sqrt(squared/n),'mae':math.fsum(abs(a-b) for a,b in zip(y,prediction))/n,
                    'r2':1-squared/variance}
        for key,value in values.items():
            assert math.isclose(record[key],value,rel_tol=1e-11,abs_tol=1e-12),(case.name,key,value,record[key])
        reports=[json.loads(p.read_text()) for p in sorted(case.glob('report-*.json'))]
        for report in reports:
            assert report['status']=='ready' and report['source_executions']==1
            assert report['test_used_for_selection'] is False and report['validation_used_for_selection'] is True
            assert report['source_rows']==report['train_rows']+report['validation_rows']+report['test_rows']==record['outer_train_rows']
            assert report['selected_training_rows']<=report['train_rows']
            completed=[t for t in report['trials'] if t['status']=='completed']
            key=report['metric']; sign=-1 if key in ('accuracy','balanced_accuracy','roc_auc') else 1
            objective=lambda t:sign*t['validation'][key]
            expected=min(objective(t) for t in completed)
            assert math.isclose(sign*report['validation'][key],expected,rel_tol=1e-11,abs_tol=1e-12)
            assert sign*report['validation'][key]<sign*report['validation_baseline'][key]
            assert report['trial_count']==len(report['trials']) and report['completed_trials']==len(completed)
            assert report['threads']==1 and report['elapsed_ms']<report['budget_seconds']*1000
            report_count+=1
        model=json.loads((case/'model.json').read_text());child=model['learner']
        assert child['algorithm']==record['selected_algorithm']
        assert len(model['features'])==record['raw_features']
        # Independently replay saved numeric preprocessing and native inference.
        # Tree predictions are metric-audited but NOT claimed to be independently
        # verified tree traversal by this script.
        if child['format']=='json' and child['algorithm'] in ('linear_regression','logistic_regression','multiclass'):
            learned=json.loads(bytes.fromhex(child['data_hex']));weights=learned['weights'];width=learned['n_coefficients']
            def sigmoid(z):
                return 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
            for input_row,output in zip(raw,preds):
                encoded=[0.0]*model['nencoded']
                for j,feature in enumerate(model['features']):
                    assert not feature['categorical']
                    encoded[feature['offset']]=(float(input_row[j+2])-feature['mean'])/feature['scale']
                vector=[1.0]+encoded
                scores=[math.fsum(w*x for w,x in zip(weights[c*width:(c+1)*width],vector))
                        for c in range(model['nclasses'] if child['algorithm']=='multiclass' else 1)]
                if not classification:
                    expected=scores[0]*model['target_scale']+model['target_mean']
                    assert math.isclose(expected,float(output['prediction']),rel_tol=1e-10,abs_tol=1e-9)
                else:
                    probabilities=([1-sigmoid(scores[0]),sigmoid(scores[0])] if len(scores)==1 else [sigmoid(z) for z in scores])
                    total=math.fsum(probabilities); probabilities=[p/total for p in probabilities]
                    assert all(math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12) for a,b in zip(probabilities,output['probabilities']))
                native_rows+=1
        prediction_rows+=n
    receipt={'cases':len(rows),'successful_cases':sum(r['status']=='ok' for r in rows),
             'selection_and_budget_reports_checked':report_count,'outer_predictions_metric_checked':prediction_rows,
             'native_preprocess_and_inference_rows_checked':native_rows,
             'scope':'independent stdlib metrics/selection/budget checks; saved native model inference, no independent XGBoost traversal claim',
             'results_sha256':hashlib.sha256((root/'results.json').read_bytes()).hexdigest(),
             'no_estimator_or_database_executed':True,'passed':True}
    (root/'audit.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2));return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path);args=p.parse_args();audit(args.results)
