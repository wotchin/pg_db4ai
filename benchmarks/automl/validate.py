#!/usr/bin/env python3
"""Current-only AutoML evaluation on frozen outer splits; no external estimator.

The old standardized NPZ is inverted with its saved parameters to recover raw
features (within roundoff). Only the frozen outer TRAIN is supplied to AutoML;
AutoML makes internal train/validation/test splits. The frozen outer TEST is
never used for search, parameter selection or publication. A disposable DB is
required: this script drops its own am_bench_input/am_bench_holdout tables.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, pathlib, statistics, subprocess, time
import numpy as np

def write(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def run_sql(bin, text, timeout=90):
    r=subprocess.run([str(bin/'psql'),'-X','-q','-At','-v','ON_ERROR_STOP=1'],input=text,text=True,capture_output=True,timeout=timeout)
    if r.returncode: raise RuntimeError(r.stderr[-4000:])
    return r.stdout

def quality(y,p,classification):
    if not classification:
        d=p-y;ss=float(np.sum((y-y.mean())**2))
        return dict(rmse=float(np.sqrt(np.mean(d*d))),mae=float(np.mean(np.abs(d))),r2=1-float(np.sum(d*d))/ss if ss else None)
    labels=np.unique(y); rec=[]; f1=[]
    for label in labels:
        tp=int(np.sum((y==label)&(p==label)));fp=int(np.sum((y!=label)&(p==label)));fn=int(np.sum((y==label)&(p!=label)))
        rec.append(tp/(tp+fn));f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return dict(accuracy=float(np.mean(y==p)),balanced_accuracy=float(np.mean(rec)),macro_f1=float(np.mean(f1)))

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--inputs',type=pathlib.Path,required=True);a.add_argument('--out',type=pathlib.Path,required=True)
    a.add_argument('--budget',type=float,default=2);a.add_argument('--repeats',type=int,default=3);a.add_argument('--seed',type=int)
    args=a.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True);out.chmod(0o777)
    bin=pathlib.Path(subprocess.check_output([os.environ.get('PG_CONFIG','pg_config'),'--bindir'],text=True).strip())
    records=[]
    cases=sorted(args.inputs.glob('*/meta.json'))
    for mp in cases:
        meta=json.loads(mp.read_text()); name=mp.parent.name
        if args.seed is not None and meta['seed']!=args.seed:continue
        folder=out/name;folder.mkdir(exist_ok=True);folder.chmod(0o777)
        data=np.load(mp.parent/'data.npz');classification=meta['task']!='regression'
        xtrain=data['X_train']*np.asarray(meta['x_scale'])+np.asarray(meta['x_mean'])
        xtest=data['X_test']*np.asarray(meta['x_scale'])+np.asarray(meta['x_mean'])
        ytrain=data['y_train']*meta['y_scale']+meta['y_mean'];ytest=data['y_test']
        cols=','.join(f'x{j} float8' for j in range(xtrain.shape[1]));names=','.join(f'x{j}' for j in range(xtrain.shape[1]))
        np.savetxt(folder/'train.csv',np.column_stack([np.arange(len(ytrain)),ytrain,xtrain]),delimiter=',',fmt='%.17g')
        np.savetxt(folder/'test.csv',np.column_stack([np.arange(len(ytest)),ytest,xtest]),delimiter=',',fmt='%.17g')
        typ='integer' if classification else 'float8'
        # ids and integer labels must not use exponent notation in COPY.
        for fn in ['train.csv','test.csv']:
            text=(folder/fn).read_text().splitlines();clean=[]
            for line in text:
                v=line.split(',');v[0]=str(int(float(v[0])))
                if classification:v[1]=str(int(float(v[1])))
                clean.append(','.join(v))
            (folder/fn).write_text('\n'.join(clean)+'\n');(folder/fn).chmod(0o644)
        sql=f"DROP TABLE IF EXISTS am_bench_input,am_bench_holdout; CREATE TABLE am_bench_input(id integer,y {typ},{cols}); CREATE TABLE am_bench_holdout(LIKE am_bench_input); COPY am_bench_input FROM '{folder}/train.csv' CSV; COPY am_bench_holdout FROM '{folder}/test.csv' CSV;"
        run_sql(bin,sql)
        options=dict(time_budget_seconds=args.budget,seed=meta['seed'],max_trials=8)
        task='classification' if classification else 'regression'
        timings=[]; reports=[];status='ok';error=None
        for repeat in range(-1,args.repeats):
            model='auto_'+name.replace('-','_')+'_'+str(int(args.budget*1000))+'_'+str(repeat+1)
            query=f"SELECT db4ai.automl('{model}','{task}','SELECT y,{names} FROM public.am_bench_input ORDER BY id','{json.dumps(options)}');"
            try:
                output=run_sql(bin,query);report=json.loads(output.strip())
                write(folder/f'report-{repeat}.json',report)
                if report['status']!='ready':status=report['status'];break
                if repeat>=0:timings.append(report['elapsed_ms']);reports.append(report)
            except Exception as exc:
                status='error';error=str(exc);(folder/f'error-{repeat}.txt').write_text(error);break
        record=dict(dataset=meta['dataset'],seed=meta['seed'],budget_seconds=args.budget,status=status,
                    outer_train_rows=len(ytrain),outer_test_rows=len(ytest),raw_features=xtrain.shape[1],error=error)
        if status=='ok':
            artifact=json.loads(run_sql(bin,f"SELECT convert_from(db4ai.export_model('{model}'),'UTF8')::jsonb;").strip());write(folder/'model.json',artifact)
            preds=run_sql(bin,f"SELECT id,db4ai.predict('{model}',to_jsonb(t)) FROM public.am_bench_holdout t ORDER BY id;")
            outputs=[json.loads(line.split('|',1)[1]) for line in preds.splitlines()];p=np.asarray([float(r['prediction']) for r in outputs])
            write(folder/'predictions.json',outputs);np.save(folder/'target.npy',ytest)
            q=quality(ytest,p,classification)
            if classification:
                labs=outputs[0]['classes']; probs=np.asarray([r['probabilities'] for r in outputs]);ids=[labs.index(str(int(y))) for y in ytest]
                assert np.isfinite(probs).all() and np.min(probs)>=0 and np.allclose(probs.sum(axis=1),1,atol=1e-12)
                q['log_loss']=float(-np.log(np.maximum(1e-15,probs[np.arange(len(ids)),ids])).mean())
            assert report['source_executions']==1 and report['test_used_for_selection'] is False
            record.update(q,elapsed_median_ms=statistics.median(timings),elapsed_trials_ms=timings,
                          cpu_seconds_median=statistics.median(r['cpu_seconds'] for r in reports),
                          selected_algorithm=report['selected_algorithm'],trial_count=report['trial_count'],
                          budget_stopped=report['budget_stopped'],selected_training_rows=report['selected_training_rows'],
                          internal_validation=report['validation'],internal_test=report['test'],encoded_features=report['encoded_features'])
        record['frozen_npz_sha256']=hashlib.sha256((mp.parent/'data.npz').read_bytes()).hexdigest()
        write(folder/'result.json',record);write(folder/'frozen-meta.json',meta)
        records.append(record);write(out/'results.json',records);print(json.dumps(record),flush=True)
    fields=['dataset','seed','status','budget_seconds','selected_algorithm','outer_train_rows','outer_test_rows','selected_training_rows','encoded_features','trial_count','elapsed_median_ms','cpu_seconds_median','accuracy','balanced_accuracy','macro_f1','log_loss','rmse','mae','r2','budget_stopped','error']
    with (out/'per-split.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)
    summary=[]
    for ds in sorted(set(r['dataset'] for r in records)):
        rows=[r for r in records if r['dataset']==ds];good=[r for r in rows if r['status']=='ok'];row=dict(dataset=ds,cases=len(rows),completed=len(good))
        if good:
            for key in ['accuracy','balanced_accuracy','macro_f1','log_loss','rmse','mae','r2']:
                vals=[r[key] for r in good if key in r and r[key] is not None]
                if vals:row[key+'_mean']=statistics.mean(vals);row[key+'_std']=statistics.stdev(vals) if len(vals)>1 else None
            row['elapsed_median_ms']=statistics.median(r['elapsed_median_ms'] for r in good)
            row['selected_algorithms']=sorted(set(r['selected_algorithm'] for r in good))
        summary.append(row)
    write(out/'summary.json',summary)
    write(out/'environment.json',dict(pg=run_sql(bin,'SELECT version();').strip(),build=json.loads(run_sql(bin,'SELECT db4ai.build_info();').strip()),cpu=subprocess.check_output(['lscpu'],text=True),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
      protocol='current-only; frozen outer train only; raw features inverse-transformed from saved scaler; AutoML internal 60/20/20; outer test never used in search; one warmup + repeated fits; no external baselines',budget=args.budget,repeats=args.repeats))
    print('FINISHED',len(records),'cases; successful',sum(r['status']=='ok' for r in records),flush=True)
if __name__=='__main__':main()
