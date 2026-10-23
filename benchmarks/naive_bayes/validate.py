#!/usr/bin/env python3
"""Validate ONLY current pg_db4ai Gaussian NB on already-frozen classification splits.
No sklearn estimator, external engine or previous pg_db4ai version is executed.
Requires numpy and a disposable PG database with pg_db4ai 0.4 installed.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import time
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frozen',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--psql',default='psql')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    def sql(q,log=None):
        p=subprocess.run([args.psql,'-X','-Atq','-v','ON_ERROR_STOP=1'],input=q,text=True,capture_output=True,check=True)
        if log:
            log.write_text(p.stdout+'\n-- STDERR --\n'+p.stderr)
            log.with_suffix('.sql').write_text(q)
        return p.stdout.strip()
    records=[]
    for case in sorted(args.frozen.glob('*')):
        if not (case/'data.npz').exists():continue
        meta=json.loads((case/'meta.json').read_text())
        if meta['task'] not in ['binary','multiclass']:continue
        z=np.load(case/'data.npz');X=z['X_train'];Xt=z['X_test'];y=z['y_train'].astype(int);yt=z['y_test'].astype(int)
        assert hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()==meta['train_X_sha256']
        assert not set(z['train_ids']).intersection(z['test_ids'])
        caseout=args.out/case.name;caseout.mkdir(exist_ok=True)
        q="SET jit=off; SET max_parallel_workers_per_gather=0; DROP TABLE IF EXISTS public.nb_validate_train,public.nb_validate_test;\n"
        for split,xx,yy in [('train',X,y),('test',Xt,yt)]:
            q+=f'CREATE TABLE public.nb_validate_{split}(id int,y int,x float8[]);\nCOPY public.nb_validate_{split} FROM STDIN;\n'
            q+=''.join(f'{i}\t{label}\t{{'+','.join(format(a,'.17g') for a in row)+'}\n' for i,(label,row) in enumerate(zip(yy,xx)))+'\\.\n'
        sql(q,caseout/'load.log')
        models=[];times=[]
        for r in range(-1,3):
            name=f'nb_{meta["dataset"]}_{meta["seed"]}_{r+1}';models.append(name)
            sql(f"SELECT db4ai.drop_model('{name}');")
            q="SET jit=off; SET max_parallel_workers_per_gather=0;\n\\timing on\n"+f"SELECT db4ai.train('{name}','gaussian_naive_bayes','SELECT y,x FROM public.nb_validate_train ORDER BY id');"
            log=sql(q,caseout/f'fit-{r}.log');t=float(re.findall(r'Time:\s*([\d.]+)\s*ms',log)[-1])
            if r>=0:times.append(t)
        name=models[-1]
        classes=np.array(json.loads(sql(f"SELECT to_json(db4ai.class_labels('{name}'));")),dtype=int)
        means=np.array([X[y==c].mean(axis=0) for c in classes])
        var=np.array([X[y==c].var(axis=0) for c in classes])+max(1e-12,1e-9*X.var(axis=0).max())
        prior=np.array([(y==c).mean() for c in classes])
        payload=json.loads(sql(f"SELECT convert_from(db4ai.export_model('{name}'),'UTF8')::jsonb;"))
        fitted_means=np.array(payload['means']).reshape(means.shape)
        fitted_var=np.array(payload['variances']).reshape(var.shape)
        fitted_prior=np.array(payload['class_priors'])
        # Validate fitting separately from inference: tiny moment-rounding errors
        # can be amplified in log probabilities with nearly zero variances.
        moment_delta=float(max(abs(fitted_means-means).max(),abs(fitted_var-var).max()))
        assert np.allclose(fitted_means,means,rtol=1e-12,atol=1e-12)
        assert np.allclose(fitted_var,var,rtol=1e-12,atol=1e-12)
        assert np.allclose(fitted_prior,prior,rtol=1e-14,atol=1e-14)
        means,var,prior=fitted_means,fitted_var,fitted_prior
        logscore=np.array([np.log(p)-.5*(np.log(2*np.pi)+np.log(v)+(Xt-m)**2/v).sum(axis=1) for m,v,p in zip(means,var,prior)]).T
        # Independent centered likelihood-ratio equations in extended precision
        # for rows whose common tail penalty would destroy float64 log odds.
        for row_index in np.where(logscore.max(axis=1)<-1e6)[0]:
            xx=Xt[row_index].astype(np.longdouble)
            mu=means.astype(np.longdouble);vv=var.astype(np.longdouble)
            anchor=int(logscore[row_index].argmax())
            relative=[]
            for c in range(len(classes)):
                terms=(mu[anchor]-mu[c])*(2*xx-mu[c]-mu[anchor])/vv[c]
                terms+=(xx-mu[anchor])**2/vv[anchor]*((vv[anchor]-vv[c])/vv[c])
                relative.append(np.log(np.longdouble(prior[c])/prior[anchor])-.5*(np.log(vv[c]/vv[anchor])+terms).sum())
            relative=np.asarray(relative,dtype=np.longdouble)
            logscore[row_index]=(relative-relative.max()).astype(float)
        shifted=logscore-logscore.max(axis=1,keepdims=True)
        expected_log=shifted-np.log(np.exp(shifted).sum(axis=1,keepdims=True));expected_p=np.exp(expected_log)
        out=sql(f"COPY (SELECT id,db4ai.predict_label('{name}',x,NULL::int),to_json(db4ai.predict_proba('{name}',x)),to_json(db4ai.predict_log_proba('{name}',x)) FROM public.nb_validate_test ORDER BY id) TO STDOUT CSV;",caseout/'predict.log')
        (caseout/'predictions.csv').write_text(out+'\n')
        data=list(csv.reader(io.StringIO(out)));assert [int(r[0]) for r in data]==list(range(len(yt)))
        pred=np.array([int(r[1]) for r in data]);p=np.array([json.loads(r[2]) for r in data]);lp=np.array([json.loads(r[3]) for r in data])
        assert np.allclose(p,expected_p,rtol=1e-9,atol=1e-10)
        assert np.allclose(lp,expected_log,rtol=1e-10,atol=1e-8)
        assert np.array_equal(pred,classes[expected_log.argmax(axis=1)])
        assert np.all(np.isfinite(lp)) and np.all((p>=0)&(p<=1)) and np.allclose(p.sum(axis=1),1,rtol=0,atol=2e-14)
        recalls=[];f1=[]
        for c in classes:
            tp=np.sum((pred==c)&(yt==c));fp=np.sum((pred==c)&(yt!=c));fn=np.sum((pred!=c)&(yt==c))
            recalls.append(float(tp/(tp+fn)));f1.append(float(2*tp/(2*tp+fp+fn)))
        ix={int(c):i for i,c in enumerate(classes)}
        record=dict(dataset=meta['dataset'],seed=meta['seed'],n_train=len(y),n_test=len(yt),nfeatures=X.shape[1],classes=classes.tolist(),
            accuracy=float((pred==yt).mean()),balanced_accuracy=float(np.mean(recalls)),macro_f1=float(np.mean(f1)),
            log_loss=float(-lp[np.arange(len(yt)),[ix[int(c)] for c in yt]].mean()),
            fit_ms=times,fit_median_ms=float(np.median(times)),
            training_moment_max_abs_difference=moment_delta,max_probability_difference=float(abs(p-expected_p).max()),max_log_probability_difference=float(abs(lp-expected_log).max()),
            train_X_sha256=meta['train_X_sha256'],train_ids_sha256=meta['train_ids_sha256'],test_ids_sha256=meta['test_ids_sha256'],
            status='passed')
        records.append(record);print(json.dumps(record),flush=True)
        (caseout/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
        (caseout/'model.json').write_text(json.dumps(payload,indent=2)+'\n')
    (args.out/'results.json').write_text(json.dumps(records,indent=2)+'\n')
    environment=dict(postgres=sql('SELECT version();'),build=json.loads(sql('SELECT db4ai.build_info();')),
        cpu=subprocess.check_output(['lscpu'],text=True),numpy=np.__version__,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        cases=len(records),external_baselines_executed=0,protocol='frozen inputs, default parameters, no tuning; one warmup and three fit repetitions; independent direct-equation probability checks on every holdout row')
    (args.out/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    print('All current Gaussian NB validation cases passed:',len(records))

if __name__=='__main__':main()
