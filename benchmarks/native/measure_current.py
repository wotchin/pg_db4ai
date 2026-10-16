#!/usr/bin/env python3
"""Measure ONLY the current pg_db4ai native implementation on frozen inputs.
No external estimator is fitted; no openGauss container or old extension runs.
Use PGHOST/PGPORT/PGUSER/PGDATABASE and --psql for an isolated test instance.
"""
from __future__ import annotations
import argparse, hashlib, io, json, os, re, subprocess, time
from pathlib import Path
import numpy as np


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def settings(work_mem):
    return ("SET client_min_messages=warning; SET jit=off; SET max_parallel_workers_per_gather=0; "
            "SET statement_timeout='180s'; SET work_mem='64MB'; "
            f"SET db4ai.work_mem={int(work_mem)};\n")


def run_sql(psql, text, log=None):
    result = subprocess.run([psql, '-X', '-q', '-A', '-t', '-v', 'ON_ERROR_STOP=1'],
                            input=text, text=True, capture_output=True, timeout=600, check=False)
    if log:
        log.write_text(result.stdout+'\n-- STDERR --\n'+result.stderr)
        log.with_suffix('.sql').write_text(text)
    if result.returncode:
        raise RuntimeError(result.stderr + '\n' + result.stdout[-2000:])
    return result.stdout


def configurations(meta):
    if meta['task']=='regression': return ['linear_gd', 'linear_ngd']
    if meta['task']=='binary': return ['logistic_gd','logistic_ngd','svm_linear','svm_gaussian']
    return ['ova_logistic', 'ova_svm']


def options(method, seed):
    result = dict(profile='opengauss', max_iterations=200, batch_size=128,
                  optimizer='ngd' if method.endswith('ngd') else 'gd',
                  learning_rate=.8 if method.endswith('ngd') else .05, decay=1,
                  tolerance=1e-30, seed=seed, max_seconds=0,
                  shuffle='batch', shuffle_buffer_size=128, null_policy='error')
    if method.startswith('svm') or method=='ova_svm': result['lambda']=.5
    if method=='svm_gaussian': result.update(kernel='gaussian',gamma=.5,components=128)
    if method.startswith('ova'): result['classifier']='logistic_regression' if method=='ova_logistic' else 'svm_classification'
    return result


def algorithm(method):
    if method.startswith('linear'): return 'linear_regression'
    if method.startswith('logistic'): return 'logistic_regression'
    if method.startswith('svm'): return 'svm_classification'
    return 'multiclass'


def quality(y, pred, meta):
    if meta['task']=='regression':
        pred=pred*meta['y_scale']+meta['y_mean']
        diff=pred-y
        return dict(rmse=float(np.sqrt(np.mean(diff*diff))),mae=float(np.abs(diff).mean()),
                    r2=float(1-np.dot(diff,diff)/np.dot(y-y.mean(),y-y.mean())))
    recall=[];f1=[]
    for cls in np.unique(y):
        tp=np.count_nonzero((pred==cls)&(y==cls));fp=np.count_nonzero((pred==cls)&(y!=cls));fn=np.count_nonzero((pred!=cls)&(y==cls))
        recall.append(tp/(tp+fn));f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return dict(accuracy=float(np.mean(pred==y)),balanced_accuracy=float(np.mean(recall)),macro_f1=float(np.mean(f1)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--psql',default='psql')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--datasets',default='breast_cancer,banknote,wine,digits,diabetes,wine_red,concrete,california')
    parser.add_argument('--variants',default='batch,corgipile')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--work-mem',type=int,default=65536)
    parser.add_argument('--kernel-ablation',action='store_true')
    a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.repeats<1: raise ValueError('repeats must be positive')
    for v in a.variants.split(','):
        if v not in ('batch','corgipile'): raise ValueError('unsupported variant')
    env=dict(source_revision=os.environ.get('SOURCE_REVISION','local-working-tree'),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
             version=run_sql(a.psql,'SELECT version();').strip(),
             extension=run_sql(a.psql,"SELECT extversion FROM pg_extension WHERE extname='pg_db4ai';").strip(),
             build_info=json.loads(run_sql(a.psql,'SELECT db4ai.build_info();')), 
             cpu=subprocess.run(['lscpu'],capture_output=True,text=True).stdout,
             seeds=[a.seed],repeats=a.repeats,work_mem_kib=a.work_mem,
             baseline_policy='Frozen external/old-extension results only; no external estimator or old extension executed.',
             fit_scope='SQL wall time including input, conversion, training, diagnostics, model serialization/persistence; excluding client startup and COPY',
             metric_scope='one last-repeat model per method/variant/split; not repeats as independent samples')
    write(a.out/'environment.json',env)
    result=[]
    for dataset in a.datasets.split(','):
        case=a.data/f'{dataset}-{a.seed}'; meta=json.loads((case/'meta.json').read_text());npz=np.load(case/'data.npz')
        if hashlib.sha256(np.ascontiguousarray(npz['X_train']).tobytes()).hexdigest()!=meta['train_X_sha256']:
            raise ValueError('frozen input fingerprint mismatch')
        dst=a.out/case.name;dst.mkdir(exist_ok=True);write(dst/'input-meta.json',meta)
        cols=','.join(f'x{i}' for i in range(meta['features']))
        declarations=','.join(f'x{i} float8' for i in range(meta['features']))
        label='float8' if meta['task']=='regression' else 'integer'
        text='DROP TABLE IF EXISTS native_train,native_test;\n'
        for split in ('train','test'):
            text+=f'CREATE TABLE native_{split}(id integer,y {label},{declarations});\n'
            text+=f'COPY native_{split} FROM STDIN CSV;\n'+(case/(split+'.csv')).read_text()+'\\.\n'
            text+=f'ANALYZE native_{split};\n'
        run_sql(a.psql,text)
        tasks=[(m,v) for m in configurations(meta) for v in a.variants.split(',')]
        if a.kernel_ablation and 'svm_gaussian' in configurations(meta): tasks.append(('svm_gaussian','uncached'))
        np.random.default_rng(a.seed).shuffle(tasks)
        for method,variant in tasks:
            o=options(method,a.seed)
            if variant=='corgipile':
                del o['shuffle_buffer_size'];o.update(shuffle='corgipile',shuffle_block_rows=256,shuffle_window_blocks=8)
            if variant=='uncached':o['kernel_cache']=False
            prefix=f'{method}-{variant}';script=settings(a.work_mem)
            model=f'n_{dataset}_{method}_{variant}'
            for repeat in range(-1,a.repeats):
                key=model+str(repeat+1)
                script+=f'SELECT db4ai.drop_model({quote(key)});\n'
                script+='\\timing on\n'+f'\\echo __TIME__fit_{repeat}\n'
                script+=f'SELECT db4ai.train({quote(key)},{quote(algorithm(method))},{quote("SELECT y,"+cols+" FROM native_train ORDER BY id")},{quote(json.dumps(o))}::jsonb);\n\\timing off\n'
            last=model+str(a.repeats)
            fn='predict_float8' if meta['task']=='regression' else 'predict_label'
            expr=f'db4ai.{fn}({quote(last)},ARRAY[{cols}]::float8[]'+(',NULL::integer)' if fn=='predict_label' else ')')
            for repeat in range(-1,a.repeats):
                script+=f'\\timing on\n\\echo __TIME__predict_{repeat}\nSELECT sum(({expr})::float8) FROM native_test;\n\\timing off\n'
            output=run_sql(a.psql,script,dst/(prefix+'.log'))
            timing={};marker=None
            for line in output.splitlines():
                if line.startswith('__TIME__'):marker=line[len('__TIME__'):]
                match=re.match(r'Time:\s*([\d.]+)\s*ms',line)
                if marker and match:timing[marker]=float(match.group(1));marker=None
            data=run_sql(a.psql,f'COPY (SELECT id,{expr} FROM native_test ORDER BY id) TO STDOUT CSV;')
            (dst/(prefix+'-predictions.csv')).write_text(data)
            pred=np.loadtxt(io.StringIO(data),delimiter=',',ndmin=2)
            assert np.array_equal(pred[:,0],np.arange(meta['n_test']))
            q=quality(npz['y_test'],pred[:,1],meta)
            modelinfo=json.loads(run_sql(a.psql,f'SELECT db4ai.explain_model({quote(last)});').strip())
            write(dst/(prefix+'-model.json'),modelinfo)
            payload=run_sql(a.psql,f'SELECT md5(db4ai.export_model({quote(last)}));').strip()
            record=dict(dataset=dataset,method=method,variant=variant,seed=a.seed,
                        parameters=o,fit_ms=[timing[f'fit_{i}'] for i in range(a.repeats)],
                        predict_ms=[timing[f'predict_{i}'] for i in range(a.repeats)],
                        fit_median_ms=float(np.median([timing[f'fit_{i}'] for i in range(a.repeats)])),
                        predict_median_ms=float(np.median([timing[f'predict_{i}'] for i in range(a.repeats)])),
                        metrics=modelinfo['metrics'],quality=q,model_md5=payload,n_train=meta['n_train'],n_test=meta['n_test'])
            oldpath=case/('pg_'+method+'-predictions.csv')
            if variant=='batch' and oldpath.exists():
                old=np.loadtxt(oldpath,delimiter=',',ndmin=2)
                record['frozen_old_pg_prediction_max_abs_diff']=float(np.max(np.abs(pred[:,1]-old[:,1])))
            result.append(record);write(a.out/'results.json',result)
            print(dataset,method,variant,record['fit_median_ms'],q,flush=True)
    for dataset in a.datasets.split(','):
        rows={r['variant']:r for r in result if r['dataset']==dataset and r['method']=='svm_gaussian'}
        if 'uncached' in rows and 'batch' in rows:
            assert rows['uncached']['model_md5']==rows['batch']['model_md5'], 'cache changes model!'
    write(a.out/'completion.json',dict(complete=True,cases=len(result),external_baselines_executed=0))

if __name__=='__main__':main()
