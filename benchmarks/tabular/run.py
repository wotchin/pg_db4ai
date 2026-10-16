#!/usr/bin/env python3
"""Same-host Docker benchmark. Timings exclude docker exec and client startup.
SQL timings include query execution and model persistence (training), or model
lookup/load plus scan and sum (prediction). Python resident-batch timings are
separately labeled and are not a pure database-overhead ratio.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import time
import traceback
from pathlib import Path
import numpy as np
from common import SOURCES, dump, prepare, metrics
from export_transport import collect_export, require_final_server

ROOT=Path(__file__).resolve().parents[2]
OG_IMAGE='opengauss-db4ai-benchmark:local'
PG_IMAGE='pg-db4ai-benchmark:local'


def command(args, text=None, timeout=180):
    r=subprocess.run(args,input=text,text=True,capture_output=True,timeout=timeout)
    if r.returncode:
        raise RuntimeError(f'command failed ({r.returncode}): {args}\n{r.stdout[-6000:]}\n{r.stderr[-6000:]}')
    return r.stdout


def sql(engine, text, out:Path|None=None, timeout=300):
    if out is None and text == 'SELECT version();':
        require_final_server(engine)
    cli='psql' if engine=='pg' else 'gsql'; user='postgres' if engine=='pg' else 'omm'
    cmd=['docker','exec','-i','-u',user,'bench-'+engine,'bash','-lc',f'{cli} -d postgres -X -q -A -t -v ON_ERROR_STOP=1']
    r=subprocess.run(cmd,input=text,text=True,capture_output=True,timeout=timeout)
    if out:
        out.write_text(r.stdout+'\n--- STDERR ---\n'+r.stderr)
        out.with_suffix('.sql').write_text(text)
    if r.returncode: raise RuntimeError(r.stderr[-4000:]+'\n'+r.stdout[-4000:])
    if engine == 'og' and out is not None and out.name.endswith('-prediction-export.log'):
        target = out.with_name(out.name.replace('-prediction-export.log', '-predictions.csv'))
        collect_export(target, '/bench/' + out.parent.name + '/' + target.name, engine)
    return r.stdout


def quote(s): return "'"+str(s).replace("'","''")+"'"

def header(engine):
    common="SET client_min_messages=warning; SET work_mem='64MB'; SET statement_timeout='120s';\n"
    return common+("SET jit=off; SET max_parallel_workers_per_gather=0; SET db4ai.work_mem=65536;\n" if engine=='pg' else "SET query_dop=1;\n")


def timings(output):
    found={}; current=None
    for line in output.splitlines():
        if line.startswith('__BENCH__'): current=line.strip()[9:]
        m=re.match(r'Time:\s*([\d.]+)\s*ms',line)
        if m and current is not None:
            if current in found: raise ValueError('duplicate timing marker '+current)
            found[current]=float(m.group(1)); current=None
    return found


def cases_methods(meta):
    if meta['task']=='regression': return ['linear_gd','linear_ngd','xgboost']
    if meta['task']=='binary': return ['logistic_gd','logistic_ngd','svm_linear','svm_gaussian','xgboost']
    return ['ova_logistic','ova_svm']


def config(method, seed):
    if method=='xgboost':
        return dict(max_iterations=100,learning_rate=0.1,max_depth=6,l2=1,
                    gamma=0,min_child_weight=1,seed=seed,tree_method='hist',nthread=1,eval_metric='rmse')
    o=dict(profile='opengauss',max_iterations=200,batch_size=128,optimizer='ngd' if method.endswith('ngd') else 'gd',
           learning_rate=0.8 if method.endswith('ngd') else 0.05,decay=1,tolerance=1e-30,seed=seed,
           max_seconds=0,shuffle='batch',shuffle_buffer_size=128,null_policy='error')
    if method.startswith('svm') or method=='ova_svm': o['lambda']=0.5
    if method=='svm_gaussian': o.update(kernel='gaussian',gamma=0.5,components=128)
    if method.startswith('ova'): o['classifier']='logistic_regression' if method=='ova_logistic' else 'svm_classification'
    return o


def algorithm(method, meta):
    if method.startswith('linear'): return 'linear_regression'
    if method.startswith('logistic'): return 'logistic_regression'
    if method.startswith('svm'): return 'svm_classification'
    if method.startswith('ova'): return 'multiclass'
    return 'xgboost_regression_squarederror' if meta['task']=='regression' else 'xgboost_binary_logistic'


def train_statement(engine, method, meta, name, columns):
    opts=config(method,meta['seed']); alg=algorithm(method,meta)
    if engine=='pg':
        query=f'SELECT y,{columns} FROM public.train_set ORDER BY id'
        return f'SELECT db4ai.train({quote(name)},{quote(alg)},{quote(query)},{quote(json.dumps(opts))}::jsonb);'
    if method=='xgboost':
        og=dict(n_iter=100,eta=0.1,max_depth=6,gamma=0,min_child_weight=1,seed=meta['seed'],
                tree_method='hist',nthread=1,verbosity=0,batch_size=meta['n_train'],eval_metric='rmse')
    else:
        og={k:v for k,v in opts.items() if k not in ('profile','shuffle','shuffle_buffer_size','null_policy')}
    param=', '.join(f'{k}={quote(v) if isinstance(v,str) else v}' for k,v in og.items())
    return f'CREATE MODEL {name} USING {alg} FEATURES {columns} TARGET y FROM (SELECT * FROM public.train_set ORDER BY id) s WITH {param};'


def predict_expression(engine,method,meta,name,columns):
    if engine=='og': return f'PREDICT BY {name}(FEATURES {columns})'
    if method=='xgboost' or meta['task']=='regression': return f'db4ai.predict_float8({quote(name)},ARRAY[{columns}]::float8[])'
    return f'db4ai.predict_label({quote(name)},ARRAY[{columns}]::float8[],NULL::integer)'


def load_case(engine,case,meta):
    cols=', '.join(f'x{i} double precision' for i in range(meta['features']))
    target='double precision' if meta['task']=='regression' else 'integer'
    txt=header(engine)+'DROP TABLE IF EXISTS public.train_set; DROP TABLE IF EXISTS public.test_set;\n'
    for split in ('train','test'):
        txt+=f'CREATE TABLE public.{split}_set(id integer,y {target},{cols});\n'
        txt+=f"\\timing on\n\\echo __BENCH__load_{split}\nCOPY public.{split}_set FROM '/bench/{case.name}/{split}.csv' CSV;\n\\timing off\nANALYZE public.{split}_set;\n"
    return timings(sql(engine,txt,case/(engine+'-load.log')))


def benchmark_db(engine,case,meta,method,repeats):
    records=[]; prefix=engine+'_'+method
    columns=','.join('x'+str(i) for i in range(meta['features']))
    name='b_'+method+'_'+meta['dataset']+'_'+str(meta['seed'])
    start=time.perf_counter()
    try:
        txt=header(engine)
        for r in range(-1,repeats):
            model=name+'_'+str(r+1)
            txt+=f'\\timing on\n\\echo __BENCH__train_{r}\n'+train_statement(engine,method,meta,model,columns)+'\n\\timing off\n'
        last=name+'_'+str(repeats)
        expr=predict_expression(engine,method,meta,last,columns)
        for r in range(-1,repeats):
            txt+=f'\\timing on\n\\echo __BENCH__predict_{r}\nSELECT sum(({expr})::double precision) FROM public.test_set;\n\\timing off\n'
        text=sql(engine,txt,case/(prefix+'-timings.log'),timeout=120*(repeats+2))
        ts=timings(text)
        expr=predict_expression(engine,method,meta,last,columns)
        path=f'/bench/{case.name}/{prefix}-predictions.csv'
        sql(engine,header(engine)+f'COPY (SELECT id,{expr} FROM public.test_set ORDER BY id) TO {quote(path)} CSV;',case/(prefix+'-prediction-export.log'))
        a=np.loadtxt(case/(prefix+'-predictions.csv'),delimiter=',',ndmin=2)
        assert np.array_equal(a[:,0],np.arange(meta['n_test'])), 'prediction ID alignment'
        raw=a[:,1]; pred=(raw>=0.5).astype(int) if method=='xgboost' and meta['task']=='binary' else raw
        score=raw if method=='xgboost' and meta['task']=='binary' else None
        if engine=='pg' and meta['task']=='binary' and method!='xgboost':
            sexpr=f'db4ai.predict_float8({quote(last)},ARRAY[{columns}]::float8[])'
            sp=f'/bench/{case.name}/{prefix}-scores.csv'
            sql(engine,header(engine)+f'COPY (SELECT id,{sexpr} FROM public.test_set ORDER BY id) TO {quote(sp)} CSV;')
            score=np.loadtxt(case/(prefix+'-scores.csv'),delimiter=',',ndmin=2)[:,1]
        mm=metrics(meta,pred,score)
        train_metrics={}; model_bytes=None
        try:
            if engine=='pg':
                text=sql(engine,f'SELECT db4ai.explain_model({quote(last)});')
                explanation=json.loads(text.strip()); dump(case/(prefix+'-model.json'),explanation)
                train_metrics=explanation['metrics']; model_bytes=explanation['model_bytes']
                if method=='xgboost':
                    p=f'/bench/{case.name}/pg_xgboost.hex'
                    sql(engine,f'COPY (SELECT encode(db4ai.export_model({quote(last)}),\'hex\')) TO {quote(p)};')
            else:
                text=sql(engine,f"SELECT iterations FROM gs_model_warehouse WHERE modelname={quote(last)};")
                train_metrics={'iterations':int(text.strip())}
                (case/(prefix+'-explain.log')).write_text(sql(engine,'EXPLAIN MODEL '+last+';'))
        except Exception as exc:
            train_metrics['metadata_error']=str(exc)
        for r in range(repeats):
            records.append(dict(engine=engine,method=method,dataset=meta['dataset'],seed=meta['seed'],repeat=r,status='ok',
                fit_ms=ts['train_'+str(r)],predict_ms=ts['predict_'+str(r)],
                fit_scope='server-local client statement wall time including model persistence; excludes client/container startup',
                predict_scope='one SQL statement: model lookup/load + full holdout scan + aggregate',
                n_train=meta['n_train'],n_test=meta['n_test'],parameters=config(method,meta['seed']),
                training_metrics=train_metrics,model_bytes=model_bytes,**mm))
    except Exception as exc:
        traceback.print_exc()
        records.append(dict(engine=engine,method=method,dataset=meta['dataset'],seed=meta['seed'],status='error',error=str(exc)))
    dump(case/(prefix+'-results.json'),records)
    print('DB',engine,case.name,method,'wall',round(time.perf_counter()-start,2),records[-1],flush=True)
    return records


def start_servers(root,cpu,pg_image,og_image):
    for engine in ('pg','og'): subprocess.run(['docker','rm','-f','bench-'+engine],capture_output=True)
    limits=['--cpuset-cpus',str(cpu),'--cpus','1','--memory','4g','--shm-size','1g','--network','none',
            '-v',str(root)+':/bench','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1']
    command(['docker','run','-d','--name','bench-pg',*limits,'-e','POSTGRES_PASSWORD=BenchOnly_2026!',pg_image,
             '-c','shared_buffers=256MB','-c','max_parallel_workers_per_gather=0','-c','jit=off'],timeout=120)
    command(['docker','run','-d','--name','bench-og',*limits,'--cap-add=SYS_PTRACE','--security-opt','seccomp=unconfined',
             '-e','GS_PASSWORD=BenchOnly_2026!',og_image],timeout=120)
    for engine in ('pg','og'):
        error=None
        for _ in range(90):
            try:
                version=sql(engine,'SELECT version();'); (root/(engine+'-version.txt')).write_text(version); break
            except Exception as exc: error=exc; time.sleep(2)
        else: raise RuntimeError(f'{engine} failed to start: {error}')
    sql('pg','CREATE EXTENSION IF NOT EXISTS pg_db4ai;')
    for engine in ('pg','og'):
        (root/(engine+'-settings.txt')).write_text(sql(engine,'SELECT name,setting FROM pg_settings ORDER BY name;'))
        (root/(engine+'-container.json')).write_text(command(['docker','inspect','bench-'+engine]))
    (root/'og-xgboost-library.txt').write_text(command(['docker','exec','bench-og','bash','-lc',
       'sha256sum /usr/local/opengauss/lib/libxgboost.so; strings /usr/local/opengauss/lib/libxgboost.so | grep -m5 -E "XGBoost|version" || true']))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seeds',default='42,2024,7'); p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--datasets',default=','.join(SOURCES)); p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)))
    p.add_argument('--pg-image',default=PG_IMAGE); p.add_argument('--og-image',default=OG_IMAGE)
    args=p.parse_args(); root=args.out.resolve(); root.mkdir(parents=True,exist_ok=True); os.chmod(root,0o777)
    if args.repeats<1: raise ValueError('repeats must be positive')
    names=args.datasets.split(','); seeds=list(map(int,args.seeds.split(',')))
    environment=dict(cpu=args.cpu,seeds=seeds,repeats=args.repeats,pg_image=args.pg_image,og_image=args.og_image,
                     source_commit=command(['git','-C',str(ROOT),'rev-parse','HEAD']).strip(),
                     cpuinfo=command(['lscpu']),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                     protocol='group-disjoint holdouts, fixed configs, no tuning; one CPU; sequential methods; warm code/data')
    dump(root/'environment.json',environment)
    records=[]
    try:
        cases=prepare(root,names,seeds)
        for path in root.rglob('*'):
            if path.is_dir(): path.chmod(0o777)
            else: path.chmod(0o666)
        start_servers(root,args.cpu,args.pg_image,args.og_image)
        for case_name in cases:
            case=root/case_name; meta=json.loads((case/'meta.json').read_text()); meta['_case']=case
            for engine in ('pg','og'): dump(case/(engine+'-load.json'),load_case(engine,case,meta))
            tasks=[(e,m) for m in cases_methods(meta) for e in ('pg','og')]
            np.random.default_rng(meta['seed']).shuffle(tasks)
            for engine,method in tasks:
                records+=benchmark_db(engine,case,meta,method,args.repeats)
                dump(root/'results.json',records)
            log=command(['docker','run','--rm','--network','none','--cpuset-cpus',str(args.cpu),'--cpus','1','--memory','4g',
                         '-v',str(root)+':/bench','--entrypoint','/opt/bench/bin/python',args.pg_image,
                         '/src/benchmarks/tabular/python_baseline.py','/bench/'+case_name,'--repeats',str(args.repeats)],timeout=600)
            (case/'python.log').write_text(log)
            rows=json.loads((case/'python-results.json').read_text()); records+=rows; dump(root/'results.json',records)
            print('COMPLETE CASE',case_name,flush=True)
    finally:
        for engine in ('pg','og'):
            r=subprocess.run(['docker','logs','bench-'+engine],text=True,capture_output=True)
            (root/(engine+'-server.log')).write_text(r.stdout+r.stderr)
    import pandas as pd
    flat=[{k:v for k,v in row.items() if not isinstance(v,(dict,list))} for row in records]
    pd.DataFrame(flat).to_csv(root/'results.csv',index=False)
    print('BENCHMARK FINISHED',root,len(records),'records',flush=True)
    failures=[r for r in records if r['status']!='ok']
    dump(root/'completion.json',dict(records=len(records),failed_cases=len(failures),complete=not failures))
    if failures: raise SystemExit(f'{len(failures)} engine/method cases failed; raw errors retained in results.json')

if __name__=='__main__': main()
