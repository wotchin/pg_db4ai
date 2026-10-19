#!/usr/bin/env python3
"""Current-code SIMD ablation on immutable public splits; no external baseline runs.
Uses an isolated PG instance from PGHOST/PGPORT/PGUSER/PGDATABASE. SQL timings
include read/train/diagnostics/model persistence, or model load/scan/predict/SUM.
Modes alternate in seeded random order within each repetition. Same .so, host,
CPU, input order and parameters. OFF retains normal compiler autovectorization.
"""
from __future__ import annotations
import argparse, hashlib, io, json, os, re, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'native'))
from measure_current import options, algorithm, configurations, quality, quote, settings


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--psql',default='psql');ap.add_argument('--seeds',default='42,2024,7')
    ap.add_argument('--repeats',type=int,default=7)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.repeats<3:raise ValueError('at least three measured repeats required')
    def sql(text,log=None):
        r=subprocess.run([args.psql,'-X','-qAt','-v','ON_ERROR_STOP=1'],input=text,text=True,capture_output=True,timeout=300)
        if log:log.write_text(r.stdout+'\n-- STDERR --\n'+r.stderr);log.with_suffix('.sql').write_text(text)
        if r.returncode:raise RuntimeError(r.stderr+r.stdout[-2000:])
        return r.stdout
    def save(name,data):
        (args.out/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    def times(s):
        out={};key=None
        for line in s.splitlines():
            if line.startswith('__SIMD__'):key=line[8:]
            m=re.match(r'Time:\s*([\d.]+)\s*ms',line)
            if key and m:
                if key in out:raise AssertionError('duplicate timing marker')
                out[key]=float(m.group(1));key=None
        return out
    modes=['off','portable','auto'];results=[];env={}
    for mode in modes:
        env[mode]=json.loads(sql("SET db4ai.simd="+quote(mode)+"; SELECT db4ai.build_info();"))
    env.update(version=sql('SELECT version();').strip(),cpu=subprocess.run(['lscpu'],capture_output=True,text=True).stdout,
               repeats=args.repeats,seed_order=args.seeds,protocol='Same current binary; paired randomized mode order; 1 warmup + N repeats; SQL-local timing',
               external_baselines_executed=0,scalar_compiler_autovectorization_disabled=False,
               fp='local FP reductions reordered; contraction disabled; no fast-math',
               settings=sql('SHOW jit; SHOW max_parallel_workers_per_gather; SHOW shared_buffers;'))
    save('environment.json',env)
    datasets=['breast_cancer','banknote','wine','digits','diabetes','wine_red','concrete','california']
    for seed in map(int,args.seeds.split(',')):
      for dataset in datasets:
        case=args.data/f'{dataset}-{seed}';meta=json.loads((case/'meta.json').read_text());z=np.load(case/'data.npz')
        assert hashlib.sha256(np.ascontiguousarray(z['X_train']).tobytes()).hexdigest()==meta['train_X_sha256']
        dst=args.out/case.name;dst.mkdir(exist_ok=True)
        cols=','.join('x'+str(j) for j in range(meta['features']));decl=','.join('x'+str(j)+' float8' for j in range(meta['features']))
        typ='float8' if meta['task']=='regression' else 'integer'
        load='DROP TABLE IF EXISTS simd_train,simd_test;\n'
        for split in ['train','test']:
            load+=f'CREATE TABLE simd_{split}(id integer,y {typ},{decl});\nCOPY simd_{split} FROM STDIN CSV;\n'+(case/(split+'.csv')).read_text()+'\\.\n'+f'ANALYZE simd_{split};\n'
        sql(load)
        for method in configurations(meta):
            params=options(method,seed);query='SELECT y,'+cols+' FROM simd_train ORDER BY id'
            rng=np.random.default_rng(seed+71);text=settings(65536);orders=[]
            def model(mode):return f'simd_{dataset}_{seed}_{method}_{mode}'
            for rep in range(-1,args.repeats):
                order=list(modes);rng.shuffle(order);orders.append(order)
                for mode in order:
                    text+=f"SET db4ai.simd={quote(mode)}; SELECT db4ai.drop_model({quote(model(mode))});\n\\timing on\n\\echo __SIMD__fit_{mode}_{rep}\n"
                    text+=f'SELECT db4ai.train({quote(model(mode))},{quote(algorithm(method))},{quote(query)},{quote(json.dumps(params))}::jsonb);\n\\timing off\n'
            # All prediction timings use the SAME model trained in OFF mode.
            expr=f'db4ai.predict_float8({quote(model("off"))},ARRAY[{cols}]::float8[])'
            for rep in range(-1,args.repeats):
                order=list(modes);rng.shuffle(order)
                for mode in order:
                    text+=f'SET db4ai.simd={quote(mode)};\n\\timing on\n\\echo __SIMD__predict_{mode}_{rep}\nSELECT sum({expr}) FROM simd_test;\n\\timing off\n'
            timing=times(sql(text,dst/(method+'.log')))
            predictions={};fixed={};qualities={};metrics={}
            for mode in modes:
                fn='predict_float8' if meta['task']=='regression' else 'predict_label'
                predict=f'db4ai.{fn}({quote(model(mode))},ARRAY[{cols}]::float8[]'+(',NULL::integer)' if fn=='predict_label' else ')')
                raw=sql(f'SET db4ai.simd={quote(mode)}; COPY (SELECT id,{predict} FROM simd_test ORDER BY id) TO STDOUT CSV;')
                (dst/(method+'-'+mode+'-predictions.csv')).write_text(raw)
                a=np.loadtxt(io.StringIO(raw),delimiter=',',ndmin=2);assert np.array_equal(a[:,0],np.arange(meta['n_test']))
                predictions[mode]=a[:,1];qualities[mode]=quality(z['y_test'],a[:,1],meta)
                raw=sql(f'SET db4ai.simd={quote(mode)}; COPY (SELECT id,{expr} FROM simd_test ORDER BY id) TO STDOUT CSV;')
                (dst/(method+'-'+mode+'-fixed-model.csv')).write_text(raw)
                fixed[mode]=np.loadtxt(io.StringIO(raw),delimiter=',',ndmin=2)[:,1]
                m=json.loads(sql(f'SELECT db4ai.explain_model({quote(model(mode))});'))
                metrics[mode]=m['metrics'];(dst/(method+'-'+mode+'-model.json')).write_text(json.dumps(m,indent=2)+'\n')
            diffs={mode:float(np.max(np.abs(predictions[mode]-predictions['off']))) for mode in modes}
            fixed_diff={mode:float(np.max(np.abs(fixed[mode]-fixed['off']))) for mode in modes}
            # A changed class or material prediction is a regression, not a speedup.
            for mode in modes:
                assert np.isfinite(predictions[mode]).all()
                assert np.allclose(predictions[mode],predictions['off'],rtol=1e-9,atol=1e-10),(case.name,method,mode,diffs)
                assert np.allclose(fixed[mode],fixed['off'],rtol=1e-11,atol=1e-12),(case.name,method,mode,fixed_diff)
            record=dict(dataset=dataset,seed=seed,method=method,features=meta['features'],n_train=meta['n_train'],n_test=meta['n_test'],
                        parameters=params,train_X_sha256=meta['train_X_sha256'],train_order_sha256=meta['train_ids_sha256'],
                        mode_orders=orders,training_quality=qualities,training_metrics=metrics,
                        trained_model_max_abs_diff=diffs,same_model_max_abs_diff=fixed_diff)
            for operation in ['fit','predict']:
                for mode in modes:record[operation+'_'+mode+'_ms']=[timing[f'{operation}_{mode}_{r}'] for r in range(args.repeats)]
                for mode in ['portable','auto']:
                    ratios=np.array(record[operation+'_off_ms'])/np.array(record[operation+'_'+mode+'_ms'])
                    record[operation+'_'+mode+'_speedup']=float(np.median(ratios))
            results.append(record);save('results.json',results)
            print(case.name,method,'fit speedup',round(record['fit_auto_speedup'],3),'predict',round(record['predict_auto_speedup'],3),'delta',diffs['auto'],flush=True)
    save('completion.json',dict(complete=True,cases=len(results),numeric_quality_checks=len(results)*len(modes),external_baselines_executed=0))
if __name__=='__main__':main()
