#!/usr/bin/env python3
"""Paired, same-host A/B for the gbtree in-place predictor.
Both images train the same fixed model on identical real data. Identical UBJSON
and all-row predictions are prerequisites for reporting any timing ratio.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np
from common import dump
from run import command, quote, timings, train_statement


def sql(name: str, text: str, log: Path | None = None):
    args=['docker','exec','-i','-u','postgres',name,'psql','-d','postgres','-X','-q','-A','-t','-v','ON_ERROR_STOP=1']
    r=subprocess.run(args,input=text,text=True,capture_output=True,timeout=180)
    if log:
        log.write_text(r.stdout+'\n--- STDERR ---\n'+r.stderr)
        log.with_suffix('.sql').write_text(text)
    if r.returncode: raise RuntimeError(f'{name}: {r.stderr}\n{r.stdout}')
    return r.stdout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--before',default='pg-db4ai-before:local');p.add_argument('--after',default='pg-db4ai-benchmark:local')
    p.add_argument('--cpu',type=int,default=min(os.sched_getaffinity(0)));p.add_argument('--repeats',type=int,default=11)
    args=p.parse_args(); data=args.data.resolve(); out=args.out.resolve(); out.mkdir(parents=True,exist_ok=True);out.chmod(0o777)
    if args.repeats<3: raise ValueError('use at least three paired repetitions')
    # Stop only the named ephemeral benchmark containers, not unrelated services.
    for name in ('bench-pg','bench-og','bench-ab-before','bench-ab-after'):
        subprocess.run(['docker','rm','-f',name],capture_output=True)
    images={'before':args.before,'after':args.after}; results=[]
    settings="SET client_min_messages=warning; SET jit=off; SET max_parallel_workers_per_gather=0; SET work_mem='64MB';\n"
    try:
        for label,image in images.items():
            name='bench-ab-'+label
            command(['docker','run','-d','--name',name,'--network','none','--cpuset-cpus',str(args.cpu),'--cpus','1',
                     '--memory','4g','--shm-size','1g','-e','POSTGRES_PASSWORD=BenchOnly_2026!',
                     '-v',str(data)+':/bench:ro','-v',str(out)+':/ab',image,
                     '-c','shared_buffers=256MB','-c','jit=off','-c','max_parallel_workers_per_gather=0'])
            for attempt in range(60):
                try: sql(name,'SELECT 1;'); break
                except Exception: time.sleep(1)
            else: raise RuntimeError(name+' did not start')
            sql(name,'CREATE EXTENSION pg_db4ai;')
            (out/(label+'-container.json')).write_text(command(['docker','inspect',name]))
            (out/(label+'-version.txt')).write_text(sql(name,'SELECT version();'))
        cases=sorted(c for c in data.iterdir() if c.is_dir() and c.name.endswith('-42') and (c/'pg_xgboost.hex').exists())
        if len(cases)!=6: raise ValueError(f'expected six binary/regression cases, got {len(cases)}')
        for case in cases:
            meta=json.loads((case/'meta.json').read_text()); dst=out/case.name;dst.mkdir();dst.chmod(0o777)
            columns=','.join(f'x{i}' for i in range(meta['features']))
            declarations=','.join(f'x{i} double precision' for i in range(meta['features']))
            target='double precision' if meta['task']=='regression' else 'integer'
            for label in images:
                name='bench-ab-'+label
                text=settings+'DROP TABLE IF EXISTS train_set,test_set;\n'
                for split in ('train','test'):
                    text+=f'CREATE TABLE {split}_set(id integer,y {target},{declarations});\n'
                    text+=f"COPY {split}_set FROM '/bench/{case.name}/{split}.csv' CSV; ANALYZE {split}_set;\n"
                text+=train_statement('pg','xgboost',meta,meta['dataset'],columns)+'\n'
                expr=f'db4ai.predict_float8({quote(meta["dataset"])},ARRAY[{columns}]::float8[])'
                text+=f"COPY (SELECT encode(db4ai.export_model({quote(meta['dataset'])}),'hex')) TO '/ab/{case.name}/{label}-model.hex';\n"
                text+=f"COPY (SELECT id,{expr} FROM test_set ORDER BY id) TO '/ab/{case.name}/{label}-predictions.csv' CSV;\n"
                sql(name,text,dst/(label+'-setup.log'))
            payload=[bytes.fromhex((dst/(label+'-model.hex')).read_text().strip()) for label in images]
            if payload[0]!=payload[1]: raise AssertionError('model bytes differ: '+case.name)
            predictions=[np.loadtxt(dst/(label+'-predictions.csv'),delimiter=',',ndmin=2) for label in images]
            delta=float(np.max(np.abs(predictions[0]-predictions[1])))
            if delta!=0: raise AssertionError(f'prediction differences: {case.name}: {delta}')
            # Every trial starts a fresh backend on each side, intentionally
            # including model loading but excluding psql/docker process startup.
            query=settings+f'\\timing on\n\\echo __BENCH__prediction\nSELECT sum({expr}) FROM test_set;\n'
            rng=np.random.default_rng(314159)
            trials=[]
            for repeat in range(-2,args.repeats):
                order=list(images);rng.shuffle(order); row={'repeat':repeat,'order':order}
                for label in order:
                    output=sql('bench-ab-'+label,query,dst/(f'{label}-{repeat}.log'))
                    row[label+'_ms']=timings(output)['prediction']
                if repeat>=0: trials.append(row)
            before=np.array([r['before_ms'] for r in trials]);after=np.array([r['after_ms'] for r in trials])
            record=dict(dataset=meta['dataset'],n_test=meta['n_test'],features=meta['features'],
                        model_sha256=hashlib.sha256(payload[0]).hexdigest(),prediction_max_abs_diff=delta,
                        before_median_ms=float(np.median(before)),after_median_ms=float(np.median(after)),
                        speedup=float(np.median(before)/np.median(after)),
                        before_p25_ms=float(np.quantile(before,.25)),before_p75_ms=float(np.quantile(before,.75)),
                        after_p25_ms=float(np.quantile(after,.25)),after_p75_ms=float(np.quantile(after,.75)),
                        scope='same CPU, new backend per trial; model load + full test-table predict/sum; excludes client startup',
                        trials=trials)
            results.append(record);dump(out/'results.json',results)
            print(record,flush=True)
        dump(out/'environment.json',dict(cpu=args.cpu,before=args.before,after=args.after,repeats=args.repeats,
                                         before_source='bde6ee99fc14ad0b2e7cc8e27868470702fe78f8',
                                         after_source=command(['git','rev-parse','HEAD']).strip(),cpuinfo=command(['lscpu'])))
    finally:
        for label in images:
            r=subprocess.run(['docker','logs','bench-ab-'+label],text=True,capture_output=True)
            (out/(label+'-server.log')).write_text(r.stdout+r.stderr)
            subprocess.run(['docker','rm','-f','bench-ab-'+label],capture_output=True)
    print('PAIRED A/B COMPLETE: identical models and predictions in all cases',flush=True)

if __name__=='__main__': main()
