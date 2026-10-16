#!/usr/bin/env python3
"""Current-version shuffle ablation; no external baseline is run.
Predeclared stress design: label-ordered copies of frozen Banknote training rows,
unchanged holdout. Traces are independent fixed-budget fits, not resumed models.
"""
import argparse, json, io, re
from pathlib import Path
import numpy as np
from measure_current import run_sql, quote, settings, quality, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--psql',default='psql');p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    case=a.data/'banknote-42';meta=json.loads((case/'meta.json').read_text());z=np.load(case/'data.npz')
    declarations=','.join(f'x{i} float8' for i in range(meta['features']));cols=','.join(f'x{i}' for i in range(meta['features']))
    setup='DROP TABLE IF EXISTS shuffle_stress_train,shuffle_stress_test;\n'
    for split in ('train','test'):
        setup+=f'CREATE TABLE shuffle_stress_{split}(id integer,y integer,{declarations});\n'
        setup+=f'COPY shuffle_stress_{split} FROM STDIN CSV;\n'+(case/(split+'.csv')).read_text()+'\\.\n'
    run_sql(a.psql,setup)
    query='SELECT y,'+cols+' FROM shuffle_stress_train CROSS JOIN generate_series(1,16) AS repetition ORDER BY y,id,repetition'
    records=[]
    tasks=[(mode,epochs) for mode in ('none','batch','corgipile') for epochs in (1,5,20)]
    np.random.default_rng(42).shuffle(tasks)
    for mode,epochs in tasks:
        opts=dict(profile='opengauss',optimizer='gd',batch_size=128,learning_rate=.05,decay=1,
                  tolerance=0,max_iterations=epochs,seed=42,shuffle=mode)
        if mode=='batch':opts['shuffle_buffer_size']=16
        if mode=='corgipile':opts.update(shuffle_block_rows=256,shuffle_window_blocks=8)
        name=f'stress_{mode}_{epochs}';script=settings(256)
        for i in range(-1,a.repeats):
            key=name+str(i+1)
            script+=f'SELECT db4ai.drop_model({quote(key)});\n\\timing on\n'
            script+=f'SELECT db4ai.train({quote(key)},\'logistic_regression\',{quote(query)},{quote(json.dumps(opts))}::jsonb);\n\\timing off\n'
        text=run_sql(a.psql,script,a.out/f'{name}.log')
        times=[float(m.group(1)) for line in text.splitlines() if (m:=re.match(r'Time:\s*([\d.]+)\s*ms',line))][1:]
        last=name+str(a.repeats)
        model=json.loads(run_sql(a.psql,f'SELECT db4ai.explain_model({quote(last)});'))
        predtext=run_sql(a.psql,f'COPY (SELECT id,db4ai.predict_label({quote(last)},ARRAY[{cols}],NULL::integer) FROM shuffle_stress_test ORDER BY id) TO STDOUT CSV;')
        pred=np.loadtxt(io.StringIO(predtext),delimiter=',')[:,1]
        (a.out/f'{name}-predictions.csv').write_text(predtext)
        record=dict(mode=mode,epoch_budget=epochs,fit_ms=times,median_fit_ms=float(np.median(times)),
                    metrics=model['metrics'],quality=quality(z['y_test'],pred,meta))
        records.append(record);write(a.out/'results.json',records)
        print(mode,epochs,record['median_fit_ms'],model['metrics']['training_loss'],record['quality'],flush=True)
    write(a.out/'protocol.json',dict(dataset='Banknote frozen training rows, repeated sixteen times then sorted by label',
          source_train_X_sha256=meta['train_X_sha256'],original_training_rows=meta['n_train'],expanded_training_rows=meta['n_train']*16,
          test_rows=meta['n_test'],work_mem_kib=256,seed=42,iteration_budgets=[1,5,20],
          protocol='Fixed-parameter stagewise fits; not continuation checkpoints or independent generalization splits',
          external_baselines_executed=0))

if __name__=='__main__':main()
