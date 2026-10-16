#!/usr/bin/env python3
"""Untimed supplementary audit of executed OG iterations and model metadata.
Recreates the exact fixed datasets/options; hashes allow joining to measured
runs. This is not a substitute for timings or same-run model inspection.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import time
from common import SOURCES, prepare, dump
import run
from export_transport import require_final_server


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=a.out.resolve();root.mkdir(parents=True,exist_ok=True);root.chmod(0o777)
    cases=prepare(root,list(SOURCES),[42,2024,7])
    for path in root.rglob('*'):path.chmod(0o777 if path.is_dir() else 0o666)
    subprocess.run(['docker','rm','-f','bench-og'],capture_output=True)
    cpu=min(os.sched_getaffinity(0))
    run.command(['docker','run','-d','--name','bench-og','--network','none',
                 '--cpuset-cpus',str(cpu),'--cpus','1','--memory','4g','--shm-size','1g',
                 '--cap-add=SYS_PTRACE','--security-opt','seccomp=unconfined',
                 '-e','GS_PASSWORD=BenchOnly_2026!','-e','OMP_NUM_THREADS=1',
                 '-v',str(root)+':/bench','opengauss-db4ai-benchmark:local'])
    records=[]
    try:
        for attempt in range(90):
            try:
                require_final_server('og');version=run.sql('og','SELECT version();');break
            except Exception:time.sleep(2)
        else:raise RuntimeError('final openGauss server did not start')
        (root/'version.txt').write_text(version)
        for c in cases:
            case=root/c;meta=json.loads((case/'meta.json').read_text())
            run.load_case('og',case,meta)
            cols=','.join(f'x{i}' for i in range(meta['features']))
            for method in run.cases_methods(meta):
                name='inspect_'+method+'_'+c.replace('-','_')
                statement=run.train_statement('og',method,meta,name,cols)
                output=run.sql('og',run.header('og')+statement+'\n',case/(method+'-train.log'))
                # Catalog column is modelname, not the extension's model_name.
                metadata=run.sql('og',f"SELECT row_to_json(m) FROM (SELECT modelname,iterations,processedtuples,discardedtuples,exectime,hyperparametersnames,hyperparametersvalues FROM gs_model_warehouse WHERE modelname={run.quote(name)}) m;")
                record=dict(dataset=meta['dataset'],seed=meta['seed'],method=method,
                            train_X_sha256=meta['train_X_sha256'],train_ids_sha256=meta['train_ids_sha256'],
                            catalog=json.loads(metadata.strip()),source_sql=statement)
                records.append(record);dump(root/'executed-iterations.json',records)
                print(json.dumps(record),flush=True)
        dump(root/'audit-completion.json',dict(expected=78,actual=len(records),complete=len(records)==78,
             scope='supplementary untimed rerun with exact data/config hashes; not original timed instances'))
    finally:
        r=subprocess.run(['docker','logs','bench-og'],text=True,capture_output=True)
        (root/'server.log').write_text(r.stdout+r.stderr)
        subprocess.run(['docker','rm','-f','bench-og'],capture_output=True)

if __name__=='__main__':main()
