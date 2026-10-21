#!/usr/bin/env python3
"""Supplementary current-code KMeans/PCA ablation on frozen Digits-42."""
from __future__ import annotations
import argparse,io,json,os,re,subprocess
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--psql',default='psql');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 def sql(text,log=None):
  r=subprocess.run([a.psql,'-X','-qAt','-v','ON_ERROR_STOP=1'],input=text,text=True,capture_output=True,timeout=300)
  if log:log.write_text(r.stdout+'\n-- STDERR --\n'+r.stderr);log.with_suffix('.sql').write_text(text)
  if r.returncode:raise RuntimeError(r.stderr)
  return r.stdout
 def q(s):return "'"+s.replace("'","''")+"'"
 case=a.data/'digits-42';meta=json.loads((case/'meta.json').read_text());cols=','.join(f'x{i}' for i in range(64));decl=','.join(f'x{i} float8' for i in range(64))
 text='DROP TABLE IF EXISTS simd_u_train,simd_u_test;\n'
 for s in ['train','test']:
  text+=f'CREATE TABLE simd_u_{s}(id integer,y integer,{decl});\nCOPY simd_u_{s} FROM STDIN CSV;\n'+(case/(s+'.csv')).read_text()+'\\.\n'+f'ANALYZE simd_u_{s};\n'
 sql(text);results=[]
 tasks=[('kmeans',dict(num_centroids=10,seeding_function='Random++',max_iterations=20,tolerance=0,seed=42)),('pca',dict(number_components=8,pca_solver='stochastic',center=True,max_iterations=20,tolerance=0,batch_size=128,seed=42))]
 for algo,opts in tasks:
  text="SET client_min_messages=warning; SET jit=off; SET max_parallel_workers_per_gather=0; SET db4ai.work_mem=65536;\n";rng=np.random.default_rng(413)
  for r in range(-1,7):
   order=['off','portable','auto'];rng.shuffle(order)
   for m in order:
    name=f'u_{algo}_{m}';text+=f'SET db4ai.simd={q(m)};SELECT db4ai.drop_model({q(name)});\n\\timing on\n\\echo __U__fit_{m}_{r}\nSELECT db4ai.train({q(name)},{q(algo)},{q("SELECT "+cols+" FROM simd_u_train ORDER BY id")},{q(json.dumps(opts))});\n\\timing off\n'
  if algo=='kmeans':expr=f'db4ai.predict_cluster(\'u_kmeans_off\',ARRAY[{cols}])::float8'
  else:expr=f'(db4ai.transform(\'u_pca_off\',ARRAY[{cols}]))[1]'
  for r in range(-1,7):
   order=['off','portable','auto'];rng.shuffle(order)
   for m in order:text+=f'SET db4ai.simd={q(m)};\n\\timing on\n\\echo __U__predict_{m}_{r}\nSELECT sum({expr}) FROM simd_u_test;\n\\timing off\n'
  raw=sql(text,a.out/(algo+'.log'));times={};marker=None
  for line in raw.splitlines():
   if line.startswith('__U__'):marker=line[5:]
   match=re.match(r'Time:\s*([\d.]+)\s*ms',line)
   if marker and match:times[marker]=float(match.group(1));marker=None
  pred={}
  for m in ['off','portable','auto']:
   if algo=='kmeans':expr=f'db4ai.predict_cluster({q("u_kmeans_"+m)},ARRAY[{cols}])'
   else:expr=','.join(f'(db4ai.transform({q("u_pca_"+m)},ARRAY[{cols}]))[{i+1}]' for i in range(8))
   raw=sql(f'SET db4ai.simd={q(m)}; COPY (SELECT id,{expr} FROM simd_u_test ORDER BY id) TO STDOUT CSV;');(a.out/(algo+'-'+m+'.csv')).write_text(raw);pred[m]=np.loadtxt(io.StringIO(raw),delimiter=',')[:,1:]
   assert np.array_equal(np.loadtxt(io.StringIO(raw),delimiter=',' )[:,0],np.arange(meta['n_test']))
  differences={m:float(np.max(np.abs(pred[m]-pred['off']))) for m in pred}
  for m in pred:assert np.allclose(pred[m],pred['off'],rtol=1e-7,atol=1e-8),(algo,m,differences)
  entry=dict(algorithm=algo,parameters=opts,dataset='digits',seed=42,features=64,train_X_sha256=meta['train_X_sha256'],prediction_max_abs_diff=differences)
  for op in ['fit','predict']:
   for m in pred:entry[op+'_'+m+'_ms']=[times[f'{op}_{m}_{r}'] for r in range(7)]
   entry[op+'_auto_speedup']=float(np.median(np.array(entry[op+'_off_ms'])/entry[op+'_auto_ms']))
  results.append(entry);(a.out/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(entry,flush=True)
if __name__=='__main__':main()
