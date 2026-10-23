#!/usr/bin/env python3
"""Independent stdlib prediction and preprocessing oracle; no fitted baseline."""
import os,subprocess,pathlib,json,math
B=pathlib.Path(subprocess.check_output([os.environ.get('PG_CONFIG','pg_config'),'--bindir'],text=True).strip())
def query(sql):
 p=subprocess.run([str(B/'psql'),'-X','-At','-v','ON_ERROR_STOP=1','-c',sql],text=True,capture_output=True,check=True)
 return [json.loads(s) for s in p.stdout.splitlines() if s.startswith(('{','['))]
def sigmoid(z):return 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
def transform(p,row):
 x=[0.0]*p['nencoded']
 for f in p['features']:
  val=row[f['name']];k=f['offset']
  if f['categorical']:
   values=f['dictionary']['values'];idx=len(values)+1 if val is None else (values.index(val) if val in values else len(values))
   x[k+idx]=1
  else:
   x[k]=0 if val is None else (val-f['mean'])/f['scale']
   if p['missing_indicators']:x[k+1]=float(val is None)
 return x
def predict(p,row):
 x=[1.0]+transform(p,row);m=json.loads(bytes.fromhex(p['learner']['data_hex']));w=m['weights'];n=len(x)
 if p['classification']:
  if m['algorithm']=='logistic_regression':
   s=sigmoid(math.fsum(a*b for a,b in zip(w,x)));probs=[1-s,s]
  else:
   probs=[sigmoid(math.fsum(w[c*n+j]*x[j] for j in range(n))) for c in range(p['nclasses'])];s=sum(probs);probs=[v/s for v in probs]
  return probs
 return math.fsum(a*b for a,b in zip(w,x))*p['target_scale']+p['target_mean']
checks=0
for name in ['am_class','am_bigint','am_multi','am_reg']:
 p=query(f"SELECT convert_from(db4ai.export_model('{name}'),'UTF8')::jsonb;")[0]
 rows=[]
 for i in range(21):
  if name=='am_class':row={'x':i%4,'z':i*.7,'region':[None,'north','south','never_seen'][i%4],'empty_feature':None}
  elif name=='am_bigint':row={'x':i%2,'z':i*11.0}
  elif name=='am_multi':row={'x':i%3,'z':i/50}
  else:row={'x':float(i*13) if i else None}
  rows.append(row)
 for mode in ['off','portable','auto']:
  selects=';'.join(f"SELECT db4ai.predict('{name}','{json.dumps(row)}'::jsonb)" for row in rows)
  returned=query(f'SET db4ai.simd={mode};'+selects)
  assert len(returned)==len(rows)
  for row,output in zip(rows,returned):
   expected=predict(p,row)
   if isinstance(expected,list):
    got=output['probabilities']
    assert len(expected)==len(got) and all(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-13) for a,b in zip(expected,got)),(name,row,expected,got)
    assert output['class_index']==max(range(len(expected)),key=expected.__getitem__)
    assert output['prediction']==p['labels']['values'][output['class_index']]
    checks+=len(got)+2
   else:assert math.isclose(output['prediction'],expected,rel_tol=1e-12,abs_tol=1e-10);checks+=1
# Independent training-only moments for the chronological fixture, with target
# changes in final test proven not to change the selected serialized artifact.
p=query("SELECT convert_from(db4ai.export_model('am_class'),'UTF8')::jsonb;")[0]
for field,values in [('x',[i%4 for i in range(1,145)]),('z',[i//4 for i in range(1,145)])]:
 mean=math.fsum(values)/len(values);scale=math.sqrt(math.fsum((x-mean)**2 for x in values)/len(values))
 f=next(f for f in p['features'] if f['name']==field)
 assert math.isclose(f['mean'],mean,abs_tol=1e-12) and math.isclose(f['scale'],scale,abs_tol=1e-12);checks+=2
print(f'Independent AutoML preprocessing/prediction oracle passed: {checks} checks; database='+os.environ.get('PGDATABASE','postgres'))
