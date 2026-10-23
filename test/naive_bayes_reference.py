#!/usr/bin/env python3
"""Independent Gaussian NB equations; no sklearn model and no old baseline run.
Two-pass math.fsum moments deliberately differ from the C Welford estimator.
Executed in a fresh backend and again after logical dump/restore.
"""
import json
import math
import os
import random
import subprocess

BIN = subprocess.check_output([os.environ.get('PG_CONFIG', 'pg_config'), '--bindir'], text=True).strip()
def sql(q):
    return subprocess.check_output([BIN+'/psql','-X','-Atq','-v','ON_ERROR_STOP=1'],input=q,text=True).strip()
def close(a,b,name,atol=1e-11,rtol=2e-10):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<=atol+rtol*max(abs(a),abs(b)),(name,a,b)
def reference(rows,labels,smoothing=1e-9,floor=1e-12,empirical=True):
    X=[r[1:] for r in rows];n=len(X);d=len(X[0]);groups=[[r[1:] for r in rows if str(r[0])==c] for c in labels]
    def moments(x):
        mean=[math.fsum(r[j] for r in x)/len(x) for j in range(d)]
        var=[math.fsum((r[j]-mean[j])**2 for r in x)/len(x) for j in range(d)]
        return mean,var
    _,global_var=moments(X);epsilon=max(floor,smoothing*max(global_var))
    means=[];variances=[]
    for g in groups:
        mu,var=moments(g);means.append(mu);variances.append([v+epsilon for v in var])
    priors=[len(g)/n if empirical else 1/len(groups) for g in groups]
    return means,variances,priors,epsilon

def posterior(x,means,variances,priors):
    scores=[math.log(p)-.5*math.fsum(math.log(2*math.pi)+math.log(v)+(a-mu)**2/v
            for a,mu,v in zip(x,m,vs)) for m,vs,p in zip(means,variances,priors)]
    maximum=max(scores);shift=[s-maximum for s in scores];normal=math.log(math.fsum(math.exp(s) for s in shift))
    lp=[s-normal for s in shift]
    return [math.exp(s) for s in lp],lp

rows=json.loads(sql("SELECT json_agg(r ORDER BY label,x,z) FROM (SELECT label,x,z FROM public.nb_reference)r;"))
rows=[[r['label'],r['x'],r['z']] for r in rows]

rng=random.Random(94617)
wide=[[f'c{i%3}']+[rng.gauss((i%3)*.7+j*.01,1+j*.02) for j in range(33)] for i in range(90)]
if sql("SELECT count(*) FROM db4ai.pg_db4ai_models WHERE model_name='nb_oracle_wide';")=='0':
    vals=','.join("('%s',ARRAY[%s]::float8[])"%(r[0],','.join(repr(x) for x in r[1:])) for r in wide)
    sql("CREATE TABLE public.nb_oracle_wide(label text,x float8[]); INSERT INTO public.nb_oracle_wide VALUES "+vals+";"
        "SELECT db4ai.train('nb_oracle_wide','gaussian_naive_bayes','SELECT label,x FROM public.nb_oracle_wide');")

checks=0
for name,rows,points in [('nb_reference',rows,[[x,z] for x in (-4.,-2.,0.,2.,4.) for z in (0.,2.,4.,6.)]),
                         ('nb_oracle_wide',wide,[r[1:] for r in wide[:12]])]:
    model=json.loads(sql(f"SELECT convert_from(db4ai.export_model('{name}'),'UTF8')::jsonb;"))
    labels=json.loads(sql(f"SELECT to_json(db4ai.class_labels('{name}'));"))
    mu,var,priors,epsilon=reference(rows,labels)
    close(model['variance_epsilon'],epsilon,name+' epsilon')
    for field,expected in [('means',[a for c in mu for a in c]),('variances',[a for c in var for a in c]),('class_priors',priors)]:
        assert len(model[field])==len(expected)
        for a,b in zip(model[field],expected):close(a,b,name+' '+field);checks+=1
    for x in points:
        arr='ARRAY['+','.join(repr(a) for a in x)+']::float8[]'
        p,lp=posterior(x,mu,var,priors)
        for mode in ('off','portable','auto'):
            actual=json.loads(sql(f"SET db4ai.simd='{mode}'; SELECT json_build_object('p',db4ai.predict_proba('{name}',{arr}),"
                                 f"'lp',db4ai.predict_log_proba('{name}',{arr}),'label',db4ai.predict_text('{name}',{arr}));"))
            for a,b in zip(actual['p'],p):close(a,b,name+' probability');checks+=1
            for a,b in zip(actual['lp'],lp):close(a,b,name+' log probability');checks+=1
            close(sum(actual['p']),1,name+' normalization')
            assert actual['label']==labels[max(range(len(lp)),key=lambda i:lp[i])]
    print('PASS independent Gaussian NB moments/posterior and SIMD modes:',name)
print(f'Gaussian NB oracle: {checks} numerical comparisons passed')
