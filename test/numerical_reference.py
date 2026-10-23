#!/usr/bin/env python3
"""Independent stdlib numerical checks, not an openGauss server parity claim.

RFF / random Maclaurin definitions: pinned openGauss matrix.cpp and matrix.h.
The only deliberate mapping correction is the homogeneous coef0=0 coefficient.
The test also recomputes inference from exported JSON, independent of C loading.
"""
import json
import math
import os
import subprocess

BIN = subprocess.check_output([os.environ.get('PG_CONFIG', 'pg_config'), '--bindir'], text=True).strip()

def sql(query):
    return subprocess.check_output([BIN + '/psql', '-X', '-At', '-v', 'ON_ERROR_STOP=1', '-c', query], text=True).strip()

def uniform48(seed):
    state = ((seed & 0xffffffff) << 16) | 0x330e
    while True:
        state = (state * 0x5deece66d + 11) & 0xffffffffffff
        yield state / 281474976710656.0

def close(a, b, label, tolerance=1e-11):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tolerance*max(1,abs(a),abs(b)), (label, a, b)

def compare(a, b, label):
    assert len(a) == len(b), label
    for i, (x, y) in enumerate(zip(a,b)):
        close(x,y,f'{label}[{i}]')

for name in ['svm_gaussian', 'svm_polynomial', 'svm_homogeneous']:
    model = json.loads(sql(f"SELECT public.model_json('{name}')"))
    options = json.loads(sql(f"SELECT hyperparameters FROM db4ai.pg_db4ai_models WHERE model_name='{name}'"))
    k, d = model['kernel_components'], model['n_features']
    rng = uniform48(options['seed'])
    if model['kernel'] == 1:
        reference = []
        while len(reference) < d*k:
            u = next(rng)
            while u <= 2**-23:
                u = next(rng)
            angle = 2*math.pi*next(rng)
            mag = math.sqrt(2*options['gamma'])*math.sqrt(-2*math.log(u))
            reference.extend([mag*math.cos(angle), mag*math.sin(angle)])
        compare(model['kernel_weights'], reference[:d*k], name+' Box-Muller')
        offset_rng = uniform48(options['seed']+1)
        compare(model['kernel_offsets'],[2*math.pi*next(offset_rng) for _ in range(k)],name+' offsets')
    else:
        repetitions, coefficients = [], []
        for _ in range(k):
            while True:
                u = next(rng)
                degree = int(math.log2(1/u)) if u else 0
                if 1 <= degree <= options['degree']:
                    break
            repetitions.append(degree)
            coefficients.append(math.sqrt(math.comb(options['degree'],degree)*options['coef0']**(options['degree']-degree)*2**(degree+1)))
        compare(model['kernel_repetitions'],repetitions,name+' repetitions')
        compare(model['kernel_coefs'],coefficients,name+' coefficients')
        signs = uniform48(options['seed'])
        compare(model['kernel_weights'],[-1 if next(signs)<0.5 else 1 for _ in range(sum(repetitions)*d)],name+' signs')
    for point in [[-.75,.25],[1.,-1.],[0.,0.],[2.,3.]]:
        transformed = [1.]
        if model['kernel'] == 1:
            for c in range(k):
                angle = model['kernel_offsets'][c]+sum(point[j]*model['kernel_weights'][j*k+c] for j in range(d))
                transformed.append(math.sqrt(2/k)*math.cos(angle))
        else:
            row = 0
            for c in range(k):
                product = 1.
                for _ in range(int(model['kernel_repetitions'][c])):
                    product *= sum(point[j]*model['kernel_weights'][row*d+j] for j in range(d))
                    row += 1
                transformed.append(product*model['kernel_coefs'][c]/math.sqrt(k))
        expected = sum(a*b for a,b in zip(transformed,model['weights']))
        actual = float(sql(f"SELECT db4ai.predict_float8('{name}',ARRAY[{','.join(map(str,point))}]::float8[])"))
        close(actual,expected,name+' exported-model inference')
    print('PASS:',name,'mapping and independent inference')

for name in ['pca', 'stochastic_pca', 'stochastic_center']:
    model = json.loads(sql(f"SELECT public.model_json('{name}')"))
    d, k = model['n_features'], model['n_outputs']
    w = [model['weights'][i*d:(i+1)*d] for i in range(k)]
    for i in range(k):
        for j in range(k):
            close(sum(a*b for a,b in zip(w[i],w[j])),float(i==j),name+' orthonormality')
    point=[.25,.5]
    expected = [sum((x-u)*v for x,u,v in zip(point,model['mean'],component)) for component in w]
    actual = json.loads(sql(f"SELECT to_json(db4ai.transform('{name}',ARRAY[.25,.5]::float8[]))"))
    compare(actual,expected,name+' exported PCA inference')
    print('PASS:',name,'orthonormality and independent inference')

print('All retained independent numerical reference checks passed.')
