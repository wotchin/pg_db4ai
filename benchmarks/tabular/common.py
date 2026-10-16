"""Public tabular benchmark: shared, leakage-safe data and metric definitions."""
from __future__ import annotations
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, mean_squared_error, mean_absolute_error, r2_score)

SOURCES = {
 'breast_cancer': ('https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic','binary'),
 'banknote': ('https://archive.ics.uci.edu/dataset/267/banknote+authentication','binary'),
 'wine': ('https://archive.ics.uci.edu/dataset/109/wine','multiclass'),
 'digits': ('https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits','multiclass'),
 'diabetes': ('https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html','regression'),
 'wine_red': ('https://archive.ics.uci.edu/dataset/186/wine+quality','regression'),
 'concrete': ('https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength','regression'),
 'california': ('https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html','regression'),
}

def dump(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def download(url: str, root: Path, name: str):
    p = root / name
    if not p.exists():
        error = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    p.write_bytes(r.read())
                break
            except Exception as exc:
                error = exc
                time.sleep(attempt + 1)
        if not p.exists():
            raise RuntimeError(f'download failed: {url}: {error}')
    return p.read_bytes()

def load_data(name: str, root: Path):
    if name in ('breast_cancer', 'wine', 'digits'):
        b = getattr(datasets, 'load_' + name)()
        return np.asarray(b.data, dtype=np.float64), np.asarray(b.target), list(b.feature_names)
    if name == 'diabetes':
        b = datasets.load_diabetes(scaled=False)
        return b.data, b.target, list(b.feature_names)
    if name == 'california':
        b = datasets.fetch_california_housing(data_home=str(root / 'sklearn'))
        return b.data, b.target, list(b.feature_names)
    urls = {
        'banknote': ('https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip', 'data_banknote_authentication.txt'),
        'wine_red': ('https://archive.ics.uci.edu/static/public/186/wine+quality.zip', 'winequality-red.csv'),
        'concrete': ('https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip', 'Concrete_Data.xls'),
    }
    url, member = urls[name]
    raw = download(url, root, name + '.zip')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        choices = [f for f in z.namelist() if f.lower().endswith(member.lower())]
        if len(choices) != 1:
            raise ValueError(f'unexpected dataset archive layout: {z.namelist()}')
        content = io.BytesIO(z.read(choices[0]))
        frame = pd.read_excel(content) if name == 'concrete' else pd.read_csv(
            content, sep=';' if name == 'wine_red' else ',', header=0 if name == 'wine_red' else None)
    return frame.iloc[:, :-1].to_numpy(dtype=np.float64), frame.iloc[:, -1].to_numpy(), [str(c) for c in frame.columns[:-1]]

def prepare(root: Path, names: list[str], seeds: list[int]):
    rawroot = root / 'raw'; rawroot.mkdir(parents=True, exist_ok=True)
    result = []
    for name in names:
        start = time.perf_counter()
        X, y, columns = load_data(name, rawroot)
        task = SOURCES[name][1]
        X = np.asarray(X, dtype='<f8'); y = np.asarray(y, dtype='<f8')
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError(f'{name}: non-finite raw data; no silent imputation')
        # Identical feature vectors are kept in the same fold, even if labels conflict.
        groups = pd.util.hash_pandas_object(pd.DataFrame(X), index=False).to_numpy()
        _, groups = np.unique(groups, return_inverse=True)
        for seed in seeds:
            if task == 'regression':
                tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X,y,groups))
            else:
                tr, te = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X,y,groups))
                assert set(y[tr]) == set(y[te]), 'each fold must contain all classes'
            assert not set(groups[tr]).intersection(groups[te])
            # Frozen train order is shuffled by split seed, shared across engines.
            np.random.default_rng(seed).shuffle(tr)
            scaler = StandardScaler().fit(X[tr])
            Xtr = scaler.transform(X[tr]); Xte = scaler.transform(X[te])
            ym, ys = (float(y[tr].mean()), float(y[tr].std()) or 1.0) if task == 'regression' else (0.0,1.0)
            ytr = (y[tr]-ym)/ys; yte = y[te]
            case = root / f'{name}-{seed}'; case.mkdir(exist_ok=True)
            np.savez(case/'data.npz', X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte, train_ids=tr, test_ids=te)
            for split, xx, yy in [('train',Xtr,ytr), ('test',Xte,(yte-ym)/ys)]:
                table = np.column_stack([np.arange(len(yy)),yy,xx])
                fmts = ['%d', '%.17g' if task=='regression' else '%d'] + ['%.17g']*X.shape[1]
                np.savetxt(case/(split+'.csv'),table,delimiter=',',fmt=fmts)
            meta = dict(dataset=name,task=task,seed=seed,rows=len(X),features=X.shape[1],
                        n_train=len(tr),n_test=len(te),n_classes=len(np.unique(y)) if task!='regression' else 0,
                        feature_names=columns,source_url=SOURCES[name][0],raw_X_sha256=digest(X),raw_y_sha256=digest(y),
                        train_ids_sha256=digest(tr),test_ids_sha256=digest(te),train_X_sha256=digest(Xtr),
                        y_mean=ym,y_scale=ys,x_mean=scaler.mean_.tolist(),x_scale=scaler.scale_.tolist(),
                        duplicate_feature_rows=len(X)-len(np.unique(groups)),
                        split='StratifiedGroupKFold first of 5' if task!='regression' else 'GroupShuffleSplit 80/20',
                        preprocessing='StandardScaler fit on train only; regression y standardized on train; metrics in original y units')
            dump(case/'meta.json',meta); result.append(str(case.name))
        print('prepared',name,X.shape,'seconds',round(time.perf_counter()-start,2),flush=True)
    hashes={str(p.relative_to(rawroot)):hashlib.sha256(p.read_bytes()).hexdigest() for p in rawroot.rglob('*') if p.is_file()}
    dump(root/'datasets.json',dict(cases=result,seeds=seeds,raw_file_sha256=hashes))
    return result

def metrics(meta, predictions, scores=None):
    case = np.load(meta['_case'] / 'data.npz') if '_case' in meta else None
    y = case['y_test']
    pred=np.asarray(predictions,dtype=float).reshape(-1)
    if len(pred)!=len(y) or not np.isfinite(pred).all():
        raise ValueError('prediction count/non-finite mismatch')
    if meta['task']=='regression':
        pred=pred*meta['y_scale']+meta['y_mean']
        return dict(rmse=float(np.sqrt(mean_squared_error(y,pred))),mae=float(mean_absolute_error(y,pred)),r2=float(r2_score(y,pred)))
    out=dict(accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
             macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)))
    if meta['task']=='binary' and scores is not None:
        out['roc_auc']=float(roc_auc_score(y,np.asarray(scores).reshape(-1)))
    return out
