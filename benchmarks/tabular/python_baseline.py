"""Measured Python estimators run inside the same pinned-CPU Docker image as PG."""
from __future__ import annotations
import argparse
import json
import pickle
import time
import warnings
from pathlib import Path
import numpy as np
import lightgbm as lgb
import sklearn
import xgboost as xgb
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
from common import metrics, dump


def run(case: Path, repeats: int):
    meta=json.loads((case/'meta.json').read_text()); meta['_case']=case
    a=np.load(case/'data.npz'); X=a['X_train']; Xt=a['X_test']; y=a['y_train']
    regression=meta['task']=='regression'; binary=meta['task']=='binary'
    if not regression: y=y.astype(int)
    result=[]
    methods=['sklearn_linear' if regression else 'sklearn_logistic','python_xgboost','lightgbm']
    if not regression: methods.append('sklearn_linear_svc')
    for method in methods:
        try:
            rows=[]; last_pred=None; drift=0.0; model=None
            for repeat in range(-1,repeats):
                params={}
                if method=='python_xgboost':
                    params=dict(objective='reg:squarederror' if regression else 'binary:logistic' if binary else 'multi:softprob',
                                tree_method='hist',max_depth=6,eta=0.1,reg_lambda=1.0,min_child_weight=1,
                                gamma=0,subsample=1,colsample_bytree=1,nthread=1,seed=meta['seed'])
                    if not regression and not binary: params['num_class']=meta['n_classes']
                    start=time.perf_counter()
                    dm=xgb.DMatrix(X.astype(np.float32),label=y.astype(np.float32),nthread=1)
                    model=xgb.train(params,dm,num_boost_round=100)
                    fit_ms=(time.perf_counter()-start)*1000
                    start=time.perf_counter(); payload=model.save_raw(raw_format='ubj'); save_ms=(time.perf_counter()-start)*1000
                    start=time.perf_counter(); loaded=xgb.Booster(); loaded.load_model(payload); loaded.set_param({'nthread':1})
                    load_ms=(time.perf_counter()-start)*1000
                    predict=lambda: loaded.inplace_predict(Xt)
                else:
                    if method=='sklearn_linear': model=LinearRegression(n_jobs=1)
                    elif method=='sklearn_logistic':
                        base=LogisticRegression(C=1,max_iter=1000,tol=1e-8,solver='lbfgs',random_state=meta['seed'])
                        model=OneVsRestClassifier(base,n_jobs=1) if not binary else base
                    elif method=='sklearn_linear_svc': model=LinearSVC(C=1,max_iter=10000,random_state=meta['seed'])
                    else:
                        cls=lgb.LGBMRegressor if regression else lgb.LGBMClassifier
                        model=cls(n_estimators=100,learning_rate=0.1,num_leaves=31,max_depth=6,n_jobs=1,
                                  random_state=meta['seed'],deterministic=True,force_col_wise=True,verbosity=-1)
                    params={k:v for k,v in model.get_params().items() if isinstance(v,(str,int,float,bool,type(None)))}
                    start=time.perf_counter(); model.fit(X,y); fit_ms=(time.perf_counter()-start)*1000
                    start=time.perf_counter(); payload=pickle.dumps(model,protocol=5); save_ms=(time.perf_counter()-start)*1000
                    start=time.perf_counter(); loaded=pickle.loads(payload); load_ms=(time.perf_counter()-start)*1000
                    predict=(lambda:loaded.predict_proba(Xt)) if not regression and method!='sklearn_linear_svc' else (lambda:loaded.predict(Xt))
                predict()
                start=time.perf_counter(); raw=predict(); pred_ms=(time.perf_counter()-start)*1000
                if regression: pred=raw; score=None
                elif method=='sklearn_linear_svc': pred=raw; score=loaded.decision_function(Xt) if binary else None
                elif method=='python_xgboost':
                    pred=(raw>=0.5).astype(int) if binary else raw.argmax(axis=1); score=raw if binary else None
                else: pred=loaded.classes_[raw.argmax(axis=1)]; score=raw[:,1] if binary else None
                if last_pred is not None: drift=max(drift,float(np.max(np.abs(pred-last_pred))))
                last_pred=pred.copy()
                if repeat>=0:
                    row=dict(engine='python',method=method,dataset=meta['dataset'],seed=meta['seed'],repeat=repeat,status='ok',
                             fit_ms=fit_ms,serialize_ms=save_ms,fit_persist_ms=fit_ms+save_ms,load_ms=load_ms,
                             predict_ms=pred_ms,predict_scope='resident model, vectorized full holdout batch; excludes SQL scan',
                             n_train=len(X),n_test=len(Xt),model_bytes=len(payload),parameters=params,**metrics(meta,pred,score))
                    rows.append(row)
            np.save(case/(method+'-predictions.npy'),last_pred)
            for row in rows: row['repeat_prediction_max_abs_diff']=drift
            result.extend(rows)
            print(meta['dataset'],meta['seed'],method,rows[-1],flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            result.append(dict(engine='python',method=method,dataset=meta['dataset'],seed=meta['seed'],status='error',error=str(exc)))
    modelhex=case/'pg_xgboost.hex'
    if modelhex.exists():
        try:
            loaded=xgb.Booster(); start=time.perf_counter()
            loaded.load_model(bytearray.fromhex(modelhex.read_text().strip())); loaded.set_param({'nthread':1})
            load_ms=(time.perf_counter()-start)*1000
            t=time.perf_counter(); raw=loaded.inplace_predict(Xt); pred_ms=(time.perf_counter()-t)*1000
            pg=np.loadtxt(case/'pg_xgboost-predictions.csv',delimiter=',',ndmin=2)
            delta=float(np.max(np.abs(raw-pg[:,1])))
            dump(case/'xgboost_roundtrip.json',dict(max_abs_difference=delta,rows=len(raw),load_ms=load_ms,predict_ms=pred_ms,
                 build=xgb.build_info(),version=xgb.__version__))
            assert delta<1e-6, f'exported PG model prediction mismatch: {delta}'
        except Exception as exc:
            result.append(dict(engine='python',method='xgboost_roundtrip',dataset=meta['dataset'],seed=meta['seed'],status='error',error=str(exc)))
    dummy=np.full(len(Xt),0 if regression else int(np.bincount(y).argmax()))
    result.append(dict(engine='control',method='mean' if regression else 'majority',dataset=meta['dataset'],seed=meta['seed'],
                       status='ok',**metrics(meta,dummy)))
    dump(case/'python-results.json',result)
    dump(case/'python-versions.json',dict(sklearn=sklearn.__version__,lightgbm=lgb.__version__,xgboost=xgb.__version__,
                                         xgboost_build=xgb.build_info()))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('case',type=Path); p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args()
    with threadpool_limits(limits=1):
        run(a.case,a.repeats)
