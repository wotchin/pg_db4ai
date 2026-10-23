#!/usr/bin/env python3
"""Cancellation, whole-call deadlines and admission limits. No external estimator."""
import os, pathlib, subprocess, time
BIN=pathlib.Path(subprocess.check_output([os.environ.get('PG_CONFIG','pg_config'),'--bindir'],text=True).strip())
def sql(text, ok=True):
    r=subprocess.run([str(BIN/'psql'),'-X','-At','-v','ON_ERROR_STOP=1','-c',text],text=True,capture_output=True,timeout=30)
    if (r.returncode==0)!=ok: raise AssertionError(r.stdout+'\n'+r.stderr)
    return r
for name,query,budget in [('am_input_timeout','SELECT 1 AS y,1 AS x FROM pg_sleep(5)',.05),
                          ('am_sort_timeout','SELECT i::float8 AS y,i::float8 AS x FROM generate_series(1,10000000)i ORDER BY random()',.05)]:
    start=time.monotonic()
    r=sql(f"SELECT db4ai.automl('{name}','regression',$q${query}$q$,'{{\"time_budget_seconds\":{budget}}}');",False)
    assert 'statement timeout' in r.stderr, r.stderr
    assert time.monotonic()-start<4, 'deadline not polled'
    assert sql(f"SELECT count(*) FROM db4ai.pg_db4ai_models WHERE model_name='{name}'").stdout.strip()=='0'
r=sql("SET statement_timeout='30ms'; SELECT db4ai.automl('am_parent_timeout','regression','SELECT 1 AS y,1 AS x FROM pg_sleep(5)','{\"time_budget_seconds\":10}');",False)
assert 'statement timeout' in r.stderr
# Catch the cancellation in a PL/pgSQL subtransaction, then a longer statement
# must not inherit our canceled timer or active-search global.
sql("""DO $$ BEGIN
 BEGIN PERFORM db4ai.automl('am_caught_timeout','regression','SELECT 1 AS y,1 AS x FROM pg_sleep(5)','{"time_budget_seconds":0.02}');
 EXCEPTION WHEN query_canceled THEN NULL; END;
 PERFORM pg_sleep(0.08);
 IF current_setting('statement_timeout') <> '0' OR current_setting('work_mem') <> '4MB' THEN RAISE EXCEPTION 'GUC leak'; END IF;
END $$;""")
# Full read of oversized input must fail; never turn a prefix into a sample.
for query,opts in [("SELECT i AS y,repeat('x',100000) AS cat,i AS x FROM generate_series(1,100)i",'{"max_input_mb":1}'),
                   ("SELECT i AS y,i AS x,i AS z,i AS a,i AS b FROM generate_series(1,20000)i",'{"max_encoded_mb":1}')]:
    r=sql(f"SELECT db4ai.automl('am_admission','regression',$q${query}$q$,'{opts}');",False)
    assert 'exceeds' in r.stderr
# Forced raw and encoded spills using current implementation only.
import json
cols=','.join(f'(i%{j+7})::float8 AS x{j}' for j in range(16))
r=sql(f"SELECT db4ai.automl('am_spill','regression','SELECT i::float8 AS y,i::float8 AS trend,{cols} FROM generate_series(1,40000)i','{{\"memory_mb\":16,\"max_trials\":1,\"pilot_rows\":256,\"use_xgboost\":false,\"time_budget_seconds\":10}}');")
report=json.loads(r.stdout.strip())
assert report['status']=='ready' and report['input_spilled'] and report['encoded_spilled'],report
assert report['source_executions']==1 and report['selected_training_rows']==256
# Recursion is a clear error, not nested timer/global mutation.
r=sql("""SELECT db4ai.automl('am_recursive','regression',
$q$SELECT (db4ai.automl('am_inner','regression','SELECT i AS y,i AS x FROM generate_series(1,100)i')->>'source_rows')::float8 AS y,i AS x FROM generate_series(1,100)i$q$);""",False)
assert 'recursively' in r.stderr,r.stderr
sql("SELECT public.assert_true(abs(db4ai.predict_float8('am_reg',ARRAY[100.0])-203)<1,'AutoML usable after canceled trials');")
print('AutoML deadline, parent timer, nested invocation, admission and forced-spill tests passed')
