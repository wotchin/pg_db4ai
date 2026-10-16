FILES['test/corgipile_ddl.sql'] = {'old': None, 'new': '1259b0072c8ba1b41ccd200fbed465b2cda427b6', 'base': 'test/corgipile_ddl.sql', 'mode': '100644','edits':[
(0,0,r'''\set ON_ERROR_STOP 1
-- Functional audit additions: explicit DDL, diagnostics, dense replay and CorgiPile.
SELECT public.assert_true((SELECT extversion='0.3.0' FROM pg_extension WHERE extname='pg_db4ai'),'0.3 installed');
SELECT db4ai.execute_ddl($ddl$
 CREATE /* nested /* comments */ remain lexical */ MODEL "模型 DDL" USING linear_regression
 FEATURES x, x*x TARGET y FROM public.regression ORDER BY x
 WITH max_iterations=1500, learning_rate=.15, tolerance=1e-12, decay=1.0;
$ddl$);
SELECT public.assert_true(abs(db4ai.predict_float8('模型 DDL',ARRAY[.25,.0625])-1.3125)<0.001,'quoted UTF-8 DDL model and expressions');
SELECT public.assert_true(db4ai.execute_ddl('EXPLAIN MODEL "模型 DDL"')->>'algorithm'='linear_regression','EXPLAIN model DDL');
SELECT public.assert_true(db4ai.parse_ddl($d$CREATE MODEL x USING kmeans FEATURES "WITH" FROM (SELECT 1 AS "WITH") s WITH num_centroids=1$d$)->>'command'='CREATE MODEL','quoted WITH is not an option delimiter');
SELECT db4ai.execute_ddl($d$CREATE MODEL ddl_kmeans USING kmeans FEATURES "WITH" FROM (SELECT 1 AS "WITH" UNION ALL SELECT 2) s WITH num_centroids=1$d$);
SELECT public.assert_true(jsonb_array_length(db4ai.explain_model('ddl_kmeans')->'metrics'->'centroids')=1,'centroid diagnostics exposed');
SELECT public.assert_true(jsonb_array_length(db4ai.explain_model('pca')->'metrics'->'principal_components')=1,'PCA component diagnostics exposed');
SELECT public.expect_error($q$SELECT db4ai.execute_ddl('CREATE MODEL a USING linear_regression FEATURES x TARGET y,z FROM public.regression')$q$,'42601');
SELECT public.expect_error($q$SELECT db4ai.execute_ddl('CREATE MODEL a USING linear_regression FEATURES x TARGET y FROM public.regression; DROP TABLE public.regression')$q$,'42601');
SELECT public.expect_error($q$SELECT db4ai.execute_ddl('CREATE MODEL a USING linear_regression FEATURES x TARGET y FROM public.regression WITH seed=1, seed=2')$q$,'42601');
SELECT public.expect_error($q$SELECT db4ai.execute_ddl('CREATE MODEL a USING linear_regression FEATURES x TARGET y FROM public.regression WITH seed=(SELECT 1)')$q$,'42601');
SELECT public.expect_error($q$SELECT db4ai.execute_ddl('CREATE MODEL a USING linear_regression FEATURES x TARGET y FROM public.regression WITH verbose=+true')$q$,'42601');
SELECT public.expect_error($q$SELECT db4ai.execute_ddl('CREATE MODEL a USING linear_regression FEATURES x TARGET y FROM public.regression WITH seed=1,')$q$,'42601');
SELECT public.expect_error($q$SELECT db4ai.execute_ddl('EXPLAIN MODEL missing_model')$q$,'42704');
SELECT public.expect_error($q$SELECT db4ai.execute_ddl('DROP MODEL missing_model')$q$,'42704');
SELECT db4ai.execute_ddl('DROP MODEL IF EXISTS missing_model');
SELECT public.expect_error($q$SELECT db4ai.execute_ddl('CREATE MODEL bad_ddl USING linear_regression FEATURES x TARGET y FROM public.regression FOR UPDATE')$q$,'25006');
BEGIN;
SELECT db4ai.execute_ddl('CREATE MODEL ddl_rollback USING linear_regression FEATURES x TARGET y FROM public.regression');
ROLLBACK;
SELECT public.assert_true(db4ai.explain_model('ddl_rollback') IS NULL,'DDL rollback');
CREATE SEQUENCE public.ddl_reads;
SELECT db4ai.parse_ddl($d$CREATE MODEL dry_run USING linear_regression FEATURES nextval('public.ddl_reads') TARGET y FROM public.regression$d$);
SELECT public.assert_true((SELECT NOT is_called FROM public.ddl_reads),'parse_ddl does not execute source expressions');
SET ROLE db4ai_bob;
SELECT public.expect_error($q$SELECT db4ai.execute_ddl('CREATE MODEL no_priv USING linear_regression FEATURES x TARGET y FROM public.private_training')$q$,'42P01');
SELECT public.assert_true(db4ai.explain_model('模型 DDL') IS NULL,'model DDL does not bypass role isolation');
RESET ROLE;
REVOKE EXECUTE ON FUNCTION db4ai.train(text,text,text,jsonb,text) FROM PUBLIC;
SET ROLE db4ai_alice;
SELECT public.expect_error($q$SELECT db4ai.execute_ddl('CREATE MODEL denied_train USING linear_regression FEATURES x TARGET y FROM public.regression')$q$,'42501');
RESET ROLE;
GRANT EXECUTE ON FUNCTION db4ai.train(text,text,text,jsonb,text) TO PUBLIC;

-- Every actual iterator emits each index once, including partial final blocks.
DO $$ DECLARE mode text; epoch int; n int; a bigint[]; b bigint[]; BEGIN
 FOREACH mode IN ARRAY ARRAY['none','batch','row','corgipile'] LOOP
  FOREACH n IN ARRAY ARRAY[0,1,7,128,129,1009] LOOP
   FOREACH epoch IN ARRAY ARRAY[0,1,2,2147483647] LOOP
    IF mode='corgipile' THEN
     a=db4ai.shuffle_indices(n,jsonb_build_object('shuffle',mode,'batch_size',17,'shuffle_block_rows',23,'shuffle_window_blocks',3,'seed',42),epoch);
     b=db4ai.shuffle_indices(n,jsonb_build_object('shuffle',mode,'batch_size',17,'shuffle_block_rows',23,'shuffle_window_blocks',3,'seed',42),epoch);
    ELSE
     a=db4ai.shuffle_indices(n,jsonb_build_object('shuffle',mode,'batch_size',17,'shuffle_buffer_size',3,'seed',42),epoch);
     b=db4ai.shuffle_indices(n,jsonb_build_object('shuffle',mode,'batch_size',17,'shuffle_buffer_size',3,'seed',42),epoch);
    END IF;
    PERFORM public.assert_true(a=b,'seed/epoch reproducibility');
    PERFORM public.assert_true((SELECT count(*)=n AND count(DISTINCT x)=n AND coalesce(min(x),0)=0 AND coalesce(max(x),-1)=n-1 FROM unnest(a)t(x)),'no duplicate/dropped/out-of-range rows');
   END LOOP;
  END LOOP;
 END LOOP;
END $$;
SELECT public.assert_true(db4ai.shuffle_indices(100,'{"shuffle":"corgipile","shuffle_block_rows":7,"shuffle_window_blocks":3,"shuffle_first_epoch":false}',0)=(SELECT array_agg(i::bigint) FROM generate_series(0,99)i),'explicit sequential first epoch');
-- Each output window is a union of complete randomly selected source blocks.
WITH t AS (SELECT x,ord FROM unnest(db4ai.shuffle_indices(1000,'{"shuffle":"corgipile","shuffle_block_rows":10,"shuffle_window_blocks":3}',1)) WITH ORDINALITY t(x,ord)),
 windows AS (SELECT (ord-1)/30 AS w,count(DISTINCT x/10) AS blocks FROM t GROUP BY 1)
SELECT public.assert_true(bool_and(blocks=CASE WHEN w=33 THEN 1 ELSE 3 END),'two-level complete-block windows') FROM windows;
SELECT public.assert_true(db4ai.shuffle_indices(1000,'{"shuffle":"corgipile","shuffle_block_rows":10,"shuffle_window_blocks":3}',1)<>db4ai.shuffle_indices(1000,'{"shuffle":"corgipile","shuffle_block_rows":10,"shuffle_window_blocks":3}',2),'new epoch reshuffles');

-- Cache optimization must not change learned weights for unchanged batch order.
SELECT db4ai.train('map_cached','svm_classification','SELECT label,x FROM public.classification ORDER BY x',
 '{"kernel":"gaussian","components":16,"max_iterations":20,"batch_size":7,"shuffle_buffer_size":128,"kernel_cache":true}');
SELECT db4ai.train('map_uncached','svm_classification','SELECT label,x FROM public.classification ORDER BY x',
 '{"kernel":"gaussian","components":16,"max_iterations":20,"batch_size":7,"shuffle_buffer_size":128,"kernel_cache":false}');
SELECT public.assert_true(db4ai.export_model('map_cached')=db4ai.export_model('map_uncached'),'cached feature maps preserve model bytes');
SELECT public.assert_true((db4ai.explain_model('map_cached')->'metrics'->>'kernel_transformed_rows')::bigint=20,'kernel map performed once per admitted row');

CREATE SEQUENCE public.corgi_reads;
SET db4ai.work_mem='64kB';
SELECT db4ai.train('corgi_spilled','linear_regression',
 'SELECT 1+2*x,x+0*nextval(''public.corgi_reads'') FROM (SELECT i::float8/5000 AS x FROM generate_series(1,5000)i)s',
 '{"max_iterations":20,"batch_size":32,"shuffle":"corgipile","shuffle_block_rows":32,"shuffle_window_blocks":3,"tolerance":0,"learning_rate":0.2}');
SELECT public.assert_true((SELECT last_value=5000 FROM public.corgi_reads),'CorgiPile never reruns source SQL');
SELECT public.assert_true((db4ai.explain_model('corgi_spilled')->'metrics'->>'input_spilled')::boolean,'normalized replay spills');
SELECT public.assert_true((db4ai.explain_model('corgi_spilled')->'metrics'->>'prepared_cache_spilled')::boolean,'prepared replay spills');
SELECT public.assert_true(abs(db4ai.predict_float8('corgi_spilled',ARRAY[.4])-1.8)<0.01,'spilled two-level training converges');
SELECT public.expect_error($q$SELECT db4ai.train('bad_window','linear_regression','SELECT i,i FROM generate_series(1,10000)i','{"shuffle":"corgipile","shuffle_block_rows":1000,"shuffle_window_blocks":32}')$q$,'54000');
SELECT public.expect_error($q$SELECT db4ai.train('bad_mode','linear_regression','SELECT 1,1','{"shuffle":"batch","shuffle_block_rows":1}')$q$,'22023');
SELECT public.expect_error($q$SELECT db4ai.train('bad_pca','pca','SELECT 1,1','{"pca_solver":"jacobi","shuffle":"corgipile"}')$q$,'22023');
SET temp_file_limit='1kB';
SELECT public.expect_error($q$SELECT db4ai.train('quota_failure','linear_regression','SELECT i,i FROM generate_series(1,10000)i')$q$,'53400');
RESET temp_file_limit;
RESET db4ai.work_mem;
SELECT public.assert_true(db4ai.explain_model('quota_failure') IS NULL,'temp failure does not publish a partial model');
SELECT db4ai.execute_ddl('DROP MODEL "模型 DDL"');
SELECT 'CorgiPile, dense replay, model DDL and diagnostics passed' AS result;
-- Native DEFAULT values, option validation and fractional time budgets.
SELECT public.assert_true(NOT (db4ai.parse_ddl('CREATE MODEL dd_default USING linear_regression FEATURES x TARGET y FROM t WITH max_iterations=DEFAULT')->'hyperparameters' ? 'max_iterations'),'DDL DEFAULT omitted from overrides');
SELECT public.expect_error($q$SELECT db4ai.parse_ddl('CREATE MODEL d USING linear_regression FEATURES x TARGET y FROM t WITH made_up=DEFAULT')$q$,'22023');
SELECT public.expect_error($q$SELECT db4ai.parse_ddl('CREATE MODEL d USING linear_regression FEATURES x TARGET y FROM t WITH kernel=DEFAULT')$q$,'22023');
SELECT db4ai.train('fractional_time','linear_regression','SELECT i::float8,i::float8 FROM generate_series(1,1000)i','{"max_seconds":0.1,"max_iterations":3}');
SELECT public.assert_true((db4ai.explain_model('fractional_time')->'hyperparameters'->>'max_seconds')::float8=0.1,'fractional max_seconds preserved');
-- DDL source permissions and RLS are the caller's, not the extension owner's.
CREATE TABLE public.ddl_private(y float8,x float8,username name);
INSERT INTO public.ddl_private VALUES(2,1,'db4ai_bob'),(10,1,'someone_else');
ALTER TABLE public.ddl_private ENABLE ROW LEVEL SECURITY;
CREATE POLICY ddl_visible ON public.ddl_private USING(username=CURRENT_USER);
GRANT SELECT ON public.ddl_private TO db4ai_bob;
SET ROLE db4ai_bob;
SELECT db4ai.execute_ddl('CREATE MODEL ddl_rls USING linear_regression FEATURES x TARGET y FROM public.ddl_private WITH max_iterations=2');
SELECT public.assert_true((db4ai.explain_model('ddl_rls')->'metrics'->>'rows')::integer=1,'DDL respects source RLS');
RESET ROLE;
REVOKE SELECT ON public.ddl_private FROM db4ai_bob;
SET ROLE db4ai_bob;
SELECT public.expect_error($q$SELECT db4ai.execute_ddl('CREATE MODEL ddl_denied USING linear_regression FEATURES x TARGET y FROM public.ddl_private')$q$,'42501');
RESET ROLE;
-- CorgiPile range reads should not amplify each block to a full replay chunk.
SET db4ai.work_mem=64;
SELECT db4ai.train('corgi_io','linear_regression',
 'SELECT i::float8/10000, (i%7)::float8 FROM generate_series(1,10000)i',
 '{"shuffle":"corgipile","shuffle_block_rows":31,"shuffle_window_blocks":3,"batch_size":17,"max_iterations":3,"tolerance":0}');
SELECT public.assert_true(
 (db4ai.explain_model('corgi_io')->'metrics'->>'prepared_bytes_read')::bigint=10000*3*8*3*2,
 'three full shuffle/loss passes read exactly normalized bytes without chunk amplification');
RESET db4ai.work_mem;
SELECT 'CorgiPile, dense replay, DDL, functional diagnostics tests passed' result;
''')
]}
FILES['test/partial_batch_reference.py'] = {'old': None, 'new': 'f276478684177ceb1f1d273bfafe6453df2b8efe', 'base': 'test/partial_batch_reference.py', 'mode': '100644','edits':[
(0,0,r'''#!/usr/bin/env python3
"""Independent scalar update oracle for partial shuffle units.
Only tests the CURRENT extension on 33 synthetic rows; no baseline engine runs.
"""
import json, os, subprocess
from pathlib import Path
pg_config=os.environ.get('PG_CONFIG','pg_config')
binpath=Path(subprocess.check_output([pg_config,'--bindir'],text=True).strip())
def sql(text):
    return subprocess.check_output([str(binpath/'psql'),'-X','-A','-t','-v','ON_ERROR_STOP=1','-c',text],text=True).strip()
for mode in ['batch','row','corgipile']:
    options={'profile':'opengauss','max_iterations':5,'batch_size':8,'learning_rate':.05,'decay':1,'tolerance':0,'seed':42,'shuffle':mode}
    if mode=='corgipile':options.update(shuffle_block_rows=7,shuffle_window_blocks=3)
    else:options['shuffle_buffer_size']=64
    spec=json.dumps(options)
    name='partial_oracle_'+mode
    sql(f"SELECT db4ai.drop_model('{name}'); SELECT db4ai.train('{name}','linear_regression','SELECT 1+2*((i-16)::float8/10), (i-16)::float8/10 FROM generate_series(0,32)i','{spec}')")
    actual=json.loads(sql(f"SELECT convert_from(db4ai.export_model('{name}'),'UTF8')"))['weights']
    w=[0.,0.]
    for epoch in range(5):
        order=json.loads(sql(f"SELECT to_json(db4ai.shuffle_indices(33,'{spec}',{epoch}))"))
        assert sorted(order)==list(range(33))
        for off in range(0,len(order),8):
            batch=order[off:off+8];grad=[0.,0.]
            for i in batch:
                x=(i-16)/10.;y=1+2*x;err=w[0]+w[1]*x-y
                grad[0]+=err;grad[1]+=err*x
            for j in range(2):w[j]-=.05*grad[j]/len(batch)
    assert max(abs(a-b) for a,b in zip(actual,w))<1e-13,(mode,actual,w)
    print('PASS independent partial-minibatch reassembly oracle:',mode)
''')
]}
FILES['test/run.sh'] = {'old': '2abb255ce4f4f7ca83d91d76b6fa3fd9eedd3144', 'new': '6ddc1d4f48aefd1f73d08a83590fd1db9a3e7799', 'base': 'test/run.sh', 'mode': '100755','edits':[
(22,22,r'''"$BIN/psql" -X -v ON_ERROR_STOP=1 -f "$ROOT/test/corgipile_ddl.sql"
'''),
(23,23,r'''python3 "$ROOT/test/partial_batch_reference.py"
''')
]}
FILES['test/upgrade.sql'] = {'old': '9a1ce2bef11690f1006c9821cd2cb63a290a699a', 'new': 'ac1f460e0f46800c8a189e440f0e0d1e9e6b0c39', 'base': 'test/upgrade.sql', 'mode': '100644','edits':[
(30,30,r'''ALTER EXTENSION pg_db4ai UPDATE TO '0.3.0';
DO $$ BEGIN
  IF EXISTS(SELECT FROM original_payload p JOIN db4ai.pg_db4ai_models m USING(model_name) WHERE p.checksum<>md5(m.model_data)) THEN RAISE EXCEPTION '0.3 upgrade rewrote model'; END IF;
  IF db4ai.predict_float8('v01_linear',ARRAY[3.0])<>7 THEN RAISE EXCEPTION '0.3 legacy model load'; END IF;
  IF array_length(db4ai.shuffle_indices(200,'{"shuffle":"corgipile","shuffle_block_rows":7,"shuffle_window_blocks":3}',1),1)<>200 THEN RAISE EXCEPTION '0.3 shuffle upgrade'; END IF;
END $$;
SELECT db4ai.execute_ddl('CREATE MODEL after_v03 USING linear_regression FEATURES x TARGET y FROM (SELECT 2.0 AS y,1.0 AS x)t WITH max_iterations=2');
SELECT '0.2 to 0.3 upgrade passed' result;
''')
]}
FILES['test/xgboost.sql'] = {'old': 'fd5beeda1eebb537701e6eadb522a689130adbeb', 'new': 'dbcb06e9852af3319d454c90ecba400dd0e98a30', 'base': 'test/xgboost.sql', 'mode': '100644','edits':[
(54,54,r'''-- nthread is a real training capability only in explicitly enabled builds.
DO $$ BEGIN
 IF (db4ai.build_info()->>'openmp')::boolean THEN
   PERFORM db4ai.train('xgb_threads','xgboost_regression_squarederror',
     'SELECT y,x,x2 FROM public.regression','{"nthread":2,"max_iterations":15}');
   IF (db4ai.explain_model('xgb_threads')->'hyperparameters'->>'nthread')::int<>2
     THEN RAISE EXCEPTION 'requested training threads not preserved'; END IF;
 ELSE
   PERFORM public.expect_error($q$SELECT db4ai.train('xgb_threads','xgboost_regression_squarederror',
       'SELECT y,x,x2 FROM public.regression','{"nthread":2}')$q$,'0A000');
 END IF;
END $$;
SELECT public.assert_true((db4ai.build_info()->>'native_create_model_grammar')::boolean=false,
                          'explicit DDL must not claim native grammar');
''')
]}
