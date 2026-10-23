\set ON_ERROR_STOP 1
CREATE TABLE public.am_reference AS
 SELECT i, date '2020-01-01'+i AS stamp,
        CASE WHEN i%4<2 THEN 'cold' ELSE 'hot' END::text AS label,
        (i%4)::float8 AS x, (i/4)::float8 AS z,
        CASE WHEN i>144 THEN 'holdout_only' WHEN i%4<2 THEN 'north' ELSE 'south' END::text AS region,
        NULL::float8 AS empty_feature
 FROM generate_series(1,240)i;
CREATE TABLE public.am_reports(name text PRIMARY KEY, report jsonb);
INSERT INTO public.am_reports SELECT 'am_class',db4ai.automl('am_class','classification',
 'SELECT label,x,z,region,empty_feature,stamp FROM public.am_reference ORDER BY i',
 '{"split":"time","time_column":"stamp","time_budget_seconds":10,"use_xgboost":false,"max_trials":4}');
SELECT public.assert_true((SELECT report->>'status'='ready' AND (report->>'train_rows')::int=144 AND
 (report->>'validation_rows')::int=48 AND (report->>'test_rows')::int=48 AND
 (report->>'source_executions')::int=1 AND (report->>'trial_count')::int<=4
 FROM public.am_reports WHERE name='am_class'),'AutoML three partitions, bounded trials and publication');
SELECT public.assert_true((db4ai.predict('am_class','{"x":3,"z":10,"region":"south","empty_feature":null}')->>'prediction')='hot',
 'mixed feature inference');
SELECT public.assert_true((SELECT abs(sum(v::float8)-1)<1e-12 FROM jsonb_array_elements_text(
 db4ai.predict('am_class','{"x":0,"z":10,"region":"unknown","empty_feature":null}')->'probabilities')v),'unknown category posterior');
SELECT public.assert_true((SELECT (f->>'mean')::float8=17.75 FROM jsonb_array_elements(
 convert_from(db4ai.export_model('am_class'),'UTF8')::jsonb->'features')f WHERE f->>'name'='z'), 'training-only mean');
SELECT public.assert_true(convert_from(db4ai.export_model('am_class'),'UTF8') NOT LIKE '%holdout_only%', 'holdout categories not fitted');
SELECT public.expect_error($q$SELECT db4ai.predict_proba('am_class',ARRAY[0.0,1.0,0.0,0.0])$q$,'42804');
SELECT public.expect_error($q$SELECT db4ai.predict('am_class','{"x":0,"z":1}')$q$,'22023');
SELECT public.expect_error($q$SELECT db4ai.predict('am_class','{"x":"bad","z":1,"region":"north","empty_feature":null}')$q$,'42804');
-- Completely alter TEST target, not validation or input; selected parameters/artifact must not change.
SELECT db4ai.automl('am_test_changed','classification',
 $$SELECT CASE WHEN i>192 THEN CASE WHEN label='cold' THEN 'hot' ELSE 'cold' END ELSE label END AS label,
 x,z,region,empty_feature,stamp FROM public.am_reference ORDER BY i$$,
 '{"split":"time","time_column":"stamp","time_budget_seconds":10,"use_xgboost":false,"max_trials":4}');
SELECT public.assert_true(db4ai.export_model('am_class')=db4ai.export_model('am_test_changed'),'test labels never select/refit/preprocess');
-- Numeric API and JSON API apply the exact same fitted transform. Large typed labels remain exact.
SELECT db4ai.automl('am_bigint','classification',
 $$SELECT CASE WHEN i%2=0 THEN 9007199254740993::bigint ELSE 9007199254740992::bigint END AS y,
 (i%2)::float8 AS x, i::float8 AS z FROM generate_series(1,300)i$$,
 '{"use_xgboost":false,"time_budget_seconds":10}');
SELECT public.assert_true(db4ai.predict_label('am_bigint',ARRAY[0.0,120.0],NULL::bigint)=9007199254740993::bigint,'typed bigint class retained');
SELECT public.assert_true(to_jsonb(db4ai.predict_proba('am_bigint',ARRAY[0.0,120.0]))=
 db4ai.predict('am_bigint','{"x":0,"z":120}')->'probabilities','array and named JSON preprocessing match');
SELECT public.expect_error($q$SELECT db4ai.predict_bool('am_bigint',ARRAY[0.0,120.0])$q$,'42804');
SELECT public.expect_error($q$SELECT db4ai.predict_label('am_bigint',ARRAY[0.0,120.0],NULL::text)$q$,'42804');
SELECT db4ai.automl('am_bool','classification','SELECT i%2=0 AS y,(i%2)::float8 AS x,i::float8 AS z FROM generate_series(1,300)i',
 '{"use_xgboost":false,"time_budget_seconds":10}');
SELECT public.assert_true(db4ai.predict_bool('am_bool',ARRAY[0.0,120.0]) AND NOT db4ai.predict_bool('am_bool',ARRAY[1.0,120.0]),'boolean class semantics');
SELECT db4ai.automl('am_multi','classification','SELECT (i%3)::text AS y,(i%3)::float8 AS x,i::float8/1000 AS z FROM generate_series(1,300)i',
 '{"use_xgboost":false,"time_budget_seconds":10,"metric":"log_loss"}');
SELECT public.assert_true(array_length(db4ai.predict_proba('am_multi',ARRAY[2.0,0.1]),1)=3,'multiclass normalized probabilities');
SELECT public.assert_true((SELECT abs(sum(p)-1)<1e-12 FROM unnest(db4ai.predict_proba('am_multi',ARRAY[2.0,0.1]))p),'multiclass probability sum');
SELECT public.expect_error($q$SELECT db4ai.predict_float8('am_multi',ARRAY[2.0,0.1])$q$,'42804');
-- Regression target scaling is automatically inverted, prediction accepts missing numerical values.
INSERT INTO public.am_reports SELECT 'am_reg',db4ai.automl('am_reg','regression',
 'SELECT 3+2*i::float8 AS y,i::float8 AS x FROM generate_series(1,300)i',
 '{"use_xgboost":false,"time_budget_seconds":10}');
SELECT public.assert_true(abs(db4ai.predict_float8('am_reg',ARRAY[100.0])-203)<1,'regression inverse transform');
SELECT public.assert_true((db4ai.predict('am_reg','{"x":100}')->>'prediction')::float8=db4ai.predict_float8('am_reg',ARRAY[100.0]),'regression named/array equality');
SELECT public.assert_true(db4ai.predict_float8('am_reg',ARRAY[NULL::float8]) IS NOT NULL,'mean imputation reused at inference');
SELECT public.expect_error($q$SELECT db4ai.predict_proba('am_reg',ARRAY[1.0])$q$,'42804');
-- Group key never becomes a feature. Replication within groups does not leak groups.
INSERT INTO public.am_reports SELECT 'am_group',db4ai.automl('am_group','regression',
 'SELECT 3+2*g::float8 AS y,g::text AS entity,g::float8 AS x FROM generate_series(1,100)g CROSS JOIN generate_series(1,3)r ORDER BY g,r',
 '{"split":"group","group_column":"entity","use_xgboost":false,"time_budget_seconds":10,"max_trials":2}');
SELECT public.assert_true((SELECT (report->>'raw_features')::int=1 AND (report->>'train_rows')::int%3=0 AND
 (report->>'validation_rows')::int%3=0 AND (report->>'test_rows')::int%3=0 FROM public.am_reports WHERE name='am_group'),'whole groups and key excluded');
SELECT public.assert_true((db4ai.automl('am_nogain','regression','SELECT 1.0 AS y,i::float8 AS x FROM generate_series(1,100)i',
 '{"use_xgboost":false,"max_trials":1,"time_budget_seconds":10}')->>'status')='no_validation_improvement','no improvement is explicit');
SELECT public.assert_true(NOT EXISTS(SELECT FROM db4ai.pg_db4ai_models WHERE model_name='am_nogain'),'unqualified candidates not published');
-- Real source SELECT is consumed once, not once per candidate.
CREATE SEQUENCE public.am_source_calls;
SELECT db4ai.automl('am_once','regression',
 $$SELECT 1+2*i::float8 AS y,nextval('public.am_source_calls')::float8 AS x FROM generate_series(1,120)i$$,
 '{"use_xgboost":false,"time_budget_seconds":10}');
SELECT public.assert_true((SELECT last_value=120 FROM public.am_source_calls),'AutoML source once');
-- Invalid input and budget admission must not create any model.
SELECT public.expect_error($q$SELECT db4ai.automl('am_bad','infer','SELECT 1,2')$q$,'22023');
SELECT public.expect_error($q$SELECT db4ai.automl('am_bad','regression','SELECT 1 AS y,2 AS x','{"unknown":1}')$q$,'22023');
SELECT public.expect_error($q$SELECT db4ai.automl('am_bad','regression','SELECT 1 AS y,2 AS x','{"time_budget_seconds":0}')$q$,'22023');
SELECT public.expect_error($q$SELECT db4ai.automl('am_bad','regression','SELECT 1 AS y,2 AS x','{"split":"time"}')$q$,'22023');
SELECT public.expect_error($q$SELECT db4ai.automl('am_bad','regression','SELECT 1 AS y,2 AS x; SELECT 3')$q$,'22023');
SELECT public.expect_error($q$SELECT db4ai.automl('am_bad','classification','SELECT NULL::text AS y,i AS x FROM generate_series(1,100)i')$q$,'22004');
SELECT public.expect_error($q$SELECT db4ai.automl('am_bad','regression','SELECT i AS y,i AS x FROM generate_series(1,100)i','{"max_rows":30}')$q$,'54000');
SELECT public.expect_error($q$SELECT db4ai.automl('am_bad','regression','SELECT i AS y,ARRAY[i]::float8[] AS x FROM generate_series(1,100)i')$q$,'42804');
SELECT public.expect_error($q$SELECT db4ai.automl('am_bad','regression','SELECT i AS y,i AS x,i AS x FROM generate_series(1,100)i')$q$,'42701');
SELECT public.expect_error($q$SELECT db4ai.automl('am_bad','regression','SELECT i AS y,''NaN''::float8 AS x,i AS z FROM generate_series(1,100)i')$q$,'22003');
SELECT public.expect_error($q$SELECT db4ai.automl('am_reg','regression','SELECT 1,2')$q$,'42710');
SELECT public.expect_error($q$SELECT db4ai.train('am_bad','automl','SELECT 1,2')$q$,'0A000');
SELECT public.expect_error($q$SELECT db4ai.automl('am_bad','classification',
 'SELECT i::text AS y,i AS x FROM generate_series(1,100)i','{"use_xgboost":false}')$q$,'22023');
-- Read-only validation forbids SQL mutation even before actual source execution.
SELECT public.expect_error($q$SELECT db4ai.automl('am_bad','regression',
 'WITH d AS (DELETE FROM public.am_reference RETURNING i) SELECT i AS y,i AS x FROM d')$q$,'25006');
SELECT public.assert_true((SELECT count(*)=240 FROM public.am_reference),'source CTE rejection before deletion');
-- Outer rollback removes the model; settings restored.
BEGIN;
SELECT db4ai.automl('am_rollback','regression','SELECT i AS y,i AS x FROM generate_series(1,100)i','{"use_xgboost":false,"max_trials":1}');
ROLLBACK;
SELECT public.assert_true(NOT EXISTS(SELECT FROM db4ai.pg_db4ai_models WHERE model_name='am_rollback'),'AutoML rollback');
SELECT public.assert_true(current_setting('work_mem')='4MB' AND current_setting('statement_timeout')='0','settings restored');
-- Corrupt pipeline artifacts are rejected instead of out-of-bounds decoding.
DO $$DECLARE original bytea; BEGIN
 SELECT model_data INTO original FROM db4ai.pg_db4ai_models WHERE model_name='am_reg';
 UPDATE db4ai.pg_db4ai_models SET model_data=convert_to(jsonb_set(convert_from(original,'UTF8')::jsonb,'{features,0,offset}','1')::text,'UTF8') WHERE model_name='am_reg';
 PERFORM public.expect_error($q$SELECT db4ai.predict_float8('am_reg',ARRAY[1.0])$q$,'XX001');
 UPDATE db4ai.pg_db4ai_models SET model_data=original WHERE model_name='am_reg';
END$$;
SELECT public.assert_true(abs(db4ai.predict_float8('am_reg',ARRAY[100.0])-203)<1,'cache invalidation after corrupt row rollback/repair');
-- Ordinary invoker reads, RLS, owner isolation and existing training permission.
CREATE ROLE am_user;
CREATE TABLE public.am_rls(y float8,x float8,visible bool);
INSERT INTO public.am_rls SELECT i,i,i<=120 FROM generate_series(1,200)i;
ALTER TABLE public.am_rls ENABLE ROW LEVEL SECURITY;
CREATE POLICY am_visible ON public.am_rls FOR SELECT TO am_user USING(visible);
GRANT SELECT ON public.am_rls TO am_user;
GRANT EXECUTE ON FUNCTION public.assert_true(boolean,text),public.expect_error(text,text) TO am_user;
SET ROLE am_user;
SELECT public.assert_true((db4ai.automl('am_rls','regression','SELECT y,x FROM public.am_rls','{"use_xgboost":false,"max_trials":1}')->>'source_rows')::int=120,'caller RLS training');
SELECT public.expect_error($q$SELECT db4ai.predict_float8('am_reg',ARRAY[1.0])$q$,'42704');
SELECT public.expect_error($q$SELECT db4ai.automl('am_denied','regression','SELECT i AS y,x FROM public.am_reference')$q$,'42501');
RESET ROLE;
REVOKE EXECUTE ON FUNCTION db4ai.train(text,text,text,jsonb,text) FROM PUBLIC;
SET ROLE am_user;
SELECT public.expect_error($q$SELECT db4ai.automl('am_denied','regression','SELECT y,x FROM public.am_rls')$q$,'42501');
RESET ROLE;
GRANT EXECUTE ON FUNCTION db4ai.train(text,text,text,jsonb,text) TO PUBLIC;
SELECT 'AutoML pipeline, leakage, budget admission and permissions tests passed' AS result;
-- Null-only categorical columns, numeric-label magnitude and bounded vocabulary.
SELECT db4ai.automl('am_allnullcat','regression',
 'SELECT i AS y,i AS x,NULL::text AS cat FROM generate_series(1,100)i','{"use_xgboost":false,"max_trials":1}');
SELECT public.assert_true((db4ai.predict('am_allnullcat','{"x":50,"cat":null}')->>'prediction')::float8 BETWEEN 40 AND 60,'null-only categorical reload');
SELECT db4ai.automl('am_large_numeric','classification',
 $$SELECT CASE WHEN i%2=0 THEN 1e1000::numeric ELSE 2e1000::numeric END AS y,(i%2)::float8 AS x,i::float8 AS z FROM generate_series(1,300)i$$,
 '{"use_xgboost":false,"max_trials":2}');
SELECT public.assert_true(db4ai.predict_label('am_large_numeric',ARRAY[0.0,120.0],NULL::numeric)=1e1000::numeric,'classification labels are not converted to float');
SELECT db4ai.automl('am_categorycap','regression',
 'SELECT i AS y,i AS x,(''c''||i::text) AS cat FROM generate_series(1,100)i','{"use_xgboost":false,"max_trials":1,"max_categories":2}');
SELECT public.assert_true((SELECT f->>'capped'='true' AND (f->>'width')::int=4 FROM jsonb_array_elements(
 convert_from(db4ai.export_model('am_categorycap'),'UTF8')::jsonb->'features')f WHERE f->>'name'='cat'),'bounded training vocabulary');
