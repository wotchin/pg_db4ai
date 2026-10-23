\set ON_ERROR_STOP 1
-- Public pipeline supports multi:softprob without inventing scalar multiclass semantics.
CREATE TABLE public.am_xgb_reports(name text, report jsonb);
INSERT INTO public.am_xgb_reports SELECT 'am_xgb_three',db4ai.automl('am_xgb_three','classification',
 'SELECT (i%3)::text AS y,(i%3)::float8 AS x,i::float8/1000 AS z FROM generate_series(1,300)i',
 '{"time_budget_seconds":10,"metric":"log_loss"}');
SELECT public.assert_true((SELECT report->>'status'='ready' AND report->>'selected_algorithm'='xgboost_multiclass'
 FROM public.am_xgb_reports WHERE name='am_xgb_three'),'actual multiclass XGBoost candidate selected');
SELECT public.assert_true(db4ai.predict_text('am_xgb_three',ARRAY[2.0,0.1])='2','multiclass XGBoost label');
SELECT public.assert_true((SELECT abs(sum(p)-1)<1e-12 FROM unnest(db4ai.predict_proba('am_xgb_three',ARRAY[2.0,0.1]))p),'multiclass XGBoost posterior');
SELECT public.assert_true(db4ai.predict_proba('am_xgb_three',ARRAY[2.0,0.1])<>db4ai.predict_proba('am_xgb_three',ARRAY[0.0,0.1]),'proxy consumes new rows');
SELECT public.expect_error($q$SELECT db4ai.predict_float8('am_xgb_three',ARRAY[2.0,0.1])$q$,'42804');
INSERT INTO public.am_xgb_reports SELECT 'am_xgb_xor',db4ai.automl('am_xgb_xor','classification',
 'SELECT ((i%4=1) OR (i%4=2)) AS y,(i%2)::float8 AS x,((i/2)%2)::float8 AS z,i::float8/10000 AS idnoise FROM generate_series(1,800)i',
 '{"time_budget_seconds":10,"metric":"log_loss"}');
SELECT public.assert_true((SELECT report->>'selected_algorithm'='xgboost_binary_logistic' FROM public.am_xgb_reports WHERE name='am_xgb_xor'),'nonlinear problem selects tree family');
SELECT public.assert_true(db4ai.predict_bool('am_xgb_xor',ARRAY[1.0,0.0,0.01]) AND NOT db4ai.predict_bool('am_xgb_xor',ARRAY[1.0,1.0,0.01]),'tree XOR predictions');
SELECT public.expect_error($q$SELECT db4ai.train('am_bare_multi','xgboost_multiclass','SELECT 1,2')$q$,'0A000');
SELECT 'AutoML XGBoost binary and multiclass integration passed' AS result;
