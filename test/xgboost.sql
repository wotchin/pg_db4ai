\set ON_ERROR_STOP 1
SELECT db4ai.train('xgb_ubj','xgboost_regression_squarederror',
 'SELECT y,x,x2 FROM public.regression','{"max_iterations":80,"learning_rate":0.15,"max_depth":3}');
SELECT db4ai.train('xgb_json','xgboost_regression_squarederror',
 'SELECT y,x,x2 FROM public.regression','{"max_iterations":80,"learning_rate":0.15,"max_depth":3}','json');
SELECT public.assert_true((SELECT model_format='ubj' FROM db4ai.pg_db4ai_models WHERE model_name='xgb_ubj'),'UBJSON default');
SELECT public.assert_true((convert_from(db4ai.export_model('xgb_json'),'UTF8')::jsonb) ? 'learner','export is reloadable model JSON');
SELECT public.assert_true(abs(db4ai.predict_float8('xgb_json',ARRAY[0.25,0.0625])-db4ai.predict_float8('xgb_ubj',ARRAY[0.25,0.0625]))<1e-7,'JSON and UBJSON predict equally');
SELECT public.assert_true(abs(db4ai.predict_float8('xgb_ubj',ARRAY[0.25,0.0625])-1.3125)<0.3,'XGBoost regression');
SELECT db4ai.train('xgb_classifier','xgboost_binary_logistic','SELECT label,x FROM public.classification',
                  '{"max_iterations":40,"learning_rate":0.2,"min_child_weight":0}');
SELECT public.assert_true(db4ai.predict_bool('xgb_classifier',ARRAY[1.0]) AND NOT db4ai.predict_bool('xgb_classifier',ARRAY[-1.0]),'XGBoost classification');
SELECT db4ai.train('xgb_gamma','xgboost_regression_gamma','SELECT x+2,x FROM public.regression','{"max_iterations":5}');
SELECT db4ai.train('xgb_logistic','xgboost_regression_logistic','SELECT (x+1)/2,x FROM public.regression','{"max_iterations":5}');
SELECT public.assert_true(db4ai.predict_float8('xgb_gamma',ARRAY[0.0])>0,'gamma objective');
SELECT public.assert_true(db4ai.predict_float8('xgb_logistic',ARRAY[0.0]) BETWEEN 0 AND 1,'logistic regression objective');
SELECT public.expect_error($q$SELECT db4ai.train('bad_gamma','xgboost_regression_gamma','SELECT -1,1')$q$,'22023');
SELECT public.expect_error($q$SELECT db4ai.train('bad_label','xgboost_binary_logistic','SELECT 2,1')$q$,'22023');
SET db4ai.xgboost_max_input_mb=1;
SELECT public.expect_error($q$SELECT db4ai.train('too_large','xgboost_regression_squarederror','SELECT i::float8,i::float8 FROM generate_series(1,140000) i','{"max_iterations":1}')$q$,'54000');
RESET db4ai.xgboost_max_input_mb;
SELECT 'XGBoost tests passed' AS result;

SELECT db4ai.train('xgb_exact','xgboost_regression_squarederror','SELECT y,x,x2 FROM public.regression',
  '{"n_iterations":15,"eta":0.2,"gamma":0.1,"tree_method":"exact","eval_metric":"mae","booster":"gbtree"}');
SELECT public.assert_true(db4ai.explain_model('xgb_exact')->'metrics'->>'training_evaluation' LIKE '%train-mae:%','XGBoost metric and aliases');
SELECT public.expect_error($q$SELECT db4ai.train('bad_metric','xgboost_binary_logistic','SELECT 1,1','{"eval_metric":"auc"}')$q$,'22023');
SELECT public.expect_error($q$SELECT db4ai.train('bad_tree','xgboost_binary_logistic','SELECT 1,1','{"tree_method":"gpu_hist"}')$q$,'22023');
SELECT public.expect_error($q$SELECT db4ai.train('bad_alias','xgboost_binary_logistic','SELECT 1,1','{"eta":0.1,"learning_rate":0.2}')$q$,'22023');

-- Reused in-place buffers must see each new row, not cached input values.
CREATE TEMP TABLE xgb_varying_predictions AS
SELECT x, db4ai.predict_float8('xgb_ubj', ARRAY[x,x2]) AS prediction
FROM public.regression ORDER BY x;
SELECT public.assert_true((SELECT count(DISTINCT prediction)>10 FROM xgb_varying_predictions),
                          'in-place prediction consumes changing row values');
SELECT public.assert_true((SELECT max(abs(p.prediction-db4ai.predict_float8('xgb_json',ARRAY[r.x,r.x2])))<1e-7
                           FROM xgb_varying_predictions p JOIN public.regression r USING(x)),
                          'in-place JSON/UBJSON all-row equality');
-- Non-gbtree boosters retain the compatible DMatrix fallback.
SELECT db4ai.train('xgb_linear_fallback','xgboost_regression_squarederror',
  'SELECT 1+2*x AS y,x FROM public.regression',
  '{"booster":"gblinear","max_iterations":100,"learning_rate":0.3,"l2":0}');
SELECT public.assert_true(abs(db4ai.predict_float8('xgb_linear_fallback',ARRAY[0.25])-1.5)<0.02,
                          'gblinear fallback remains functional');
SELECT db4ai.train('xgb_dart_fallback','xgboost_regression_squarederror',
  'SELECT y,x,x2 FROM public.regression',
  '{"booster":"dart","max_iterations":30,"learning_rate":0.2,"max_depth":3}');
SELECT public.assert_true(abs(db4ai.predict_float8('xgb_dart_fallback',ARRAY[0.25,0.0625])-1.3125)<0.5,
                          'dart fallback remains functional');
SELECT public.expect_error($q$SELECT db4ai.predict_float8('xgb_ubj',ARRAY[1e300,1.0])$q$,'22003');
SELECT public.assert_true(abs(db4ai.predict_float8('xgb_ubj',ARRAY[0.25,0.0625])-1.3125)<0.3,
                          'predictor usable after invalid row fails in subtransaction');
SELECT 'XGBoost in-place/fallback regressions passed' AS result;
