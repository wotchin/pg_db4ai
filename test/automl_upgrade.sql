\set ON_ERROR_STOP 1
CREATE EXTENSION pg_db4ai VERSION '0.4.0';
SELECT db4ai.train('pre_auto','linear_regression','SELECT i::float8 AS y,i::float8/100 AS x FROM generate_series(1,100)i','{"max_iterations":400,"learning_rate":0.1}');
CREATE TABLE old_payload AS SELECT model_data FROM db4ai.pg_db4ai_models WHERE model_name='pre_auto';
ALTER EXTENSION pg_db4ai UPDATE TO '0.5.0';
DO $$BEGIN
 IF (SELECT model_data FROM db4ai.pg_db4ai_models WHERE model_name='pre_auto') != (SELECT model_data FROM old_payload) THEN RAISE EXCEPTION 'old artifact changed'; END IF;
 IF to_regprocedure('db4ai.automl(text,text,text,jsonb)') IS NULL OR to_regprocedure('db4ai.predict(text,jsonb)') IS NULL THEN RAISE EXCEPTION 'missing API'; END IF;
END$$;
SELECT db4ai.predict_float8('pre_auto',ARRAY[0.5]);
SELECT db4ai.automl('post_auto','regression','SELECT i AS y,i AS x FROM generate_series(1,100)i','{"use_xgboost":false,"max_trials":1}');
SELECT db4ai.predict('post_auto','{"x":50}');
