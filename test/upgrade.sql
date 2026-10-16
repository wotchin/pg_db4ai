\set ON_ERROR_STOP on
-- Explicit hand-authored fixtures in the original extension's 0.1 JSON format.
-- They are not openGauss memory snapshots, and no retraining is performed.
CREATE EXTENSION pg_db4ai VERSION '0.1.0';
INSERT INTO db4ai.pg_db4ai_models(owner_oid,model_name,algorithm,model_format,n_features,n_outputs,
  feature_names,hyperparameters,metrics,model_data,training_sql,rows_processed)
VALUES (CURRENT_USER::regrole,'v01_linear','linear_regression','json',1,1,ARRAY['x'],'{}','{}',
  convert_to('{"format_version":1,"algorithm":"linear_regression","n_features":1,"n_outputs":1,"weights":[1,2]}','UTF8'),'v0.1 format fixture',2),
 (CURRENT_USER::regrole,'v01_logistic','logistic_regression','json',1,1,ARRAY['x'],'{}','{}',
  convert_to('{"format_version":1,"algorithm":"logistic_regression","n_features":1,"n_outputs":1,"weights":[0,1]}','UTF8'),'v0.1 format fixture',2),
 (CURRENT_USER::regrole,'v01_kmeans','kmeans','json',1,2,ARRAY['x'],'{}','{}',
  convert_to('{"format_version":1,"algorithm":"kmeans","n_features":1,"n_outputs":2,"weights":[-10,10]}','UTF8'),'v0.1 format fixture',2),
 (CURRENT_USER::regrole,'v01_pca','pca','json',2,1,ARRAY['x','y'],'{}','{}',
  convert_to('{"format_version":1,"algorithm":"pca","n_features":2,"n_outputs":1,"weights":[0.4472135954999579,0.8944271909999159],"mean":[0,0]}','UTF8'),'v0.1 format fixture',2);
CREATE TEMP TABLE original_payload AS SELECT model_name,md5(model_data) checksum FROM db4ai.pg_db4ai_models;
ALTER EXTENSION pg_db4ai UPDATE TO '0.2.0';
DO $$ BEGIN
  IF EXISTS(SELECT FROM original_payload p JOIN db4ai.pg_db4ai_models m USING(model_name) WHERE p.checksum<>md5(m.model_data)) THEN RAISE EXCEPTION 'upgrade rewrote model'; END IF;
  IF db4ai.predict_float8('v01_linear',ARRAY[3.0])<>7 THEN RAISE EXCEPTION 'v0.1 linear load'; END IF;
  IF abs(db4ai.predict_float8('v01_logistic',ARRAY[1.0])-1/(1+exp(-1.0)))>1e-14 THEN RAISE EXCEPTION 'v0.1 logistic load'; END IF;
  IF db4ai.predict_cluster('v01_kmeans',ARRAY[-9.0])=db4ai.predict_cluster('v01_kmeans',ARRAY[9.0]) THEN RAISE EXCEPTION 'v0.1 kmeans load'; END IF;
  IF abs((db4ai.transform('v01_pca',ARRAY[1.0,2.0]))[1]-sqrt(5.0))>1e-14 THEN RAISE EXCEPTION 'v0.1 PCA load'; END IF;
END $$;
SELECT db4ai.train('post_upgrade','multiclass',
  $$SELECT 'a'::text,1,0,0 UNION ALL SELECT 'b',0,1,0 UNION ALL SELECT 'c',0,0,1$$,
  '{"classifier":"logistic_regression","max_iterations":100,"learning_rate":0.3}');
DO $$ BEGIN
 IF db4ai.predict_text('post_upgrade',ARRAY[0.0,1.0,0.0])<>'b' THEN RAISE EXCEPTION 'new algorithm after upgrade'; END IF;
END $$;
SELECT '0.1 to 0.2 upgrade and legacy model fixtures passed' result;
