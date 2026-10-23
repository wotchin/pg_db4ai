\set ON_ERROR_STOP 1
CREATE EXTENSION pg_db4ai VERSION '0.3.0';
-- This is an opaque legacy-artifact preservation fixture, not a new-model fit.
INSERT INTO db4ai.pg_db4ai_models(owner_oid,model_name,algorithm,model_format,n_features,n_outputs,
 feature_names,hyperparameters,metrics,model_data,training_sql,rows_processed)
VALUES(current_user::regrole,'legacy_joint','bayes_net_internal','json',1,1,ARRAY['x'],'{}','{}',
 convert_to('{"format_version":1,"algorithm":"bayes_net_internal","nfeatures":1}','UTF8'),'SELECT x',1);
CREATE TEMP TABLE original_bytes AS SELECT db4ai.export_model('legacy_joint') AS bytes;
CREATE VIEW public.dependent_joint_api AS SELECT db4ai.predict_joint('legacy_joint',1);
DO $$ BEGIN
 BEGIN
  EXECUTE 'ALTER EXTENSION pg_db4ai UPDATE TO ''0.4.0''';
  RAISE EXCEPTION 'upgrade unexpectedly dropped a user-dependent view';
 EXCEPTION WHEN dependent_objects_still_exist THEN NULL;
 END;
END$$;
DROP VIEW public.dependent_joint_api;
ALTER EXTENSION pg_db4ai UPDATE TO '0.4.0';
DO $$BEGIN
 IF (SELECT bytes FROM original_bytes)<>db4ai.export_model('legacy_joint') THEN
  RAISE EXCEPTION 'upgrade modified old Bayes payload';
 END IF;
 IF to_regprocedure('db4ai.predict_joint(text,"any")') IS NOT NULL THEN
  RAISE EXCEPTION 'joint API survived upgrade';
 END IF;
 BEGIN
  PERFORM db4ai.predict_float8('legacy_joint',ARRAY[0.0]);
  RAISE EXCEPTION 'old model silently reinterpreted';
 EXCEPTION WHEN feature_not_supported THEN NULL;
 END;
END$$;
SELECT db4ai.train('upgraded_nb','gaussian_naive_bayes','SELECT * FROM (VALUES(0,-1.0),(1,1.0))t(y,x)');
DO $$BEGIN
 IF db4ai.predict_label('upgraded_nb',ARRAY[1.0],NULL::int)<>1 THEN RAISE EXCEPTION 'new predictor failed';END IF;
 IF NOT db4ai.drop_model('legacy_joint') THEN RAISE EXCEPTION 'legacy drop failed';END IF;
END$$;
