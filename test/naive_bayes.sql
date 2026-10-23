\set ON_ERROR_STOP 1
-- Fixed supervised data retained for cross-backend and logical-restore oracle.
CREATE TABLE public.nb_reference(label text, x float8, z float8);
INSERT INTO public.nb_reference VALUES
 ('red',-3,1),('red',-1,2),('red',-2,3),
 ('blue',1,1),('blue',3,2),('blue',2,3),
 ('green',0,4),('green',0,6),('green',0,5);
SELECT db4ai.train('nb_reference','gaussian_naive_bayes',
  'SELECT * FROM public.nb_reference ORDER BY label,x,z',
  '{"var_smoothing":1e-9,"variance_floor":1e-12}');
SELECT public.assert_true(db4ai.predict_text('nb_reference',ARRAY[-2.0,2.0])='red','NB actual label prediction');
SELECT public.assert_true(db4ai.predict_label('nb_reference',ARRAY[2.0,2.0],NULL::text)='blue','NB polymorphic typed label');
SELECT public.assert_true((SELECT abs(sum(p)-1)<1e-14 AND min(p)>=0 AND max(p)<=1
  FROM unnest(db4ai.predict_proba('nb_reference',ARRAY[0.0,3.0]))p),'NB normalized posterior');
SELECT public.assert_true(db4ai.predict_scores('nb_reference',ARRAY[0.0,3.0])=
 db4ai.predict_proba('nb_reference',ARRAY[0.0,3.0]),'NB scores are documented probabilities');
SELECT public.assert_true(db4ai.class_labels('nb_reference')=ARRAY['blue','green','red'],'NB explicit class order');
SELECT public.expect_error($q$SELECT db4ai.predict_float8('nb_reference',ARRAY[0.0,3.0])$q$,'42804');
SELECT public.expect_error($q$SELECT db4ai.predict_bool('nb_reference',ARRAY[0.0,3.0])$q$,'42804');
SELECT public.expect_error($q$SELECT db4ai.predict_label('nb_reference',ARRAY[0.0,3.0],NULL::int)$q$,'42804');
SELECT public.expect_error($q$SELECT db4ai.predict_proba('reg',ARRAY[0.0,0.0])$q$,'42804');
SELECT public.expect_error($q$SELECT db4ai.predict_proba('nb_reference',ARRAY[1.0])$q$,'42804');
SELECT public.expect_error($q$SELECT db4ai.predict_proba('nb_reference',ARRAY['NaN'::float8,1])$q$,'22003');
SELECT public.expect_error($q$SELECT db4ai.predict_proba('nb_reference',ARRAY[1e308,1])$q$,'22003');
SELECT public.assert_true(db4ai.predict_text('nb_reference',ARRAY[-2.0,2.0])='red','NB error recovery');
SELECT public.assert_true(db4ai.predict_proba('nb_reference',NULL) IS NULL,'NULL array follows strict SQL API');
-- Constants/singletons have positive smoothed variances, not division by zero.
SELECT db4ai.train('nb_prior','gaussian_naive_bayes',
 $$SELECT label, 7.0 FROM (VALUES('a'),('a'),('a'),('b'))t(label)$$);
SELECT public.assert_true(abs((db4ai.predict_proba('nb_prior',ARRAY[7.0]))[1]-0.75)<1e-14,'NB empirical prior');
SELECT db4ai.train('nb_uniform','gaussian_naive_bayes',
 $$SELECT label, 7.0 FROM (VALUES('a'),('a'),('a'),('b'))t(label)$$,'{"fit_prior":false}');
SELECT public.assert_true(db4ai.predict_proba('nb_uniform',ARRAY[7.0])=ARRAY[0.5,0.5]::float8[],'NB uniform prior');
SELECT db4ai.train('nb_bool','gaussian_naive_bayes',
 'SELECT label,x FROM (VALUES(true,2.0),(true,3.0),(false,-3.0),(false,-2.0))t(label,x)');
SELECT public.assert_true(db4ai.class_labels('nb_bool')=ARRAY['f','t'] AND
 db4ai.predict_bool('nb_bool',ARRAY[3.0]) AND NOT db4ai.predict_bool('nb_bool',ARRAY[-3.0]),'NB boolean identity');
SELECT public.assert_true(db4ai.predict_float8('nb_bool',ARRAY[3.0])>0.99,'binary scalar is probability of class_labels[2]');
SELECT db4ai.train('nb_bigint','gaussian_naive_bayes',
 'SELECT y,x FROM (VALUES(9007199254740992::bigint,-1.0),(9007199254740993::bigint,1.0))t(y,x)');
SELECT public.assert_true(db4ai.predict_label('nb_bigint',ARRAY[1.0],NULL::bigint)=9007199254740993::bigint,'NB label bigint exact');
SELECT db4ai.train('nb_array','gaussian_naive_bayes',
 'SELECT label,ARRAY[x,z]::float8[] FROM public.nb_reference ORDER BY label,x,z');
SELECT public.assert_true(db4ai.export_model('nb_array')=db4ai.export_model('nb_reference'),'array/scalar representation same model');
SELECT public.expect_error($q$SELECT db4ai.train('nb_one','gaussian_naive_bayes','SELECT 1,1')$q$,'22023');
SELECT public.expect_error($q$SELECT db4ai.train('nb_null','gaussian_naive_bayes','SELECT NULL::text,1')$q$,'22004');
SELECT public.expect_error($q$SELECT db4ai.train('nb_textfeature','gaussian_naive_bayes',$s$SELECT 0,'red'::text$s$)$q$,'42804');
SELECT public.expect_error($q$SELECT db4ai.train('nb_badvar','gaussian_naive_bayes','SELECT 0,0','{"variance_floor":0}')$q$,'22023');
SELECT public.expect_error($q$SELECT db4ai.train('nb_badsmooth','gaussian_naive_bayes','SELECT 0,0','{"var_smoothing":-1}')$q$,'22023');
SELECT public.expect_error($q$SELECT db4ai.train('nb_badopt','gaussian_naive_bayes','SELECT 0,0','{"max_iterations":2}')$q$,'22023');
SELECT public.expect_error($q$SELECT db4ai.train('nb_badgraph','gaussian_naive_bayes','SELECT 0,0','{"edges_of_network":"0,1"}')$q$,'22023');
SELECT public.expect_error($q$SELECT db4ai.train('nb_old','bayes_net_internal','SELECT 0,0')$q$,'0A000');
SELECT public.assert_true(to_regprocedure('db4ai.predict_joint(text,"any")') IS NULL,'new installation omits joint estimator');
-- Source query is not re-executed for statistics/evaluation passes.
CREATE SEQUENCE public.nb_source_calls;
SELECT db4ai.train('nb_once','gaussian_naive_bayes',
 $$SELECT i%2, nextval('public.nb_source_calls')::float8 FROM generate_series(1,20)i$$);
SELECT public.assert_true((SELECT last_value=20 FROM public.nb_source_calls),'NB source once');
-- Statement-scoped cache keys must follow models and roles, not first row only.
SELECT public.assert_true(bool_and(abs((db4ai.predict_proba(name,ARRAY[7.0]))[1]-p)<1e-14),'NB model-switch cache')
 FROM (VALUES('nb_prior',0.75),('nb_uniform',0.5),('nb_prior',0.75))t(name,p);
BEGIN;
SELECT db4ai.train('nb_rollback','gaussian_naive_bayes','SELECT * FROM public.nb_reference');
ROLLBACK;
SELECT public.assert_true(NOT EXISTS(SELECT FROM db4ai.pg_db4ai_models WHERE model_name='nb_rollback'),'NB transactional model save');
-- DDL reuses the supervised path and rejects a missing TARGET.
SELECT db4ai.execute_ddl($ddl$CREATE MODEL nb_ddl USING gaussian_naive_bayes
 FEATURES x,z TARGET label FROM public.nb_reference ORDER BY label,x,z WITH var_smoothing=1e-9$ddl$);
SELECT public.assert_true(db4ai.predict_text('nb_ddl',ARRAY[-2.0,2.0])='red','NB model DDL');
-- Numeric storage must remain spillable.
SET db4ai.work_mem=64;
SELECT db4ai.train('nb_spilled','gaussian_naive_bayes',
 'SELECT (i%3)::text,i::float8,i::float8/2 FROM generate_series(1,10000)i');
SELECT public.assert_true((db4ai.explain_model('nb_spilled')->'metrics'->>'input_spilled')::bool,'NB spill');
RESET db4ai.work_mem;
SET db4ai.max_model_mb=1;
SELECT public.expect_error($q$SELECT db4ai.train('nb_state_limit','gaussian_naive_bayes',
 'SELECT i, array_fill(i::float8,ARRAY[512]) FROM generate_series(1,256)i')$q$,'54000');
RESET db4ai.max_model_mb;
SELECT 'Gaussian Naive Bayes user API checks passed' AS result;
-- Rejection must be atomic; no partial artifacts are published.
SELECT public.assert_true(NOT EXISTS(SELECT FROM db4ai.pg_db4ai_models
 WHERE model_name IN ('nb_state_limit','nb_badopt','nb_badvar','nb_old')),'NB failed fits publish nothing');
-- Legacy NULL behavior is explicit, not silent implicit imputation by default.
SELECT db4ai.train('nb_null_legacy','gaussian_naive_bayes',
 $$SELECT * FROM (VALUES('a',NULL::float8),('b',1.0),(NULL::text,100.0))t(y,x)$$,
 '{"null_policy":"legacy"}');
SELECT public.assert_true((db4ai.explain_model('nb_null_legacy')->'metrics'->>'discarded_rows')::int=1,
 'NB legacy skips NULL label');
-- Invalid serialized arrays/priors must fail, rather than read beyond bounds.
BEGIN;
UPDATE db4ai.pg_db4ai_models SET model_data=convert_to(
 ((convert_from(model_data,'UTF8')::jsonb)||'{"variances":[0]}')::text,'UTF8') WHERE model_name='nb_reference';
SELECT public.expect_error($q$SELECT db4ai.predict_proba('nb_reference',ARRAY[0.0,1.0])$q$,'XX001');
ROLLBACK;
BEGIN;
UPDATE db4ai.pg_db4ai_models SET model_data=convert_to(
 ((convert_from(model_data,'UTF8')::jsonb)||'{"class_priors":[0.8,0.8,0.8]}')::text,'UTF8') WHERE model_name='nb_reference';
SELECT public.expect_error($q$SELECT db4ai.predict_proba('nb_reference',ARRAY[0.0,1.0])$q$,'XX001');
ROLLBACK;
-- New algorithm follows source RLS and model-owner isolation.
CREATE ROLE nb_reader;
CREATE TABLE public.nb_rls(owner_name name, y boolean, x float8);
INSERT INTO public.nb_rls VALUES('nb_reader',false,-2),('nb_reader',true,2),
 ('somebody_else',false,-200),('somebody_else',true,200);
ALTER TABLE public.nb_rls ENABLE ROW LEVEL SECURITY;
CREATE POLICY nb_source_policy ON public.nb_rls USING(owner_name=current_user);
GRANT SELECT ON public.nb_rls TO nb_reader;
SET ROLE nb_reader;
SELECT db4ai.train('nb_private','gaussian_naive_bayes','SELECT y,x FROM public.nb_rls');
SELECT public.assert_true((db4ai.explain_model('nb_private')->>'rows_processed')::int=2,'NB respects source RLS');
SELECT public.assert_true(db4ai.predict_bool('nb_private',ARRAY[2.0]),'NB owner can predict');
SELECT public.expect_error($q$SELECT db4ai.predict_proba('nb_reference',ARRAY[0.0,1.0])$q$,'42704');
RESET ROLE;
SELECT public.assert_true(NOT EXISTS(SELECT FROM db4ai.pg_db4ai_models WHERE owner_oid=current_user::regrole AND model_name='nb_private'),
 'NB role-specific artifact owner');
SELECT 'Gaussian NB corruption/permissions/NULL checks passed' AS result;
-- A huge class-independent tail penalty must not erase meaningful class odds.
SELECT db4ai.train('nb_remote_tail','gaussian_naive_bayes',
 'SELECT * FROM (VALUES(false,0.0,-2.0),(false,0.0,-1.0),(true,0.0,1.0),(true,0.0,2.0))t(y,constant,x)');
SELECT public.assert_true(abs((db4ai.predict_log_proba('nb_remote_tail',ARRAY[0.0,1.5]))[1]-
 (db4ai.predict_log_proba('nb_remote_tail',ARRAY[1e8,1.5]))[1])<1e-10,
 'NB common-tail cancellation preserves log odds');
SELECT public.assert_true(db4ai.predict_bool('nb_remote_tail',ARRAY[1e8,1.5]),'NB remote tail preserves class');
