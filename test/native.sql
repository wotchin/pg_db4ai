\set ON_ERROR_STOP 1
CREATE EXTENSION pg_db4ai;
CREATE FUNCTION public.assert_true(value boolean, message text) RETURNS void
LANGUAGE plpgsql AS $$ BEGIN
    IF value IS DISTINCT FROM true THEN RAISE EXCEPTION 'assertion failed: %', message; END IF;
END $$;
CREATE FUNCTION public.expect_error(statement text, expected text) RETURNS void
LANGUAGE plpgsql AS $$ BEGIN
    BEGIN
        EXECUTE statement;
    EXCEPTION WHEN OTHERS THEN
        IF SQLSTATE = expected THEN RETURN; END IF;
        RAISE;
    END;
    RAISE EXCEPTION 'expected SQLSTATE %, statement succeeded: %', expected, statement;
END $$;

CREATE TABLE public.regression AS
SELECT 1 + 2*x - 3*x*x AS y, x, x*x AS x2
FROM (SELECT i::float8/50 AS x FROM generate_series(-50,50) AS i) s;
SELECT db4ai.train('reg', 'linear_regression', 'SELECT y,x,x2 FROM public.regression ORDER BY x',
                  '{"max_iterations":1500,"learning_rate":0.15,"tolerance":1e-12}');
SELECT public.assert_true(abs(db4ai.predict_float8('reg', ARRAY[0.25,0.0625])-1.3125)<0.001, 'linear regression');
SELECT db4ai.train('constant', 'linear_regression', 'SELECT 10::float8,0::float8,0::float8',
                  '{"max_iterations":400,"learning_rate":0.2,"tolerance":0}');
SELECT public.assert_true(bool_and(abs(db4ai.predict_float8(name, ARRAY[0.25,0.0625])-expected)<0.001),
                          'model name can change per row')
FROM (VALUES ('reg',1.3125),('constant',10.0),('reg',1.3125)) AS v(name,expected);

CREATE TABLE public.classification AS
SELECT (i>0)::int AS label, i::float8/10 AS x FROM generate_series(-10,10) i WHERE i<>0;
SELECT db4ai.train('logreg','logistic_regression','SELECT label,x FROM public.classification',
                  '{"max_iterations":300,"learning_rate":0.1,"tolerance":0}');
SELECT public.assert_true(db4ai.predict_bool('logreg',ARRAY[1.0]) AND NOT db4ai.predict_bool('logreg',ARRAY[-1.0]), 'logistic classes');
SELECT public.assert_true(db4ai.predict_float8('logreg',ARRAY[1.0])>0.8, 'logistic probability');
SELECT db4ai.train('svm','svm_classification','SELECT label,x FROM public.classification',
                  '{"max_iterations":300,"learning_rate":0.1,"tolerance":0}');
SELECT public.assert_true(db4ai.predict_bool('svm',ARRAY[1.0]) AND NOT db4ai.predict_bool('svm',ARRAY[-1.0]), 'SVM classes');

SELECT db4ai.train('clusters','kmeans',
    'SELECT (CASE WHEN i<10 THEN -10 ELSE 10 END)::float8+i*0.01 AS x FROM generate_series(0,19) i',
    '{"k":2,"seed":42}');
SELECT public.assert_true(db4ai.predict_cluster('clusters',ARRAY[-10.0])<>db4ai.predict_cluster('clusters',ARRAY[10.0]), 'kmeans separates clusters');
SELECT public.assert_true(db4ai.predict_cluster('clusters',ARRAY[-10.0])=db4ai.predict_cluster('clusters',ARRAY[-9.9]), 'kmeans stable cluster');

SELECT db4ai.train('pca','pca','SELECT i::float8 AS x, (2*i)::float8 AS x2 FROM generate_series(-10,10) i', '{"n_components":1}');
SELECT public.assert_true(abs((db4ai.transform('pca',ARRAY[1.0,2.0]))[1]-sqrt(5.0))<1e-8, 'PCA projection');
SELECT public.assert_true((db4ai.explain_model('pca')->>'n_outputs')::int=1, 'PCA metadata');

SELECT public.expect_error($q$SELECT db4ai.train('bad','linear_regression','SELECT 1,2','{"typo":2}')$q$, '22023');
SELECT public.expect_error($q$SELECT db4ai.train('bad','linear_regression','SELECT 1,2; SELECT 3,4')$q$, '22023');
SELECT public.expect_error($q$SELECT db4ai.train('bad','linear_regression','WITH x AS (DELETE FROM public.regression RETURNING *) SELECT y,x,x2 FROM x')$q$, '25006');
SELECT public.expect_error($q$SELECT db4ai.train('bad','linear_regression','SELECT y,x FROM public.regression FOR UPDATE')$q$, '25006');
SELECT public.expect_error($q$SELECT db4ai.train('bad','linear_regression','SELECT 1,NULL::float8')$q$, '22004');
SELECT public.expect_error($q$SELECT db4ai.train('bad','linear_regression','SELECT 1,2 WHERE false')$q$, 'P0002');
SELECT public.expect_error($q$SELECT db4ai.train('bad','logistic_regression','SELECT 2,1')$q$, '22023');
SELECT public.expect_error($q$SELECT db4ai.predict_float8('reg',ARRAY[1.0])$q$, '42804');
SELECT public.expect_error($q$SELECT db4ai.predict_float8('reg',ARRAY[1.0,NULL])$q$, '22004');
SELECT public.expect_error($q$SELECT db4ai.predict_float8('reg',ARRAY['NaN'::float8,1.0])$q$, '22003');
SELECT public.expect_error($q$SELECT db4ai.predict_bool('reg',ARRAY[1.0,1.0])$q$, '42804');

CREATE SEQUENCE public.source_reads;
SET db4ai.work_mem='64kB';
SELECT db4ai.train('spill','linear_regression',
  'SELECT 2*x, x + 0*nextval(''public.source_reads'') FROM (SELECT i::float8/5000 AS x FROM generate_series(1,5000) i) s',
  '{"max_iterations":5,"tolerance":0}');
SELECT public.assert_true((SELECT last_value=5000 FROM public.source_reads), 'input SELECT evaluated exactly once across spill and epochs');
SELECT public.assert_true((SELECT rows_processed=5000 FROM db4ai.pg_db4ai_models WHERE model_name='spill'), 'spilled row count');
RESET db4ai.work_mem;

BEGIN;
SELECT db4ai.train('rolledback','linear_regression','SELECT 1,1');
ROLLBACK;
SELECT public.assert_true(NOT EXISTS(SELECT FROM db4ai.pg_db4ai_models WHERE model_name='rolledback'), 'transaction rollback');

-- One outer statement, same cached expression, intervening writes and a new model ID.
DO $$ DECLARE i int; result float8; BEGIN
  FOR i IN 1..3 LOOP
    PERFORM db4ai.train('loop','linear_regression',format('SELECT %s::float8,0::float8',i),
                       '{"max_iterations":400,"learning_rate":0.2,"tolerance":0}');
    SELECT db4ai.predict_float8('loop',ARRAY[0.0]) INTO result;
    PERFORM public.assert_true(abs(result-i)<1e-8,'cache invalidation after model replacement');
    PERFORM db4ai.drop_model('loop');
  END LOOP;
END $$;

SELECT db4ai.train('precision_a','linear_regression','SELECT y,x,x2 FROM public.regression','{"max_iterations":20}');
SET extra_float_digits=-15;
SELECT db4ai.train('precision_b','linear_regression','SELECT y,x,x2 FROM public.regression','{"max_iterations":20}');
RESET extra_float_digits;
SELECT public.assert_true(db4ai.export_model('precision_a')=db4ai.export_model('precision_b'),'model encoding independent of extra_float_digits');

CREATE ROLE db4ai_alice;
CREATE ROLE db4ai_bob;
GRANT SELECT ON public.regression TO db4ai_alice, db4ai_bob;
CREATE TABLE public.secret(x float8);
INSERT INTO public.secret VALUES (1);
SET ROLE db4ai_alice;
SELECT db4ai.train('private','linear_regression','SELECT y,x,x2 FROM public.regression','{"max_iterations":5}');
SELECT public.expect_error($q$SELECT db4ai.train('private','linear_regression','SELECT 1,1')$q$, '23505');
SELECT public.assert_true(CURRENT_USER='db4ai_alice', 'identity restored after protected registry error');
SELECT public.expect_error($q$SELECT db4ai.train('forbidden','linear_regression','SELECT x,x FROM public.secret')$q$, '42501');
SELECT public.expect_error($q$UPDATE db4ai.pg_db4ai_models SET model_data='\x00'::bytea$q$, '42501');
SELECT public.assert_true((SELECT count(*)=1 FROM db4ai.pg_db4ai_models), 'RLS only exposes owned model');
RESET ROLE;
SET ROLE db4ai_bob;
SELECT public.assert_true((SELECT count(*)=0 FROM db4ai.pg_db4ai_models), 'cross-role model isolation');
SELECT public.expect_error($q$SELECT db4ai.predict_float8('private',ARRAY[1.0,1.0])$q$, '42704');
SELECT public.assert_true(NOT db4ai.drop_model('private'), 'cannot drop another role model');
RESET ROLE;
SELECT public.assert_true((SELECT count(*)=1 FROM db4ai.pg_db4ai_models WHERE model_name='private'), 'other role model retained');
SELECT 'native and security tests passed' AS result;
