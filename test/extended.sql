\set ON_ERROR_STOP on
LOAD 'pg_db4ai';
CREATE FUNCTION public.model_json(n text) RETURNS jsonb LANGUAGE SQL STABLE AS $$ SELECT convert_from(db4ai.export_model(n),'UTF8')::jsonb $$;
CREATE FUNCTION public.plan_json(q text) RETURNS jsonb LANGUAGE plpgsql AS $$ DECLARE p jsonb; BEGIN EXECUTE 'EXPLAIN (ANALYZE, COSTS OFF, TIMING OFF, FORMAT JSON) '||q INTO p; RETURN p; END $$;

-- Hand-calculated original GD and normalized GD. NGD is not natural gradient.
SELECT db4ai.train('gd_reference','linear_regression','SELECT 2::float8 y,1::float8 x UNION ALL SELECT 4,2',
    '{"optimizer":"gd","shuffle":"none","max_iterations":1,"batch_size":2,"learning_rate":0.1,"tolerance":0}');
SELECT public.assert_true(abs((public.model_json('gd_reference')->'weights'->>0)::float8-0.3)<1e-14 AND
                         abs((public.model_json('gd_reference')->'weights'->>1)::float8-0.5)<1e-14,'GD reference update');
SELECT db4ai.train('ngd_reference','linear_regression','SELECT 2::float8 y,1::float8 x UNION ALL SELECT 4,2',
    '{"optimizer":"ngd","shuffle":"none","max_iterations":1,"batch_size":2,"learning_rate":0.1,"tolerance":0}');
SELECT public.assert_true(abs((public.model_json('ngd_reference')->'weights'->>0)::float8-0.15)<1e-14 AND
                         abs((public.model_json('ngd_reference')->'weights'->>1)::float8-0.0625)<1e-14,'NGD per-feature scale and global scale-rate');
SELECT db4ai.train('ngd_zero','linear_regression','SELECT 1,0 FROM generate_series(1,3)',
    '{"optimizer":"ngd","max_iterations":1,"learning_rate":0.1}');
SELECT public.assert_true((public.model_json('ngd_zero')->'weights'->>1)::float8=0,'NGD zero feature scale');
SELECT db4ai.train('decay','linear_regression','SELECT 1,1',
    '{"max_iterations":2,"learning_rate":0.1,"decay":0.5,"tolerance":0,"shuffle":"none"}');
SELECT public.assert_true(abs((public.model_json('decay')->'weights'->>0)::float8-0.14)<1e-14,'learning-rate decay');

-- Arbitrary hashable SQL labels; never convert label identity through float8.
SELECT db4ai.train('text_labels','logistic_regression',
    $$SELECT CASE WHEN i<0 THEN 'red' ELSE 'blue' END, i::float8 FROM generate_series(-4,4)i WHERE i<>0$$,
    '{"max_iterations":150,"learning_rate":0.2,"tolerance":0}');
SELECT public.assert_true(db4ai.predict_text('text_labels',ARRAY[-2.0])='red' AND db4ai.predict_label('text_labels',ARRAY[2.0],NULL::text)='blue','text labels');
SELECT public.expect_error($q$SELECT db4ai.predict_label('text_labels',ARRAY[1.0],NULL::int)$q$,'42804');
SELECT db4ai.train('numeric_labels','logistic_regression',
    $$SELECT CASE WHEN i<0 THEN 9007199254740992::numeric ELSE 9007199254740993::numeric END, i::float8 FROM generate_series(-3,3)i WHERE i<>0$$,
    '{"max_iterations":150,"learning_rate":0.2,"tolerance":0}');
SELECT public.assert_true(db4ai.predict_label('numeric_labels',ARRAY[2.0],NULL::numeric)=9007199254740993::numeric AND
                         db4ai.predict_label('numeric_labels',ARRAY[-2.0],NULL::numeric)=9007199254740992::numeric,'numeric category precision beyond 2^53');
SELECT db4ai.train('huge_numeric_labels','logistic_regression',
    $$SELECT CASE WHEN i<0 THEN 1e400::numeric ELSE 2e400::numeric END, i::float8 FROM generate_series(-3,3)i WHERE i<>0$$,
    '{"max_iterations":100,"learning_rate":0.2}');
SELECT public.assert_true(db4ai.predict_label('huge_numeric_labels',ARRAY[2.0],NULL::numeric)=2e400::numeric,'numeric label does not require float8 conversion');
SELECT db4ai.train('bool_reverse','logistic_regression',
    'SELECT true, 1.0 UNION ALL SELECT false,-1.0','{"max_iterations":100,"learning_rate":0.4,"tolerance":0}');
SELECT public.assert_true(db4ai.predict_bool('bool_reverse',ARRAY[1.0]) AND NOT db4ai.predict_bool('bool_reverse',ARRAY[-1.0]),'boolean label order');
SELECT public.assert_true(db4ai.predict_float8('bool_reverse',ARRAY[1.0])>0.9,'v0.1 probability-of-1 compatibility');
SELECT db4ai.train('numeric_reverse','logistic_regression',
    'SELECT 1,1.0 UNION ALL SELECT 0,-1.0','{"max_iterations":100,"learning_rate":0.4,"tolerance":0}');
SELECT public.assert_true(db4ai.predict_float8('numeric_reverse',ARRAY[1.0])>0.9,'numeric 0/1 label order');
CREATE TYPE public.db4ai_label_enum AS ENUM ('cold','warm','hot');
SELECT db4ai.train('enum_labels','logistic_regression',
    $$SELECT 'cold'::public.db4ai_label_enum,-1.0 UNION ALL SELECT 'hot'::public.db4ai_label_enum,1.0$$,
    '{"max_iterations":150,"learning_rate":0.3}');
SELECT public.assert_true(db4ai.predict_label('enum_labels',ARRAY[1.0],NULL::public.db4ai_label_enum)='hot','enum labels');
SELECT db4ai.train('timestamp_labels','logistic_regression',
    $$SELECT '2025-01-02 10:00+00'::timestamptz,-1.0 UNION ALL SELECT '2025-08-09 10:00+00'::timestamptz,1.0$$,
    '{"max_iterations":150,"learning_rate":0.3}');
SET timezone='America/Toronto'; SET datestyle='SQL,DMY';
SELECT public.assert_true(db4ai.predict_label('timestamp_labels',ARRAY[1.0],NULL::timestamptz)='2025-08-09 10:00+00'::timestamptz,'timestamp label durable I/O');
RESET timezone; RESET datestyle;

-- Array feature expansion, discarded-row dictionary correctness, NULL policy.
SELECT db4ai.train('arrays','linear_regression',
    'SELECT 1+2*x-3*x*x,ARRAY[x,x*x]::float4[] FROM (SELECT i::float8/50 x FROM generate_series(-50,50)i)s',
    '{"max_iterations":1000,"learning_rate":0.15,"tolerance":1e-12}');
SELECT public.assert_true(abs(db4ai.predict_float8('arrays',ARRAY[0.25,0.0625])-1.3125)<0.001,'array features');
SELECT db4ai.train('discard_labels','logistic_regression',
    $$SELECT 'discard',NULL::float8[] UNION ALL SELECT 'a',ARRAY[-1.0] UNION ALL SELECT 'b',ARRAY[1.0]$$,
    '{"null_policy":"legacy","max_iterations":50,"learning_rate":0.2}');
SELECT public.assert_true(cardinality(db4ai.class_labels('discard_labels'))=2 AND
    (db4ai.explain_model('discard_labels')->'metrics'->>'discarded_rows')::int=1,'discarded row creates no phantom class');
SELECT public.expect_error($q$SELECT db4ai.train('bad_array','linear_regression','SELECT 1,ARRAY[1.0]::float8[] UNION ALL SELECT 2,ARRAY[1.0,2.0]::float8[]')$q$,'42804');
SELECT public.expect_error($q$SELECT db4ai.train('bad_array','linear_regression','SELECT 1,ARRAY[1.0,NULL]::float8[]')$q$,'22004');

-- One-vs-all with either original binary learner; independent NGD scales/class.
CREATE TABLE public.multidata AS
SELECT (ARRAY['apple','pear','plum'])[i%3+1] label,
       (i%3=0)::int::float8 x0,(i%3=1)::int::float8 x1,(i%3=2)::int::float8 x2
FROM generate_series(0,89)i;
SELECT db4ai.train('ova_logistic','multiclass','SELECT * FROM public.multidata',
    '{"classifier":"logistic_regression","max_iterations":100,"learning_rate":0.3,"batch_size":9,"shuffle":"batch","shuffle_buffer_size":20,"tolerance":0}');
SELECT db4ai.train('ova_svm_ngd','multiclass','SELECT * FROM public.multidata',
    '{"classifier":"svm_classification","optimizer":"ngd","max_iterations":100,"learning_rate":0.5,"batch_size":9,"shuffle":"row","shuffle_buffer_size":7,"tolerance":0}');
SELECT public.assert_true(bool_and(db4ai.predict_text('ova_logistic',ARRAY[x0,x1,x2])=label AND
                                 db4ai.predict_text('ova_svm_ngd',ARRAY[x0,x1,x2])=label),'OVA logistic/SVM predictions') FROM public.multidata;
SELECT public.assert_true(cardinality(db4ai.predict_scores('ova_logistic',ARRAY[1.0,0.0,0.0]))=3,'OVA scores');
SELECT public.assert_true((db4ai.explain_model('ova_logistic')->'metrics'->>'accuracy')::float8=1,'multiclass accuracy');

-- Both original nonlinear random-feature transformations, not an RBF name alias.
CREATE TABLE public.xor_data AS SELECT (x*y<0)::int label,x::float8,y::float8 FROM (VALUES(-1),(1))a(x) CROSS JOIN (VALUES(-1),(1))b(y) CROSS JOIN generate_series(1,8);
SELECT db4ai.train('svm_gaussian','svm_classification','SELECT * FROM public.xor_data',
    '{"kernel":"gaussian","gamma":0.5,"components":128,"learning_rate":0.3,"max_iterations":200,"batch_size":8,"tolerance":0,"l2":0}');
SELECT db4ai.train('svm_polynomial','svm_classification','SELECT * FROM public.xor_data',
    '{"kernel":"polynomial","degree":2,"coef0":1,"components":128,"learning_rate":0.1,"max_iterations":100,"batch_size":8,"tolerance":0,"l2":0}');
SELECT db4ai.train('svm_homogeneous','svm_classification','SELECT * FROM public.xor_data',
    '{"kernel":"polynomial","degree":2,"coef0":0,"components":128,"learning_rate":0.1,"max_iterations":100,"batch_size":8,"tolerance":0,"l2":0}');
SELECT public.assert_true(bool_and(db4ai.predict_label('svm_gaussian',ARRAY[x,y],NULL::int)=label AND
                                  db4ai.predict_label('svm_polynomial',ARRAY[x,y],NULL::int)=label AND
                                  db4ai.predict_label('svm_homogeneous',ARRAY[x,y],NULL::int)=label),'nonlinear XOR classifiers') FROM public.xor_data;
SELECT public.expect_error($q$SELECT db4ai.train('bad_kernel','multiclass','SELECT * FROM public.multidata','{"classifier":"logistic_regression","kernel":"gaussian"}')$q$,'22023');

-- Seeding families and all original norms. K=1 gives an analytical centroid.
DO $$ DECLARE d text; expected float8; method text; BEGIN
  FOREACH d IN ARRAY ARRAY['L1','L2','L2_Squared','Linf'] LOOP
    expected=CASE d WHEN 'L1' THEN 3.5 WHEN 'L2' THEN 2.5 WHEN 'L2_Squared' THEN 6.25 ELSE 2 END;
    PERFORM db4ai.train('norm_'||d,'kmeans','SELECT 0.0,0.0 UNION ALL SELECT 3.0,4.0',
      jsonb_build_object('num_centroids',1,'distance_function',d,'seeding_function','Random++'));
    PERFORM public.assert_true(abs((db4ai.explain_model('norm_'||d)->'metrics'->>'training_loss')::float8-expected)<1e-12,'KMeans norm '||d);
  END LOOP;
  FOREACH method IN ARRAY ARRAY['Random++','KMeans||','reservoir'] LOOP
    PERFORM db4ai.train('seed_'||method,'kmeans',
      'SELECT (CASE WHEN i<20 THEN -10 ELSE 10 END)+i*0.001 FROM generate_series(0,39)i',
      jsonb_build_object('num_centroids',2,'num_features',1,'seeding_function',method));
    PERFORM public.assert_true(db4ai.predict_cluster('seed_'||method,ARRAY[-10.0])<>db4ai.predict_cluster('seed_'||method,ARRAY[10.0]),'KMeans seeding '||method);
  END LOOP;
END $$;
SELECT public.expect_error($q$SELECT db4ai.train('bad_features','kmeans','SELECT 1,2','{"num_features":3}')$q$,'22023');

-- Original matrix-free stochastic PCA plus retained, explicit Jacobi option.
SELECT db4ai.train('stochastic_pca','pca','SELECT i::float8,2*i::float8 FROM generate_series(-30,30)i',
    '{"profile":"opengauss","number_components":1,"max_iterations":20,"batch_size":8,"shuffle":"row","shuffle_buffer_size":16,"tolerance":0}');
SELECT public.assert_true(abs(abs((db4ai.transform('stochastic_pca',ARRAY[1.0,2.0]))[1])-sqrt(5.0))<1e-8,'stochastic PCA principal direction');
SELECT db4ai.train('stochastic_center','pca','SELECT 100+i::float8,200+2*i::float8 FROM generate_series(-30,30)i',
    '{"pca_solver":"stochastic","center":true,"n_components":2,"max_iterations":20,"batch_size":8,"tolerance":0}');
SELECT public.assert_true(abs((db4ai.transform('stochastic_center',ARRAY[100.0,200.0]))[1])<1e-8,'centered stochastic PCA');

-- Training shuffle: cache-fit / replacement, partial last batch, determinism.
CREATE SEQUENCE public.shuffle_train_reads;
SELECT db4ai.train('shuffle_full','linear_regression',
    $$SELECT x*x,x+0*nextval('public.shuffle_train_reads') FROM (SELECT i::float8/37 x FROM generate_series(1,37)i)s$$,
    '{"max_iterations":4,"batch_size":8,"shuffle_buffer_size":10,"tolerance":0,"learning_rate":0.1}');
SELECT public.assert_true((SELECT last_value=37 FROM public.shuffle_train_reads),'training shuffle runs source once');
SELECT public.assert_true((db4ai.explain_model('shuffle_full')->'metrics'->>'shuffle_full_cache')::bool,'full training batch cache');
SELECT db4ai.train('shuffle_bounded','linear_regression',
    'SELECT x*x,x FROM (SELECT i::float8/37 x FROM generate_series(1,37)i)s',
    '{"max_iterations":4,"batch_size":8,"shuffle_buffer_size":2,"tolerance":0,"learning_rate":0.1}');
SELECT db4ai.train('shuffle_bounded_again','linear_regression',
    'SELECT x*x,x FROM (SELECT i::float8/37 x FROM generate_series(1,37)i)s',
    '{"max_iterations":4,"batch_size":8,"shuffle_buffer_size":2,"tolerance":0,"learning_rate":0.1}');
SELECT public.assert_true(NOT (db4ai.explain_model('shuffle_bounded')->'metrics'->>'shuffle_full_cache')::bool,'bounded training cache');
SELECT public.assert_true(db4ai.export_model('shuffle_bounded')=db4ai.export_model('shuffle_bounded_again'),'deterministic batch shuffle seed');
SELECT db4ai.train('shuffle_other_seed','linear_regression',
    'SELECT x*x,x FROM (SELECT i::float8/37 x FROM generate_series(1,37)i)s',
    '{"max_iterations":4,"batch_size":8,"shuffle_buffer_size":2,"tolerance":0,"learning_rate":0.1,"seed":43}');
SELECT public.assert_true(db4ai.export_model('shuffle_bounded')<>db4ai.export_model('shuffle_other_seed'),'different shuffle seed changes training trajectory');

-- Bayes joint/selectivity estimation, categorical DAG and Gaussian parent.
SELECT db4ai.train('bayes_chain','bayes_net_internal',
    'SELECT x,y FROM (VALUES(0,0),(0,0),(0,1),(1,1))t(x,y)',
    '{"edges_of_network":"0,1","num_edges":1,"num_nodes":2}');
SELECT public.assert_true(abs(db4ai.predict_joint('bayes_chain',0,0)-0.5)<1e-14 AND
                         abs(db4ai.predict_joint('bayes_chain',0,1)-0.25)<1e-14 AND
                         db4ai.predict_joint('bayes_chain',1,0)=0,'Bayes analytical joint');
SELECT db4ai.train('bayes_mixed','bayes_net_internal',
    $$SELECT x,y FROM (VALUES ('yes'::text,true),('yes',true),(NULL::text,false),('no',false))t(x,y)$$,
    '{"edges_of_network":"0,1"}');
SELECT public.assert_true(abs(db4ai.predict_joint('bayes_mixed',NULL::text,false)-0.25)<1e-14,'Bayes NULL is observed category');
SELECT public.expect_error($q$SELECT db4ai.predict_joint('bayes_mixed',1,false)$q$,'42804');
SELECT db4ai.train('bayes_gaussian','bayes_net_internal',
    $$SELECT x,y,z FROM (VALUES ('a'::text,1,0.0::float8),('a',1,1.0::float8),('b',0,2.0::float8),('b',0,NULL::float8))t(x,y,z)$$,
    '{"edges_of_network":"0,1,2,1","num_edges":2}');
SELECT public.assert_true(abs(db4ai.predict_joint('bayes_gaussian','a'::text,1,0.0::float8)-0.17067237303427146)<1e-12 AND
                         abs(db4ai.predict_joint('bayes_gaussian','b'::text,0,NULL::float8)-0.25)<1e-14,'Gaussian parent CDF and NULL fraction');
SELECT db4ai.train('bayes_diamond','bayes_net_internal',
    'SELECT a,b,c,d FROM generate_series(0,1)a CROSS JOIN generate_series(0,1)b CROSS JOIN generate_series(0,1)c CROSS JOIN generate_series(0,1)d',
    '{"edges_of_network":"0,1,0,2,1,3,2,3","num_edges":4}');
SELECT public.assert_true(abs(db4ai.predict_joint('bayes_diamond',1,0,1,0)-0.0625)<1e-14,'diamond DAG is not a cycle');
SELECT db4ai.train('bayes_mcv','bayes_net_internal', $$SELECT CASE WHEN i<=8 THEN 'common' ELSE 'rare_'||i END FROM generate_series(1,10)i$$,
    '{"num_mcvs":1}');
SELECT public.assert_true(abs(db4ai.predict_joint('bayes_mcv','common'::text)-0.8)<1e-14 AND
                         abs(db4ai.predict_joint('bayes_mcv','rare_9'::text)-0.1)<1e-14,'MCV and other bucket distinct correction');
SELECT db4ai.train('bayes_bins','bayes_net_internal','SELECT i FROM generate_series(1,100)i', '{"num_bins":5}');
SELECT public.assert_true(abs(db4ai.predict_joint('bayes_bins',40)-0.01)<1e-14 AND db4ai.predict_joint('bayes_bins',101)=0,'numeric bins and outside-domain prediction');
SELECT db4ai.train('bayes_sample','bayes_net_internal','SELECT i FROM generate_series(1,100)i',
    '{"num_bins":5,"num_sample_rows":100,"num_total_rows":1000}');
SELECT public.assert_true(abs(db4ai.predict_joint('bayes_sample',10)-0.001)<1e-14,'Haas-Stokes sample distinct estimate');
SELECT db4ai.train('bayes_int64','bayes_net_internal','SELECT 9007199254740992::bigint UNION ALL SELECT 9007199254740993::bigint');
SELECT public.assert_true(db4ai.predict_joint('bayes_int64',9007199254740993::bigint)=0.5,'Bayes int64 identity');
SELECT public.expect_error($q$SELECT db4ai.train('cycle','bayes_net_internal','SELECT 0,1','{"edges_of_network":"0,1,1,0"}')$q$,'22023');
SELECT public.expect_error($q$SELECT db4ai.train('edge','bayes_net_internal','SELECT 0,1','{"edges_of_network":"0,1,0,1"}')$q$,'22023');
SELECT public.expect_error($q$SELECT db4ai.train('edge','bayes_net_internal','SELECT 0,1','{"edges_of_network":"0,2"}')$q$,'22023');
SELECT public.expect_error($q$SELECT db4ai.train('edge','bayes_net_internal','SELECT 0,1','{"edges_of_network":"0,1","num_edges":0}')$q$,'22023');
SELECT public.expect_error($q$SELECT db4ai.predict_float8('bayes_chain',ARRAY[0.0,1.0])$q$,'42804');

-- Real CustomScan, not a cosmetically renamed FunctionScan.
SELECT public.assert_true((public.plan_json($q$SELECT x*2 FROM db4ai.shuffle_scan('SELECT i FROM generate_series(1,30)i',42,3,1)s(x int) WHERE x>10$q$)->0->'Plan'->>'Custom Plan Provider')='ShuffleScan','CustomScan provider');
SELECT public.assert_true(count(*)=100 AND count(DISTINCT x)=100 AND sum(x)=5050,'bounded SQL shuffle row conservation')
FROM db4ai.shuffle_scan('SELECT i FROM generate_series(1,100)i',42,3,1)s(x int);
SELECT public.assert_true((SELECT array_agg(x) FROM db4ai.shuffle_scan('SELECT i FROM generate_series(1,100)i',42,200,1)s(x int))=
                         (SELECT array_agg(x) FROM db4ai.shuffle_scan('SELECT i FROM generate_series(1,100)i',42,200,1)s(x int)),'full SQL shuffle determinism');
SELECT public.assert_true((SELECT array_agg(x) FROM db4ai.shuffle_scan('SELECT i FROM generate_series(1,100)i',42,200,1)s(x int))<>
                         (SELECT array_agg(x) FROM db4ai.shuffle_scan('SELECT i FROM generate_series(1,100)i',42,200,2)s(x int)),'different epoch permutation');
SELECT public.assert_true((SELECT array_agg(x) FROM db4ai.shuffle_scan('SELECT i FROM generate_series(1,20)i',42,3,0)s(x int))=ARRAY(SELECT generate_series(1,20)),'epoch zero sequential');
SELECT public.assert_true(count(*)=50 AND count(t)=25 AND max(length(t))=500,'varlena and NULL shuffle')
FROM db4ai.shuffle_scan($q$SELECT i,CASE WHEN i%2=0 THEN repeat('x',500) ELSE NULL END FROM generate_series(1,50)i$q$,5,3,1)s(x int,t text);
SELECT public.assert_true((SELECT to_jsonb(s) FROM db4ai.shuffle_scan('SELECT 1, ''abc''::text',42,3,1)s(x int,t text))='{"x":1,"t":"abc"}'::jsonb,'whole-row FunctionScan fallback');
-- Dynamic arguments use the same implementation through ordinary SRF fallback.
SELECT public.assert_true(count(*)=30,'LATERAL/SRF fallback')
FROM (VALUES(10::bigint),(20::bigint),(30::bigint))v(seed)
CROSS JOIN LATERAL db4ai.shuffle_scan('SELECT i FROM generate_series(1,10)i',v.seed,3,1)s(x int);
CREATE SEQUENCE public.shuffle_rescans;
SET enable_material=off;
WITH RECURSIVE q(n) AS (
 SELECT 1 UNION ALL
 SELECT q.n+1 FROM q CROSS JOIN db4ai.shuffle_scan('SELECT nextval(''public.shuffle_rescans'')',42,4,1)s(v bigint) WHERE q.n<4
) SELECT public.assert_true(max(n)=4,'recursive rescan output') FROM q;
RESET enable_material;
SELECT public.assert_true((SELECT last_value=1 FROM public.shuffle_rescans),'CustomScan rescan does not rerun source SQL');
SET db4ai.work_mem='64kB';
SELECT public.assert_true(count(*)=5000 AND sum(x)=12502500,'SQL shuffle spilled input')
FROM db4ai.shuffle_scan('SELECT i FROM generate_series(1,5000)i',42,1000,1)s(x int);
SELECT public.assert_true((public.plan_json($q$SELECT * FROM db4ai.shuffle_scan('SELECT i FROM generate_series(1,5000)i',42,1000,1)s(x int)$q$)->0->'Plan'->>'Input Spilled')::bool,'CustomScan exposes spill');
RESET db4ai.work_mem;
PREPARE shuffle_prepared AS SELECT sum(x) FROM db4ai.shuffle_scan('SELECT i FROM generate_series(1,20)i',42,3,1)s(x int);
EXECUTE shuffle_prepared; EXECUTE shuffle_prepared; DEALLOCATE shuffle_prepared;
SELECT public.expect_error($q$SELECT * FROM db4ai.shuffle_scan('SELECT 1')s(x text)$q$,'42804');
SELECT public.expect_error($q$SELECT * FROM db4ai.shuffle_scan('DELETE FROM public.secret RETURNING x')s(x float8)$q$,'22023');
SELECT public.expect_error($q$SELECT * FROM db4ai.shuffle_scan('SELECT x FROM public.secret FOR UPDATE')s(x float8)$q$,'25006');
SELECT public.expect_error($q$SELECT * FROM db4ai.shuffle_scan('SELECT 1',42,0,1)s(x int)$q$,'22023');
REVOKE EXECUTE ON FUNCTION db4ai.shuffle_scan(text,bigint,int,int) FROM PUBLIC;
SET ROLE db4ai_alice;
SELECT public.expect_error($q$SELECT * FROM db4ai.shuffle_scan('SELECT 1',42,3,1)s(x int)$q$,'42501');
RESET ROLE;
GRANT EXECUTE ON FUNCTION db4ai.shuffle_scan(text,bigint,int,int) TO PUBLIC;
SET ROLE db4ai_alice;
SELECT public.expect_error($q$SELECT * FROM db4ai.shuffle_scan('SELECT x FROM public.secret',42,3,1)s(x float8)$q$,'42501');
SELECT public.expect_error($q$SELECT db4ai.predict_text('text_labels',ARRAY[1.0])$q$,'42704');
RESET ROLE;
-- Planner inspection must not execute a volatile source expression.
CREATE SEQUENCE public.shuffle_explain_only;
EXPLAIN (COSTS OFF) SELECT * FROM db4ai.shuffle_scan(
  'SELECT nextval(''public.shuffle_explain_only'')',42,3,1)s(x bigint);
SELECT public.assert_true(NOT (SELECT is_called FROM public.shuffle_explain_only),'EXPLAIN-only does not run source');

-- Source RLS applies even though both APIs execute server-side C code.
CREATE TABLE public.shuffle_rls(owner_name name, x int);
INSERT INTO public.shuffle_rls VALUES ('db4ai_alice',1),('db4ai_bob',2);
ALTER TABLE public.shuffle_rls ENABLE ROW LEVEL SECURITY;
CREATE POLICY per_user ON public.shuffle_rls USING (owner_name=current_user);
GRANT SELECT ON public.shuffle_rls TO db4ai_alice;
SET ROLE db4ai_alice;
SELECT public.assert_true((SELECT array_agg(x) FROM db4ai.shuffle_scan(
  'SELECT x FROM public.shuffle_rls',42,8,1)s(x int))=ARRAY[1],'ShuffleScan caller RLS');
SELECT db4ai.train('rls_native','linear_regression','SELECT x,x FROM public.shuffle_rls','{"max_iterations":1}');
SELECT public.assert_true((db4ai.explain_model('rls_native')->>'rows_processed')::bigint=1,'training caller RLS');
RESET ROLE;
SELECT public.expect_error($q$SELECT db4ai.train('pca_bad_option','pca','SELECT i,i FROM generate_series(1,10)i','{"pca_solver":"jacobi","shuffle":"row"}')$q$,'22023');
SELECT 'extended algorithms and ShuffleScan tests passed' result;
