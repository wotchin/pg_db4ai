\set ON_ERROR_STOP 1
-- Same persisted model, backend, role and tuple shape; GUC can change safely.
SET db4ai.simd=off;
SELECT public.assert_true(db4ai.build_info()->>'simd_backend'='off','scalar-source mode selected');
CREATE TEMP TABLE simd_rows AS
 SELECT i, ARRAY(SELECT sin((i+j*13)*0.017)::float8 FROM generate_series(1,31)j) AS x,
        sin((i+13)*0.017)*2+sin((i+26)*0.017) AS y
 FROM generate_series(1,129)i;
SELECT db4ai.train('simd_reg_off','linear_regression','SELECT y,x FROM simd_rows ORDER BY i',
 '{"max_iterations":40,"learning_rate":0.01,"tolerance":0,"batch_size":17}');
CREATE TEMP TABLE simd_expected AS SELECT i,db4ai.predict_float8('simd_reg_off',x) AS p FROM simd_rows;
SET db4ai.simd=portable;
SELECT public.assert_true((SELECT max(abs(p-db4ai.predict_float8('simd_reg_off',x)))<1e-11
 FROM simd_expected JOIN simd_rows USING(i)), 'portable same-model dot tail');
SET db4ai.simd=auto;
SELECT public.assert_true((SELECT max(abs(p-db4ai.predict_float8('simd_reg_off',x)))<1e-11
 FROM simd_expected JOIN simd_rows USING(i)), 'runtime SIMD same-model dot tail');
SELECT db4ai.train('simd_reg_auto','linear_regression','SELECT y,x FROM simd_rows ORDER BY i',
 '{"max_iterations":40,"learning_rate":0.01,"tolerance":0,"batch_size":17}');
SELECT public.assert_true((SELECT max(abs(p-db4ai.predict_float8('simd_reg_auto',x)))<1e-9
 FROM simd_expected JOIN simd_rows USING(i)), 'reduction training error bounded');
BEGIN;
SET LOCAL db4ai.simd=off;
SELECT public.assert_true(db4ai.build_info()->>'simd_backend'='off','local mode');
ROLLBACK;
SELECT public.assert_true(db4ai.build_info()->>'simd_backend' IN ('portable','avx2','off'),'mode reset callback');
SELECT public.expect_error($q$SET db4ai.simd='avx512'$q$, '22023');
SELECT public.expect_error($q$SELECT db4ai.predict_float8('simd_reg_off',array_fill('NaN'::float8,ARRAY[31]))$q$,'22003');
SELECT public.expect_error($q$SELECT db4ai.predict_float8('simd_reg_off',array_fill('Infinity'::float8,ARRAY[31]))$q$,'22003');
SELECT public.assert_true((SELECT count(*)=129 FROM simd_rows WHERE db4ai.predict_float8('simd_reg_off',x) IS NOT NULL),'usable after invalid inputs');
RESET db4ai.simd;
