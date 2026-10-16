-- Public int4 epoch and int8 seed boundaries; cast before adding one.
LOAD 'pg_db4ai';
SELECT public.assert_true(count(*)=100 AND count(DISTINCT x)=100 AND sum(x)=5050,
  'maximum epoch and minimum seed preserve all rows')
FROM db4ai.shuffle_scan('SELECT i FROM generate_series(1,100)i',
  '-9223372036854775808'::bigint,3,2147483647)s(x int);
SELECT public.assert_true(
  (SELECT array_agg(x) FROM db4ai.shuffle_scan('SELECT i FROM generate_series(1,100)i',
    9223372036854775807,200,2147483647)s(x int))=
  (SELECT array_agg(x) FROM db4ai.shuffle_scan('SELECT i FROM generate_series(1,100)i',
    9223372036854775807,200,2147483647)s(x int)),
  'maximum seed and epoch remain deterministic');
SELECT public.assert_true(count(*)=200 AND count(DISTINCT (seed,x))=200,
  'SRF fallback supports extreme seed and epoch')
FROM (VALUES('-9223372036854775808'::bigint),(9223372036854775807::bigint))v(seed)
CROSS JOIN LATERAL db4ai.shuffle_scan('SELECT i FROM generate_series(1,100)i',
  v.seed,3,2147483647)s(x int);
CREATE SEQUENCE public.shuffle_boundary_rescans;
SET enable_material=off;
WITH RECURSIVE q(n) AS (
 SELECT 1 UNION ALL
 SELECT q.n+1 FROM q CROSS JOIN db4ai.shuffle_scan(
   'SELECT nextval(''public.shuffle_boundary_rescans'')',
   9223372036854775807,4,2147483647)s(v bigint) WHERE q.n<4
) SELECT public.assert_true(max(n)=4,'maximum epoch rescan output') FROM q;
RESET enable_material;
SELECT public.assert_true((SELECT last_value=1 FROM public.shuffle_boundary_rescans),
  'maximum epoch rescan replays one source execution');
