\echo Use "ALTER EXTENSION pg_db4ai UPDATE TO '0.2.0'" to load this file. \quit

-- 0.2: native category dictionaries, Bayes selectivity and an explicit ShuffleScan.
CREATE FUNCTION db4ai.predict_text(model_name text, features double precision[])
RETURNS text AS 'MODULE_PATHNAME', 'pg_db4ai_predict_text'
LANGUAGE C STRICT STABLE PARALLEL UNSAFE SECURITY INVOKER;

CREATE FUNCTION db4ai.predict_label(model_name text, features double precision[], type_witness anyelement)
RETURNS anyelement AS 'MODULE_PATHNAME', 'pg_db4ai_predict_label'
LANGUAGE C STABLE PARALLEL UNSAFE SECURITY INVOKER;

CREATE FUNCTION db4ai.predict_scores(model_name text, features double precision[])
RETURNS double precision[] AS 'MODULE_PATHNAME', 'pg_db4ai_predict_scores'
LANGUAGE C STRICT STABLE PARALLEL UNSAFE SECURITY INVOKER;

CREATE FUNCTION db4ai.class_labels(model_name text)
RETURNS text[] AS 'MODULE_PATHNAME', 'pg_db4ai_class_labels'
LANGUAGE C STRICT STABLE PARALLEL UNSAFE SECURITY INVOKER;

CREATE FUNCTION db4ai.predict_joint(model_name text, VARIADIC variables "any")
RETURNS double precision AS 'MODULE_PATHNAME', 'pg_db4ai_predict_joint'
LANGUAGE C STABLE PARALLEL UNSAFE SECURITY INVOKER;

CREATE FUNCTION db4ai.shuffle_scan(source_sql text, seed bigint DEFAULT 42,
                                 buffer_rows integer DEFAULT 1024, epoch integer DEFAULT 1)
RETURNS SETOF record AS 'MODULE_PATHNAME', 'pg_db4ai_shuffle_scan'
LANGUAGE C VOLATILE PARALLEL UNSAFE SECURITY INVOKER COST 100 ROWS 1000;

COMMENT ON FUNCTION db4ai.predict_label(text, double precision[], anyelement) IS
  'Typed original classification label. Pass NULL::label_type as the third argument.';
COMMENT ON FUNCTION db4ai.predict_scores(text, double precision[]) IS
  'Scores in class_labels order. OVA logistic scores are not softmax probabilities; SVM scores are margins.';
COMMENT ON FUNCTION db4ai.predict_joint(text, "any") IS
  'Bayes equality joint/selectivity estimate; SQL NULL is an observed NULL category, not a wildcard.';
COMMENT ON FUNCTION db4ai.shuffle_scan(text,bigint,integer,integer) IS
  'Fixed-input seeded row shuffle. Epoch 0 is sequential. Bounded replacement is not globally uniform. LOAD pg_db4ai before planning to enable CustomScan.';
