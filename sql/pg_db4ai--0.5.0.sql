\echo Use "CREATE EXTENSION pg_db4ai" to load this file. \quit

CREATE TABLE db4ai.pg_db4ai_models (
    model_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    owner_oid regrole NOT NULL,
    model_name text NOT NULL CHECK (octet_length(model_name) BETWEEN 1 AND 128),
    algorithm text NOT NULL,
    format_version integer NOT NULL DEFAULT 1,
    model_format text NOT NULL CHECK (model_format IN ('json', 'ubj')),
    n_features integer NOT NULL CHECK (n_features BETWEEN 1 AND 1024),
    n_outputs integer NOT NULL CHECK (n_outputs BETWEEN 1 AND 256),
    feature_names text[] NOT NULL,
    hyperparameters jsonb NOT NULL,
    metrics jsonb NOT NULL,
    model_data bytea NOT NULL,
    training_sql text NOT NULL,
    rows_processed bigint NOT NULL CHECK (rows_processed > 0),
    created_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    UNIQUE (owner_oid, model_name)
);

COMMENT ON TABLE db4ai.pg_db4ai_models IS
  'Immutable, role-owned model artifacts. Only extension functions write this ordinary WAL-logged table.';
COMMENT ON COLUMN db4ai.pg_db4ai_models.owner_oid IS
  'regrole preserves symbolic role names in logical dumps instead of dumping raw OIDs.';

ALTER TABLE db4ai.pg_db4ai_models ENABLE ROW LEVEL SECURITY;
CREATE POLICY db4ai_owner_select ON db4ai.pg_db4ai_models FOR SELECT TO PUBLIC
    USING (owner_oid = CURRENT_USER::pg_catalog.regrole);
GRANT USAGE ON SCHEMA db4ai TO PUBLIC;
GRANT SELECT ON db4ai.pg_db4ai_models TO PUBLIC;
REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON db4ai.pg_db4ai_models FROM PUBLIC;

SELECT pg_catalog.pg_extension_config_dump('db4ai.pg_db4ai_models', '');
SELECT pg_catalog.pg_extension_config_dump(
    pg_catalog.pg_get_serial_sequence('db4ai.pg_db4ai_models', 'model_id')::regclass, '');

CREATE FUNCTION db4ai.train(model_name text, algorithm text, training_sql text,
                            hyperparameters jsonb DEFAULT '{}'::jsonb,
                            model_format text DEFAULT NULL)
RETURNS bigint AS 'MODULE_PATHNAME', 'pg_db4ai_train'
LANGUAGE C VOLATILE PARALLEL UNSAFE SECURITY INVOKER;

CREATE FUNCTION db4ai.predict_float8(model_name text, features double precision[])
RETURNS double precision AS 'MODULE_PATHNAME', 'pg_db4ai_predict_float8'
LANGUAGE C STRICT STABLE PARALLEL UNSAFE SECURITY INVOKER;

CREATE FUNCTION db4ai.predict_bool(model_name text, features double precision[])
RETURNS boolean AS 'MODULE_PATHNAME', 'pg_db4ai_predict_bool'
LANGUAGE C STRICT STABLE PARALLEL UNSAFE SECURITY INVOKER;

CREATE FUNCTION db4ai.predict_cluster(model_name text, features double precision[])
RETURNS integer AS 'MODULE_PATHNAME', 'pg_db4ai_predict_cluster'
LANGUAGE C STRICT STABLE PARALLEL UNSAFE SECURITY INVOKER;

CREATE FUNCTION db4ai.transform(model_name text, features double precision[])
RETURNS double precision[] AS 'MODULE_PATHNAME', 'pg_db4ai_transform'
LANGUAGE C STRICT STABLE PARALLEL UNSAFE SECURITY INVOKER;

CREATE FUNCTION db4ai.drop_model(model_name text)
RETURNS boolean AS 'MODULE_PATHNAME', 'pg_db4ai_drop_model'
LANGUAGE C STRICT VOLATILE PARALLEL UNSAFE SECURITY INVOKER;

CREATE FUNCTION db4ai.export_model(model_name text)
RETURNS bytea AS 'MODULE_PATHNAME', 'pg_db4ai_export_model'
LANGUAGE C STRICT STABLE PARALLEL UNSAFE SECURITY INVOKER;

CREATE FUNCTION db4ai.explain_model(model_name text)
RETURNS jsonb LANGUAGE SQL STRICT STABLE PARALLEL UNSAFE SECURITY INVOKER
SET search_path = pg_catalog, pg_temp
AS $$
    SELECT pg_catalog.jsonb_build_object(
        'model_id', m.model_id, 'model_name', m.model_name,
        'algorithm', m.algorithm, 'format_version', m.format_version,
        'model_format', m.model_format, 'n_features', m.n_features,
        'n_outputs', m.n_outputs, 'feature_names', m.feature_names,
        'hyperparameters', m.hyperparameters, 'metrics', m.metrics,
        'rows_processed', m.rows_processed, 'model_bytes', pg_catalog.octet_length(m.model_data))
    FROM db4ai.pg_db4ai_models m
    WHERE m.owner_oid = CURRENT_USER::pg_catalog.regrole AND m.model_name = $1
$$;

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

CREATE FUNCTION db4ai.shuffle_scan(source_sql text, seed bigint DEFAULT 42,
                                 buffer_rows integer DEFAULT 1024, epoch integer DEFAULT 1)
RETURNS SETOF record AS 'MODULE_PATHNAME', 'pg_db4ai_shuffle_scan'
LANGUAGE C VOLATILE PARALLEL UNSAFE SECURITY INVOKER COST 100 ROWS 1000;

COMMENT ON FUNCTION db4ai.predict_label(text, double precision[], anyelement) IS
  'Typed original classification label. Pass NULL::label_type as the third argument.';
COMMENT ON FUNCTION db4ai.predict_scores(text, double precision[]) IS
  'Scores in class_labels order. Gaussian NB: normalized posterior; OVA: independent scores, not softmax; SVM: margins.';
COMMENT ON FUNCTION db4ai.shuffle_scan(text,bigint,integer,integer) IS
  'Fixed-input seeded row shuffle. Epoch 0 is sequential. Bounded replacement is not globally uniform. LOAD pg_db4ai before planning to enable CustomScan.';

-- Model-DDL compatibility without changing PostgreSQL's top-level grammar.
CREATE FUNCTION db4ai.execute_ddl(statement text)
RETURNS jsonb AS 'MODULE_PATHNAME', 'pg_db4ai_execute_ddl'
LANGUAGE C STRICT VOLATILE PARALLEL UNSAFE SECURITY INVOKER;
CREATE FUNCTION db4ai.parse_ddl(statement text)
RETURNS jsonb AS 'MODULE_PATHNAME', 'pg_db4ai_parse_ddl'
LANGUAGE C STRICT STABLE PARALLEL UNSAFE SECURITY INVOKER;
COMMENT ON FUNCTION db4ai.execute_ddl(text) IS
  'Explicit CREATE/DROP/EXPLAIN MODEL compatibility parser. Not a stock-PG raw grammar hook. Executes as caller.';
COMMENT ON FUNCTION db4ai.parse_ddl(text) IS
  'Parse one model DDL into a structured request without executing training SQL or accessing data.';

CREATE FUNCTION db4ai.shuffle_indices(row_count integer, options jsonb DEFAULT '{}'::jsonb, epoch integer DEFAULT 1)
RETURNS bigint[] AS 'MODULE_PATHNAME', 'pg_db4ai_shuffle_indices'
LANGUAGE C STRICT VOLATILE PARALLEL UNSAFE SECURITY INVOKER;
COMMENT ON FUNCTION db4ai.shuffle_indices(integer,jsonb,integer) IS
  'Bounded diagnostic for the actual training shuffle iterator. Returns 0-based indexes (maximum 100000), not a dataset.';

CREATE FUNCTION db4ai.build_info() RETURNS jsonb
AS 'MODULE_PATHNAME', 'pg_db4ai_build_info'
LANGUAGE C STABLE PARALLEL SAFE SECURITY INVOKER;

-- User-facing supervised Gaussian Naive Bayes. Never reinterpret legacy Bayes artifacts.
CREATE FUNCTION db4ai.predict_proba(model_name text, features double precision[])
RETURNS double precision[] AS 'MODULE_PATHNAME', 'pg_db4ai_predict_proba'
LANGUAGE C STRICT STABLE PARALLEL UNSAFE SECURITY INVOKER;
CREATE FUNCTION db4ai.predict_log_proba(model_name text, features double precision[])
RETURNS double precision[] AS 'MODULE_PATHNAME', 'pg_db4ai_predict_log_proba'
LANGUAGE C STRICT STABLE PARALLEL UNSAFE SECURITY INVOKER;
COMMENT ON FUNCTION db4ai.predict_proba(text,double precision[]) IS
 'Gaussian NB posterior probabilities in class_labels order, summing to one; not calibrated confidence.';
COMMENT ON FUNCTION db4ai.predict_log_proba(text,double precision[]) IS
 'Gaussian NB normalized log posteriors; preserves small probabilities without exp underflow.';

-- Bounded model selection. Source SELECT executes as caller, not extension owner.
CREATE FUNCTION db4ai.automl(model_name text, task text, training_sql text,
                            options jsonb DEFAULT '{}'::jsonb)
RETURNS jsonb AS 'MODULE_PATHNAME', 'pg_db4ai_automl'
LANGUAGE C VOLATILE PARALLEL UNSAFE SECURITY INVOKER;
COMMENT ON FUNCTION db4ai.automl(text,text,text,jsonb) IS
 'Budgeted portfolio search with train-only preprocessing, disjoint validation/test, and transactional publication. No model is published without validation improvement.';
CREATE FUNCTION db4ai.predict(model_name text, features jsonb)
RETURNS jsonb AS 'MODULE_PATHNAME', 'pg_db4ai_predict'
LANGUAGE C STRICT STABLE PARALLEL UNSAFE SECURITY INVOKER;
COMMENT ON FUNCTION db4ai.predict(text,jsonb) IS
 'Predict from named raw AutoML features using the persisted preprocessing. Classification labels are text with explicit class order; probabilities are not calibrated confidence.';

COMMENT ON FUNCTION db4ai.predict_proba(text,double precision[]) IS
 'Gaussian NB or AutoML normalized probabilities in class_labels order; not calibrated confidence. AutoML inputs are original numerical features.';
COMMENT ON FUNCTION db4ai.predict_log_proba(text,double precision[]) IS
 'Standalone NB: stable log posterior. AutoML: log of normalized probabilities; zero maps to -Infinity (not clipped).';
COMMENT ON FUNCTION db4ai.predict_scores(text,double precision[]) IS
 'AutoML and Gaussian NB: normalized probabilities; standalone OVA: independent scores; standalone SVM: margins.';
