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
