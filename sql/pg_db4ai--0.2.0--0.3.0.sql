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
