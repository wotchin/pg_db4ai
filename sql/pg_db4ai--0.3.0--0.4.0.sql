-- RESTRICT deliberately refuses to destroy user views depending on the old API.
DROP FUNCTION db4ai.predict_joint(text,"any");
-- No DELETE/UPDATE of pg_db4ai_models: old artifacts remain exportable/deletable.
COMMENT ON FUNCTION db4ai.predict_scores(text,double precision[]) IS
 'Scores in class_labels order. Gaussian NB: normalized posterior; OVA: independent scores, not softmax; SVM: margins.';

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
