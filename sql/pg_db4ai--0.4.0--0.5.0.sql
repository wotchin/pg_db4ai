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
