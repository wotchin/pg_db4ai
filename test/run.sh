#!/bin/sh
# Isolated integration tests. Run after make install, as a non-root OS user.
set -eu
if [ "$(id -u)" = 0 ]; then
  echo 'Run tests as a non-root OS user; PostgreSQL refuses to run as root.' >&2
  exit 1
fi
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BIN=$("${PG_CONFIG:-pg_config}" --bindir)
WORK=$(mktemp -d "${TMPDIR:-/tmp}/pg-db4ai.XXXXXXXX")
cleanup() {
  "$BIN/pg_ctl" -D "$WORK/data" -m immediate stop >/dev/null 2>&1 || true
  if [ "${KEEP_TEST_DATA:-0}" = 1 ]; then echo "Test files: $WORK"; else rm -rf "$WORK"; fi
}
trap cleanup EXIT HUP INT TERM
"$BIN/initdb" -D "$WORK/data" -A trust --no-locale -E UTF8 -U db4ai_test >/dev/null
"$BIN/pg_ctl" -D "$WORK/data" -l "$WORK/server.log" \
  -o "-F -k $WORK -c listen_addresses='' -p 65432" -w start >/dev/null
export PGHOST="$WORK" PGPORT=65432 PGUSER=db4ai_test PGDATABASE=postgres
"$BIN/psql" -X -v ON_ERROR_STOP=1 -f "$ROOT/test/native.sql"
"$BIN/psql" -X -v ON_ERROR_STOP=1 -f "$ROOT/test/extended.sql"
"$BIN/psql" -X -v ON_ERROR_STOP=1 -f "$ROOT/test/naive_bayes.sql"
python3 "$ROOT/test/naive_bayes_reference.py"
"$BIN/psql" -X -v ON_ERROR_STOP=1 -f "$ROOT/test/shufflescan_boundaries.sql"
"$BIN/psql" -X -v ON_ERROR_STOP=1 -f "$ROOT/test/corgipile_ddl.sql"
python3 "$ROOT/test/numerical_reference.py"
python3 "$ROOT/test/partial_batch_reference.py"
"$BIN/psql" -X -v ON_ERROR_STOP=1 -f "$ROOT/test/simd.sql"
sh "$ROOT/test/run_numeric.sh"
"$BIN/createdb" db4ai_upgrade
"$BIN/psql" -X -d db4ai_upgrade -v ON_ERROR_STOP=1 -f "$ROOT/test/upgrade.sql"
"$BIN/createdb" db4ai_bayes_upgrade
"$BIN/psql" -X -d db4ai_bayes_upgrade -v ON_ERROR_STOP=1 -f "$ROOT/test/naive_bayes_upgrade.sql"
if [ "${WITH_XGBOOST:-1}" = 1 ]; then
  "$BIN/psql" -X -v ON_ERROR_STOP=1 -f "$ROOT/test/xgboost.sql"
fi
# A new backend must load the durable model, rather than use an in-process pointer.
"$BIN/psql" -X -v ON_ERROR_STOP=1 -c "SELECT public.assert_true(abs(db4ai.predict_float8('reg',ARRAY[0.25,0.0625])-1.3125)<0.001,'new backend reload');"
# Timeout must not publish a partial model.
if "$BIN/psql" -X -v ON_ERROR_STOP=1 -c "SET statement_timeout='50ms'; SELECT db4ai.train('cancelled','linear_regression','SELECT 1,1 FROM pg_sleep(10)');" >"$WORK/cancel.log" 2>&1; then
  echo 'Expected statement timeout' >&2; exit 1
fi
grep -q 'statement timeout' "$WORK/cancel.log"
"$BIN/psql" -X -v ON_ERROR_STOP=1 -c "SELECT public.assert_true(NOT EXISTS(SELECT FROM db4ai.pg_db4ai_models WHERE model_name='cancelled'),'cancel cleanup');"
# Compare all model payloads across dump/restore. Roles already exist in this cluster.
"$BIN/psql" -X -At -c "SELECT owner_oid::text,model_name,md5(model_data) FROM db4ai.pg_db4ai_models ORDER BY 1,2" >"$WORK/before"
"$BIN/pg_dump" -Fc -f "$WORK/models.dump" postgres
"$BIN/createdb" db4ai_restored
"$BIN/pg_restore" --exit-on-error --dbname=db4ai_restored "$WORK/models.dump"
"$BIN/psql" -X -At -d db4ai_restored -c "SELECT owner_oid::text,model_name,md5(model_data) FROM db4ai.pg_db4ai_models ORDER BY 1,2" >"$WORK/after"
cmp "$WORK/before" "$WORK/after"
"$BIN/psql" -X -d db4ai_restored -v ON_ERROR_STOP=1 <<'SQL'
SELECT public.assert_true(abs(db4ai.predict_float8('reg',ARRAY[0.25,0.0625])-1.3125)<0.001,'restored predictor');
SELECT db4ai.train('post_restore','linear_regression','SELECT 1,1');
SELECT public.assert_true(db4ai.predict_text('text_labels',ARRAY[2.0])='blue','restored typed labels');
SELECT public.assert_true(db4ai.predict_label('numeric_labels',ARRAY[2.0],NULL::numeric)=9007199254740993::numeric,'restored numeric label');
SELECT public.assert_true(db4ai.predict_label('enum_labels',ARRAY[1.0],NULL::public.db4ai_label_enum)='hot','restored enum label');
SELECT public.assert_true(db4ai.predict_text('ova_svm_ngd',ARRAY[0.0,1.0,0.0])='pear','restored multiclass NGD');
SQL
if [ "${WITH_XGBOOST:-1}" = 1 ]; then
  "$BIN/psql" -X -d db4ai_restored -v ON_ERROR_STOP=1 -c "SELECT public.assert_true(abs(db4ai.predict_float8('xgb_ubj',ARRAY[0.25,0.0625])-1.3125)<0.3,'restored XGBoost predictor');"
fi
# Both kernel maps and PCA bases must remain exactly usable after a new backend
# and logical restore, independently of in-memory handles.
PGDATABASE=db4ai_restored python3 "$ROOT/test/numerical_reference.py"
PGDATABASE=db4ai_restored python3 "$ROOT/test/naive_bayes_reference.py"
printf '\nAll integration tests passed (PostgreSQL: %s; WITH_XGBOOST=%s).\n' "$("$BIN/pg_config" --version)" "${WITH_XGBOOST:-1}"
