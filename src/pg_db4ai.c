/* SPDX-License-Identifier: MulanPSL-2.0 */
#include "postgres.h"
#include "db4ai.h"
#include "access/xact.h"
#include "catalog/pg_type_d.h"
#include "lib/stringinfo.h"
#include "miscadmin.h"
#include "utils/builtins.h"
#include "utils/fmgrprotos.h"
#include "utils/guc.h"
#include "utils/timestamp.h"

PG_MODULE_MAGIC;
PG_FUNCTION_INFO_V1(pg_db4ai_train);
PG_FUNCTION_INFO_V1(pg_db4ai_predict_float8);
PG_FUNCTION_INFO_V1(pg_db4ai_predict_bool);
PG_FUNCTION_INFO_V1(pg_db4ai_predict_cluster);
PG_FUNCTION_INFO_V1(pg_db4ai_transform);
PG_FUNCTION_INFO_V1(pg_db4ai_drop_model);
PG_FUNCTION_INFO_V1(pg_db4ai_export_model);

int db4ai_work_mem = 16384;
int db4ai_xgboost_max_input_mb = 128;
int db4ai_max_model_mb = 64;
int db4ai_max_rows = 10000000;

void _PG_init(void);
void
_PG_init(void)
{
    DefineCustomIntVariable("db4ai.work_mem", "Memory for normalized training tuplestore, in kB.", NULL,
                            &db4ai_work_mem, 16384, 64, 1048576, PGC_USERSET, GUC_UNIT_KB, NULL, NULL, NULL);
    DefineCustomIntVariable("db4ai.xgboost_max_input_mb", "Maximum dense XGBoost input copy, in MB.", NULL,
                            &db4ai_xgboost_max_input_mb, 128, 1, 1024, PGC_SUSET, 0, NULL, NULL, NULL);
    DefineCustomIntVariable("db4ai.max_model_mb", "Maximum serialized model size, in MB.", NULL,
                            &db4ai_max_model_mb, 64, 1, 512, PGC_SUSET, 0, NULL, NULL, NULL);
    DefineCustomIntVariable("db4ai.max_rows", "Maximum training input rows per call.", NULL,
                            &db4ai_max_rows, 10000000, 1, 100000000, PGC_SUSET, 0, NULL, NULL, NULL);
}

static char *
model_name(FunctionCallInfo fcinfo)
{
    char *name = text_to_cstring(PG_GETARG_TEXT_PP(0));
    if (*name == '\0' || strlen(name) > 128)
        ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("model name must contain 1 to 128 bytes")));
    return name;
}

Datum
pg_db4ai_train(PG_FUNCTION_ARGS)
{
    MemoryContext parent = CurrentMemoryContext;
    MemoryContext context;
    Oid caller = GetUserId();
    char *name;
    char *query;
    char *format;
    Db4aiAlgorithm algorithm;
    Db4aiOptions options;
    volatile int64 id = 0;
    int i;
    for (i = 0; i < 4; i++)
        if (PG_ARGISNULL(i))
            ereport(ERROR, (errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED), errmsg("training arguments must not be NULL")));
    name = model_name(fcinfo);
    query = text_to_cstring(PG_GETARG_TEXT_PP(2));
    algorithm = db4ai_algorithm(text_to_cstring(PG_GETARG_TEXT_PP(1)));
    db4ai_options(PG_GETARG_JSONB_P(3), algorithm, &options);
    format = PG_ARGISNULL(4) ? (db4ai_is_xgboost(algorithm) ? "ubj" : "json") : text_to_cstring(PG_GETARG_TEXT_PP(4));
    if ((strcmp(format, "json") != 0 && strcmp(format, "ubj") != 0) ||
        (!db4ai_is_xgboost(algorithm) && strcmp(format, "json") != 0))
        ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("native models require json; XGBoost supports json or ubj")));
    context = AllocSetContextCreate(parent, "DB4AI training", ALLOCSET_DEFAULT_SIZES);
    PG_TRY();
    {
        Db4aiData *data;
        Db4aiModel model;
        MemoryContextSwitchTo(context);
        memset(&model, 0, sizeof(model));
        data = db4ai_data_open(query, algorithm);
        model.algorithm = algorithm;
        model.nfeatures = data->nfeatures;
        if (db4ai_is_xgboost(algorithm))
            db4ai_xgb_train(data, algorithm, &options, format, &model);
        else
        {
            StringInfoData metrics;
            Db4aiNative *native = db4ai_native_train(data, algorithm, &options);
            char *loss = DatumGetCString(DirectFunctionCall1(float8out, Float8GetDatum(native->loss)));
            model.data = db4ai_native_save(native);
            model.format = "json";
            model.noutputs = native->noutputs;
            initStringInfo(&metrics);
            appendStringInfo(&metrics, "{\"rows\":" INT64_FORMAT ",\"iterations\":%d,\"training_loss\":%s}",
                              data->nrows, native->iterations, loss);
            model.metrics = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(metrics.data)));
        }
        id = db4ai_model_store(name, query, data, &model, db4ai_options_json(&options, algorithm), caller);
        db4ai_data_close(data);
        MemoryContextSwitchTo(parent);
        MemoryContextDelete(context);
    }
    PG_CATCH();
    {
        MemoryContextSwitchTo(parent);
        MemoryContextDelete(context);
        PG_RE_THROW();
    }
    PG_END_TRY();
    PG_RETURN_INT64(id);
}

typedef struct Db4aiCache
{
    MemoryContext context;
    TimestampTz statement;
    CommandId command_id;
    Oid role;
    char *name;
    Db4aiModel *model;
    Db4aiNative *native;
    void *xgb;
} Db4aiCache;

static Db4aiCache *
get_predictor(FunctionCallInfo fcinfo)
{
    Db4aiCache *cache = (Db4aiCache *) fcinfo->flinfo->fn_extra;
    MemoryContext caller = CurrentMemoryContext;
    char *name = model_name(fcinfo);
    TimestampTz statement = GetCurrentStatementStartTimestamp();
    Oid role = GetUserId();
    CommandId command_id = GetCurrentCommandId(false);
    if (cache == NULL)
    {
        cache = MemoryContextAllocZero(fcinfo->flinfo->fn_mcxt, sizeof(*cache));
        fcinfo->flinfo->fn_extra = cache;
    }
    if (cache->context != NULL && cache->role == role && cache->statement == statement && cache->command_id == command_id && strcmp(cache->name, name) == 0)
    {
        pfree(name);
        return cache;
    }
    if (cache->context != NULL)
        MemoryContextDelete(cache->context);
    cache->context = AllocSetContextCreate(fcinfo->flinfo->fn_mcxt, "DB4AI predictor", ALLOCSET_DEFAULT_SIZES);
    PG_TRY();
    {
        MemoryContextSwitchTo(cache->context);
        cache->name = pstrdup(name);
        cache->model = db4ai_model_load(name, role);
        cache->native = NULL;
        cache->xgb = NULL;
        if (db4ai_is_xgboost(cache->model->algorithm))
            cache->xgb = db4ai_xgb_load(cache->model);
        else
            cache->native = db4ai_native_load(cache->model);
        cache->role = role;
        cache->statement = statement;
        cache->command_id = command_id;
        MemoryContextSwitchTo(caller);
    }
    PG_CATCH();
    {
        MemoryContextSwitchTo(caller);
        MemoryContextDelete(cache->context);
        cache->context = NULL;
        PG_RE_THROW();
    }
    PG_END_TRY();
    pfree(name);
    return cache;
}

static double
predict(Db4aiCache *cache, ArrayType *array)
{
    double *features = db4ai_features(array, cache->model->nfeatures);
    double result = cache->xgb != NULL ? db4ai_xgb_predict(cache->xgb, features, cache->model->nfeatures) :
                                        db4ai_native_predict(cache->native, features);
    pfree(features);
    return result;
}

Datum
pg_db4ai_predict_float8(PG_FUNCTION_ARGS)
{
    Db4aiCache *cache = get_predictor(fcinfo);
    PG_RETURN_FLOAT8(predict(cache, PG_GETARG_ARRAYTYPE_P(1)));
}

Datum
pg_db4ai_predict_bool(PG_FUNCTION_ARGS)
{
    Db4aiCache *cache = get_predictor(fcinfo);
    Db4aiAlgorithm algorithm = cache->model->algorithm;
    double result;
    if (algorithm != DB4AI_LOGISTIC && algorithm != DB4AI_SVM && algorithm != DB4AI_XGB_BINARY)
        ereport(ERROR, (errcode(ERRCODE_DATATYPE_MISMATCH), errmsg("predict_bool requires a binary classifier")));
    result = predict(cache, PG_GETARG_ARRAYTYPE_P(1));
    PG_RETURN_BOOL(result >= (algorithm == DB4AI_SVM ? 0.0 : 0.5));
}

Datum
pg_db4ai_predict_cluster(PG_FUNCTION_ARGS)
{
    Db4aiCache *cache = get_predictor(fcinfo);
    if (cache->model->algorithm != DB4AI_KMEANS)
        ereport(ERROR, (errcode(ERRCODE_DATATYPE_MISMATCH), errmsg("predict_cluster requires a kmeans model")));
    PG_RETURN_INT32((int32) predict(cache, PG_GETARG_ARRAYTYPE_P(1)));
}

Datum
pg_db4ai_transform(PG_FUNCTION_ARGS)
{
    Db4aiCache *cache = get_predictor(fcinfo);
    double *features;
    double *out;
    Datum *values;
    int n = cache->model->noutputs;
    int i;
    ArrayType *result;
    if (cache->model->algorithm != DB4AI_PCA)
        ereport(ERROR, (errcode(ERRCODE_DATATYPE_MISMATCH), errmsg("transform requires a PCA model")));
    features = db4ai_features(PG_GETARG_ARRAYTYPE_P(1), cache->model->nfeatures);
    out = palloc(n * sizeof(double));
    values = palloc(n * sizeof(Datum));
    db4ai_native_transform(cache->native, features, out);
    for (i = 0; i < n; i++)
        values[i] = Float8GetDatum(out[i]);
    result = construct_array(values, n, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, TYPALIGN_DOUBLE);
    pfree(features); pfree(out); pfree(values);
    PG_RETURN_ARRAYTYPE_P(result);
}

Datum
pg_db4ai_drop_model(PG_FUNCTION_ARGS)
{
    PG_RETURN_BOOL(db4ai_model_delete(model_name(fcinfo), GetUserId()));
}

Datum
pg_db4ai_export_model(PG_FUNCTION_ARGS)
{
    Db4aiModel *model = db4ai_model_load(model_name(fcinfo), GetUserId());
    PG_RETURN_BYTEA_P(model->data);
}
