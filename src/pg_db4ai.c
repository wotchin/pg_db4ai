/* SPDX-License-Identifier: MulanPSL-2.0 */
#include "postgres.h"
#include "db4ai.h"
#include "access/xact.h"
#include "catalog/pg_type_d.h"
#include "common/shortest_dec.h"
#include "lib/stringinfo.h"
#include "miscadmin.h"
#include "utils/builtins.h"
#include "utils/fmgrprotos.h"
#include "utils/guc.h"
#include "utils/timestamp.h"
#include "utils/datum.h"
#include "utils/lsyscache.h"

PG_MODULE_MAGIC;
PG_FUNCTION_INFO_V1(pg_db4ai_train);
PG_FUNCTION_INFO_V1(pg_db4ai_predict_float8);
PG_FUNCTION_INFO_V1(pg_db4ai_predict_bool);
PG_FUNCTION_INFO_V1(pg_db4ai_predict_cluster);
PG_FUNCTION_INFO_V1(pg_db4ai_transform);
PG_FUNCTION_INFO_V1(pg_db4ai_drop_model);
PG_FUNCTION_INFO_V1(pg_db4ai_export_model);
PG_FUNCTION_INFO_V1(pg_db4ai_predict_text);
PG_FUNCTION_INFO_V1(pg_db4ai_predict_label);
PG_FUNCTION_INFO_V1(pg_db4ai_predict_scores);
PG_FUNCTION_INFO_V1(pg_db4ai_class_labels);
PG_FUNCTION_INFO_V1(pg_db4ai_predict_joint);

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
    db4ai_shuffle_init();
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
    TimestampTz started = GetCurrentTimestamp();
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
        data = db4ai_data_open(query, algorithm, &options);
        model.algorithm = algorithm;
        model.nfeatures = data->nfeatures;
        if (db4ai_is_xgboost(algorithm))
            db4ai_xgb_train(data, algorithm, &options, format, &model);
        else if (algorithm == DB4AI_BAYES)
            db4ai_bayes_train(data, &options, &model);
        else
        {
            StringInfoData metrics;
            Db4aiNative *native = db4ai_native_train(data, algorithm, &options);
            char loss[DOUBLE_SHORTEST_DECIMAL_LEN];
            double_to_shortest_decimal_buf(native->loss,loss);
            model.data = db4ai_native_save(native);
            model.format = "json";
            model.noutputs = native->noutputs;
            initStringInfo(&metrics);
            appendStringInfo(&metrics, "{\"rows\":" INT64_FORMAT ",\"iterations\":%d,\"training_loss\":%s}",
                              data->nrows, native->iterations, loss);
            if (db4ai_classifier(algorithm))
            {
                double *row = palloc(sizeof(double) * data->ncolumns);
                int64 correct = 0, tp = 0, fp = 0, fn = 0;
                db4ai_data_rewind(data);
                while (db4ai_data_next(data, row))
                {
                    int label = db4ai_native_class(native, row + 1);
                    int actual = (int) row[0];
                    CHECK_FOR_INTERRUPTS();
                    if (label == actual) correct++;
                    if (native->nclasses == 2)
                    {
                        if (label == 1 && actual == 1) tp++;
                        if (label == 1 && actual == 0) fp++;
                        if (label == 0 && actual == 1) fn++;
                    }
                }
                metrics.len--; metrics.data[metrics.len] = '\0';
                appendStringInfo(&metrics, ",\"classes\":%d,\"accuracy\":%.17g", native->nclasses, (double) correct / data->nrows);
                if (native->nclasses == 2)
                {
                    double precision = tp + fp ? (double) tp / (tp + fp) : 0;
                    double recall = tp + fn ? (double) tp / (tp + fn) : 0;
                    appendStringInfo(&metrics, ",\"precision\":%.17g,\"recall\":%.17g,\"f1\":%.17g",
                                     precision, recall, precision+recall>0 ? 2*precision*recall/(precision+recall) : 0);
                }
                appendStringInfoChar(&metrics, '}');
                pfree(row);
            }
            model.metrics = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(metrics.data)));
        }
        {
            StringInfoData stats;
            long seconds; int micros;
            TimestampDifference(started, GetCurrentTimestamp(), &seconds, &micros);
            initStringInfo(&stats);
            appendStringInfo(&stats, "{\"discarded_rows\":" INT64_FORMAT ",\"input_passes\":" INT64_FORMAT
                             ",\"input_spilled\":%s,\"shuffle_cache_units\":%d,\"shuffle_full_cache\":%s,\"elapsed_seconds\":%.6f}",
                             data->discarded, data->passes, data->spilled?"true":"false", data->shuffle_capacity,
                             data->shuffle_full_cache?"true":"false", seconds + micros/1000000.0);
            model.metrics = DatumGetJsonbP(DirectFunctionCall2(jsonb_concat, JsonbPGetDatum(model.metrics),
                DirectFunctionCall1(jsonb_in, CStringGetDatum(stats.data))));
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
    Db4aiBayes *bayes;
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
        cache->bayes = NULL;
        if (db4ai_is_xgboost(cache->model->algorithm))
            cache->xgb = db4ai_xgb_load(cache->model);
        else if (cache->model->algorithm == DB4AI_BAYES)
            cache->bayes = db4ai_bayes_load(cache->model);
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
    double *features;
    double result;
    if (cache->bayes != NULL)
        ereport(ERROR, (errcode(ERRCODE_DATATYPE_MISMATCH), errmsg("use predict_joint for Bayes models")));
    features = db4ai_features(array, cache->model->nfeatures);
    result = cache->xgb != NULL ? db4ai_xgb_predict(cache->xgb, features, cache->model->nfeatures) :
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
    if (cache->native != NULL && cache->native->labels != NULL && cache->native->labels->type == BOOLOID)
    {
        double *features = db4ai_features(PG_GETARG_ARRAYTYPE_P(1), cache->model->nfeatures);
        int index = db4ai_native_class(cache->native, features);
        pfree(features);
        PG_RETURN_BOOL(DatumGetBool(cache->native->labels->values[index]));
    }
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

/* Polymorphic witness fixes the output SQL type at parse time; the model's
 * dictionary must match. This is not an unchecked Datum reinterpretation. */
static Db4aiCache *
classifier_predictor(FunctionCallInfo fcinfo)
{
    Db4aiCache *cache;
    if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
        ereport(ERROR, (errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED), errmsg("model and features must not be NULL")));
    cache=get_predictor(fcinfo);
    if (!db4ai_classifier(cache->model->algorithm) || cache->native->labels == NULL)
        ereport(ERROR, (errcode(ERRCODE_DATATYPE_MISMATCH), errmsg("model has no typed classification dictionary")));
    return cache;
}

Datum
pg_db4ai_predict_label(PG_FUNCTION_ARGS)
{
    Db4aiCache *cache=classifier_predictor(fcinfo);
    Db4aiDictionary *d=cache->native->labels;
    double *features=db4ai_features(PG_GETARG_ARRAYTYPE_P(1), cache->model->nfeatures);
    Oid rettype=get_fn_expr_rettype(fcinfo->flinfo);
    int index;
    if (rettype != d->type)
        ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("classification label type is %s, not %s",format_type_be(d->type),format_type_be(rettype))));
    index=db4ai_native_class(cache->native,features); pfree(features);
    PG_RETURN_DATUM(datumCopy(d->values[index],d->typbyval,d->typlen));
}

Datum
pg_db4ai_predict_text(PG_FUNCTION_ARGS)
{
    Db4aiCache *cache=classifier_predictor(fcinfo);
    double *features=db4ai_features(PG_GETARG_ARRAYTYPE_P(1),cache->model->nfeatures);
    int index=db4ai_native_class(cache->native,features);
    char *label=db4ai_dictionary_text(cache->native->labels,index);
    pfree(features);
    PG_RETURN_TEXT_P(cstring_to_text(label));
}

Datum
pg_db4ai_predict_scores(PG_FUNCTION_ARGS)
{
    Db4aiCache *cache=classifier_predictor(fcinfo);
    int n=cache->native->nclasses,i;
    double *features=db4ai_features(PG_GETARG_ARRAYTYPE_P(1),cache->model->nfeatures);
    double *scores=palloc(sizeof(double)*n);
    Datum *values=palloc(sizeof(Datum)*n);
    ArrayType *result;
    db4ai_native_scores(cache->native,features,scores);
    for(i=0;i<n;i++) { db4ai_check_finite(scores[i]); values[i]=Float8GetDatum(scores[i]); }
    result=construct_array(values,n,FLOAT8OID,sizeof(float8),FLOAT8PASSBYVAL,TYPALIGN_DOUBLE);
    pfree(features); pfree(scores); pfree(values);
    PG_RETURN_ARRAYTYPE_P(result);
}

Datum
pg_db4ai_class_labels(PG_FUNCTION_ARGS)
{
    Db4aiCache *cache=get_predictor(fcinfo);
    Db4aiDictionary *d;
    Datum *values; ArrayType *result; int i;
    if(!cache->native || !cache->native->labels)
        ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("model has no classification dictionary")));
    d=cache->native->labels; values=palloc(sizeof(Datum)*d->count);
    for(i=0;i<d->count;i++) values[i]=CStringGetTextDatum(db4ai_dictionary_text(d,i));
    result=construct_array(values,d->count,TEXTOID,-1,false,TYPALIGN_INT);
    PG_RETURN_ARRAYTYPE_P(result);
}

Datum
pg_db4ai_predict_joint(PG_FUNCTION_ARGS)
{
    Db4aiCache *cache;
    Datum values[64]; bool nulls[64]; Oid types[64]; int i,n=PG_NARGS()-1;
    if(PG_ARGISNULL(0)) PG_RETURN_NULL();
    if(n<1 || n>64) ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("Bayes expects 1..64 variables")));
    cache=get_predictor(fcinfo);
    if(!cache->bayes) ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("predict_joint requires a bayes_net_internal model")));
    for(i=0;i<n;i++)
    { values[i]=PG_GETARG_DATUM(i+1); nulls[i]=PG_ARGISNULL(i+1); types[i]=get_fn_expr_argtype(fcinfo->flinfo,i+1); }
    PG_RETURN_FLOAT8(db4ai_bayes_predict(cache->bayes,values,nulls,types,n));
}
