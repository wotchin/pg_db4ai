/* SPDX-License-Identifier: MulanPSL-2.0
 * Only the upstream C API crosses this boundary. No PostgreSQL callbacks are
 * invoked from inside XGBoost's C++ frames. All PG errors happen after return.
 */
#include "postgres.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include "db4ai.h"
#include "common/shortest_dec.h"
#include "lib/stringinfo.h"
#include "miscadmin.h"
#include "utils/builtins.h"
#include "utils/fmgrprotos.h"
#include "utils/json.h"
#include "utils/memutils.h"
#include "utils/timestamp.h"

#ifdef DB4AI_WITH_XGBOOST
#include "xgboost/c_api.h"

typedef struct Db4aiXgb
{
    MemoryContextCallback cleanup;
    BoosterHandle booster;
    DMatrixHandle matrix;
    DMatrixHandle proxy;
    bool inplace;
    int nfeatures;
    float *prediction_row;
    char *array_interface;
} Db4aiXgb;

static void
xgb_check(int result)
{
    if (result != 0)
        ereport(ERROR, (errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION), errmsg("XGBoost: %s", XGBGetLastError())));
}

static void
xgb_release(void *arg)
{
    Db4aiXgb *resource = (Db4aiXgb *) arg;
    /* Context cleanup must never throw another PG error. */
    if (resource->matrix != NULL)
    {
        (void) XGDMatrixFree(resource->matrix);
        resource->matrix = NULL;
    }
    if (resource->proxy != NULL)
    {
        (void) XGDMatrixFree(resource->proxy);
        resource->proxy = NULL;
    }
    if (resource->booster != NULL)
    {
        (void) XGBoosterFree(resource->booster);
        resource->booster = NULL;
    }
}

static Db4aiXgb *
xgb_resource(void)
{
    Db4aiXgb *resource = palloc0(sizeof(*resource));
    resource->cleanup.func = xgb_release;
    resource->cleanup.arg = resource;
    MemoryContextRegisterResetCallback(CurrentMemoryContext, &resource->cleanup);
    return resource;
}

static void
set_number(BoosterHandle booster, const char *key, double number)
{
    char value[DOUBLE_SHORTEST_DECIMAL_LEN];
    double_to_shortest_decimal_buf(number, value);
    xgb_check(XGBoosterSetParam(booster, key, value));
}

static void
set_integer(BoosterHandle booster, const char *key, int number)
{
    char value[32];
    snprintf(value, sizeof(value), "%d", number);
    xgb_check(XGBoosterSetParam(booster, key, value));
}

static const char *
objective(Db4aiAlgorithm algorithm)
{
    switch (algorithm)
    {
        case DB4AI_XGB_SQUARED: return "reg:squarederror";
        case DB4AI_XGB_BINARY: return "binary:logistic";
        case DB4AI_XGB_LOGISTIC: return "reg:logistic";
        case DB4AI_XGB_GAMMA: return "reg:gamma";
        default: elog(ERROR, "invalid XGBoost algorithm");
    }
    return NULL;
}

void
db4ai_xgb_train(Db4aiData *data, Db4aiAlgorithm algorithm, const Db4aiOptions *o,
                const char *format, Db4aiModel *out)
{
    Db4aiXgb *r = xgb_resource();
    Size cells = (Size) data->nrows * data->nfeatures;
    Size bytes;
    float *features;
    float *labels;
    double *row;
    int64 i = 0;
    int64 positives = 0;
    int j, epoch, major, minor, patch;
    bst_ulong model_size;
    const char *model_data;
    const char *evaluation;
    const char *eval_names[] = {"train"};
    StringInfoData metrics;
    TimestampTz started=GetCurrentTimestamp();
    int completed=0;

    if (cells > MaxAllocSize / sizeof(float) || (Size) data->nrows > MaxAllocSize / sizeof(float))
        ereport(ERROR, (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED), errmsg("XGBoost dense input is too large")));
    bytes = (cells + (Size) data->nrows) * sizeof(float);
    if (bytes > (Size) db4ai_xgboost_max_input_mb * 1024 * 1024)
        ereport(ERROR, (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
                        errmsg("XGBoost input exceeds db4ai.xgboost_max_input_mb"),
                        errdetail("This limit covers the dense input copy, not all internal XGBoost allocations.")));
    features = palloc(cells * sizeof(float));
    labels = palloc(data->nrows * sizeof(float));
    row = palloc(data->ncolumns * sizeof(double));
    db4ai_data_rewind(data);
    while (db4ai_data_next(data, row))
    {
        CHECK_FOR_INTERRUPTS();
        for (j = 0; j < data->ncolumns; j++)
            if (fabs(row[j]) > FLT_MAX)
                ereport(ERROR, (errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE), errmsg("XGBoost input exceeds float32 range")));
        if ((algorithm == DB4AI_XGB_BINARY && row[0] != 0 && row[0] != 1) ||
            (algorithm == DB4AI_XGB_LOGISTIC && (row[0] < 0 || row[0] > 1)) ||
            (algorithm == DB4AI_XGB_GAMMA && row[0] <= 0))
            ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("label is outside the objective's supported range")));
        if(row[0]>0) positives++;
        labels[i] = (float) row[0];
        for (j = 0; j < data->nfeatures; j++)
            features[i * data->nfeatures + j] = (float) row[j + 1];
        i++;
    }
    if((strcmp(o->eval_metric,"auc")==0 || strcmp(o->eval_metric,"aucpr")==0) &&
       (positives==0 || positives==data->nrows))
        ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("auc/aucpr require both positive and negative labels")));
    xgb_check(XGDMatrixCreateFromMat(features, data->nrows, data->nfeatures, NAN, &r->matrix));
    xgb_check(XGDMatrixSetFloatInfo(r->matrix, "label", labels, data->nrows));
    pfree(features); pfree(labels); pfree(row);
    xgb_check(XGBoosterCreate(&r->matrix, 1, &r->booster));
    xgb_check(XGBoosterSetParam(r->booster, "objective", objective(algorithm)));
    xgb_check(XGBoosterSetParam(r->booster, "tree_method", o->tree_method));
    xgb_check(XGBoosterSetParam(r->booster, "booster", o->booster));
    if(*o->eval_metric) xgb_check(XGBoosterSetParam(r->booster, "eval_metric", o->eval_metric));
    xgb_check(XGBoosterSetParam(r->booster, "device", "cpu"));
    set_integer(r->booster, "nthread", o->nthread);
    xgb_check(XGBoosterSetParam(r->booster, "verbosity", o->verbose?"1":"0"));
    set_integer(r->booster, "max_depth", o->max_depth);
    set_integer(r->booster, "seed", o->seed);
    set_number(r->booster, "gamma", o->gamma);
    set_number(r->booster, "eta", o->learning_rate);
    set_number(r->booster, "lambda", o->l2);
    set_number(r->booster, "min_child_weight", o->min_child_weight);
    set_number(r->booster, "subsample", o->subsample);
    set_number(r->booster, "colsample_bytree", o->colsample_bytree);
    /* Every tree is trained on the complete DMatrix, not on independent chunks. */
    for (epoch = 0; epoch < o->iterations; epoch++)
    {
        CHECK_FOR_INTERRUPTS();
        xgb_check(XGBoosterUpdateOneIter(r->booster, epoch, r->matrix));
        completed=epoch+1;
        if(o->max_seconds && TimestampDifferenceExceeds(started,GetCurrentTimestamp(),o->max_seconds*1000)) break;
    }
    xgb_check(XGBoosterSaveModelToBuffer(r->booster,
                strcmp(format, "json") == 0 ? "{\"format\":\"json\"}" : "{\"format\":\"ubj\"}",
                &model_size, &model_data));
    out->data = db4ai_bytes(model_data, model_size);
    out->format = pstrdup(format);
    out->noutputs = 1;
    xgb_check(XGBoosterEvalOneIter(r->booster, completed - 1, &r->matrix, eval_names, 1, &evaluation));
    XGBoostVersion(&major, &minor, &patch);
    initStringInfo(&metrics);
    appendStringInfo(&metrics, "{\"rows\":" INT64_FORMAT ",\"iterations\":%d,\"library_version\":\"%d.%d.%d\","
                              "\"training_evaluation\":", data->nrows, completed, major, minor, patch);
    escape_json(&metrics, evaluation);
    appendStringInfoChar(&metrics, '}');
    out->metrics = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(metrics.data)));
    xgb_release(r);
}

/* Inspect the booster once. Non-tree boosters keep the existing DMatrix path. */
static bool
xgb_supports_inplace(BoosterHandle booster)
{
    bst_ulong length;
    const char *buffer;
    char *text;
    char *name;
    Jsonb *config;
    JsonbValue *value;
    bool result;
    xgb_check(XGBoosterSaveJsonConfig(booster, &length, &buffer));
    if (length >= MaxAllocSize)
        ereport(ERROR, (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED), errmsg("XGBoost configuration is too large")));
    text = pnstrdup(buffer, (Size) length);
    config = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(text)));
    value = db4ai_json_field(config, "learner", true);
    if (value->type != jbvBinary || !JsonContainerIsObject(value->val.binary.data))
        ereport(ERROR, (errcode(ERRCODE_DATA_CORRUPTED), errmsg("invalid XGBoost learner configuration")));
    config = JsonbValueToJsonb(value);
    value = db4ai_json_field(config, "gradient_booster", true);
    if (value->type != jbvBinary || !JsonContainerIsObject(value->val.binary.data))
        ereport(ERROR, (errcode(ERRCODE_DATA_CORRUPTED), errmsg("invalid XGBoost booster configuration")));
    config = JsonbValueToJsonb(value);
    name = db4ai_json_string(config, "name", NULL);
    if (name == NULL)
        ereport(ERROR, (errcode(ERRCODE_DATA_CORRUPTED), errmsg("missing XGBoost booster name")));
    /* Deliberately retain the old path for dart and gblinear. */
    result = strcmp(name, "gbtree") == 0;
    pfree(name);
    pfree(text);
    return result;
}

void *
db4ai_xgb_load(const Db4aiModel *model)
{
    Db4aiXgb *r = xgb_resource();
    bst_ulong nfeatures;
    if (strcmp(model->format, "json") != 0 && strcmp(model->format, "ubj") != 0)
        ereport(ERROR, (errcode(ERRCODE_DATA_CORRUPTED), errmsg("unsupported XGBoost model format")));
    xgb_check(XGBoosterCreate(NULL, 0, &r->booster));
    xgb_check(XGBoosterLoadModelFromBuffer(r->booster, VARDATA_ANY(model->data), VARSIZE_ANY_EXHDR(model->data)));
    xgb_check(XGBoosterSetParam(r->booster, "nthread", "1"));
    xgb_check(XGBoosterSetParam(r->booster, "device", "cpu"));
    xgb_check(XGBoosterGetNumFeature(r->booster, &nfeatures));
    if (nfeatures != model->nfeatures)
        ereport(ERROR, (errcode(ERRCODE_DATA_CORRUPTED), errmsg("XGBoost feature schema does not match model metadata")));
    r->nfeatures = model->nfeatures;
    r->prediction_row = palloc(r->nfeatures * sizeof(float));
    r->inplace = xgb_supports_inplace(r->booster);
    if (r->inplace)
    {
#ifdef WORDS_BIGENDIAN
        const char *typestr = ">f4";
#else
        const char *typestr = "<f4";
#endif
        /* Stable PG-owned buffer and reusable proxy: no per-row DMatrix build.
         * PredictFromDense is synchronous; the array remains valid until this
         * predictor's MemoryContext is reset. Never persist its address. */
        r->array_interface = psprintf(
            "{\"data\":[" UINT64_FORMAT ",false],\"shape\":[1,%d],\"typestr\":\"%s\",\"version\":3}",
            (uint64) (uintptr_t) r->prediction_row, r->nfeatures, typestr);
        xgb_check(XGProxyDMatrixCreate(&r->proxy));
    }
    return r;
}

double
db4ai_xgb_predict(void *predictor, const double *features, int nfeatures)
{
    Db4aiXgb *r = (Db4aiXgb *) predictor;
    float *row = r->prediction_row;
    const bst_ulong *shape;
    bst_ulong ndim;
    const float *result;
    volatile double out = 0;
    int i;
    if (nfeatures != r->nfeatures)
        ereport(ERROR, (errcode(ERRCODE_DATATYPE_MISMATCH), errmsg("XGBoost feature count changed")));
    for (i = 0; i < nfeatures; i++)
    {
        if (fabs(features[i]) > FLT_MAX)
            ereport(ERROR, (errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE), errmsg("XGBoost feature exceeds float32 range")));
        row[i] = (float) features[i];
    }
    CHECK_FOR_INTERRUPTS();
    PG_TRY();
    {
        if (r->inplace)
            xgb_check(XGBoosterPredictFromDense(r->booster, r->array_interface,
                      "{\"type\":0,\"iteration_begin\":0,\"iteration_end\":0,\"strict_shape\":true,\"missing\":NaN}",
                      r->proxy, &shape, &ndim, &result));
        else
        {
            xgb_check(XGDMatrixCreateFromMat(row, 1, nfeatures, NAN, &r->matrix));
            xgb_check(XGBoosterPredictFromDMatrix(r->booster, r->matrix,
                      "{\"type\":0,\"training\":false,\"iteration_begin\":0,\"iteration_end\":0,\"strict_shape\":true}",
                      &shape, &ndim, &result));
        }
        if (ndim != 2 || shape[0] != 1 || shape[1] != 1)
            ereport(ERROR, (errcode(ERRCODE_DATA_CORRUPTED), errmsg("unexpected XGBoost prediction shape")));
        out = result[0];
        if (r->matrix != NULL)
        {
            xgb_check(XGDMatrixFree(r->matrix));
            r->matrix = NULL;
        }
    }
    PG_CATCH();
    {
        if (r->matrix != NULL)
        {
            (void) XGDMatrixFree(r->matrix);
            r->matrix = NULL;
        }
        PG_RE_THROW();
    }
    PG_END_TRY();
    db4ai_check_finite(out);
    return out;
}
#else
static void
unavailable(void)
{
    ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                    errmsg("pg_db4ai was built without XGBoost"),
                    errhint("Build with WITH_XGBOOST=1 using the vendored source tree.")));
}
void db4ai_xgb_train(Db4aiData *d, Db4aiAlgorithm a, const Db4aiOptions *o, const char *f, Db4aiModel *m) { unavailable(); }
void *db4ai_xgb_load(const Db4aiModel *m) { unavailable(); return NULL; }
double db4ai_xgb_predict(void *p, const double *f, int n) { unavailable(); return 0; }
#endif
