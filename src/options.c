/* SPDX-License-Identifier: MulanPSL-2.0 */
#include "postgres.h"
#include <math.h>
#include "db4ai.h"
#include "common/shortest_dec.h"
#include "lib/stringinfo.h"
#include "utils/builtins.h"
#include "utils/fmgrprotos.h"

static const char *const names[] = {
    "linear_regression", "logistic_regression", "svm_classification", "kmeans", "pca",
    "xgboost_regression_squarederror", "xgboost_binary_logistic",
    "xgboost_regression_logistic", "xgboost_regression_gamma"
};

Db4aiAlgorithm
db4ai_algorithm(const char *name)
{
    int i;
    for (i = 0; i < lengthof(names); i++)
        if (strcmp(name, names[i]) == 0)
            return (Db4aiAlgorithm) i;
    ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                    errmsg("unsupported DB4AI algorithm: %s", name)));
    return DB4AI_LINEAR;
}

const char *
db4ai_algorithm_name(Db4aiAlgorithm algorithm)
{
    Assert(algorithm >= DB4AI_LINEAR && algorithm <= DB4AI_XGB_GAMMA);
    return names[algorithm];
}

bool
db4ai_supervised(Db4aiAlgorithm algorithm)
{
    return algorithm != DB4AI_KMEANS && algorithm != DB4AI_PCA;
}

bool
db4ai_is_xgboost(Db4aiAlgorithm algorithm)
{
    return algorithm >= DB4AI_XGB_SQUARED;
}

void
db4ai_check_finite(double x)
{
    if (!isfinite(x))
        ereport(ERROR, (errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
                        errmsg("DB4AI requires finite numbers"),
                        errhint("Scale features or lower the learning rate if training diverges.")));
}

JsonbValue *
db4ai_json_field(Jsonb *json, const char *key, bool required)
{
    JsonbValue k;
    JsonbValue *value;
    if (!JB_ROOT_IS_OBJECT(json))
        ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("expected a JSON object")));
    k.type = jbvString;
    k.val.string.val = (char *) key;
    k.val.string.len = strlen(key);
    value = findJsonbValueFromContainer(&json->root, JB_FOBJECT, &k);
    if (value == NULL && required)
        ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("missing JSON field: %s", key)));
    return value;
}

double
db4ai_json_number(Jsonb *json, const char *key, double def)
{
    JsonbValue *v = db4ai_json_field(json, key, false);
    double result;
    if (v == NULL)
        return def;
    if (v->type != jbvNumeric)
        ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("%s must be a number", key)));
    result = DatumGetFloat8(DirectFunctionCall1(numeric_float8, NumericGetDatum(v->val.numeric)));
    db4ai_check_finite(result);
    return result;
}

static double
bounded(Jsonb *json, const char *key, double def, double low, double high, bool integral)
{
    double v = db4ai_json_number(json, key, def);
    if (v < low || v > high || (integral && floor(v) != v))
        ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                        errmsg("invalid value for hyperparameter %s", key),
                        errdetail("Expected %s in [%g, %g].", integral ? "an integer" : "a number", low, high)));
    return v;
}

void
db4ai_options(Jsonb *json, Db4aiAlgorithm algorithm, Db4aiOptions *o)
{
    JsonbIterator *it;
    JsonbIteratorToken token;
    JsonbValue v;
    bool xgb = db4ai_is_xgboost(algorithm);
    bool gd = algorithm <= DB4AI_SVM;
    if (!JB_ROOT_IS_OBJECT(json))
        ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("hyperparameters must be a JSON object")));
    it = JsonbIteratorInit(&json->root);
    while ((token = JsonbIteratorNext(&it, &v, true)) != WJB_DONE)
    {
        if (token == WJB_KEY)
        {
            char *key = pnstrdup(v.val.string.val, v.val.string.len);
            bool allowed = strcmp(key, "max_iterations") == 0;
            allowed |= (gd || xgb) && (strcmp(key, "learning_rate") == 0 || strcmp(key, "l2") == 0);
            allowed |= gd && strcmp(key, "batch_size") == 0;
            allowed |= !xgb && strcmp(key, "tolerance") == 0;
            allowed |= (algorithm == DB4AI_KMEANS || xgb) && strcmp(key, "seed") == 0;
            allowed |= algorithm == DB4AI_KMEANS && strcmp(key, "k") == 0;
            allowed |= algorithm == DB4AI_PCA && strcmp(key, "n_components") == 0;
            allowed |= xgb && (strcmp(key, "max_depth") == 0 || strcmp(key, "min_child_weight") == 0 ||
                              strcmp(key, "subsample") == 0 || strcmp(key, "colsample_bytree") == 0);
            if (!allowed)
                ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                                errmsg("unknown or inapplicable hyperparameter: %s", key)));
            pfree(key);
        }
    }
    o->iterations = (int) bounded(json, "max_iterations", 100, 1, 100000, true);
    o->batch_size = (int) bounded(json, "batch_size", 128, 1, 1048576, true);
    o->clusters = (int) bounded(json, "k", 8, 1, 128, true);
    o->components = (int) bounded(json, "n_components", 2, 1, 256, true);
    o->seed = (int) bounded(json, "seed", 42, 0, PG_INT32_MAX, true);
    o->max_depth = (int) bounded(json, "max_depth", 6, 1, 16, true);
    o->learning_rate = bounded(json, "learning_rate", xgb ? 0.1 : 0.01, 1e-12, 1, false);
    o->l2 = bounded(json, "l2", algorithm == DB4AI_SVM ? 0.01 : 0, 0, 1e6, false);
    o->tolerance = bounded(json, "tolerance", 1e-8, 0, 1, false);
    o->min_child_weight = bounded(json, "min_child_weight", 1, 0, 1e6, false);
    o->subsample = bounded(json, "subsample", 1, 1e-6, 1, false);
    o->colsample_bytree = bounded(json, "colsample_bytree", 1, 1e-6, 1, false);
}

static void
option_number(StringInfo b, const char *name, double value)
{
    char number[DOUBLE_SHORTEST_DECIMAL_LEN];
    double_to_shortest_decimal_buf(value, number);
    appendStringInfo(b, ",\"%s\":%s", name, number);
}

Jsonb *
db4ai_options_json(const Db4aiOptions *o, Db4aiAlgorithm algorithm)
{
    StringInfoData b;
    bool xgb = db4ai_is_xgboost(algorithm);
    bool gd = algorithm <= DB4AI_SVM;
    initStringInfo(&b);
    appendStringInfo(&b, "{\"max_iterations\":%d", o->iterations);
    if (gd || xgb)
    {
        option_number(&b, "learning_rate", o->learning_rate);
        option_number(&b, "l2", o->l2);
    }
    if (gd)
        option_number(&b, "batch_size", o->batch_size);
    if (!xgb)
        option_number(&b, "tolerance", o->tolerance);
    if (algorithm == DB4AI_KMEANS || xgb)
        option_number(&b, "seed", o->seed);
    if (algorithm == DB4AI_KMEANS)
        option_number(&b, "k", o->clusters);
    if (algorithm == DB4AI_PCA)
        option_number(&b, "n_components", o->components);
    if (xgb)
    {
        option_number(&b, "max_depth", o->max_depth);
        option_number(&b, "min_child_weight", o->min_child_weight);
        option_number(&b, "subsample", o->subsample);
        option_number(&b, "colsample_bytree", o->colsample_bytree);
    }
    appendStringInfoChar(&b, '}');
    return DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(b.data)));
}

bytea *
db4ai_bytes(const void *data, Size len)
{
    bytea *out;
    if (len > (Size) db4ai_max_model_mb * 1024 * 1024 || len > MaxAllocSize - VARHDRSZ)
        ereport(ERROR, (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED), errmsg("model exceeds db4ai.max_model_mb")));
    out = (bytea *) palloc(VARHDRSZ + len);
    SET_VARSIZE(out, VARHDRSZ + len);
    memcpy(VARDATA(out), data, len);
    return out;
}
