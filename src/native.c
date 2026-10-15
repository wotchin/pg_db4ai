/*
 * SPDX-License-Identifier: MulanPSL-2.0
 *
 * PostgreSQL C implementation of the DB4AI dense numeric training path.
 * The linear/logistic gradient equations follow openGauss DB4AI
 * (Copyright (c) 2020 Huawei Technologies Co.,Ltd.). The PG data source,
 * bounded mini-batch driver and portable model encoding are new.
 * See docs/PORTING.md for deliberate differences from the openGauss solvers.
 */
#include "postgres.h"
#include <float.h>
#include <math.h>
#include "db4ai.h"
#include "common/pg_prng.h"
#include "common/shortest_dec.h"
#include "lib/stringinfo.h"
#include "miscadmin.h"
#include "utils/builtins.h"
#include "utils/fmgrprotos.h"
#include "utils/json.h"

static double
sigmoid(double x)
{
    if (x >= 0)
        return 1.0 / (1.0 + exp(-x));
    x = exp(x);
    return x / (1.0 + x);
}

static double
score(const Db4aiNative *m, const double *x)
{
    double z = m->weights[0];
    int j;
    for (j = 0; j < m->nfeatures; j++)
        z += m->weights[j + 1] * x[j];
    db4ai_check_finite(z);
    return z;
}

static double
distance2(const double *a, const double *b, int n)
{
    double s = 0;
    int j;
    for (j = 0; j < n; j++)
    {
        double d = a[j] - b[j];
        s += d * d;
    }
    db4ai_check_finite(s);
    return s;
}

static int
closest(const Db4aiNative *m, const double *x, double *distance)
{
    double best = DBL_MAX;
    int result = 0;
    int c;
    for (c = 0; c < m->noutputs; c++)
    {
        double d = distance2(x, m->weights + c * m->nfeatures, m->nfeatures);
        if (d < best)
        {
            best = d;
            result = c;
        }
    }
    if (distance != NULL)
        *distance = best;
    return result;
}

static void
apply_gradient(Db4aiNative *m, double *gradient, int batch, const Db4aiOptions *o)
{
    int j;
    for (j = 0; j <= m->nfeatures; j++)
    {
        double regularization = j == 0 ? 0 : o->l2 * m->weights[j];
        m->weights[j] -= o->learning_rate * (gradient[j] / batch + regularization);
        db4ai_check_finite(m->weights[j]);
        gradient[j] = 0;
    }
}

static void
train_gd(Db4aiNative *m, Db4aiData *data, const Db4aiOptions *o)
{
    double *row = palloc(sizeof(double) * data->ncolumns);
    double *gradient = palloc0(sizeof(double) * (m->nfeatures + 1));
    double previous = DBL_MAX;
    int epoch;
    m->noutputs = 1;
    m->weights = palloc0(sizeof(double) * (m->nfeatures + 1));
    for (epoch = 0; epoch < o->iterations; epoch++)
    {
        int batch = 0;
        double loss = 0;
        db4ai_data_rewind(data);
        while (db4ai_data_next(data, row))
        {
            double y = row[0];
            double z = score(m, row + 1);
            double error;
            int j;
            CHECK_FOR_INTERRUPTS();
            if (m->algorithm == DB4AI_LOGISTIC || m->algorithm == DB4AI_SVM)
                if (y != 0.0 && y != 1.0)
                    ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                                    errmsg("binary classification labels must be 0 or 1")));
            if (m->algorithm == DB4AI_LOGISTIC)
                error = sigmoid(z) - y;
            else if (m->algorithm == DB4AI_SVM)
            {
                double target = y * 2.0 - 1.0;
                error = target * z < 1.0 ? -target : 0.0;
            }
            else
                error = z - y;
            gradient[0] += error;
            for (j = 0; j < m->nfeatures; j++)
                gradient[j + 1] += error * row[j + 1];
            if (++batch == o->batch_size)
            {
                apply_gradient(m, gradient, batch, o);
                batch = 0;
            }
        }
        if (batch != 0)
            apply_gradient(m, gradient, batch, o);
        /* Evaluate the final weights of this epoch, not a mixture of iterates. */
        db4ai_data_rewind(data);
        while (db4ai_data_next(data, row))
        {
            double z = score(m, row + 1);
            double y = row[0];
            CHECK_FOR_INTERRUPTS();
            if (m->algorithm == DB4AI_LOGISTIC)
                loss += Max(z, 0.0) - y * z + log1p(exp(-fabs(z)));
            else if (m->algorithm == DB4AI_SVM)
                loss += Max(0.0, 1.0 - (y * 2.0 - 1.0) * z);
            else
                loss += (z - y) * (z - y);
        }
        m->loss = loss / data->nrows;
        db4ai_check_finite(m->loss);
        m->iterations = epoch + 1;
        if (epoch > 0 && fabs(previous - m->loss) <= o->tolerance * Max(1.0, fabs(previous)))
            break;
        previous = m->loss;
    }
    pfree(gradient);
    pfree(row);
}

static void
train_kmeans(Db4aiNative *m, Db4aiData *data, const Db4aiOptions *o)
{
    pg_prng_state rng;
    double *row = palloc(sizeof(double) * m->nfeatures);
    double *sums;
    int64 *counts;
    int64 seen = 0;
    int epoch;
    int c;
    int j;
    if (o->clusters > data->nrows)
        ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("k exceeds the number of input rows")));
    m->noutputs = o->clusters;
    m->weights = palloc0(sizeof(double) * m->nfeatures * m->noutputs);
    sums = palloc0(sizeof(double) * m->nfeatures * m->noutputs);
    counts = palloc0(sizeof(int64) * m->noutputs);
    pg_prng_seed(&rng, (uint64) o->seed);
    /* Uniform reservoir seeds. No full input copy is needed, including on spill. */
    db4ai_data_rewind(data);
    while (db4ai_data_next(data, row))
    {
        uint64 slot = seen < m->noutputs ? (uint64) seen : pg_prng_uint64_range(&rng, 0, (uint64) seen);
        CHECK_FOR_INTERRUPTS();
        if (slot < m->noutputs)
            memcpy(m->weights + slot * m->nfeatures, row, sizeof(double) * m->nfeatures);
        seen++;
    }
    for (epoch = 0; epoch < o->iterations; epoch++)
    {
        double movement = 0;
        memset(sums, 0, sizeof(double) * m->nfeatures * m->noutputs);
        memset(counts, 0, sizeof(int64) * m->noutputs);
        db4ai_data_rewind(data);
        while (db4ai_data_next(data, row))
        {
            CHECK_FOR_INTERRUPTS();
            c = closest(m, row, NULL);
            counts[c]++;
            for (j = 0; j < m->nfeatures; j++)
                sums[c * m->nfeatures + j] += row[j];
        }
        for (c = 0; c < m->noutputs; c++)
            if (counts[c] != 0)
                for (j = 0; j < m->nfeatures; j++)
                {
                    int pos = c * m->nfeatures + j;
                    double value = sums[pos] / counts[c];
                    db4ai_check_finite(value);
                    movement = Max(movement, fabs(value - m->weights[pos]));
                    m->weights[pos] = value;
                }
        /* Empty clusters retain their preceding centroid. */
        m->iterations = epoch + 1;
        if (movement <= o->tolerance)
            break;
    }
    m->loss = 0;
    db4ai_data_rewind(data);
    while (db4ai_data_next(data, row))
    {
        double d;
        CHECK_FOR_INTERRUPTS();
        (void) closest(m, row, &d);
        m->loss += d;
    }
    m->loss /= data->nrows;
    db4ai_check_finite(m->loss);
    pfree(row);
    pfree(sums);
    pfree(counts);
}

static void
train_pca(Db4aiNative *m, Db4aiData *data, const Db4aiOptions *o)
{
    int n = m->nfeatures;
    int i, j, p, q, sweep;
    int64 count = 0;
    double *a;
    double *v;
    double *row;
    double *delta;
    bool converged = false;
    if (n > 256 || o->components > n || data->nrows < 2)
        ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                        errmsg("PCA requires 2+ rows, at most 256 features, and n_components <= n_features")));
    m->noutputs = o->components;
    m->mean = palloc0(n * sizeof(double));
    m->weights = palloc(n * m->noutputs * sizeof(double));
    a = palloc0(n * n * sizeof(double));
    v = palloc0(n * n * sizeof(double));
    row = palloc(n * sizeof(double));
    delta = palloc(n * sizeof(double));
    db4ai_data_rewind(data);
    /* Welford covariance. The input is centered, not standardized. */
    while (db4ai_data_next(data, row))
    {
        CHECK_FOR_INTERRUPTS();
        count++;
        for (i = 0; i < n; i++)
        {
            delta[i] = row[i] - m->mean[i];
            m->mean[i] += delta[i] / count;
        }
        for (i = 0; i < n; i++)
            for (j = i; j < n; j++)
                a[i * n + j] += delta[i] * (row[j] - m->mean[j]);
    }
    for (i = 0; i < n; i++)
    {
        v[i * n + i] = 1;
        for (j = i; j < n; j++)
        {
            a[i * n + j] /= count - 1;
            db4ai_check_finite(a[i * n + j]);
            a[j * n + i] = a[i * n + j];
        }
    }
    /* Symmetric Jacobi eigensolver, intentionally free of BLAS/LAPACK dependencies. */
    for (sweep = 0; sweep < o->iterations; sweep++)
    {
        double off = 0;
        double scale = 0;
        CHECK_FOR_INTERRUPTS();
        for (i = 0; i < n; i++)
        {
            scale = Max(scale, fabs(a[i * n + i]));
            for (j = i + 1; j < n; j++)
                off = Max(off, fabs(a[i * n + j]));
        }
        if (off <= Max(o->tolerance, DBL_EPSILON) * Max(scale, DBL_MIN))
        {
            converged = true;
            break;
        }
        for (p = 0; p < n; p++)
            for (q = p + 1; q < n; q++)
            {
                double apq = a[p * n + q];
                double angle, cs, sn, app, aqq;
                if (apq == 0)
                    continue;
                angle = 0.5 * atan2(2 * apq, a[q * n + q] - a[p * n + p]);
                cs = cos(angle);
                sn = sin(angle);
                app = a[p * n + p];
                aqq = a[q * n + q];
                for (i = 0; i < n; i++)
                {
                    double vip = v[i * n + p];
                    double viq = v[i * n + q];
                    v[i * n + p] = cs * vip - sn * viq;
                    v[i * n + q] = sn * vip + cs * viq;
                    if (i != p && i != q)
                    {
                        double aip = a[i * n + p];
                        double aiq = a[i * n + q];
                        a[i * n + p] = a[p * n + i] = cs * aip - sn * aiq;
                        a[i * n + q] = a[q * n + i] = sn * aip + cs * aiq;
                    }
                }
                a[p * n + p] = cs * cs * app - 2 * sn * cs * apq + sn * sn * aqq;
                a[q * n + q] = sn * sn * app + 2 * sn * cs * apq + cs * cs * aqq;
                a[p * n + q] = a[q * n + p] = 0;
            }
    }
    if (!converged)
        ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                        errmsg("PCA eigensolver did not converge"), errhint("Increase max_iterations.")));
    m->iterations = sweep;
    for (p = 0; p < m->noutputs; p++)
    {
        int best = 0;
        double sign = 1;
        int anchor = 0;
        for (i = 1; i < n; i++)
            if (a[i * n + i] > a[best * n + best])
                best = i;
        for (i = 1; i < n; i++)
            if (fabs(v[i * n + best]) > fabs(v[anchor * n + best]))
                anchor = i;
        if (v[anchor * n + best] < 0)
            sign = -1;
        for (i = 0; i < n; i++)
            m->weights[p * n + i] = sign * v[i * n + best];
        a[best * n + best] = -DBL_MAX;
    }
    m->loss = 0; /* Residual sample variance after the selected components. */
    for (i = 0; i < n; i++)
        if (a[i * n + i] != -DBL_MAX)
            m->loss += Max(0.0, a[i * n + i]);
    pfree(a); pfree(v); pfree(row); pfree(delta);
}

Db4aiNative *
db4ai_native_train(Db4aiData *data, Db4aiAlgorithm algorithm, const Db4aiOptions *options)
{
    Db4aiNative *model = palloc0(sizeof(*model));
    model->algorithm = algorithm;
    model->nfeatures = data->nfeatures;
    if (algorithm == DB4AI_KMEANS)
        train_kmeans(model, data, options);
    else if (algorithm == DB4AI_PCA)
        train_pca(model, data, options);
    else
        train_gd(model, data, options);
    return model;
}

static void
write_array(StringInfo b, const double *values, int count)
{
    int i;
    appendStringInfoChar(b, '[');
    for (i = 0; i < count; i++)
    {
        char value[DOUBLE_SHORTEST_DECIMAL_LEN];
        db4ai_check_finite(values[i]);
        double_to_shortest_decimal_buf(values[i], value);
        if (i != 0)
            appendStringInfoChar(b, ',');
        appendStringInfoString(b, value);
    }
    appendStringInfoChar(b, ']');
}

bytea *
db4ai_native_save(const Db4aiNative *m)
{
    StringInfoData b;
    int count = m->algorithm <= DB4AI_SVM ? m->nfeatures + 1 : m->nfeatures * m->noutputs;
    initStringInfo(&b);
    appendStringInfo(&b, "{\"format_version\":%d,\"algorithm\":", DB4AI_FORMAT_VERSION);
    escape_json(&b, db4ai_algorithm_name(m->algorithm));
    appendStringInfo(&b, ",\"n_features\":%d,\"n_outputs\":%d,\"weights\":", m->nfeatures, m->noutputs);
    write_array(&b, m->weights, count);
    if (m->mean != NULL)
    {
        appendStringInfoString(&b, ",\"mean\":");
        write_array(&b, m->mean, m->nfeatures);
    }
    appendStringInfoChar(&b, '}');
    return db4ai_bytes(b.data, b.len);
}

static double *
read_array(Jsonb *json, const char *name, int expected)
{
    JsonbValue *v = db4ai_json_field(json, name, true);
    JsonbIterator *it;
    JsonbIteratorToken token;
    JsonbValue elem;
    double *out = palloc(sizeof(double) * expected);
    int i = 0;
    if (v->type != jbvBinary || !JsonContainerIsArray(v->val.binary.data) ||
        JsonContainerSize(v->val.binary.data) != expected)
        ereport(ERROR, (errcode(ERRCODE_DATA_CORRUPTED), errmsg("invalid model array: %s", name)));
    it = JsonbIteratorInit(v->val.binary.data);
    while ((token = JsonbIteratorNext(&it, &elem, true)) != WJB_DONE)
        if (token == WJB_ELEM)
        {
            if (elem.type != jbvNumeric || i >= expected)
                ereport(ERROR, (errcode(ERRCODE_DATA_CORRUPTED), errmsg("invalid numeric model array")));
            out[i] = DatumGetFloat8(DirectFunctionCall1(numeric_float8, NumericGetDatum(elem.val.numeric)));
            db4ai_check_finite(out[i++]);
        }
    if (i != expected)
        ereport(ERROR, (errcode(ERRCODE_DATA_CORRUPTED), errmsg("truncated model array")));
    return out;
}

Db4aiNative *
db4ai_native_load(const Db4aiModel *model)
{
    char *text = pnstrdup(VARDATA_ANY(model->data), VARSIZE_ANY_EXHDR(model->data));
    Jsonb *json = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(text)));
    JsonbValue *algorithm = db4ai_json_field(json, "algorithm", true);
    Db4aiNative *m = palloc0(sizeof(*m));
    int count;
    if (db4ai_json_number(json, "format_version", -1) != DB4AI_FORMAT_VERSION ||
        db4ai_json_number(json, "n_features", -1) != model->nfeatures ||
        db4ai_json_number(json, "n_outputs", -1) != model->noutputs ||
        algorithm->type != jbvString ||
        db4ai_algorithm(pnstrdup(algorithm->val.string.val, algorithm->val.string.len)) != model->algorithm)
        ereport(ERROR, (errcode(ERRCODE_DATA_CORRUPTED), errmsg("unsupported or inconsistent native model header")));
    m->algorithm = model->algorithm;
    m->nfeatures = model->nfeatures;
    m->noutputs = model->noutputs;
    if (m->nfeatures < 1 || m->nfeatures > DB4AI_MAX_FEATURES || m->noutputs < 1 || m->noutputs > 256 ||
        (m->algorithm <= DB4AI_SVM && m->noutputs != 1))
        ereport(ERROR, (errcode(ERRCODE_DATA_CORRUPTED), errmsg("invalid native model dimensions")));
    count = m->algorithm <= DB4AI_SVM ? m->nfeatures + 1 : m->nfeatures * m->noutputs;
    m->weights = read_array(json, "weights", count);
    if (m->algorithm == DB4AI_PCA)
        m->mean = read_array(json, "mean", m->nfeatures);
    return m;
}

double
db4ai_native_predict(const Db4aiNative *m, const double *features)
{
    double z;
    if (m->algorithm == DB4AI_PCA)
        ereport(ERROR, (errcode(ERRCODE_DATATYPE_MISMATCH), errmsg("use db4ai.transform for PCA models")));
    if (m->algorithm == DB4AI_KMEANS)
        return closest(m, features, NULL) + 1;
    z = score(m, features);
    return m->algorithm == DB4AI_LOGISTIC ? sigmoid(z) : z;
}

void
db4ai_native_transform(const Db4aiNative *m, const double *features, double *out)
{
    int c, j;
    if (m->algorithm != DB4AI_PCA)
        ereport(ERROR, (errcode(ERRCODE_DATATYPE_MISMATCH), errmsg("transform requires a PCA model")));
    for (c = 0; c < m->noutputs; c++)
    {
        out[c] = 0;
        for (j = 0; j < m->nfeatures; j++)
            out[c] += (features[j] - m->mean[j]) * m->weights[c * m->nfeatures + j];
        db4ai_check_finite(out[c]);
    }
}
