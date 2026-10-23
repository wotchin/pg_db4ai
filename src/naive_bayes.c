/* SPDX-License-Identifier: MulanPSL-2.0
 * Supervised Gaussian Naive Bayes, not the retired ABO joint estimator.
 * Label is column zero; numerical features follow. PG owns all state.
 * Welford population variances, explicit smoothing, log-space posteriors.
 */
#include "postgres.h"
#include "db4ai.h"
#include <math.h>
#include "miscadmin.h"
#include "utils/builtins.h"
#include "utils/fmgrprotos.h"
#include "utils/timestamp.h"

#define NB_LOG_2PI 1.83787706640934548356

static void
check_budget(int classes, int features, Size dictionary_bytes)
{
    Size arrays = ((Size)2 * classes * features + (Size)3 * classes +
                   (Size)4 * features + 1) * sizeof(double);
    Size limit = (Size) db4ai_max_model_mb * 1024 * 1024;
    if (dictionary_bytes > limit || arrays > limit - dictionary_bytes)
        ereport(ERROR, (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
                        errmsg("Gaussian NB state exceeds db4ai.max_model_mb")));
}

/* This explicit boundary prevents finite-but-extreme inputs from producing
 * NaN posteriors. Never silently clip a likelihood into a different model. */
static void
finite_likelihood(double x)
{
    if (!isfinite(x))
        ereport(ERROR, (errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
                        errmsg("Gaussian NB numeric range exceeded"),
                        errhint("Scale numerical features using training-set statistics, or increase variance_floor.")));
}

static void
prepare_normalizers(Db4aiNative *m)
{
    int c, j, n = m->nfeatures;
    m->nb_log_normalizer = palloc(sizeof(double) * m->nclasses);
    for (c = 0; c < m->nclasses; c++)
    {
        double sum = 0;
        for (j = 0; j < n; j++)
            sum += NB_LOG_2PI + log(m->nb_variances[(Size)c * n + j]);
        m->nb_log_normalizer[c] = log(m->nb_priors[c]) - 0.5 * sum;
        finite_likelihood(m->nb_log_normalizer[c]);
    }
}

/* Rare tail rows can have an enormous class-independent Gaussian penalty.
 * Subtract it per FEATURE before reduction, rather than subtracting two huge
 * final scores and erasing meaningful log odds. Keep the ordinary SIMD path
 * for well-scaled rows. Long double is only an extra guard, not a model format. */
static double
relative_tail_scores(const Db4aiNative *m, const double *x, double *out)
{
    long double relative[256], maximum = -HUGE_VALL;
    int c, j, reference = 0, n = m->nfeatures;
    /* Anchor at the best preliminary class, not an arbitrarily remote class. */
    for (c = 1; c < m->nclasses; c++) if (out[c] > out[reference]) reference = c;
    for (c = 0; c < m->nclasses; c++)
    {
        long double sum = 0;
        for (j = 0; j < n; j++)
        {
            long double a = m->nb_means[(Size)c*n+j], b = m->nb_means[(Size)reference*n+j];
            long double va = m->nb_variances[(Size)c*n+j], vb = m->nb_variances[(Size)reference*n+j];
            long double value = x[j], delta = value - b;
            /* Algebraic difference of squared distances, not square-minus-square.
             * Identical means/variances cancel exactly even in remote tails. */
            long double q = (b-a)*(2*value-a-b)/va;
            if (va != vb) q += (delta*delta/vb)*((vb-va)/va);
            sum += logl(va)-logl(vb)+q;
        }
        relative[c] = logl(m->nb_priors[c])-logl(m->nb_priors[reference])-0.5L*sum;
        if (!isfinite(relative[c])) finite_likelihood(HUGE_VAL);
        if (relative[c] > maximum) maximum = relative[c];
    }
    for (c = 0; c < m->nclasses; c++)
    {
        out[c] = (double)(relative[c]-maximum);
        finite_likelihood(out[c]);
    }
    return 0.0; /* scores are already centered */
}

void
db4ai_nb_log_proba(const Db4aiNative *m, const double *x, double *out)
{
    double residual[DB4AI_MAX_FEATURES];
    double maximum = -HUGE_VAL, total = 0;
    int c, j, n = m->nfeatures;
    for (c = 0; c < m->nclasses; c++)
    {
        const double *mean = m->nb_means + (Size)c * n;
        const double *variance = m->nb_variances + (Size)c * n;
        db4ai_automl_check();
        for (j = 0; j < n; j++)
            residual[j] = (x[j] - mean[j]) / sqrt(variance[j]);
        /* Reuse the audited runtime-dispatched SIMD squared-norm kernel.
         * No fabricated alignment, no PG calls inside the vector loop. */
        out[c] = m->nb_log_normalizer[c] - 0.5 * db4ai_dot(residual, residual, n);
        finite_likelihood(out[c]);
        maximum = Max(maximum, out[c]);
    }
    if (maximum < -1e6)
        maximum = relative_tail_scores(m, x, out);
    /* Subtract max BEFORE computing logsumexp; avoid large-score cancellation. */
    for (c = 0; c < m->nclasses; c++)
    {
        out[c] -= maximum;
        finite_likelihood(out[c]);
        total += exp(out[c]);
    }
    total = log(total); /* at least one term is exp(0), so total >= 1 */
    for (c = 0; c < m->nclasses; c++)
        out[c] -= total;
}

void
db4ai_nb_proba(const Db4aiNative *m, const double *x, double *out)
{
    int c;
    db4ai_nb_log_proba(m, x, out);
    for (c = 0; c < m->nclasses; c++) out[c] = exp(out[c]);
}

/* One-pass centered population statistics. The class/global populations have
 * separate counts; categories are never cast from original SQL values to float. */
static void
update_moments(const double *x, double *mean, double *m2, double count, int n)
{
    int j;
    for (j = 0; j < n; j++)
    {
        double delta = x[j] - mean[j];
        mean[j] += delta / count;
        m2[j] += delta * (x[j] - mean[j]);
        finite_likelihood(mean[j]);
        finite_likelihood(m2[j]);
    }
}

void
db4ai_nb_train(Db4aiNative *m, Db4aiData *data, const Db4aiOptions *o)
{
    int c, j, n = data->nfeatures, k = data->labels->count;
    double *row, *global_mean, *global_m2;
    double max_variance = 0, seen = 0, loss = 0;
    double logp[256];
    TimestampTz started = GetCurrentTimestamp();
    StringInfoData diagnostics;
    check_budget(k, n, data->labels->bytes);
    m->labels = data->labels;
    m->nclasses = m->noutputs = k;
    m->ncoefficients = n;
    m->iterations = 1;
    m->nb_counts = palloc0(sizeof(double) * k);
    m->nb_priors = palloc(sizeof(double) * k);
    m->nb_means = palloc0(sizeof(double) * (Size)k * n);
    m->nb_variances = palloc0(sizeof(double) * (Size)k * n); /* M2 until finalized */
    row = palloc(sizeof(double) * data->ncolumns);
    global_mean = palloc0(sizeof(double) * n);
    global_m2 = palloc0(sizeof(double) * n);
    db4ai_data_rewind(data);
    while (db4ai_data_next(data, row))
    {
        db4ai_automl_check();
        c = (int) row[0];
        Assert(c >= 0 && c < k);
        seen++;
        m->nb_counts[c]++;
        update_moments(row + 1, m->nb_means + (Size)c * n,
                       m->nb_variances + (Size)c * n, m->nb_counts[c], n);
        update_moments(row + 1, global_mean, global_m2, seen, n);
    }
    for (j = 0; j < n; j++) max_variance = Max(max_variance, global_m2[j] / seen);
    m->nb_epsilon = Max(o->variance_floor, o->var_smoothing * max_variance);
    finite_likelihood(m->nb_epsilon);
    for (c = 0; c < k; c++)
    {
        m->nb_priors[c] = o->fit_prior ? m->nb_counts[c] / seen : 1.0 / k;
        for (j = 0; j < n; j++)
        {
            Size index = (Size)c * n + j;
            m->nb_variances[index] = Max(0, m->nb_variances[index] / m->nb_counts[c]) + m->nb_epsilon;
            finite_likelihood(m->nb_variances[index]);
        }
    }
    prepare_normalizers(m);
    data->optimize_seconds = (GetCurrentTimestamp() - started) / 1000000.0;
    started = GetCurrentTimestamp();
    db4ai_data_rewind(data);
    while (db4ai_data_next(data, row))
    {
        db4ai_nb_log_proba(m, row + 1, logp);
        loss -= logp[(int) row[0]] / data->nrows;
    }
    finite_likelihood(loss);
    m->loss = loss;
    data->evaluation_seconds = (GetCurrentTimestamp() - started) / 1000000.0;
    initStringInfo(&diagnostics);
    appendStringInfoString(&diagnostics, "{\"training_loss_kind\":\"mean_negative_log_posterior\","
                          "\"probability_semantics\":\"Gaussian naive Bayes posterior; not calibrated\","
                          "\"class_priors\":");
    db4ai_write_array(&diagnostics, m->nb_priors, k);
    appendStringInfoString(&diagnostics, ",\"class_counts\":");
    db4ai_write_array(&diagnostics, m->nb_counts, k);
    appendStringInfo(&diagnostics, ",\"variance_epsilon\":%.17g}", m->nb_epsilon);
    m->diagnostics = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(diagnostics.data)));
    pfree(row); pfree(global_mean); pfree(global_m2); pfree(diagnostics.data);
}

bytea *
db4ai_nb_save(const Db4aiNative *m)
{
    StringInfoData out;
    bytea *result;
    initStringInfo(&out);
    appendStringInfo(&out, "{\"format_version\":1,\"algorithm\":\"gaussian_naive_bayes\","
                     "\"n_features\":%d,\"n_outputs\":%d,\"variance_epsilon\":%.17g,\"labels\":",
                     m->nfeatures, m->nclasses, m->nb_epsilon);
    db4ai_dictionary_save(&out, m->labels);
    appendStringInfoString(&out, ",\"class_counts\":"); db4ai_write_array(&out, m->nb_counts, m->nclasses);
    appendStringInfoString(&out, ",\"class_priors\":"); db4ai_write_array(&out, m->nb_priors, m->nclasses);
    appendStringInfoString(&out, ",\"means\":"); db4ai_write_array(&out, m->nb_means, m->nfeatures * m->nclasses);
    appendStringInfoString(&out, ",\"variances\":"); db4ai_write_array(&out, m->nb_variances, m->nfeatures * m->nclasses);
    appendStringInfoChar(&out, '}');
    result = db4ai_bytes(out.data, out.len);
    pfree(out.data);
    return result;
}

Db4aiNative *
db4ai_nb_load(const Db4aiModel *model)
{
    char *text, *algorithm;
    Jsonb *json;
    JsonbValue *labels;
    Db4aiNative *m;
    double prior_sum = 0;
    int c, j, n = model->nfeatures, k = model->noutputs;
    if (strcmp(model->format, "json") != 0 || n < 1 || n > DB4AI_MAX_FEATURES || k < 2 || k > 256)
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid Gaussian NB model dimensions or format")));
    check_budget(k, n, 0);
    text = pnstrdup(VARDATA_ANY(model->data), VARSIZE_ANY_EXHDR(model->data));
    json = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetDatum(text)));
    algorithm = db4ai_json_string(json, "algorithm", "");
    if (strcmp(algorithm, "gaussian_naive_bayes") || db4ai_json_number(json, "format_version", -1) != 1 ||
        db4ai_json_number(json, "n_features", -1) != n || db4ai_json_number(json, "n_outputs", -1) != k)
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("inconsistent Gaussian NB model header")));
    m = palloc0(sizeof(*m));
    m->algorithm = DB4AI_GAUSSIAN_NB;
    m->nfeatures = m->ncoefficients = n;
    m->nclasses = m->noutputs = k;
    m->nb_epsilon = db4ai_json_number(json, "variance_epsilon", -1);
    if (m->nb_epsilon <= 0)
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid Gaussian NB variance floor")));
    labels = db4ai_json_field(json, "labels", true);
    if (labels->type != jbvBinary || !JsonContainerIsObject(labels->val.binary.data))
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid Gaussian NB labels")));
    m->labels = db4ai_dictionary_load(JsonbValueToJsonb(labels));
    if (m->labels->count != k)
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("Gaussian NB label count mismatch")));
    check_budget(k, n, m->labels->bytes);
    m->nb_counts = db4ai_read_array(json, "class_counts", k);
    m->nb_priors = db4ai_read_array(json, "class_priors", k);
    m->nb_means = db4ai_read_array(json, "means", k * n);
    m->nb_variances = db4ai_read_array(json, "variances", k * n);
    for (c = 0; c < k; c++)
    {
        if (m->labels->nulls[c] || m->nb_counts[c] < 1 || floor(m->nb_counts[c]) != m->nb_counts[c] ||
            m->nb_priors[c] <= 0 || m->nb_priors[c] > 1)
            ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid Gaussian NB class statistics")));
        prior_sum += m->nb_priors[c];
        for (j = 0; j < n; j++)
            if (m->nb_variances[(Size)c * n + j] < m->nb_epsilon)
                ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid Gaussian NB variance")));
    }
    if (fabs(prior_sum - 1.0) > 1e-10)
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("Gaussian NB priors must sum to one")));
    prepare_normalizers(m);
    pfree(algorithm); pfree(text); pfree(json);
    return m;
}
