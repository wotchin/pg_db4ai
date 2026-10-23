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
#include "utils/timestamp.h"
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
distance_metric(const double *a, const double *b, int n, int metric)
{
    double sum=db4ai_numeric->distance(a,b,n,metric);
    db4ai_check_finite(sum); return sum;
}

static int
closest(const Db4aiNative *m, const double *x, double *distance)
{
    double best = DBL_MAX;
    int result = 0;
    int c;
    for (c = 0; c < m->noutputs; c++)
    {
        double d = distance_metric(x, m->weights + c * m->nfeatures, m->nfeatures, m->distance);
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

/* Uniform candidate sampling (Random++) or scalable distance sampling
 * (KMeans||), followed by weighted k-means++ reduction. No STL containers.
 */
static int
candidate_closest(const double *points, int count, const double *row, int d, int metric, double *cost)
{
    int i,best=0; double distance=DBL_MAX;
    for(i=0;i<count;i++)
    {
        double v=distance_metric(points+(Size)i*d,row,d,metric);
        if(v<distance) { distance=v; best=i; }
    }
    if(cost) *cost=distance;
    return best;
}

static void
seed_kmeans(Db4aiNative *m, Db4aiData *data, const Db4aiOptions *o)
{
    pg_prng_state rng;
    int k=m->noutputs,d=m->nfeatures;
    int cap=(int)Min((int64)k*o->oversampling*o->seeding_rounds+1,data->nrows);
    Size budget=(Size)db4ai_work_mem*1024;
    double *points, *weight, *cost, *row;
    bool *chosen;
    int nc=0,c,j,round;
    int maxcap=(int)Min(data->nrows,(int64)(budget/(d*sizeof(double)+2*sizeof(double)+sizeof(bool)+32)));
    int64 seen=0;
    if(o->seeding==0) cap=(int)Min(data->nrows,(int64)k*o->oversampling);
    if(o->seeding==2) cap=k;
    if((Size)cap*(d*sizeof(double)+sizeof(double)*2+sizeof(bool))>budget)
        ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("KMeans seeding candidates exceed db4ai.work_mem"),errhint("Reduce k/oversampling/seeding_rounds or increase db4ai.work_mem.")));
    points=palloc((Size)cap*d*sizeof(double)); weight=palloc0(cap*sizeof(double));
    cost=palloc(cap*sizeof(double)); chosen=palloc0(cap*sizeof(bool)); row=palloc(d*sizeof(double));
    pg_prng_seed(&rng,(uint64)o->seed);
    db4ai_data_rewind(data);
    /* Initial uniform sample; Random++ keeps an oversampled candidate set. */
    while(db4ai_data_next(data,row))
    {
        int target=o->seeding==1?1:cap;
        db4ai_automl_check();
        if(seen<target) memcpy(points+(Size)seen*d,row,d*sizeof(double));
        else { uint64 pos=pg_prng_uint64_range(&rng,0,seen); if(pos<(uint64)target) memcpy(points+(Size)pos*d,row,d*sizeof(double)); }
        seen++;
    }
    nc=o->seeding==1?1:cap;
    if(o->seeding==1)
        for(round=0;round<o->seeding_rounds;round++)
        {
            double total=0;
            int original=nc;
            db4ai_data_rewind(data);
            while(db4ai_data_next(data,row))
            { double v; db4ai_automl_check(); candidate_closest(points,original,row,d,o->distance,&v); total+=v; }
            db4ai_check_finite(total);
            if(total==0) break;
            db4ai_data_rewind(data);
            while(db4ai_data_next(data,row))
            {
                double v;
                db4ai_automl_check(); candidate_closest(points,original,row,d,o->distance,&v);
                if(pg_prng_double(&rng)<Min(1.0,(double)o->oversampling*k*v/total))
                {
                    if(nc==cap)
                    {
                        int oldcap=cap;
                        if(cap>=maxcap)
                            ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("KMeans|| candidate cache exhausted db4ai.work_mem"),
                                           errhint("Increase db4ai.work_mem or reduce oversampling. Candidates are never silently truncated.")));
                        cap=Min(maxcap,cap*2);
                        points=repalloc(points,(Size)cap*d*sizeof(double));
                        weight=repalloc(weight,cap*sizeof(double)); cost=repalloc(cost,cap*sizeof(double));
                        chosen=repalloc(chosen,cap*sizeof(bool)); memset(chosen+oldcap,0,(cap-oldcap)*sizeof(bool));
                    }
                    memcpy(points+(Size)nc*d,row,d*sizeof(double)); nc++;
                }
            }
        }
    /* Degenerate/identical data may produce fewer candidates than k. */
    if(nc<k)
    {
        db4ai_data_rewind(data);
        while(nc<k && db4ai_data_next(data,row)) { memcpy(points+(Size)nc*d,row,d*sizeof(double)); nc++; }
    }
    for(c=0;c<nc;c++) { weight[c]=o->seeding==1?0:1; cost[c]=DBL_MAX; }
    if(o->seeding==1)
    {
        db4ai_data_rewind(data);
        while(db4ai_data_next(data,row)) { db4ai_automl_check(); weight[candidate_closest(points,nc,row,d,o->distance,NULL)]+=1; }
    }
    if(o->seeding==2) memcpy(m->weights,points,(Size)k*d*sizeof(double));
    else for(c=0;c<k;c++)
    {
        int selected=-1;
        if(c==0) selected=(int)pg_prng_uint64_range(&rng,0,nc-1);
        else
        {
            double sum=0,draw;
            for(j=0;j<nc;j++) if(!chosen[j]) sum+=weight[j]*cost[j];
            db4ai_check_finite(sum);
            if(sum>0)
            {
                draw=pg_prng_double(&rng)*sum;
                for(j=0;j<nc;j++) if(!chosen[j]) { draw-=weight[j]*cost[j]; if(draw<=0) { selected=j; break; } }
            }
            if(selected<0) for(j=0;j<nc;j++) if(!chosen[j]) { selected=j; break; }
        }
        Assert(selected>=0); chosen[selected]=true;
        memcpy(m->weights+(Size)c*d,points+(Size)selected*d,d*sizeof(double));
        for(j=0;j<nc;j++) cost[j]=Min(cost[j],distance_metric(points+(Size)j*d,points+(Size)selected*d,d,o->distance));
    }
    pfree(points); pfree(weight); pfree(cost); pfree(chosen); pfree(row);
}

static void
train_kmeans(Db4aiNative *m, Db4aiData *data, const Db4aiOptions *o)
{
    double *row = palloc(sizeof(double) * m->nfeatures);
    double *sums;
    int64 *counts;
    int epoch;
    int c;
    int j;
    double previous_cost=DBL_MAX;
    TimestampTz started=GetCurrentTimestamp();
    if (o->clusters > data->nrows && !o->legacy_profile)
        ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("k exceeds the number of input rows")));
    m->noutputs = (int)Min((int64)o->clusters,data->nrows);
    m->weights = palloc0(sizeof(double) * m->nfeatures * m->noutputs);
    sums = palloc0(sizeof(double) * m->nfeatures * m->noutputs);
    counts = palloc0(sizeof(int64) * m->noutputs);
    seed_kmeans(m,data,o);
    for (epoch = 0; epoch < o->iterations; epoch++)
    {
        double movement = 0;
        double objective = 0;
        memset(sums, 0, sizeof(double) * m->nfeatures * m->noutputs);
        memset(counts, 0, sizeof(int64) * m->noutputs);
        db4ai_data_rewind(data);
        while (db4ai_data_next(data, row))
        {
            double distance;
            db4ai_automl_check();
            c = closest(m, row, &distance);
            objective += distance;
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
        db4ai_check_finite(objective);
        if(o->verbose) ereport(NOTICE,(errmsg("DB4AI kmeans iteration %d cost %.12g movement %.12g",m->iterations,objective,movement)));
        if(o->legacy_profile)
        {
            if(data->nrows<=m->noutputs || objective==0 ||
               (epoch>0 && previous_cost/objective-1 < o->tolerance)) break;
        }
        else if(movement <= o->tolerance) break;
        previous_cost=objective;
        if(o->max_seconds && TimestampDifferenceExceeds(started,GetCurrentTimestamp(),o->max_seconds*1000)) break;
    }
    m->loss = 0;
    db4ai_data_rewind(data);
    while (db4ai_data_next(data, row))
    {
        double d;
        db4ai_automl_check();
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
    TimestampTz started=GetCurrentTimestamp();
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
        db4ai_automl_check();
        count++;
        for (i = 0; i < n; i++)
        {
            delta[i] = row[i] - m->mean[i];
            if(o->center) m->mean[i] += delta[i] / count;
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
            a[i * n + j] /= o->center ? count - 1 : count;
            db4ai_check_finite(a[i * n + j]);
            a[j * n + i] = a[i * n + j];
        }
    }
    /* Symmetric Jacobi eigensolver, intentionally free of BLAS/LAPACK dependencies. */
    for (sweep = 0; sweep < o->iterations; sweep++)
    {
        double off = 0;
        double scale = 0;
        db4ai_automl_check();
        if(o->max_seconds && TimestampDifferenceExceeds(started,GetCurrentTimestamp(),o->max_seconds*1000)) break;
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
    m->eigenvalues=palloc0(m->noutputs*sizeof(double));
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
        m->eigenvalues[p]=Max(0.0,a[best*n+best]);
        a[best * n + best] = -DBL_MAX;
    }
    m->loss = 0; /* Residual sample variance after the selected components. */
    for (i = 0; i < n; i++)
        if (a[i * n + i] != -DBL_MAX)
            m->loss += Max(0.0, a[i * n + i]);
    pfree(a); pfree(v); pfree(row); pfree(delta);
}

/* User-visible model diagnostics are DB4AI functionality, unlike ABO's use of
 * Bayes estimates inside an optimizer. Store these once with the model. */
static void
native_diagnostics(Db4aiNative *m, Db4aiData *data)
{
    StringInfoData json;
    int c;
    initStringInfo(&json);
    if(m->algorithm==DB4AI_KMEANS)
    {
        int k=m->noutputs;
        int64 *count=palloc0(k*sizeof(int64));
        double *mean=palloc0(k*sizeof(double)), *m2=palloc0(k*sizeof(double));
        double *min=palloc(k*sizeof(double)), *max=palloc0(k*sizeof(double));
        double *row=palloc(data->ncolumns*sizeof(double));
        for(c=0;c<k;c++) min[c]=DBL_MAX;
        db4ai_data_rewind(data);
        while(db4ai_data_next(data,row))
        {
            double distance,delta;
            db4ai_automl_check();
            c=closest(m,row,&distance);
            delta=distance-mean[c];
            mean[c]+=delta/(++count[c]);
            m2[c]+=delta*(distance-mean[c]);
            min[c]=Min(min[c],distance); max[c]=Max(max[c],distance);
        }
        appendStringInfo(&json,"{\"centroids\":[");
        for(c=0;c<k;c++)
        {
            if(c) appendStringInfoChar(&json,',');
            appendStringInfo(&json,"{\"id\":%d,\"size\":" INT64_FORMAT ",\"coordinates\":",c,count[c]);
            db4ai_write_array(&json,m->weights+(Size)c*m->nfeatures,m->nfeatures);
            if(count[c])
            {
                double stats[]={mean[c]*count[c],min[c],max[c],mean[c],sqrt(Max(0.0,m2[c]/count[c]))};
                appendStringInfoString(&json,",\"distance_stats\":");
                db4ai_write_array(&json,stats,5);
            }
            else appendStringInfoString(&json,",\"distance_stats\":null");
            appendStringInfoChar(&json,'}');
        }
        appendStringInfoString(&json,"],\"distance_stat_order\":[\"sum\",\"min\",\"max\",\"mean\",\"population_stddev\"]}");
        pfree(row); pfree(count); pfree(mean); pfree(m2); pfree(min); pfree(max);
    }
    else if(m->algorithm==DB4AI_PCA)
    {
        double total=0;
        for(c=0;c<m->noutputs;c++) total+=m->eigenvalues[c];
        appendStringInfoString(&json,"{\"principal_components\":[");
        for(c=0;c<m->noutputs;c++)
        {
            double stats[]={m->eigenvalues[c],total>0?m->eigenvalues[c]/total:0};
            if(c) appendStringInfoChar(&json,',');
            appendStringInfo(&json,"{\"id\":%d,\"coordinates\":",c);
            db4ai_write_array(&json,m->weights+(Size)c*m->nfeatures,m->nfeatures);
            appendStringInfoString(&json,",\"eigenvalue_and_selected_fraction\":");
            db4ai_write_array(&json,stats,2);
            appendStringInfoString(&json,",\"proportion_stddev_estimate\":");
            if(m->eigen_stddev) db4ai_write_array(&json,m->eigen_stddev+c,1);
            else appendStringInfoString(&json,"null");
            appendStringInfoChar(&json,'}');
        }
        appendStringInfoString(&json,"]}");
    }
    else { pfree(json.data); return; }
    m->diagnostics=DatumGetJsonbP(DirectFunctionCall1(jsonb_in,CStringGetDatum(json.data)));
    pfree(json.data);
}

Db4aiNative *
db4ai_native_train(Db4aiData *data, Db4aiAlgorithm algorithm, const Db4aiOptions *options)
{
    Db4aiNative *model = palloc0(sizeof(*model));
    model->algorithm = algorithm;
    model->nfeatures = data->nfeatures;
    model->distance=options->distance;
    if (algorithm == DB4AI_KMEANS)
        train_kmeans(model, data, options);
    else if (algorithm == DB4AI_PCA)
    {
        if(options->pca_solver) db4ai_stochastic_pca(model,data,options);
        else train_pca(model, data, options);
    }
    else if (algorithm == DB4AI_GAUSSIAN_NB)
        db4ai_nb_train(model, data, options);
    else
        db4ai_gradient_train(model, data, options);
    native_diagnostics(model,data);
    return model;
}

void
db4ai_write_array(StringInfo b, const double *values, int count)
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
    bool gd=m->algorithm<=DB4AI_SVM || m->algorithm==DB4AI_MULTICLASS;
    int count=gd?m->ncoefficients*(m->algorithm==DB4AI_MULTICLASS?m->nclasses:1):m->nfeatures*m->noutputs;
    if (m->algorithm == DB4AI_GAUSSIAN_NB) return db4ai_nb_save(m);
    initStringInfo(&b);
    appendStringInfoString(&b,"{\"format_version\":2,\"algorithm\":");
    escape_json(&b,db4ai_algorithm_name(m->algorithm));
    appendStringInfo(&b,",\"n_features\":%d,\"n_outputs\":%d,\"n_coefficients\":%d,\"n_classes\":%d,\"classifier\":%d,\"distance\":%d,\"weights\":",
                     m->nfeatures,m->noutputs,m->ncoefficients,m->nclasses,m->classifier,m->distance);
    db4ai_write_array(&b,m->weights,count);
    if(m->mean) { appendStringInfoString(&b,",\"mean\":"); db4ai_write_array(&b,m->mean,m->nfeatures); }
    if(m->labels) { appendStringInfoString(&b,",\"labels\":"); db4ai_dictionary_save(&b,m->labels); }
    if(m->eigenvalues) { appendStringInfoString(&b,",\"eigenvalues\":"); db4ai_write_array(&b,m->eigenvalues,m->noutputs); }
    if(m->eigen_stddev) { appendStringInfoString(&b,",\"eigen_stddev\":"); db4ai_write_array(&b,m->eigen_stddev,m->noutputs); }
    if(m->kernel)
    {
        appendStringInfo(&b,",\"kernel\":%d,\"kernel_components\":%d,\"kernel_rows\":%d,\"kernel_weights\":",m->kernel,m->kernel_components,m->kernel_rows);
        db4ai_write_array(&b,m->kernel_weights,m->kernel_rows*(m->kernel==1?m->kernel_components:m->nfeatures));
        if(m->kernel==1) { appendStringInfoString(&b,",\"kernel_offsets\":"); db4ai_write_array(&b,m->kernel_offsets,m->kernel_components); }
        else
        {
            appendStringInfoString(&b,",\"kernel_repetitions\":"); db4ai_write_array(&b,m->kernel_repetitions,m->kernel_components);
            appendStringInfoString(&b,",\"kernel_coefs\":"); db4ai_write_array(&b,m->kernel_coefs,m->kernel_components);
        }
    }
    appendStringInfoChar(&b,'}'); return db4ai_bytes(b.data,b.len);
}

double *
db4ai_read_array(Jsonb *json, const char *name, int expected)
{
    JsonbValue *v = db4ai_json_field(json, name, true);
    JsonbIterator *it;
    JsonbIteratorToken token;
    JsonbValue elem;
    double *out;
    int i = 0;
    if (v->type != jbvBinary || !JsonContainerIsArray(v->val.binary.data) ||
        JsonContainerSize(v->val.binary.data) != expected)
        ereport(ERROR, (errcode(ERRCODE_DATA_CORRUPTED), errmsg("invalid model array: %s", name)));
    if(expected<0 || (Size)expected>(Size)db4ai_max_model_mb*1024*1024/sizeof(double))
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("model array exceeds db4ai.max_model_mb")));
    out=palloc(sizeof(double)*expected);
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

static int
model_integer(Jsonb *json, const char *key, int def, int low, int high)
{
    double value=db4ai_json_number(json,key,def);
    if(value<low || value>high || floor(value)!=value)
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid native model field: %s",key)));
    return (int)value;
}

Db4aiNative *
db4ai_native_load(const Db4aiModel *model)
{
    char *text;
    Jsonb *json;
    JsonbValue *algorithm;
    Db4aiNative *m;
    bool gd=model->algorithm<=DB4AI_SVM || model->algorithm==DB4AI_MULTICLASS;
    int count; JsonbValue *v;
    if (model->algorithm == DB4AI_GAUSSIAN_NB) return db4ai_nb_load(model);
    text=pnstrdup(VARDATA_ANY(model->data),VARSIZE_ANY_EXHDR(model->data));
    json=DatumGetJsonbP(DirectFunctionCall1(jsonb_in,CStringGetDatum(text)));
    algorithm=db4ai_json_field(json,"algorithm",true);
    m=palloc0(sizeof(*m));
    (void)model_integer(json,"format_version",-1,1,2);
    if(db4ai_json_number(json,"n_features",-1)!=model->nfeatures || db4ai_json_number(json,"n_outputs",-1)!=model->noutputs ||
       algorithm->type!=jbvString || db4ai_algorithm(pnstrdup(algorithm->val.string.val,algorithm->val.string.len))!=model->algorithm)
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("inconsistent native model header")));
    m->algorithm=model->algorithm; m->nfeatures=model->nfeatures; m->noutputs=model->noutputs;
    m->kernel=model_integer(json,"kernel",0,0,2);
    m->kernel_components=model_integer(json,"kernel_components",0,0,4096);
    m->kernel_rows=model_integer(json,"kernel_rows",0,0,4096*9);
    m->ncoefficients=model_integer(json,"n_coefficients",gd?m->nfeatures+1:0,0,4097);
    m->nclasses=model_integer(json,"n_classes",db4ai_classifier(m->algorithm)?2:0,0,256);
    m->classifier=model_integer(json,"classifier",DB4AI_SVM,DB4AI_LINEAR,DB4AI_SVM);
    if(m->algorithm==DB4AI_MULTICLASS && m->classifier!=DB4AI_LOGISTIC && m->classifier!=DB4AI_SVM)
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid OVA classifier")));
    m->distance=model_integer(json,"distance",2,0,3);
    if(gd && (m->ncoefficients!=1+(m->kernel?m->kernel_components:m->nfeatures) ||
       (m->algorithm==DB4AI_MULTICLASS?m->noutputs!=m->nclasses:m->noutputs!=1)))
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid gradient model dimensions")));
    if(db4ai_classifier(m->algorithm) && (m->nclasses<2 || (m->algorithm!=DB4AI_MULTICLASS && m->nclasses!=2)))
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid classifier dimensions")));
    count=gd?m->ncoefficients*(m->algorithm==DB4AI_MULTICLASS?m->nclasses:1):m->nfeatures*m->noutputs;
    m->weights=db4ai_read_array(json,"weights",count);
    if(m->algorithm==DB4AI_PCA) m->mean=db4ai_read_array(json,"mean",m->nfeatures);
    v=db4ai_json_field(json,"labels",false);
    if(v)
    {
        if(v->type!=jbvBinary || !JsonContainerIsObject(v->val.binary.data)) ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid label dictionary")));
        m->labels=db4ai_dictionary_load(JsonbValueToJsonb(v));
        if(m->labels->count!=m->nclasses) ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("label count does not match weights")));
    }
    if(db4ai_json_field(json,"eigenvalues",false)) m->eigenvalues=db4ai_read_array(json,"eigenvalues",m->noutputs);
    if(db4ai_json_field(json,"eigen_stddev",false)) m->eigen_stddev=db4ai_read_array(json,"eigen_stddev",m->noutputs);
    if(m->kernel)
    {
        int i,total=0;
        if(!gd || !m->kernel_components || !m->kernel_rows || (m->kernel==1 && m->kernel_rows!=m->nfeatures))
            ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid kernel dimensions")));
        m->kernel_weights=db4ai_read_array(json,"kernel_weights",m->kernel_rows*(m->kernel==1?m->kernel_components:m->nfeatures));
        if(m->kernel==1) m->kernel_offsets=db4ai_read_array(json,"kernel_offsets",m->kernel_components);
        else
        {
            m->kernel_repetitions=db4ai_read_array(json,"kernel_repetitions",m->kernel_components);
            m->kernel_coefs=db4ai_read_array(json,"kernel_coefs",m->kernel_components);
            for(i=0;i<m->kernel_components;i++)
            {
                double r=m->kernel_repetitions[i];
                if(r<1 || r>9 || floor(r)!=r) ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid kernel repetition")));
                total+=(int)r;
            }
            if(total!=m->kernel_rows) ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("kernel row count mismatch")));
        }
    }
    return m;
}

double
db4ai_native_predict(const Db4aiNative *m, const double *features)
{
    double *v,z;
    if(m->algorithm==DB4AI_GAUSSIAN_NB)
    {
        double probabilities[256];
        if(m->nclasses!=2)
            ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("multiclass Gaussian NB requires predict_label or predict_proba")));
        db4ai_nb_proba(m,features,probabilities);
        return probabilities[1];
    }
    if(m->algorithm==DB4AI_PCA) ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("use db4ai.transform for PCA models")));
    if(m->algorithm==DB4AI_KMEANS) return closest(m,features,NULL)+1;
    if(m->algorithm==DB4AI_MULTICLASS) return db4ai_native_class(m,features);
    v=palloc(m->ncoefficients*sizeof(double)); db4ai_kernel_transform(m,features,v);
    z=db4ai_dot(v,m->weights,m->ncoefficients);
    pfree(v); db4ai_check_finite(z);
    return m->algorithm==DB4AI_LOGISTIC?sigmoid(z):z;
}

void
db4ai_native_transform(const Db4aiNative *m, const double *features, double *out)
{
    int c;
    if (m->algorithm != DB4AI_PCA)
        ereport(ERROR, (errcode(ERRCODE_DATATYPE_MISMATCH), errmsg("transform requires a PCA model")));
    for (c = 0; c < m->noutputs; c++)
    {
        out[c] = db4ai_numeric->centered_dot(features,m->mean,
                        m->weights+(Size)c*m->nfeatures,m->nfeatures);
        db4ai_check_finite(out[c]);
    }
}
