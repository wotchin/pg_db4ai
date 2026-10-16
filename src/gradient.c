/* Copyright (c) 2020 Huawei Technologies Co.,Ltd.
 * SPDX-License-Identifier: MulanPSL-2.0
 * C adaptation of DB4AI GD/NGD/OVA and matrix-free stochastic PCA.
 * NGD follows optimizer_ngd.cpp (normalized, NOT natural gradient descent).
 */
#include "postgres.h"
#include "db4ai.h"
#include "miscadmin.h"
#include "utils/timestamp.h"
#include <float.h>
#include <math.h>

static double
logistic(double z)
{
    if(z>=0) return 1/(1+exp(-z));
    z=exp(z); return z/(1+z);
}

static double
linear_score(const double *w, const double *x, int n)
{
    double value=0; int i;
    for(i=0;i<n;i++) value+=w[i]*x[i];
    db4ai_check_finite(value); return value;
}

void
db4ai_native_scores(const Db4aiNative *m, const double *x, double *scores)
{
    double *v=palloc(sizeof(double)*m->ncoefficients);
    int c;
    db4ai_kernel_transform(m,x,v);
    if(m->algorithm==DB4AI_MULTICLASS)
        for(c=0;c<m->nclasses;c++)
        {
            double z=linear_score(m->weights+(Size)c*m->ncoefficients,v,m->ncoefficients);
            scores[c]=m->classifier==DB4AI_LOGISTIC?logistic(z):z;
        }
    else
    {
        double z=linear_score(m->weights,v,m->ncoefficients);
        scores[1]=m->algorithm==DB4AI_LOGISTIC?logistic(z):z;
        scores[0]=m->algorithm==DB4AI_LOGISTIC?1-scores[1]:-z;
    }
    pfree(v);
}

int
db4ai_native_class(const Db4aiNative *m, const double *x)
{
    double scores[256]; int best=0,c;
    if(!db4ai_classifier(m->algorithm)) ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("model is not a classifier")));
    db4ai_native_scores(m,x,scores);
    if(m->nclasses==2 && m->algorithm!=DB4AI_MULTICLASS) return scores[1]>=scores[0]?1:0;
    for(c=1;c<m->nclasses;c++) if(scores[c]>scores[best]) best=c;
    return best;
}

static void
update_batch(Db4aiNative *m, const Db4aiOptions *o, const double *features, const double *labels,
             int rows, int epoch, double rate, double *scales, double *scale_rate, double *grad)
{
    int models=m->algorithm==DB4AI_MULTICLASS?m->nclasses:1;
    bool classifier=db4ai_classifier(m->algorithm);
    bool svm=m->algorithm==DB4AI_SVM || (m->algorithm==DB4AI_MULTICLASS && m->classifier==DB4AI_SVM);
    int c,r,j,n=m->ncoefficients;
    for(c=0;c<models;c++)
    {
        double *w=m->weights+(Size)c*n;
        double *s=scales+(Size)c*n;
        memset(grad,0,n*sizeof(double));
        if(o->optimizer==1 && epoch==0)
            for(r=0;r<rows;r++) for(j=0;j<n;j++)
            {
                double v=features[(Size)r*n+j], q=v*v;
                db4ai_check_finite(q);
                if(q>s[j]) { w[j]*=s[j]/q; s[j]=q; }
                if(s[j]>0) scale_rate[c]+=q/s[j];
            }
        for(r=0;r<rows;r++)
        {
            const double *x=features+(Size)r*n;
            double y=classifier?(m->algorithm==DB4AI_MULTICLASS?(labels[r]==c?1.0:0.0):labels[r]):labels[r];
            double z=linear_score(w,x,n);
            double error;
            CHECK_FOR_INTERRUPTS();
            if(svm)
            {
                double target=2*y-1;
                error=target*z<1?(-target*2*o->svm_lambda):0;
            }
            else error=classifier?logistic(z)-y:z-y;
            for(j=0;j<n;j++) grad[j]+=error*x[j];
        }
        for(j=0;j<n;j++)
        {
            double g=grad[j];
            if(o->optimizer==1) g=s[j]>0 && scale_rate[c]>0 ? g/s[j]/scale_rate[c] : 0;
            else g/=rows;
            if(j) g+=o->l2*w[j];
            w[j]-=rate*g; db4ai_check_finite(w[j]);
        }
    }
}

void
db4ai_gradient_train(Db4aiNative *m, Db4aiData *data, const Db4aiOptions *o)
{
    Db4aiShuffle *shuffle;
    double *batch, *labels, *scales, *scale_rate, *gradient, *row, *v;
    int n, epoch, models;
    double rate=o->learning_rate, previous=DBL_MAX;
    TimestampTz started=GetCurrentTimestamp();
    m->labels=data->labels; m->nclasses=data->labels?data->labels->count:0;
    m->noutputs=m->algorithm==DB4AI_MULTICLASS?m->nclasses:1;
    m->classifier=o->classifier?DB4AI_SVM:DB4AI_LOGISTIC;
    db4ai_kernel_prepare(m,o); n=m->ncoefficients;
    models=m->algorithm==DB4AI_MULTICLASS?m->nclasses:1;
    if((Size)o->batch_size*(n+1)*sizeof(double)>(Size)db4ai_work_mem*1024)
        ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("transformed gradient batch exceeds db4ai.work_mem")));
    m->weights=palloc0((Size)models*n*sizeof(double)); scales=palloc0((Size)models*n*sizeof(double));
    scale_rate=palloc0(models*sizeof(double)); gradient=palloc0(n*sizeof(double));
    batch=palloc((Size)o->batch_size*n*sizeof(double)); labels=palloc(o->batch_size*sizeof(double));
    row=palloc(data->ncolumns*sizeof(double)); v=palloc(n*sizeof(double));
    shuffle=db4ai_shuffle_create(data,o);
    for(epoch=0;epoch<o->iterations;epoch++)
    {
        const double *rows;
        int count, r, fill=0;
        double loss=0;
        db4ai_shuffle_start(shuffle,epoch);
        while((count=db4ai_shuffle_next(shuffle,&rows))>0)
            for(r=0;r<count;r++)
            {
                const double *raw=rows+(Size)r*data->ncolumns;
                db4ai_kernel_transform(m,raw+1,batch+(Size)fill*n); labels[fill]=raw[0];
                if(++fill==o->batch_size)
                {
                    update_batch(m,o,batch,labels,fill,epoch,rate,scales,scale_rate,gradient); fill=0;
                }
            }
        if(fill) update_batch(m,o,batch,labels,fill,epoch,rate,scales,scale_rate,gradient);
        db4ai_data_rewind(data);
        while(db4ai_data_next(data,row))
        {
            int c;
            CHECK_FOR_INTERRUPTS(); db4ai_kernel_transform(m,row+1,v);
            for(c=0;c<models;c++)
            {
                double z=linear_score(m->weights+(Size)c*n,v,n);
                double y=m->algorithm==DB4AI_MULTICLASS?(row[0]==c?1:0):row[0];
                if(m->algorithm==DB4AI_SVM || (m->algorithm==DB4AI_MULTICLASS && m->classifier==DB4AI_SVM))
                    loss+=Max(0,1-(2*y-1)*z);
                else if(db4ai_classifier(m->algorithm)) loss+=Max(z,0)-y*z+log1p(exp(-fabs(z)));
                else loss+=(z-y)*(z-y)*(o->legacy_profile?0.5:1.0);
            }
        }
        m->loss=loss/data->nrows; m->iterations=epoch+1; db4ai_check_finite(m->loss);
        if(o->verbose) ereport(NOTICE,(errmsg("DB4AI epoch %d loss %.12g",m->iterations,m->loss)));
        if(epoch && fabs(previous-m->loss)<=o->tolerance*(o->legacy_profile?1:Max(1,fabs(previous)))) break;
        previous=m->loss; rate*=o->decay; db4ai_check_finite(rate);
        if(o->max_seconds && TimestampDifferenceExceeds(started,GetCurrentTimestamp(),o->max_seconds*1000)) break;
    }
    db4ai_shuffle_free(shuffle);
    pfree(batch); pfree(labels); pfree(scales); pfree(scale_rate); pfree(gradient); pfree(row); pfree(v);
}

/* Reorthogonalized modified Gram-Schmidt; protects nearly dependent columns. */
static void
orthonormalize(double *w, int n, int k)
{
    int c,j,p,pass;
    for(c=0;c<k;c++)
    {
        double norm;
        for(pass=0;pass<2;pass++) for(p=0;p<c;p++)
        {
            double dot=0;
            for(j=0;j<n;j++) dot+=w[(Size)c*n+j]*w[(Size)p*n+j];
            for(j=0;j<n;j++) w[(Size)c*n+j]-=dot*w[(Size)p*n+j];
        }
        norm=0; for(j=0;j<n;j++) norm=hypot(norm,w[(Size)c*n+j]);
        if(norm<DBL_EPSILON)
        {
            /* Select a coordinate outside the already selected subspace. */
            int a; bool found=false;
            for(a=0;a<n && !found;a++)
            {
                memset(w+(Size)c*n,0,n*sizeof(double)); w[(Size)c*n+a]=1;
                for(p=0;p<c;p++)
                { double dot=w[(Size)p*n+a]; for(j=0;j<n;j++) w[(Size)c*n+j]-=dot*w[(Size)p*n+j]; }
                norm=0; for(j=0;j<n;j++) norm=hypot(norm,w[(Size)c*n+j]);
                found=norm>1e-10;
            }
            if(!found) ereport(ERROR,(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),errmsg("PCA basis is numerically singular")));
        }
        db4ai_check_finite(norm); for(j=0;j<n;j++) w[(Size)c*n+j]/=norm;
    }
}

void
db4ai_stochastic_pca(Db4aiNative *m, Db4aiData *data, const Db4aiOptions *o)
{
    int n=m->nfeatures,k=o->components,epoch,i,j,c;
    double *row=palloc(n*sizeof(double)); double *gradient;
    double *m2; double *average; double previous=DBL_MAX; int64 batches=0,seen=0;
    pg_prng_state rng;
    Db4aiShuffle *shuffle;
    TimestampTz started=GetCurrentTimestamp();
    if(o->legacy_profile) k=Min(k,n);
    if(k>n || data->nrows<2) ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("PCA requires 2+ rows and components <= features")));
    m->noutputs=k; m->mean=palloc0(n*sizeof(double)); m->weights=palloc((Size)n*k*sizeof(double));
    m->eigenvalues=palloc0(k*sizeof(double)); m->eigen_stddev=palloc0(k*sizeof(double));
    average=palloc0(k*sizeof(double)); m2=palloc0(k*sizeof(double)); gradient=palloc0((Size)n*k*sizeof(double));
    if(o->center)
    {
        db4ai_data_rewind(data);
        while(db4ai_data_next(data,row)) { CHECK_FOR_INTERRUPTS(); seen++; for(j=0;j<n;j++) m->mean[j]+=(row[j]-m->mean[j])/seen; }
    }
    pg_prng_seed(&rng,(uint64)o->seed ^ UINT64CONST(0x274066db9441e851));
    for(c=0;c<k;c++) for(j=0;j<n;j++) m->weights[(Size)c*n+j]=pg_prng_double(&rng);
    orthonormalize(m->weights,n,k);
    shuffle=db4ai_shuffle_create(data,o);
    for(epoch=0;epoch<o->iterations;epoch++)
    {
        const double *rows; int count;
        db4ai_shuffle_start(shuffle,epoch);
        while((count=db4ai_shuffle_next(shuffle,&rows))>0)
        {
            double total=0;
            memset(gradient,0,(Size)n*k*sizeof(double)); memset(m->eigenvalues,0,k*sizeof(double));
            for(i=0;i<count;i++)
            {
                CHECK_FOR_INTERRUPTS();
                for(j=0;j<n;j++) row[j]=rows[(Size)i*n+j]-m->mean[j];
                for(c=0;c<k;c++)
                {
                    double dot=linear_score(m->weights+(Size)c*n,row,n);
                    m->eigenvalues[c]+=dot*dot;
                    for(j=0;j<n;j++) gradient[(Size)c*n+j]+=row[j]*dot;
                }
            }
            for(c=0;c<k;c++) total+=m->eigenvalues[c];
            ++batches;
            for(c=0;c<k;c++)
            {
                double fraction=total>0?m->eigenvalues[c]/total:0;
                double delta=fraction-average[c]; average[c]+=delta/batches; m2[c]+=delta*(fraction-average[c]);
                m->eigen_stddev[c]=sqrt(Max(0,m2[c]/batches));
                for(j=0;j<n;j++) m->weights[(Size)c*n+j]+=gradient[(Size)c*n+j]/count;
            }
            orthonormalize(m->weights,n,k);
        }
        m->loss=0; for(c=0;c<k;c++) m->loss+=m->eigen_stddev[c];
        m->iterations=epoch+1;
        if(epoch>0 && fabs(previous-m->loss)<=o->tolerance) break;
        previous=m->loss;
        if(o->max_seconds && TimestampDifferenceExceeds(started,GetCurrentTimestamp(),o->max_seconds*1000)) break;
    }
    /* Final eigenvalue estimates and deterministic component signs. */
    memset(m->eigenvalues,0,k*sizeof(double)); db4ai_data_rewind(data);
    while(db4ai_data_next(data,row))
    {
        CHECK_FOR_INTERRUPTS(); for(j=0;j<n;j++) row[j]-=m->mean[j];
        for(c=0;c<k;c++) { double dot=linear_score(m->weights+(Size)c*n,row,n); m->eigenvalues[c]+=dot*dot/data->nrows; }
    }
    for(c=0;c<k;c++)
    {
        int best=c, anchor=0;
        for(i=c+1;i<k;i++) if(m->eigenvalues[i]>m->eigenvalues[best]) best=i;
        if(best!=c)
        {
            double tmp=m->eigenvalues[c]; m->eigenvalues[c]=m->eigenvalues[best]; m->eigenvalues[best]=tmp;
            tmp=m->eigen_stddev[c]; m->eigen_stddev[c]=m->eigen_stddev[best]; m->eigen_stddev[best]=tmp;
            for(j=0;j<n;j++) { tmp=m->weights[(Size)c*n+j]; m->weights[(Size)c*n+j]=m->weights[(Size)best*n+j]; m->weights[(Size)best*n+j]=tmp; }
        }
        for(j=1;j<n;j++) if(fabs(m->weights[(Size)c*n+j])>fabs(m->weights[(Size)c*n+anchor])) anchor=j;
        if(m->weights[(Size)c*n+anchor]<0) for(j=0;j<n;j++) m->weights[(Size)c*n+j]*=-1;
    }
    db4ai_shuffle_free(shuffle); pfree(row); pfree(gradient); pfree(average); pfree(m2);
}
