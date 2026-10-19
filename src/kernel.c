/* Copyright (c) 2020 Huawei Technologies Co.,Ltd.
 * SPDX-License-Identifier: MulanPSL-2.0
 * C adaptation of openGauss matrix.cpp / matrix.h random feature mappings.
 */
#include "postgres.h"
#include "db4ai.h"
#include "miscadmin.h"
#include <math.h>
#include <float.h>

/* Portable drand48 stream, rather than platform-specific *_r or C++ STL. */
static double
uniform48(uint64 *state)
{
    *state=(*state*UINT64CONST(0x5deece66d)+11)&UINT64CONST(0xffffffffffff);
    return (double)*state/281474976710656.0;
}

void
db4ai_kernel_prepare(Db4aiNative *m, const Db4aiOptions *o)
{
    int i,j,k;
    uint64 rng=((uint64)(uint32)o->seed<<16)|0x330e;
    m->kernel=o->kernel;
    m->kernel_components=m->kernel?(o->kernel_components?o->kernel_components:Max(128,2*m->nfeatures)):0;
    m->ncoefficients=1+(m->kernel?m->kernel_components:m->nfeatures);
    if(!m->kernel) return;
    k=m->kernel_components;
    if((Size)k*m->nfeatures*o->degree*sizeof(double)>(Size)db4ai_max_model_mb*1024*1024)
        ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("kernel map exceeds db4ai.max_model_mb")));
    if(m->kernel==1)
    {
        Size count=(Size)m->nfeatures*k, n=0;
        double sigma=sqrt(2.0*o->gamma);
        m->kernel_rows=m->nfeatures;
        m->kernel_weights=palloc(count*sizeof(double));
        m->kernel_offsets=palloc(k*sizeof(double));
        while(n<count)
        {
            double u,v,mag;
            CHECK_FOR_INTERRUPTS();
            do { u=uniform48(&rng); } while(u<=FLT_EPSILON);
            v=2.0*M_PI*uniform48(&rng); mag=sigma*sqrt(-2.0*log(u));
            m->kernel_weights[n++]=mag*cos(v);
            if(n<count) m->kernel_weights[n++]=mag*sin(v);
        }
        rng=((uint64)((uint32)o->seed+1)<<16)|0x330e;
        for(i=0;i<k;i++) m->kernel_offsets[i]=2.0*M_PI*uniform48(&rng);
    }
    else
    {
        int rows=0;
        m->kernel_repetitions=palloc(k*sizeof(double)); m->kernel_coefs=palloc(k*sizeof(double));
        for(i=0;i<k;i++)
        {
            int rep;
            double coeff=1;
            do { double u=uniform48(&rng); rep=u>0?(int)log2(1.0/u):0; } while(rep==0 || rep>o->degree);
            rows+=rep; m->kernel_repetitions[i]=rep;
            for(j=1;j<=rep;j++) coeff*= (double)(o->degree-j+1)/j;
            /* Correct homogeneous coef0=0: the highest-degree coefficient is 1. */
            coeff*=pow(o->coef0,o->degree-rep);
            m->kernel_coefs[i]=sqrt(coeff*(1U<<(rep+1)));
        }
        m->kernel_rows=rows; m->kernel_weights=palloc((Size)rows*m->nfeatures*sizeof(double));
        rng=((uint64)(uint32)o->seed<<16)|0x330e;
        for(i=0;i<rows;i++) for(j=0;j<m->nfeatures;j++)
            m->kernel_weights[(Size)i*m->nfeatures+j]=uniform48(&rng)<0.5?-1.0:1.0;
    }
}

void
db4ai_kernel_transform(const Db4aiNative *m, const double *input, double *out)
{
    int c;
    out[0]=1.0;
    if(!m->kernel) { memcpy(out+1,input,m->nfeatures*sizeof(double)); return; }
    if(m->kernel==1)
    {
        double scale=sqrt(2.0/m->kernel_components);
        db4ai_numeric->gaussian(out+1,input,m->kernel_weights,m->kernel_offsets,
                                 m->nfeatures,m->kernel_components);
        for(c=0;c<m->kernel_components;c++)
        {
            double v=out[c+1];
            db4ai_check_finite(v); out[c+1]=cos(v)*scale;
        }
    }
    else
    {
        int row=0;
        for(c=0;c<m->kernel_components;c++)
        {
            int r; double product=1;
            for(r=0;r<(int)m->kernel_repetitions[c];r++,row++)
            {
                double value=db4ai_dot(input,m->kernel_weights+(Size)row*m->nfeatures,m->nfeatures);
                product*=value;
            }
            out[c+1]=product*m->kernel_coefs[c]/sqrt((double)m->kernel_components);
            db4ai_check_finite(out[c+1]);
        }
    }
}
