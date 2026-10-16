/*
 * Copyright (c) 2020 Huawei Technologies Co.,Ltd.
 * SPDX-License-Identifier: MulanPSL-2.0
 * Adapted design: openGauss executor/optimize/gd/shuffle_cache.cpp.
 * First-epoch sequential traversal, full-cache random permutations, and a
 * bounded replacement cache when the working set does not fit. Every source
 * batch is returned once per epoch; the final partial batch is never dropped.
 * Unlike the original fallback, rescans read a PG tuplestore, not source SQL.
 */
#include "postgres.h"
#include "db4ai.h"
#include "miscadmin.h"

struct Db4aiShuffle
{
    MemoryContext context;
    Db4aiData *data;
    int mode;
    int unit;
    int result_rows;
    double *row_batch;
    int seed;
    bool first_shuffle;
    bool full;
    bool sequential;
    int capacity;
    int active;
    int cursor;
    int *order;
    int *counts;
    double **cache;
    double *output;
    Size unit_bytes;
    int64 emitted;
    pg_prng_state rng;
};

static int
read_unit(Db4aiShuffle *s, double *out)
{
    int n=0;
    while(n<s->unit)
    {
        CHECK_FOR_INTERRUPTS();
        if(!db4ai_data_next(s->data,out+(Size)n*s->data->ncolumns)) break;
        n++;
    }
    return n;
}

Db4aiShuffle *
db4ai_shuffle_create(Db4aiData *data, const Db4aiOptions *o)
{
    MemoryContext ctx=AllocSetContextCreate(CurrentMemoryContext,"DB4AI ShuffleScan",ALLOCSET_DEFAULT_SIZES);
    MemoryContext old=MemoryContextSwitchTo(ctx);
    Db4aiShuffle *s=palloc0(sizeof(*s));
    Size budget=(Size)db4ai_work_mem*1024;
    Size rowbytes=(Size)data->ncolumns*sizeof(double);
    Size unitbytes, reserved;
    int64 units;
    int capacity;
    int i;
    s->context=ctx; s->data=data; s->mode=o->shuffle_mode; s->unit=s->mode==DB4AI_SHUFFLE_ROW?1:o->batch_size;
    s->seed=o->seed; s->first_shuffle=o->shuffle_first;
    if((Size)s->unit>budget/rowbytes/2)
        ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("batch_size does not fit db4ai.work_mem for ShuffleScan"),errhint("Reduce batch_size or increase db4ai.work_mem.")));
    unitbytes=(Size)s->unit*rowbytes; s->unit_bytes=unitbytes;
    s->result_rows=o->batch_size;
    reserved=unitbytes+(s->mode==DB4AI_SHUFFLE_ROW?(Size)o->batch_size*rowbytes:0);
    if(reserved>budget/2)
        ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("shuffle output batch exceeds db4ai.work_mem")));
    capacity=(int)Min((Size)o->shuffle_buffer,(budget-reserved)/(unitbytes+64+sizeof(double *)+2*sizeof(int)));
    if(s->mode==DB4AI_SHUFFLE_NONE) capacity=1;
    capacity=Max(1,capacity);
    units=(data->nrows+s->unit-1)/s->unit;
    s->capacity=(int)Min((int64)capacity,units);
    s->full=units<=capacity && s->mode!=DB4AI_SHUFFLE_NONE;
    s->output=palloc(unitbytes);
    if(s->mode==DB4AI_SHUFFLE_ROW) s->row_batch=palloc((Size)o->batch_size*rowbytes);
    s->cache=palloc0(s->capacity*sizeof(double *));
    s->counts=palloc0(s->capacity*sizeof(int)); s->order=palloc0(s->capacity*sizeof(int));
    for(i=0;i<s->capacity;i++) s->cache[i]=palloc(unitbytes);
    if(s->full)
    {
        db4ai_data_rewind(data);
        for(i=0;i<s->capacity;i++) s->counts[i]=read_unit(s,s->cache[i]);
    }
    data->shuffle_capacity=s->capacity; data->shuffle_full_cache=s->full;
    MemoryContextSwitchTo(old); return s;
}

void
db4ai_shuffle_start(Db4aiShuffle *s, int epoch)
{
    int i;
    s->emitted=0; s->cursor=0; s->active=0;
    s->sequential=s->mode==DB4AI_SHUFFLE_NONE || (epoch==0 && !s->first_shuffle);
    pg_prng_seed(&s->rng,(uint64)s->seed ^ ((uint64)epoch*UINT64CONST(0x9e3779b97f4a7c15)));
    if(s->full)
    {
        for(i=0;i<s->capacity;i++) s->order[i]=i;
        if(!s->sequential) for(i=s->capacity-1;i>0;i--)
        {
            int j=(int)pg_prng_uint64_range(&s->rng,0,i); int tmp=s->order[i];
            s->order[i]=s->order[j]; s->order[j]=tmp;
        }
    }
    else
    {
        db4ai_data_rewind(s->data);
        if(!s->sequential) for(i=0;i<s->capacity;i++)
        {
            s->counts[i]=read_unit(s,s->cache[i]);
            if(!s->counts[i]) break;
            s->active++;
        }
    }
}

static int
next_unit(Db4aiShuffle *s, const double **rows)
{
    int n=0;
    CHECK_FOR_INTERRUPTS();
    if(s->full)
    {
        if(s->cursor<s->capacity)
        {
            int i=s->order[s->cursor++]; n=s->counts[i]; *rows=s->cache[i];
        }
    }
    else if(s->sequential) { n=read_unit(s,s->output); *rows=s->output; }
    else if(s->active)
    {
        int i=(int)pg_prng_uint64_range(&s->rng,0,s->active-1);
        n=s->counts[i]; memcpy(s->output,s->cache[i],(Size)n*s->data->ncolumns*sizeof(double));
        s->counts[i]=read_unit(s,s->cache[i]);
        if(!s->counts[i])
        {
            double *tmp=s->cache[i];
            --s->active; s->cache[i]=s->cache[s->active]; s->cache[s->active]=tmp;
            s->counts[i]=s->counts[s->active];
        }
        *rows=s->output;
    }
    s->emitted+=n;
    if(!n && s->emitted!=s->data->nrows)
        elog(ERROR,"ShuffleScan violated exact-once epoch accounting");
    return n;
}

int
db4ai_shuffle_next(Db4aiShuffle *s, const double **rows)
{
    int count=0;
    const double *one;
    if(s->mode!=DB4AI_SHUFFLE_ROW) return next_unit(s,rows);
    while(count<s->result_rows && next_unit(s,&one)>0)
    {
        memcpy(s->row_batch+(Size)count*s->data->ncolumns,one,s->data->ncolumns*sizeof(double));
        count++;
    }
    *rows=s->row_batch;
    return count;
}

void
db4ai_shuffle_free(Db4aiShuffle *s)
{
    MemoryContextDelete(s->context);
}
