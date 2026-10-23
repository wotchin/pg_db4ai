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
    /* CorgiPile: random block order, then a uniform permutation of rows in
     * a bounded multi-block window. Index bytes are included in the budget. */
    int64 *blocks;
    int *row_order;
    double *window;
    int block_rows;
    int window_blocks;
    int nblocks;
    int next_block;
    int window_count;
    int window_cursor;

};

static int
read_unit(Db4aiShuffle *s, double *out)
{
    int n;
    db4ai_automl_check();
    n=db4ai_rows_read(s->data->dense,s->data->position,s->unit,out);
    s->data->position+=n;
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
    if (s->mode==DB4AI_SHUFFLE_CORGIPILE)
    {
        int64 blocks=(data->nrows+o->shuffle_block_rows-1)/o->shuffle_block_rows;
        int64 windowblocks=Min(blocks,(int64)o->shuffle_window_blocks);
        int64 windowrows=Min(data->nrows,windowblocks*o->shuffle_block_rows);
        Size indexbytes;
        Size outputbytes=(Size)o->batch_size*rowbytes;
        if (blocks>INT_MAX || windowrows>INT_MAX ||
            (uint64)blocks>MaxAllocSize/sizeof(int64) ||
            (uint64)windowrows>MaxAllocSize/rowbytes || outputbytes>MaxAllocSize)
            ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("CorgiPile buffer exceeds allocation limits")));
        indexbytes=(Size)blocks*sizeof(int64)+(Size)windowrows*sizeof(int);
        if (indexbytes+outputbytes+(Size)windowrows*rowbytes+4096>budget)
            ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
                errmsg("CorgiPile blocks, row indexes and output do not fit db4ai.work_mem"),
                errhint("Reduce shuffle_block_rows, shuffle_window_blocks or batch_size; or increase db4ai.work_mem.")));
        s->block_rows=o->shuffle_block_rows; s->nblocks=(int)blocks;
        s->window_blocks=(int)windowblocks; s->result_rows=o->batch_size;
        s->blocks=palloc((Size)blocks*sizeof(int64));
        s->row_order=palloc((Size)windowrows*sizeof(int));
        s->window=palloc((Size)windowrows*rowbytes);
        s->output=palloc(outputbytes);
        data->shuffle_blocks=blocks;
        data->shuffle_window_rows=(int)windowrows;
        data->shuffle_capacity=(int)windowrows;
        data->shuffle_full_cache=windowrows==data->nrows;
        MemoryContextSwitchTo(old);
        return s;
    }
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
    if (s->mode==DB4AI_SHUFFLE_CORGIPILE)
    {
        s->next_block=0; s->window_count=0; s->window_cursor=0;
        for(i=0;i<s->nblocks;i++) s->blocks[i]=i;
        if(!s->sequential) for(i=s->nblocks-1;i>0;i--)
        {
            int j=(int)pg_prng_uint64_range(&s->rng,0,i);
            int64 t=s->blocks[i]; s->blocks[i]=s->blocks[j]; s->blocks[j]=t;
        }
        s->data->passes++;
        return;
    }
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
    db4ai_automl_check();
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

static int
corgipile_next(Db4aiShuffle *s, const double **rows)
{
    int filled=0;
    Size width=s->data->ncolumns;
    while(filled<s->result_rows)
    {
        int index;
        if(s->window_cursor==s->window_count)
        {
            int k,i;
            if(s->next_block==s->nblocks) break;
            s->window_count=0; s->window_cursor=0;
            for(k=0;k<s->window_blocks && s->next_block<s->nblocks;k++)
            {
                int64 start=s->blocks[s->next_block++]*s->block_rows;
                int n;
                db4ai_automl_check();
                n=db4ai_rows_read(s->data->dense,start,s->block_rows,
                                 s->window+(Size)s->window_count*width);
                if(n<1) elog(ERROR,"CorgiPile encountered an empty block");
                s->window_count+=n;
            }
            for(i=0;i<s->window_count;i++) s->row_order[i]=i;
            if(!s->sequential) for(i=s->window_count-1;i>0;i--)
            {
                int j=(int)pg_prng_uint64_range(&s->rng,0,i);
                int t=s->row_order[i]; s->row_order[i]=s->row_order[j]; s->row_order[j]=t;
            }
        }
        index=s->row_order[s->window_cursor++];
        memcpy(s->output+(Size)filled*width,s->window+(Size)index*width,width*sizeof(double));
        filled++;
    }
    s->emitted+=filled;
    if(!filled && s->emitted!=s->data->nrows) elog(ERROR,"CorgiPile violated exact-once epoch accounting");
    *rows=s->output;
    return filled;
}

int
db4ai_shuffle_next(Db4aiShuffle *s, const double **rows)
{
    int count=0;
    const double *one;
    if(s->mode==DB4AI_SHUFFLE_CORGIPILE) return corgipile_next(s,rows);
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

/* Bounded diagnostic interface used to audit the actual training iterator.
 * This returns row indexes only; it neither reads tables nor trains a model. */
PG_FUNCTION_INFO_V1(pg_db4ai_shuffle_indices);
Datum
pg_db4ai_shuffle_indices(PG_FUNCTION_ARGS)
{
    MemoryContext parent=CurrentMemoryContext;
    MemoryContext context;
    ArrayType *result=NULL;
    int nrows=PG_GETARG_INT32(0);
    int epoch=PG_GETARG_INT32(2);
    if(nrows<0 || nrows>100000 || epoch<0)
        ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                       errmsg("shuffle diagnostic requires 0..100000 rows and a nonnegative epoch")));
    context=AllocSetContextCreate(parent,"DB4AI shuffle diagnostic",ALLOCSET_DEFAULT_SIZES);
    PG_TRY();
    {
        Db4aiData data;
        Db4aiOptions options;
        Db4aiShuffle *scan;
        const double *rows;
        Datum *values;
        int n,i,position=0;
        MemoryContextSwitchTo(context);
        memset(&data,0,sizeof(data));
        data.ncolumns=1; data.nfeatures=1; data.nrows=nrows;
        db4ai_options(PG_GETARG_JSONB_P(1),DB4AI_LINEAR,&options);
        data.dense=db4ai_rows_create(1,(Size)db4ai_work_mem*1024);
        for(i=0;i<nrows;i++) { double v=i; db4ai_rows_append(data.dense,&v); }
        db4ai_rows_seal(data.dense);
        scan=db4ai_shuffle_create(&data,&options);
        db4ai_shuffle_start(scan,epoch);
        values=palloc((Size)Max(nrows,1)*sizeof(Datum));
        while((n=db4ai_shuffle_next(scan,&rows))>0)
            for(i=0;i<n;i++)
            {
                if(position>=nrows) elog(ERROR,"shuffle emitted excess rows");
                values[position++]=Int64GetDatum((int64)rows[i]);
            }
        if(position!=nrows) elog(ERROR,"shuffle dropped rows");
        MemoryContextSwitchTo(parent);
        result=nrows?construct_array(values,nrows,INT8OID,8,FLOAT8PASSBYVAL,TYPALIGN_DOUBLE):construct_empty_array(INT8OID);
        MemoryContextDelete(context);
    }
    PG_CATCH();
    {
        MemoryContextSwitchTo(parent); MemoryContextDelete(context); PG_RE_THROW();
    }
    PG_END_TRY();
    PG_RETURN_ARRAYTYPE_P(result);
}
