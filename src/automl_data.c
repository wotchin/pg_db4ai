/* SPDX-License-Identifier: MulanPSL-2.0
 * One authorized SELECT, deterministic group-safe holdouts, train-only fitting.
 * The row/input caps are admission checks, never silent LIMIT sampling.
 */
#include "postgres.h"
#include "automl.h"
#include <math.h>
#include "catalog/pg_type_d.h"
#include "common/hashfn.h"
#include "executor/spi.h"
#include "miscadmin.h"
#include "nodes/parsenodes.h"
#include "parser/parser.h"
#include "utils/builtins.h"
#include "utils/fmgrprotos.h"
#include "utils/lsyscache.h"
#include "utils/plancache.h"
#include "utils/date.h"

typedef struct RowKey { uint64 hash; int64 time; int index; } RowKey;

static bool
am_numeric_type(Oid type)
{
    return type==FLOAT8OID || type==FLOAT4OID || type==INT8OID || type==INT4OID ||
           type==INT2OID || type==NUMERICOID || type==BOOLOID;
}

static double
value_double(Datum v, Oid type)
{
    double x;
    switch(type)
    {
        case FLOAT8OID: x=DatumGetFloat8(v); break;
        case FLOAT4OID: x=DatumGetFloat4(v); break;
        case INT8OID: x=(double)DatumGetInt64(v); break;
        case INT4OID: x=DatumGetInt32(v); break;
        case INT2OID: x=DatumGetInt16(v); break;
        case BOOLOID: x=DatumGetBool(v)?1:0; break;
        case NUMERICOID: x=DatumGetFloat8(DirectFunctionCall1(numeric_float8,v)); break;
        default: ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("AutoML regression target must be numeric"))); x=0;
    }
    db4ai_check_finite(x); return x;
}

static void
moment(double x,int64 *count,double *mean,double *m2)
{
    double delta=x-*mean;
    (*count)++; *mean+=delta/(*count); *m2+=delta*(x-*mean);
    db4ai_check_finite(*mean); db4ai_check_finite(*m2);
}

static int
compare_hash(const void *a,const void *b)
{
    const RowKey *x=a,*y=b;
    if(x->hash!=y->hash) return x->hash<y->hash?-1:1;
    return x->index<y->index?-1:(x->index>y->index);
}

static int
compare_time(const void *a,const void *b)
{
    const RowKey *x=a,*y=b;
    if(x->time!=y->time) return x->time<y->time?-1:1;
    return x->index<y->index?-1:(x->index>y->index);
}

static int
find_column(TupleDesc desc,const char *name)
{
    int i;
    if(!name || !*name) return -1;
    for(i=1;i<desc->natts;i++)
        if(strcmp(NameStr(TupleDescAttr(desc,i)->attname),name)==0) return i;
    ereport(ERROR,(errcode(ERRCODE_UNDEFINED_COLUMN),errmsg("AutoML split column %s is not in the input SELECT",name)));
    return -1;
}

static bool
same_key(const RowKey *a,const RowKey *b,bool temporal)
{
    return temporal?a->time==b->time:a->hash==b->hash;
}

Db4aiAutoInput *
db4ai_automl_input(const char *query,bool classification,const Db4aiAutoOptions *o)
{
    MemoryContext context=CurrentMemoryContext,rowcontext;
    Db4aiAutoInput *in=palloc0(sizeof(*in));
    Db4aiPipeline *p=palloc0(sizeof(*p));
    SPIPlanPtr plan; Portal portal;
    List *raw; ListCell *lc;
    RowKey *keys=NULL;
    int capacity=0,groupcol,timecol,i,j,k,groups,rank;
    FmgrInfo *hashes;
    bool temporal=strcmp(o->split,"time")==0;
    TupleTableSlot *slot;
    Oid labeltype;
    double *encoded;
    int64 train_classes[256]={0},valid_classes[256]={0},test_classes[256]={0};
    Size store_budget=(Size)o->memory_mb*1024*1024/8;
    in->pipeline=p; p->classification=classification; p->target_scale=1;
    p->missing_indicators=o->missing_indicators;
    raw=raw_parser(query,RAW_PARSE_DEFAULT);
    if(list_length(raw)!=1 || !IsA(((RawStmt *)linitial(raw))->stmt,SelectStmt) ||
       ((SelectStmt *)((RawStmt *)linitial(raw))->stmt)->intoClause)
        ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("AutoML input must be one SELECT without INTO")));
    if(SPI_connect()!=SPI_OK_CONNECT) elog(ERROR,"SPI_connect failed");
    plan=SPI_prepare_cursor(query,0,NULL,CURSOR_OPT_NO_SCROLL);
    if(!plan || !SPI_is_cursor_plan(plan)) ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("AutoML input is not a SELECT cursor")));
    foreach(lc,SPI_plan_get_plan_sources(plan))
    {
        CachedPlanSource *s=lfirst(lc); ListCell *qc;
        foreach(qc,s->query_list)
        {
            Query *q=lfirst(qc);
            if(q->commandType!=CMD_SELECT || q->hasModifyingCTE || q->rowMarks!=NIL)
                ereport(ERROR,(errcode(ERRCODE_READ_ONLY_SQL_TRANSACTION),errmsg("AutoML source must be read-only without row locks or modifying CTEs")));
        }
    }
    portal=SPI_cursor_open(NULL,plan,NULL,NULL,true);
    MemoryContextSwitchTo(context);
    in->desc=CreateTupleDescCopy(portal->tupDesc);
    if(in->desc->natts<2 || in->desc->natts>DB4AI_AM_MAX_COLUMNS+2)
        ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("AutoML accepts one target and at most 128 scalar feature columns")));
    /* Ambiguous names make named prediction unsafe. */
    for(i=0;i<in->desc->natts;i++) for(j=0;j<i;j++)
        if(strcmp(NameStr(TupleDescAttr(in->desc,i)->attname),NameStr(TupleDescAttr(in->desc,j)->attname))==0)
            ereport(ERROR,(errcode(ERRCODE_DUPLICATE_COLUMN),errmsg("AutoML input column names must be unique; use SQL aliases")));
    groupcol=find_column(in->desc,o->group_column);
    timecol=find_column(in->desc,o->time_column);
    p->features=palloc0(sizeof(Db4aiFeature)*DB4AI_AM_MAX_COLUMNS);
    labeltype=getBaseType(TupleDescAttr(in->desc,0)->atttypid);
    if(!classification && !am_numeric_type(labeltype))
        ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("regression target must be numeric")));
    for(i=1;i<in->desc->natts;i++)
    {
        Form_pg_attribute a=TupleDescAttr(in->desc,i);
        Oid t=getBaseType(a->atttypid);
        Db4aiFeature *f;
        if(i==groupcol || i==timecol) continue;
        if(p->nraw>=DB4AI_AM_MAX_COLUMNS) ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("too many raw features")));
        f=&p->features[p->nraw++]; f->name=pstrdup(NameStr(a->attname)); f->source_column=i; f->scale=1;
        f->categorical=!am_numeric_type(t);
        if(f->categorical && t!=TEXTOID && t!=VARCHAROID && t!=BPCHAROID && get_typtype(t)!=TYPTYPE_ENUM)
            ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("AutoML feature %s must be numeric, boolean, text or enum",f->name),
                           errhint("Project timestamp components and expand arrays explicitly in SELECT; split time columns are excluded from features.")));
        if(f->categorical)
        {
            f->dictionary=db4ai_dictionary_new(a->atttypid,a->atttypmod,a->attcollation,o->max_categories);
            getTypeInputInfo(t,&f->input_function,&f->input_parameter);
        }
    }
    if(p->nraw<1) ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("AutoML needs at least one feature besides the split key")));
    hashes=palloc0(sizeof(FmgrInfo)*in->desc->natts);
    for(i=1;i<in->desc->natts;i++)
    {
        TypeCacheEntry *tc=lookup_type_cache(TupleDescAttr(in->desc,i)->atttypid,TYPECACHE_HASH_PROC_FINFO);
        if(!OidIsValid(tc->hash_proc_finfo.fn_oid))
            ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("input feature/split type lacks a stable equality hash")));
        fmgr_info_copy(&hashes[i],&tc->hash_proc_finfo,context);
    }
    in->raw=tuplestore_begin_heap(true,false,(int)(store_budget/1024));
    for(;;)
    {
        uint64 r;
        SPI_cursor_fetch(portal,true,256); MemoryContextSwitchTo(context);
        if(!SPI_processed) { if(SPI_tuptable) SPI_freetuptable(SPI_tuptable); break; }
        for(r=0;r<SPI_processed;r++)
        {
            HeapTuple tuple=SPI_tuptable->vals[r];
            uint64 h=(uint64)o->seed; bool isnull; Datum value;
            db4ai_automl_check();
            if(in->nrows>=o->max_rows || in->nrows>=db4ai_max_rows)
                ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("AutoML source exceeds max_rows; no prefix sample was trained"),
                               errhint("Use a representative bounded query, or explicitly increase the admitted working set.")));
            in->input_bytes+=tuple->t_len;
            if(in->input_bytes>(uint64)o->max_input_mb*1024*1024)
                ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("AutoML source exceeds max_input_mb")));
            value=SPI_getbinval(tuple,in->desc,1,&isnull);
            if(isnull) ereport(ERROR,(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),errmsg("AutoML target must not be NULL")));
            if(!classification || labeltype==FLOAT4OID || labeltype==FLOAT8OID) (void)value_double(value,labeltype);
            if(in->nrows==capacity)
            {
                int newcap=Min(o->max_rows,Max(256,capacity*2));
                if((Size)newcap*(sizeof(RowKey)+sizeof(uint8))>(Size)o->memory_mb*1024*1024/4)
                    ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("AutoML split inventory exceeds its memory allocation")));
                keys=capacity?repalloc(keys,sizeof(RowKey)*newcap):palloc(sizeof(RowKey)*newcap); capacity=newcap;
            }
            keys[in->nrows].index=in->nrows; keys[in->nrows].time=0;
            if(temporal)
            {
                Oid t=getBaseType(TupleDescAttr(in->desc,timecol)->atttypid);
                value=SPI_getbinval(tuple,in->desc,timecol+1,&isnull);
                if(isnull) ereport(ERROR,(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),errmsg("time split key must not be NULL")));
                if(t==TIMESTAMPOID || t==TIMESTAMPTZOID)
                {
                    keys[in->nrows].time=DatumGetTimestamp(value);
                    if(TIMESTAMP_NOT_FINITE(keys[in->nrows].time)) ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("time split key must be finite")));
                }
                else if(t==DATEOID)
                {
                    DateADT d=DatumGetDateADT(value);
                    if(DATE_NOT_FINITE(d)) ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("date split key must be finite")));
                    keys[in->nrows].time=(int64)d;
                }
                else ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("time_column must be date, timestamp or timestamptz")));
            }
            if(groupcol>=0)
            {
                value=SPI_getbinval(tuple,in->desc,groupcol+1,&isnull);
                if(isnull) ereport(ERROR,(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),errmsg("group split key must not be NULL")));
                h=hash_combine64(h,DatumGetUInt32(FunctionCall1Coll(&hashes[groupcol],TupleDescAttr(in->desc,groupcol)->attcollation,value)));
            }
            else
                for(j=0;j<p->nraw;j++)
                {
                    int col=p->features[j].source_column;
                    value=SPI_getbinval(tuple,in->desc,col+1,&isnull);
                    h=hash_combine64(h,isnull?UINT64CONST(0x9e3779b97f4a7c15):DatumGetUInt32(FunctionCall1Coll(&hashes[col],TupleDescAttr(in->desc,col)->attcollation,value)));
                }
            keys[in->nrows].hash=hash_bytes_extended((unsigned char*)&h,sizeof(h),(uint64)o->seed);
            tuplestore_puttuple(in->raw,tuple); in->nrows++;
        }
        SPI_freetuptable(SPI_tuptable);
    }
    SPI_cursor_close(portal); SPI_freeplan(plan); SPI_finish(); MemoryContextSwitchTo(context);
    if(in->nrows<30) ereport(ERROR,(errcode(ERRCODE_NO_DATA_FOUND),errmsg("AutoML requires at least 30 labeled rows for train/validation/test separation")));
    in->spilled=!tuplestore_in_memory(in->raw);
    qsort(keys,in->nrows,sizeof(RowKey),temporal?compare_time:compare_hash);
    in->partition=palloc0(in->nrows);
    groups=1;
    for(i=1;i<in->nrows;i++) if(!same_key(&keys[i-1],&keys[i],temporal)) groups++;
    if(groups<10) ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("AutoML needs at least 10 distinct split groups/time values"),
                                errhint("Identical feature rows stay together in the default split. Use an explicit appropriate group or time key.")));
    rank=0;
    for(i=0;i<in->nrows;i++)
    {
        double fraction;
        int part;
        if(i && !same_key(&keys[i-1],&keys[i],temporal)) rank++;
        fraction=(rank+0.5)/groups;
        part=fraction<1-o->validation_fraction-o->test_fraction?0:(fraction<1-o->test_fraction?1:2);
        in->partition[keys[i].index]=(uint8)part;
        if(part==0) in->ntrain++; else if(part==1) in->nvalid++; else in->ntest++;
    }
    if(in->ntrain<10 || in->nvalid<5 || in->ntest<5)
        ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("split groups do not provide enough rows in all three partitions")));
    in->split_hash=hash_bytes_extended(in->partition,in->nrows,(uint64)o->seed);
    pfree(keys); pfree(hashes);
    if(classification)
    {
        Form_pg_attribute a=TupleDescAttr(in->desc,0);
        p->labels=db4ai_dictionary_new(a->atttypid,a->atttypmod,a->attcollation,256);
    }
    rowcontext=AllocSetContextCreate(context,"DB4AI AutoML row",ALLOCSET_SMALL_SIZES);
    slot=MakeSingleTupleTableSlot(in->desc,&TTSOpsMinimalTuple);
    tuplestore_rescan(in->raw); i=0;
    while(tuplestore_gettupleslot(in->raw,true,false,slot))
    {
        db4ai_automl_check();
        if(in->partition[i++]==0)
        {
            slot_getallattrs(slot); MemoryContextSwitchTo(rowcontext);
            if(classification) db4ai_dictionary_index(p->labels,slot->tts_values[0],false,true);
            else moment(value_double(slot->tts_values[0],labeltype),&p->target_count,&p->target_mean,&p->target_m2);
            for(j=0;j<p->nraw;j++)
            {
                Db4aiFeature *f=&p->features[j]; int c=f->source_column;
                if(slot->tts_isnull[c]) continue;
                if(f->categorical)
                {
                    int id=db4ai_dictionary_index(f->dictionary,slot->tts_values[c],false,false);
                    if(id>=0 || f->dictionary->count<o->max_categories)
                        db4ai_dictionary_index(f->dictionary,slot->tts_values[c],false,true);
                    else f->capped=true;
                }
                else moment(value_double(slot->tts_values[c],getBaseType(TupleDescAttr(in->desc,c)->atttypid)),&f->observed,&f->mean,&f->m2);
            }
            MemoryContextSwitchTo(context); MemoryContextReset(rowcontext);
        }
        ExecClearTuple(slot);
    }
    p->nclasses=classification?p->labels->count:0;
    if(classification && p->nclasses<2) ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("training partition must contain at least two classes")));
    if(!classification)
    {
        p->target_scale=sqrt(Max(0,p->target_m2/p->target_count));
        if(p->target_scale==0) p->target_scale=1;
    }
    for(j=0;j<p->nraw;j++)
    {
        Db4aiFeature *f=&p->features[j];
        f->offset=p->nencoded;
        if(f->categorical) f->width=f->dictionary->count+2;
        else
        {
            f->width=p->missing_indicators?2:1;
            f->scale=f->observed?sqrt(Max(0,f->m2/f->observed)):1;
            if(f->scale==0) f->scale=1;
        }
        p->nencoded+=f->width;
    }
    if(p->nencoded>DB4AI_MAX_FEATURES)
        ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("encoded AutoML features exceed 1024; lower max_categories or project fewer columns")));
    if((uint64)in->nrows*(p->nencoded+1)*sizeof(double)>(uint64)o->max_encoded_mb*1024*1024)
        ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("encoded AutoML working set exceeds max_encoded_mb")));
    for(k=0;k<3;k++)
    {
        Db4aiData *d=k==0?&in->train:(k==1?&in->valid:&in->test);
        d->nfeatures=p->nencoded; d->ncolumns=p->nencoded+1; d->labels=p->labels;
        d->dense=db4ai_rows_create(d->ncolumns,store_budget);
        d->feature_names=palloc(sizeof(char*)*p->nencoded);
        for(j=0;j<p->nencoded;j++) d->feature_names[j]=psprintf("encoded_%d",j);
    }
    encoded=palloc(sizeof(double)*(p->nencoded+1));
    tuplestore_rescan(in->raw); i=0;
    while(tuplestore_gettupleslot(in->raw,true,false,slot))
    {
        Datum vals[DB4AI_AM_MAX_COLUMNS]; bool nulls[DB4AI_AM_MAX_COLUMNS]; Oid types[DB4AI_AM_MAX_COLUMNS];
        int part=in->partition[i++]; Db4aiData *d=part==0?&in->train:(part==1?&in->valid:&in->test);
        db4ai_automl_check(); slot_getallattrs(slot); MemoryContextSwitchTo(rowcontext);
        if(classification)
        {
            int label=db4ai_dictionary_index(p->labels,slot->tts_values[0],false,false);
            if(label<0) ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("holdout contains a class absent from training"),
                                     errhint("Use more examples or an appropriate split key; AutoML never learns labels from the holdout.")));
            encoded[0]=label;
            if(part==0) train_classes[label]++; else if(part==1) valid_classes[label]++; else test_classes[label]++;
        }
        else encoded[0]=(value_double(slot->tts_values[0],labeltype)-p->target_mean)/p->target_scale;
        for(j=0;j<p->nraw;j++)
        {
            int c=p->features[j].source_column;
            vals[j]=slot->tts_values[c]; nulls[j]=slot->tts_isnull[c]; types[j]=getBaseType(TupleDescAttr(in->desc,c)->atttypid);
        }
        db4ai_pipeline_transform(p,vals,nulls,types,encoded+1);
        db4ai_check_finite(encoded[0]);
        db4ai_rows_append(d->dense,encoded); d->nrows++;
        MemoryContextSwitchTo(context); ExecClearTuple(slot); MemoryContextReset(rowcontext);
    }
    if(classification)
        for(j=0;j<p->nclasses;j++)
            if(!valid_classes[j] || !test_classes[j])
                ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("each class must occur in validation and test for this AutoML protocol"),
                               errhint("Use more labeled groups, another predefined seed, or an appropriate time/group split; do not pick a seed by test score.")));
    for(k=0;k<3;k++)
    {
        Db4aiData *d=k==0?&in->train:(k==1?&in->valid:&in->test);
        db4ai_rows_seal(d->dense); d->spilled=db4ai_rows_spilled(d->dense);
    }
    pfree(encoded); ExecDropSingleTupleTableSlot(slot); MemoryContextDelete(rowcontext);
    tuplestore_end(in->raw); in->raw=NULL;
    return in;
}
