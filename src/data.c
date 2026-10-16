/* SPDX-License-Identifier: MulanPSL-2.0 */
#include "postgres.h"
#include "db4ai.h"
#include "catalog/pg_type_d.h"
#include "executor/spi.h"
#include "miscadmin.h"
#include "nodes/parsenodes.h"
#include "parser/parser.h"
#include "utils/builtins.h"
#include "utils/fmgrprotos.h"
#include "utils/plancache.h"
#include "utils/lsyscache.h"

static bool
numeric_type(Oid type)
{
    return type == FLOAT8OID || type == FLOAT4OID || type == INT8OID ||
           type == INT4OID || type == INT2OID || type == NUMERICOID || type == BOOLOID;
}

static double
as_double(Datum value, Oid type)
{
    double out;
    switch (type)
    {
        case FLOAT8OID: out = DatumGetFloat8(value); break;
        case FLOAT4OID: out = DatumGetFloat4(value); break;
        case INT8OID: out = (double) DatumGetInt64(value); break;
        case INT4OID: out = DatumGetInt32(value); break;
        case INT2OID: out = DatumGetInt16(value); break;
        case BOOLOID: out = DatumGetBool(value) ? 1.0 : 0.0; break;
        case NUMERICOID: out = DatumGetFloat8(DirectFunctionCall1(numeric_float8, value)); break;
        default: elog(ERROR, "unsupported DB4AI input type"); out = 0;
    }
    db4ai_check_finite(out);
    return out;
}

/* Maintain the 0.1 probability-of-1 API for numeric 0/1 and boolean labels,
 * regardless of which category appears first. Other dictionaries retain their
 * first-occurrence ordering. Never coerce category identity through float8. */
static bool
one_then_zero(Db4aiDictionary *d)
{
    Datum zero,one;
    if(d->count!=2) return false;
    switch(d->type)
    {
        case BOOLOID: zero=BoolGetDatum(false); one=BoolGetDatum(true); break;
        case INT2OID: zero=Int16GetDatum(0); one=Int16GetDatum(1); break;
        case INT4OID: zero=Int32GetDatum(0); one=Int32GetDatum(1); break;
        case INT8OID: zero=Int64GetDatum(0); one=Int64GetDatum(1); break;
        case FLOAT4OID: zero=Float4GetDatum(0); one=Float4GetDatum(1); break;
        case FLOAT8OID: zero=Float8GetDatum(0); one=Float8GetDatum(1); break;
        case NUMERICOID: zero=DirectFunctionCall1(int4_numeric,Int32GetDatum(0)); one=DirectFunctionCall1(int4_numeric,Int32GetDatum(1)); break;
        default: return false;
    }
    return DatumGetBool(FunctionCall2Coll(&d->equal,d->collation,d->values[0],one)) &&
           DatumGetBool(FunctionCall2Coll(&d->equal,d->collation,d->values[1],zero));
}

/* Materialize source rows once before normalization: this also lets array
 * shapes be inferred from later non-NULL rows without rerunning volatile SQL.
 * Both stores independently obey db4ai.work_mem and may spill to disk.
 */
Db4aiData *
db4ai_data_open(const char *query, Db4aiAlgorithm algorithm, const Db4aiOptions *options)
{
    MemoryContext context=CurrentMemoryContext;
    MemoryContext rowcontext;
    Db4aiData *data=palloc0(sizeof(*data));
    SPIPlanPtr plan; Portal portal; List *raw; ListCell *lc;
    Tuplestorestate *input;
    TupleTableSlot *slot;
    TupleDesc desc;
    Datum *values; bool *nulls;
    int *width;
    int source_cols, first_feature=db4ai_supervised(algorithm)?1:0;
    int i, nf=0;
    int64 seen=0;
    bool bayes=algorithm==DB4AI_BAYES;
    bool categorical=db4ai_classifier(algorithm);
    raw=raw_parser(query,RAW_PARSE_DEFAULT);
    if(list_length(raw)!=1 || !IsA(((RawStmt *)linitial(raw))->stmt,SelectStmt) ||
       ((SelectStmt *)((RawStmt *)linitial(raw))->stmt)->intoClause!=NULL)
        ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("training_sql must contain exactly one SELECT without INTO")));
    if(SPI_connect()!=SPI_OK_CONNECT) elog(ERROR,"SPI_connect failed");
    plan=SPI_prepare_cursor(query,0,NULL,CURSOR_OPT_NO_SCROLL);
    if(plan==NULL || !SPI_is_cursor_plan(plan)) ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("training query is not a SELECT cursor")));
    foreach(lc,SPI_plan_get_plan_sources(plan))
    {
        CachedPlanSource *source=(CachedPlanSource *)lfirst(lc); ListCell *qc;
        foreach(qc,source->query_list)
        {
            Query *q=(Query *)lfirst(qc);
            if(q->commandType!=CMD_SELECT || q->hasModifyingCTE || q->rowMarks!=NIL)
                ereport(ERROR,(errcode(ERRCODE_READ_ONLY_SQL_TRANSACTION),errmsg("training SQL must be read-only; modifying CTEs and row locks are forbidden")));
        }
    }
    portal=SPI_cursor_open(NULL,plan,NULL,NULL,true);
    MemoryContextSwitchTo(context);
    desc=CreateTupleDescCopy(portal->tupDesc); source_cols=desc->natts;
    if(source_cols<first_feature+1 || source_cols>DB4AI_MAX_FEATURES+first_feature || (bayes && source_cols>64))
        ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("invalid number of training columns")));
    width=palloc0(sizeof(int)*source_cols);
    input=tuplestore_begin_heap(true,false,db4ai_work_mem);
    for(;;)
    {
        uint64 r;
        SPI_cursor_fetch(portal,true,Min(options->batch_size,1024));
        MemoryContextSwitchTo(context);
        if(!SPI_processed) { if(SPI_tuptable) SPI_freetuptable(SPI_tuptable); break; }
        for(r=0;r<SPI_processed;r++)
        {
            CHECK_FOR_INTERRUPTS();
            if(seen++>=db4ai_max_rows) ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("training input exceeds db4ai.max_rows")));
            tuplestore_puttuple(input,SPI_tuptable->vals[r]);
            for(i=first_feature;i<source_cols && !bayes;i++)
            {
                Oid type=getBaseType(TupleDescAttr(desc,i)->atttypid);
                if(type==FLOAT8ARRAYOID || type==FLOAT4ARRAYOID)
                {
                    bool isnull; Datum d=SPI_getbinval(SPI_tuptable->vals[r],desc,i+1,&isnull);
                    if(!isnull)
                    {
                        ArrayType *array=DatumGetArrayTypeP(d);
                        if(ARR_NDIM(array)!=1 || ARR_DIMS(array)[0]<1 || ARR_DIMS(array)[0]>DB4AI_MAX_FEATURES ||
                           (width[i]!=0 && width[i]!=ARR_DIMS(array)[0]))
                            ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("feature array dimensions must be positive and consistent")));
                        width[i]=ARR_DIMS(array)[0];
                        if((Pointer)array!=DatumGetPointer(d)) pfree(array);
                    }
                }
                else
                {
                    if(!numeric_type(type)) ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("training feature %s must be numeric, boolean, float4[] or float8[]",NameStr(TupleDescAttr(desc,i)->attname))));
                    width[i]=1;
                }
            }
        }
        SPI_freetuptable(SPI_tuptable);
    }
    SPI_cursor_close(portal); SPI_freeplan(plan); SPI_finish(); MemoryContextSwitchTo(context);
    if(!seen) ereport(ERROR,(errcode(ERRCODE_NO_DATA_FOUND),errmsg("training query returned no rows")));
    if(bayes)
    {
        data->domains=palloc0(sizeof(Db4aiDictionary *)*source_cols);
        for(i=0;i<source_cols;i++)
        {
            Form_pg_attribute attr=TupleDescAttr(desc,i);
            data->domains[i]=db4ai_dictionary_new(attr->atttypid,attr->atttypmod,attr->attcollation,65536);
            width[i]=1;
        }
    }
    for(i=first_feature;i<source_cols;i++)
    {
        if(width[i]==0) ereport(ERROR,(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),errmsg("cannot infer the shape of an entirely NULL feature array")));
        nf+=width[i];
    }
    if(nf>DB4AI_MAX_FEATURES || (options->num_features>0 && options->num_features!=nf))
        ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("training feature count does not match limits/num_features")));
    data->nfeatures=nf; data->ncolumns=nf+first_feature;
    if(categorical)
    {
        Form_pg_attribute label=TupleDescAttr(desc,0);
        data->labels=db4ai_dictionary_new(label->atttypid,label->atttypmod,label->attcollation,algorithm==DB4AI_MULTICLASS?256:2);
    }
    else if(first_feature && !numeric_type(getBaseType(TupleDescAttr(desc,0)->atttypid)))
        ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("regression label must be numeric")));
    data->desc=CreateTemplateTupleDesc(data->ncolumns);
    data->feature_names=palloc0(sizeof(char *)*nf);
    for(i=0;i<data->ncolumns;i++) TupleDescInitEntry(data->desc,i+1,NULL,FLOAT8OID,-1,0);
    nf=0;
    for(i=first_feature;i<source_cols;i++)
    {
        int j;
        for(j=0;j<width[i];j++)
            data->feature_names[nf++]=width[i]==1?pstrdup(NameStr(TupleDescAttr(desc,i)->attname)):
                psprintf("%s[%d]",NameStr(TupleDescAttr(desc,i)->attname),j+1);
    }
    data->store=tuplestore_begin_heap(true,false,db4ai_work_mem);
    data->slot=MakeSingleTupleTableSlot(data->desc,&TTSOpsMinimalTuple);
    slot=MakeSingleTupleTableSlot(desc,&TTSOpsMinimalTuple);
    values=palloc0(sizeof(Datum)*data->ncolumns); nulls=palloc0(sizeof(bool)*data->ncolumns);
    rowcontext=AllocSetContextCreate(context,"DB4AI normalize row",ALLOCSET_SMALL_SIZES);
    tuplestore_rescan(input);
    while(tuplestore_gettupleslot(input,true,false,slot))
    {
        int column=0;
        bool discard=false;
        CHECK_FOR_INTERRUPTS(); slot_getallattrs(slot); MemoryContextSwitchTo(rowcontext);
        if(first_feature && slot->tts_isnull[0] && options->null_policy) discard=true;
        /* Reject/skip the complete row before inserting its classification label.
         * Otherwise a skipped NULL-array row could create a phantom category. */
        if(options->null_policy && !bayes)
            for(i=first_feature;i<source_cols;i++)
            {
                Oid t=getBaseType(TupleDescAttr(desc,i)->atttypid);
                if(slot->tts_isnull[i] && (t==FLOAT8ARRAYOID || t==FLOAT4ARRAYOID || db4ai_is_xgboost(algorithm))) discard=true;
            }
        for(i=0;i<source_cols && !discard;i++)
        {
            Datum v=slot->tts_values[i]; bool isnull=slot->tts_isnull[i];
            Oid type=getBaseType(TupleDescAttr(desc,i)->atttypid);
            if(bayes)
            {
                if(!isnull && (type==FLOAT4OID || type==FLOAT8OID)) (void)as_double(v,type);
                if(!isnull && type==NUMERICOID && (numeric_is_nan(DatumGetNumeric(v)) || numeric_is_inf(DatumGetNumeric(v))))
                    ereport(ERROR,(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),errmsg("non-finite Bayes numeric category")));
                values[column++]=Float8GetDatum((double)db4ai_dictionary_index(data->domains[i],v,isnull,true));
            }
            else if(categorical && i==0)
            {
                if(isnull) ereport(ERROR,(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),errmsg("NULL training label")));
                if(type==FLOAT4OID || type==FLOAT8OID) (void)as_double(v,type);
                if(type==NUMERICOID && (numeric_is_nan(DatumGetNumeric(v)) || numeric_is_inf(DatumGetNumeric(v))))
                    ereport(ERROR,(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),errmsg("non-finite classification label")));
                values[column++]=Float8GetDatum((double)db4ai_dictionary_index(data->labels,v,false,true));
            }
            else if(type==FLOAT8ARRAYOID || type==FLOAT4ARRAYOID)
            {
                ArrayType *array;
                Datum *items; bool *missing; int n,j; Oid elem=type==FLOAT8ARRAYOID?FLOAT8OID:FLOAT4OID;
                if(isnull)
                {
                    if(options->null_policy) { discard=true; break; }
                    ereport(ERROR,(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),errmsg("NULL feature array")));
                }
                array=DatumGetArrayTypeP(v);
                deconstruct_array(array,elem,elem==FLOAT8OID?8:4,elem==FLOAT8OID?FLOAT8PASSBYVAL:true,elem==FLOAT8OID?'d':'i',&items,&missing,&n);
                for(j=0;j<n;j++)
                {
                    if(missing[j]) ereport(ERROR,(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),errmsg("feature arrays must not contain NULL elements")));
                    values[column++]=Float8GetDatum(as_double(items[j],elem));
                }
            }
            else
            {
                if(isnull && !options->null_policy) ereport(ERROR,(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),errmsg("NULL training value in column %d",i+1)));
                values[column++]=Float8GetDatum(isnull?0.0:as_double(v,type));
            }
        }
        MemoryContextSwitchTo(context);
        if(bayes)
        {
            Size bytes=0;
            for(i=0;i<source_cols;i++) bytes+=data->domains[i]->bytes;
            if(bytes>(Size)db4ai_max_model_mb*1024*1024)
                ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("combined Bayes dictionaries exceed db4ai.max_model_mb")));
        }
        if(discard) data->discarded++;
        else { tuplestore_putvalues(data->store,data->desc,values,nulls); data->nrows++; }
        ExecClearTuple(slot); MemoryContextReset(rowcontext);
    }
    data->spilled=!tuplestore_in_memory(data->store);
    ExecDropSingleTupleTableSlot(slot); tuplestore_end(input); FreeTupleDesc(desc); MemoryContextDelete(rowcontext);
    pfree(values); pfree(nulls); pfree(width);
    if(!data->nrows) ereport(ERROR,(errcode(ERRCODE_NO_DATA_FOUND),errmsg("training input has no usable rows")));
    if(categorical && data->labels->count<2) ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("classification requires at least two distinct labels")));
    if(categorical && algorithm!=DB4AI_MULTICLASS && one_then_zero(data->labels))
    {
        Db4aiDictionary *old=data->labels;
        Tuplestorestate *ordered=tuplestore_begin_heap(true,false,db4ai_work_mem);
        data->labels=db4ai_dictionary_new(old->type,old->typmod,old->collation,old->limit);
        db4ai_dictionary_index(data->labels,old->values[1],false,true);
        db4ai_dictionary_index(data->labels,old->values[0],false,true);
        data->labels->frequency[0]=old->frequency[1]; data->labels->frequency[1]=old->frequency[0];
        tuplestore_rescan(data->store);
        while(tuplestore_gettupleslot(data->store,true,false,data->slot))
        {
            CHECK_FOR_INTERRUPTS(); slot_getallattrs(data->slot);
            data->slot->tts_values[0]=Float8GetDatum(1.0-DatumGetFloat8(data->slot->tts_values[0]));
            tuplestore_putvalues(ordered,data->desc,data->slot->tts_values,data->slot->tts_isnull);
            ExecClearTuple(data->slot);
        }
        tuplestore_end(data->store); data->store=ordered;
        data->spilled=data->spilled || !tuplestore_in_memory(ordered);
    }
    tuplestore_rescan(data->store);
    return data;
}

bool
db4ai_data_next(Db4aiData *data, double *row)
{
    int i;
    if (!tuplestore_gettupleslot(data->store, true, false, data->slot))
        return false;
    slot_getallattrs(data->slot);
    for (i = 0; i < data->ncolumns; i++)
        row[i] = DatumGetFloat8(data->slot->tts_values[i]);
    ExecClearTuple(data->slot);
    return true;
}

void
db4ai_data_rewind(Db4aiData *data)
{
    ExecClearTuple(data->slot);
    tuplestore_rescan(data->store);
    data->passes++;
}

void
db4ai_data_close(Db4aiData *data)
{
    ExecDropSingleTupleTableSlot(data->slot);
    tuplestore_end(data->store);
    FreeTupleDesc(data->desc);
}

double *
db4ai_features(ArrayType *array, int expected)
{
    Datum *values;
    bool *nulls;
    int count;
    int i;
    double *out;
    if (ARR_NDIM(array) != 1 || ARR_ELEMTYPE(array) != FLOAT8OID || ARR_DIMS(array)[0] != expected)
        ereport(ERROR, (errcode(ERRCODE_DATATYPE_MISMATCH),
                        errmsg("expected a one-dimensional float8 array of %d features", expected)));
    deconstruct_array(array, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, TYPALIGN_DOUBLE, &values, &nulls, &count);
    out = palloc(sizeof(double) * count);
    for (i = 0; i < count; i++)
    {
        if (nulls[i])
            ereport(ERROR, (errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED), errmsg("prediction features must not contain NULL")));
        out[i] = DatumGetFloat8(values[i]);
        db4ai_check_finite(out[i]);
    }
    pfree(values);
    pfree(nulls);
    return out;
}
