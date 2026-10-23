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
    double *values;
    int *width;
    int source_cols, first_feature=db4ai_supervised(algorithm)?1:0;
    int i, nf=0;
    int64 seen=0;
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
    if(source_cols<first_feature+1 || source_cols>DB4AI_MAX_FEATURES+first_feature)
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
            for(i=first_feature;i<source_cols;i++)
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
        data->labels=db4ai_dictionary_new(label->atttypid,label->atttypmod,label->attcollation,(algorithm==DB4AI_MULTICLASS || algorithm==DB4AI_GAUSSIAN_NB)?256:2);
    }
    else if(first_feature && !numeric_type(getBaseType(TupleDescAttr(desc,0)->atttypid)))
        ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("regression label must be numeric")));
    data->feature_names=palloc0(sizeof(char *)*nf);
    nf=0;
    for(i=first_feature;i<source_cols;i++)
    {
        int j;
        for(j=0;j<width[i];j++)
            data->feature_names[nf++]=width[i]==1?pstrdup(NameStr(TupleDescAttr(desc,i)->attname)):
                psprintf("%s[%d]",NameStr(TupleDescAttr(desc,i)->attname),j+1);
    }
    data->dense=db4ai_rows_create(data->ncolumns,(Size)db4ai_work_mem*1024);
    slot=MakeSingleTupleTableSlot(desc,&TTSOpsMinimalTuple);
    values=palloc0(sizeof(double)*data->ncolumns);
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
        if(options->null_policy)
            for(i=first_feature;i<source_cols;i++)
            {
                Oid t=getBaseType(TupleDescAttr(desc,i)->atttypid);
                if(slot->tts_isnull[i] && (t==FLOAT8ARRAYOID || t==FLOAT4ARRAYOID || db4ai_is_xgboost(algorithm))) discard=true;
            }
        for(i=0;i<source_cols && !discard;i++)
        {
            Datum v=slot->tts_values[i]; bool isnull=slot->tts_isnull[i];
            Oid type=getBaseType(TupleDescAttr(desc,i)->atttypid);
            if(categorical && i==0)
            {
                if(isnull) ereport(ERROR,(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),errmsg("NULL training label")));
                if(type==FLOAT4OID || type==FLOAT8OID) (void)as_double(v,type);
                if(type==NUMERICOID && (numeric_is_nan(DatumGetNumeric(v)) || numeric_is_inf(DatumGetNumeric(v))))
                    ereport(ERROR,(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),errmsg("non-finite classification label")));
                values[column++]=(double)db4ai_dictionary_index(data->labels,v,false,true);
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
                    values[column++]=as_double(items[j],elem);
                }
            }
            else
            {
                if(isnull && !options->null_policy) ereport(ERROR,(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),errmsg("NULL training value in column %d",i+1)));
                values[column++]=isnull?0.0:as_double(v,type);
            }
        }
        MemoryContextSwitchTo(context);
        if(discard) data->discarded++;
        else { db4ai_rows_append(data->dense,values); data->nrows++; }
        ExecClearTuple(slot); MemoryContextReset(rowcontext);
    }
    db4ai_rows_seal(data->dense);
    data->spilled=db4ai_rows_spilled(data->dense);
    ExecDropSingleTupleTableSlot(slot); tuplestore_end(input); FreeTupleDesc(desc); MemoryContextDelete(rowcontext);
    pfree(values); pfree(width);
    if(!data->nrows) ereport(ERROR,(errcode(ERRCODE_NO_DATA_FOUND),errmsg("training input has no usable rows")));
    if(categorical && data->labels->count<2) ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("classification requires at least two distinct labels")));
    if(categorical && algorithm!=DB4AI_MULTICLASS && one_then_zero(data->labels))
    {
        Db4aiDictionary *old=data->labels;
        Db4aiRowStore *ordered=db4ai_rows_create(data->ncolumns,(Size)db4ai_work_mem*1024);
        double *row=palloc(data->ncolumns*sizeof(double));
        data->labels=db4ai_dictionary_new(old->type,old->typmod,old->collation,old->limit);
        db4ai_dictionary_index(data->labels,old->values[1],false,true);
        db4ai_dictionary_index(data->labels,old->values[0],false,true);
        data->labels->frequency[0]=old->frequency[1]; data->labels->frequency[1]=old->frequency[0];
        data->position=0;
        while(db4ai_data_next(data,row))
        {
            CHECK_FOR_INTERRUPTS();
            row[0]=1.0-row[0];
            db4ai_rows_append(ordered,row);
        }
        db4ai_rows_seal(ordered);
        db4ai_rows_free(data->dense); data->dense=ordered;
        data->spilled=data->spilled || db4ai_rows_spilled(ordered);
        pfree(row);
    }
    data->position=0;
    return data;
}

bool
db4ai_data_next(Db4aiData *data, double *row)
{
    const double *source;
    if (!db4ai_rows_slice(data->dense,data->position,1,&source))
        return false;
    memcpy(row,source,(Size)data->ncolumns*sizeof(double));
    data->position++;
    return true;
}

void
db4ai_data_rewind(Db4aiData *data)
{
    data->position=0;
    data->passes++;
}

void
db4ai_data_close(Db4aiData *data)
{
    db4ai_rows_free(data->dense);
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
