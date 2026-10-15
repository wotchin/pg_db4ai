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

Db4aiData *
db4ai_data_open(const char *query, Db4aiAlgorithm algorithm)
{
    MemoryContext context = CurrentMemoryContext;
    Db4aiData *data = palloc0(sizeof(*data));
    SPIPlanPtr plan;
    Portal portal;
    List *raw;
    ListCell *lc;
    Datum *values;
    bool *nulls;
    int first_feature = db4ai_supervised(algorithm) ? 1 : 0;
    int i;

    raw = raw_parser(query, RAW_PARSE_DEFAULT);
    if (list_length(raw) != 1 || !IsA(((RawStmt *) linitial(raw))->stmt, SelectStmt) ||
        ((SelectStmt *) ((RawStmt *) linitial(raw))->stmt)->intoClause != NULL)
        ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                        errmsg("training_sql must contain exactly one SELECT without INTO")));
    if (SPI_connect() != SPI_OK_CONNECT)
        elog(ERROR, "SPI_connect failed");
    plan = SPI_prepare_cursor(query, 0, NULL, CURSOR_OPT_NO_SCROLL);
    if (plan == NULL || !SPI_is_cursor_plan(plan))
        ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("training query is not a SELECT cursor")));
    foreach(lc, SPI_plan_get_plan_sources(plan))
    {
        CachedPlanSource *source = (CachedPlanSource *) lfirst(lc);
        ListCell *qc;
        foreach(qc, source->query_list)
        {
            Query *q = (Query *) lfirst(qc);
            if (q->commandType != CMD_SELECT || q->hasModifyingCTE || q->rowMarks != NIL)
                ereport(ERROR, (errcode(ERRCODE_READ_ONLY_SQL_TRANSACTION),
                                errmsg("training SQL must be read-only; modifying CTEs and row locks are forbidden")));
        }
    }
    portal = SPI_cursor_open(NULL, plan, NULL, NULL, true);
    MemoryContextSwitchTo(context);
    data->ncolumns = portal->tupDesc->natts;
    data->nfeatures = data->ncolumns - first_feature;
    if (data->nfeatures < 1 || data->nfeatures > DB4AI_MAX_FEATURES)
        ereport(ERROR, (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
                        errmsg("expected 1 to %d scalar features", DB4AI_MAX_FEATURES)));
    data->desc = CreateTemplateTupleDesc(data->ncolumns);
    data->feature_names = palloc(sizeof(char *) * data->nfeatures);
    for (i = 0; i < data->ncolumns; i++)
    {
        Form_pg_attribute attr = TupleDescAttr(portal->tupDesc, i);
        if (!numeric_type(attr->atttypid))
            ereport(ERROR, (errcode(ERRCODE_DATATYPE_MISMATCH),
                            errmsg("training column %s must be numeric or boolean", NameStr(attr->attname)),
                            errhint("Cast domains and expand feature arrays in the training SELECT.")));
        TupleDescInitEntry(data->desc, i + 1, NameStr(attr->attname), FLOAT8OID, -1, 0);
        if (i >= first_feature)
            data->feature_names[i - first_feature] = pstrdup(NameStr(attr->attname));
    }
    data->store = tuplestore_begin_heap(true, false, db4ai_work_mem);
    data->slot = MakeSingleTupleTableSlot(data->desc, &TTSOpsMinimalTuple);
    values = palloc(sizeof(Datum) * data->ncolumns);
    nulls = palloc0(sizeof(bool) * data->ncolumns);
    for (;;)
    {
        uint64 r;
        SPI_cursor_fetch(portal, true, 1024);
        MemoryContextSwitchTo(context);
        if (SPI_processed == 0)
        {
            if (SPI_tuptable != NULL)
                SPI_freetuptable(SPI_tuptable);
            break;
        }
        for (r = 0; r < SPI_processed; r++)
        {
            CHECK_FOR_INTERRUPTS();
            if (data->nrows >= db4ai_max_rows)
                ereport(ERROR, (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED), errmsg("training input exceeds db4ai.max_rows")));
            for (i = 0; i < data->ncolumns; i++)
            {
                bool isnull;
                Datum v = SPI_getbinval(SPI_tuptable->vals[r], SPI_tuptable->tupdesc, i + 1, &isnull);
                if (isnull)
                    ereport(ERROR, (errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
                                    errmsg("NULL training value at row " INT64_FORMAT ", column %d", data->nrows + 1, i + 1),
                                    errhint("Apply an explicit filtering or imputation policy in training_sql.")));
                values[i] = Float8GetDatum(as_double(v, TupleDescAttr(portal->tupDesc, i)->atttypid));
            }
            tuplestore_putvalues(data->store, data->desc, values, nulls);
            data->nrows++;
        }
        SPI_freetuptable(SPI_tuptable);
    }
    SPI_cursor_close(portal);
    SPI_freeplan(plan);
    SPI_finish();
    MemoryContextSwitchTo(context);
    pfree(values);
    pfree(nulls);
    if (data->nrows == 0)
        ereport(ERROR, (errcode(ERRCODE_NO_DATA_FOUND), errmsg("training query returned no rows")));
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
