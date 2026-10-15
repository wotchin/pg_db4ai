/* SPDX-License-Identifier: MulanPSL-2.0
 * Only fixed, parameterized registry writes use the extension owner's identity.
 * Training SQL is never executed here or under that identity.
 */
#include "postgres.h"
#include "db4ai.h"
#include "catalog/pg_extension.h"
#include "catalog/pg_type_d.h"
#include "commands/extension.h"
#include "executor/spi.h"
#include "miscadmin.h"
#include "utils/builtins.h"
#include "utils/syscache.h"

static Oid
registry_owner(void)
{
    Oid extension = get_extension_oid("pg_db4ai", false);
    HeapTuple tuple = SearchSysCache1(EXTENSIONOID, ObjectIdGetDatum(extension));
    Oid owner;
    if (!HeapTupleIsValid(tuple))
        elog(ERROR, "pg_db4ai extension metadata is missing");
    owner = ((Form_pg_extension) GETSTRUCT(tuple))->extowner;
    ReleaseSysCache(tuple);
    return owner;
}

int64
db4ai_model_store(const char *name, const char *query, const Db4aiData *data,
                  const Db4aiModel *model, Jsonb *options, Oid caller)
{
    Oid saved_user;
    int saved_security;
    volatile int64 id = 0;
    Oid types[12] = {REGROLEOID, TEXTOID, TEXTOID, INT4OID, INT4OID, TEXTOID,
                     BYTEAOID, JSONBOID, JSONBOID, TEXTOID, TEXTARRAYOID, INT8OID};
    Datum values[12];
    Datum *features = palloc(sizeof(Datum) * data->nfeatures);
    int i;
    for (i = 0; i < data->nfeatures; i++)
        features[i] = CStringGetTextDatum(data->feature_names[i]);
    values[0] = ObjectIdGetDatum(caller);
    values[1] = CStringGetTextDatum(name);
    values[2] = CStringGetTextDatum(db4ai_algorithm_name(model->algorithm));
    values[3] = Int32GetDatum(model->nfeatures);
    values[4] = Int32GetDatum(model->noutputs);
    values[5] = CStringGetTextDatum(model->format);
    values[6] = PointerGetDatum(model->data);
    values[7] = JsonbPGetDatum(options);
    values[8] = JsonbPGetDatum(model->metrics);
    values[9] = CStringGetTextDatum(query);
    values[10] = PointerGetDatum(construct_array(features, data->nfeatures, TEXTOID, -1, false, TYPALIGN_INT));
    values[11] = Int64GetDatum(data->nrows);

    GetUserIdAndSecContext(&saved_user, &saved_security);
    SetUserIdAndSecContext(registry_owner(), saved_security | SECURITY_LOCAL_USERID_CHANGE | SECURITY_RESTRICTED_OPERATION);
    PG_TRY();
    {
        bool isnull;
        if (SPI_connect() != SPI_OK_CONNECT)
            elog(ERROR, "SPI_connect failed");
        if (SPI_execute_with_args(
            "INSERT INTO db4ai.pg_db4ai_models "
            "(owner_oid, model_name, algorithm, n_features, n_outputs, model_format, model_data, "
            " hyperparameters, metrics, training_sql, feature_names, rows_processed) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12) RETURNING model_id",
            12, types, values, NULL, false, 1) != SPI_OK_INSERT_RETURNING || SPI_processed != 1)
            elog(ERROR, "model insertion failed");
        id = DatumGetInt64(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
        SPI_finish();
        SetUserIdAndSecContext(saved_user, saved_security);
    }
    PG_CATCH();
    {
        SetUserIdAndSecContext(saved_user, saved_security);
        PG_RE_THROW();
    }
    PG_END_TRY();
    return id;
}

Db4aiModel *
db4ai_model_load(const char *name, Oid caller)
{
    MemoryContext context = CurrentMemoryContext;
    Db4aiModel *model;
    Oid types[2] = {REGROLEOID, TEXTOID};
    Datum values[2] = {ObjectIdGetDatum(caller), CStringGetTextDatum(name)};
    bool isnull;
    Datum value;
    int version;
    if (SPI_connect() != SPI_OK_CONNECT)
        elog(ERROR, "SPI_connect failed");
    if (SPI_execute_with_args(
        "SELECT model_id, algorithm, n_features, n_outputs, model_format, model_data, metrics, format_version "
        "FROM db4ai.pg_db4ai_models WHERE owner_oid OPERATOR(pg_catalog.=) $1 "
        "AND model_name OPERATOR(pg_catalog.=) $2",
        2, types, values, NULL, true, 1) != SPI_OK_SELECT)
        elog(ERROR, "model lookup failed");
    if (SPI_processed != 1)
        ereport(ERROR, (errcode(ERRCODE_UNDEFINED_OBJECT), errmsg("model does not exist or is not owned by the current role: %s", name)));
    MemoryContextSwitchTo(context);
    model = palloc0(sizeof(*model));
    model->id = DatumGetInt64(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
    model->algorithm = db4ai_algorithm(SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 2));
    model->nfeatures = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 3, &isnull));
    model->noutputs = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 4, &isnull));
    model->format = SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 5);
    value = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 6, &isnull);
    model->data = DatumGetByteaPCopy(value);
    value = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 7, &isnull);
    model->metrics = DatumGetJsonbPCopy(value);
    version = DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 8, &isnull));
    SPI_finish();
    MemoryContextSwitchTo(context);
    if (version != DB4AI_FORMAT_VERSION || model->nfeatures < 1 || model->nfeatures > DB4AI_MAX_FEATURES ||
        model->noutputs < 1 || model->noutputs > 256 ||
        VARSIZE_ANY_EXHDR(model->data) > (Size) db4ai_max_model_mb * 1024 * 1024)
        ereport(ERROR, (errcode(ERRCODE_DATA_CORRUPTED), errmsg("invalid or unsupported model metadata")));
    return model;
}

bool
db4ai_model_delete(const char *name, Oid caller)
{
    Oid saved_user;
    int saved_security;
    volatile bool found = false;
    Oid types[2] = {REGROLEOID, TEXTOID};
    Datum values[2] = {ObjectIdGetDatum(caller), CStringGetTextDatum(name)};
    GetUserIdAndSecContext(&saved_user, &saved_security);
    SetUserIdAndSecContext(registry_owner(), saved_security | SECURITY_LOCAL_USERID_CHANGE | SECURITY_RESTRICTED_OPERATION);
    PG_TRY();
    {
        if (SPI_connect() != SPI_OK_CONNECT)
            elog(ERROR, "SPI_connect failed");
        if (SPI_execute_with_args(
            "DELETE FROM db4ai.pg_db4ai_models WHERE owner_oid OPERATOR(pg_catalog.=) $1 "
            "AND model_name OPERATOR(pg_catalog.=) $2", 2, types, values, NULL, false, 0) != SPI_OK_DELETE)
            elog(ERROR, "model deletion failed");
        found = SPI_processed != 0;
        SPI_finish();
        SetUserIdAndSecContext(saved_user, saved_security);
    }
    PG_CATCH();
    {
        SetUserIdAndSecContext(saved_user, saved_security);
        PG_RE_THROW();
    }
    PG_END_TRY();
    return found;
}
