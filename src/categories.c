/* SPDX-License-Identifier: MulanPSL-2.0
 * Typed category dictionaries use PostgreSQL equality/hash functions, never
 * lossy float casts or truncated string keys. Artifacts contain names and text
 * values, not OIDs or Datum memory images.
 */
#include "postgres.h"
#include <math.h>
#include "db4ai.h"
#include "catalog/namespace.h"
#include "catalog/pg_collation.h"
#include "catalog/pg_type.h"
#include "catalog/pg_type_d.h"
#include "common/shortest_dec.h"
#include "nodes/makefuncs.h"
#include "utils/builtins.h"
#include "utils/datum.h"
#include "utils/json.h"
#include "utils/lsyscache.h"
#include "utils/syscache.h"
#include "utils/guc.h"

typedef struct CategoryBucket { uint32 hash; int first; } CategoryBucket;

Db4aiDictionary *
db4ai_dictionary_new(Oid type, int32 typmod, Oid coll, int limit)
{
    Db4aiDictionary *d=palloc0(sizeof(*d));
    TypeCacheEntry *tc;
    HASHCTL ctl;
    Oid output; bool varlena;
    d->context=CurrentMemoryContext;
    d->type=getBaseTypeAndTypmod(type,&typmod);
    d->typmod=typmod; d->collation=coll; d->limit=limit;
    if(get_typtype(d->type)==TYPTYPE_PSEUDO)
        ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("anonymous record and pseudo-type labels are not durable"),
                       errhint("Cast to a named composite or concrete SQL type.")));
    tc=lookup_type_cache(d->type,TYPECACHE_EQ_OPR_FINFO|TYPECACHE_HASH_PROC_FINFO);
    if (!OidIsValid(tc->eq_opr_finfo.fn_oid) || !OidIsValid(tc->hash_proc_finfo.fn_oid))
        ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("category type %s requires equality and hashing",format_type_be(d->type))));
    fmgr_info_copy(&d->equal,&tc->eq_opr_finfo,d->context);
    fmgr_info_copy(&d->hash,&tc->hash_proc_finfo,d->context);
    getTypeOutputInfo(d->type,&output,&varlena); fmgr_info(output,&d->output);
    get_typlenbyval(d->type,&d->typlen,&d->typbyval);
    memset(&ctl,0,sizeof(ctl)); ctl.keysize=sizeof(uint32); ctl.entrysize=sizeof(CategoryBucket); ctl.hcxt=d->context;
    d->buckets=hash_create("DB4AI category buckets",32,&ctl,HASH_ELEM|HASH_BLOBS|HASH_CONTEXT);
    return d;
}

int
db4ai_dictionary_index(Db4aiDictionary *d, Datum value, bool isnull, bool insert)
{
    uint32 h=isnull ? 0x9e3779b9U : DatumGetUInt32(FunctionCall1Coll(&d->hash,d->collation,value));
    CategoryBucket *bucket;
    bool found;
    int index;
    MemoryContext old=MemoryContextSwitchTo(d->context);
    bucket=hash_search(d->buckets,&h,insert?HASH_ENTER:HASH_FIND,&found);
    if (bucket && found)
        for (index=bucket->first;index>=0;index=d->next[index])
            if (d->nulls[index]==isnull && (isnull || DatumGetBool(FunctionCall2Coll(&d->equal,d->collation,d->values[index],value))))
            {
                if (insert) d->frequency[index]++;
                MemoryContextSwitchTo(old); return index;
            }
    if (!insert) { MemoryContextSwitchTo(old); return -1; }
    if (!found) bucket->first=-1;
    if (d->count==d->limit)
        ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("too many categories (limit %d)",d->limit)));
    if (d->count==d->capacity)
    {
        int cap=Min(d->limit,Max(16,d->capacity*2));
        if (d->capacity==0)
        {
            d->values=palloc(sizeof(Datum)*cap); d->nulls=palloc(sizeof(bool)*cap);
            d->frequency=palloc(sizeof(int64)*cap); d->next=palloc(sizeof(int)*cap);
        }
        else
        {
            d->values=repalloc(d->values,sizeof(Datum)*cap); d->nulls=repalloc(d->nulls,sizeof(bool)*cap);
            d->frequency=repalloc(d->frequency,sizeof(int64)*cap); d->next=repalloc(d->next,sizeof(int)*cap);
        }
        d->capacity=cap;
    }
    d->bytes+=sizeof(Datum)+sizeof(int64)+sizeof(bool)+sizeof(int)+(isnull?0:datumGetSize(value,d->typbyval,d->typlen));
    if (d->bytes>(Size)db4ai_max_model_mb*1024*1024)
        ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("category dictionary exceeds db4ai.max_model_mb")));
    index=d->count++;
    d->values[index]=isnull? (Datum)0 : datumCopy(value,d->typbyval,d->typlen);
    d->nulls[index]=isnull; d->frequency[index]=1; d->next[index]=bucket->first; bucket->first=index;
    MemoryContextSwitchTo(old); return index;
}

char *
db4ai_dictionary_text(const Db4aiDictionary *d, int index)
{
    char number[DOUBLE_SHORTEST_DECIMAL_LEN];
    if(index<0 || index>=d->count) ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid category index")));
    if(d->nulls[index]) return NULL;
    if(d->type==FLOAT8OID)
    {
        double_to_shortest_decimal_buf(DatumGetFloat8(d->values[index]),number); return pstrdup(number);
    }
    if(d->type==FLOAT4OID)
    {
        float_to_shortest_decimal_buf(DatumGetFloat4(d->values[index]),number); return pstrdup(number);
    }
    return OutputFunctionCall((FmgrInfo *)&d->output,d->values[index]);
}

void
db4ai_dictionary_save(StringInfo out, const Db4aiDictionary *d)
{
    HeapTuple tuple=SearchSysCache1(TYPEOID,ObjectIdGetDatum(d->type));
    Form_pg_type t;
    int i, nest;
    if (!HeapTupleIsValid(tuple)) elog(ERROR,"category type was removed");
    t=(Form_pg_type)GETSTRUCT(tuple);
    appendStringInfoString(out,"{\"type_schema\":"); escape_json(out,get_namespace_name(t->typnamespace));
    appendStringInfoString(out,",\"type_name\":"); escape_json(out,NameStr(t->typname));
    appendStringInfo(out,",\"typmod\":%d,\"collation\":",d->typmod);
    ReleaseSysCache(tuple);
    if(OidIsValid(d->collation))
    {
        tuple=SearchSysCache1(COLLOID,ObjectIdGetDatum(d->collation));
        if (!HeapTupleIsValid(tuple)) elog(ERROR,"category collation was removed");
        { Form_pg_collation c=(Form_pg_collation)GETSTRUCT(tuple);
          appendStringInfoChar(out,'['); escape_json(out,get_namespace_name(c->collnamespace));
          appendStringInfoChar(out,','); escape_json(out,NameStr(c->collname)); appendStringInfoChar(out,']'); }
        ReleaseSysCache(tuple);
    }
    else appendStringInfoString(out,"null");
    appendStringInfoString(out,",\"values\":[");
    /* Stable built-in textual I/O, including dates/timestamps across dump/restore. */
    nest=NewGUCNestLevel();
    PG_TRY();
    {
        (void)set_config_option("DateStyle","ISO,YMD",PGC_USERSET,PGC_S_SESSION,GUC_ACTION_SAVE,true,0,false);
        (void)set_config_option("IntervalStyle","postgres",PGC_USERSET,PGC_S_SESSION,GUC_ACTION_SAVE,true,0,false);
        (void)set_config_option("TimeZone","UTC",PGC_USERSET,PGC_S_SESSION,GUC_ACTION_SAVE,true,0,false);
        for(i=0;i<d->count;i++)
        {
            char *text=db4ai_dictionary_text(d,i);
            if(i) appendStringInfoChar(out,',');
            if(text) { escape_json(out,text); pfree(text); } else appendStringInfoString(out,"null");
        }
        AtEOXact_GUC(true,nest);
    }
    PG_CATCH(); { AtEOXact_GUC(false,nest); PG_RE_THROW(); } PG_END_TRY();
    appendStringInfoString(out,"]}");
}

Db4aiDictionary *
db4ai_dictionary_load(Jsonb *json)
{
    char *schema=db4ai_json_string(json,"type_schema",NULL);
    char *name=db4ai_json_string(json,"type_name",NULL);
    Oid type, input, io, coll=InvalidOid;
    JsonbValue *values=db4ai_json_field(json,"values",true);
    JsonbValue *cv=db4ai_json_field(json,"collation",true);
    JsonbIterator *it; JsonbIteratorToken token; JsonbValue v;
    Db4aiDictionary *d; int nest, n=0;
    double tm=db4ai_json_number(json,"typmod",-1);
    if(!schema || !name || tm < -1 || tm > PG_INT32_MAX || floor(tm)!=tm)
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid category type metadata")));
    type=GetSysCacheOid2(TYPENAMENSP,Anum_pg_type_oid,CStringGetDatum(name),ObjectIdGetDatum(get_namespace_oid(schema,false)));
    if(!OidIsValid(type)) ereport(ERROR,(errcode(ERRCODE_UNDEFINED_OBJECT),errmsg("model category type %s.%s is unavailable",schema,name)));
    if(cv->type==jbvBinary && JsonContainerIsArray(cv->val.binary.data) && JsonContainerSize(cv->val.binary.data)==2)
    {
        List *parts=NIL;
        it=JsonbIteratorInit(cv->val.binary.data);
        while((token=JsonbIteratorNext(&it,&v,true))!=WJB_DONE) if(token==WJB_ELEM)
        {
            if(v.type!=jbvString) ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid model collation")));
            parts=lappend(parts,makeString(pnstrdup(v.val.string.val,v.val.string.len)));
        }
        coll=get_collation_oid(parts,false);
    }
    else if(cv->type!=jbvNull) ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid model collation")));
    if(values->type!=jbvBinary || !JsonContainerIsArray(values->val.binary.data) || JsonContainerSize(values->val.binary.data)>65536)
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid category values")));
    d=db4ai_dictionary_new(type,(int32)tm,coll,65536);
    getTypeInputInfo(type,&input,&io);
    nest=NewGUCNestLevel();
    PG_TRY();
    {
        (void)set_config_option("DateStyle","ISO,YMD",PGC_USERSET,PGC_S_SESSION,GUC_ACTION_SAVE,true,0,false);
        (void)set_config_option("IntervalStyle","postgres",PGC_USERSET,PGC_S_SESSION,GUC_ACTION_SAVE,true,0,false);
        (void)set_config_option("TimeZone","UTC",PGC_USERSET,PGC_S_SESSION,GUC_ACTION_SAVE,true,0,false);
        it=JsonbIteratorInit(values->val.binary.data);
        while((token=JsonbIteratorNext(&it,&v,true))!=WJB_DONE) if(token==WJB_ELEM)
        {
            Datum value=(Datum)0; int index;
            if(v.type==jbvString)
            {
                char *text=pnstrdup(v.val.string.val,v.val.string.len);
                value=OidInputFunctionCall(input,text,io,d->typmod); pfree(text);
            }
            else if(v.type!=jbvNull) ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid serialized category")));
            index=db4ai_dictionary_index(d,value,v.type==jbvNull,true);
            if(index!=n++) ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("duplicate category in model")));
            if(v.type!=jbvNull && !d->typbyval) pfree(DatumGetPointer(value));
        }
        AtEOXact_GUC(true,nest);
    }
    PG_CATCH(); { AtEOXact_GUC(false,nest); PG_RE_THROW(); } PG_END_TRY();
    return d;
}
