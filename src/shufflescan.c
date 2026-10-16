/* SPDX-License-Identifier: MulanPSL-2.0
 * Explicit PostgreSQL ShuffleScan provider and SRF fallback.
 * Source query is executed once under the caller. Rescans replay materialized
 * tuples, never the source query. A bounded replacement shuffle is NOT a uniform
 * permutation unless the entire input fits the cache. No planner rewriting of
 * arbitrary user queries or implicit ORDER BY random().
 */
#include "postgres.h"
#include "db4ai.h"
#include "access/htup_details.h"
#include "catalog/dependency.h"
#include "catalog/pg_proc.h"
#include "catalog/pg_type_d.h"
#include "commands/extension.h"
#include "commands/explain.h"
#if PG_VERSION_NUM >= 180000
#include "commands/explain_format.h"
#endif
#include "executor/executor.h"
#include "executor/spi.h"
#include "funcapi.h"
#include "miscadmin.h"
#include "nodes/extensible.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "optimizer/paths.h"
#include "optimizer/pathnode.h"
#include "optimizer/restrictinfo.h"
#include "parser/parser.h"
#include "utils/builtins.h"
#include "utils/acl.h"
#include "utils/syscache.h"
#include "utils/lsyscache.h"
#include "utils/plancache.h"

PG_FUNCTION_INFO_V1(pg_db4ai_shuffle_scan);

typedef struct TupleShuffle
{
    MemoryContext context;
    Tuplestorestate *input;
    TupleDesc desc;
    TupleTableSlot *readslot;
    MinimalTuple *cache;
    int capacity, cached;
    int64 rows, emitted;
    Size max_tuple;
    uint64 seed;
    int epoch;
    bool initialized, spilled;
    pg_prng_state rng;
} TupleShuffle;

typedef struct ShuffleScanState
{
    CustomScanState css;
    MemoryContext context;
    TupleShuffle *reader;
    char *sql;
    int64 seed;
    int buffer_rows, epoch;
} ShuffleScanState;

static set_rel_pathlist_hook_type previous_path_hook=NULL;
static const CustomExecMethods shuffle_exec_methods;
static const CustomScanMethods shuffle_scan_methods;
static const CustomPathMethods shuffle_path_methods;

static TupleShuffle *
open_shuffle(const char *sql, TupleDesc expected, int64 seed, int buffer_rows, int epoch)
{
    MemoryContext context=CurrentMemoryContext;
    TupleShuffle *s=palloc0(sizeof(*s));
    SPIPlanPtr plan; Portal portal; List *raw; ListCell *lc;
    Size budget=(Size)db4ai_work_mem*1024, footprint; int i;
    if(buffer_rows<1 || buffer_rows>1000000 || epoch<0)
        ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("shuffle buffer_rows must be 1..1000000 and epoch must be nonnegative")));
    s->context=context; s->seed=(uint64)seed; s->epoch=epoch;
    raw=raw_parser(sql,RAW_PARSE_DEFAULT);
    if(list_length(raw)!=1 || !IsA(((RawStmt*)linitial(raw))->stmt,SelectStmt) ||
       ((SelectStmt*)((RawStmt*)linitial(raw))->stmt)->intoClause)
        ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("shuffle_scan requires exactly one SELECT without INTO")));
    if(SPI_connect()!=SPI_OK_CONNECT) elog(ERROR,"SPI_connect failed");
    plan=SPI_prepare_cursor(sql,0,NULL,CURSOR_OPT_NO_SCROLL);
    if(!plan || !SPI_is_cursor_plan(plan)) ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("shuffle_scan query is not a cursor")));
    foreach(lc,SPI_plan_get_plan_sources(plan))
    {
        CachedPlanSource *source=(CachedPlanSource*)lfirst(lc); ListCell *qc;
        foreach(qc,source->query_list)
        {
            Query *q=(Query*)lfirst(qc);
            if(q->commandType!=CMD_SELECT || q->hasModifyingCTE || q->rowMarks!=NIL)
                ereport(ERROR,(errcode(ERRCODE_READ_ONLY_SQL_TRANSACTION),errmsg("shuffle_scan query must be read-only")));
        }
    }
    portal=SPI_cursor_open(NULL,plan,NULL,NULL,true);
    MemoryContextSwitchTo(context);
    s->desc=CreateTupleDescCopy(expected);
    if(expected->natts!=portal->tupDesc->natts)
        ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("shuffle_scan column definition does not match source")));
    for(i=0;i<expected->natts;i++)
    {
        Form_pg_attribute a=TupleDescAttr(expected,i),b=TupleDescAttr(portal->tupDesc,i);
        if(a->attisdropped || a->atttypid!=b->atttypid || (a->atttypmod>=0 && a->atttypmod!=b->atttypmod))
            ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("shuffle_scan column %d type does not match source",i+1)));
    }
    s->input=tuplestore_begin_heap(true,false,db4ai_work_mem);
    for(;;)
    {
        uint64 r;
        SPI_cursor_fetch(portal,true,64); MemoryContextSwitchTo(context);
        if(!SPI_processed) { if(SPI_tuptable) SPI_freetuptable(SPI_tuptable); break; }
        for(r=0;r<SPI_processed;r++)
        {
            CHECK_FOR_INTERRUPTS();
            if(s->rows++>=db4ai_max_rows)
                ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("shuffle_scan input exceeds db4ai.max_rows")));
            s->max_tuple=Max(s->max_tuple,(Size)SPI_tuptable->vals[r]->t_len+sizeof(HeapTupleHeaderData));
            tuplestore_puttuple(s->input,SPI_tuptable->vals[r]);
        }
        SPI_freetuptable(SPI_tuptable);
    }
    SPI_cursor_close(portal); SPI_freeplan(plan); SPI_finish(); MemoryContextSwitchTo(context);
    s->spilled=!tuplestore_in_memory(s->input);
    s->readslot=MakeSingleTupleTableSlot(s->desc,&TTSOpsMinimalTuple);
    /* Include pointer storage, allocator overhead, read/replacement/output slots.
     * Tuplestore and SPI have their own budgets; this is the shuffle cache cap. */
    footprint=MAXALIGN(Max(s->max_tuple,(Size)32))+64+sizeof(MinimalTuple);
    if(footprint>budget/4)
        ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("shuffle_scan tuple is too large for db4ai.work_mem")));
    s->capacity=Max(1,Min((int64)buffer_rows,Min(Max(s->rows,(int64)1),(int64)(budget/footprint-3))));
    s->cache=palloc0(sizeof(MinimalTuple)*s->capacity);
    tuplestore_rescan(s->input);
    pg_prng_seed(&s->rng,s->seed^(UINT64CONST(0x9e3779b97f4a7c15)*((uint64)epoch+1)));
    return s;
}

static MinimalTuple
read_tuple(TupleShuffle *s)
{
    MinimalTuple result;
    MemoryContext old=MemoryContextSwitchTo(s->context);
    if(!tuplestore_gettupleslot(s->input,true,false,s->readslot))
    { MemoryContextSwitchTo(old); return NULL; }
    result=ExecCopySlotMinimalTuple(s->readslot); ExecClearTuple(s->readslot);
    MemoryContextSwitchTo(old); return result;
}

static MinimalTuple
next_tuple(TupleShuffle *s)
{
    MinimalTuple result, fresh;
    int chosen;
    CHECK_FOR_INTERRUPTS();
    if(s->epoch==0)
    { result=read_tuple(s); if(result) s->emitted++; return result; }
    if(!s->initialized)
    {
        while(s->cached<s->capacity && (result=read_tuple(s))!=NULL) s->cache[s->cached++]=result;
        s->initialized=true;
    }
    if(!s->cached)
    {
        if(s->emitted!=s->rows) elog(ERROR,"ShuffleScan row conservation failed");
        return NULL;
    }
    chosen=(int)pg_prng_uint64_range(&s->rng,0,s->cached-1);
    result=s->cache[chosen]; fresh=read_tuple(s);
    if(fresh) s->cache[chosen]=fresh;
    else s->cache[chosen]=s->cache[--s->cached];
    s->emitted++; return result;
}

static void
rewind_shuffle(TupleShuffle *s)
{
    int i;
    for(i=0;i<s->cached;i++) pfree(s->cache[i]);
    s->cached=0; s->emitted=0; s->initialized=false;
    tuplestore_rescan(s->input);
    pg_prng_seed(&s->rng,s->seed^(UINT64CONST(0x9e3779b97f4a7c15)*((uint64)s->epoch+1)));
}

static void
close_shuffle(TupleShuffle *s)
{
    if(s)
    { ExecDropSingleTupleTableSlot(s->readslot); tuplestore_end(s->input); }
}

Datum
pg_db4ai_shuffle_scan(PG_FUNCTION_ARGS)
{
    ReturnSetInfo *rsinfo=(ReturnSetInfo*)fcinfo->resultinfo;
    MemoryContext old,context;
    TupleShuffle *s; MinimalTuple t; TupleTableSlot *slot;
    int i;
    for(i=0;i<4;i++) if(PG_ARGISNULL(i)) ereport(ERROR,(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),errmsg("shuffle_scan arguments must not be NULL")));
    InitMaterializedSRF(fcinfo,MAT_SRF_USE_EXPECTED_DESC);
    old=CurrentMemoryContext;
    context=AllocSetContextCreate(old,"DB4AI ShuffleScan SRF",ALLOCSET_DEFAULT_SIZES);
    MemoryContextSwitchTo(context);
    s=open_shuffle(text_to_cstring(PG_GETARG_TEXT_PP(0)),rsinfo->setDesc,PG_GETARG_INT64(1),PG_GETARG_INT32(2),PG_GETARG_INT32(3));
    slot=MakeSingleTupleTableSlot(s->desc,&TTSOpsMinimalTuple);
    while((t=next_tuple(s))!=NULL)
    {
        ExecStoreMinimalTuple(t,slot,true); tuplestore_puttupleslot(rsinfo->setResult,slot); ExecClearTuple(slot);
    }
    ExecDropSingleTupleTableSlot(slot); close_shuffle(s);
    MemoryContextSwitchTo(old); MemoryContextDelete(context);
    return (Datum)0;
}

static Plan *
plan_shuffle(PlannerInfo *root, RelOptInfo *rel, CustomPath *best,
             List *tlist, List *clauses, List *custom_plans)
{
    RangeTblFunction *f=(RangeTblFunction*)linitial(best->custom_private);
    FuncExpr *call=(FuncExpr*)f->funcexpr;
    CustomScan *scan=makeNode(CustomScan);
    ListCell *ty,*tm,*co,*na; int col=1;
    scan->scan.plan.targetlist=tlist;
    scan->scan.plan.qual=extract_actual_clauses(clauses,false);
    scan->scan.scanrelid=0;
    forfour(ty,f->funccoltypes,tm,f->funccoltypmods,co,f->funccolcollations,na,f->funccolnames)
    {
        Var *v=makeVar(rel->relid,col,lfirst_oid(ty),lfirst_int(tm),lfirst_oid(co),0);
        scan->custom_scan_tlist=lappend(scan->custom_scan_tlist,makeTargetEntry((Expr*)v,col,pstrdup(strVal(lfirst(na))),false));
        col++;
    }
    scan->custom_private=lappend((List*)copyObjectImpl(call->args),makeConst(OIDOID,-1,InvalidOid,sizeof(Oid),ObjectIdGetDatum(call->funcid),false,true));
    scan->custom_exprs=list_make1(copyObjectImpl(call)); /* records function plan invalidation dependencies */
    scan->custom_relids=bms_make_singleton(rel->relid);
    scan->flags=CUSTOMPATH_SUPPORT_PROJECTION;
    scan->methods=&shuffle_scan_methods;
    return &scan->scan.plan;
}

static bool
unsupported_scan_var(Node *node, void *context)
{
    if(!node) return false;
    if(IsA(node,Var) && ((Var*)node)->varattno<=0) return true;
    return expression_tree_walker(node,unsupported_scan_var,context);
}

static void
shuffle_rel_path(PlannerInfo *root, RelOptInfo *rel, Index rti, RangeTblEntry *rte)
{
    RangeTblFunction *f; FuncExpr *call; ListCell *lc; Oid ext;
    CustomPath *path;
    if(previous_path_hook) previous_path_hook(root,rel,rti,rte);
    if(rte->rtekind!=RTE_FUNCTION || rte->funcordinality || list_length(rte->functions)!=1 || rel->lateral_relids)
        return;
    f=(RangeTblFunction*)linitial(rte->functions);
    if(!IsA(f->funcexpr,FuncExpr) || f->funccolcount<1 || f->funccoltypes==NIL) return;
    call=(FuncExpr*)f->funcexpr;
    if(list_length(call->args)!=4 || strcmp(get_func_name(call->funcid),"shuffle_scan")!=0) return;
    ext=get_extension_oid("pg_db4ai",true);
    if(!OidIsValid(ext) || getExtensionOfObject(ProcedureRelationId,call->funcid)!=ext) return;
    {
        HeapTuple tuple=SearchSysCache1(PROCOID,ObjectIdGetDatum(call->funcid));
        bool isnull;
        Form_pg_proc proc;
        if(!HeapTupleIsValid(tuple)) return;
        proc=(Form_pg_proc)GETSTRUCT(tuple);
        (void)SysCacheGetAttr(PROCOID,tuple,Anum_pg_proc_proconfig,&isnull);
        if(proc->prosecdef || !isnull) { ReleaseSysCache(tuple); return; }
        ReleaseSysCache(tuple);
    }
    foreach(lc,call->args) if(!IsA(lfirst(lc),Const) || ((Const*)lfirst(lc))->constisnull) return;
    /* Whole-row null-extension semantics must remain PostgreSQL's own.
     * Do not rewrite whole-row or system-attribute expressions to scalar Vars. */
    if(unsupported_scan_var((Node*)rel->reltarget->exprs,NULL)) return;
    foreach(lc,rel->baserestrictinfo)
        if(unsupported_scan_var((Node*)((RestrictInfo*)lfirst(lc))->clause,NULL)) return;
    path=makeNode(CustomPath);
    path->path.pathtype=T_CustomScan;
    path->path.parent=rel; path->path.pathtarget=rel->reltarget;
    path->path.rows=rel->rows; path->path.parallel_safe=false;
    path->path.startup_cost=1; path->path.total_cost=1+rel->rows;
    path->flags=CUSTOMPATH_SUPPORT_PROJECTION;
    path->custom_private=list_make1(copyObjectImpl(f)); path->methods=&shuffle_path_methods;
    /* Only this explicit function call is replaced. Dynamic/LATERAL calls keep
     * PostgreSQL's ordinary FunctionScan and execute the same SRF semantics. */
    rel->pathlist=NIL; rel->partial_pathlist=NIL;
    add_path(rel,&path->path);
}

static Node *
create_shuffle_state(CustomScan *plan)
{
    ShuffleScanState *state=palloc0(sizeof(*state));
    NodeSetTag(&state->css,T_CustomScanState);
    state->css.methods=&shuffle_exec_methods;
    /* PG 16/17 have no CustomScanState.slotOps. Virtual slot works throughout. */
    return (Node*)&state->css;
}

static void
begin_shuffle(CustomScanState *css, EState *estate, int eflags)
{
    ShuffleScanState *s=(ShuffleScanState*)css;
    CustomScan *p=(CustomScan*)css->ss.ps.plan;
    Oid function=DatumGetObjectId(((Const*)list_nth(p->custom_private,4))->constvalue);
    AclResult acl=object_aclcheck(ProcedureRelationId,function,GetUserId(),ACL_EXECUTE);
    if(acl!=ACLCHECK_OK) aclcheck_error(acl,OBJECT_FUNCTION,get_func_name(function));
    s->sql=TextDatumGetCString(((Const*)list_nth(p->custom_private,0))->constvalue);
    s->seed=DatumGetInt64(((Const*)list_nth(p->custom_private,1))->constvalue);
    s->buffer_rows=DatumGetInt32(((Const*)list_nth(p->custom_private,2))->constvalue);
    s->epoch=DatumGetInt32(((Const*)list_nth(p->custom_private,3))->constvalue);
    s->context=AllocSetContextCreate(estate->es_query_cxt,"DB4AI ShuffleScan",ALLOCSET_DEFAULT_SIZES);
}

static TupleTableSlot *
shuffle_access(ScanState *ss)
{
    ShuffleScanState *s=(ShuffleScanState*)ss;
    MemoryContext old;
    MinimalTuple t;
    TupleTableSlot *slot=ss->ss_ScanTupleSlot;
    ExecClearTuple(slot);
    if(!s->reader)
    {
        old=MemoryContextSwitchTo(s->context);
        s->reader=open_shuffle(s->sql,slot->tts_tupleDescriptor,s->seed,s->buffer_rows,s->epoch);
        MemoryContextSwitchTo(old);
    }
    t=next_tuple(s->reader);
    if(!t) return slot;
    /* ExecForceStoreMinimalTuple supports virtual slots on every supported PG.
     * It materializes/copies as needed and releases the owned minimal tuple. */
    ExecForceStoreMinimalTuple(t,slot,true);
    return slot;
}

static bool
shuffle_recheck(ScanState *ss, TupleTableSlot *slot) { return true; }

static TupleTableSlot *
exec_shuffle(CustomScanState *css)
{
    return ExecScan(&css->ss,shuffle_access,shuffle_recheck);
}

static void
end_shuffle(CustomScanState *css)
{
    ShuffleScanState *s=(ShuffleScanState*)css;
    ExecClearTuple(css->ss.ss_ScanTupleSlot);
    close_shuffle(s->reader);
    if(s->context) { MemoryContextDelete(s->context); s->context=NULL; s->reader=NULL; }
}

static void
rescan_shuffle(CustomScanState *css)
{
    ShuffleScanState *s=(ShuffleScanState*)css;
    ExecScanReScan(&css->ss);
    if(s->reader) rewind_shuffle(s->reader);
}

static void
explain_shuffle(CustomScanState *css, List *ancestors, ExplainState *es)
{
    ShuffleScanState *s=(ShuffleScanState*)css;
    ExplainPropertyInteger("Seed",NULL,s->seed,es);
    ExplainPropertyInteger("Epoch",NULL,s->epoch,es);
    ExplainPropertyInteger("Requested Cache Rows",NULL,s->buffer_rows,es);
    ExplainPropertyText("Rescan","replay fixed input; same seed and epoch",es);
    if(s->reader)
    {
        ExplainPropertyInteger("Input Rows",NULL,s->reader->rows,es);
        ExplainPropertyInteger("Cache Rows",NULL,s->reader->capacity,es);
        ExplainPropertyBool("Input Spilled",s->reader->spilled,es);
        ExplainPropertyText("Shuffle",s->epoch==0?"sequential":(s->reader->rows<=s->reader->capacity?"full-cache permutation":"bounded replacement (not globally uniform)"),es);
    }
}

static const CustomPathMethods shuffle_path_methods={.CustomName="ShuffleScan",.PlanCustomPath=plan_shuffle};
static const CustomScanMethods shuffle_scan_methods={.CustomName="ShuffleScan",.CreateCustomScanState=create_shuffle_state};
static const CustomExecMethods shuffle_exec_methods={
    .CustomName="ShuffleScan",.BeginCustomScan=begin_shuffle,.ExecCustomScan=exec_shuffle,
    .EndCustomScan=end_shuffle,.ReScanCustomScan=rescan_shuffle,.ExplainCustomScan=explain_shuffle
};

void
db4ai_shuffle_init(void)
{
    RegisterCustomScanMethods(&shuffle_scan_methods);
    previous_path_hook=set_rel_pathlist_hook; set_rel_pathlist_hook=shuffle_rel_path;
}
