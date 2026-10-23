/* SPDX-License-Identifier: MulanPSL-2.0
 * Bounded sequential portfolio search. No Python service, external AutoML
 * runtime, parallel trial workers, or persisted losing candidate models.
 */
#include "postgres.h"
#include "automl.h"
#include <float.h>
#include <math.h>
#include <sys/resource.h>
#include "access/xact.h"
#include "catalog/pg_type_d.h"
#include "catalog/pg_proc.h"
#include "parser/parse_func.h"
#include "nodes/makefuncs.h"
#include "utils/acl.h"
#include "utils/lsyscache.h"
#include "executor/spi.h"
#include "miscadmin.h"
#include "utils/builtins.h"
#include "utils/fmgrprotos.h"
#include "utils/guc.h"
#include "utils/json.h"
#include "utils/resowner.h"
#include "utils/timeout.h"

PG_FUNCTION_INFO_V1(pg_db4ai_automl);
static bool automl_active = false;

typedef struct Evaluation
{
    double accuracy, balanced_accuracy, macro_f1, log_loss, auc;
    double rmse, mae, r2;
    bool has_auc;
    double objective; /* Minimized, even for higher-is-better metrics. */
} Evaluation;

typedef struct RankedProbability { double p; int label; } RankedProbability;

typedef struct Candidate
{
    Db4aiAlgorithm algorithm;
    double l2, rate;
    int depth;
    bool complete;
    double objective;
} Candidate;

typedef struct Search
{
    MemoryContext context, best_context;
    Db4aiAutoOptions options;
    Db4aiAutoInput *input;
    Candidate candidates[5];
    int ncandidates, trials, completed;
    Db4aiModel best;
    Db4aiOptions best_options;
    Evaluation best_validation;
    double best_objective, best_fit_seconds;
    int64 best_rows;
    StringInfoData records;
    double started, search_deadline;
    bool budget_stop;
    char *metric;
} Search;

static double
cpu_seconds(void)
{
    struct rusage r;
    if(getrusage(RUSAGE_SELF,&r)!=0) return 0;
    return r.ru_utime.tv_sec+r.ru_utime.tv_usec/1e6+r.ru_stime.tv_sec+r.ru_stime.tv_usec/1e6;
}

static double
bounded(Jsonb *j,const char *key,double def,double low,double high,bool integral)
{
    double v=db4ai_json_number(j,key,def);
    if(v<low || v>high || (integral && v!=floor(v)))
        ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("invalid AutoML option %s",key),errdetail("Expected %s between %g and %g.",integral?"integer":"number",low,high)));
    return v;
}

static void
parse_options(Jsonb *json,bool classification,Db4aiAutoOptions *o)
{
    static const char *keys[]={"time_budget_seconds","max_trials","max_rows","memory_mb","max_input_mb",
        "max_categories","seed","validation_fraction","test_fraction","metric","split","group_column","time_column",
        "use_xgboost","include_naive_bayes","missing_indicators","pilot_rows","min_improvement","max_encoded_mb"};
    JsonbIterator *it; JsonbValue v; JsonbIteratorToken token;
    int i;
    if(!JB_ROOT_IS_OBJECT(json)) ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("AutoML options must be a JSON object")));
    it=JsonbIteratorInit(&json->root);
    while((token=JsonbIteratorNext(&it,&v,true))!=WJB_DONE)
        if(token==WJB_KEY)
        {
            bool found=false;
            for(i=0;i<lengthof(keys);i++)
                if(strlen(keys[i])==v.val.string.len && memcmp(keys[i],v.val.string.val,v.val.string.len)==0) { found=true; break; }
            if(!found) ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("unknown AutoML option: %.*s",v.val.string.len,v.val.string.val)));
        }
    o->seconds=bounded(json,"time_budget_seconds",30,0.01,3600,false);
    o->max_trials=(int)bounded(json,"max_trials",8,1,DB4AI_AM_MAX_TRIALS,true);
    o->max_rows=(int)bounded(json,"max_rows",200000,30,1000000,true);
    o->memory_mb=(int)bounded(json,"memory_mb",128,16,4096,true);
    o->max_input_mb=(int)bounded(json,"max_input_mb",256,1,8192,true);
    o->max_encoded_mb=(int)bounded(json,"max_encoded_mb",512,1,8192,true);
    o->max_categories=(int)bounded(json,"max_categories",32,1,128,true);
    o->seed=(int)bounded(json,"seed",42,0,INT_MAX,true);
    o->pilot_rows=(int)bounded(json,"pilot_rows",4096,256,1000000,true);
    o->validation_fraction=bounded(json,"validation_fraction",0.2,0.05,0.4,false);
    o->test_fraction=bounded(json,"test_fraction",0.2,0.05,0.4,false);
    if(o->validation_fraction+o->test_fraction>0.6)
        ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("at least 40 percent of split groups must be assigned to training")));
    o->metric=db4ai_json_string(json,"metric",classification?"balanced_accuracy":"rmse");
    if(classification)
    {
        if(strcmp(o->metric,"accuracy") && strcmp(o->metric,"balanced_accuracy") && strcmp(o->metric,"log_loss") && strcmp(o->metric,"roc_auc"))
            ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("classification metric must be accuracy, balanced_accuracy, log_loss or roc_auc")));
    }
    else if(strcmp(o->metric,"rmse") && strcmp(o->metric,"mae"))
        ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("regression metric must be rmse or mae")));
    o->split=db4ai_json_string(json,"split","random");
    o->group_column=db4ai_json_string(json,"group_column",NULL);
    o->time_column=db4ai_json_string(json,"time_column",NULL);
    if(strcmp(o->split,"random") && strcmp(o->split,"group") && strcmp(o->split,"time"))
        ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("split must be random, group or time")));
    if((strcmp(o->split,"group")==0)!= (o->group_column!=NULL) ||
       (strcmp(o->split,"time")==0)!=(o->time_column!=NULL))
        ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("group/time split requires exactly its matching key column")));
    o->use_xgboost=db4ai_json_boolean(json,"use_xgboost",true);
    o->include_nb=db4ai_json_boolean(json,"include_naive_bayes",false);
    if(o->include_nb && !classification)
        ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("Naive Bayes is not a regression candidate")));
    o->missing_indicators=db4ai_json_boolean(json,"missing_indicators",true);
    o->min_improvement=bounded(json,"min_improvement",0,0,1e100,false);
}

static int
rank_compare(const void *a,const void *b)
{
    const RankedProbability *x=a,*y=b;
    return x->p<y->p?-1:(x->p>y->p?1:0);
}

static Evaluation
evaluate(const Db4aiPipeline *p,Db4aiData *data,const char *metric,bool dummy)
{
    Evaluation e={0};
    double *row=palloc(sizeof(double)*data->ncolumns);
    double probabilities[256],train_total=0;
    int64 actual[256]={0},predicted[256]={0},correct[256]={0};
    RankedProbability *ranks=NULL;
    double squared=0,absolute=0,ym=0,ym2=0;
    int64 seen=0; int c,majority=0;
    if(p->classification)
    {
        for(c=0;c<p->nclasses;c++)
        {
            train_total+=p->labels->frequency[c];
            if(p->labels->frequency[c]>p->labels->frequency[majority]) majority=c;
        }
        if(p->nclasses==2) ranks=palloc(sizeof(RankedProbability)*data->nrows);
    }
    db4ai_data_rewind(data);
    while(db4ai_data_next(data,row))
    {
        db4ai_automl_check();
        if(p->classification)
        {
            int label=(int)row[0],prediction=dummy?majority:0;
            if(dummy) for(c=0;c<p->nclasses;c++) probabilities[c]=p->labels->frequency[c]/train_total;
            else
            {
                db4ai_pipeline_proba(p,row+1,probabilities);
                for(c=1;c<p->nclasses;c++) if(probabilities[c]>probabilities[prediction]) prediction=c;
            }
            actual[label]++; predicted[prediction]++;
            if(label==prediction) correct[label]++;
            /* Explicit probability clipping for comparable finite log-loss.
             * This value is disclosed in every report, not hidden as calibration. */
            e.log_loss-=log(Max(1e-15,probabilities[label]));
            if(ranks) { ranks[seen].p=probabilities[1]; ranks[seen].label=label; }
        }
        else
        {
            double y=row[0]*p->target_scale+p->target_mean;
            double prediction=dummy?p->target_mean:db4ai_pipeline_value(p,row+1);
            double d=prediction-y,delta=y-ym;
            absolute+=fabs(d); squared+=d*d;
            ym+=delta/(seen+1); ym2+=delta*(y-ym);
            db4ai_check_finite(squared); db4ai_check_finite(ym2);
        }
        seen++;
    }
    if(!seen) elog(ERROR,"cannot evaluate an empty partition");
    if(p->classification)
    {
        for(c=0;c<p->nclasses;c++)
        {
            e.accuracy+=correct[c];
            e.balanced_accuracy+=actual[c]?(double)correct[c]/actual[c]:0;
            e.macro_f1+=(actual[c]+predicted[c])?2.0*correct[c]/(actual[c]+predicted[c]):0;
        }
        e.accuracy/=seen; e.balanced_accuracy/=p->nclasses; e.macro_f1/=p->nclasses; e.log_loss/=seen;
        if(ranks && actual[0] && actual[1])
        {
            int64 begin=0; double rank_sum=0;
            qsort(ranks,seen,sizeof(RankedProbability),rank_compare);
            while(begin<seen)
            {
                int64 end=begin+1,r;
                while(end<seen && ranks[end].p==ranks[begin].p) end++;
                for(r=begin;r<end;r++) if(ranks[r].label==1) rank_sum+=(begin+1+end)/2.0;
                begin=end;
            }
            e.auc=(rank_sum-actual[1]*(actual[1]+1.0)/2)/(actual[0]*(double)actual[1]); e.has_auc=true;
        }
        if(!strcmp(metric,"accuracy")) e.objective=-e.accuracy;
        else if(!strcmp(metric,"balanced_accuracy")) e.objective=-e.balanced_accuracy;
        else if(!strcmp(metric,"roc_auc"))
        {
            if(!e.has_auc) ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("roc_auc requires a binary task with both classes in the holdout")));
            e.objective=-e.auc;
        }
        else e.objective=e.log_loss;
    }
    else
    {
        e.rmse=sqrt(squared/seen); e.mae=absolute/seen;
        e.r2=ym2>0?1-squared/ym2:(squared==0?1:0);
        e.objective=!strcmp(metric,"mae")?e.mae:e.rmse;
    }
    if(ranks) pfree(ranks);
    pfree(row);
    db4ai_check_finite(e.objective);
    return e;
}

static void
write_evaluation(StringInfo s,const Evaluation *e,bool classification)
{
    if(classification)
    {
        appendStringInfo(s,"{\"accuracy\":%.17g,\"balanced_accuracy\":%.17g,\"macro_f1\":%.17g,\"log_loss\":%.17g,\"roc_auc\":",
                         e->accuracy,e->balanced_accuracy,e->macro_f1,e->log_loss);
        if(e->has_auc) appendStringInfo(s,"%.17g",e->auc); else appendStringInfoString(s,"null");
        appendStringInfoChar(s,'}');
    }
    else appendStringInfo(s,"{\"rmse\":%.17g,\"mae\":%.17g,\"r2\":%.17g}",e->rmse,e->mae,e->r2);
}

/* Nested bounded pilot sample. First include one example of each training
 * class, then reservoir-sample the remaining rows. No holdout is sampled. */
static Db4aiData *
pilot_data(Search *s)
{
    Db4aiData *full=&s->input->train,*small;
    int size=Min((int)full->nrows,s->options.pilot_rows);
    double *reservoir,*row;
    bool included[256]={0};
    pg_prng_state rng;
    int occupied=0,mandatory=0,r;
    int64 rest=0;
    if(size==full->nrows) return full;
    if((Size)size*full->ncolumns*sizeof(double)>(Size)s->options.memory_mb*1024*1024/4)
        size=Max(s->input->pipeline->nclasses,(int)((Size)s->options.memory_mb*1024*1024/4/(full->ncolumns*sizeof(double))));
    if(size<s->input->pipeline->nclasses || size<32)
        ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("pilot working set exceeds its buffer budget")));
    reservoir=palloc(sizeof(double)*(Size)size*full->ncolumns);
    row=palloc(sizeof(double)*full->ncolumns);
    pg_prng_seed(&rng,s->options.seed);
    if(s->input->pipeline->classification)
    {
        db4ai_data_rewind(full);
        while(db4ai_data_next(full,row))
        {
            int label=(int)row[0]; db4ai_automl_check();
            if(!included[label]) { included[label]=true; memcpy(reservoir+(Size)mandatory*full->ncolumns,row,sizeof(double)*full->ncolumns); mandatory++; }
        }
    }
    memset(included,0,sizeof(included)); occupied=mandatory;
    db4ai_data_rewind(full);
    while(db4ai_data_next(full,row))
    {
        uint64 position;
        db4ai_automl_check();
        if(s->input->pipeline->classification && !included[(int)row[0]]) { included[(int)row[0]]=true; continue; }
        if(occupied<size) { memcpy(reservoir+(Size)occupied*full->ncolumns,row,sizeof(double)*full->ncolumns); occupied++; }
        else
        {
            position=pg_prng_uint64_range(&rng,0,rest);
            if(position<(uint64)(size-mandatory)) memcpy(reservoir+(Size)(position+mandatory)*full->ncolumns,row,sizeof(double)*full->ncolumns);
        }
        rest++;
    }
    /* Random order also prevents a class-sorted SELECT from biasing SGD. */
    for(r=occupied-1;r>0;r--)
    {
        int other=(int)pg_prng_uint64_range(&rng,0,r);
        memcpy(row,reservoir+(Size)r*full->ncolumns,sizeof(double)*full->ncolumns);
        memcpy(reservoir+(Size)r*full->ncolumns,reservoir+(Size)other*full->ncolumns,sizeof(double)*full->ncolumns);
        memcpy(reservoir+(Size)other*full->ncolumns,row,sizeof(double)*full->ncolumns);
    }
    small=palloc0(sizeof(*small)); *small=*full; small->nrows=occupied; small->position=0;
    small->dense=db4ai_rows_create(full->ncolumns,(Size)s->options.memory_mb*1024*1024/8);
    for(r=0;r<occupied;r++) db4ai_rows_append(small->dense,reservoir+(Size)r*full->ncolumns);
    db4ai_rows_seal(small->dense); small->spilled=db4ai_rows_spilled(small->dense);
    pfree(reservoir); pfree(row); return small;
}

static void
candidate_options(Search *s,const Candidate *c,int stage,Db4aiOptions *out)
{
    Jsonb *empty=DatumGetJsonbP(DirectFunctionCall1(jsonb_in,CStringGetDatum("{}")));
    db4ai_options(empty,c->algorithm,out);
    out->seed=s->options.seed; out->batch_size=Min(256,Max(16,(int)((Size)db4ai_work_mem*1024/(4*(s->input->pipeline->nencoded+2)*sizeof(double)))));
    out->iterations=db4ai_is_xgboost(c->algorithm)?(stage?160:24):(stage?200:32);
    out->learning_rate=c->rate; out->l2=c->l2; out->max_depth=c->depth;
    out->shuffle_mode=DB4AI_SHUFFLE_ROW; out->shuffle_first=true; out->shuffle_buffer=128;
    out->classifier=0; /* Pipeline OVA scores are logistic, never SVM margins. */
    out->nthread=1; out->tolerance=1e-8; out->max_seconds=0;
    if(db4ai_is_xgboost(c->algorithm)) out->eval_metric=s->input->pipeline->classification?
        (s->input->pipeline->nclasses==2?"logloss":"mlogloss"):"rmse";
}

static bool
run_candidate(Search *s,Candidate *candidate,Db4aiData *training,int stage,double trial_seconds)
{
    MemoryContext outer=CurrentMemoryContext;
    MemoryContext trial=AllocSetContextCreate(outer,"DB4AI AutoML trial",ALLOCSET_DEFAULT_SIZES);
    ResourceOwner owner=CurrentResourceOwner;
    volatile bool success=false;
    double start=db4ai_automl_now();
    int trial_index=++s->trials;
    BeginInternalSubTransaction(NULL);
    PG_TRY();
    {
        Db4aiPipeline runtime=*s->input->pipeline;
        Db4aiData data=*training;
        Db4aiOptions options;
        Db4aiModel model={0}; Evaluation evaluation;
        double elapsed;
        MemoryContextSwitchTo(trial);
        db4ai_automl_soft_deadline(Min(s->search_deadline,start+trial_seconds));
        candidate_options(s,candidate,stage,&options);
        model.algorithm=candidate->algorithm; model.nfeatures=runtime.nencoded;
        runtime.native=NULL; runtime.xgb=NULL;
        if(db4ai_is_xgboost(candidate->algorithm))
        {
            Size input=(Size)data.nrows*(data.nfeatures+1)*sizeof(float);
            if(input>(Size)s->options.memory_mb*1024*1024/4)
                ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("candidate dense input exceeds AutoML admission budget")));
            db4ai_xgb_train(&data,candidate->algorithm,&options,"ubj",&model);
            runtime.xgb=db4ai_xgb_load(&model);
        }
        else
        {
            runtime.native=db4ai_native_train(&data,candidate->algorithm,&options);
            model.data=db4ai_native_save(runtime.native); model.format="json"; model.noutputs=runtime.native->noutputs;
        }
        evaluation=evaluate(&runtime,&s->input->valid,s->metric,false);
        elapsed=db4ai_automl_now()-start;
        MemoryContextSwitchTo(s->context);
        if(s->records.len) appendStringInfoChar(&s->records,',');
        appendStringInfo(&s->records,"{\"trial\":%d,\"stage\":%d,\"algorithm\":",trial_index,stage);
        escape_json(&s->records,db4ai_algorithm_name(candidate->algorithm));
        appendStringInfo(&s->records,",\"status\":\"completed\",\"training_rows\":" INT64_FORMAT ",\"elapsed_ms\":%.6f,\"parameters\":",
                         data.nrows,elapsed*1000);
        appendStringInfoString(&s->records,JsonbToCString(NULL,&db4ai_options_json(&options,candidate->algorithm)->root,0));
        appendStringInfoString(&s->records,",\"validation\":"); write_evaluation(&s->records,&evaluation,runtime.classification); appendStringInfoChar(&s->records,'}');
        candidate->complete=true; candidate->objective=evaluation.objective;
        if(evaluation.objective<s->best_objective-1e-12 ||
           (fabs(evaluation.objective-s->best_objective)<=1e-12 &&
            (runtime.classification ? evaluation.log_loss<s->best_validation.log_loss-1e-12 :
                                      evaluation.mae<s->best_validation.mae-1e-12)))
        {
            MemoryContextReset(s->best_context);
            MemoryContextSwitchTo(s->best_context);
            s->best=model; s->best.data=DatumGetByteaPCopy(PointerGetDatum(model.data)); s->best.format=pstrdup(model.format);
            s->best_options=options;
            s->best_options.eval_metric=pstrdup(options.eval_metric); s->best_options.tree_method=pstrdup(options.tree_method); s->best_options.booster=pstrdup(options.booster);
            s->best_validation=evaluation; s->best_objective=evaluation.objective;
            s->best_fit_seconds=elapsed; s->best_rows=data.nrows;
        }
        db4ai_automl_soft_deadline(0);
        MemoryContextSwitchTo(outer); MemoryContextDelete(trial);
        ReleaseCurrentSubTransaction(); CurrentResourceOwner=owner;
        s->completed++; success=true;
    }
    PG_CATCH();
    {
        ErrorData *error;
        int code;
        MemoryContextSwitchTo(outer); error=CopyErrorData(); code=error->sqlerrcode; FlushErrorState();
        db4ai_automl_soft_deadline(0);
        /* Close PG-owned spill files before aborting their resource owner. */
        MemoryContextDelete(trial);
        RollbackAndReleaseCurrentSubTransaction(); MemoryContextSwitchTo(outer); CurrentResourceOwner=owner;
        if(code!=DB4AI_AUTOML_DEADLINE && code!=ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE &&
           code!=ERRCODE_PROGRAM_LIMIT_EXCEEDED && code!=ERRCODE_EXTERNAL_ROUTINE_EXCEPTION)
            ReThrowError(error); /* Includes user cancel / statement timeout / permissions. */
        if(s->records.len) appendStringInfoChar(&s->records,',');
        appendStringInfo(&s->records,"{\"trial\":%d,\"stage\":%d,\"algorithm\":",trial_index,stage);
        escape_json(&s->records,db4ai_algorithm_name(candidate->algorithm));
        appendStringInfo(&s->records,",\"status\":\"%s\",\"training_rows\":" INT64_FORMAT ",\"elapsed_ms\":%.6f,\"reason\":",
            code==DB4AI_AUTOML_DEADLINE?"budget_stopped":"failed",training->nrows,(db4ai_automl_now()-start)*1000);
        escape_json(&s->records,error->message); appendStringInfoChar(&s->records,'}');
        FreeErrorData(error);
        if(db4ai_automl_now()>=s->search_deadline) s->budget_stop=true;
    }
    PG_END_TRY();
    MemoryContextSwitchTo(outer);
    return success;
}

static void
add_candidate(Search *s,Db4aiAlgorithm a,double l2,double rate,int depth)
{
    Candidate *c=&s->candidates[s->ncandidates++];
    c->algorithm=a; c->l2=l2; c->rate=rate; c->depth=depth; c->objective=DBL_MAX;
}

static void
require_training_privilege(void)
{
    Oid types[5]={TEXTOID,TEXTOID,TEXTOID,JSONBOID,TEXTOID};
    Oid function=LookupFuncName(list_make2(makeString("db4ai"),makeString("train")),5,types,false);
    AclResult acl=object_aclcheck(ProcedureRelationId,function,GetUserId(),ACL_EXECUTE);
    if(acl!=ACLCHECK_OK) aclcheck_error(acl,OBJECT_FUNCTION,get_func_name(function));
}

static void
ensure_name_free(const char *name)
{
    Oid types[2]={TEXTOID,REGROLEOID}; Datum values[2]={CStringGetTextDatum(name),ObjectIdGetDatum(GetUserId())};
    if(SPI_connect()!=SPI_OK_CONNECT) elog(ERROR,"SPI_connect failed");
    if(SPI_execute_with_args("SELECT 1 FROM db4ai.pg_db4ai_models WHERE model_name OPERATOR(pg_catalog.=) $1 AND owner_oid OPERATOR(pg_catalog.=) $2",2,types,values,NULL,true,1)!=SPI_OK_SELECT)
        elog(ERROR,"model lookup failed");
    if(SPI_processed) ereport(ERROR,(errcode(ERRCODE_DUPLICATE_OBJECT),errmsg("model already exists: %s",name)));
    SPI_finish();
}

Datum
pg_db4ai_automl(PG_FUNCTION_ARGS)
{
    MemoryContext caller=CurrentMemoryContext;
    MemoryContext context;
    Search *s;
    char *name,*task,*query;
    bool classification;
    Jsonb *options;
    volatile Jsonb *result=NULL;
    bool parent_timer; TimestampTz parent_finish=0,own_finish;
    int guc;
    double started=db4ai_automl_now(),started_cpu=cpu_seconds();
    int i;
    for(i=0;i<4;i++) if(PG_ARGISNULL(i)) ereport(ERROR,(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),errmsg("AutoML arguments must not be NULL")));
    name=text_to_cstring(PG_GETARG_TEXT_PP(0)); task=text_to_cstring(PG_GETARG_TEXT_PP(1)); query=text_to_cstring(PG_GETARG_TEXT_PP(2)); options=PG_GETARG_JSONB_P(3);
    if(!*name || strlen(name)>128) ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("model name must contain 1..128 bytes")));
    if(strcmp(task,"classification") && strcmp(task,"regression"))
        ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("task must be classification or regression; numeric targets are not guessed")));
    classification=strcmp(task,"classification")==0;
    if(automl_active) ereport(ERROR,(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),errmsg("AutoML cannot be invoked recursively")));
    require_training_privilege();
    context=AllocSetContextCreate(caller,"DB4AI AutoML",ALLOCSET_DEFAULT_SIZES);
    MemoryContextSwitchTo(context);
    s=palloc0(sizeof(*s)); s->context=context; s->started=started; s->best_objective=DBL_MAX;
    s->best_context=AllocSetContextCreate(context,"DB4AI AutoML champion",ALLOCSET_DEFAULT_SIZES);
    parse_options(options,classification,&s->options); s->metric=s->options.metric;
    initStringInfo(&s->records);
    guc=NewGUCNestLevel();
    parent_timer=get_timeout_active(STATEMENT_TIMEOUT);
    if(parent_timer) parent_finish=get_timeout_finish_time(STATEMENT_TIMEOUT);
    own_finish=TimestampTzPlusMilliseconds(GetCurrentTimestamp(),Max(1,(int)ceil((s->options.seconds-(db4ai_automl_now()-started))*1000)));
    /* Backend statement timeout also covers blocked/expensive source SQL.
     * Do not extend any caller's earlier deadline. Native library calls can
     * defer interrupt handling; no sub-millisecond hard-preemption claim. */
    if(!parent_timer || own_finish<parent_finish) enable_timeout_at(STATEMENT_TIMEOUT,own_finish);
    automl_active=true;
    PG_TRY();
    {
        Db4aiData *pilot;
        Db4aiPipeline *p;
        Evaluation baseline,test={0},test_baseline={0};
        StringInfoData report;
        int order[5],norder=0,j;
        double prepared,remaining;
        bool qualifies=false;
        int64 model_id=0;
        char *memory=psprintf("%d",s->options.memory_mb*1024/8);
        (void)set_config_option("db4ai.work_mem",memory,PGC_USERSET,PGC_S_SESSION,GUC_ACTION_SAVE,true,0,false);
        (void)set_config_option("work_mem",memory,PGC_USERSET,PGC_S_SESSION,GUC_ACTION_SAVE,true,0,false);
        (void)set_config_option("max_parallel_workers_per_gather","0",PGC_USERSET,PGC_S_SESSION,GUC_ACTION_SAVE,true,0,false);
        (void)set_config_option("jit","off",PGC_USERSET,PGC_S_SESSION,GUC_ACTION_SAVE,true,0,false);
        ensure_name_free(name);
        s->input=db4ai_automl_input(query,classification,&s->options); p=s->input->pipeline;
        if(!strcmp(s->metric,"roc_auc") && p->nclasses!=2)
            ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("roc_auc is supported only for binary AutoML")));
        baseline=evaluate(p,&s->input->valid,s->metric,true);
        prepared=db4ai_automl_now()-started;
        /* Reserve final test evaluation and serialization. Search deadline is
         * measured from call start, not reset after data preparation. */
        s->search_deadline=started+s->options.seconds*0.8;
        add_candidate(s,classification?(p->nclasses==2?DB4AI_LOGISTIC:DB4AI_MULTICLASS):DB4AI_LINEAR,0.001,0.1,3);
#ifdef DB4AI_WITH_XGBOOST
        if(s->options.use_xgboost) add_candidate(s,classification?(p->nclasses==2?DB4AI_XGB_BINARY:DB4AI_XGB_MULTICLASS):DB4AI_XGB_SQUARED,1,0.1,3);
#endif
        add_candidate(s,classification?(p->nclasses==2?DB4AI_LOGISTIC:DB4AI_MULTICLASS):DB4AI_LINEAR,0.1,0.05,3);
#ifdef DB4AI_WITH_XGBOOST
        if(s->options.use_xgboost) add_candidate(s,classification?(p->nclasses==2?DB4AI_XGB_BINARY:DB4AI_XGB_MULTICLASS):DB4AI_XGB_SQUARED,3,0.08,6);
#endif
        if(s->options.include_nb) add_candidate(s,DB4AI_GAUSSIAN_NB,0,0,0);
        pilot=pilot_data(s);
        for(i=0;i<s->ncandidates && s->trials<s->options.max_trials;i++)
        {
            remaining=s->search_deadline-db4ai_automl_now();
            if(remaining<=0.002) { s->budget_stop=true; break; }
            (void)run_candidate(s,&s->candidates[i],pilot,0,Max(0.001,remaining/(s->ncandidates-i+2)));
        }
        for(i=0;i<s->ncandidates;i++) if(s->candidates[i].complete && s->candidates[i].algorithm!=DB4AI_GAUSSIAN_NB) order[norder++]=i;
        for(i=0;i<norder;i++) for(j=i+1;j<norder;j++)
            if(s->candidates[order[j]].objective<s->candidates[order[i]].objective) { int t=order[i]; order[i]=order[j]; order[j]=t; }
        /* At most two finalist configurations. Preserve one other family when
         * available: small-sample rank is not proof that a family cannot learn. */
        if(norder>2 && db4ai_is_xgboost(s->candidates[order[0]].algorithm)==db4ai_is_xgboost(s->candidates[order[1]].algorithm))
            for(j=2;j<norder;j++) if(db4ai_is_xgboost(s->candidates[order[j]].algorithm)!=db4ai_is_xgboost(s->candidates[order[0]].algorithm))
            { int t=order[1]; order[1]=order[j]; order[j]=t; break; }
        for(i=0;i<Min(2,norder) && s->trials<s->options.max_trials;i++)
        {
            remaining=s->search_deadline-db4ai_automl_now();
            if(remaining<=0.002) { s->budget_stop=true; break; }
            (void)run_candidate(s,&s->candidates[order[i]],&s->input->train,1,remaining/(Min(2,norder)-i));
        }
        db4ai_automl_soft_deadline(0);
        if(s->completed)
        {
            Db4aiModel wrapper={0};
            wrapper.algorithm=DB4AI_AUTOML; wrapper.nfeatures=p->nraw; wrapper.noutputs=classification?p->nclasses:1;
            wrapper.format="json"; wrapper.data=db4ai_pipeline_save(p,&s->best);
            p=db4ai_pipeline_load(&wrapper);
            /* Test is observed exactly once, after selecting the immutable
             * champion. It does not select, calibrate, refit or gate a model. */
            test=evaluate(p,&s->input->test,s->metric,false);
            test_baseline=evaluate(s->input->pipeline,&s->input->test,s->metric,true);
            qualifies=s->best_objective+s->options.min_improvement<baseline.objective-1e-12;
        }
        initStringInfo(&report);
        appendStringInfo(&report,"{\"status\":\"%s\",\"task\":",!s->completed?"no_completed_candidate":(qualifies?"ready":"no_validation_improvement"));
        escape_json(&report,task); appendStringInfoString(&report,",\"metric\":"); escape_json(&report,s->metric);
        appendStringInfo(&report,",\"source_rows\":%d,\"train_rows\":%d,\"validation_rows\":%d,\"test_rows\":%d,"
            "\"selected_training_rows\":" INT64_FORMAT ",\"raw_features\":%d,\"encoded_features\":%d,\"source_executions\":1,"
            "\"input_bytes\":" UINT64_FORMAT ",\"input_spilled\":%s,\"split\":",
            s->input->nrows,s->input->ntrain,s->input->nvalid,s->input->ntest,s->best_rows,p->nraw,p->nencoded,
            s->input->input_bytes,s->input->spilled?"true":"false");
        escape_json(&report,s->options.split);
        appendStringInfo(&report,",\"split_fingerprint\":\"%016llx\",\"seed\":%d,\"trial_count\":%d,\"completed_trials\":%d,"
            "\"budget_stopped\":%s,\"budget_seconds\":%.6f,\"preparation_ms\":%.6f,\"search_and_finalization_ms\":%.6f,"
            "\"elapsed_ms_before_publish\":%.6f,\"cpu_seconds_before_publish\":%.6f,\"threads\":1,"
            "\"log_loss_clip\":1e-15,\"validation_used_for_selection\":true,\"test_used_for_selection\":false,"
            "\"refit\":\"none; retained training-partition candidate\",\"memory_budget_scope\":\"admission and individual working buffers, not backend RSS hard cap\","
            "\"selected_algorithm\":",
            (unsigned long long)s->input->split_hash,s->options.seed,s->trials,s->completed,s->budget_stop?"true":"false",s->options.seconds,
            prepared*1000,(db4ai_automl_now()-started-prepared)*1000,(db4ai_automl_now()-started)*1000,cpu_seconds()-started_cpu);
        if(s->completed) escape_json(&report,db4ai_algorithm_name(s->best.algorithm)); else appendStringInfoString(&report,"null");
        appendStringInfo(&report,",\"encoded_bytes\":" UINT64_FORMAT ",\"encoded_spilled\":%s",
            (uint64)s->input->nrows*(p->nencoded+1)*sizeof(double),
            (s->input->train.spilled || s->input->valid.spilled || s->input->test.spilled)?"true":"false");
        appendStringInfoString(&report,",\"class_order\":[");
        if(classification) for(i=0;i<p->nclasses;i++)
        { if(i) appendStringInfoChar(&report,','); escape_json(&report,db4ai_dictionary_text(p->labels,i)); }
        appendStringInfoString(&report,"],\"probabilities_calibrated\":false,\"candidate_limit_semantics\":\"upper bound; finite portfolio may finish earlier\"");
        {
            uint64 total_read=0,total_write=0,total_seek=0;
            Db4aiData *parts[3]={&s->input->train,&s->input->valid,&s->input->test};
            for(i=0;i<3;i++)
            {
                uint64 rd,wr,sk; db4ai_rows_stats(parts[i]->dense,&rd,&wr,&sk);
                total_read+=rd; total_write+=wr; total_seek+=sk;
            }
            appendStringInfo(&report,",\"replay_logical_bytes_read\":" UINT64_FORMAT ",\"replay_logical_bytes_written\":" UINT64_FORMAT ",\"replay_seeks\":" UINT64_FORMAT,
                             total_read,total_write,total_seek);
        }
        appendStringInfoString(&report,",\"selected_parameters\":");
        if(s->completed) appendStringInfoString(&report,JsonbToCString(NULL,&db4ai_options_json(&s->best_options,s->best.algorithm)->root,0));
        else appendStringInfoString(&report,"null");
        appendStringInfo(&report,",\"stop_reason\":\"%s\"",s->budget_stop?"search_budget":(s->trials>=s->options.max_trials?"trial_limit":"portfolio_exhausted"));
#ifdef DB4AI_WITH_XGBOOST
        appendStringInfoString(&report,",\"xgboost_available\":true");
#else
        appendStringInfoString(&report,",\"xgboost_available\":false");
#endif
        appendStringInfoString(&report,",\"validation_baseline\":"); write_evaluation(&report,&baseline,classification);
        appendStringInfoString(&report,",\"validation\":");
        if(s->completed) write_evaluation(&report,&s->best_validation,classification); else appendStringInfoString(&report,"null");
        appendStringInfoString(&report,",\"test\":");
        if(s->completed) write_evaluation(&report,&test,classification); else appendStringInfoString(&report,"null");
        appendStringInfoString(&report,",\"test_baseline\":");
        if(s->completed) write_evaluation(&report,&test_baseline,classification); else appendStringInfoString(&report,"null");
        appendStringInfo(&report,",\"trials\":[%s]}",s->records.data);
        if(qualifies)
        {
            Db4aiModel model={0}; Db4aiData metadata={0};
            model.algorithm=DB4AI_AUTOML; model.nfeatures=p->nraw; model.noutputs=classification?p->nclasses:1;
            model.format="json"; model.data=db4ai_pipeline_save(s->input->pipeline,&s->best);
            model.metrics=DatumGetJsonbP(DirectFunctionCall1(jsonb_in,CStringGetDatum(report.data)));
            metadata.nfeatures=p->nraw; metadata.nrows=s->best_rows;
            metadata.feature_names=palloc(sizeof(char*)*p->nraw);
            for(i=0;i<p->nraw;i++) metadata.feature_names[i]=p->features[i].name;
            model_id=db4ai_model_store(name,query,&metadata,&model,options,GetUserId());
        }
        report.len--; report.data[report.len]='\0';
        appendStringInfo(&report,",\"model_id\":%s,\"elapsed_ms\":%.6f,\"cpu_seconds\":%.6f}",model_id?psprintf(INT64_FORMAT,model_id):"null",
                         (db4ai_automl_now()-started)*1000,cpu_seconds()-started_cpu);
        MemoryContextSwitchTo(caller);
        result=DatumGetJsonbPCopy(DirectFunctionCall1(jsonb_in,CStringGetDatum(report.data)));
        db4ai_automl_soft_deadline(0);
        AtEOXact_GUC(true,guc);
        CHECK_FOR_INTERRUPTS();
        if(parent_timer) enable_timeout_at(STATEMENT_TIMEOUT,parent_finish);
        else disable_timeout(STATEMENT_TIMEOUT,false);
        MemoryContextDelete(context);
        automl_active=false;
    }
    PG_CATCH();
    {
        db4ai_automl_soft_deadline(0); automl_active=false;
        MemoryContextSwitchTo(caller); AtEOXact_GUC(false,guc);
        if(parent_timer) enable_timeout_at(STATEMENT_TIMEOUT,parent_finish);
        else disable_timeout(STATEMENT_TIMEOUT,false);
        MemoryContextDelete(context); PG_RE_THROW();
    }
    PG_END_TRY();
    PG_RETURN_JSONB_P((Jsonb *)result);
}
