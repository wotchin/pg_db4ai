/*
 * Copyright (c) 2020 Huawei Technologies Co.,Ltd.
 * PostgreSQL/C adaptation: pg_db4ai contributors, 2026.
 * SPDX-License-Identifier: MulanPSL-2.0
 *
 * The original bayes_net_internal is an equality-selectivity / joint-probability
 * estimator, not a Naive Bayes classifier. Preserve its P(X|Y)P(Y)/P(X)
 * message propagation, MCV/bin distinct corrections and Gaussian parents.
 * Explicit, typed JSON replaces C++ objects, Datum/OID dumps and Boost.
 */
#include "postgres.h"
#include "db4ai.h"
#include <math.h>
#include <float.h>
#include <ctype.h>
#include <errno.h>
#include "catalog/pg_type_d.h"
#include "miscadmin.h"
#include "utils/builtins.h"
#include "utils/fmgrprotos.h"
#include "utils/lsyscache.h"
#include "utils/json.h"

typedef struct BayesVariable
{
    int mode;                 /* 0 full dictionary, 1 MCV + other, 2 numeric bins */
    Db4aiDictionary *dict;     /* for bins: lowers, explicit maximum, optional NULL */
    FmgrInfo compare;
    int states;               /* dictionary count plus unknown / other */
    int *map;                 /* transient original dictionary ID -> state */
    double *coeff;
    double *counts;
    double *marginal;
} BayesVariable;

typedef struct BayesEdge
{
    int parent, child;
    bool continuous;
    double *prob;             /* discrete: child states * parent states; normal: *5 */
    double *min, *max;        /* transient Gaussian accumulators */
    uint64 *bits;             /* 128 occupancy buckets, two uint64 per child state */
} BayesEdge;

struct Db4aiBayes
{
    int n, nedges;
    int order[64];            /* DFS order matching original direct prediction */
    bool children[64];
    BayesVariable *variables;
    BayesEdge *edges;
    Size allocated;
};

typedef struct SortValues { Db4aiDictionary *d; FmgrInfo *cmp; bool frequency; } SortValues;

static void *
bayes_alloc(Db4aiBayes *b, Size count, Size width)
{
    Size size;
    if (count > MaxAllocSize / width)
        ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("Bayes state size overflow")));
    size=count*width;
    if (size > (Size)db4ai_max_model_mb*1024*1024-b->allocated)
        ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("Bayes working state exceeds db4ai.max_model_mb"),
                       errhint("Reduce num_mcvs / num_bins, graph edges, or categorical cardinality.")));
    b->allocated+=size;
    return palloc0(Max(size,(Size)1));
}

static int
value_compare(const void *av, const void *bv, void *arg)
{
    int a=*(const int*)av, b=*(const int*)bv;
    SortValues *s=(SortValues*)arg;
    if(s->frequency && s->d->frequency[a]!=s->d->frequency[b])
        return s->d->frequency[a]>s->d->frequency[b]?-1:1;
    if(s->d->nulls[a] || s->d->nulls[b])
        return s->d->nulls[a]==s->d->nulls[b] ? 0 : (s->d->nulls[a]?1:-1);
    if(s->cmp)
        return DatumGetInt32(FunctionCall2Coll(s->cmp,s->d->collation,s->d->values[a],s->d->values[b]));
    return a>b?1:(a<b?-1:0); /* stable first-occurrence tie break for unsortable types */
}

static bool
number_type(Oid t)
{
    return t==INT2OID || t==INT4OID || t==INT8OID || t==FLOAT4OID || t==FLOAT8OID || t==NUMERICOID;
}

static double
float_value(Db4aiDictionary *d, int i)
{
    double v=d->type==FLOAT4OID ? DatumGetFloat4(d->values[i]) : DatumGetFloat8(d->values[i]);
    db4ai_check_finite(v);
    return v;
}

/* Haas-Stokes Duj1, as used by original DB4AI. Integer truncation is retained. */
static double
distinct_estimate(double n, double N, double d, double singletons)
{
    double result;
    if(n<1 || d<1) return 1;
    if(n>=N) return Max(1,d);
    singletons=Min(singletons,n);
    result=n*d/(n-singletons+singletons*n/N);
    db4ai_check_finite(result);
    return Max(1,floor(result));
}

static int
variable_index(BayesVariable *v, Datum value, bool isnull)
{
    int id=db4ai_dictionary_index(v->dict,value,isnull,false);
    int left, right, middle, last;
    if(v->mode!=2 || id>=0) return id>=0?id:v->dict->count;
    if(isnull) return v->dict->count;
    last=v->dict->count-1;
    if(v->dict->nulls[last]) last--;
    if(last<0 || DatumGetInt32(FunctionCall2Coll(&v->compare,v->dict->collation,value,v->dict->values[0]))<0 ||
       DatumGetInt32(FunctionCall2Coll(&v->compare,v->dict->collation,value,v->dict->values[last]))>0)
        return v->dict->count;
    left=0; right=last;
    while(left+1<right)
    {
        middle=left+(right-left)/2;
        if(DatumGetInt32(FunctionCall2Coll(&v->compare,v->dict->collation,value,v->dict->values[middle]))<0) right=middle;
        else left=middle;
    }
    return left;
}

static void
variable_prepare(Db4aiBayes *b, BayesVariable *v, Db4aiDictionary *full,
                 const Db4aiOptions *o, int64 rows)
{
    int i, j, k, nullid=-1, nonnull=full->count, *sorted, *ndistinct, *singletons;
    int64 nullrows=0, maxfreq=0, cum=0, bucket_size;
    TypeCacheEntry *tc=lookup_type_cache(full->type,TYPECACHE_CMP_PROC_FINFO);
    SortValues sort;
    double sample=o->num_sample_rows?o->num_sample_rows:rows;
    double total=o->num_total_rows?o->num_total_rows:sample;
    sort.d=full; sort.cmp=NULL; sort.frequency=false;
    if(OidIsValid(tc->cmp_proc_finfo.fn_oid))
    { fmgr_info_copy(&v->compare,&tc->cmp_proc_finfo,CurrentMemoryContext); sort.cmp=&v->compare; }
    sorted=bayes_alloc(b,full->count,sizeof(int));
    v->map=bayes_alloc(b,full->count,sizeof(int));
    for(i=0;i<full->count;i++)
    {
        sorted[i]=i;
        if(full->nulls[i]) { nullid=i; nullrows=full->frequency[i]; nonnull--; }
        else maxfreq=Max(maxfreq,full->frequency[i]);
        if(full->type==FLOAT4OID || full->type==FLOAT8OID)
            if(!full->nulls[i]) (void)float_value(full,i);
    }
    v->dict=db4ai_dictionary_new(full->type,full->typmod,full->collation,65536);
    if(o->num_bins>0 && number_type(full->type) && full->count>o->num_bins &&
       maxfreq<(rows-nullrows)/o->num_bins)
    {
        v->mode=2;
        qsort_arg(sorted,full->count,sizeof(int),value_compare,&sort);
        bucket_size=(int64)ceil((double)(rows-nullrows)/o->num_bins);
        for(i=0;i<nonnull;i++)
        {
            if(cum/bucket_size>=v->dict->count)
                db4ai_dictionary_index(v->dict,full->values[sorted[i]],false,true);
            cum+=full->frequency[sorted[i]];
        }
        db4ai_dictionary_index(v->dict,full->values[sorted[nonnull-1]],false,true);
        if(nullid>=0) db4ai_dictionary_index(v->dict,0,true,true);
    }
    else if(o->num_mcvs>0 && full->count>o->num_mcvs)
    {
        v->mode=1; sort.frequency=true;
        qsort_arg(sorted,full->count,sizeof(int),value_compare,&sort);
        for(i=0;i<o->num_mcvs;i++)
            db4ai_dictionary_index(v->dict,full->values[sorted[i]],full->nulls[sorted[i]],true);
    }
    else
    {
        v->mode=0;
        for(i=0;i<full->count;i++) db4ai_dictionary_index(v->dict,full->values[i],full->nulls[i],true);
    }
    v->states=v->dict->count+1;
    v->coeff=bayes_alloc(b,v->states,sizeof(double));
    v->counts=bayes_alloc(b,v->states,sizeof(double));
    v->marginal=bayes_alloc(b,v->states,sizeof(double));
    ndistinct=bayes_alloc(b,v->states,sizeof(int)); singletons=bayes_alloc(b,v->states,sizeof(int));
    for(i=0;i<full->count;i++)
    {
        j=variable_index(v,full->values[i],full->nulls[i]);
        v->map[i]=j; v->counts[j]+=full->frequency[i]; ndistinct[j]++;
        if(full->frequency[i]==1) singletons[j]++;
    }
    for(j=0;j<v->states;j++)
    {
        int distinct=ndistinct[j];
        if(!distinct) { v->coeff[j]=0; continue; }
        if(v->mode==1 && j==v->dict->count)
        {
            double n=sample-(rows-v->counts[j]);
            v->coeff[j]=1.0/distinct_estimate(n,total*n/sample,distinct,singletons[j]);
        }
        else if(v->mode==2 && j<v->dict->count && !v->dict->nulls[j])
        {
            k=v->dict->count-1; if(v->dict->nulls[k]) k--;
            v->coeff[j]=j==k?1.0:1.0/distinct_estimate(sample/o->num_bins,total/o->num_bins,distinct,singletons[j]);
        }
        else v->coeff[j]=1;
        v->marginal[j]=v->counts[j]/rows*v->coeff[j];
    }
    pfree(ndistinct); pfree(singletons);
    pfree(sorted);
}

static void
dfs(Db4aiBayes *b, int node, int *colors, int *pos)
{
    int parent,e;
    if(colors[node]==1) ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("Bayes graph contains a cycle")));
    if(colors[node]==2) return;
    colors[node]=1;
    for(parent=0;parent<b->n;parent++)
        for(e=0;e<b->nedges;e++) if(b->edges[e].child==node && b->edges[e].parent==parent) dfs(b,parent,colors,pos);
    colors[node]=2; b->order[(*pos)++]=node;
}

static void
graph_order(Db4aiBayes *b)
{
    int colors[64]={0}, pos=0, i,j;
    for(i=0;i<b->nedges;i++)
    {
        BayesEdge *e=&b->edges[i];
        if(e->parent<0 || e->parent>=b->n || e->child<0 || e->child>=b->n || e->child==e->parent)
            ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("invalid Bayes edge")));
        for(j=0;j<i;j++) if(b->edges[j].parent==e->parent && b->edges[j].child==e->child)
            ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("duplicate Bayes edge")));
        b->children[e->parent]=true;
    }
    for(i=0;i<b->n;i++) dfs(b,i,colors,&pos);
}

static void
parse_graph(Db4aiBayes *b, const Db4aiOptions *o)
{
    const char *s=o->edges?o->edges:"";
    int ids[4032], count=0,i;
    while(*s)
    {
        char *end; long id;
        while(isspace((unsigned char)*s)) s++;
        if(!*s) break;
        errno=0; id=strtol(s,&end,10);
        if(end==s || errno || id<0 || id>=b->n || count>=lengthof(ids))
            ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("invalid edges_of_network"),errhint("Use comma-separated zero-based parent,child pairs.")));
        ids[count++]=(int)id; s=end;
        while(isspace((unsigned char)*s)) s++;
        if(*s && *s!=',') ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("expected comma in edges_of_network")));
        if(*s==',') { s++; while(isspace((unsigned char)*s)) s++; if(!*s) ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("trailing comma in edges_of_network"))); }
    }
    if(count%2 || (o->num_edges>=0 && count/2!=o->num_edges))
        ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("num_edges does not match parent,child pairs")));
    b->nedges=count/2; b->edges=bayes_alloc(b,b->nedges,sizeof(BayesEdge));
    for(i=0;i<b->nedges;i++) { b->edges[i].parent=ids[2*i]; b->edges[i].child=ids[2*i+1]; }
    graph_order(b);
}

static double
expected_distinct(double rows,double possibilities)
{
    double small=Min(rows,possibilities), big=Max(rows,possibilities);
    if(small<=1) return Max(1,small);
    return small * -expm1(big*log1p(-1.0/small));
}

static void
bayes_serialize(Db4aiBayes *b, Db4aiModel *model)
{
    StringInfoData s; int i;
    initStringInfo(&s);
    appendStringInfo(&s,"{\"format_version\":1,\"algorithm\":\"bayes_net_internal\",\"nfeatures\":%d,\"variables\":[",b->n);
    for(i=0;i<b->n;i++)
    {
        BayesVariable *v=&b->variables[i];
        if(i) appendStringInfoChar(&s,',');
        appendStringInfo(&s,"{\"mode\":%d,\"dictionary\":",v->mode); db4ai_dictionary_save(&s,v->dict);
        appendStringInfoString(&s,",\"coefficients\":"); db4ai_write_array(&s,v->coeff,v->states);
        appendStringInfoString(&s,",\"marginal\":"); db4ai_write_array(&s,v->marginal,v->states); appendStringInfoChar(&s,'}');
    }
    appendStringInfoString(&s,"],\"edges\":[");
    for(i=0;i<b->nedges;i++)
    {
        BayesEdge *e=&b->edges[i];
        int len=b->variables[e->child].states*(e->continuous?5:b->variables[e->parent].states);
        if(i) appendStringInfoChar(&s,',');
        appendStringInfo(&s,"{\"parent\":%d,\"child\":%d,\"continuous\":%s,\"parameters\":",e->parent,e->child,e->continuous?"true":"false");
        db4ai_write_array(&s,e->prob,len); appendStringInfoChar(&s,'}');
    }
    appendStringInfoString(&s,"]}"); model->data=db4ai_bytes(s.data,s.len); model->format="json"; model->noutputs=1;
}

void
db4ai_bayes_train(Db4aiData *data, const Db4aiOptions *o, Db4aiModel *model)
{
    Db4aiBayes *b=palloc0(sizeof(*b));
    int i,j,k,e, pass;
    double *row=palloc(sizeof(double)*data->ncolumns);
    double ratio=o->num_total_rows ? (double)o->num_total_rows/(o->num_sample_rows?o->num_sample_rows:data->nrows):1;
    StringInfoData metrics;
    b->n=data->ncolumns;
    if(b->n<1 || b->n>64 || (o->num_nodes>0 && o->num_nodes!=b->n))
        ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("Bayes num_nodes must equal the input column count (1..64)")));
    if((o->num_sample_rows>0 && o->num_sample_rows!=data->nrows) || ratio<1)
        ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("Bayes num_sample_rows must equal input rows and num_total_rows must not be smaller")));
    parse_graph(b,o);
    b->variables=bayes_alloc(b,b->n,sizeof(BayesVariable));
    for(i=0;i<b->n;i++) variable_prepare(b,&b->variables[i],data->domains[i],o,data->nrows);
    for(e=0;e<b->nedges;e++)
    {
        BayesEdge *edge=&b->edges[e];
        Oid pt=data->domains[edge->parent]->type;
        int nt=b->variables[edge->child].states, np=b->variables[edge->parent].states;
        edge->continuous=pt==FLOAT4OID || pt==FLOAT8OID;
        edge->prob=bayes_alloc(b,nt, sizeof(double)*(edge->continuous?5:np));
        if(edge->continuous)
        {
            edge->min=bayes_alloc(b,nt,sizeof(double)); edge->max=bayes_alloc(b,nt,sizeof(double));
            edge->bits=bayes_alloc(b,nt,2*sizeof(uint64));
            for(i=0;i<nt;i++) { edge->min[i]=DBL_MAX; edge->max[i]=-DBL_MAX; }
        }
    }
    for(pass=0;pass<2;pass++)
    {
        db4ai_data_rewind(data);
        while(db4ai_data_next(data,row))
        {
            CHECK_FOR_INTERRUPTS();
            for(e=0;e<b->nedges;e++)
            {
                BayesEdge *edge=&b->edges[e];
                int source=(int)row[edge->parent];
                i=b->variables[edge->child].map[(int)row[edge->child]];
                j=b->variables[edge->parent].map[source];
                if(!edge->continuous)
                { if(!pass) edge->prob[i*b->variables[edge->parent].states+j]++; }
                else
                {
                    double *p=edge->prob+5*i;
                    Db4aiDictionary *d=data->domains[edge->parent];
                    if(!pass)
                    {
                        p[3]++;
                        if(d->nulls[source]) p[2]++;
                        else
                        {
                            double x=float_value(d,source);
                            p[0]+=(x-p[0])/(p[3]-p[2]);
                            edge->min[i]=Min(edge->min[i],x); edge->max[i]=Max(edge->max[i],x);
                        }
                    }
                    else if(!d->nulls[source])
                    {
                        double x=float_value(d,source), diff=x-p[0];
                        int bin=edge->max[i]>edge->min[i] ? (int)((x-edge->min[i])/(edge->max[i]-edge->min[i])*127):0;
                        bin=Min(127,Max(0,bin));
                        p[1]+=diff*diff;
                        edge->bits[2*i+bin/64]|=UINT64CONST(1)<<(bin%64);
                    }
                }
            }
        }
    }
    for(e=0;e<b->nedges;e++)
    {
        BayesEdge *edge=&b->edges[e];
        BayesVariable *child=&b->variables[edge->child], *parent=&b->variables[edge->parent];
        for(i=0;i<child->states;i++)
        {
            if(edge->continuous)
            {
                double *p=edge->prob+5*i;
                int c=0; double distinct;
                for(k=0;k<2;k++) { uint64 bits=edge->bits[2*i+k]; while(bits) { bits&=bits-1; c++; } }
                if(c>=127) distinct=floor((127+p[3]-p[2])/2);
                else
                {
                    double coeff=1,sum=0;
                    for(j=c;j<=p[3] && coeff>=1e-3;j++) { sum+=j*coeff; coeff*=(double)c/128; }
                    distinct=Max(1,Min(p[3],floor(sum)));
                }
                p[1]=p[3]>0?sqrt(p[1]/p[3]):0;
                p[4]=p[3]>p[2] ? (edge->max[i]-edge->min[i])/Max(1,distinct):0;
            }
            else
                for(j=0;j<parent->states;j++)
                {
                    double *p=&edge->prob[i*parent->states+j];
                    if(*p>0 && child->coeff[i]>0 && parent->coeff[j]>0)
                        *p=*p/expected_distinct(*p*ratio,Max(1,1.0/(child->coeff[i]*parent->coeff[j])))/
                             (child->counts[i]*child->coeff[i]);
                    else *p=0;
                }
        }
    }
    bayes_serialize(b,model);
    initStringInfo(&metrics);
    appendStringInfo(&metrics,"{\"rows\":" INT64_FORMAT ",\"nodes\":%d,\"edges\":%d,\"estimator\":\"bayes_net_equality\",\"sample_ratio\":%.17g}",
                     data->nrows,b->n,b->nedges,ratio);
    model->metrics=DatumGetJsonbP(DirectFunctionCall1(jsonb_in,CStringGetDatum(metrics.data)));
}

static int
json_integer(Jsonb *json,const char *key,int low,int high)
{
    double v=db4ai_json_number(json,key,-1);
    if(v<low || v>high || floor(v)!=v)
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid Bayes model integer: %s",key)));
    return (int)v;
}

Db4aiBayes *
db4ai_bayes_load(const Db4aiModel *model)
{
    char *text=pnstrdup(VARDATA_ANY(model->data),VARSIZE_ANY_EXHDR(model->data));
    Jsonb *json=DatumGetJsonbP(DirectFunctionCall1(jsonb_in,CStringGetDatum(text)));
    Db4aiBayes *b=palloc0(sizeof(*b));
    JsonbValue *a=db4ai_json_field(json,"variables",true), v;
    JsonbIterator *it; JsonbIteratorToken token;
    int i=0,j;
    if(strcmp(model->format,"json")!=0 || json_integer(json,"format_version",1,1)!=1 ||
       strcmp(db4ai_json_string(json,"algorithm",""),"bayes_net_internal")!=0)
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("unsupported Bayes model format")));
    b->n=json_integer(json,"nfeatures",1,64);
    if(b->n!=model->nfeatures || a->type!=jbvBinary || !JsonContainerIsArray(a->val.binary.data) ||
       JsonContainerSize(a->val.binary.data)!=b->n)
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid Bayes variables")));
    b->variables=bayes_alloc(b,b->n,sizeof(BayesVariable));
    it=JsonbIteratorInit(a->val.binary.data);
    while((token=JsonbIteratorNext(&it,&v,true))!=WJB_DONE)
        if(token==WJB_ELEM)
        {
            BayesVariable *var=&b->variables[i++]; Jsonb *obj=JsonbValueToJsonb(&v);
            JsonbValue *dict=db4ai_json_field(obj,"dictionary",true);
            var->mode=json_integer(obj,"mode",0,2);
            var->dict=db4ai_dictionary_load(JsonbValueToJsonb(dict)); var->states=var->dict->count+1;
            if(var->dict->count<1) ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("empty Bayes dictionary")));
            var->coeff=db4ai_read_array(obj,"coefficients",var->states);
            var->marginal=db4ai_read_array(obj,"marginal",var->states);
            for(j=0;j<var->states;j++)
                if(var->coeff[j]<0 || var->coeff[j]>1 || var->marginal[j]<0 || var->marginal[j]>1)
                    ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid Bayes probability")));
            if(var->mode==2)
            {
                TypeCacheEntry *tc=lookup_type_cache(var->dict->type,TYPECACHE_CMP_PROC_FINFO);
                if(!number_type(var->dict->type) || !OidIsValid(tc->cmp_proc_finfo.fn_oid))
                    ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid Bayes bin type")));
                fmgr_info_copy(&var->compare,&tc->cmp_proc_finfo,CurrentMemoryContext);
                for(j=1;j<var->dict->count;j++)
                    if(!var->dict->nulls[j] && (var->dict->nulls[j-1] ||
                        DatumGetInt32(FunctionCall2Coll(&var->compare,var->dict->collation,var->dict->values[j-1],var->dict->values[j]))>=0))
                        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("unsorted Bayes bins")));
            }
        }
    a=db4ai_json_field(json,"edges",true);
    if(a->type!=jbvBinary || !JsonContainerIsArray(a->val.binary.data) || JsonContainerSize(a->val.binary.data)>2016)
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid Bayes edges")));
    b->nedges=JsonContainerSize(a->val.binary.data); b->edges=bayes_alloc(b,b->nedges,sizeof(BayesEdge));
    it=JsonbIteratorInit(a->val.binary.data); i=0;
    while((token=JsonbIteratorNext(&it,&v,true))!=WJB_DONE)
        if(token==WJB_ELEM)
        {
            BayesEdge *e=&b->edges[i++]; Jsonb *obj=JsonbValueToJsonb(&v); Oid t; Size count;
            e->parent=json_integer(obj,"parent",0,b->n-1); e->child=json_integer(obj,"child",0,b->n-1);
            e->continuous=db4ai_json_boolean(obj,"continuous",false); t=b->variables[e->parent].dict->type;
            if(e->continuous!=(t==FLOAT4OID || t==FLOAT8OID))
                ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid Bayes continuous parent")));
            count=(Size)b->variables[e->child].states*(e->continuous?5:b->variables[e->parent].states);
            if(count>MaxAllocSize/sizeof(double) || count>(Size)db4ai_max_model_mb*1024*1024/sizeof(double))
                ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("oversized Bayes conditional table")));
            e->prob=db4ai_read_array(obj,"parameters",(int)count);
            for(j=0;j<(int)count;j++)
                if((!e->continuous || j%5!=0) && e->prob[j]<0)
                    ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("negative Bayes conditional parameter")));
        }
    graph_order(b); return b;
}

double
db4ai_bayes_predict(Db4aiBayes *b, Datum *values, bool *nulls, Oid *types, int count)
{
    int states[64],p,i,j,e; double message[64],result=1;
    if(count!=b->n) ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("Bayes expects %d input variables",b->n)));
    for(i=0;i<b->n;i++)
    {
        if(!nulls[i] && getBaseType(types[i])!=b->variables[i].dict->type)
            ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("Bayes variable %d expects %s",i,format_type_be(b->variables[i].dict->type))));
        states[i]=variable_index(&b->variables[i],values[i],nulls[i]); message[i]=1;
    }
    for(p=0;p<b->n;p++)
    {
        i=b->order[p]; message[i]=b->variables[i].marginal[states[i]];
        for(j=0;j<b->n;j++) for(e=0;e<b->nedges;e++)
        {
            BayesEdge *edge=&b->edges[e]; double conditional,px;
            if(edge->child!=i || edge->parent!=j) continue;
            px=b->variables[j].marginal[states[j]];
            if(edge->continuous)
            {
                double *par=edge->prob+states[i]*5;
                if(nulls[j]) conditional=par[3]>0?par[2]/par[3]:0;
                else
                {
                    double x=getBaseType(types[j])==FLOAT4OID?DatumGetFloat4(values[j]):DatumGetFloat8(values[j]);
                    db4ai_check_finite(x);
                    if(par[1]==0) conditional=(x<=par[0] && par[0]<=x+par[4])?1:0;
                    else
                    {
                        double a=(x-par[0])/(par[1]*sqrt(2.0)),z=(x+par[4]-par[0])/(par[1]*sqrt(2.0));
                        conditional=a>0 ? .5*(erfc(a)-erfc(z)) : .5*(erfc(-z)-erfc(-a));
                    }
                }
            }
            else conditional=edge->prob[states[i]*b->variables[j].states+states[j]];
            message[i]*=px>0?conditional/px*message[j]:0;
        }
        /* Mark each consumed parent once, after all parent contributions. */
        for(e=0;e<b->nedges;e++) if(b->edges[e].child==i) message[b->edges[e].parent]=1;
        if(!b->children[i]) result*=message[i];
        db4ai_check_finite(result);
    }
    return result;
}
