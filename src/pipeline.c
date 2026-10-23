/* SPDX-License-Identifier: MulanPSL-2.0
 * Durable fitted preprocessing + one learner, owned by the ordinary model table.
 * Numeric arrays and named JSON inputs share exactly the same transform.
 */
#include "postgres.h"
#include "automl.h"
#include <math.h>
#include "catalog/pg_type_d.h"
#include "utils/builtins.h"
#include "utils/fmgrprotos.h"
#include "utils/json.h"
#include "utils/lsyscache.h"

static double
number(Datum value, Oid type)
{
    double v;
    switch (getBaseType(type))
    {
        case FLOAT8OID: v=DatumGetFloat8(value); break;
        case FLOAT4OID: v=DatumGetFloat4(value); break;
        case INT8OID: v=(double)DatumGetInt64(value); break;
        case INT4OID: v=DatumGetInt32(value); break;
        case INT2OID: v=DatumGetInt16(value); break;
        case NUMERICOID: v=DatumGetFloat8(DirectFunctionCall1(numeric_float8,value)); break;
        case BOOLOID: v=DatumGetBool(value)?1:0; break;
        default: ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("pipeline numeric feature has an incompatible type"))); v=0;
    }
    db4ai_check_finite(v);
    return v;
}

void
db4ai_pipeline_transform(const Db4aiPipeline *p, const Datum *values,
                          const bool *nulls, const Oid *types, double *out)
{
    int j;
    memset(out,0,sizeof(double)*p->nencoded);
    for(j=0;j<p->nraw;j++)
    {
        const Db4aiFeature *f=&p->features[j];
        if(f->categorical)
        {
            int k;
            if(getBaseType(types[j])!=f->dictionary->type)
                ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("pipeline feature %s has an incompatible categorical type",f->name)));
            k=nulls[j]?f->dictionary->count+1:db4ai_dictionary_index(f->dictionary,values[j],false,false);
            if(k<0) k=f->dictionary->count; /* Dedicated unseen/overflow bucket. */
            out[f->offset+k]=1;
        }
        else
        {
            double v=nulls[j]?f->mean:number(values[j],types[j]);
            out[f->offset]=(v-f->mean)/f->scale;
            db4ai_check_finite(out[f->offset]);
            if(p->missing_indicators) out[f->offset+1]=nulls[j]?1:0;
        }
    }
}

double *
db4ai_pipeline_array(const Db4aiPipeline *p, ArrayType *array)
{
    Datum *values; bool *nulls; int n,j;
    Oid *types;
    double *out;
    if(ARR_ELEMTYPE(array)!=FLOAT8OID || ARR_NDIM(array)!=1)
        ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("pipeline features must be a one-dimensional float8 array")));
    deconstruct_array(array,FLOAT8OID,8,FLOAT8PASSBYVAL,TYPALIGN_DOUBLE,&values,&nulls,&n);
    if(n!=p->nraw)
        ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("pipeline expects %d original features, got %d",p->nraw,n)));
    types=palloc(sizeof(Oid)*n); out=palloc(sizeof(double)*p->nencoded);
    for(j=0;j<n;j++)
    {
        if(p->features[j].categorical)
            ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("mixed-type pipeline requires db4ai.predict(model_name, jsonb)")));
        types[j]=FLOAT8OID;
    }
    db4ai_pipeline_transform(p,values,nulls,types,out);
    pfree(types); pfree(values); pfree(nulls);
    return out;
}

double *
db4ai_pipeline_json(const Db4aiPipeline *p, Jsonb *json)
{
    Datum values[DB4AI_AM_MAX_COLUMNS];
    bool nulls[DB4AI_AM_MAX_COLUMNS];
    Oid types[DB4AI_AM_MAX_COLUMNS];
    double *out=palloc(sizeof(double)*p->nencoded);
    int j;
    if(!JB_ROOT_IS_OBJECT(json))
        ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("prediction input must be a JSON object with named features")));
    for(j=0;j<p->nraw;j++)
    {
        const Db4aiFeature *f=&p->features[j];
        JsonbValue *v=db4ai_json_field(json,f->name,true);
        nulls[j]=v->type==jbvNull; values[j]=(Datum)0;
        types[j]=f->categorical?f->dictionary->type:FLOAT8OID;
        if(nulls[j]) continue;
        if(f->categorical)
        {
            char *text;
            if(v->type!=jbvString)
                ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("categorical feature %s must be a JSON string or null",f->name)));
            text=pnstrdup(v->val.string.val,v->val.string.len);
            values[j]=OidInputFunctionCall(f->input_function,text,f->input_parameter,f->dictionary->typmod);
            pfree(text);
        }
        else
        {
            double x;
            if(v->type==jbvNumeric) x=DatumGetFloat8(DirectFunctionCall1(numeric_float8,NumericGetDatum(v->val.numeric)));
            else if(v->type==jbvBool) x=v->val.boolean?1:0;
            else ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("numeric feature %s must be a JSON number, boolean or null",f->name)));
            db4ai_check_finite(x); values[j]=Float8GetDatum(x);
        }
    }
    db4ai_pipeline_transform(p,values,nulls,types,out);
    /* InputFunction allocations, if any, are in the caller's per-row context. */
    return out;
}

void
db4ai_pipeline_proba(const Db4aiPipeline *p,const double *x,double *out)
{
    int c;
    double sum=0;
    if(!p->classification) ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("regression pipeline has no class probabilities")));
    if(p->xgb)
    {
        if(p->nclasses==2)
        { out[1]=db4ai_xgb_predict(p->xgb,x,p->nencoded); out[0]=1-out[1]; }
        else db4ai_xgb_predict_vector(p->xgb,x,p->nencoded,out,p->nclasses);
    }
    else db4ai_native_scores(p->native,x,out);
    /* OVA logistic scores are normalized explicitly for this pipeline only;
     * the pre-existing predict_scores API for standalone OVA is unchanged. */
    for(c=0;c<p->nclasses;c++)
    {
        db4ai_check_finite(out[c]);
        if(out[c]<0 || out[c]>1+1e-6)
            ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid pipeline class probability")));
        sum+=out[c];
    }
    if(sum<=0) for(c=0;c<p->nclasses;c++) out[c]=1.0/p->nclasses;
    else for(c=0;c<p->nclasses;c++) out[c]/=sum;
}

int
db4ai_pipeline_class(const Db4aiPipeline *p,const double *x)
{
    double probabilities[256]; int c,best=0;
    db4ai_pipeline_proba(p,x,probabilities);
    for(c=1;c<p->nclasses;c++) if(probabilities[c]>probabilities[best]) best=c;
    return best;
}

double
db4ai_pipeline_value(const Db4aiPipeline *p,const double *x)
{
    double v;
    if(p->classification)
    {
        double probabilities[256];
        if(p->nclasses!=2) ereport(ERROR,(errcode(ERRCODE_DATATYPE_MISMATCH),errmsg("scalar probability requires a binary pipeline")));
        db4ai_pipeline_proba(p,x,probabilities); return probabilities[1];
    }
    v=p->xgb?db4ai_xgb_predict(p->xgb,x,p->nencoded):db4ai_native_predict(p->native,x);
    v=v*p->target_scale+p->target_mean; db4ai_check_finite(v); return v;
}

bytea *
db4ai_pipeline_save(const Db4aiPipeline *p,const Db4aiModel *child)
{
    StringInfoData s;
    int j;
    Size bytes=VARSIZE_ANY_EXHDR(child->data);
    char *hex;
    if(bytes > ((Size)db4ai_max_model_mb*1024*1024)/2)
        ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("learner exceeds pipeline model serialization budget")));
    hex=palloc(bytes*2+1); hex_encode(VARDATA_ANY(child->data),bytes,hex); hex[bytes*2]='\0';
    initStringInfo(&s);
    appendStringInfo(&s,"{\"pipeline_version\":1,\"algorithm\":\"automl\",\"classification\":%s,"
        "\"nraw\":%d,\"nencoded\":%d,\"nclasses\":%d,\"target_mean\":%.17g,\"target_scale\":%.17g,"
        "\"missing_indicators\":%s,\"features\":[",p->classification?"true":"false",p->nraw,p->nencoded,
        p->nclasses,p->target_mean,p->target_scale,p->missing_indicators?"true":"false");
    for(j=0;j<p->nraw;j++)
    {
        const Db4aiFeature *f=&p->features[j];
        if(j) appendStringInfoChar(&s,',');
        appendStringInfoString(&s,"{\"name\":"); escape_json(&s,f->name);
        appendStringInfo(&s,",\"categorical\":%s,\"offset\":%d,\"width\":%d,\"mean\":%.17g,\"scale\":%.17g,\"capped\":%s",
            f->categorical?"true":"false",f->offset,f->width,f->mean,f->scale,f->capped?"true":"false");
        if(f->categorical) { appendStringInfoString(&s,",\"dictionary\":"); db4ai_dictionary_save(&s,f->dictionary); }
        appendStringInfoChar(&s,'}');
    }
    appendStringInfoString(&s,"],\"labels\":");
    if(p->classification) db4ai_dictionary_save(&s,p->labels); else appendStringInfoString(&s,"null");
    appendStringInfoString(&s,",\"learner\":{\"algorithm\":"); escape_json(&s,db4ai_algorithm_name(child->algorithm));
    appendStringInfoString(&s,",\"format\":"); escape_json(&s,child->format);
    appendStringInfo(&s,",\"noutputs\":%d,\"data_hex\":",child->noutputs); escape_json(&s,hex);
    appendStringInfoString(&s,"}}"); pfree(hex);
    return db4ai_bytes(s.data,s.len);
}

static int
integer(Jsonb *j,const char *key,int low,int high)
{
    double v=db4ai_json_number(j,key,-1);
    if(v<low || v>high || floor(v)!=v)
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid pipeline %s",key)));
    return (int)v;
}

static Jsonb *
object(Jsonb *j,const char *key)
{
    JsonbValue *v=db4ai_json_field(j,key,true);
    if(v->type!=jbvBinary || !JsonContainerIsObject(v->val.binary.data))
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid pipeline object %s",key)));
    return JsonbValueToJsonb(v);
}

Db4aiPipeline *
db4ai_pipeline_load(const Db4aiModel *wrapper)
{
    Db4aiPipeline *p=palloc0(sizeof(*p));
    Jsonb *root,*learner;
    JsonbValue *items,v;
    JsonbIterator *it;
    JsonbIteratorToken token;
    int j=0,offset=0;
    char *hex;
    Size n;
    if(strcmp(wrapper->format,"json")!=0)
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("pipeline artifact must be JSON")));
    root=DatumGetJsonbP(DirectFunctionCall1(jsonb_in,CStringGetDatum(pnstrdup(VARDATA_ANY(wrapper->data),VARSIZE_ANY_EXHDR(wrapper->data)))));
    (void)integer(root,"pipeline_version",1,1);
    if(strcmp(db4ai_json_string(root,"algorithm",""),"automl"))
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid pipeline algorithm header")));
    p->classification=db4ai_json_boolean(root,"classification",false);
    p->nraw=integer(root,"nraw",1,DB4AI_AM_MAX_COLUMNS);
    p->nencoded=integer(root,"nencoded",1,DB4AI_MAX_FEATURES);
    p->nclasses=integer(root,"nclasses",p->classification?2:0,p->classification?256:0);
    p->target_mean=db4ai_json_number(root,"target_mean",0);
    p->target_scale=db4ai_json_number(root,"target_scale",0);
    if(p->target_scale<=0 || p->nraw!=wrapper->nfeatures || wrapper->noutputs!=(p->classification?p->nclasses:1))
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("pipeline metadata mismatch")));
    p->missing_indicators=db4ai_json_boolean(root,"missing_indicators",true);
    p->features=palloc0(sizeof(Db4aiFeature)*p->nraw);
    items=db4ai_json_field(root,"features",true);
    if(items->type!=jbvBinary || !JsonContainerIsArray(items->val.binary.data) ||
       JsonContainerSize(items->val.binary.data)!=p->nraw)
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid pipeline feature list")));
    it=JsonbIteratorInit(items->val.binary.data);
    while((token=JsonbIteratorNext(&it,&v,true))!=WJB_DONE)
    {
        Db4aiFeature *f; Jsonb *feature; int k;
        if(token!=WJB_ELEM) continue;
        if(v.type!=jbvBinary || !JsonContainerIsObject(v.val.binary.data) || j>=p->nraw)
            ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid pipeline feature")));
        f=&p->features[j]; feature=JsonbValueToJsonb(&v);
        f->name=db4ai_json_string(feature,"name",NULL);
        if(!f->name || !*f->name) ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("missing pipeline feature name")));
        for(k=0;k<j;k++) if(strcmp(p->features[k].name,f->name)==0)
            ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("duplicate pipeline feature name")));
        f->categorical=db4ai_json_boolean(feature,"categorical",false);
        f->offset=integer(feature,"offset",0,DB4AI_MAX_FEATURES-1);
        f->width=integer(feature,"width",1,DB4AI_MAX_FEATURES);
        f->mean=db4ai_json_number(feature,"mean",0); f->scale=db4ai_json_number(feature,"scale",0);
        if(f->scale<=0 || f->offset!=offset || offset+f->width>p->nencoded)
            ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid pipeline transform layout")));
        if(f->categorical)
        {
            f->dictionary=db4ai_dictionary_load(object(feature,"dictionary"));
            if(f->width!=f->dictionary->count+2)
                ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("categorical pipeline dimensions differ")));
            getTypeInputInfo(f->dictionary->type,&f->input_function,&f->input_parameter);
        }
        else if(f->width!=(p->missing_indicators?2:1))
            ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("numeric pipeline dimensions differ")));
        offset+=f->width; j++;
    }
    if(offset!=p->nencoded || j!=p->nraw) ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("pipeline dimensions differ")));
    if(p->classification)
    {
        p->labels=db4ai_dictionary_load(object(root,"labels"));
        if(p->labels->count!=p->nclasses) ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("pipeline class count mismatch")));
    }
    learner=object(root,"learner");
    p->child.algorithm=db4ai_algorithm(db4ai_json_string(learner,"algorithm",""));
    if(p->child.algorithm!=DB4AI_LINEAR && p->child.algorithm!=DB4AI_LOGISTIC && p->child.algorithm!=DB4AI_MULTICLASS &&
       p->child.algorithm!=DB4AI_GAUSSIAN_NB && p->child.algorithm!=DB4AI_XGB_SQUARED &&
       p->child.algorithm!=DB4AI_XGB_BINARY && p->child.algorithm!=DB4AI_XGB_MULTICLASS)
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("unsupported pipeline learner")));
    if(p->classification==(p->child.algorithm==DB4AI_LINEAR || p->child.algorithm==DB4AI_XGB_SQUARED))
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("pipeline task and learner differ")));
    p->child.nfeatures=p->nencoded;
    p->child.noutputs=integer(learner,"noutputs",1,256);
    if((p->child.algorithm==DB4AI_XGB_MULTICLASS && (p->nclasses<3 || p->child.noutputs!=p->nclasses)) ||
       (p->child.algorithm==DB4AI_XGB_BINARY && (p->nclasses!=2 || p->child.noutputs!=1)) ||
       (p->child.algorithm==DB4AI_LOGISTIC && p->nclasses!=2) ||
       (!p->classification && p->child.noutputs!=1))
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("pipeline learner output dimensions differ")));
    p->child.format=db4ai_json_string(learner,"format","");
    hex=db4ai_json_string(learner,"data_hex",""); n=strlen(hex);
    if(n==0 || n%2 || n/2>(Size)db4ai_max_model_mb*1024*1024)
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid pipeline learner payload")));
    p->child.data=palloc(VARHDRSZ+n/2); SET_VARSIZE(p->child.data,VARHDRSZ+n/2);
    if(hex_decode(hex,n,VARDATA(p->child.data))!=n/2)
        ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("invalid pipeline learner encoding")));
    if(db4ai_is_xgboost(p->child.algorithm)) p->xgb=db4ai_xgb_load(&p->child);
    else
    {
        p->native=db4ai_native_load(&p->child);
        if(p->child.algorithm==DB4AI_MULTICLASS && p->native->classifier!=DB4AI_LOGISTIC)
            ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("pipeline OVA learner must use logistic scores")));
        if(p->classification && (p->native->nclasses!=p->nclasses || !p->native->labels || p->native->labels->type!=p->labels->type))
            ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("pipeline learner labels mismatch")));
        if(p->classification)
            for(j=0;j<p->nclasses;j++)
                if(p->labels->nulls[j] || p->native->labels->nulls[j] ||
                   !DatumGetBool(FunctionCall2Coll(&p->labels->equal,p->labels->collation,p->labels->values[j],p->native->labels->values[j])))
                    ereport(ERROR,(errcode(ERRCODE_DATA_CORRUPTED),errmsg("pipeline learner class order mismatch")));
    }
    return p;
}
