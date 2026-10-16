/* SPDX-License-Identifier: MulanPSL-2.0 */
#include "postgres.h"
#include <math.h>
#include "db4ai.h"
#include "common/shortest_dec.h"
#include "lib/stringinfo.h"
#include "utils/builtins.h"
#include "utils/fmgrprotos.h"
#include "utils/json.h"

static const char *const names[] = {
    "linear_regression", "logistic_regression", "svm_classification", "kmeans", "pca",
    "xgboost_regression_squarederror", "xgboost_binary_logistic",
    "xgboost_regression_logistic", "xgboost_regression_gamma",
    "multiclass", "bayes_net_internal"
};

Db4aiAlgorithm
db4ai_algorithm(const char *name)
{
    int i;
    for (i = 0; i < lengthof(names); i++)
        if (strcmp(name, names[i]) == 0)
            return (Db4aiAlgorithm) i;
    ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                    errmsg("unsupported DB4AI algorithm: %s", name)));
    return DB4AI_LINEAR;
}

const char *
db4ai_algorithm_name(Db4aiAlgorithm algorithm)
{
    Assert(algorithm >= DB4AI_LINEAR && algorithm <= DB4AI_BAYES);
    return names[algorithm];
}

bool
db4ai_supervised(Db4aiAlgorithm algorithm)
{
    return algorithm != DB4AI_KMEANS && algorithm != DB4AI_PCA && algorithm != DB4AI_BAYES;
}

bool
db4ai_is_xgboost(Db4aiAlgorithm algorithm)
{
    return algorithm >= DB4AI_XGB_SQUARED && algorithm <= DB4AI_XGB_GAMMA;
}

bool
db4ai_classifier(Db4aiAlgorithm a)
{
    return a == DB4AI_LOGISTIC || a == DB4AI_SVM || a == DB4AI_MULTICLASS;
}

void
db4ai_check_finite(double x)
{
    if (!isfinite(x))
        ereport(ERROR, (errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
                        errmsg("DB4AI requires finite numbers"),
                        errhint("Scale features or lower the learning rate if training diverges.")));
}

JsonbValue *
db4ai_json_field(Jsonb *json, const char *key, bool required)
{
    JsonbValue k;
    JsonbValue *value;
    if (!JB_ROOT_IS_OBJECT(json))
        ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("expected a JSON object")));
    k.type = jbvString;
    k.val.string.val = (char *) key;
    k.val.string.len = strlen(key);
    value = findJsonbValueFromContainer(&json->root, JB_FOBJECT, &k);
    if (value == NULL && required)
        ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("missing JSON field: %s", key)));
    return value;
}

double
db4ai_json_number(Jsonb *json, const char *key, double def)
{
    JsonbValue *v = db4ai_json_field(json, key, false);
    double result;
    if (v == NULL)
        return def;
    if (v->type != jbvNumeric)
        ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("%s must be a number", key)));
    result = DatumGetFloat8(DirectFunctionCall1(numeric_float8, NumericGetDatum(v->val.numeric)));
    db4ai_check_finite(result);
    return result;
}

static double
bounded(Jsonb *json, const char *key, double def, double low, double high, bool integral)
{
    double v = db4ai_json_number(json, key, def);
    if (v < low || v > high || (integral && floor(v) != v))
        ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                        errmsg("invalid value for hyperparameter %s", key),
                        errdetail("Expected %s in [%g, %g].", integral ? "an integer" : "a number", low, high)));
    return v;
}

typedef struct OptionKey
{
    const char *key;
    unsigned int algorithms;
} OptionKey;
#define A(x) (1U << (x))
#define GD (A(DB4AI_LINEAR)|A(DB4AI_LOGISTIC)|A(DB4AI_SVM)|A(DB4AI_MULTICLASS))
#define SVM (A(DB4AI_SVM)|A(DB4AI_MULTICLASS))
#define XGB (A(DB4AI_XGB_SQUARED)|A(DB4AI_XGB_BINARY)|A(DB4AI_XGB_LOGISTIC)|A(DB4AI_XGB_GAMMA))
#define KM A(DB4AI_KMEANS)
#define PCA A(DB4AI_PCA)
#define BAYES A(DB4AI_BAYES)
#define ALL (GD|XGB|KM|PCA|BAYES)
static const OptionKey keys[] = {
    {"profile", ALL}, {"max_iterations",ALL & ~BAYES}, {"n_iterations",XGB},
    {"batch_size",ALL}, {"seed", ALL & ~BAYES}, {"verbose",ALL & ~BAYES},
    {"learning_rate",GD|XGB}, {"eta",XGB}, {"l2",GD|XGB},
    {"tolerance",GD|PCA|KM}, {"decay",GD}, {"optimizer",GD},
    {"max_seconds",GD|PCA|KM|XGB}, {"null_policy",GD|KM|PCA|XGB},
    {"shuffle",GD|PCA}, {"shuffle_first_epoch",GD|PCA}, {"shuffle_buffer_size",GD|PCA},
    {"kernel",SVM}, {"components",SVM}, {"gamma",SVM|XGB}, {"degree",SVM}, {"coef0",SVM},
    {"lambda",SVM|XGB}, {"classifier",A(DB4AI_MULTICLASS)},
    {"k",KM}, {"num_centroids",KM}, {"num_features",KM},
    {"seeding_function",KM}, {"distance_function",KM}, {"seeding_rounds",KM}, {"oversampling",KM},
    {"n_components",PCA}, {"number_components",PCA}, {"pca_solver",PCA}, {"center",PCA},
    {"max_depth",XGB}, {"min_child_weight",XGB}, {"subsample",XGB}, {"colsample_bytree",XGB},
    {"nthread",XGB}, {"verbosity",XGB}, {"booster",XGB}, {"tree_method",XGB}, {"eval_metric",XGB},
    {"edges_of_network",BAYES}, {"num_edges",BAYES}, {"num_nodes",BAYES},
    {"num_bins",BAYES}, {"num_mcvs",BAYES}, {"num_sample_rows",BAYES}, {"num_total_rows",BAYES}
};

char *
db4ai_json_string(Jsonb *json, const char *key, const char *def)
{
    JsonbValue *v = db4ai_json_field(json, key, false);
    if (v == NULL) return def ? pstrdup(def) : NULL;
    if (v->type != jbvString || memchr(v->val.string.val, 0, v->val.string.len))
        ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("%s must be a string", key)));
    return pnstrdup(v->val.string.val, v->val.string.len);
}

bool
db4ai_json_boolean(Jsonb *json, const char *key, bool def)
{
    JsonbValue *v = db4ai_json_field(json, key, false);
    if (v == NULL) return def;
    if (v->type != jbvBool)
        ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("%s must be boolean", key)));
    return v->val.boolean;
}

static const char *
aliased(Jsonb *json, const char *key, const char *alias)
{
    if (alias && db4ai_json_field(json, alias, false))
    {
        if (db4ai_json_field(json, key, false))
            ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                            errmsg("do not specify both %s and its alias %s",key,alias)));
        return alias;
    }
    return key;
}

static int
choice(Jsonb *json, const char *key, const char *def, const char *a, const char *b, const char *c, const char *d)
{
    char *s = db4ai_json_string(json,key,def);
    const char *values[4] = {a,b,c,d};
    int i;
    for (i=0;i<4;i++) if (values[i] && strcmp(s,values[i])==0) { pfree(s); return i; }
    ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("invalid %s: %s",key,s)));
    return 0;
}

void
db4ai_options(Jsonb *json, Db4aiAlgorithm algorithm, Db4aiOptions *o)
{
    JsonbIterator *it;
    JsonbIteratorToken token;
    JsonbValue v;
    bool xgb = db4ai_is_xgboost(algorithm);
    unsigned int bit=A(algorithm);
    bool legacy;
    memset(o,0,sizeof(*o));
    if (!JB_ROOT_IS_OBJECT(json))
        ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("hyperparameters must be a JSON object")));
    it=JsonbIteratorInit(&json->root);
    while ((token=JsonbIteratorNext(&it,&v,true))!=WJB_DONE)
        if (token==WJB_KEY)
        {
            int i; bool found=false;
            char *key=pnstrdup(v.val.string.val,v.val.string.len);
            for (i=0;i<lengthof(keys);i++)
                if (strcmp(key,keys[i].key)==0 && (keys[i].algorithms & bit)) { found=true; break; }
            if (!found) ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("unknown or inapplicable hyperparameter: %s",key)));
            pfree(key);
        }
    o->legacy_profile=choice(json,"profile","stable","stable","opengauss",NULL,NULL)==1;
    legacy=o->legacy_profile;
    o->iterations=bounded(json,aliased(json,"max_iterations",xgb?"n_iterations":NULL),(legacy && algorithm==DB4AI_KMEANS)?10:100,1,100000,true);
    o->batch_size=bounded(json,"batch_size",legacy?1000:128,1,1048576,true);
    o->clusters=bounded(json,aliased(json,"k","num_centroids"),legacy?1:8,1,256,true);
    o->components=bounded(json,aliased(json,"n_components","number_components"),legacy?1:2,1,256,true);
    o->seed=bounded(json,"seed",42,0,PG_INT32_MAX,true);
    o->max_depth=bounded(json,"max_depth",6,1,16,true);
    o->learning_rate=bounded(json,aliased(json,"learning_rate",xgb?"eta":NULL),xgb?0.1:(legacy?0.8:0.01),1e-12,1e6,false);
    o->l2=bounded(json,aliased(json,"l2",xgb?"lambda":NULL),(!legacy && algorithm==DB4AI_SVM)?0.01:0,0,1e6,false);
    o->tolerance=bounded(json,"tolerance",legacy?(algorithm==DB4AI_KMEANS?0.00001:0.0005):1e-8,0,1e6,false);
    o->min_child_weight=bounded(json,"min_child_weight",1,0,1e6,false);
    o->subsample=bounded(json,"subsample",1,1e-6,1,false);
    o->colsample_bytree=bounded(json,"colsample_bytree",1,1e-6,1,false);
    o->optimizer=choice(json,"optimizer","gd","gd","ngd",NULL,NULL);
    o->kernel=choice(json,"kernel","linear","linear","gaussian","polynomial",NULL);
    o->kernel_components=bounded(json,"components",0,0,4096,true);
    o->degree=bounded(json,"degree",2,2,9,true);
    o->gamma=bounded(json,"gamma",xgb?0:0.5,0,1e6,false);
    if (o->kernel==1 && o->gamma==0) ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("Gaussian gamma must be positive")));
    o->coef0=bounded(json,"coef0",1,0,1e6,false);
    o->decay=bounded(json,"decay",legacy?0.95:1,1e-12,1e6,false);
    o->svm_lambda=xgb?0:bounded(json,"lambda",legacy?0.01:0.5,1e-12,1e6,false);
    o->classifier=choice(json,"classifier","svm_classification","logistic_regression","svm_classification",NULL,NULL);
    o->shuffle_mode=choice(json,"shuffle","batch","none","batch","row",NULL);
    o->shuffle_first=db4ai_json_boolean(json,"shuffle_first_epoch",false);
    o->shuffle_buffer=bounded(json,"shuffle_buffer_size",128,1,1048576,true);
    o->seeding=choice(json,"seeding_function","Random++","Random++","KMeans||","reservoir",NULL);
    o->distance=choice(json,"distance_function","L2_Squared","L1","L2","L2_Squared","Linf");
    o->seeding_rounds=bounded(json,"seeding_rounds",5,1,10,true);
    o->oversampling=bounded(json,"oversampling",4,1,32,true);
    o->pca_solver=choice(json,"pca_solver",legacy?"stochastic":"jacobi","jacobi","stochastic",NULL,NULL);
    o->center=db4ai_json_boolean(json,"center",!legacy);
    o->null_policy=choice(json,"null_policy",legacy?"legacy":"error","error","legacy",NULL,NULL);
    o->max_seconds=bounded(json,"max_seconds",0,0,86400,true);
    v.type=jbvNull;
    {
        JsonbValue *vv=db4ai_json_field(json,"verbose",false);
        if (vv && vv->type==jbvNumeric) o->verbose=bounded(json,"verbose",0,0,2,true)!=0;
        else o->verbose=db4ai_json_boolean(json,"verbose",false);
    }
    o->num_features=bounded(json,"num_features",0,0,DB4AI_MAX_FEATURES,true);
    o->num_nodes=bounded(json,"num_nodes",0,0,64,true);
    o->num_edges=db4ai_json_field(json,"num_edges",false)?bounded(json,"num_edges",0,0,2016,true):-1;
    o->num_bins=bounded(json,"num_bins",0,0,4096,true);
    o->num_mcvs=bounded(json,"num_mcvs",0,0,4096,true);
    /* Restrict exactly representable integers rather than round an int64 through double. */
    o->num_sample_rows=(int64)bounded(json,"num_sample_rows",0,0,9007199254740991.0,true);
    o->num_total_rows=(int64)bounded(json,"num_total_rows",0,0,9007199254740991.0,true);
    o->edges=db4ai_json_string(json,"edges_of_network","");
    o->eval_metric=db4ai_json_string(json,"eval_metric","");
    o->tree_method=db4ai_json_string(json,"tree_method","hist");
    o->booster=db4ai_json_string(json,"booster","gbtree");
    if(xgb && strcmp(o->tree_method,"hist") && strcmp(o->tree_method,"approx") && strcmp(o->tree_method,"exact") && strcmp(o->tree_method,"auto"))
        ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("unsupported CPU XGBoost tree_method")));
    if(xgb && strcmp(o->booster,"gbtree") && strcmp(o->booster,"dart") && strcmp(o->booster,"gblinear"))
        ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("unsupported XGBoost booster")));
    o->nthread=bounded(json,"nthread",1,1,1,true); /* bundled build intentionally has no OpenMP */
    if (db4ai_json_field(json,"verbosity",false)) o->verbose=bounded(json,"verbosity",0,0,3,true)>0;
    if (algorithm==DB4AI_PCA && o->pca_solver==0)
    {
        if (db4ai_json_field(json,"shuffle",false) ||
            db4ai_json_field(json,"shuffle_first_epoch",false) ||
            db4ai_json_field(json,"shuffle_buffer_size",false))
            ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                           errmsg("shuffle options require pca_solver=stochastic")));
        o->shuffle_mode=0;
    }
    if (algorithm==DB4AI_MULTICLASS && o->classifier==0 && o->kernel!=0)
        ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("nonlinear kernels require classifier=svm_classification")));
}

static void
option_number(StringInfo b, const char *name, double value)
{
    char number[DOUBLE_SHORTEST_DECIMAL_LEN];
    double_to_shortest_decimal_buf(value, number);
    appendStringInfo(b, ",\"%s\":%s", name, number);
}

static void
option_string(StringInfo b, const char *key, const char *value)
{
    appendStringInfo(b, ",\"%s\":",key);
    escape_json(b,value);
}

Jsonb *
db4ai_options_json(const Db4aiOptions *o, Db4aiAlgorithm a)
{
    StringInfoData b;
    bool xgb=db4ai_is_xgboost(a);
    bool gd=a<=DB4AI_SVM || a==DB4AI_MULTICLASS;
    initStringInfo(&b);
    appendStringInfo(&b,"{\"profile\":\"%s\",\"batch_size\":%d",o->legacy_profile?"opengauss":"stable",o->batch_size);
    if (a!=DB4AI_BAYES) { option_number(&b,"max_iterations",o->iterations); option_number(&b,"seed",o->seed); }
    if (gd || xgb) { option_number(&b,"learning_rate",o->learning_rate); option_number(&b,"l2",o->l2); }
    if (gd || a==DB4AI_PCA || a==DB4AI_KMEANS) option_number(&b,"tolerance",o->tolerance);
    if (gd) { option_string(&b,"optimizer",o->optimizer?"ngd":"gd"); option_number(&b,"decay",o->decay); }
    if (gd || (a==DB4AI_PCA && o->pca_solver==1))
    {
        option_string(&b,"shuffle",o->shuffle_mode==0?"none":(o->shuffle_mode==1?"batch":"row"));
        option_number(&b,"shuffle_buffer_size",o->shuffle_buffer);
        appendStringInfo(&b,",\"shuffle_first_epoch\":%s",o->shuffle_first?"true":"false");
    }
    if (a==DB4AI_SVM || a==DB4AI_MULTICLASS)
    {
        option_string(&b,"kernel",o->kernel==0?"linear":(o->kernel==1?"gaussian":"polynomial"));
        option_number(&b,"components",o->kernel_components); option_number(&b,"degree",o->degree);
        option_number(&b,"gamma",o->gamma); option_number(&b,"coef0",o->coef0); option_number(&b,"lambda",o->svm_lambda);
        if(a==DB4AI_MULTICLASS) option_string(&b,"classifier",o->classifier?"svm_classification":"logistic_regression");
    }
    if(a==DB4AI_KMEANS)
    {
        const char *dist[]={"L1","L2","L2_Squared","Linf"};
        option_number(&b,"k",o->clusters); option_number(&b,"num_features",o->num_features);
        option_string(&b,"seeding_function",o->seeding==0?"Random++":(o->seeding==1?"KMeans||":"reservoir"));
        option_string(&b,"distance_function",dist[o->distance]);
        option_number(&b,"seeding_rounds",o->seeding_rounds); option_number(&b,"oversampling",o->oversampling);
    }
    if(a==DB4AI_PCA)
    {
        option_number(&b,"n_components",o->components);
        option_string(&b,"pca_solver",o->pca_solver?"stochastic":"jacobi");
        appendStringInfo(&b,",\"center\":%s",o->center?"true":"false");
    }
    if(xgb)
    {
        option_number(&b,"max_depth",o->max_depth); option_number(&b,"min_child_weight",o->min_child_weight);
        option_number(&b,"subsample",o->subsample); option_number(&b,"colsample_bytree",o->colsample_bytree);
        option_number(&b,"gamma",o->gamma); option_number(&b,"nthread",o->nthread);
        option_string(&b,"eval_metric",o->eval_metric); option_string(&b,"tree_method",o->tree_method); option_string(&b,"booster",o->booster);
    }
    if(a==DB4AI_BAYES)
    {
        option_string(&b,"edges_of_network",o->edges); option_number(&b,"num_nodes",o->num_nodes);
        if(o->num_edges>=0) option_number(&b,"num_edges",o->num_edges);
        option_number(&b,"num_bins",o->num_bins); option_number(&b,"num_mcvs",o->num_mcvs);
        option_number(&b,"num_sample_rows",o->num_sample_rows); option_number(&b,"num_total_rows",o->num_total_rows);
    }
    else { option_string(&b,"null_policy",o->null_policy?"legacy":"error"); option_number(&b,"max_seconds",o->max_seconds); }
    if(a!=DB4AI_BAYES) appendStringInfo(&b,",\"verbose\":%s",o->verbose?"true":"false");
    appendStringInfoChar(&b,'}');
    return DatumGetJsonbP(DirectFunctionCall1(jsonb_in,CStringGetDatum(b.data)));
}

bytea *
db4ai_bytes(const void *data, Size len)
{
    bytea *out;
    if (len > (Size) db4ai_max_model_mb * 1024 * 1024 || len > MaxAllocSize - VARHDRSZ)
        ereport(ERROR, (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED), errmsg("model exceeds db4ai.max_model_mb")));
    out = (bytea *) palloc(VARHDRSZ + len);
    SET_VARSIZE(out, VARHDRSZ + len);
    memcpy(VARDATA(out), data, len);
    return out;
}
