/* SPDX-License-Identifier: MulanPSL-2.0
 * Explicit model-DDL compatibility entry point, not a PostgreSQL raw-parser
 * hook. Stock PostgreSQL rejects bare CREATE MODEL before utility hooks run.
 * Reuse PG's lexer (comments, quoted identifiers, escape/dollar strings), and
 * let PG parse/authorize the generated SELECT. No regex-based SQL rewriting.
 */
#include "postgres.h"
#include "db4ai.h"
#include "catalog/pg_type_d.h"
#include "executor/spi.h"
#include "miscadmin.h"
#include "parser/scanner.h"
#include "parser/parser.h"
#include "parser/scansup.h"
#include "utils/builtins.h"
#include "utils/datum.h"
#include "utils/fmgrprotos.h"
#include "utils/json.h"

PG_FUNCTION_INFO_V1(pg_db4ai_execute_ddl);
PG_FUNCTION_INFO_V1(pg_db4ai_parse_ddl);

/* scanner.h specifies this stable scanner-token prefix; gram.h is not an
 * installed extension header. Do not depend on the remaining grammar tokens. */
enum { DDL_IDENT=258, DDL_UIDENT, DDL_FCONST, DDL_SCONST, DDL_USCONST,
       DDL_BCONST, DDL_XCONST, DDL_OP, DDL_ICONST };

typedef struct DdlToken
{
    int kind;
    int start;
    int end;
    int depth;
    char *value;
} DdlToken;

typedef struct ModelDdl
{
    const char *source;
    DdlToken *tokens;
    int count;
    int end;
    char *command;
    char *name;
    char *algorithm;
    char *query;
    char *format;
    Jsonb *options;
    bool if_exists;
} ModelDdl;

static void
syntax(ModelDdl *d, int index, const char *message)
{
    int position=index<d->count?d->tokens[index].start:d->end;
    ereport(ERROR,(errcode(ERRCODE_SYNTAX_ERROR),errmsg("model DDL: %s",message),
                   internalerrposition(position+1),internalerrquery(d->source)));
}

static bool
word(ModelDdl *d, int index, const char *keyword)
{
    DdlToken *t;
    Size length=strlen(keyword);
    if(index>=d->count) return false;
    t=&d->tokens[index];
    return t->end-t->start==(int)length &&
           pg_strncasecmp(d->source+t->start,keyword,length)==0;
}

static char *
identifier(ModelDdl *d, int index)
{
    DdlToken *t;
    char c;
    if(index>=d->count) syntax(d,index,"expected an identifier");
    t=&d->tokens[index]; c=d->source[t->start];
    if(t->kind==DDL_IDENT) return pstrdup(t->value);
    if(t->kind==DDL_UIDENT)
        syntax(d,index,"Unicode-escaped model identifiers are not supported; use UTF-8 quoted identifiers");
    if(t->kind==DDL_SCONST || t->kind==DDL_USCONST || t->kind==DDL_BCONST ||
       t->kind==DDL_XCONST || t->kind==DDL_FCONST || t->kind==DDL_ICONST)
        syntax(d,index,"expected an identifier, not a literal");
    if(!((c>='a' && c<='z') || (c>='A' && c<='Z') || c=='_' || (unsigned char)c>=128))
        syntax(d,index,"expected an identifier");
    return downcase_truncate_identifier(d->source+t->start,t->end-t->start,false);
}

static void
lex(ModelDdl *d)
{
    core_yy_extra_type extra;
    core_yyscan_t scanner;
    core_YYSTYPE value;
    YYLTYPE location;
    int kind;
    int capacity=128;
    int depth=0;
    char stack[256];
    if(strlen(d->source)>16*1024*1024)
        ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("model DDL exceeds 16 MiB")));
    d->tokens=palloc(capacity*sizeof(DdlToken));
    scanner=scanner_init(d->source,&extra,&ScanKeywords,ScanKeywordTokens);
    while((kind=core_yylex(&value,&location,scanner))!=0)
    {
        DdlToken *t;
        CHECK_FOR_INTERRUPTS();
        if(d->count==262144)
            ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("too many model DDL tokens")));
        if(d->count==capacity)
        {
            capacity*=2;
            d->tokens=repalloc(d->tokens,capacity*sizeof(DdlToken));
        }
        t=&d->tokens[d->count++];
        memset(t,0,sizeof(*t));
        t->kind=kind; t->start=location;
        t->end=location+strlen(extra.scanbuf+location);
        if(kind==DDL_IDENT || kind==DDL_SCONST || kind==DDL_FCONST)
            t->value=pstrdup(value.str);
        else if(kind==DDL_ICONST) t->value=psprintf("%d",value.ival);
        if(kind==')' || kind==']')
        {
            if(depth==0 || stack[depth-1]!=(kind==')'?'(': '[')) syntax(d,d->count-1,"unbalanced delimiters");
            depth--;
        }
        t->depth=depth;
        if(kind=='(' || kind=='[')
        {
            if(depth==lengthof(stack)) syntax(d,d->count-1,"nesting exceeds 256 levels");
            stack[depth++]=(char)kind;
        }
    }
    scanner_finish(scanner);
    d->end=strlen(d->source);
    if(depth) syntax(d,d->count,"unbalanced delimiters");
    if(d->count && d->tokens[d->count-1].kind==';')
        d->end=d->tokens[--d->count].start;
    for(kind=0;kind<d->count;kind++)
        if(d->tokens[kind].kind==';') syntax(d,kind,"exactly one model statement is required");
}

static int
find_word(ModelDdl *d, int start, const char *keyword)
{
    int i;
    for(i=start;i<d->count;i++)
        if(d->tokens[i].depth==0 && word(d,i,keyword)) return i;
    return -1;
}

static Jsonb *
parameters(ModelDdl *d, int index)
{
    StringInfoData out;
    List *seen=NIL;
    initStringInfo(&out);
    /* DDL defaults select the original-style profile; an explicit profile is
     * allowed to override it. Duplicate user-supplied keys remain an error. */
    appendStringInfoString(&out,db4ai_algorithm(d->algorithm)==DB4AI_GAUSSIAN_NB ?
                           "{\"batch_size\":1024" : "{\"profile\":\"opengauss\"");
    while(index<d->count)
    {
        char *key=identifier(d,index++);
        ListCell *cell;
        bool negative=false;
        bool signed_value=false;
        DdlToken *t;
        foreach(cell,seen)
            if(strcmp(key,(char *)lfirst(cell))==0) syntax(d,index-1,"duplicate hyperparameter");
        seen=lappend(seen,key);
        if(index==d->count || d->tokens[index++].kind!='=') syntax(d,index-1,"expected '=' after hyperparameter");
        if(index==d->count) syntax(d,index,"missing hyperparameter value");
        if(d->tokens[index].kind=='-' || d->tokens[index].kind=='+')
        {
            signed_value=true; negative=d->tokens[index].kind=='-'; index++;
        }
        if(index==d->count) syntax(d,index,"missing numeric value");
        t=&d->tokens[index];
        if(signed_value && t->kind!=DDL_ICONST && t->kind!=DDL_FCONST)
            syntax(d,index,"a sign requires a numeric literal");
        if(!signed_value && word(d,index,"default"))
        {
            if(strcmp(key,"model_format")!=0) db4ai_validate_option_name(key,db4ai_algorithm(d->algorithm));
            /* An explicit DEFAULT selects the chosen profile's value. */
        }
        else if(strcmp(key,"model_format")==0)
        {
            if(t->kind!=DDL_SCONST || negative) syntax(d,index,"model_format must be a string");
            d->format=pstrdup(t->value);
        }
        else
        {
            appendStringInfoChar(&out,','); escape_json(&out,key); appendStringInfoChar(&out,':');
            if(t->kind==DDL_ICONST || t->kind==DDL_FCONST)
            {
                /* PG numeric input validates the literal. Output gives legal
                 * JSON numbers also for .1, 1. and exponent notation. */
                char *number=negative?psprintf("-%s",t->value):t->value;
                Datum parsed=DirectFunctionCall3(numeric_in,CStringGetDatum(number),ObjectIdGetDatum(InvalidOid),Int32GetDatum(-1));
                char *normalized=DatumGetCString(DirectFunctionCall1(numeric_out,parsed));
                appendStringInfoString(&out,normalized);
            }
            else if(t->kind==DDL_SCONST && !negative) escape_json(&out,t->value);
            else if(!negative && (word(d,index,"true") || word(d,index,"false")))
                appendStringInfoString(&out,word(d,index,"true")?"true":"false");
            else syntax(d,index,"hyperparameters require numeric, string or boolean literals");
        }
        index++;
        if(index<d->count && d->tokens[index++].kind!=',') syntax(d,index-1,"expected ',' between hyperparameters");
        else if(index==d->count && d->tokens[index-1].kind==',') syntax(d,index,"trailing hyperparameter comma");
    }
    appendStringInfoChar(&out,'}');
    return DatumGetJsonbP(DirectFunctionCall1(jsonb_in,CStringGetDatum(out.data)));
}

static ModelDdl *
parse(const char *source)
{
    ModelDdl *d=palloc0(sizeof(*d));
    int features,target,from,with;
    int i;
    StringInfoData query;
    Db4aiOptions checked;
    Db4aiAlgorithm algorithm;
    d->source=source;
    lex(d);
    if(word(d,0,"create") && word(d,1,"model"))
    {
        d->command="CREATE MODEL";
        d->name=identifier(d,2);
        if(!word(d,3,"using")) syntax(d,3,"expected USING algorithm");
        d->algorithm=identifier(d,4);
        algorithm=db4ai_algorithm(d->algorithm);
        features=find_word(d,5,"features"); target=find_word(d,5,"target"); from=find_word(d,5,"from");
        if(from<0) syntax(d,5,"expected FROM source");
        if(features!=5 && target!=5 && from!=5) syntax(d,5,"expected FEATURES, TARGET or FROM");
        if(features>=0 && (features!=5 || features>from)) syntax(d,features,"misplaced FEATURES clause");
        if(target>=0 && (target>from || (features>=0 && target<features))) syntax(d,target,"misplaced TARGET clause");
        if(db4ai_supervised(algorithm) && target<0) syntax(d,from,"supervised algorithms require TARGET");
        if(!db4ai_supervised(algorithm) && target>=0) syntax(d,target,"unsupervised algorithms do not accept TARGET");
        if(db4ai_supervised(algorithm) && features<0) syntax(d,from,"supervised algorithms require explicit FEATURES");
        with=find_word(d,from+1,"with");
        d->options=parameters(d,with<0?d->count:with+1);
        if(with>=0 && with+1==d->count) syntax(d,with,"empty WITH clause");
        db4ai_options(d->options,algorithm,&checked);
        if(d->format==NULL) d->format=db4ai_is_xgboost(algorithm)?"ubj":"json";
        if(strcmp(d->format,"json")!=0 && !(db4ai_is_xgboost(algorithm) && strcmp(d->format,"ubj")==0))
            syntax(d,with<0?from:with,"native models require json; XGBoost supports json or ubj");
        initStringInfo(&query);
        appendStringInfoString(&query,"SELECT ");
        if(target>=0)
        {
            if(target+1>=from) syntax(d,target,"empty TARGET expression");
            appendBinaryStringInfo(&query,source+d->tokens[target].end,d->tokens[from].start-d->tokens[target].end);
            appendStringInfoChar(&query,',');
        }
        if(features>=0)
        {
            int end=target<0?from:target;
            if(features+1>=end) syntax(d,features,"empty FEATURES list");
            appendBinaryStringInfo(&query,source+d->tokens[features].end,d->tokens[end].start-d->tokens[features].end);
        }
        else appendStringInfoChar(&query,'*');
        appendStringInfoString(&query," FROM ");
        if(from+1>=(with<0?d->count:with)) syntax(d,from,"empty FROM source");
        appendBinaryStringInfo(&query,source+d->tokens[from].end,(with<0?d->end:d->tokens[with].start)-d->tokens[from].end);
        d->query=query.data;
        {
            List *statements=raw_parser(d->query,RAW_PARSE_DEFAULT);
            if(list_length(statements)!=1 || !IsA(((RawStmt *)linitial(statements))->stmt,SelectStmt))
                syntax(d,from,"expected a SELECT-compatible source");
        }
        if(target>=0)
        {
            char *expression=pnstrdup(source+d->tokens[target].end,d->tokens[from].start-d->tokens[target].end);
            List *statements=raw_parser(psprintf("SELECT %s",expression),RAW_PARSE_DEFAULT);
            SelectStmt *select=(SelectStmt *)((RawStmt *)linitial(statements))->stmt;
            if(list_length(select->targetList)!=1) syntax(d,target,"TARGET must be one expression");
        }
    }
    else if(word(d,0,"drop") && word(d,1,"model"))
    {
        d->command="DROP MODEL"; i=2;
        if(word(d,i,"if") && word(d,i+1,"exists")) { d->if_exists=true; i+=2; }
        d->name=identifier(d,i++);
        if(i!=d->count) syntax(d,i,"unexpected tokens after model name");
    }
    else if(word(d,0,"explain") && word(d,1,"model"))
    {
        d->command="EXPLAIN MODEL";
        d->name=identifier(d,2);
        if(d->count!=3) syntax(d,3,"EXPLAIN MODEL returns JSON; no trailing options are supported");
    }
    else syntax(d,0,"expected CREATE MODEL, DROP MODEL or EXPLAIN MODEL");
    if(strlen(d->name)<1 || strlen(d->name)>128)
        ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("model name must contain 1 to 128 bytes")));
    return d;
}

Datum
pg_db4ai_parse_ddl(PG_FUNCTION_ARGS)
{
    ModelDdl *d=parse(text_to_cstring(PG_GETARG_TEXT_PP(0)));
    StringInfoData out;
    initStringInfo(&out);
    appendStringInfoString(&out,"{\"command\":"); escape_json(&out,d->command);
    appendStringInfoString(&out,",\"model_name\":"); escape_json(&out,d->name);
    appendStringInfo(&out,",\"if_exists\":%s",d->if_exists?"true":"false");
    if(d->query)
    {
        char *options=JsonbToCString(NULL,&d->options->root,VARSIZE(d->options));
        appendStringInfoString(&out,",\"algorithm\":"); escape_json(&out,d->algorithm);
        appendStringInfoString(&out,",\"training_sql\":"); escape_json(&out,d->query);
        appendStringInfoString(&out,",\"model_format\":"); escape_json(&out,d->format);
        appendStringInfo(&out,",\"hyperparameters\":%s",options);
    }
    appendStringInfoChar(&out,'}');
    PG_RETURN_DATUM(DirectFunctionCall1(jsonb_in,CStringGetDatum(out.data)));
}

Datum
pg_db4ai_execute_ddl(PG_FUNCTION_ARGS)
{
    ModelDdl *d=parse(text_to_cstring(PG_GETARG_TEXT_PP(0)));
    MemoryContext caller=CurrentMemoryContext;
    Oid types[5]={TEXTOID,TEXTOID,TEXTOID,JSONBOID,TEXTOID};
    Datum values[5];
    Datum value;
    Jsonb *copy;
    bool isnull;
    int status;
    values[0]=CStringGetTextDatum(d->name);
    if(SPI_connect()!=SPI_OK_CONNECT) elog(ERROR,"SPI_connect failed");
    if(d->query)
    {
        values[1]=CStringGetTextDatum(d->algorithm); values[2]=CStringGetTextDatum(d->query);
        values[3]=JsonbPGetDatum(d->options); values[4]=CStringGetTextDatum(d->format);
        status=SPI_execute_with_args(
            "SELECT pg_catalog.jsonb_build_object('command','CREATE MODEL','model_name',$1,'model_id',"
            "db4ai.train($1,$2,$3,$4,$5))",5,types,values,NULL,false,1);
    }
    else if(strcmp(d->command,"DROP MODEL")==0)
    {
        status=SPI_execute_with_args("SELECT db4ai.drop_model($1)",1,types,values,NULL,false,1);
        if(status!=SPI_OK_SELECT || SPI_processed!=1) elog(ERROR,"model deletion returned no result");
        value=SPI_getbinval(SPI_tuptable->vals[0],SPI_tuptable->tupdesc,1,&isnull);
        if(!d->if_exists && (isnull || !DatumGetBool(value)))
            ereport(ERROR,(errcode(ERRCODE_UNDEFINED_OBJECT),errmsg("model \"%s\" does not exist for current role",d->name)));
        status=SPI_execute_with_args("SELECT pg_catalog.jsonb_build_object('command','DROP MODEL','model_name',$1)",1,types,values,NULL,false,1);
    }
    else status=SPI_execute_with_args("SELECT db4ai.explain_model($1)",1,types,values,NULL,true,1);
    if(status!=SPI_OK_SELECT || SPI_processed!=1) elog(ERROR,"model DDL returned no result");
    value=SPI_getbinval(SPI_tuptable->vals[0],SPI_tuptable->tupdesc,1,&isnull);
    if(isnull) ereport(ERROR,(errcode(ERRCODE_UNDEFINED_OBJECT),errmsg("model \"%s\" does not exist for current role",d->name)));
    MemoryContextSwitchTo(caller);
    copy=(Jsonb *)DatumGetPointer(datumCopy(value,false,-1));
    SPI_finish();
    PG_RETURN_JSONB_P(copy);
}
