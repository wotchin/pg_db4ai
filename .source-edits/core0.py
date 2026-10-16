FILES['Makefile'] = {'old': '2c14cbeff03d97781fdd978d070369799a05c707', 'new': '905ee44578b1711bbd6cadcc289838c4a2a271be', 'base': 'Makefile', 'mode': '100644','edits':[
(3,5,r'''OBJS = src/pg_db4ai.o src/options.o src/data.o src/models.o src/native.o src/xgboost.o src/categories.o src/shuffle.o src/kernel.o src/gradient.o src/bayes.o src/shufflescan.o src/rowstore.o src/model_ddl.o
DATA = sql/pg_db4ai--0.3.0.sql sql/pg_db4ai--0.2.0--0.3.0.sql sql/pg_db4ai--0.1.0.sql sql/pg_db4ai--0.2.0.sql sql/pg_db4ai--0.1.0--0.2.0.sql
'''),
(8,8,r'''WITH_OPENMP ?= 0
ifneq ($(filter $(WITH_OPENMP),0 1),$(WITH_OPENMP))
$(error WITH_OPENMP must be 0 or 1)
endif
ifeq ($(WITH_OPENMP),1)
ifneq ($(WITH_XGBOOST),1)
$(error WITH_OPENMP=1 requires WITH_XGBOOST=1)
endif
endif
'''),
(12,12,r'''ifeq ($(WITH_OPENMP),1)
PG_CPPFLAGS += -DDB4AI_WITH_OPENMP
SHLIB_LINK += -fopenmp
endif
'''),
(24,24,r'''# First-party header changes and build switches must invalidate every object.
$(OBJS): src/db4ai.h src/extended.h build/extension-config
.PHONY: FORCE
FORCE:
build/extension-config: FORCE
	@mkdir -p build
	@printf '%s\n' '$(PG_CONFIG)|$(shell $(PG_CONFIG) --version)|$(WITH_XGBOOST)|$(WITH_OPENMP)' > $@.tmp
	@cmp -s $@.tmp $@ && rm $@.tmp || mv $@.tmp $@

build/xgboost-config: FORCE
	@mkdir -p build
	@printf '%s\n' '$(WITH_OPENMP)' > $@.tmp
	@cmp -s $@.tmp $@ && rm $@.tmp || mv $@.tmp $@

'''),
(26,27,r'''build/xgboost.stamp: third_party/xgboost.lock.json build/xgboost-config
'''),
(31,32,r'''	  -DUSE_OPENMP=$(if $(filter 1,$(WITH_OPENMP)),ON,OFF) -DUSE_CUDA=OFF -DUSE_NCCL=OFF \
'''),
(42,43,r'''	PG_CONFIG='$(PG_CONFIG)' WITH_XGBOOST='$(WITH_XGBOOST)' WITH_OPENMP='$(WITH_OPENMP)' sh test/run.sh
''')
]}
FILES['pg_db4ai.control'] = {'old': '5baebb39f479a118224a68b5d91ff3b1a8973f8e', 'new': '909feecf062adf16fc840e1b39911b8d6310a7d2', 'base': 'pg_db4ai.control', 'mode': '100644','edits':[
(1,2,r'''default_version = '0.3.0'
''')
]}
FILES['sql/pg_db4ai--0.2.0--0.3.0.sql'] = {'old': None, 'new': 'def95bc05a4310b4ebcd0677d557a67171861e18', 'base': 'sql/pg_db4ai--0.2.0--0.3.0.sql', 'mode': '100644','edits':[
(0,0,r'''-- Model-DDL compatibility without changing PostgreSQL's top-level grammar.
CREATE FUNCTION db4ai.execute_ddl(statement text)
RETURNS jsonb AS 'MODULE_PATHNAME', 'pg_db4ai_execute_ddl'
LANGUAGE C STRICT VOLATILE PARALLEL UNSAFE SECURITY INVOKER;
CREATE FUNCTION db4ai.parse_ddl(statement text)
RETURNS jsonb AS 'MODULE_PATHNAME', 'pg_db4ai_parse_ddl'
LANGUAGE C STRICT STABLE PARALLEL UNSAFE SECURITY INVOKER;
COMMENT ON FUNCTION db4ai.execute_ddl(text) IS
  'Explicit CREATE/DROP/EXPLAIN MODEL compatibility parser. Not a stock-PG raw grammar hook. Executes as caller.';
COMMENT ON FUNCTION db4ai.parse_ddl(text) IS
  'Parse one model DDL into a structured request without executing training SQL or accessing data.';

CREATE FUNCTION db4ai.shuffle_indices(row_count integer, options jsonb DEFAULT '{}'::jsonb, epoch integer DEFAULT 1)
RETURNS bigint[] AS 'MODULE_PATHNAME', 'pg_db4ai_shuffle_indices'
LANGUAGE C STRICT VOLATILE PARALLEL UNSAFE SECURITY INVOKER;
COMMENT ON FUNCTION db4ai.shuffle_indices(integer,jsonb,integer) IS
  'Bounded diagnostic for the actual training shuffle iterator. Returns 0-based indexes (maximum 100000), not a dataset.';

CREATE FUNCTION db4ai.build_info() RETURNS jsonb
AS 'MODULE_PATHNAME', 'pg_db4ai_build_info'
LANGUAGE C STABLE PARALLEL SAFE SECURITY INVOKER;
''')
]}
FILES['sql/pg_db4ai--0.3.0.sql'] = {'old': None, 'new': 'b5820cd5383dd9b0ca9722f4e1e0b1367ddf0296', 'base': 'sql/pg_db4ai--0.2.0.sql', 'mode': '100644','edits':[
(116,116,r'''
-- Model-DDL compatibility without changing PostgreSQL's top-level grammar.
CREATE FUNCTION db4ai.execute_ddl(statement text)
RETURNS jsonb AS 'MODULE_PATHNAME', 'pg_db4ai_execute_ddl'
LANGUAGE C STRICT VOLATILE PARALLEL UNSAFE SECURITY INVOKER;
CREATE FUNCTION db4ai.parse_ddl(statement text)
RETURNS jsonb AS 'MODULE_PATHNAME', 'pg_db4ai_parse_ddl'
LANGUAGE C STRICT STABLE PARALLEL UNSAFE SECURITY INVOKER;
COMMENT ON FUNCTION db4ai.execute_ddl(text) IS
  'Explicit CREATE/DROP/EXPLAIN MODEL compatibility parser. Not a stock-PG raw grammar hook. Executes as caller.';
COMMENT ON FUNCTION db4ai.parse_ddl(text) IS
  'Parse one model DDL into a structured request without executing training SQL or accessing data.';

CREATE FUNCTION db4ai.shuffle_indices(row_count integer, options jsonb DEFAULT '{}'::jsonb, epoch integer DEFAULT 1)
RETURNS bigint[] AS 'MODULE_PATHNAME', 'pg_db4ai_shuffle_indices'
LANGUAGE C STRICT VOLATILE PARALLEL UNSAFE SECURITY INVOKER;
COMMENT ON FUNCTION db4ai.shuffle_indices(integer,jsonb,integer) IS
  'Bounded diagnostic for the actual training shuffle iterator. Returns 0-based indexes (maximum 100000), not a dataset.';

CREATE FUNCTION db4ai.build_info() RETURNS jsonb
AS 'MODULE_PATHNAME', 'pg_db4ai_build_info'
LANGUAGE C STABLE PARALLEL SAFE SECURITY INVOKER;
''')
]}
FILES['src/data.c'] = {'old': 'e5ef26f0c18369556eb71d911a6d7c3cba89fc26', 'new': '720e490c000bf59e958487d9703b545148fd05b4', 'base': 'src/data.c', 'mode': '100644','edits':[
(76,77,r'''    double *values;
'''),
(170,171,r''''''),
(172,173,r''''''),
(181,183,r'''    data->dense=db4ai_rows_create(data->ncolumns,(Size)db4ai_work_mem*1024);
'''),
(184,185,r'''    values=palloc0(sizeof(double)*data->ncolumns);
'''),
(210,211,r'''                values[column++]=(double)db4ai_dictionary_index(data->domains[i],v,isnull,true);
'''),
(218,219,r'''                values[column++]=(double)db4ai_dictionary_index(data->labels,v,false,true);
'''),
(234,235,r'''                    values[column++]=as_double(items[j],elem);
'''),
(240,241,r'''                values[column++]=isnull?0.0:as_double(v,type);
'''),
(252,253,r'''        else { db4ai_rows_append(data->dense,values); data->nrows++; }
'''),
(255,256,r'''    db4ai_rows_seal(data->dense);
    data->spilled=db4ai_rows_spilled(data->dense);
'''),
(257,258,r'''    pfree(values); pfree(width);
'''),
(263,264,r'''        Db4aiRowStore *ordered=db4ai_rows_create(data->ncolumns,(Size)db4ai_work_mem*1024);
        double *row=palloc(data->ncolumns*sizeof(double));
'''),
(268,270,r'''        data->position=0;
        while(db4ai_data_next(data,row))
'''),
(271,275,r'''            CHECK_FOR_INTERRUPTS();
            row[0]=1.0-row[0];
            db4ai_rows_append(ordered,row);
'''),
(276,278,r'''        db4ai_rows_seal(ordered);
        db4ai_rows_free(data->dense); data->dense=ordered;
        data->spilled=data->spilled || db4ai_rows_spilled(ordered);
        pfree(row);
'''),
(279,280,r'''    data->position=0;
'''),
(286,288,r'''    const double *source;
    if (!db4ai_rows_slice(data->dense,data->position,1,&source))
'''),
(289,293,r'''    memcpy(row,source,(Size)data->ncolumns*sizeof(double));
    data->position++;
'''),
(299,301,r'''    data->position=0;
'''),
(307,310,r'''    db4ai_rows_free(data->dense);
''')
]}
FILES['src/db4ai.h'] = {'old': '8384732db7b99100a3c3aa196f4ed2d5905aeea0', 'new': '369efcc8210531c738c2540bcbc7310c8a2d23bb', 'base': 'src/db4ai.h', 'mode': '100644','edits':[
(49,49,r'''    int shuffle_block_rows;
    int shuffle_window_blocks;
    bool kernel_cache;
'''),
(56,57,r'''    double max_seconds;
'''),
(76,79,r'''    struct Db4aiRowStore *dense;
    int64 position;
'''),
(90,90,r'''    int64 shuffle_blocks;
    int shuffle_window_rows;
    bool kernel_cached;
    bool kernel_spilled;
    int64 kernel_rows;
    double input_seconds;
    double feature_seconds;
    double optimize_seconds;
    double evaluation_seconds;
    uint64 prepared_bytes_read;
    uint64 prepared_bytes_written;
    uint64 prepared_seeks;
'''),
(116,116,r'''    Jsonb *diagnostics;
'''),
(139,140,r'''extern void db4ai_validate_option_name(const char *name, Db4aiAlgorithm algorithm);
void db4ai_options(Jsonb *json, Db4aiAlgorithm algorithm, Db4aiOptions *out);
''')
]}
FILES['src/extended.h'] = {'old': '3e894f2eeaeb24991239ae1c3fec615c2c8391d0', 'new': 'afabceeb147ee646fc3f9fb127150632b5eed091', 'base': 'src/extended.h', 'mode': '100644','edits':[
(34,35,r'''    DB4AI_SHUFFLE_NONE, DB4AI_SHUFFLE_BATCH, DB4AI_SHUFFLE_ROW, DB4AI_SHUFFLE_CORGIPILE
'''),
(37,37,r'''typedef struct Db4aiRowStore Db4aiRowStore;
extern Db4aiRowStore *db4ai_rows_create(int ncolumns, Size budget);
extern void db4ai_rows_append(Db4aiRowStore *store, const double *row);
extern void db4ai_rows_seal(Db4aiRowStore *store);
extern int db4ai_rows_slice(Db4aiRowStore *store, int64 start, int maximum, const double **rows);
extern int db4ai_rows_read(Db4aiRowStore *store, int64 start, int maximum, double *out);
extern bool db4ai_rows_spilled(Db4aiRowStore *store);
extern void db4ai_rows_stats(Db4aiRowStore *store, uint64 *read, uint64 *written, uint64 *seeks);
extern void db4ai_rows_free(Db4aiRowStore *store);
''')
]}
FILES['src/gradient.c'] = {'old': '7fa8577fee13db52720c3e92fb9a6f065f8f7b52', 'new': 'e3f498751559a97c6bf12e5fd3dc2a7e7d9c7f39', 'base': 'src/gradient.c', 'mode': '100644','edits':[
(24,25,r'''    if (unlikely(!isfinite(value))) db4ai_check_finite(value);
    return value;
'''),
(60,61,r'''update_batch(Db4aiNative *m, const Db4aiOptions *o, const double *features,
'''),
(75,76,r'''                double v=features[(Size)r*(n+1)+1+j], q=v*v;
'''),
(82,84,r'''            const double *x=features+(Size)r*(n+1)+1;
            double label=features[(Size)r*(n+1)];
            double y=classifier?(m->algorithm==DB4AI_MULTICLASS?(label==c?1.0:0.0):label):label;
'''),
(93,94,r'''            if (error!=0.0) for(j=0;j<n;j++) grad[j]+=error*x[j];
'''),
(101,102,r'''            w[j]-=rate*g; if (unlikely(!isfinite(w[j]))) db4ai_check_finite(w[j]);
'''),
(104,104,r'''}

/* Precompute once, including bias and random maps. The cached representation
 * is still spillable and temporary; serialized models contain maps, not rows.
 * Disabling the cache is an explicit ablation/debug path, not a new model. */
static void
prepare_rows(Db4aiNative *m, Db4aiData *source, Db4aiData *prepared)
{
    double *row=palloc((m->ncoefficients+1)*sizeof(double));
    int64 cursor=0;
    *prepared=*source;
    prepared->ncolumns=m->ncoefficients+1;
    prepared->nfeatures=m->ncoefficients;
    prepared->position=0; prepared->passes=0;
    prepared->dense=db4ai_rows_create(prepared->ncolumns,(Size)db4ai_work_mem*1024);
    while(cursor<source->nrows)
    {
        const double *rows;
        int i, count=db4ai_rows_slice(source->dense,cursor,4096,&rows);
        for(i=0;i<count;i++)
        {
            const double *raw=rows+(Size)i*source->ncolumns;
            CHECK_FOR_INTERRUPTS();
            row[0]=raw[0];
            db4ai_kernel_transform(m,raw+1,row+1);
            db4ai_rows_append(prepared->dense,row);
        }
        cursor+=count;
    }
    db4ai_rows_seal(prepared->dense);
    prepared->spilled=db4ai_rows_spilled(prepared->dense);
    pfree(row);
'''),
(110,111,r'''    Db4aiData prepared;
    Db4aiData *train=data;
    double *batch=NULL, *assembly=NULL, *scales, *scale_rate, *gradient, *v;
'''),
(114,114,r'''    TimestampTz stage;
'''),
(117,117,r'''    stage=GetCurrentTimestamp();
'''),
(123,126,r'''    v=palloc(n*sizeof(double));
    if(o->kernel_cache)
    {
        prepare_rows(m,data,&prepared);
        train=&prepared;
        data->kernel_cached=true;
        data->kernel_spilled=prepared.spilled;
        data->kernel_rows=m->kernel?data->nrows:0;
    }
    else batch=palloc((Size)o->batch_size*(n+1)*sizeof(double));
    data->feature_seconds=(GetCurrentTimestamp()-stage)/1000000.0;
    shuffle=db4ai_shuffle_create(train,o);
'''),
(130,130,r'''        int64 cursor=0;
'''),
(131,131,r'''        stage=GetCurrentTimestamp();
'''),
(133,133,r'''        {
            const double *features=rows;
            if(!o->kernel_cache)
            {
                for(r=0;r<count;r++)
                {
                    const double *raw=rows+(Size)r*data->ncolumns;
                    double *target=batch+(Size)r*(n+1);
                    target[0]=raw[0];
                    db4ai_kernel_transform(m,raw+1,target+1);
                }
                features=batch;
                if(m->kernel) data->kernel_rows+=count;
            }
            /* A shuffled short unit can occur in the middle of an epoch.
             * Reassemble minibatches across such unit boundaries, exactly as
             * the original row-fed trainer did; do not change update sizes. */
            if(fill==0 && count==o->batch_size)
                update_batch(m,o,features,count,epoch,rate,scales,scale_rate,gradient);
            else
            {
                int used=0;
                if(assembly==NULL) assembly=palloc((Size)o->batch_size*(n+1)*sizeof(double));
                while(used<count)
                {
                    int take=Min(count-used,o->batch_size-fill);
                    memcpy(assembly+(Size)fill*(n+1),features+(Size)used*(n+1),(Size)take*(n+1)*sizeof(double));
                    fill+=take; used+=take;
                    if(fill==o->batch_size)
                    {
                        update_batch(m,o,assembly,fill,epoch,rate,scales,scale_rate,gradient);
                        fill=0;
                    }
                }
            }
        }
        if(fill) update_batch(m,o,assembly,fill,epoch,rate,scales,scale_rate,gradient);
        data->optimize_seconds+=(GetCurrentTimestamp()-stage)/1000000.0;
        stage=GetCurrentTimestamp();
        train->passes++;
        while(cursor<train->nrows)
        {
            count=db4ai_rows_slice(train->dense,cursor,4096,&rows);
'''),
(135,138,r'''                const double *raw=rows+(Size)r*train->ncolumns;
                const double *x=raw+1;
                int c;
                CHECK_FOR_INTERRUPTS();
                if(!o->kernel_cache)
'''),
(139,140,r'''                    db4ai_kernel_transform(m,raw+1,v); x=v;
                    if(m->kernel) data->kernel_rows++;
                }
                for(c=0;c<models;c++)
                {
                    double z=linear_score(m->weights+(Size)c*n,x,n);
                    double y=m->algorithm==DB4AI_MULTICLASS?(raw[0]==c?1:0):raw[0];
                    if(m->algorithm==DB4AI_SVM || (m->algorithm==DB4AI_MULTICLASS && m->classifier==DB4AI_SVM))
                        loss+=Max(0,1-(2*y-1)*z);
                    else if(db4ai_classifier(m->algorithm)) loss+=Max(z,0)-y*z+log1p(exp(-fabs(z)));
                    else loss+=(z-y)*(z-y)*(o->legacy_profile?0.5:1.0);
'''),
(142,157,r'''            cursor+=count;
'''),
(158,158,r'''        data->evaluation_seconds+=(GetCurrentTimestamp()-stage)/1000000.0;
'''),
(164,164,r'''    data->shuffle_capacity=train->shuffle_capacity;
    data->shuffle_full_cache=train->shuffle_full_cache;
    data->shuffle_blocks=train->shuffle_blocks;
    data->shuffle_window_rows=train->shuffle_window_rows;
'''),
(165,166,r'''    if(o->kernel_cache)
    {
        data->passes+=prepared.passes;
        db4ai_rows_stats(prepared.dense, &data->prepared_bytes_read,
                        &data->prepared_bytes_written, &data->prepared_seeks);
        db4ai_rows_free(prepared.dense);
    }
    if(batch) pfree(batch);
    if(assembly) pfree(assembly);
    pfree(scales); pfree(scale_rate); pfree(gradient); pfree(v);
''')
]}
