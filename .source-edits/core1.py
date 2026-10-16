FILES['src/native.c'] = {'old': 'cc718f56da16de5519a6059323f58397248b6eda', 'new': '48c0eb6ff16560dc5d867e435aae5a81b471a722', 'base': 'src/native.c', 'mode': '100644','edits':[
(354,354,r'''    m->eigenvalues=palloc0(m->noutputs*sizeof(double));
'''),
(369,369,r'''        m->eigenvalues[p]=Max(0.0,a[best*n+best]);
'''),
(376,376,r'''}

/* User-visible model diagnostics are DB4AI functionality, unlike ABO's use of
 * Bayes estimates inside an optimizer. Store these once with the model. */
static void
native_diagnostics(Db4aiNative *m, Db4aiData *data)
{
    StringInfoData json;
    int c;
    initStringInfo(&json);
    if(m->algorithm==DB4AI_KMEANS)
    {
        int k=m->noutputs;
        int64 *count=palloc0(k*sizeof(int64));
        double *mean=palloc0(k*sizeof(double)), *m2=palloc0(k*sizeof(double));
        double *min=palloc(k*sizeof(double)), *max=palloc0(k*sizeof(double));
        double *row=palloc(data->ncolumns*sizeof(double));
        for(c=0;c<k;c++) min[c]=DBL_MAX;
        db4ai_data_rewind(data);
        while(db4ai_data_next(data,row))
        {
            double distance,delta;
            CHECK_FOR_INTERRUPTS();
            c=closest(m,row,&distance);
            delta=distance-mean[c];
            mean[c]+=delta/(++count[c]);
            m2[c]+=delta*(distance-mean[c]);
            min[c]=Min(min[c],distance); max[c]=Max(max[c],distance);
        }
        appendStringInfo(&json,"{\"centroids\":[");
        for(c=0;c<k;c++)
        {
            if(c) appendStringInfoChar(&json,',');
            appendStringInfo(&json,"{\"id\":%d,\"size\":" INT64_FORMAT ",\"coordinates\":",c,count[c]);
            db4ai_write_array(&json,m->weights+(Size)c*m->nfeatures,m->nfeatures);
            if(count[c])
            {
                double stats[]={mean[c]*count[c],min[c],max[c],mean[c],sqrt(Max(0.0,m2[c]/count[c]))};
                appendStringInfoString(&json,",\"distance_stats\":");
                db4ai_write_array(&json,stats,5);
            }
            else appendStringInfoString(&json,",\"distance_stats\":null");
            appendStringInfoChar(&json,'}');
        }
        appendStringInfoString(&json,"],\"distance_stat_order\":[\"sum\",\"min\",\"max\",\"mean\",\"population_stddev\"]}");
        pfree(row); pfree(count); pfree(mean); pfree(m2); pfree(min); pfree(max);
    }
    else if(m->algorithm==DB4AI_PCA)
    {
        double total=0;
        for(c=0;c<m->noutputs;c++) total+=m->eigenvalues[c];
        appendStringInfoString(&json,"{\"principal_components\":[");
        for(c=0;c<m->noutputs;c++)
        {
            double stats[]={m->eigenvalues[c],total>0?m->eigenvalues[c]/total:0};
            if(c) appendStringInfoChar(&json,',');
            appendStringInfo(&json,"{\"id\":%d,\"coordinates\":",c);
            db4ai_write_array(&json,m->weights+(Size)c*m->nfeatures,m->nfeatures);
            appendStringInfoString(&json,",\"eigenvalue_and_selected_fraction\":");
            db4ai_write_array(&json,stats,2);
            appendStringInfoString(&json,",\"proportion_stddev_estimate\":");
            if(m->eigen_stddev) db4ai_write_array(&json,m->eigen_stddev+c,1);
            else appendStringInfoString(&json,"null");
            appendStringInfoChar(&json,'}');
        }
        appendStringInfoString(&json,"]}");
    }
    else { pfree(json.data); return; }
    m->diagnostics=DatumGetJsonbP(DirectFunctionCall1(jsonb_in,CStringGetDatum(json.data)));
    pfree(json.data);
'''),
(394,394,r'''    native_diagnostics(model,data);
''')
]}
FILES['src/options.c'] = {'old': '13cab7b4997e496af0341602ea8968fc8142d80d', 'new': 'e31f06261e7d046f6ea7c1efe129345a5aa87ccb', 'base': 'src/options.c', 'mode': '100644','edits':[
(124,124,r'''    {"shuffle_block_rows",GD|PCA}, {"shuffle_window_blocks",GD|PCA}, {"kernel_cache",GD},
'''),
(180,180,r'''db4ai_validate_option_name(const char *name, Db4aiAlgorithm algorithm)
{
    int i;
    unsigned int bit=A(algorithm);
    for(i=0;i<lengthof(keys);i++)
        if(strcmp(name,keys[i].key)==0 && (keys[i].algorithms & bit)) return;
    ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("unknown or inapplicable hyperparameter: %s",name)));
}

void
'''),
(226,228,r'''    o->shuffle_mode=choice(json,"shuffle","batch","none","batch","row","corgipile");
    o->shuffle_first=db4ai_json_boolean(json,"shuffle_first_epoch",o->shuffle_mode==DB4AI_SHUFFLE_CORGIPILE);
'''),
(229,229,r'''    o->shuffle_block_rows=bounded(json,"shuffle_block_rows",256,1,1048576,true);
    o->shuffle_window_blocks=bounded(json,"shuffle_window_blocks",8,2,65536,true);
    o->kernel_cache=db4ai_json_boolean(json,"kernel_cache",true);
    if(o->shuffle_mode!=DB4AI_SHUFFLE_CORGIPILE &&
       (db4ai_json_field(json,"shuffle_block_rows",false) || db4ai_json_field(json,"shuffle_window_blocks",false)))
        ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),errmsg("block/window options require shuffle=corgipile")));
'''),
(236,237,r'''    o->max_seconds=bounded(json,"max_seconds",0,0,86400,false);
'''),
(259,260,r'''    o->nthread=bounded(json,"nthread",1,1,64,true);
#ifndef DB4AI_WITH_OPENMP
    if(o->nthread != 1)
        ereport(ERROR,(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                       errmsg("nthread > 1 requires WITH_OPENMP=1"),
                       errhint("Rebuild the bundled XGBoost with WITH_XGBOOST=1 WITH_OPENMP=1; default builds remain single-threaded.")));
#endif
'''),
(265,266,r'''            db4ai_json_field(json,"shuffle_buffer_size",false) ||
            db4ai_json_field(json,"shuffle_block_rows",false) || db4ai_json_field(json,"shuffle_window_blocks",false))
'''),
(303,304,r'''        option_string(&b,"shuffle",o->shuffle_mode==0?"none":(o->shuffle_mode==1?"batch":(o->shuffle_mode==2?"row":"corgipile")));
'''),
(306,306,r'''    }
    if (gd) appendStringInfo(&b,",\"kernel_cache\":%s",o->kernel_cache?"true":"false");
    if (o->shuffle_mode==DB4AI_SHUFFLE_CORGIPILE)
    {
        option_number(&b,"shuffle_block_rows",o->shuffle_block_rows);
        option_number(&b,"shuffle_window_blocks",o->shuffle_window_blocks);
''')
]}
FILES['src/pg_db4ai.c'] = {'old': '4bbd82699e40167452217599fa02bcb4b6ed95bc', 'new': '951a5081a810254726c67f5a5cb90efe1039dc00', 'base': 'src/pg_db4ai.c', 'mode': '100644','edits':[
(38,39,r'''    DefineCustomIntVariable("db4ai.work_mem", "Memory per training replay store and shuffle buffer, in kB.", NULL,
'''),
(91,91,r'''        data->input_seconds=(GetCurrentTimestamp()-started)/1000000.0;
'''),
(140,140,r'''            if(native->diagnostics)
                model.metrics=DatumGetJsonbP(DirectFunctionCall2(jsonb_concat,
                    JsonbPGetDatum(model.metrics),JsonbPGetDatum(native->diagnostics)));

'''),
(152,152,r'''        }
        {
            StringInfoData phases;
            initStringInfo(&phases);
            appendStringInfo(&phases,"{\"input_seconds\":%.9f,\"feature_prepare_seconds\":%.9f,"
                "\"optimize_seconds\":%.9f,\"evaluation_seconds\":%.9f,"
                "\"prepared_cache\":%s,\"prepared_cache_spilled\":%s,\"kernel_transformed_rows\":" INT64_FORMAT ","
                "\"shuffle_blocks\":" INT64_FORMAT ",\"shuffle_window_rows\":%d,\"replay_format\":\"dense-float8\"}",
                data->input_seconds,data->feature_seconds,data->optimize_seconds,data->evaluation_seconds,
                data->kernel_cached?"true":"false",data->kernel_spilled?"true":"false",data->kernel_rows,
                data->shuffle_blocks,data->shuffle_window_rows);
            model.metrics=DatumGetJsonbP(DirectFunctionCall2(jsonb_concat,JsonbPGetDatum(model.metrics),
                DirectFunctionCall1(jsonb_in,CStringGetDatum(phases.data))));
        }
        if(data->dense)
        {
            uint64 read,written,seeks;
            StringInfoData io;
            db4ai_rows_stats(data->dense,&read,&written,&seeks);
            initStringInfo(&io);
            appendStringInfo(&io,"{\"replay_bytes_read\":" UINT64_FORMAT
                ",\"replay_bytes_written\":" UINT64_FORMAT ",\"replay_seeks\":" UINT64_FORMAT
                ",\"prepared_bytes_read\":" UINT64_FORMAT ",\"prepared_bytes_written\":" UINT64_FORMAT
                ",\"prepared_seeks\":" UINT64_FORMAT "}",read,written,seeks,
                data->prepared_bytes_read,data->prepared_bytes_written,data->prepared_seeks);
            model.metrics=DatumGetJsonbP(DirectFunctionCall2(jsonb_concat,JsonbPGetDatum(model.metrics),
                DirectFunctionCall1(jsonb_in,CStringGetDatum(io.data))));
'''),
(401,401,r'''
/* Compilation capabilities; this does not promise a compatible top-level SQL grammar. */
PG_FUNCTION_INFO_V1(pg_db4ai_build_info);
Datum
pg_db4ai_build_info(PG_FUNCTION_ARGS)
{
#ifdef DB4AI_WITH_XGBOOST
    const char *xgboost="true";
#else
    const char *xgboost="false";
#endif
#ifdef DB4AI_WITH_OPENMP
    const char *openmp="true";
#else
    const char *openmp="false";
#endif
    char *info=psprintf("{\"extension_version\":\"0.3.0\",\"postgres_major\":%d,"
        "\"xgboost\":%s,\"openmp\":%s,\"training_corgipile\":true,"
        "\"dense_replay\":true,\"model_ddl\":\"explicit execute_ddl entrypoint\","
        "\"native_create_model_grammar\":false}",PG_VERSION_NUM/10000,xgboost,openmp);
    PG_RETURN_DATUM(DirectFunctionCall1(jsonb_in,CStringGetDatum(info)));
}
''')
]}
FILES['src/shuffle.c'] = {'old': '99f2f5ad57c4888e78cbe2e9167db7278d6bbfeb', 'new': '47eb20ba53085e8bb261e6b69bae52569838f707', 'base': 'src/shuffle.c', 'mode': '100644','edits':[
(35,35,r'''    /* CorgiPile: random block order, then a uniform permutation of rows in
     * a bounded multi-block window. Index bytes are included in the budget. */
    int64 *blocks;
    int *row_order;
    double *window;
    int block_rows;
    int window_blocks;
    int nblocks;
    int next_block;
    int window_count;
    int window_cursor;

'''),
(40,47,r'''    int n;
    CHECK_FOR_INTERRUPTS();
    n=db4ai_rows_read(s->data->dense,s->data->position,s->unit,out);
    s->data->position+=n;
'''),
(64,64,r'''    if (s->mode==DB4AI_SHUFFLE_CORGIPILE)
    {
        int64 blocks=(data->nrows+o->shuffle_block_rows-1)/o->shuffle_block_rows;
        int64 windowblocks=Min(blocks,(int64)o->shuffle_window_blocks);
        int64 windowrows=Min(data->nrows,windowblocks*o->shuffle_block_rows);
        Size indexbytes;
        Size outputbytes=(Size)o->batch_size*rowbytes;
        if (blocks>INT_MAX || windowrows>INT_MAX ||
            (uint64)blocks>MaxAllocSize/sizeof(int64) ||
            (uint64)windowrows>MaxAllocSize/rowbytes || outputbytes>MaxAllocSize)
            ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),errmsg("CorgiPile buffer exceeds allocation limits")));
        indexbytes=(Size)blocks*sizeof(int64)+(Size)windowrows*sizeof(int);
        if (indexbytes+outputbytes+(Size)windowrows*rowbytes+4096>budget)
            ereport(ERROR,(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
                errmsg("CorgiPile blocks, row indexes and output do not fit db4ai.work_mem"),
                errhint("Reduce shuffle_block_rows, shuffle_window_blocks or batch_size; or increase db4ai.work_mem.")));
        s->block_rows=o->shuffle_block_rows; s->nblocks=(int)blocks;
        s->window_blocks=(int)windowblocks; s->result_rows=o->batch_size;
        s->blocks=palloc((Size)blocks*sizeof(int64));
        s->row_order=palloc((Size)windowrows*sizeof(int));
        s->window=palloc((Size)windowrows*rowbytes);
        s->output=palloc(outputbytes);
        data->shuffle_blocks=blocks;
        data->shuffle_window_rows=(int)windowrows;
        data->shuffle_capacity=(int)windowrows;
        data->shuffle_full_cache=windowrows==data->nrows;
        MemoryContextSwitchTo(old);
        return s;
    }
'''),
(98,98,r'''    if (s->mode==DB4AI_SHUFFLE_CORGIPILE)
    {
        s->next_block=0; s->window_count=0; s->window_cursor=0;
        for(i=0;i<s->nblocks;i++) s->blocks[i]=i;
        if(!s->sequential) for(i=s->nblocks-1;i>0;i--)
        {
            int j=(int)pg_prng_uint64_range(&s->rng,0,i);
            int64 t=s->blocks[i]; s->blocks[i]=s->blocks[j]; s->blocks[j]=t;
        }
        s->data->passes++;
        return;
    }
'''),
(151,151,r'''static int
corgipile_next(Db4aiShuffle *s, const double **rows)
{
    int filled=0;
    Size width=s->data->ncolumns;
    while(filled<s->result_rows)
    {
        int index;
        if(s->window_cursor==s->window_count)
        {
            int k,i;
            if(s->next_block==s->nblocks) break;
            s->window_count=0; s->window_cursor=0;
            for(k=0;k<s->window_blocks && s->next_block<s->nblocks;k++)
            {
                int64 start=s->blocks[s->next_block++]*s->block_rows;
                int n;
                CHECK_FOR_INTERRUPTS();
                n=db4ai_rows_read(s->data->dense,start,s->block_rows,
                                 s->window+(Size)s->window_count*width);
                if(n<1) elog(ERROR,"CorgiPile encountered an empty block");
                s->window_count+=n;
            }
            for(i=0;i<s->window_count;i++) s->row_order[i]=i;
            if(!s->sequential) for(i=s->window_count-1;i>0;i--)
            {
                int j=(int)pg_prng_uint64_range(&s->rng,0,i);
                int t=s->row_order[i]; s->row_order[i]=s->row_order[j]; s->row_order[j]=t;
            }
        }
        index=s->row_order[s->window_cursor++];
        memcpy(s->output+(Size)filled*width,s->window+(Size)index*width,width*sizeof(double));
        filled++;
    }
    s->emitted+=filled;
    if(!filled && s->emitted!=s->data->nrows) elog(ERROR,"CorgiPile violated exact-once epoch accounting");
    *rows=s->output;
    return filled;
}

'''),
(156,156,r'''    if(s->mode==DB4AI_SHUFFLE_CORGIPILE) return corgipile_next(s,rows);
'''),
(171,171,r'''
/* Bounded diagnostic interface used to audit the actual training iterator.
 * This returns row indexes only; it neither reads tables nor trains a model. */
PG_FUNCTION_INFO_V1(pg_db4ai_shuffle_indices);
Datum
pg_db4ai_shuffle_indices(PG_FUNCTION_ARGS)
{
    MemoryContext parent=CurrentMemoryContext;
    MemoryContext context;
    ArrayType *result=NULL;
    int nrows=PG_GETARG_INT32(0);
    int epoch=PG_GETARG_INT32(2);
    if(nrows<0 || nrows>100000 || epoch<0)
        ereport(ERROR,(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                       errmsg("shuffle diagnostic requires 0..100000 rows and a nonnegative epoch")));
    context=AllocSetContextCreate(parent,"DB4AI shuffle diagnostic",ALLOCSET_DEFAULT_SIZES);
    PG_TRY();
    {
        Db4aiData data;
        Db4aiOptions options;
        Db4aiShuffle *scan;
        const double *rows;
        Datum *values;
        int n,i,position=0;
        MemoryContextSwitchTo(context);
        memset(&data,0,sizeof(data));
        data.ncolumns=1; data.nfeatures=1; data.nrows=nrows;
        db4ai_options(PG_GETARG_JSONB_P(1),DB4AI_LINEAR,&options);
        data.dense=db4ai_rows_create(1,(Size)db4ai_work_mem*1024);
        for(i=0;i<nrows;i++) { double v=i; db4ai_rows_append(data.dense,&v); }
        db4ai_rows_seal(data.dense);
        scan=db4ai_shuffle_create(&data,&options);
        db4ai_shuffle_start(scan,epoch);
        values=palloc((Size)Max(nrows,1)*sizeof(Datum));
        while((n=db4ai_shuffle_next(scan,&rows))>0)
            for(i=0;i<n;i++)
            {
                if(position>=nrows) elog(ERROR,"shuffle emitted excess rows");
                values[position++]=Int64GetDatum((int64)rows[i]);
            }
        if(position!=nrows) elog(ERROR,"shuffle dropped rows");
        MemoryContextSwitchTo(parent);
        result=nrows?construct_array(values,nrows,INT8OID,8,FLOAT8PASSBYVAL,TYPALIGN_DOUBLE):construct_empty_array(INT8OID);
        MemoryContextDelete(context);
    }
    PG_CATCH();
    {
        MemoryContextSwitchTo(parent); MemoryContextDelete(context); PG_RE_THROW();
    }
    PG_END_TRY();
    PG_RETURN_ARRAYTYPE_P(result);
}
''')
]}
FILES['src/xgboost.c'] = {'old': '9dd9e4f2862aa237c42f2622ce185f72a4114c6d', 'new': 'b4c549ff1582a9328195ca005279f4120503ff1e', 'base': 'src/xgboost.c', 'mode': '100644','edits':[
(162,163,r'''    set_integer(r->booster, "nthread", o->nthread);
''')
]}
