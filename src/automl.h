/* SPDX-License-Identifier: MulanPSL-2.0
 * Private PostgreSQL-native AutoML interfaces. No independent ML framework.
 */
#ifndef DB4AI_AUTOML_H
#define DB4AI_AUTOML_H
#include "db4ai.h"
#include "utils/timestamp.h"

#define DB4AI_AUTOML_DEADLINE MAKE_SQLSTATE('D','B','0','0','1')
#define DB4AI_AM_MAX_TRIALS 24
#define DB4AI_AM_MAX_COLUMNS 128

typedef struct Db4aiFeature
{
    char *name;
    int source_column;
    bool categorical;
    int offset;
    int width;
    int64 observed;
    double mean, m2, scale;
    Db4aiDictionary *dictionary;
    bool capped;
    Oid input_function;
    Oid input_parameter;
} Db4aiFeature;

typedef struct Db4aiPipeline
{
    bool classification;
    int nraw, nencoded, nclasses;
    Db4aiFeature *features;
    Db4aiDictionary *labels;
    double target_mean, target_scale;
    int64 target_count;
    double target_m2;
    bool missing_indicators;
    Db4aiModel child;
    Db4aiNative *native;
    void *xgb;
} Db4aiPipeline;

typedef struct Db4aiAutoOptions
{
    double seconds, validation_fraction, test_fraction;
    int max_trials, max_rows, max_categories, seed, memory_mb, max_input_mb;
    int pilot_rows, max_encoded_mb;
    char *metric, *split, *group_column, *time_column;
    bool use_xgboost, include_nb, missing_indicators;
    double min_improvement;
} Db4aiAutoOptions;

typedef struct Db4aiAutoInput
{
    Tuplestorestate *raw;
    TupleDesc desc;
    uint8 *partition; /* 0=train, 1=validation, 2=untouched final test */
    int nrows, ntrain, nvalid, ntest;
    uint64 input_bytes, split_hash;
    bool spilled;
    Db4aiPipeline *pipeline;
    Db4aiData train, valid, test;
} Db4aiAutoInput;

extern double db4ai_automl_now(void);
extern void db4ai_automl_check(void);
extern void db4ai_automl_soft_deadline(double deadline);
extern Db4aiAutoInput *db4ai_automl_input(const char *query, bool classification,
                                        const Db4aiAutoOptions *options);
extern Db4aiPipeline *db4ai_pipeline_load(const Db4aiModel *wrapper);
extern bytea *db4ai_pipeline_save(const Db4aiPipeline *pipeline, const Db4aiModel *child);
extern void db4ai_pipeline_transform(const Db4aiPipeline *pipeline,
                                     const Datum *values, const bool *nulls,
                                     const Oid *types, double *out);
extern double *db4ai_pipeline_array(const Db4aiPipeline *pipeline, ArrayType *array);
extern double *db4ai_pipeline_json(const Db4aiPipeline *pipeline, Jsonb *json);
extern double db4ai_pipeline_value(const Db4aiPipeline *pipeline, const double *encoded);
extern void db4ai_pipeline_proba(const Db4aiPipeline *pipeline, const double *encoded, double *out);
extern int db4ai_pipeline_class(const Db4aiPipeline *pipeline, const double *encoded);
extern void db4ai_xgb_predict_vector(void *predictor, const double *features, int nfeatures,
                                    double *out, int noutputs);
#endif
