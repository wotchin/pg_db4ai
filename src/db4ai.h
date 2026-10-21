/* SPDX-License-Identifier: MulanPSL-2.0 */
#ifndef PG_DB4AI_H
#define PG_DB4AI_H

#include "numeric.h"
void db4ai_numeric_init(void);

#include "postgres.h"
#include "fmgr.h"
#include "executor/tuptable.h"
#include "utils/jsonb.h"
#include "utils/array.h"
#include "utils/memutils.h"
#include "utils/numeric.h"
#include "utils/tuplestore.h"

#define DB4AI_FORMAT_VERSION 1
#define DB4AI_MAX_FEATURES 1024

typedef enum Db4aiAlgorithm
{
    DB4AI_LINEAR, DB4AI_LOGISTIC, DB4AI_SVM, DB4AI_KMEANS, DB4AI_PCA,
    DB4AI_XGB_SQUARED, DB4AI_XGB_BINARY, DB4AI_XGB_LOGISTIC, DB4AI_XGB_GAMMA,
    DB4AI_MULTICLASS, DB4AI_BAYES
} Db4aiAlgorithm;

typedef struct Db4aiOptions
{
    int iterations;
    int batch_size;
    int clusters;
    int components;
    int seed;
    int max_depth;
    double learning_rate;
    double l2;
    double tolerance;
    double min_child_weight;
    double subsample;
    double colsample_bytree;
    int optimizer;             /* 0: GD, 1: normalized GD (not natural GD) */
    int kernel;                /* 0: linear, 1: Gaussian RFF, 2: polynomial RM */
    int kernel_components;
    int degree;
    double gamma;
    double coef0;
    double decay;
    double svm_lambda;
    int classifier;            /* OVA binary learner, logistic or SVM */
    int shuffle_mode;
    bool shuffle_first;
    int shuffle_buffer;
    int shuffle_block_rows;
    int shuffle_window_blocks;
    bool kernel_cache;
    int seeding;               /* 0: Random++, 1: KMeans||, 2: reservoir */
    int distance;              /* 0: L1, 1: L2, 2: L2 squared, 3: Linf */
    int seeding_rounds;
    int oversampling;
    int pca_solver;            /* 0: covariance/Jacobi, 1: stochastic */
    bool center;
    int null_policy;           /* 0: error, 1: skip label / zero features */
    double max_seconds;
    bool verbose;
    bool legacy_profile;
    int num_features;
    int num_nodes;
    int num_edges;
    int num_bins;
    int num_mcvs;
    int64 num_sample_rows;
    int64 num_total_rows;
    char *edges;
    char *eval_metric;
    char *tree_method;
    char *booster;
    int nthread;

} Db4aiOptions;

typedef struct Db4aiData
{
    struct Db4aiRowStore *dense;
    int64 position;
    int ncolumns;
    int nfeatures;
    int64 nrows;
    char **feature_names;
    struct Db4aiDictionary *labels;
    struct Db4aiDictionary **domains;  /* Bayes variables, including numeric ones */
    int64 discarded;
    int64 passes;
    bool spilled;
    int shuffle_capacity;
    bool shuffle_full_cache;
    int64 shuffle_blocks;
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

} Db4aiData;

typedef struct Db4aiNative
{
    Db4aiAlgorithm algorithm;
    int nfeatures;
    int noutputs;
    int iterations;
    double loss;
    double *weights;
    double *mean;
    struct Db4aiDictionary *labels;
    int nclasses;
    int ncoefficients;
    int classifier;
    int distance;
    int kernel;
    int kernel_components;
    int kernel_rows;
    double *kernel_weights;
    double *kernel_offsets;
    double *kernel_repetitions;
    double *kernel_coefs;
    double *eigenvalues;
    double *eigen_stddev;
    Jsonb *diagnostics;

} Db4aiNative;

typedef struct Db4aiModel
{
    int64 id;
    Db4aiAlgorithm algorithm;
    int nfeatures;
    int noutputs;
    char *format;
    bytea *data;
    Jsonb *metrics;
} Db4aiModel;

extern int db4ai_work_mem;
extern int db4ai_xgboost_max_input_mb;
extern int db4ai_max_model_mb;
extern int db4ai_max_rows;

extern Db4aiAlgorithm db4ai_algorithm(const char *name);
extern const char *db4ai_algorithm_name(Db4aiAlgorithm algorithm);
extern bool db4ai_supervised(Db4aiAlgorithm algorithm);
extern bool db4ai_is_xgboost(Db4aiAlgorithm algorithm);
extern void db4ai_validate_option_name(const char *name, Db4aiAlgorithm algorithm);
void db4ai_options(Jsonb *json, Db4aiAlgorithm algorithm, Db4aiOptions *out);
extern Jsonb *db4ai_options_json(const Db4aiOptions *options, Db4aiAlgorithm algorithm);
extern JsonbValue *db4ai_json_field(Jsonb *json, const char *key, bool required);
extern double db4ai_json_number(Jsonb *json, const char *key, double def);
extern bytea *db4ai_bytes(const void *data, Size len);
extern void db4ai_check_finite(double x);

extern Db4aiData *db4ai_data_open(const char *query, Db4aiAlgorithm algorithm, const Db4aiOptions *options);
extern bool db4ai_data_next(Db4aiData *data, double *row);
extern void db4ai_data_rewind(Db4aiData *data);
extern void db4ai_data_close(Db4aiData *data);
extern double *db4ai_features(ArrayType *array, int expected);

extern Db4aiNative *db4ai_native_train(Db4aiData *data, Db4aiAlgorithm algorithm,
                                      const Db4aiOptions *options);
extern bytea *db4ai_native_save(const Db4aiNative *model);
extern Db4aiNative *db4ai_native_load(const Db4aiModel *model);
extern double db4ai_native_predict(const Db4aiNative *model, const double *features);
extern void db4ai_native_transform(const Db4aiNative *model, const double *features, double *out);

extern void db4ai_xgb_train(Db4aiData *data, Db4aiAlgorithm algorithm,
                          const Db4aiOptions *options, const char *format, Db4aiModel *out);
extern void *db4ai_xgb_load(const Db4aiModel *model);
extern double db4ai_xgb_predict(void *predictor, const double *features, int nfeatures);

extern int64 db4ai_model_store(const char *name, const char *query, const Db4aiData *data,
                              const Db4aiModel *model, Jsonb *options, Oid caller);
extern Db4aiModel *db4ai_model_load(const char *name, Oid caller);
extern bool db4ai_model_delete(const char *name, Oid caller);

#include "extended.h"

#endif
