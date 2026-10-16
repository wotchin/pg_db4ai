/* SPDX-License-Identifier: MulanPSL-2.0
 * PG-native DB4AI 0.2 interfaces. No database-independent ABI is intended.
 */
#ifndef DB4AI_EXTENDED_H
#define DB4AI_EXTENDED_H
#include "common/pg_prng.h"
#include "lib/stringinfo.h"
#include "utils/typcache.h"
#include "utils/hsearch.h"

typedef struct Db4aiDictionary
{
    Oid type;
    int32 typmod;
    Oid collation;
    FmgrInfo equal;
    FmgrInfo hash;
    FmgrInfo output;
    int16 typlen;
    bool typbyval;
    int count;
    int capacity;
    int limit;
    Datum *values;
    bool *nulls;
    int64 *frequency;
    int *next;
    HTAB *buckets;
    MemoryContext context;
    Size bytes;
} Db4aiDictionary;

typedef enum Db4aiShuffleMode
{
    DB4AI_SHUFFLE_NONE, DB4AI_SHUFFLE_BATCH, DB4AI_SHUFFLE_ROW, DB4AI_SHUFFLE_CORGIPILE
} Db4aiShuffleMode;
typedef struct Db4aiShuffle Db4aiShuffle;
typedef struct Db4aiRowStore Db4aiRowStore;
extern Db4aiRowStore *db4ai_rows_create(int ncolumns, Size budget);
extern void db4ai_rows_append(Db4aiRowStore *store, const double *row);
extern void db4ai_rows_seal(Db4aiRowStore *store);
extern int db4ai_rows_slice(Db4aiRowStore *store, int64 start, int maximum, const double **rows);
extern int db4ai_rows_read(Db4aiRowStore *store, int64 start, int maximum, double *out);
extern bool db4ai_rows_spilled(Db4aiRowStore *store);
extern void db4ai_rows_stats(Db4aiRowStore *store, uint64 *read, uint64 *written, uint64 *seeks);
extern void db4ai_rows_free(Db4aiRowStore *store);
typedef struct Db4aiBayes Db4aiBayes;

extern bool db4ai_classifier(Db4aiAlgorithm a);
extern char *db4ai_json_string(Jsonb *json, const char *key, const char *def);
extern bool db4ai_json_boolean(Jsonb *json, const char *key, bool def);
extern double *db4ai_read_array(Jsonb *json, const char *name, int count);
extern void db4ai_write_array(StringInfo out, const double *values, int count);
extern Db4aiDictionary *db4ai_dictionary_new(Oid type, int32 typmod, Oid coll, int limit);
extern int db4ai_dictionary_index(Db4aiDictionary *d, Datum value, bool isnull, bool insert);
extern void db4ai_dictionary_save(StringInfo out, const Db4aiDictionary *d);
extern Db4aiDictionary *db4ai_dictionary_load(Jsonb *json);
extern char *db4ai_dictionary_text(const Db4aiDictionary *d, int index);
extern Db4aiShuffle *db4ai_shuffle_create(Db4aiData *data, const Db4aiOptions *options);
extern void db4ai_shuffle_start(Db4aiShuffle *scan, int epoch);
extern int db4ai_shuffle_next(Db4aiShuffle *scan, const double **rows);
extern void db4ai_shuffle_free(Db4aiShuffle *scan);
extern void db4ai_shuffle_init(void);
extern void db4ai_gradient_train(Db4aiNative *model, Db4aiData *data, const Db4aiOptions *options);
extern void db4ai_stochastic_pca(Db4aiNative *model, Db4aiData *data, const Db4aiOptions *options);
extern void db4ai_kernel_prepare(Db4aiNative *model, const Db4aiOptions *options);
extern void db4ai_kernel_transform(const Db4aiNative *model, const double *input, double *output);
extern int db4ai_native_class(const Db4aiNative *model, const double *features);
extern void db4ai_native_scores(const Db4aiNative *model, const double *features, double *scores);
extern void db4ai_bayes_train(Db4aiData *data, const Db4aiOptions *options, Db4aiModel *model);
extern Db4aiBayes *db4ai_bayes_load(const Db4aiModel *model);
extern double db4ai_bayes_predict(Db4aiBayes *model, Datum *values, bool *nulls, Oid *types, int count);
#endif
