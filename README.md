# pg_db4ai 0.4: a user-facing Bayes classifier

## Lightweight AutoML (0.5)

`db4ai.automl(name, task, source_sql, options)` fits a bounded sequential portfolio,
with train-only numerical/categorical preprocessing, separate validation/final test,
explicit time/admission budgets and one durable fitted pipeline. Use
`db4ai.predict(name, jsonb)` for original named features, or existing typed APIs
for numeric-only inputs. See [AutoML contracts and examples](docs/AUTOML.md).
This is not a hard process-memory limit or a universal SOTA guarantee.


`gaussian_naive_bayes` replaces the retired internal joint-selectivity estimator.
It supports typed binary/multiclass labels and normalized `predict_proba` /
`predict_log_proba` outputs, with numeric/array features and versioned JSON.
The old `bayes_net_internal` trainer and `predict_joint` API are removed; existing
artifacts remain exportable/deletable and are not converted. Upgrade with
`ALTER EXTENSION pg_db4ai UPDATE TO '0.4.0'` after installing and reconnecting.
See [usage, assumptions and migration](docs/NAIVE_BAYES.md).

## Earlier 0.3 changes



PostgreSQL-only C11. New: dense numerical replay, prepared-feature caching,
CorgiPile block/window row shuffle, explicit model-DDL interpreter, richer model
diagnostics, fractional time budgets, and optional XGBoost training OpenMP.

ABO optimizer integration and snapshots are excluded. Numerical implementation
choices are accepted, not a missing-function checklist. Bare CREATE MODEL is NOT
registered in stock PG; use the explicit `db4ai.execute_ddl(text)` entrypoint.

[Functional audit](docs/FUNCTIONAL_COVERAGE.md) · [DDL](docs/MODEL_DDL.md) ·
[CorgiPile](docs/CORGIPILE.md) · [Current-only measurements](benchmarks/native/README.md)

Install and reconnect, then `ALTER EXTENSION pg_db4ai UPDATE TO '0.3.0';`.
Default builds still use single-thread XGBoost without OpenMP. Optional
`make WITH_XGBOOST=1 WITH_OPENMP=1` enables nthread=1..64 for training, with the
system OpenMP runtime dependency. Prediction stays single-thread per SQL call.
`SELECT db4ai.build_info();` reports actual build capabilities.

---

# pg_db4ai

PostgreSQL-native, in-process machine learning adapted from openGauss DB4AI.
First-party code is **C11** and deliberately uses PostgreSQL memory contexts,
SPI, slots, tuplestore, error handling and CustomScan. There is no
cross-database abstraction layer and no PostgreSQL core patch.

**0.2 implements all 11 algorithm entries in the pinned original registry**, plus
GD/normalized-GD, One-vs-All, Gaussian/polynomial feature mappings, stochastic
PCA, KMeans seeding/norm families, typed labels and the training shuffle cache.
It also adds an explicit SQL `ShuffleScan` CustomScan provider.
This is algorithm-path coverage, **not a claim of bitwise equivalence or a
completed live-openGauss differential qualification**. See
[porting boundaries](docs/PORTING.md) and [validation](docs/VALIDATION.md).

[中文说明与示例](README.zh-CN.md)

## Build and upgrade

Linux, PostgreSQL server development headers, GNU Make and a C compiler are
required. Bundled XGBoost also needs CMake 3.18+ and a C++17 compiler. CI targets
PG 16/17/18; each PostgreSQL major requires a separate build.

```sh
git clone https://github.com/wotchin/pg_db4ai.git
cd pg_db4ai
make PG_CONFIG=/path/to/pg_config WITH_XGBOOST=1 XGB_JOBS=2
sudo make PG_CONFIG=/path/to/pg_config WITH_XGBOOST=1 install
# Non-root OS user; creates a disposable database cluster.
make PG_CONFIG=/path/to/pg_config WITH_XGBOOST=1 test
```

Use `WITH_XGBOOST=0` for all native algorithms without a C++ runtime dependency.
Run `make clean` when changing PostgreSQL or this option; `make clean-vendor-build`
removes the third-party build cache. Normal builds download no source, use no
submodules, and require neither pip nor system `libxgboost.so`.

```sql
CREATE EXTENSION pg_db4ai; -- fresh installation
-- Or, after installing the new library/scripts, upgrade an existing 0.1:
ALTER EXTENSION pg_db4ai UPDATE TO '0.2.0';
```

Reconnect existing sessions after replacing the shared library: loaded C code is
not hot-swapped by ALTER EXTENSION. Upgrade adds APIs without rewriting existing
model payloads. Version-1 native JSON remains readable. Old 0.1 classifiers have
no typed dictionary; use their legacy prediction API or retrain for typed labels.
This does not import old openGauss binary model blobs.

## Algorithms and prediction APIs

| Registry entry | Training paths | Prediction |
|---|---|---|
| `linear_regression` | GD / normalized GD | `predict_float8` |
| `logistic_regression` | GD / NGD, typed binary labels | `predict_label`, `predict_text`, `predict_scores` |
| `svm_classification` | GD / NGD; linear, Gaussian, polynomial maps | typed labels or scalar margin |
| `multiclass` | One-vs-All logistic / SVM, GD / NGD | typed labels, class-ordered scores |
| `kmeans` | Random++, KMeans\|\|, optional reservoir; L1/L2/L2_Squared/Linf | `predict_cluster` (1-based) |
| `pca` | Original-family matrix-free stochastic updates; optional Jacobi | `transform` |
| `gaussian_naive_bayes` | Supervised Gaussian NB with typed binary/multiclass labels | `predict_label`, `predict_proba`, `predict_log_proba` |
| `xgboost_regression_squarederror` | `reg:squarederror` | `predict_float8` |
| `xgboost_binary_logistic` | `binary:logistic` | `predict_float8`, `predict_bool` |
| `xgboost_regression_logistic` | `reg:logistic` | `predict_float8` |
| `xgboost_regression_gamma` | `reg:gamma` | `predict_float8` |

Supervised input is **label first, then features**. KMeans/PCA receive only
features. Gaussian NB also uses the first column as its supervised label.
Features support numeric/boolean scalars and fixed-width `float4[]`/`float8[]`
columns, flattened in query-column order. Prediction uses the same flat order.
Native classification uses PG hash/equality and concrete label types, including
text, enum, integer and numeric values beyond float8 precision. Domain label types
are flattened to their base types. Pseudo-types and anonymous records are not
supported. Gaussian NB uses the explicit error/legacy numeric NULL policy.

```sql
SELECT db4ai.train('price', 'linear_regression',
    'SELECT price, area, age FROM public.houses',
    '{"optimizer":"ngd","max_iterations":100,"shuffle":"batch"}');
SELECT db4ai.predict_float8('price', ARRAY[120.0, 15.0]);
SELECT db4ai.explain_model('price');
SELECT db4ai.export_model('price'); -- bytea
SELECT db4ai.drop_model('price');
```

`predict_float8` retains binary probability/margin semantics and returns a class
index for multiclass. Prefer `predict_label(name, features, NULL::label_type)`
for the actual classification label. Numeric/bool 0/1 labels are ordered 0,1;
other labels use first occurrence. `class_labels` reveals the score order.
OVA logistic scores are independent binary probabilities, **not softmax**;
SVM scores are margins. Metrics are training-set metrics, not holdout validation.

### Options and input profiles

`profile:"stable"` is the default: conservative native learning-rate defaults,
Jacobi/centered PCA, and NULL rejection. `profile:"opengauss"` selects original-
style GD defaults, legacy NULL policy and stochastic/uncentered PCA. It is **not**
a switch promising an identical original execution. Explicit values override the
profile. Unknown, inapplicable and conflicting alias options raise errors.

Common GD controls: `optimizer` (`gd`/`ngd`), `batch_size`, `learning_rate`,
`decay`, `tolerance`, `max_iterations`, `max_seconds`, `seed`, `verbose`.
SVM: `kernel`, `components`, `gamma`, `degree`, `coef0`, `lambda`.
Multiclass: `classifier` (`logistic_regression`/`svm_classification`).
KMeans: `num_centroids` (`k`), `num_features`, `seeding_function`,
`distance_function`, `seeding_rounds`, `oversampling`.
PCA: `number_components` (`n_components`), `pca_solver`, `center`.
Validated ranges and effective defaults live in `src/options.c`.

Legacy NULL policy: supervised NULL labels are skipped; GD scalar NULL features
become zero; a NULL whole feature array is skipped and NULL elements are rejected.
XGBoost rows with NULL fields are skipped. Non-finite numeric features are errors.
Choose preprocessing explicitly for application use; automatic SQL feature
standardization is not performed.

## ShuffleScan: two related implementations

**Training shuffle** (`src/shuffle.c`) supplies GD/NGD/OVA/stochastic PCA. Source
SQL runs once into a normalized spillable tuplestore. Epoch zero is sequential
unless `shuffle_first_epoch:true`. Later epochs shuffle cached batches. If all
batches fit, permutation is uniform; otherwise bounded random replacement and
random drain give exact-once coverage, **not a globally uniform permutation**.
`shuffle:"row"` shuffles rows then reassembles the requested training batch size.
`shuffle:"none"` keeps the fixed input order. `shuffle_buffer_size` counts batches
in batch mode and rows in row mode, subject to the cache memory budget.

**SQL ShuffleScan** (`src/shufflescan.c`) exposes arbitrary typed tuples:

```sql
LOAD 'pg_db4ai'; -- load the path hook BEFORE planning, in each relevant session
EXPLAIN (ANALYZE, COSTS OFF)
SELECT x * 2
FROM db4ai.shuffle_scan(
    'SELECT i FROM generate_series(1,100)i', 42, 16, 1
) AS s(x integer)
WHERE x > 10;
```

Eligible calls show **`Custom Scan (ShuffleScan)`**, with input rows, spill status,
cache size, seed and epoch. `epoch=0` is sequential. ExecReScan replays the fixed
input with the same seed/epoch, not the original SQL. An EXPLAIN without ANALYZE
does not execute the source. Function EXECUTE privileges and source RLS remain
enforced. Parameterized/LATERAL/whole-row/ordinality cases can use the ordinary
FunctionScan fallback, which preserves the same SQL shuffle semantics. The hook
does not rewrite arbitrary queries or add random sorts. No preload requirement.

For reproducibility, provide a source ORDER BY and fix seed, batch/cache size,
profile and software version. A seed alone cannot stabilize a nondeterministically
ordered source. Training epochs do not use the SQL CustomScan node: they use the
batch-oriented runtime to avoid per-tuple SQL invocation overhead.

## Persistence, permissions and resources

Models live in **`db4ai.pg_db4ai_models`**, an ordinary WAL-logged extension table.
Source SQL uses caller privileges/RLS. Only fixed parameterized catalog writes
use the extension owner's identity, restored on success/error. Models are
immutable and names are per role; replace using drop/train in one transaction.
Regular users cannot update artifacts directly. Roles must exist before restore;
regrole preserves names but is not an automatic DROP ROLE dependency. Cross-role
model sharing and automatic orphan cleanup are not implemented.

Native JSON v2 stores explicit dimensions, typed dictionaries, weights, actual
random feature maps and PCA means/bases/eigenvalue estimates. Type identities use
namespace/name and typmod, not serialized OIDs. Date/time dictionary IO uses
canonical settings. Custom type input/output code and collations must remain
compatible across servers. Artifacts are inference models, not resumable training
snapshots.

Vendored **XGBoost 3.4.2** and dmlc-core remain upstream C++, used only through the
C API. Static PIC libraries are linked into `pg_db4ai.so`; standard OS/C++ runtime
libraries remain. CPU-only/no-OpenMP is the default; `nthread` is restricted to 1.
JSON/UBJSON persistence uses `XGBoosterSaveModelToBuffer` and
`XGBoosterLoadModelFromBuffer`, never runtime snapshots or tree display dumps.
The default format is `ubj`; pass `'json'` as the last training argument for JSON.
`python3 tools/vendor_xgboost.py --verify` checks pinned source hashes offline.

| Setting | Default | Meaning |
|---|---:|---|
| `db4ai.work_mem` | 16 MB | Per-tuplestore/cache budget before spill or reduction |
| `db4ai.max_rows` | 10,000,000 | Per-call input row limit |
| `db4ai.max_model_mb` | 64 MB | Serialized model cap and relevant algorithm allocation guards |
| `db4ai.xgboost_max_input_mb` | 128 MB | Dense float32 input-copy cap |

These are **not total backend RSS limits**: source and normalized stores, shuffle
cache, numerical workspace and XGBoost allocations are separate. XGBoost still
uses a complete DMatrix, not external-memory training. `max_seconds` is a soft
algorithm-phase budget; PG statement_timeout/cancellation is checked in C loops
and between XGBoost calls, not inside every upstream call. Functions are
PARALLEL UNSAFE. GPU/distributed training, production performance qualification,
old-openGauss-blob import and full SQL grammar compatibility are not implemented.

## Documentation

[Architecture](docs/ARCHITECTURE.md) · [Original-source mapping and differences](docs/PORTING.md) ·
[Validation](docs/VALIDATION.md) · [Changelog](CHANGELOG.md) · [Notices](THIRD_PARTY_NOTICES.md)

First-party and openGauss-derived work uses Mulan PSL v2. Vendored upstream
licenses remain unchanged. The independent `snapshots/` data-versioning feature
is deliberately excluded.
