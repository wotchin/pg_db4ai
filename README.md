# pg_db4ai

PostgreSQL-native, in-process machine learning based on the DB4AI design in openGauss. The first-party implementation is **C11** and intentionally uses PostgreSQL memory contexts, SPI, tuples, tuplestore and error handling. There is no database-independent abstraction layer.

**Status: working 0.1 implementation, not a drop-in migration of every openGauss DB4AI feature.** Numerical and integration tests pass locally on PostgreSQL 18.0 in both bundled-XGBoost and native-only configurations. See [validation](docs/VALIDATION.md) and [porting boundaries](docs/PORTING.md); matching algorithm names does not imply identical original solvers or old-model compatibility.

## Build and install

Linux, PostgreSQL server development headers, a C compiler and GNU Make are required. Bundled XGBoost additionally requires CMake 3.18+ and a C++17 compiler. CI targets PostgreSQL 16, 17 and 18; each major version needs its own compiled extension.

```sh
git clone https://github.com/wotchin/pg_db4ai.git
cd pg_db4ai
make PG_CONFIG=/path/to/pg_config WITH_XGBOOST=1 XGB_JOBS=2
sudo make PG_CONFIG=/path/to/pg_config WITH_XGBOOST=1 install
# Run as a non-root OS user; tests create an isolated disposable cluster.
make PG_CONFIG=/path/to/pg_config WITH_XGBOOST=1 test
```

No submodule initialization, pip installation, system libxgboost or build-time download is required. Source and dependency sources are committed under `third_party/xgboost`. Use `WITH_XGBOOST=0` for only native algorithms, requiring no C++ runtime. Run `make clean` when changing PostgreSQL versions or this option. `make clean-vendor-build` removes the third-party build cache.

```sql
CREATE EXTENSION pg_db4ai;
```

The model table is **`db4ai.pg_db4ai_models`**, an ordinary WAL-logged extension table, not a patched system catalog. No custom grammar, PostgreSQL core changes or shared_preload_libraries are required.

## SQL API

Supervised training SQL returns **label first, then scalar features**. KMeans/PCA receive only features. Numeric/boolean columns are supported; cast domains and expand arrays explicitly. NULL/non-finite values are rejected. Prediction features must match training order. Poorly conditioned features should be scaled in SQL before native gradient descent.

```sql
SELECT db4ai.train(
    'house_price_v1', 'linear_regression',
    'SELECT price, area, age FROM public.houses',
    '{"max_iterations":300,"learning_rate":0.01}'::jsonb
);
SELECT id, db4ai.predict_float8('house_price_v1', ARRAY[area, age]::float8[])
FROM public.houses;
SELECT db4ai.explain_model('house_price_v1');
SELECT db4ai.export_model('house_price_v1'); -- bytea, native JSON payload
SELECT db4ai.drop_model('house_price_v1');
```

| Algorithm | Implementation | Prediction |
|---|---|---|
| `linear_regression` | C mini-batch gradient descent, intercept | `predict_float8` |
| `logistic_regression` | C binary logistic regression, 0/1 labels | probability: `predict_float8`; class: `predict_bool` |
| `svm_classification` | C linear hinge loss, 0/1 labels | margin: `predict_float8`; class: `predict_bool` |
| `kmeans` | C Lloyd iteration, seeded reservoir initialization | `predict_cluster`, 1-based ID |
| `pca` | C centered sample covariance + Jacobi eigensolver | `transform`, component array |
| `xgboost_regression_squarederror` | XGBoost reg:squarederror | `predict_float8` |
| `xgboost_binary_logistic` | XGBoost binary:logistic | `predict_float8` / `predict_bool` |
| `xgboost_regression_logistic` | XGBoost reg:logistic, labels in [0,1] | `predict_float8` |
| `xgboost_regression_gamma` | XGBoost reg:gamma, positive labels | `predict_float8` |

Unknown/inapplicable hyperparameters are errors. Defaults: max_iterations=100; learning_rate=0.01 for native GD / 0.1 for XGBoost; l2=0 except native SVM=0.01; native batch_size=128; native tolerance=1e-8; KMeans k=8, seed=42; PCA n_components=2. XGBoost also accepts max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1, seed=42. `src/options.c` defines validated ranges. `explain_model` reports effective applicable settings and training-set metrics, not holdout validation.

## Vendored XGBoost and durable models

XGBoost **3.4.2**, including dmlc-core and bundled dependencies, is pinned by source-release SHA-256 in `third_party/xgboost.lock.json`. The manifest records file hashes; `python3 tools/vendor_xgboost.py --verify` checks a clean source checkout. The importer is a maintainer-only tool, never an ordinary build step.

Our adapter is C; upstream XGBoost remains C++ behind `xgboost/c_api.h`. Static PIC archives are linked into pg_db4ai.so. No separate libxgboost.so is installed or loaded. Standard OS/C++ runtime dependencies remain. The default CPU-only build disables OpenMP, CUDA, NCCL, federation and GPU plugins, with one XGBoost compute thread per backend.

```sql
SELECT db4ai.train('xgb_price', 'xgboost_regression_squarederror',
                  'SELECT price, area, age FROM public.houses',
                  '{"max_iterations":100,"max_depth":4}'::jsonb);
-- Defaults to UBJSON (format name: ubj).
SELECT db4ai.train('xgb_price_json', 'xgboost_regression_squarederror',
                  'SELECT price, area, age FROM public.houses',
                  '{"max_iterations":100}'::jsonb, 'json');
```

Persistence uses **XGBoosterSaveModelToBuffer / XGBoosterLoadModelFromBuffer**, not runtime snapshots or display-only tree dumps. Native models use explicit versioned JSON, never raw structs, pointers or Datum arrays. Native float encoding is independent of extra_float_digits. These artifacts preserve inference models, not a guarantee of bit-exact training-state resumption.

Both model data and the identity sequence are registered for pg_dump/pg_restore. Owners use regrole so dumps encode role names; restore required roles first. Old openGauss model blobs need an explicit converter and are not supported here. Public arbitrary third-party model import is deliberately absent.

## Transactions, permissions and resource limits

Training executes as the caller with normal source-table permissions/RLS. Only fixed parameterized registry writes use the extension owner's identity, restored on success and errors. User training SQL never runs in that privileged block. Regular users cannot modify artifacts directly; model reads are owner-filtered by RLS.

Names are unique per current role. Models are immutable; replace by drop plus train in one transaction. Publication occurs only after training succeeds, and rollback restores committed state. Cross-role model sharing is not implemented. Administrators must manage dropped-role orphan entries; regrole alone does not establish a native object dependency.

The input SELECT executes once, through batched SPI fetches, into a normalized spillable tuplestore. All epochs reread that fixed input. Add ORDER BY for repeatable GD row order. Exactly one SELECT is admitted; modifying CTEs, INTO and row locks are rejected. This is not a sandbox against side effects in invoker-permitted user-defined functions.

| Setting | Default | Scope |
|---|---:|---|
| `db4ai.work_mem` | 16 MB | Tuplestore memory before spill; user-settable |
| `db4ai.max_rows` | 10,000,000 | Maximum input rows; administrator-settable |
| `db4ai.max_model_mb` | 64 MB | Serialized-model cap; administrator-settable |
| `db4ai.xgboost_max_input_mb` | 128 MB | Dense float32 input-copy cap; administrator-settable |

**The XGBoost input cap is not a total-process memory limit.** DMatrix/histogram/model allocations are additional. External-memory XGBoost is not implemented. Maximum features: 1,024 generally, 256 for PCA. Cancellation is checked in PG/native loops and between boosting iterations, not inside every upstream call. Training holds a transaction/snapshot.

Predictors cache by model name, role, statement and command ID. Third-party handles have explicit cleanup and memory-context callbacks. SQL functions remain PARALLEL UNSAFE until separately audited. Windows/macOS support and performance claims are not established.

## Documentation and license

- [Architecture and security](docs/ARCHITECTURE.md)
- [Original-source mapping and deliberate semantic differences](docs/PORTING.md)
- [Validation record](docs/VALIDATION.md)
- [Upstream notices](THIRD_PARTY_NOTICES.md)

First-party and openGauss-derived work is licensed under Mulan PSL v2. All upstream XGBoost/dependency licenses remain in their original source directories. The openGauss snapshots/ data-versioning feature is excluded.
