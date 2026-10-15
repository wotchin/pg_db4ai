# DB4AI porting boundaries

Reference: `opengauss-mirror/openGauss-server`, commit `8f33f38a306dd4b155b6e77d1bc15554b1afcf8f`, excluding `src/gausskernel/dbmind/db4ai/snapshots/`.

This is a PostgreSQL C adaptation, **not an assertion that the original C++ source tree has already been mechanically translated and behaviorally matched in full**. The source input callbacks and algorithm lifecycle inform the runtime, but PostgreSQL-specific structures remain intentional. There is no database-independent ABI or plugin-host abstraction.

| Original area | New implementation | Boundary |
|---|---|---|
| `db4ai_api.h`, `executor/direct.cpp` | `src/db4ai.h`, `src/pg_db4ai.c` | PG-native C structs and lifecycle, no PlanState/TrainModel node |
| `nodeTrainModel.cpp` fetch/rescan | `src/data.c` | SPI cursor + normalized, spillable tuplestore; one source execution |
| `catalog/model_warehouse.cpp` | `src/models.c`, extension SQL | `db4ai.pg_db4ai_models`, ordinary table and role isolation |
| `commands/predict_by.cpp` | `src/pg_db4ai.c` | Typed SQL functions; keyed predictor cache with explicit invalidation |
| Linear/logistic gradient equations | `src/native.c` | Scalar-loop mini-batch C implementation of dense numeric path |
| SVM, KMeans, PCA | `src/native.c` | C implementations with the deliberately narrowed algorithms below |
| `algorithms/xgboost.cpp` | `src/xgboost.c` | Current upstream C API, complete-DMatrix training and model IO |
| Custom SQL grammar, DestReceiver and executor node registration | Not carried over | Function API; no PostgreSQL core patch |
| `snapshots/` data versioning | Excluded | Not the same as the required MVCC/training input consistency |

## Deliberate semantic differences

- Native GD uses fixed input order and ordinary mini-batch gradient descent. The original shuffle cache, NGD and kernel/random-feature transformations have not been ported.
- Classification currently accepts 0/1 labels, not arbitrary SQL-type category dictionaries. Logistic scalar prediction is a probability; SVM scalar prediction is a margin. Typed boolean functions return labels.
- SVM is linear hinge loss with L2 regularization; nonlinear SVM kernels are unsupported.
- KMeans uses seeded uniform-reservoir initialization and squared Euclidean distance. Original KMeans||/Random++ and other distance options are not emulated. Empty clusters retain their previous centroids.
- PCA uses centered sample covariance and a Jacobi eigensolver, not the original stochastic PCA optimizer. It does not standardize feature variance automatically.
- NULL/non-finite input is rejected rather than silently imputed, skipped or converted. Preprocessing is explicit SQL.
- XGBoost trains each boosting iteration on the entire admitted DMatrix. It does not reuse the old chunk-by-chunk warm-start procedure. All four exposed objectives have input-label checks.
- Training metrics describe the training data, not a holdout validation set. Users must create a separate evaluation split.
- Models use new durable formats. Existing openGauss serialized structs/Datum arrays and XGBoost memory snapshots are not imported directly.
- Multiclass, internal Bayes-network models, distributed/parallel training, auto-hyperparameter search and CustomScan integration are outside 0.1.

Before calling this a full replacement for a particular existing deployment, run that deployment's data, algorithms and option set against both engines. Compare prediction semantics, metrics, categorical encoding and acceptable numerical tolerances; do not infer equivalence from matching algorithm names.
