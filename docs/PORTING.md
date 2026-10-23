> **0.4 scope change:** the internal Bayes joint estimator described in this
> historical 0.2/0.3 record has been removed, and replaced by supervised
> `gaussian_naive_bayes`. New behavior and tests are in [NAIVE_BAYES.md](NAIVE_BAYES.md).
> Historical benchmark evidence is not rewritten.

# Scope update for 0.3

The detailed numerical differences below are accepted implementation choices,
not unresolved functional migration gaps. ABO optimizer integration is excluded
by project scope, as is snapshots. See [FUNCTIONAL_COVERAGE.md](FUNCTIONAL_COVERAGE.md)
for the current capability audit. Model diagnostics, fractional time limits,
optional XGBoost threads and explicit model-DDL interpretation are now included.
The new CorgiPile training iterator and dense/prepared replay are described in
[CORGIPILE.md](CORGIPILE.md); native top-level grammar remains outside ordinary
PG extension hooks, documented in [MODEL_DDL.md](MODEL_DDL.md).

## Historical numerical reference (0.2)

# Original DB4AI coverage and deliberate differences (0.2)

Reference: `opengauss-mirror/openGauss-server` commit
`8f33f38a306dd4b155b6e77d1bc15554b1afcf8f`. `snapshots/` is excluded.
Selected exact source file SHA-256 values are recorded in `UPSTREAM_REFERENCE.json`.
The registry in `catalog/aifuncs.cpp`, rather than an error-message list, defines
coverage. **All 11 registered algorithm entry points now have executing paths.**
This is not a claim that every C++ file was translated line-by-line or that a live
openGauss server has been run against this extension for complete differential
qualification.

| Original source area | 0.2 implementation | Verification and boundary |
|---|---|---|
| `aifuncs.cpp` registry | `options.c` | 11 entries, four XGBoost objectives preserved |
| `db4ai_api.h`, `direct.cpp` | `db4ai.h`, `pg_db4ai.c` | PG-native lifecycle; no cross-DB layer |
| `nodeTrainModel.cpp` fetch/rescan | `data.c` | One caller-identity query; normalized/spillable replay |
| `shuffle_cache.cpp` | `shuffle.c` | Sequential first epoch, full-cache batch permutation, bounded replacement/drain, exact-once accounting |
| Linear/logistic/SVM + `optimizer_gd.cpp` | `gradient.c` | Original gradient families, decay, convergence; hand-calculated update tests |
| `optimizer_ngd.cpp` | `gradient.c` | Normalized gradients: first-pass feature maxima, weight rescaling and accumulated global normalization |
| `multiclass.cpp`, `optimizer_ova.cpp` | `gradient.c`, `categories.c` | Logistic/SVM One-vs-All; typed class dictionary; independent solver state |
| `kernel.cpp`, `matrix.cpp` random maps | `kernel.c` | Gaussian Fourier and polynomial Maclaurin families; stored actual maps and independent oracle |
| `kmeans.cpp` | `native.c` | Random++/KMeans\|\| candidate paths, weighted reduction; four distances |
| `pca.cpp`, `optimizer_pca.cpp` | `gradient.c` | Matrix-free stochastic covariance action and orthonormalization; explicit Jacobi alternative retained |
| Internal Bayes / `bayes.h` | `bayes.c`, `categories.c` | Marginal/conditional equality estimator, DAG propagation, MCV/bins/distinct estimation, Gaussian parents |
| `predict_by.cpp` | `pg_db4ai.c` | Fixed-return and polymorphic typed-label functions, safe keyed cache |
| `model_warehouse.cpp` | `models.c`, SQL | Ordinary `db4ai.pg_db4ai_models`; versioned durable models |
| `algorithms/xgboost.cpp` | `xgboost.c` | Modern C API, four objectives, full DMatrix; JSON/UBJSON model IO |

## What ShuffleScan means here

The pinned source contains `shuffle_cache.cpp`; a repository search did not locate
a node named `ShuffleScan`. Two concrete components implement the requested
capability: a training batch shuffle runtime and a new explicit PostgreSQL SQL
CustomScan node. Training shuffle follows the original limited-cache structure,
not `ORDER BY random()` or an unused API stub. Small datasets give uniform batch
permutation; bounded replacement is deliberately described as nonuniform.
The SQL CustomScan additionally preserves arbitrary PG tuple types, filtering,
projection and rescan semantics. It is only used when its call is eligible;
complex cases use the same semantics through FunctionScan fallback.

## Numerical and behavioral differences

1. Profiles are explicit. Default `stable` keeps conservative extension defaults;
   `opengauss` selects original-style settings and stochastic/uncentered PCA,
   but does not promise original RNG sequences, convergence history or identical
   weights. Keep source ordering and all execution parameters fixed for repeats.
2. GD bias storage is first rather than last coordinate. Shuffle uses the PG PRNG;
   random kernel maps implement the original 48-bit LCG equations separately.
   Numerical reduction order/rounding can differ. NGD means **normalized**, not
   natural gradient descent. OVA models have independent NGD scaling state rather
   than reproducing cross-class shared-state behavior in the old optimizer wrapper.
3. Polynomial coef0=0 is corrected to retain the highest-degree component; the old
   branch that zeroed all coefficients is not preserved. Actual random map arrays
   are persisted and tested against an independent implementation.
4. KMeans implements the two candidate-seeding families but uses explicit seeded
   sampling/oversampling and resource caps, not bit-identical original sampling
   schedules. All four distances use arithmetic centroid updates (L1 is not
   presented as k-medians). Empty clusters retain prior centroids. `stable`
   convergence uses movement; `opengauss` uses relative objective improvement.
   Not every original per-centroid EXPLAIN diagnostic is mirrored.
5. Stochastic PCA uses covariance action and reorthogonalization with different
   initialization/numerical orthogonalization details. Persisted eigenvalue and
   variation fields are estimates. Centering is explicit; feature variance is not
   automatically standardized. Jacobi remains an explicit extension alternative,
   not mislabeled as the original stochastic solver.
6. Class dictionaries use real PG type equality/hash. Numeric/bool 0/1 labels are
   ordered 0,1 for 0.1 probability compatibility; other categories follow first
   occurrence. Domain labels resolve to base types. Anonymous record/pseudo-type
   labels are rejected; the concrete type must provide usable equality/hash/IO.
   Old 0.1 artifacts have no typed dictionaries: no labels are invented for them.
7. `null_policy:error` is default. `legacy` adds algorithm-appropriate row skipping
   and scalar imputation; array shapes/elements and non-finite features are checked.
   Bayes keeps NULL as an actual variable category. Limits on features, dictionaries,
   graph nodes and buffers are intentional and validated.
8. Internal Bayes is a **joint equality/selectivity estimator**, not Naive Bayes
   classification and not a claim of normalized exact inference on arbitrary DAGs.
   It follows conditional/marginal correction propagation; MCV/numeric bins use
   estimated bucket distinct counts. Gaussian parent CDF uses C erfc instead of
   Boost. Typed numeric comparisons, valid diamond-DAG handling, and negative/
   constant/null Gaussian cases are guarded rather than preserving old bugs.
   Optimizer-private range/marginalization helpers, automatic optimizer training
   and model use are not installed into PostgreSQL.
9. XGBoost uses one complete DMatrix across boosting iterations instead of the old
   chunk-wise warm-start procedure. Four objectives and the supported CPU options
   are exposed, but not unrestricted future XGBoost parameter passthrough. Upstream
   C++/standard runtime remains; GPU/distributed/external-memory training is
   not enabled. OpenMP is now optional in 0.3; default nthread is 1; training metrics are not held-out validation.
10. All native models use new explicit JSON and XGBoost uses model IO. Old openGauss
    struct/Datum/memory-snapshot blobs cannot be directly loaded. The extension's
    own 0.1 JSON is supported; tests distinguish this from openGauss compatibility.

## Deliberately outside this extension change

Original custom SQL grammar, full openGauss object-system diagnostics, obsolete
serialized structs, distributed/parallel execution, automatic hyperparameter
search, optimizer lifecycle integration and the separate snapshots feature are
not reproduced. First-party code remains C/PG-only, with no Boost or independent
ML service. Performance, prediction equivalence on deployment datasets and
production readiness need their own qualification; training and mathematical
regression tests are not a substitute for those claims.
