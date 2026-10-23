# 0.4.0

## 0.5.0

- Budget-bounded sequential AutoML for classification/regression with shared source replay.
- Training-only numeric imputation/scaling, bounded categorical encoding and durable pipeline inference.
- Group-safe random, entity-group or chronological train/validation/test partitions; honest non-publication outcomes.
- Internal typed XGBoost multiclass softprob; native fallback without extra dependencies.
- Caller training/RLS checks, deadline cleanup, model upgrade/restore and independent prediction tests.
- Whole-pipeline reports, finite-portfolio controls and explicitly documented memory/timeout limitations.


- Replace internal ABO-oriented Bayes estimation with supervised Gaussian Naive Bayes.
- Typed binary/multiclass prediction, normalized probabilities and log probabilities.
- Stable moments, variance smoothing/floor and common-tail log-odds cancellation guard.
- Remove predict_joint with dependency-safe SQL upgrade; preserve old artifacts for export/delete.
- New numerical, security, spill, corrupt-model and upgrade/restore tests.

# 0.3.0

- CorgiPile randomized logical blocks and row mixing within bounded windows.
- Dense numeric replay and spillable one-time prepared feature caching.
- Preserve minibatch assembly when a shuffled short unit occurs mid-epoch.
- Exact-range spill reads, phase timers and logical I/O counters.
- Explicit model-DDL scanner/interpreter, DEFAULT support, dry parsing.
- KMeans cluster distance diagnostics and PCA selected-component diagnostics.
- Fractional max_seconds; optional WITH_OPENMP training nthread support.
- Build/header/configuration invalidation; 0.2->0.3 upgrade and mathematical,
  security/RLS, spill, cancellation and partial-minibatch regressions.
- ABO excluded; numerical implementation differences accepted as design choices.
- External baselines are frozen: only current extension measurements/ablations.

# Changelog

## 0.2.0 — 2026-10-16

- Complete executing paths for the 11 algorithms in the pinned original registry.
- Add normalized GD, OVA logistic/SVM, typed classification labels and score APIs.
- Add persisted Gaussian/polynomial random feature maps and stochastic PCA.
- Add Random++/KMeans|| seeding paths and all four original norm choices.
- Add typed Bayes equality/selectivity estimator, graph validation, MCV/bins,
  sample distinct estimation and Gaussian parent statistics without Boost.
- Add full/bounded training batch shuffle, row shuffle and stable-input rescans.
- Add explicit SQL ShuffleScan CustomPath/CustomScan with FunctionScan fallback,
  ACL/RLS checks and EXPLAIN execution statistics.
- Add option profiles, original-style aliases, controlled NULL policy and input
  array support; reject conflicting/inapplicable options.
- Native JSON payload v2 stores typed labels/maps/PCA estimates; v1 remains readable.
- Add 0.1-to-0.2 SQL upgrade, numerical reference, security, rescan/spill and
  restore regressions. No PostgreSQL core or independent-DB ABI introduced.
- Do not restore legacy XGBoost memory snapshots or chunk-wise warm-start training.

See docs/PORTING.md for the exact original behavior not claimed as equivalent.

## 0.1.0 — 2026-10-15

Initial PostgreSQL-only C extension, simplified native algorithms, vendored
XGBoost, ordinary role-isolated model table, durable IO and integration tests.
