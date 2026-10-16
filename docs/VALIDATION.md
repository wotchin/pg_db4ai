# Validation record

## Local validation — 2026-10-16, 0.2

Linux x86-64, GCC, PostgreSQL 18.0 compiled from the official checksum-verified
source release. XGBoost 3.4.2 is static PIC, CPU-only, no OpenMP. First-party
sources compile as C11 with implicit declarations treated as errors.

`test/run.sh` passed with **WITH_XGBOOST=1 and WITH_XGBOOST=0**. The suite includes:

| Area | Checks |
|---|---|
| Existing behavior | Original 0.1 numerical, permission, rollback, cache, serialization and cancellation regressions |
| GD / NGD | Hand-computed one-batch weights, zero-scale safety and learning-rate decay |
| Typed labels | text, enum, bool, numeric beyond 2^53 and beyond float8 range, canonical IO, 0/1 probability compatibility |
| Multiclass | OVA logistic and SVM+NGD class results and score ordering |
| Kernels | XOR classifications, independent LCG/Box–Muller/Gaussian and polynomial-map calculations, independent inference |
| KMeans | Both seeding families and explicit reservoir, all four norm costs on analytical data |
| PCA | Stochastic centered/uncentered solutions, Jacobi, orthonormal bases and independent projection |
| Training shuffle | Partial batch, full/bounded paths, repeat/different seeds, source executed once |
| Bayes | Analytical chain, valid diamond, categorical/NULL, Gaussian CDF, MCV, bins, sample distinct estimate, int64 identity, invalid graphs |
| SQL ShuffleScan | Actual Custom Plan Provider, filtering/projection, row conservation, full/bounded deterministic permutations, epoch changes, plain EXPLAIN nonexecution |
| Rescan / storage | Recursive rescan source-once, prepared plan, LATERAL/whole-row fallback, NULL/varlena, forced spill |
| Security | Function EXECUTE revocation, source table permission, caller RLS in both shuffle and training, cross-role predictor isolation |
| Upgrade | Install 0.1 and hand-authored version-1 JSON fixtures; ALTER EXTENSION 0.2, unchanged bytes and predictions; new typed training |
| XGBoost | All four objectives, JSON/UBJSON equivalence, reload, option aliases/validation and AUC label checks |
| Logical persistence | All model payload hashes before/after pg_dump/pg_restore, identity sequence, new-backend typed/OVA/Bayes predictions and independent numerical checks again |

`test/numerical_reference.py` uses only the Python standard library and psql. It
recomputes kernel maps and predictions without invoking the extension's numerical
helpers, verifies PCA bases/projections, and evaluates the Gaussian-parent CDF
formula. It is an **independent mathematical oracle**, not a live openGauss server.
Upgrade fixtures are explicitly constructed 0.1-format JSON, not original
openGauss model exports.

Dynamic dependency checks show no libxgboost/libdmlc runtime dependency. The
native-only shared library has no C++ runtime dependency; XGBoost builds retain
standard OS/C++ runtime libraries.

## Remote validation

The committed CI matrix runs the same suite on PG 16, 17 and 18 with XGBoost and
PG 18 without XGBoost. Use the result on the exact commit/PR being deployed. A
configured matrix, old green check, or local PG18 pass is not evidence for all
other PostgreSQL versions. CI logs are uploaded as workflow artifacts.

## Not established

- Bitwise or complete execution equivalence with an independently running
  openGauss instance and every original configuration/data type.
- Direct import of original openGauss binary/memory-snapshot artifacts.
- Performance, scalability, generalization quality or production hardening.
- Total backend RSS limits or instantaneous cancellation inside XGBoost calls.
- Windows/macOS, GPU, parallel/distributed or external-memory XGBoost training.
- Exact normalized joint inference for arbitrary Bayes graphs or optimizer
  integration beyond the explicit equality estimator.
