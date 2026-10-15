# Validation record

## Local validation, 2026-10-15

- Linux x86-64, GCC, PostgreSQL 18.0 built from the official SHA-256-verified source distribution.
- Vendored XGBoost 3.4.2, static PIC build, CPU-only, OpenMP disabled.
- All first-party sources compile as C11 with `-Werror=implicit-function-declaration`.
- `test/run.sh` completed successfully with both `WITH_XGBOOST=1` and `WITH_XGBOOST=0`.
- Native numerical checks: linear regression, logistic classification/probability, linear SVM, KMeans clustering and PCA projection.
- XGBoost checks: all four objective entry points, JSON/UBJSON prediction equivalence, reloadable exported JSON, label validation and dense-input limit.
- Integration checks: model name changes per row, replacement during a single outer statement, transaction rollback, cross-role RLS, restored identity after a registry exception, forbidden input tables, prohibited direct artifact updates, malformed queries/inputs, normalized-input spill and one-time SELECT evaluation, timeout cleanup, new-backend reload and pg_dump/pg_restore including the identity sequence.
- `ldd pg_db4ai.so` showed only standard OS/C++ runtime dependencies, with no dynamically linked libxgboost or libdmlc. The native-only build had no C++ runtime dependency.

CI independently targets PostgreSQL 16, 17 and 18 and a native-only configuration. Consult the GitHub Actions result for the exact commit being deployed; a configured matrix is not itself evidence of passing tests.

## Not established by these tests

- Complete behavioral or bit-level equivalence with all original openGauss DB4AI algorithms/options.
- Import compatibility for old openGauss model artifacts.
- Production throughput, scalability or a total-memory hard limit for XGBoost.
- Windows/macOS build support, parallel-query safety, distributed training or external-memory XGBoost.
- Statistical generalization quality on application-specific data. Training-set metrics are not held-out validation.
