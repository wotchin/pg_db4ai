# PostgreSQL-only architecture (0.2)

## Components

| File | Responsibility |
|---|---|
| `src/pg_db4ai.c` | PG function entry points, training lifecycle, dispatch, metrics and keyed predictor cache |
| `src/models.c` | `db4ai.pg_db4ai_models` access, fixed parameterized privileged writes |
| `src/options.c` | 11 algorithm registry entries, per-algorithm options, aliases/profiles and JSON helpers |
| `src/data.c` | One caller-identity source execution, typed inspection, normalized spillable input |
| `src/categories.c` | PG typed dictionaries, collision-safe hash/equality, canonical persistent IO |
| `src/shuffle.c` | Seeded epoch batch/row shuffle, full-cache and bounded replacement paths |
| `src/gradient.c` | GD, normalized GD, OVA and matrix-free stochastic PCA |
| `src/kernel.c` | Gaussian Fourier / polynomial Maclaurin random feature maps |
| `src/native.c` | KMeans, Jacobi PCA, native serialization and prediction |
| `src/bayes.c` | Typed marginal/conditional statistics and original-family joint-selectivity propagation |
| `src/shufflescan.c` | Explicit SQL CustomPath/CustomScan plus FunctionScan fallback |
| `src/xgboost.c` | C API handles, full-DMatrix CPU training and JSON/UBJSON model IO |

No database-independent allocator/type/error wrapper is intended. Headers and
algorithms use PG types directly. First-party sources compile as C11. Upstream
vendored XGBoost is C++ behind a C ABI and does not call PG from worker threads.

## Training

`db4ai.train` validates algorithm/options, creates a dedicated memory context,
and consumes exactly one read-only SELECT under the caller's identity. SPI
fetches batches; raw tuples are materialized for reliable array-shape/label
inspection. Accepted data is normalized into another spillable tuplestore.
GD/NGD/OVA/PCA read it through the shuffle runtime; KMeans/Bayes perform their
required passes. No epoch reruns source SQL. The two stores and shuffle cache
have separate budgets, not a shared RSS cap.

The first epoch is sequential by default. When every batch fits, later epochs
uniformly permute batches. Under a bounded cache, input replaces a random cache
unit or is emitted directly, followed by random drain. Every row is emitted once;
partial final batches carry their actual row count. Row mode reassembles output
rows into numerical batches. The source order, seed, batch size and cache
capacity are all part of reproducibility.

Typed classification labels retain PG hash/equality semantics; normalized rows
contain dictionary indices. Bayes maintains one dictionary per variable,
including an explicit NULL category. Exported JSON contains type namespace/name,
typmod, collation and typed text, never backend-local OIDs/pointers. Dictionary IO
uses canonical date/time settings. Consumers must still provide compatible type
implementations/collations when restoring across installations.

Only after a model is successfully encoded is it written to the model table.
PG errors/cancellation unwind PG contexts. XGBoost DMatrix/Booster handles have
explicit cleanup and reset callbacks. No PG longjmp is intentionally propagated
through a first-party C++ stack because there is no first-party C++ stack.

## Explicit SQL CustomScan

`LOAD 'pg_db4ai'` registers methods and chains `set_rel_pathlist_hook` before
planning. Only a simple call to the extension-owned `db4ai.shuffle_scan` with
eligible constant arguments is replaced. The original function expression is
retained for plan dependencies, and the custom private data is serializable
PostgreSQL nodes. Complex/whole-row/LATERAL/ordinality calls retain ordinary
FunctionScan execution. A SECURITY DEFINER or function-SET modified function is
not intercepted, preserving normal fmgr semantics.

Begin rechecks EXECUTE ACL; Exec lazily runs caller-identity SPI, applies the
shuffle reader and uses ExecScan for normal qualification/projection. ReScan
rewinds fixed materialized input. End releases contexts. Explain reports rows,
spill, seed, epoch and cache capacity; plain EXPLAIN never opens the source.
Parallel flags are not advertised. The SQL tuple reader and the numerical batch
shuffle runtime are separate: training does not need to invoke this SQL node.

## Catalog and cache

`db4ai.pg_db4ai_models` remains an ordinary WAL-logged extension-owned table with
role-isolated reads and no direct non-admin artifact writes. Source SQL never
runs as the extension owner. Only fixed parameterized model writes enter a
short privileged scope, restoring identity on both success and failure.
Models are immutable; drop/train inside a transaction replaces one. Predictor
cache keys include model name, caller role, statement and command boundary;
model names may vary by row. User label types require compatible names on restore.

Both table contents and the identity sequence are registered for logical backup.
Roles must be restored first. Native JSON payload v1 and v2 are read by 0.2;
upgrade does not mutate old blobs. The table's format envelope is distinct from
the algorithm-specific JSON payload version. No arbitrary untrusted artifact
import endpoint is exposed.

## Limits

Max input rows, feature/dictionary sizes, model allocations, spill and algorithm
budgets are guarded, but these are not hard whole-backend RSS controls. XGBoost
uses a complete DMatrix and additional internal memory. `max_seconds` is a soft
training-phase limit; statement_timeout includes source SQL too. Native loops
check interrupts; an individual upstream XGBoost call may delay cancellation.
All APIs remain PARALLEL UNSAFE. No implicit distributed/GPU training or automatic
optimizer cardinality-model lifecycle is introduced.
