FILES['.github/workflows/ci.yml'] = {'old': '803754f10e6feaeee39f51f228d714e6d2c8e58d', 'new': '25aced4f4ee5e07d3a1bebbbf4a8216b33e59d58', 'base': '.github/workflows/ci.yml', 'mode': '100644','edits':[
(18,22,r'''          - {pg: '16', xgboost: '1', openmp: '0'}
          - {pg: '17', xgboost: '1', openmp: '0'}
          - {pg: '18', xgboost: '1', openmp: '0'}
          - {pg: '18', xgboost: '0', openmp: '0'}
          - {pg: '18', xgboost: '1', openmp: '1'}
'''),
(34,35,r'''        run: make -j2 PG_CONFIG=/usr/lib/postgresql/${{ matrix.pg }}/bin/pg_config WITH_XGBOOST=${{ matrix.xgboost }} WITH_OPENMP=${{ matrix.openmp }} XGB_JOBS=2
'''),
(36,37,r'''        run: sudo make PG_CONFIG=/usr/lib/postgresql/${{ matrix.pg }}/bin/pg_config WITH_XGBOOST=${{ matrix.xgboost }} WITH_OPENMP=${{ matrix.openmp }} install
'''),
(40,41,r'''          PG_CONFIG=/usr/lib/postgresql/${{ matrix.pg }}/bin/pg_config WITH_XGBOOST=${{ matrix.xgboost }} WITH_OPENMP=${{ matrix.openmp }} sh test/run.sh | tee integration-test.log
'''),
(51,52,r'''          name: test-results-pg${{ matrix.pg }}-xgb${{ matrix.xgboost }}-omp${{ matrix.openmp }}
''')
]}
FILES['CHANGELOG.md'] = {'old': 'd841c94be5338f16e2a65e157d33387ebb0bdf5a', 'new': '3092def5212f9c0a84fccfa7a8dadd16d59d3b61', 'base': 'CHANGELOG.md', 'mode': '100644','edits':[
(0,0,r'''# 0.3.0

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

''')
]}
FILES['README.md'] = {'old': '45d86b507d56957166232f09d6d529399a809782', 'new': 'f723d600a5013cb0e2fd848a38a1482b7f4a31c1', 'base': 'README.md', 'mode': '100644','edits':[
(0,0,r'''# pg_db4ai 0.3 update

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

''')
]}
FILES['README.zh-CN.md'] = {'old': 'da275ef825182d14b1bd1ed328ea4c2b7ad30728', 'new': '7d1bf1444aa9aee5cfd84cd8eb255f5f3ed5ff40', 'base': 'README.zh-CN.md', 'mode': '100644','edits':[
(1,1,r'''
## 0.3 更新

新增 CorgiPile 两级训练混排、连续数值重扫与可溢写的特征预计算缓存；补齐模型 DDL 显式解析入口、KMeans/PCA 诊断、小数 max_seconds，以及可选 `WITH_OPENMP=1` 的 XGBoost 多线程训练。

**ABO 和 snapshots 不在迁移范围。** RNG、KMeans 采样、随机 PCA 和 OVA 状态的差异是允许的实现选择，不再作为功能缺失。

`SELECT db4ai.execute_ddl($ddl$CREATE MODEL ...$ddl$)` 在插件内解析执行；**原生 PG 顶层裸 CREATE MODEL 仍不支持**。`SELECT db4ai.build_info()` 可查看编译能力。

升级：安装新库、重新连接，然后 `ALTER EXTENSION pg_db4ai UPDATE TO '0.3.0';`。

[功能核查](docs/FUNCTIONAL_COVERAGE.md) · [DDL 边界](docs/MODEL_DDL.md) · [论文与 CorgiPile 实现](docs/CORGIPILE.md) · [仅测本插件的实验](benchmarks/native/README.md)

'''),
(160,161,r'''仍未验证运行中 openGauss 的全功能逐位等价、相同随机种子的逐位一致性、生产性能、
''')
]}
FILES['docs/CORGIPILE.md'] = {'old': None, 'new': '9c932215e4b443948300b4715c419499b551ea8a', 'base': 'docs/CORGIPILE.md', 'mode': '100644','edits':[
(0,0,r'''# CorgiPile and training replay (0.3)

Reference: Xu et al., **Stochastic gradient descent without full data shuffle:
with applications to in-database machine learning and deep learning systems**, The
VLDB Journal (2024), DOI [10.1007/s00778-024-00845-0](https://doi.org/10.1007/s00778-024-00845-0).
The [extended technical report](https://arxiv.org/abs/2206.05830) describes the two-level
block/tuple shuffle and the implementation's buffering operators.

## What is implemented

`shuffle:"corgipile"` is an additional training iterator. A pass permutes the
logical blocks without replacement, reads a bounded window containing several
complete blocks, uniformly permutes the rows of that window, and emits fixed-size
minibatches. The last partial block/window/minibatch is preserved. Minibatches may
span windows. Every admitted row appears exactly once per full dataset pass.

The block size, window size and minibatch size are independent:

```sql
SELECT db4ai.train('model', 'logistic_regression',
 'SELECT label,x0,x1 FROM training_data ORDER BY id',
 '{"shuffle":"corgipile","shuffle_block_rows":256,"shuffle_window_blocks":8,
   "batch_size":128,"max_iterations":100,"seed":42}');
```

`shuffle_first_epoch` defaults to true for CorgiPile and false for legacy batch/row
modes. `shuffle:"batch"`, `"row"`, `"none"` remain available. Small tables fully
contained in the window get a global row permutation; otherwise this is NOT a
uniform permutation of the entire table. A reproducible order requires fixed
source ORDER BY, seed, epoch, block/window/batch sizes and implementation version.
The random sequence intentionally is not the original openGauss sequence.

Our epoch means one complete dataset pass. Do not transfer the paper's convergence
theorem, its epoch notation or its reported speedups to this implementation without
checking their assumptions. A fair learning comparison measures quality/loss as
well as wall time; more tuple mixing can cost CPU on small already-shuffled data.

## Practical data path

The authorized caller's source SELECT executes once through SPI and a raw
spillable tuplestore for schema inference/typed labels. Normalized numerical rows
then use contiguous double buffers, spilling to PG-managed segmented BufFile when
necessary. All later passes read this immutable dense replay, not the source SQL.

This avoids repeated TupleTableSlot/Datum deformation in the training hot path.
A replay block is a range of the **query result**, not a physical heap page.
It supports joins/expressions with ordinary permissions and RLS; it does not skip
MVCC or call a table AM directly. No persistent shuffled data copy is created.

For GD/NGD/OVA, `kernel_cache:true` (default) prepares bias and deterministic feature
maps once in a second temporary spillable store. Both updates and loss scans reuse
it. Actual map parameters still go into model JSON, not the prepared training rows.
`kernel_cache:false` retains an explicit current-version ablation/debug path.
The same seed/order/config produces identical model bytes with cache on/off when
the requested cache capacities fit and the shuffle order is the same.

Random spilled reads fetch the requested range rather than inflate each block
read to the larger sequential replay cache chunk. The block-index array, row-index
array, window and output buffer are included in the shuffle budget. Requests that
do not fit fail with a hint instead of silently changing the block size.
Budgets are per store/buffer, NOT a backend-wide hard memory limit.

`EXPLAIN MODEL` through `db4ai.explain_model` exposes input, feature preparation,
optimization and loss-evaluation times, spill flags, block/window sizes and logical
bytes/seeks. Those counters count BufFile requests, not physical disk bytes or
cache misses. Final metrics/serialization/catalog work lie outside some phase
timers; the external SQL wall time remains the end-to-end measure.

## Important boundaries

* The paper's overlapping producer/consumer double buffer is **not implemented**.
  Arbitrary C++/OpenMP threads must not call SPI, MemoryContext or other PG backend
  APIs. A safe future asynchronous path needs a separately qualified process/DSM
  or native-I/O design, cancellation and ownership protocol. No overlap claim is
  made here. XGBoost's optional internal OpenMP does not parallelize the PG reader.
* The existing explicit `db4ai.shuffle_scan(...)` SQL CustomScan remains its 0.2
  arbitrary-tuple replacement/replay operator. Training CorgiPile is a different
  entrypoint, not a renamed SQL FunctionScan. It does not force all queries or
  training into a new planner path.
* GPU/deep-learning loaders, physical-table block sampling, and distributed
  training are not delivered by this PG extension change.

## Correctness checks

`db4ai.shuffle_indices(n, options, epoch)` returns only diagnostic row indexes
(0 <= n <= 100000), using the actual training iterator. Tests check empty input,
nondivisible block/window/batch counts, index boundaries, seed/epoch determinism,
exact once, bounded memory, forced spill and exact requested read volumes.

A shuffled short batch can occur in the middle of an epoch. The trainer must
reassemble rows across that boundary rather than make an early smaller gradient
update. `test/partial_batch_reference.py` independently checks scalar GD updates
for batch, row and CorgiPile modes. Cached/uncached mapping, source-once execution,
permissions/RLS, temp-file-limit errors and cleanup also have regressions.
''')
]}
FILES['docs/FUNCTIONAL_COVERAGE.md'] = {'old': None, 'new': '704445e581afc8bd91f22ed7619f271c03f8e266', 'base': 'docs/FUNCTIONAL_COVERAGE.md', 'mode': '100644','edits':[
(0,0,r'''# DB4AI functional coverage audit (0.3)

Source registry: openGauss commit `8f33f38a306dd4b155b6e77d1bc15554b1afcf8f`,
`catalog/aifuncs.cpp` and public DB4AI headers/commands. Coverage is functionality,
not identical source, RNG sequence or bitwise solver internals.

## In scope and implemented

| Capability | Implementation/status |
|---|---|
| All 11 registered algorithms | Linear/logistic/SVM/PCA/KMeans, four XGBoost objectives, multiclass OVA, internal Bayes estimator (0.2 retained) |
| GD/NGD and OVA | Normalized GD, independent solver state, decay, typed labels, raw scores and typed prediction |
| Nonlinear feature maps | Gaussian Fourier / polynomial Maclaurin; deterministic actual maps persisted |
| KMeans/PCA options | Both seeding families, four distances, stochastic PCA; Jacobi as additional choice |
| Input and iteration | Numeric scalars/arrays, typed labels, explicit NULL policy, source-once replay, memory-bounded spill |
| Training shuffle | Existing batch/row replacement and new two-level CorgiPile window mixing |
| Train/save/load/predict/drop | Transactional PG APIs, role isolation, versioned JSON/UBJSON, export and dump/restore |
| Model commands | Explicit CREATE/DROP/EXPLAIN MODEL parser inside execute_ddl; NOT bare PG grammar |
| Model diagnostics | Added per-cluster size/coordinates/distance statistics; PCA selected eigenvalues/fractions and stochastic estimates |
| Fractional training time | max_seconds now accepts positive fractional seconds; checked at algorithm iteration boundaries |
| XGBoost training threads | nthread=1..64 with optional WITH_OPENMP=1; default build remains one thread and no OpenMP dependency |
| Upgrade | 0.1->0.2->0.3 SQL migration; previous extension models are not rewritten |

PCA `fraction_of_selected` is the fraction within the selected eigenvalues, not
misrepresented as total explained variance. Stochastic eigenvalue/variation values
are estimates. KMeans distance summaries use the selected distance and arithmetic
centroids; L1 does not turn this implementation into k-medians.

## Accepted implementation choices, not migration blockers

RNG sequence, KMeans sampling schedules, stochastic PCA initialization and
orthogonalization, independent OVA NGD state, bias coordinate layout, safer checks,
new durable serialization and PG-native memory/error management may differ.
Full original memory blobs need not be loadable. These are deliberate choices;
regression, mathematical and real-data prediction checks guard actual behavior.
Resource caps and unsupported input types are documented, not silently accepted.

## Excluded by scope

**ABO is not a missing DB4AI feature.** Its use of DB4AI Bayes in optimizer training,
plan selection, private range/marginalization helpers and optimizer lifecycle
integration is not ported and is not tracked as a DB4AI backlog item.
The separate snapshots/git-for-data feature is also excluded.

Distributed execution, GPU/deep-learning infrastructure and future parameter-search
services are not inferred requirements of this port. The audited original public
CREATE MODEL path supplies one hyperparameter configuration; generic fields for
multiple configurations alone do not establish a complete public AutoML feature.

## Explicit limits still relevant to users

Native top-level CREATE MODEL/PREDICT BY grammar and openGauss-specific plan nodes
require a PG core/parser change or a client adapter; ordinary extension hooks do
not provide them. The explicit interpreter does not claim that capability.
Old openGauss struct/Datum/XGBoost memory snapshots are not imported automatically.
XGBoost external-memory training and paper-style asynchronous double buffering
are not implemented. These are documented scope/engineering limits, not hidden
claims of complete source identity. See MODEL_DDL.md and CORGIPILE.md.
''')
]}
FILES['docs/MODEL_DDL.md'] = {'old': None, 'new': '2e0c164236a08f224086ebfe63674e4e64f812b8', 'base': 'docs/MODEL_DDL.md', 'mode': '100644','edits':[
(0,0,r'''# Model DDL compatibility entrypoint (0.3)

**Stock PostgreSQL 16–18 does not accept bare `CREATE MODEL` from this extension.**
The raw parser calls its built-in grammar before ProcessUtility hooks. Registering
a utility hook or CustomScan cannot make an unknown top-level statement parse.
See the upstream [parser](https://github.com/postgres/postgres/blob/REL_18_STABLE/src/backend/parser/parser.c).
No PostgreSQL grammar patch, wire proxy or hidden parser replacement is included.

Instead, the extension implements the model language at an explicit SQL entrypoint:

```sql
SELECT db4ai.execute_ddl($ddl$
 CREATE MODEL "房价模型" USING linear_regression
 FEATURES area, age*age TARGET price
 FROM public.houses WHERE area>0
 WITH optimizer='ngd', max_iterations=100, batch_size=64,
      shuffle='corgipile', shuffle_block_rows=128, shuffle_window_blocks=8
$ddl$);
SELECT db4ai.execute_ddl('EXPLAIN MODEL "房价模型"');
SELECT db4ai.execute_ddl('DROP MODEL IF EXISTS "房价模型"');
```

`db4ai.parse_ddl(text)` is a dry parser returning a structured request and generated
training SELECT without querying data or training. `db4ai.build_info()` explicitly
reports `native_create_model_grammar:false`. Bare SQL is rejected by stock PG.
A client adapter could wrap model DDL in execute_ddl, but that is client-side
compatibility, not a changed PostgreSQL wire grammar.

The interpreter uses PG's lexical scanner, not regex. Quoted UTF-8 identifiers,
comments, strings, dollar quoting, nested subqueries and expression features are
handled with token/depth boundaries. PG parses the reconstructed SELECT and
performs normal access checks through the existing training API.

## Accepted shape

```
CREATE MODEL name USING algorithm
 FEATURES expression [,expression...]
 [TARGET expression]
 FROM relation_or_parenthesized_query [WHERE ...] [ORDER BY ...]
 [WITH key=literal_or_DEFAULT [,key=literal_or_DEFAULT...]]
DROP MODEL [IF EXISTS] name
EXPLAIN MODEL name
```

Supervised training requires one TARGET expression and explicit FEATURES to avoid
accidental target leakage. Unsupervised algorithms reject TARGET and can omit
FEATURES to select all source fields. Parameters accept numerical/string/boolean
literals and DEFAULT; unknown and inapplicable keys are rejected, including with
DEFAULT. Default profile is `opengauss`; explicit profile can override it.
`model_format='json'/'ubj'` selects durable model IO; UBJSON is XGBoost-only.

Only one model command per call is allowed. Model names are not PG schema objects;
this API takes one identifier, not an arbitrary schema-qualified object name.
Unicode U&-escaped model identifiers and full openGauss EXPLAIN option/format syntax
are not supported; use ordinary UTF-8 quoting and the returned JSON diagnostics.
`PREDICT BY` expression grammar is not intercepted; typed db4ai.predict_* functions
provide prediction. Use ordinary PG EXPLAIN for prediction SELECTs.

## Security and transactions

The API is SECURITY INVOKER. It passes parameters to fixed schema-qualified SPI
calls to db4ai.train/drop_model/explain_model. It does not elevate training SQL.
Source permissions/RLS and underlying function EXECUTE grants apply. Failed
commands and enclosing transaction rollback do not publish partial models. The
scanner is bounded to 16 MiB / 262144 tokens / 256 nested delimiter levels.

Tests include semicolon injection, comments/dollar strings, duplicate options,
invalid targets, malformed signs, dry-run nonexecution, permission revocation,
source RLS, missing models, IF EXISTS and rollback.
''')
]}
