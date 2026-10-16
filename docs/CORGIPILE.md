# CorgiPile and training replay (0.3)

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
