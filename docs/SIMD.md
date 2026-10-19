# Native CPU SIMD kernels

## Review of 0.3

At main `9053edbe6154ed116565d762e41d7abeb9941d9e`, native training already
had dense contiguous replay and prepared feature caching. It had no explicit
SIMD pragmas, reduction contracts, audited `restrict` kernel interfaces, runtime
ISA dispatch or vectorization diagnostics. This does **not** mean zero SIMD:
PGXS inherits compiler optimizations and some old loops auto-vectorize.
OpenMP was an optional XGBoost thread feature, not native SIMD support.

## Implementation

Private numerical helpers (not a database-independent public API) cover dot
products, AXPY gradient accumulation, L1/L2/squared-L2/Linf distance, centered
PCA projections and the linear part of Gaussian random feature mapping. GD,
NGD, OVA, stochastic PCA, KMeans and native prediction call these helpers.
Polynomial maps use the same dot helper. Stateful NGD scaling, robust hypot
normalization and Jacobi control logic are not falsely declared independent.

`#pragma omp simd` and local reductions express valid loop semantics;
`-fopenmp-simd` does not create OpenMP worker threads or require libgomp/libomp.
Audited numerical kernels alone get `-O3`. Writable AXPY/map buffers use C11
`restrict`; two read-only dot inputs may alias. Gaussian traversal is changed
from strided column reads to contiguous independent-component updates. Each
component retains its original feature accumulation order.

There is deliberately no `aligned(...:32)` or `assume_aligned(32)` promise.
PG allocations, array offsets and odd-width rows are only naturally aligned;
the compiler emits unaligned loads and safe scalar tails. There are no padded
extra reads. Tiny (<16) dot/AXPY vectors retain direct scalar-source loops to
avoid dispatch/reduction setup overhead.

The baseline build is portable: no global `-march=native`, `-mavx2` or AVX512
requirement. On GCC/Clang x86, function-targeted AVX2 kernels are selected only
after `__builtin_cpu_supports("avx2")` (including OS register-state support).
Other machines use baseline compiler-vectorized kernels (e.g. SSE2 on this
x86-64 build; AArch64 can use its baseline NEON). AVX512/FMA/SVE specializations
and macOS/ARM execution measurements are not delivered in this change.

## Floating-point contract and error handling

No global fast-math, finite-math-only, reciprocal-math or `-Ofast` is enabled.
Numerical compilation rejects fast-math and disables FP contraction. NaN/Inf
checks remain effective in PG. Reductions explicitly permit local reassociation:
low-order bits may change, and near a classification boundary the label can
in principle change. Model bytes across modes/compilers are not promised equal.
`off` provides ordered scalar-source arithmetic for investigations; the compiler
may still vectorize products or independent updates without reordering sums.

Gaussian component order and AXPY per-element arithmetic are preserved exactly
in the standalone tests. Overflow, cancellation-heavy vectors, read-only
self-alias, misalignment, odd tails, guard-page boundaries and zero length are
tested. PG errors/cancellation checks stay outside SIMD regions; no palloc,
SPI, ereport or PG callback executes inside a vector loop. cos/exp/log/hypot
remain ordinary libm operations, not silently replaced by approximate math.

XGBoost's vendored C++ implementation is unchanged. These switches do not change
XGBoost tree traversal or add AVX2 requirements to the upstream library.
Branch-heavy trees and Bayes graph traversal are not dense vector operations.

## Build, inspect, control

```sh
make PG_CONFIG=/path/to/pg_config WITH_SIMD=1 WITH_XGBOOST=1
make PG_CONFIG=/path/to/pg_config WITH_SIMD=1 WITH_XGBOOST=1 install
# Compiler remarks and machine instructions, without executing a model:
make PG_CONFIG=/path/to/pg_config WITH_XGBOOST=0 simd-report
```

SIMD is on by default when the compiler accepts `-fopenmp-simd`; an unsupported
compiler or `WITH_SIMD=0` uses the fallback. Build fingerprints include the
switch/compiler flags; .o and LLVM .bc depend on the new headers. Different PG
majors still need separate builds. SQL schema/model format remain 0.3.0; existing
0.3 installations need the rebuilt library and **new connections**, not a fake
SQL upgrade script for a C-only optimization.

```sql
SELECT db4ai.build_info(); -- simd_compiled, simd_backend
SET db4ai.simd = auto;     -- CPU checked AVX2 or portable
SET db4ai.simd = portable; -- baseline ISA vectorized functions
SET db4ai.simd = off;      -- ordered scalar-source reference path
```

These per-backend GUC settings also support SET LOCAL/rollback. They are not
persisted as pointers or ISA-specific model data. A saved model can be loaded
on a different supported CPU. For strict comparisons, pin compiler and mode,
input order, algorithm parameters and random seed.

## References

- GCC, SIMD-only option: https://gcc.gnu.org/onlinedocs/gcc/C-Dialect-Options.html
- GCC, optimization/FP semantics: https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
- GCC, x86 CPU feature checks: https://gcc.gnu.org/onlinedocs/gcc/x86-Built-in-Functions.html
- LLVM vectorizers, reduction constraints and diagnostics: https://llvm.org/docs/Vectorizers.html
