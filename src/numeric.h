/* SPDX-License-Identifier: MulanPSL-2.0
 * Internal C numerical kernels. No PG callbacks, allocation or exceptions here.
 * "off" is ordered scalar-source arithmetic (compiler autovectorization remains
 * permitted). "portable" is explicit SIMD at the compiler baseline ISA. "auto"
 * additionally selects an AVX2-only target when CPU AND OS support it.
 */
#ifndef DB4AI_NUMERIC_H
#define DB4AI_NUMERIC_H
#include <stddef.h>

typedef struct Db4aiNumericOps
{
    const char *name;
    double (*dot)(const double *, const double *, int);
    void (*axpy)(double *restrict, const double *restrict, double, int);
    double (*distance)(const double *, const double *, int, int);
    double (*centered_dot)(const double *, const double *, const double *, int);
    void (*gaussian)(double *restrict, const double *restrict,
                     const double *restrict, const double *restrict, int, int);
} Db4aiNumericOps;

extern const Db4aiNumericOps db4ai_numeric_scalar;
extern const Db4aiNumericOps *db4ai_numeric;
const Db4aiNumericOps *db4ai_numeric_choose(int mode);
int db4ai_numeric_compiled(void);

/* Avoid an indirect call/reduction setup on tiny feature vectors. These loops
 * keep ordered sums, independent of runtime SIMD mode. Alias is fine for two
 * read-only dot inputs (including dot(x,x)); restrict is ONLY on writable,
 * proven-disjoint axpy/gaussian buffers. No alignment stronger than double's
 * natural alignment is assumed: PG allocations/row strides need not be 32B. */
static inline double
db4ai_dot(const double *a, const double *b, int n)
{
    double sum = 0;
    int i;
    if (n >= 16) return db4ai_numeric->dot(a, b, n);
    for (i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}
static inline void
db4ai_axpy(double *restrict y, const double *restrict x, double alpha, int n)
{
    int i;
    if (n >= 16) { db4ai_numeric->axpy(y, x, alpha, n); return; }
    for (i = 0; i < n; i++) y[i] += alpha * x[i];
}
#endif
