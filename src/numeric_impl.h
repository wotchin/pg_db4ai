/* SPDX-License-Identifier: MulanPSL-2.0
 * Intentionally included for scalar, portable and AVX2 targets.
 * NSIMD clauses authorize local reductions, NOT global fast-math. Reduction
 * rounding order may differ. Build this file with -ffp-contract=off; never use
 * finite-math-only: the PG caller still detects NaN/Inf/overflow normally.
 */
static NATTR double
NF(dot)(const double *a, const double *b, int n)
{
    double sum = 0;
    int i;
    NSIMD(reduction(+:sum))
    for (i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}
static NATTR void
NF(axpy)(double *restrict y, const double *restrict x, double alpha, int n)
{
    int i;
    NSIMD()
    for (i = 0; i < n; i++) y[i] += alpha * x[i];
}
static NATTR double
NF(distance)(const double *a, const double *b, int n, int metric)
{
    double sum = 0;
    int i;
    /* Hoist algorithm selection OUT of the feature loop. NaN inputs are
     * rejected by PG; overflow to Inf is preserved for the caller to reject. */
    if (metric == 0)
    {
        NSIMD(reduction(+:sum))
        for (i = 0; i < n; i++) sum += fabs(a[i] - b[i]);
    }
    else if (metric == 3)
    {
#if NINTERCHANGE
        /* Independent lane maxima avoid LLVM's strict-FP max-reduction limit.
         * Max over finite/nonnegative distances is exactly order-independent. */
        double lanes[8] = {0};
        int base, lane;
        for (base = 0; base <= n - 8; base += 8)
        {
            NSIMD()
            for (lane = 0; lane < 8; lane++)
            {
                double d = fabs(a[base+lane] - b[base+lane]);
                if (d > lanes[lane]) lanes[lane] = d;
            }
        }
        for (lane = 0; lane < 8; lane++)
            if (lanes[lane] > sum) sum = lanes[lane];
        for (i = base; i < n; i++)
        {
            double d = fabs(a[i] - b[i]);
            if (d > sum) sum = d;
        }
#else
        for (i = 0; i < n; i++)
        {
            double d = fabs(a[i] - b[i]);
            if (d > sum) sum = d;
        }
#endif
    }
    else
    {
        NSIMD(reduction(+:sum))
        for (i = 0; i < n; i++)
        {
            double d = a[i] - b[i];
            sum += d * d;
        }
        if (metric == 1) sum = sqrt(sum);
    }
    return sum;
}
static NATTR double
NF(centered_dot)(const double *a, const double *mean, const double *w, int n)
{
    double sum = 0;
    int i;
    NSIMD(reduction(+:sum))
    for (i = 0; i < n; i++) sum += (a[i] - mean[i]) * w[i];
    return sum;
}
static NATTR void
NF(gaussian)(double *restrict out, const double *restrict input,
             const double *restrict weights, const double *restrict offsets,
             int n, int k)
{
    int c, j;
#if NINTERCHANGE
    /* weights[j*k+c] is physically contiguous over c. Keep each output's j
     * accumulation order identical to the scalar map while vectorizing across
     * independent c outputs. No horizontal FP reassociation is needed here. */
    for (c = 0; c < k; c++) out[c] = offsets[c];
    for (j = 0; j < n; j++)
    {
        double x = input[j];
        const double *w = weights + (size_t)j * k;
        NSIMD()
        for (c = 0; c < k; c++) out[c] += x * w[c];
    }
#else
    for (c = 0; c < k; c++)
    {
        double v = offsets[c];
        for (j = 0; j < n; j++) v += input[j] * weights[(size_t)j * k + c];
        out[c] = v;
    }
#endif
}
