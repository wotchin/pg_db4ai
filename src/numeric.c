/* SPDX-License-Identifier: MulanPSL-2.0
 * Standalone C kernels; PostgreSQL integration is in numeric_pg.c.
 * AVX2 is function-targeted, never a global -march=native build requirement.
 */
#include "numeric.h"
#include <math.h>
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "DB4AI numeric kernels require NaN/Inf semantics; do not use fast-math"
#endif
#ifndef DB4AI_WITH_SIMD
#define DB4AI_WITH_SIMD 0
#endif
#define NF(name) scalar_##name
#define NATTR
#define NSIMD(...)
#define NINTERCHANGE 0
#include "numeric_impl.h"
#undef NF
#undef NINTERCHANGE
#undef NSIMD

const Db4aiNumericOps db4ai_numeric_scalar = {
    "off", scalar_dot, scalar_axpy, scalar_distance, scalar_centered_dot, scalar_gaussian
};
const Db4aiNumericOps *db4ai_numeric = &db4ai_numeric_scalar;

#if DB4AI_WITH_SIMD
#define DO_PRAGMA_(x) _Pragma(#x)
#define DO_PRAGMA(x) DO_PRAGMA_(x)
#define NSIMD(...) DO_PRAGMA(omp simd __VA_ARGS__)
#define NINTERCHANGE 1
#define NF(name) portable_##name
#include "numeric_impl.h"
#undef NF
static const Db4aiNumericOps portable_ops = {
    "portable", portable_dot, portable_axpy, portable_distance, portable_centered_dot, portable_gaussian
};
/* __builtin_cpu_supports includes OS AVX register-state support. No AVX2
 * instruction can be reached before this gate; no AVX512 or FMA requirement. */
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define DB4AI_TARGET_AVX2 1
#undef NATTR
#define NATTR __attribute__((target("avx2")))
#define NF(name) avx2_##name
#include "numeric_impl.h"
#undef NF
static const Db4aiNumericOps avx2_ops = {
    "avx2", avx2_dot, avx2_axpy, avx2_distance, avx2_centered_dot, avx2_gaussian
};
#endif
#endif

const Db4aiNumericOps *
db4ai_numeric_choose(int mode)
{
#if DB4AI_WITH_SIMD
    if (mode == 0) return &db4ai_numeric_scalar;
#ifdef DB4AI_TARGET_AVX2
    if (mode == 2)
    {
        __builtin_cpu_init();
        if (__builtin_cpu_supports("avx2")) return &avx2_ops;
    }
#endif
    return &portable_ops;
#else
    (void) mode;
    return &db4ai_numeric_scalar;
#endif
}
int
db4ai_numeric_compiled(void)
{
    return DB4AI_WITH_SIMD;
}
