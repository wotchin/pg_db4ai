/* SPDX-License-Identifier: MulanPSL-2.0 */
#include "postgres.h"
#include "numeric.h"
#include "utils/guc.h"

static int simd_mode = 2;
static const struct config_enum_entry simd_modes[] = {
    {"off", 0, false}, {"portable", 1, false}, {"auto", 2, false}, {NULL, 0, false}
};
void db4ai_numeric_init(void);

static void
assign_simd(int value, void *extra)
{
    (void) extra;
    db4ai_numeric = db4ai_numeric_choose(value);
}
void
db4ai_numeric_init(void)
{
    DefineCustomEnumVariable("db4ai.simd", "Native numeric kernel mode (not XGBoost threads).",
        "SIMD reductions may change rounding; off retains ordered scalar-source sums.",
        &simd_mode, 2, simd_modes, PGC_USERSET, 0, NULL, assign_simd, NULL);
    assign_simd(simd_mode, NULL);
}
