/* SPDX-License-Identifier: MulanPSL-2.0 */
#include "postgres.h"
#include "automl.h"
#include "miscadmin.h"
#include <time.h>

/* Backend-local and active only inside an AutoML trial. Never convert a user
 * cancel, statement_timeout, lock timeout or recovery conflict into a result. */
static double soft_deadline = 0;

double
db4ai_automl_now(void)
{
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t) != 0)
        elog(ERROR, "could not read monotonic clock");
    return (double)t.tv_sec + t.tv_nsec / 1000000000.0;
}

void
db4ai_automl_soft_deadline(double deadline)
{
    soft_deadline = deadline;
}

void
db4ai_automl_check(void)
{
    CHECK_FOR_INTERRUPTS();
    if (soft_deadline != 0 && db4ai_automl_now() >= soft_deadline)
        ereport(ERROR, (errcode(DB4AI_AUTOML_DEADLINE),
                        errmsg("AutoML candidate exhausted its allocated search time")));
}
