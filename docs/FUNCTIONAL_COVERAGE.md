# DB4AI functional coverage audit (0.3)

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
