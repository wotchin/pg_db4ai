# Lightweight PostgreSQL-native AutoML (0.5)

This is an implemented bounded **sequential portfolio**, not a general-purpose
AutoML platform, Bayesian optimizer, or a promise to beat every external model.
It reuses the existing C11 training/prediction, SIMD, row replay, permissions and
model table. There is no Python runtime, extra daemon, queue, or new ML dependency.

## Quick use

```sql
SELECT db4ai.automl(
  'customer_risk', 'classification',
  'SELECT churned, age, spend_30d, channel FROM training_customers',
  '{"time_budget_seconds":5,"max_trials":6,"memory_mb":128}'::jsonb
);
SELECT customer_id,
       db4ai.predict('customer_risk', jsonb_build_object(
         'age', age, 'spend_30d', spend_30d, 'channel', channel))
FROM scoring_customers;
SELECT db4ai.explain_model('customer_risk');
```

The first SELECT column is the target. `task` must explicitly be `classification`
or `regression`; numeric targets are not guessed. The other columns are original
features, except a declared group/time split key. Source column names must be
unique. 1..128 scalar feature columns, up to 1024 encoded dimensions, and 2..256
classification labels are accepted. Numeric/boolean and text/varchar/char/enum
features are supported. Expand arrays and project timestamp features in SQL.
Named JSON prediction requires each feature key, accepts explicit null, and
ignores extra keys (so `to_jsonb(row)` can be used). Numeric values must be JSON
numbers/booleans, categorical values strings; enum validation still applies.

For numeric-only pipelines, existing `predict_float8`, `predict_label`,
`predict_text`, `predict_proba`, `predict_scores`, and `class_labels` work with an
array of ORIGINAL features in original order. The stored transform is applied
automatically. Mixed pipelines use named JSON, not an ambiguously typed array.
The JSON output contains either `prediction` (regression), or a formatted label,
`class_index`, `classes`, and `probabilities`. `predict_label` retains PG label
identity, including large bigint/numeric values. Pipeline `predict_bool` requires
boolean labels. Binary scalar probability refers to `class_labels[2]`, not an
assumed business-positive class. Order is training first occurrence, also in the
report's `class_order`; fix source order for reproducibility.

## One source read, train-only fitted preprocessing

The authorized SELECT executes once. A bounded PG tuplestore holds its result;
repeated candidates consume three dense replay stores, which can spill using
PG-managed temporary files. Statement snapshots/RLS remain PostgreSQL's concern:
no heap-page shortcut bypasses them. The SELECT must be structurally read-only
(no INTO, row locks, or modifying CTE); ordinary volatile expressions such as
sequences still run once, so this is not a promise that arbitrary UDFs are pure.

The default split hashes feature values, **excluding the target**, and groups
identical feature vectors. Hash groups are divided approximately 60/20/20 by
GROUP count, not guaranteed row count. Hash collisions conservatively merge
rather than split groups. Explicit `split:"group"` groups by `group_column`;
`split:"time"` orders date/timestamp/timestamptz values, keeping equal timestamps
together. The key is excluded from features. A temporal split does not prove
user-supplied features are point-in-time correct or labels have matured.

At least 30 rows, ten distinct groups/time values, ten train rows and five rows
per holdout are required. Every class must occur in training, validation and
test. Rare/missing-class splits are rejected rather than secretly resampled after
seeing test scores. Source equality/hash functions and PG version/collation are
part of reproducibility; the split fingerprint is not a cryptographic data ID.

Only training rows fit numerical means/population standard deviations, target
scaling, label dictionaries and categorical vocabularies. Numerical NULLs map to
training means plus optional missing indicators. Constants/all-NULL numeric
columns use scale=1. Categories retain the first `max_categories` distinct
TRAINING values; remaining/unseen values and NULL have separate buckets. This
is bounded one-hot/overflow encoding, NOT frequency-selected rare-category
encoding or target encoding. Correlation, id leakage and unsuitable user feature
SQL are not automatically diagnosed.

Validation selects a candidate. Final test is evaluated only after the champion
is fixed and never selects, calibrates, refits, or gates publication. Reports
clearly distinguish validation and final test. Do not repeatedly adjust the
options after inspecting the same test set and call it an untouched assessment.
The implementation retains the selected training-partition model; it does NOT
refit on validation/test/full input. A pilot winner can contain fewer rows than
the training partition; `selected_training_rows` discloses that fact.

## Small, cost-bounded search

Default classification: logistic / logistic one-vs-all and XGBoost hist.
Default regression: linear regression and XGBoost squarederror. Two fixed
regularization/learning-rate configurations per family are screened; no external
estimator or old release is invoked. XGBoost multiclass `multi:softprob` is now
available inside the typed AutoML pipeline. Its standalone `train` path is
intentionally rejected rather than offering an incomplete untyped multi API.
Without XGBoost, the native family remains available and the report says so.
Gaussian NB is optional, never a default just because it exists in the registry.

A bounded training-only reservoir (with one mandatory example per class) is
used when training exceeds `pilot_rows`. This is not unbiased for rare-class
frequencies; it guarantees initial class coverage, not stratified optimum.
Training transforms use the full TRAIN partition, never validation/test. Pilot
candidates use 32 native epochs / 24 trees. At most two configurations are then
retrained on the full training partition, 200 native epochs / 160 trees, retaining
an alternate family when possible. There is no warm-start equivalence claim when
sample size changes. Existing native convergence can stop before the cap.

The validation primary metric ranks completed candidates, with log loss (class)
or MAE (regression) breaking ties. At most 6 fits run by default (7 with optional
NB); `max_trials` is a ceiling, not a promise to exhaust all possible time or
trial slots. Trial errors/soft cancellations are recorded. No ensemble, parameter
Bayesian optimization, cross-task cache, distributed tuning, or refit is hidden.

## Budget, interruption, and publication contract

`time_budget_seconds` starts at entry, not after loading data. A PostgreSQL
statement deadline covers source planning, fetching, blocked SQL, preprocessing,
search and serialization/publication. An existing earlier statement timeout is
never extended. 80% of requested time is available for preparation/search;
20% is reserved for final evaluation/publication. A trial-local monotonic budget
is polled by existing native loops and between XGBoost calls. Only this internal
soft deadline and documented numerical/admission/library failures are recoverable.
User cancel, parent timeout, permissions and recovery conflicts are NOT swallowed.

If some trials complete, the best completed one can be returned when search
ends. If the source or finalization exhausts the hard statement deadline, the
call errors and the transaction does not publish a partial model. Timeouts remain
cooperative: a long SQL operator or upstream C++ call can delay interrupt handling.
No hard real-time or always-return-a-model guarantee is made. Timers, GUCs,
contexts and files are restored/cleaned on both success and error; recursive
AutoML is rejected to keep backend-local budget state well-defined.

`memory_mb` is an admission/buffer budget, NOT a backend/process RSS cap. Input,
partition inventory, dense replay, pilot and candidate DMatrix admissions are
bounded; underlying SQL/third-party library allocations may consume more. There
is no process-wide memory or CPU-seconds kill mechanism. Trials and XGBoost use
one thread; query parallelism/JIT are temporarily disabled for controlled costs.
`max_rows`, `max_input_mb`, `max_encoded_mb`, PG temporary-file limits and the
existing model/XGBoost caps add explicit rejection boundaries. Oversized source
queries ERROR, never silently train the first N rows as a purported random sample.
This initial release is not TB-scale automatic sampling or XGBoost external-memory
training. A standby cannot publish to the registry; replica fit/publish separation
is not delivered by this change.

A candidate is published only if its validation primary metric improves over
training-derived majority/empirical-prior or mean prediction by more than
`min_improvement`. Outcomes:
- `ready`: a qualified completed candidate was persisted atomically;
- `no_validation_improvement`: candidate evaluated but none published;
- `no_completed_candidate`: budget/admission prevented a completed candidate.

`ready` is NOT a business-acceptance or statistical-significance certificate.
Final test can be worse, and the report must remain visible. Probabilities are
normalized, not calibrated. OVA logistic scores are normalized only within this
pipeline; standalone OVA/SVM semantics are unchanged. Metric log loss explicitly
clips at 1e-15; `predict_log_proba` on a pipeline takes the log of the returned
probability and returns -Infinity for zero, not a clipped/calibrated result.

## Options

| Option | Default | Meaning |
|---|---:|---|
| time_budget_seconds | 30 | Whole-call wall budget, 0.01..3600 |
| max_trials | 8 | Fit-attempt ceiling, 1..24; finite portfolio may finish earlier |
| memory_mb | 128 | Individual buffer/admission allocations, 16..4096, not RSS cap |
| max_rows | 200000 | Source row cap, 30..1000000, also existing db4ai.max_rows |
| max_input_mb / max_encoded_mb | 256 / 512 | Source tuple bytes / total dense encoded bytes |
| pilot_rows | 4096 | Pilot maximum, further constrained by buffer budget |
| max_categories | 32 | Training vocabulary cap per feature, 1..128 |
| seed | 42 | Group hash/reservoir/training seed |
| validation_fraction / test_fraction | .2 / .2 | Group fractions; at least .4 remains train |
| metric | balanced_accuracy / rmse | Class: accuracy, balanced_accuracy, log_loss, binary roc_auc; reg: rmse, mae |
| split | random | duplicate-feature-group hash, explicit group, or chronological time |
| group_column / time_column | absent | Exactly the key matching group/time split |
| use_xgboost | true | Skip tree family when false; build availability is reported |
| include_naive_bayes | false | Optional classifier candidate, not supported for regression |
| missing_indicators | true | Numeric missing-indicator columns |
| min_improvement | 0 | Absolute metric-unit improvement over simple reference |

Unknown/inapplicable options are errors. Reports include all candidate outcomes,
parameters, split counts/fingerprint, selected row count, family, validation and
test quality, source/encoded bytes, spills, logical replay I/O, wall/CPU times and
stop reason. Logical I/O is not physical device I/O; no measured peak RSS is
claimed. Stored report timings precede catalog publish; the returned JSON adds
full elapsed/CPU and model_id.

## Upgrade, security and model portability

Stop old library sessions, rebuild/install, reconnect, then:

```sql
ALTER EXTENSION pg_db4ai UPDATE TO '0.5.0';
```

No model table layout changes. Existing standalone artifacts are untouched.
One versioned JSON pipeline envelope stores fitted metadata and a nested existing
learner payload (native JSON or UBJSON, hex-encoded), not raw pointers or a second
model ID. Serialization overhead and model caps apply. Loading checks task,
feature/label layout and child compatibility. Unknown/missing categorical features
have explicit treatment; raw unsupported arrays are rejected, not guessed.

Source reads use caller permissions/RLS. AutoML additionally checks EXECUTE on
`db4ai.train` so revoking training cannot be bypassed via the new API. Model
writes use the existing restricted registry helper; only the selected artifact
is written. Existing name collisions fail, and outer transaction rollback removes
the publication. Models remain role-owned, exportable and dump/restorable.

## Verification

`test/automl.sql`, `automl_reference.py`, `automl_budget.py`,
`automl_upgrade.sql`, `automl_xgboost.sql` cover leakage, typed labels, raw/mixed
prediction, constants, errors, whole-call deadlines, stricter parent timer,
source-once, forced spill, no-publication decisions, RLS, permissions, upgrade,
corrupt artifacts and independent arithmetic across SIMD modes/new backends.
Existing complete regression and SIMD suites remain enabled.
`benchmarks/automl/validate.py` measures only current AutoML on frozen inputs;
its outer holdout is never provided to search. Historical results are not
silently relabeled as same-condition AutoML baselines.

References for methodology, not copied implementation:
- https://scikit-learn.org/stable/common_pitfalls.html
- https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML/
