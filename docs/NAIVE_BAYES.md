# User-facing Gaussian Naive Bayes (0.4)

`gaussian_naive_bayes` is a supervised binary/multiclass classifier. It replaces
(not renames) `bayes_net_internal`, whose approximate joint-selectivity/ABO
semantics were not useful as a public business prediction API. No optimizer
integration, graph parameters, marginalization, MCV or binning implementation
remains in the runtime. This is not a general Bayesian-network learner.

## Train and predict

The first query column is the label; subsequent columns are numeric/boolean
scalars or fixed-length float4[]/float8[] features. Label values retain PG type
identity: text, enum, bigint, numeric and other supported equality/hash/IO types.
There must be 2..256 classes and 1..1024 flattened features. Text/categorical
FEATURES, count-based Multinomial NB and online partial_fit are not implemented.

```sql
CREATE EXTENSION pg_db4ai;
SELECT db4ai.train('weather', 'gaussian_naive_bayes', $sql$
  SELECT label, temperature, humidity
  FROM (VALUES ('dry',20.0,30.0),('dry',24.0,35.0),
               ('wet',18.0,80.0),('wet',22.0,85.0)) t(label,temperature,humidity)
$sql$);
SELECT db4ai.predict_label('weather', ARRAY[21.0,32.0], NULL::text);
SELECT db4ai.class_labels('weather'),
       db4ai.predict_proba('weather', ARRAY[21.0,32.0]),
       db4ai.predict_log_proba('weather', ARRAY[21.0,32.0]);
SELECT db4ai.explain_model('weather');
```

`predict_label` returns the original label (type witness NULL::label_type).
`predict_text` formats the label. `predict_proba` and `predict_log_proba` return
arrays in `class_labels` order; no softmax is retrofitted to existing OVA/SVM.
`predict_scores` on this model returns the same posterior probabilities.
Binary `predict_float8` is the probability of class_labels[2], not the label.
`predict_bool` requires boolean labels for this new algorithm. Other label types
must use the typed/label APIs, avoiding an ambiguous boolean business meaning.
Binary 0/1 or bool labels are ordered 0,1/false,true; other classes preserve first
occurrence. Argmax ties choose the first class. Fix source ORDER BY for repeats.

## Statistical and numerical contract

The model assumes conditionally independent Gaussian features:

`log P(c|x) = log prior[c] - .5 * sum(log(2*pi*var[c,j]) + (x[j]-mean[c,j])^2/var[c,j]) - log_normalizer`.

One pass over normalized data computes class/global Welford population moments.
Additional replay passes compute training diagnostics; the source SELECT executes
once, as caller. Population variance uses denominator N, not N-1. Every class
variance receives `max(variance_floor, var_smoothing * max(global_variances))`.
This makes singleton classes and all-constant training data well-defined.

| Option | Default | Semantics |
|---|---:|---|
| var_smoothing | 1e-9 | Relative smoothing factor in [0,1] |
| variance_floor | 1e-12 | Positive absolute floor, [1e-300,1e100] |
| fit_prior | true | Empirical class frequencies; false chooses uniform priors |
| batch_size | 1024 | Input fetch size, not a gradient minibatch |
| null_policy | error | Reject NULL; explicit legacy skips NULL labels and zero-imputes scalar NULL features using the existing input policy |

Unrelated seed, epochs, shuffle, graph and learning-rate parameters are rejected.
NaN/Inf inputs are rejected. Out-of-range likelihood arithmetic fails explicitly
with a scaling hint and never publishes partial models. Normalize with logsumexp;
very small probabilities may underflow to zero, while log probabilities remain
available. Rare remote-tail rows use per-feature likelihood differences against
the best preliminary class to avoid cancellation of huge common penalties.
Ordinary squared norms reuse the existing runtime-selected SIMD helper.

Probabilities sum to one under this model; they are **not calibrated confidence**.
Strongly correlated, sparse or nearly deterministic features can produce grossly
overconfident wrong predictions. Use separate validation data for selecting
smoothing/calibration. Do not tune on the holdouts used for a published score.

The new versioned JSON stores class labels/counts/priors, per-class means and
variances. Derived normalizers, addresses, OIDs and CPU-specific objects are not
persisted. Loading checks dimensions, lengths, positive variances and priors.
Model ownership, RLS, rollback, export and dump/restore use the existing PG paths.

## Upgrade and removal of internal functionality

Quiesce old sessions, install the rebuilt library, reconnect, then run:

```sql
ALTER EXTENSION pg_db4ai UPDATE TO '0.4.0';
```

New installs do not create `predict_joint`. Upgrade drops it with **RESTRICT**:
user views/functions depending on it must be deliberately migrated first; the
upgrade does not silently CASCADE-delete user objects. The old C symbol is only
an error-reporting shim for installations that have not yet upgraded their SQL.
`bayes_net_internal` is recognized only to identify stored legacy artifacts.
New training and inference are rejected with a retraining message. Legacy
payloads remain unchanged and can still be explained, exported or dropped.
They are NEVER reinterpreted as supervised models. All other model formats and
algorithms are preserved. See test/naive_bayes_upgrade.sql for the migration test.

## Validation

`test/naive_bayes.sql`: labels, binary/multiclass posteriors, priors, constants,
arrays, source-once, spill, limits, corrupt artifacts, RLS/permissions, rollback,
DDL and remote-tail cancellation. `test/naive_bayes_reference.py`: independent
stdlib two-pass moment and posterior equations, all three SIMD modes, executed
again in a fresh backend after logical restore. It does not run sklearn.

`benchmarks/naive_bayes/validate.py` accepts already-frozen NPZ/meta inputs and
validates only this classifier. Fitting moments and inference from the actual
serialized parameters are checked separately: tiny moment rounding differences
can be amplified by near-zero variances. No previous-release or external baseline
is re-run. Full integration CI retains the existing PG/XGBoost/OpenMP matrix.

Statistical reference (not copied implementation):
https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes
