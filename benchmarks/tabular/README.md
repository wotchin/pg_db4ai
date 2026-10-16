# Public tabular Docker benchmark

This is a measured engineering and fixed-configuration quality benchmark, not a
reproduction of a tabular foundation-model leaderboard or a tuned SOTA claim.

## Datasets and attribution

| Dataset | Rows | Features | Task | Source |
|---|---:|---:|---|---|
| breast_cancer | 569 | 30 | Binary | [UCI WDBC](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), sklearn loader |
| banknote | 1372 | 4 | Binary | [UCI Banknote, Volker Lohweg](https://archive.ics.uci.edu/dataset/267/banknote+authentication) |
| wine | 178 | 13 | 3 classes | [UCI Wine](https://archive.ics.uci.edu/dataset/109/wine), sklearn loader |
| digits | 1797 | 64 | 10 classes | [sklearn Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) |
| diabetes | 442 | 10 | Regression | [sklearn Diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) |
| wine_red | 1599 | 11 | Regression | [UCI Wine Quality, Cortez et al.](https://archive.ics.uci.edu/dataset/186/wine+quality), red subset |
| concrete | 1030 | 8 | Regression | [UCI Concrete, I-Cheng Yeh](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) |
| california | 20640 | 8 | Regression | [sklearn California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) |

UCI distributes Banknote, Wine Quality and Concrete under CC BY 4.0; retain their
linked attribution when redistributing derived data. Loader descriptions identify
the other datasets' provenance. Digits is flattened image data, not enterprise
records. All tasks here have numeric features; mixed-category and missing-value
workloads need separate tests. Small Wine/WDBC scores can saturate.

[KumoRFM](https://docs.nvidia.com/sdgm/research/kumorfm-paper) preserves multi-table,
temporal and graph signals. RelBench is a separate relational benchmark, not a
single-table score to quote against this test. The [2025 TabPFN paper](https://www.nature.com/articles/s41586-024-08328-6)
also has its own splits, tuning and hardware. Neither model is executed here.

## Run

Use an isolated Linux x86-64 Docker host and Python 3.11+. Do not target a
production database or put private files into the benchmark output directory.
The runner removes only its named ephemeral containers `bench-pg` and `bench-og`.

```sh
docker build -f benchmarks/tabular/Dockerfile -t pg-db4ai-benchmark:local .
docker build -f benchmarks/tabular/Dockerfile.opengauss -t opengauss-db4ai-benchmark:local .
python3 -m venv .venv-bench
.venv-bench/bin/pip install numpy==2.3.3 scipy==1.16.2 pandas==2.3.3 scikit-learn==1.7.2 xlrd==2.0.2
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-bench/bin/python \
  benchmarks/tabular/run.py --out /absolute/path/benchmark-output --seeds 42,2024,7 --repeats 3
.venv-bench/bin/python benchmarks/tabular/audit.py /absolute/path/benchmark-output
.venv-bench/bin/python benchmarks/tabular/summarize.py /absolute/path/benchmark-output /absolute/path/summary
.venv-bench/bin/python -m unittest discover -s benchmarks/tabular -p test_harness.py
```

`export_transport.py` remains a compatible entrypoint for the original measured
workflows. The readiness/export guards are now called directly by `run.py` rather
than installed by monkeypatching. The final database process must have replaced
the image's temporary bootstrap server before measurement. openGauss COPY files
are read through their owning `omm` account and republished as runner-owned public
outputs, outside the timed region. Database file permissions are not relaxed.

## Images and versions

The openGauss base is immutable:

```
enmotech/opengauss@sha256:9bd81380273944e5a02a2139c90954d4f46813b71810f7b23fe8f738014d03b5
```

It contains openGauss 6.0.0 build aee4abd5 and XGBoost 1.4.1. The derived image
installs its distribution's omitted `libgomp1` runtime without replacing database
or XGBoost code. This is a packaging repair, not an algorithm speedup. The released
binary is not the master commit reviewed during the extension port.

The workflow records the resolved PostgreSQL image digest. The measured run used
PostgreSQL 18.6, XGBoost 3.4.2, sklearn 1.7.2 and LightGBM 4.6.0. Python XGBoost is
linked from the extension's exact vendored compiled static objects, not a different
pip wheel. Other Python packages are pinned in the Dockerfile. Apt packages can
change on future rebuilds: preserve image IDs, versions and artifacts for a
historical comparison; an image tag alone is not immutable.

## Protocol and limitations

1. Identical row splits and frozen train order across engines. Three seeds
   (42,2024,7), about 80/20 train/test. Identical feature vectors are grouped before
   splitting, including conflicting labels. Classification uses the first fold
   of StratifiedGroupKFold; regression uses GroupShuffleSplit. There is no extra
   geographic, patient-, writer- or time-based isolation beyond these groups.
2. StandardScaler is fit on training X only. Regression y is standardized on
   training data; RMSE/MAE/R2 are restored to original target units. California
   targets are in USD 100,000; Concrete strength is in MPa. No holdout tuning.
3. Fixed 200-iteration GD/NGD and 100-round trees. Tolerance=1e-30 is positive
   because openGauss rejects zero despite its source comment. Actual convergence
   can still stop early. The measured-run OG iteration query initially used the
   wrong catalog field, so its raw metadata errors are retained. A supplementary
   untimed rerun uses `modelname` and joins exact data/split hashes; this is not
   falsely labeled as metadata from the original timed instances. The canonical
   runner now uses the correct column.
4. One warmup and three timed repetitions per split. Quality is counted once per
   split, not three times. Mean/sample SD describes three overlapping holdouts,
   not statistical significance. Latencies use medians within each split.
5. Each measurement container is limited to one pinned CPU and 4 GiB; OpenMP/BLAS
   and database query parallelism are disabled or set to one. No network during
   measurements. The matrix uses separate hosts per seed, but all engines within
   each seed run sequentially on the same host/CPU. Compute engine ratios within
   seed before taking their median; do not divide mixed-host aggregate timings.
   Database defaults beyond explicit settings differ; no peak-memory claim.
6. Training SQL timings include read/prepare/train/serialize/catalog persistence,
   not Docker build/startup, psql/gsql startup, scaler fitting or COPY loading.
   COPY is recorded separately. Prediction SQL includes model lookup/load, table
   scan, per-row prediction and SUM, not transfer of each result row to the client.
   This is warmed code/data, not cold disk or concurrent throughput.
7. Python prediction is resident-model vectorized inference over the full holdout;
   serialization/loading are recorded separately. It excludes SQL and durable
   catalog writes. DB/Python timing ratios are not pure database overhead.
8. sklearn OLS/lbfgs/LinearSVC use different solvers and regularization. LightGBM is
   a different algorithm. These are practical quality baselines, not port parity.
   openGauss XGBoost 1.4.1 vs PG/Python 3.4.2 also confounds speed and quality.
9. OG XGBoost gets the complete training set as one batch, avoiding its old
   chunk-wise warm-start behavior. It evaluates internally every round and builds
   its internally named test DMatrix from training data. We never report that
   metric as holdout quality: quality is recomputed from exported held-out rows.
   The PG wrapper evaluates at the end. Such work is part of stack latency.
10. Keep failed runs and poor configurations. All main model executions must
    succeed for completion. Final main run: 744 records, zero execution failures.
    Native OG metadata-query failures are separately disclosed above. Independent
    audit checks raw predictions and resource settings without common.metrics or
    sklearn.metrics. XGBoost exported-model parity covers every test row; separate
    training parity compares binary labels and scaled regression outputs.

## Same-model A/B

`prediction_ab.py` builds the unmodified bde6ee99fc14 extension with current
benchmark tooling and the same PG base as the optimized image. It requires
byte-identical UBJSON models and identical predictions. Two warmups and eleven
paired trials alternate old/new order on the same CPU. Every timed statement
uses a fresh backend, including model load but excluding process startup.

The optimization reuses a PG-owned float buffer and proxy for gbtree in-place
prediction, removing per-row DMatrix construction. Training/model IO are unchanged;
dart and gblinear retain their previous compatible DMatrix path. Both an earlier
independent A/B and the final main-run A/B are disclosed; VM-specific speedups are
not a universal latency guarantee.

## Evidence

See REPORT.zh-CN.md and results/2026-10-16. Raw artifacts retain source/SQL, row
IDs, data/scaler hashes, CSV/NPZ, predictions, model metadata, round-trip checks,
settings and logs. summarize.py produces per-split/aggregate CSV and paired ratios.
No peak-memory, production, concurrency, GPU, mixed-type, huge-table, relational,
KMeans/PCA/Bayes-quality or foundation-model claim follows from this suite.
