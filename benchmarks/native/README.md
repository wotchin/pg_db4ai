# Current-only native training measurements

External baselines (openGauss, sklearn, Python XGBoost/LightGBM) and old extension
results are frozen. **Neither script executes an external estimator or old
extension version.** No parameter tuning uses holdout labels. Inputs come from the
already generated benchmark artifacts; data splits/scalers are not recomputed.

`measure_current.py` measures the current extension on eight fixed numeric datasets
with the historical GD/NGD/SVM/OVA configurations. Its variants are current-version
batch shuffle, new CorgiPile, and an optional Gaussian feature-cache ablation.
Full SQL wall time includes input, normalization, training, diagnostics and model
persistence but excludes client startup and COPY. Predictions are evaluated once
per split/configuration. Three timing repetitions are not independent quality folds.

```sh
# First build/install the current extension and create an isolated database.
# PGHOST, PGPORT, PGUSER, PGDATABASE identify ONLY this test instance.
python3 benchmarks/native/measure_current.py \
  --data /frozen/artifact/results --out /new/results --seed 42 --repeats 3 \
  --psql /path/to/psql --kernel-ablation
python3 benchmarks/native/shuffle_experiment.py \
  --data /frozen/artifact/results --out /new/stress --psql /path/to/psql
```

`shuffle_experiment.py` is a predeclared stress test, not another generalization
split: sixteen copies of the existing Banknote **training** rows, sorted by label,
256 KiB budgets, no new holdout tuning. none/batch/CorgiPile run at fixed 1/5/20-pass
budgets. The unchanged original holdout is reported. These are independent fits
at increasing budgets, not continuation checkpoints. Differences in quality and
throughput are both retained; CorgiPile need not be faster on small shuffled tables.

The workflow pins one CPU and limits the current PostgreSQL container to 4 GiB.
It validates each archived input ZIP hash and reuses the saved NPZ/CSV/meta files.
No openGauss image is pulled; no Python baseline .fit/.train is called. Its output
contains the current source commit, PG/build version, CPU, input hashes, all SQL,
model metadata, prediction CSVs and full timing repetitions.

Historical absolute timings are different-machine/time references, NOT same-host
causal speedups. The feature-cache ablation is same-current-code/same-data; it must
pass model-byte identity first. Read counters count logical BufFile requests, not
physical device traffic. No peak-RSS, concurrent throughput or universal speedup
claim follows from this test.
