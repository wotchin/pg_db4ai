# Current-code SIMD ablation

No openGauss/sklearn/LightGBM/XGBoost baseline is rerun. Reuse the frozen public
CSV/NPZ/meta files from the earlier tabular evidence, without re-splitting or
refitting scalers. A current native-only .so is tested with `db4ai.simd=off`,
`portable`, `auto`, in seeded randomized mode order per trial.

Start an isolated PG18 instance, pin its postmaster to one CPU, disable JIT and
query parallelism, and set the usual PGHOST/PGPORT/PGUSER/PGDATABASE. Never use a
production DB: scripts create/drop their specifically named benchmark tables
and overwrite `simd_*` / `u_*` benchmark models.

```sh
python3 benchmarks/simd/measure.py --data /frozen/results --out /results/simd \
    --psql /path/to/psql --seeds 42,2024,7 --repeats 7
python3 benchmarks/simd/unsupervised.py --data /frozen/results --out /results/unsupervised \
    --psql /path/to/psql
cc -std=c11 -O3 -ffp-contract=off -fopenmp-simd -DDB4AI_WITH_SIMD=1 -I src \
    src/numeric.c benchmarks/simd/micro.c -lm -o /tmp/db4ai-micro
taskset -c 0 /tmp/db4ai-micro > /results/micro.csv
sh test/run_numeric.sh
```

Python's only additional analysis dependency is NumPy; no Python estimator is
trained. Scripts retain raw SQL, timing repeats, predictions, model diagnostics,
input hashes and alignment/quality checks. Actual measured summaries are under
results/2026-10-19; full logs and predictions accompany the delivered evidence ZIP.

Training SQL includes data preparation, algorithm, diagnostics and persistence.
Prediction uses the same OFF-trained model in all modes and includes lookup/load,
table scan, prediction and SUM. This is not resident-model inference time or
concurrent throughput. Client startup/COPY and warmup are excluded. Seven timing
repeats are not seven independent quality splits. Ratios are paired within each
repeat and then summarized across the three seeds. Sub-millisecond prediction
ratios are noisy and should not be treated as meaningful universal regressions
or gains. No competing build or microbenchmark ran during the SQL measurement.

The microbenchmark intentionally includes indirect kernel dispatch, unaligned
buffers and odd widths, while data stay warm/resident. Gaussian-linear measures
only its matrix-vector stage, excludes cos and includes loop-layout improvements;
its largest ratio is NOT an end-to-end training or purely ISA speedup.

`off` means ordered scalar-source sums, **not** a compiler built with vectorization
disabled. All modes share the same binary, local optimization flags and fp-contract
setting. These results isolate current kernel selection/layout, not every build
change versus the previous release. No historical timing is reinterpreted as a
same-host causal comparison. Quality guards permit roundoff, not changed labels
or material regression output; raw model weights may differ.
