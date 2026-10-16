FILES['.github/workflows/native-optimization.yml'] = {'old': None, 'new': '074bc0e14ee72e078e6bbada7060dc7c25e92c2f', 'base': '.github/workflows/native-optimization.yml', 'mode': '100644','edits':[
(0,0,r'''name: Current extension only - CorgiPile and native training
on:
  workflow_dispatch:
  push:
    branches: [db4ai-corgipile-ddl]
    paths: [.github/workflows/native-optimization.yml]
permissions:
  contents: read
  actions: read
jobs:
  current_only:
    runs-on: ubuntu-24.04
    timeout-minutes: 25
    steps:
      - uses: actions/checkout@v4
      - name: Retrieve immutable existing splits, not new baseline runs
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          mkdir -p /tmp/db4ai-current/frozen /tmp/db4ai-current/out
          python3 - <<'PY'
          import hashlib,io,json,os,urllib.request,zipfile
          from pathlib import Path
          artifacts=[(42,12063108102,'84db2b99d6cce4ba15581bf7ab419162bbce4394bed1b58a0a1d035af160d0f2'),
                     (2024,12063183225,'10fcb7b29bf27c98b1044ebc997056a9dd57d00827681ea3af0caf54c2b0ef8a'),
                     (7,12063542146,'7d6be8920b04129aa78659d0bec6279bf8715a080806b47fb100e172b3d6a2a0')]
          dest=Path('/tmp/db4ai-current/frozen')
          for seed,i,digest in artifacts:
              req=urllib.request.Request(f'https://api.github.com/repos/wotchin/pg_db4ai/actions/artifacts/{i}/zip')
              req.add_unredirected_header('Authorization','Bearer '+os.environ['GH_TOKEN'])
              with urllib.request.urlopen(req,timeout=120) as r:raw=r.read()
              assert hashlib.sha256(raw).hexdigest()==digest
              with zipfile.ZipFile(io.BytesIO(raw)) as z:
                  for f in z.namelist():
                      path=Path(f)
                      assert not path.is_absolute() and '..' not in path.parts
                      if not f.startswith('results/') or f.endswith('/'):continue
                      if len(path.parts)<3:continue
                      if not (f.endswith('.npz') or f.endswith('.csv') or f.endswith('meta.json')):continue
                      target=dest.joinpath(*path.parts[1:]);target.parent.mkdir(parents=True,exist_ok=True)
                      target.write_bytes(z.read(f))
          Path('/tmp/db4ai-current/out/frozen-provenance.json').write_text(json.dumps(artifacts,indent=2)+'\n')
          PY
      - name: Build only the current PostgreSQL extension
        run: |
          docker pull postgres:18-bookworm
          image=$(docker image inspect postgres:18-bookworm --format '{{index .RepoDigests 0}}')
          docker image inspect postgres:18-bookworm > /tmp/db4ai-current/out/base-image.json
          docker build --build-arg PG_IMAGE="$image" -f benchmarks/native/Dockerfile -t db4ai-native-current:local .
      - name: Current-version fixed-input measurements and ablations
        run: |
          set -o pipefail
          chmod -R a+rX /tmp/db4ai-current/frozen
          chmod 777 /tmp/db4ai-current/out
          cpu=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
          docker run -d --name db4ai-native-only --cpuset-cpus "$cpu" --cpus 1 --memory 4g --shm-size 1g --network none \
            -e POSTGRES_PASSWORD='CurrentOnly_2026!' -e SOURCE_REVISION='${{ github.sha }}' \
            -v /tmp/db4ai-current/frozen:/frozen:ro -v /tmp/db4ai-current/out:/out \
            db4ai-native-current:local -c shared_buffers=256MB -c jit=off -c max_parallel_workers_per_gather=0
          ready=0
          for i in $(seq 1 90); do
            if test "$(docker exec db4ai-native-only cat /proc/1/comm)" = postgres && \
               docker exec -u postgres db4ai-native-only psql -d postgres -Atc 'SELECT 1'; then ready=1;break;fi
            sleep 1
          done
          test "$ready" = 1
          docker exec -u postgres db4ai-native-only psql -d postgres -v ON_ERROR_STOP=1 -c 'CREATE EXTENSION pg_db4ai; SELECT db4ai.build_info();'
          for seed in 42 2024 7; do
            extra=''; if test "$seed" = 42;then extra='--kernel-ablation';fi
            docker exec -u postgres db4ai-native-only /opt/native/bin/python \
              /src/benchmarks/native/measure_current.py --data /frozen --out /out/seed-$seed --seed "$seed" --repeats 3 $extra \
              2>&1 | tee /tmp/db4ai-current/out/seed-$seed.log
          done
          docker exec -u postgres db4ai-native-only /opt/native/bin/python \
            /src/benchmarks/native/shuffle_experiment.py --data /frozen --out /out/ordered-stress --repeats 3 \
            2>&1 | tee /tmp/db4ai-current/out/stress.log
      - name: Preserve evidence
        if: always()
        run: |
          docker inspect db4ai-native-only > /tmp/db4ai-current/out/container.json || true
          docker logs db4ai-native-only > /tmp/db4ai-current/out/server.log 2>&1 || true
          git archive --format=tar.gz HEAD > /tmp/db4ai-current/out/source.tar.gz
          docker rm -f db4ai-native-only || true
          sudo chown -R "$(id -u):$(id -g)" /tmp/db4ai-current/out
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: current-only-native-corgipile
          path: /tmp/db4ai-current/out
          retention-days: 30
''')
]}
FILES['benchmarks/native/Dockerfile'] = {'old': None, 'new': '58e1cdfe205f6e64ef1e10e579195b42fb5d5987', 'base': 'benchmarks/native/Dockerfile', 'mode': '100644','edits':[
(0,0,r'''ARG PG_IMAGE=postgres:18-bookworm
FROM ${PG_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-server-dev-18 build-essential python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/native && /opt/native/bin/pip install --no-cache-dir numpy==2.3.3
COPY Makefile pg_db4ai.control /src/
COPY src /src/src
COPY sql /src/sql
COPY benchmarks/native /src/benchmarks/native
WORKDIR /src
RUN make PG_CONFIG=/usr/lib/postgresql/18/bin/pg_config WITH_XGBOOST=0 -j2 \
    && make PG_CONFIG=/usr/lib/postgresql/18/bin/pg_config WITH_XGBOOST=0 install
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
''')
]}
FILES['benchmarks/native/README.md'] = {'old': None, 'new': '7f2bff431a2ceb9c192243704320d83e81c21124', 'base': 'benchmarks/native/README.md', 'mode': '100644','edits':[
(0,0,r'''# Current-only native training measurements

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
''')
]}
