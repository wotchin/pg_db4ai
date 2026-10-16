ARG PG_IMAGE=postgres:18-bookworm
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
