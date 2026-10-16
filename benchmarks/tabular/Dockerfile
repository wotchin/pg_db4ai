ARG PG_IMAGE=postgres:18-bookworm
FROM ${PG_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-server-dev-18 build-essential cmake python3 python3-venv libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/bench && /opt/bench/bin/pip install --no-cache-dir \
    numpy==2.3.3 scipy==1.16.2 pandas==2.3.3 scikit-learn==1.7.2 \
    lightgbm==4.6.0 xlrd==2.0.2 threadpoolctl==3.6.0
COPY Makefile /src/Makefile
COPY third_party /src/third_party
WORKDIR /src
RUN make PG_CONFIG=/usr/lib/postgresql/18/bin/pg_config WITH_XGBOOST=1 XGB_JOBS=2 build/xgboost.stamp
COPY src /src/src
COPY sql /src/sql
COPY pg_db4ai.control /src/pg_db4ai.control
RUN make PG_CONFIG=/usr/lib/postgresql/18/bin/pg_config WITH_XGBOOST=1 XGB_JOBS=2 \
    && make PG_CONFIG=/usr/lib/postgresql/18/bin/pg_config WITH_XGBOOST=1 install \
    && mkdir -p third_party/xgboost/python-package/xgboost/lib \
    && c++ -shared -o third_party/xgboost/python-package/xgboost/lib/libxgboost.so \
       -Wl,--whole-archive build/libxgboost.a build/libdmlc.a -Wl,--no-whole-archive -pthread \
    && PYTHONPATH=/src/third_party/xgboost/python-package /opt/bench/bin/python -c \
       'import xgboost; print(xgboost.__version__); print(xgboost.build_info())'
COPY benchmarks/tabular /src/benchmarks/tabular
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ENV PYTHONPATH=/src/third_party/xgboost/python-package
