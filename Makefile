.DEFAULT_GOAL := all
EXTENSION = pg_db4ai
MODULE_big = pg_db4ai
OBJS = src/pg_db4ai.o src/options.o src/data.o src/models.o src/native.o src/xgboost.o src/categories.o src/shuffle.o src/kernel.o src/gradient.o src/bayes.o src/shufflescan.o
DATA = sql/pg_db4ai--0.1.0.sql sql/pg_db4ai--0.2.0.sql sql/pg_db4ai--0.1.0--0.2.0.sql
PGFILEDESC = "pg_db4ai - PostgreSQL-native machine learning"
PG_CONFIG ?= pg_config
WITH_XGBOOST ?= 1
XGB_JOBS ?= 2
PG_CPPFLAGS = -I$(srcdir)/src
PG_CFLAGS = -std=c11 -Werror=implicit-function-declaration
SHLIB_LINK = -lm

ifeq ($(WITH_XGBOOST),1)
PG_CPPFLAGS += -DDB4AI_WITH_XGBOOST -I$(srcdir)/third_party/xgboost/include
SHLIB_LINK += $(CURDIR)/build/libxgboost.a $(CURDIR)/build/libdmlc.a -lstdc++ -pthread
ifeq ($(shell uname -s),Linux)
SHLIB_LINK += -Wl,--exclude-libs,ALL
endif
endif

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

ifeq ($(WITH_XGBOOST),1)
$(shlib): build/xgboost.stamp
build/xgboost.stamp: third_party/xgboost.lock.json
	@test -f third_party/xgboost/dmlc-core/CMakeLists.txt || (echo 'Vendored XGBoost source is missing; builds never download dependencies.'; exit 1)
	cmake -S third_party/xgboost -B build/xgboost \
	  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
	  -DBUILD_STATIC_LIB=ON -DKEEP_BUILD_ARTIFACTS_IN_BINARY_DIR=ON \
	  -DUSE_OPENMP=OFF -DUSE_CUDA=OFF -DUSE_NCCL=OFF \
	  -DPLUGIN_FEDERATED=OFF -DPLUGIN_RMM=OFF -DPLUGIN_SYCL=OFF \
	  -DGOOGLE_TEST=OFF -DBUILD_WITH_SYSTEM_DMLC=OFF
	cmake --build build/xgboost --target xgboost --parallel $(XGB_JOBS)
	cp "$$(find build/xgboost -name libxgboost.a -print -quit)" build/libxgboost.a
	cp "$$(find build/xgboost -name libdmlc.a -print -quit)" build/libdmlc.a
	touch $@
endif

.PHONY: test verify-vendor clean-vendor-build
test:
	PG_CONFIG='$(PG_CONFIG)' WITH_XGBOOST='$(WITH_XGBOOST)' sh test/run.sh
verify-vendor:
	python3 tools/vendor_xgboost.py --verify
clean-vendor-build:
	rm -rf build
