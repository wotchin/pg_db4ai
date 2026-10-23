.DEFAULT_GOAL := all
EXTENSION = pg_db4ai
MODULE_big = pg_db4ai
OBJS = src/pg_db4ai.o src/options.o src/data.o src/models.o src/native.o src/xgboost.o src/categories.o src/shuffle.o src/kernel.o src/gradient.o src/naive_bayes.o src/shufflescan.o src/rowstore.o src/model_ddl.o src/numeric.o src/numeric_pg.o
DATA = sql/pg_db4ai--0.4.0.sql sql/pg_db4ai--0.3.0--0.4.0.sql sql/pg_db4ai--0.3.0.sql sql/pg_db4ai--0.2.0--0.3.0.sql sql/pg_db4ai--0.1.0.sql sql/pg_db4ai--0.2.0.sql sql/pg_db4ai--0.1.0--0.2.0.sql
PGFILEDESC = "pg_db4ai - PostgreSQL-native machine learning"
PG_CONFIG ?= pg_config
WITH_XGBOOST ?= 1
WITH_OPENMP ?= 0
WITH_SIMD ?= 1
ifneq ($(filter $(WITH_SIMD),0 1),$(WITH_SIMD))
$(error WITH_SIMD must be 0 or 1)
endif
ifneq ($(filter $(WITH_OPENMP),0 1),$(WITH_OPENMP))
$(error WITH_OPENMP must be 0 or 1)
endif
ifeq ($(WITH_OPENMP),1)
ifneq ($(WITH_XGBOOST),1)
$(error WITH_OPENMP=1 requires WITH_XGBOOST=1)
endif
endif
XGB_JOBS ?= 2
PG_CPPFLAGS = -I$(srcdir)/src
PG_CFLAGS = -ffp-contract=off -std=c11 -Werror=implicit-function-declaration
# SIMD-only directives do not introduce OpenMP threads or libgomp/libomp.
# Probe the actual C compiler; unsupported compilers retain a buildable fallback.
SIMD_CC_OK := $(shell printf 'int main(void){return 0;}\n' | $(CC) -x c -fopenmp-simd -c -o /dev/null - 2>/dev/null && echo 1)
ifeq ($(WITH_SIMD)$(SIMD_CC_OK),11)
PG_CPPFLAGS += -DDB4AI_WITH_SIMD=1
SIMD_CFLAGS = -fopenmp-simd
else
PG_CPPFLAGS += -DDB4AI_WITH_SIMD=0
endif
SHLIB_LINK = -lm
ifeq ($(WITH_OPENMP),1)
PG_CPPFLAGS += -DDB4AI_WITH_OPENMP
SHLIB_LINK += -fopenmp
endif

ifeq ($(WITH_XGBOOST),1)
PG_CPPFLAGS += -DDB4AI_WITH_XGBOOST -I$(srcdir)/third_party/xgboost/include
SHLIB_LINK += $(CURDIR)/build/libxgboost.a $(CURDIR)/build/libdmlc.a -lstdc++ -pthread
ifeq ($(shell uname -s),Linux)
SHLIB_LINK += -Wl,--exclude-libs,ALL
endif
endif

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

# First-party header changes and build switches must invalidate every object.
$(OBJS) $(OBJS:.o=.bc): src/db4ai.h src/extended.h src/numeric.h build/extension-config
src/numeric.o src/numeric.bc: src/numeric_impl.h
# Optimize only audited pure numerical kernels; no fast-math or native ISA.
src/numeric.o: override CFLAGS += -O3 $(SIMD_CFLAGS) -ffp-contract=off
src/numeric.bc: override BITCODE_CFLAGS += -O3 -ffp-contract=off $(if $(filter 1,$(WITH_SIMD)),-fopenmp-simd,)
.PHONY: FORCE
FORCE:
build/extension-config: FORCE
	@mkdir -p build
	@printf '%s\n' '$(PG_CONFIG)|$(shell $(PG_CONFIG) --version)|$(WITH_XGBOOST)|$(WITH_OPENMP)|$(WITH_SIMD)|$(SIMD_CC_OK)|$(CC)|$(CFLAGS)|$(PG_CFLAGS)' > $@.tmp
	@cmp -s $@.tmp $@ && rm $@.tmp || mv $@.tmp $@

build/xgboost-config: FORCE
	@mkdir -p build
	@printf '%s\n' '$(WITH_OPENMP)' > $@.tmp
	@cmp -s $@.tmp $@ && rm $@.tmp || mv $@.tmp $@

ifeq ($(WITH_XGBOOST),1)
$(shlib): build/xgboost.stamp
build/xgboost.stamp: third_party/xgboost.lock.json build/xgboost-config
	@test -f third_party/xgboost/dmlc-core/CMakeLists.txt || (echo 'Vendored XGBoost source is missing; builds never download dependencies.'; exit 1)
	cmake -S third_party/xgboost -B build/xgboost \
	  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
	  -DBUILD_STATIC_LIB=ON -DKEEP_BUILD_ARTIFACTS_IN_BINARY_DIR=ON \
	  -DUSE_OPENMP=$(if $(filter 1,$(WITH_OPENMP)),ON,OFF) -DUSE_CUDA=OFF -DUSE_NCCL=OFF \
	  -DPLUGIN_FEDERATED=OFF -DPLUGIN_RMM=OFF -DPLUGIN_SYCL=OFF \
	  -DGOOGLE_TEST=OFF -DBUILD_WITH_SYSTEM_DMLC=OFF
	cmake --build build/xgboost --target xgboost --parallel $(XGB_JOBS)
	cp "$$(find build/xgboost -name libxgboost.a -print -quit)" build/libxgboost.a
	cp "$$(find build/xgboost -name libdmlc.a -print -quit)" build/libdmlc.a
	touch $@
endif

.PHONY: test verify-vendor clean-vendor-build
test:
	PG_CONFIG='$(PG_CONFIG)' WITH_XGBOOST='$(WITH_XGBOOST)' WITH_OPENMP='$(WITH_OPENMP)' WITH_SIMD='$(WITH_SIMD)' CC='$(CC)' sh test/run.sh
verify-vendor:
	python3 tools/vendor_xgboost.py --verify
clean-vendor-build:
	rm -rf build

# Reproducible native compiler remarks and disassembly, without training models.
.PHONY: simd-report
simd-report:
	@mkdir -p build/simd-report
	@$(CC) --version > build/simd-report/compiler.txt
	@if $(CC) --version | head -1 | grep -qi clang; then \
	  flags='-Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize'; \
	else flags='-fopt-info-vec-all'; fi; \
	$(CC) -std=c11 -O3 -ffp-contract=off $(SIMD_CFLAGS) $(PG_CPPFLAGS) $$flags \
	  -c $(srcdir)/src/numeric.c -o build/simd-report/numeric.o 2> build/simd-report/vectorization.txt
	@objdump -d build/simd-report/numeric.o > build/simd-report/disassembly.txt
