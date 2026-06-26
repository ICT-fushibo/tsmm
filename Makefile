# ============================================================================
#  TSMM — C = A^T * B  optimisation project
#
#  Node config:  see scripts/node_config.mk
#  Benchmark:    bash scripts/run_benchmark.sh
#
#  Targets:
#    make               build lib + test + benchmark (no BLAS)
#    make blas          build lib + test + benchmark (with BLAS)
#    make test          build & run correctness test
#    make benchmark     build & run performance benchmark
#    make slurm         submit benchmark job via SLURM
#    make clean         remove build artefacts
#
#  BLAS integration (choose one):
#    make blas BLAS=openblas       # use OpenBLAS  (pkg-config)
#    make blas BLAS=mkl            # use Intel MKL  ($MKLROOT)
#    make blas BLAS_LIBS=... BLAS_CFLAGS=...   # manual override
#
#  Thread control for fair comparison:
#    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./benchmark
# ============================================================================

# ---- 引入节点配置（BLAS 路径 / 硬件参数）-----------------------------------
-include scripts/node_config.mk

# --- Compiler & flags --------------------------------------------------
CC       := gcc
CFLAGS   ?= -O3 -march=native -mtune=native
CFLAGS   += -Wall -Wextra -std=c11 -pedantic
CFLAGS   += -D_POSIX_C_SOURCE=200809L   # expose posix_memalign, clock_gettime
CFLAGS   += -fno-omit-frame-pointer   # helpful for perf profiling

# Directories
SRCDIR   := src
INCDIR   := include
TESTDIR  := tests
BUILDDIR := build

# Source files (library)
LIB_SRCS := $(SRCDIR)/tsmm_naive.c $(SRCDIR)/tsmm_tiled.c $(SRCDIR)/tsmm_utils.c
LIB_OBJS := $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.o,$(LIB_SRCS))

# Test executables
TEST_CORRECT := $(BUILDDIR)/test_correctness
BENCHMARK    := $(BUILDDIR)/benchmark

# --- BLAS configuration ------------------------------------------------
# BLAS=openblas  → use pkg-config
# BLAS=mkl       → use MKLROOT and MKL link line advisor
# BLAS_LIBS / BLAS_CFLAGS → manual override

BLAS_LIBS   ?=
BLAS_CFLAGS ?=

ifeq ($(BLAS),openblas)
  BLAS_CFLAGS := $(shell pkg-config --cflags openblas 2>/dev/null)
  BLAS_LIBS   := $(shell pkg-config --libs   openblas 2>/dev/null)
  ifeq ($(BLAS_LIBS),)
    $(error pkg-config could not find openblas.  Install it or set BLAS_LIBS/BLAS_CFLAGS manually)
  endif
else ifeq ($(BLAS),mkl)
  MKLROOT ?= /opt/intel/mkl
  ifeq ($(wildcard $(MKLROOT)),)
    $(error MKLROOT=$(MKLROOT) not found.  Set MKLROOT or use BLAS_LIBS/BLAS_CFLAGS)
  endif
  BLAS_CFLAGS := -I$(MKLROOT)/include -DTSMM_USE_MKL
  BLAS_LIBS   := -L$(MKLROOT)/lib/intel64 -lmkl_rt -lpthread -lm -ldl
endif

# --- BLAS-aware build flags --------------------------------------------
HAS_BLAS := $(if $(BLAS_LIBS),1,0)

ifeq ($(HAS_BLAS),1)
  CFLAGS += -DTSMM_USE_BLAS $(BLAS_CFLAGS)
  LDFLAGS_BLAS := $(BLAS_LIBS)
endif

# OpenMP for thread control (optional)
CFLAGS += -fopenmp
LDFLAGS += -fopenmp

# --- Build targets -----------------------------------------------------
.PHONY: all blas lib test benchmark slurm clean help \
        check bench-submit profile-submit

all: lib $(TEST_CORRECT) $(BENCHMARK)
	@echo "==> Build complete (no BLAS).  Use 'make blas BLAS=openblas' for BLAS."

blas: lib $(TEST_CORRECT) $(BENCHMARK)
	@echo "==> Build complete (with BLAS: $(if $(BLAS_LIBS),$(BLAS),none))."

lib: $(LIB_OBJS)
	@echo "==> Library built."

# Object files
$(BUILDDIR)/%.o: $(SRCDIR)/%.c include/tsmm.h | $(BUILDDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

# Correctness test
$(TEST_CORRECT): $(TESTDIR)/test_correctness.c $(LIB_OBJS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) $< $(LIB_OBJS) $(LDFLAGS) $(LDFLAGS_BLAS) -o $@

# Benchmark
$(BENCHMARK): $(TESTDIR)/benchmark.c $(LIB_OBJS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) $< $(LIB_OBJS) $(LDFLAGS) $(LDFLAGS_BLAS) -o $@

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# --- Run targets -------------------------------------------------------
test: $(TEST_CORRECT)
	@echo "=== Running correctness tests ==="
	$(TEST_CORRECT)

benchmark: $(BENCHMARK)
	@echo "=== Running benchmarks (serial, fair comparison) ==="
	@echo "Setting OMP/MKL/OPENBLAS threads to 1 for fair serial comparison."
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	OPENBLAS_NUM_THREADS=1 \
	$(BENCHMARK)

benchmark-csv: $(BENCHMARK)
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	OPENBLAS_NUM_THREADS=1 \
	$(BENCHMARK) --csv

# --- SLURM submission --------------------------------------------------

# old combined script (kept for backward compat)
slurm:
	sbatch scripts/run_slurm.sh

# correctness only
check:
	sbatch scripts/run_correctness.sh

# performance benchmark (GFLOPS, BW, AI)
bench-submit:
	sbatch scripts/run_benchmark.sh

# hardware profiling (cache miss, CPI, branch miss)
profile-submit:
	sbatch scripts/run_profile.sh

# --- Cleanup -----------------------------------------------------------
clean:
	rm -rf $(BUILDDIR)
	@echo "==> Cleaned build directory."

# --- Help --------------------------------------------------------------
help:
	@echo "TSMM Optimisation Project — Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make                    build all (no BLAS)"
	@echo "  make blas BLAS=openblas build with OpenBLAS"
	@echo "  make blas BLAS=mkl      build with Intel MKL"
	@echo "  make test               run correctness tests"
	@echo "  make benchmark          run performance benchmarks"
	@echo "  make benchmark-csv      benchmarks → CSV (for analysis)"
	@echo "  make slurm              submit to SLURM"
	@echo "  make clean              remove build artefacts"
	@echo ""
	@echo "Variables:"
	@echo "  CC=gcc|icc              compiler choice"
	@echo "  BLAS=openblas|mkl       BLAS library"
	@echo "  MKLROOT=/path/to/mkl    MKL install root"
	@echo "  BLAS_LIBS=...           manual BLAS link flags"
	@echo "  BLAS_CFLAGS=...         manual BLAS compile flags"
	@echo ""
	@echo "Fair comparison (serial baseline):"
	@echo "  The benchmark sets OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"
	@echo "  OPENBLAS_NUM_THREADS=1 automatically."
