# ============================================================================
# node_config.mk — BSCC-T6 计算节点配置 & BLAS 基线设置
#
# 用法：在 Makefile 中 include 此文件
#   include scripts/node_config.mk
#
# 节点：m3ca0xxx.para.bscc (2× Intel Xeon Platinum 9242)
# ============================================================================

# ---- 节点硬件参数 -----------------------------------------------------------
NODE_CPU        := Intel(R) Xeon(R) Platinum 9242
NODE_SOCKETS    := 2
NODE_CORES      := 96          # 48 cores/socket × 2, HT off
NODE_NUMA       := 4           # 24 cores per NUMA node
NODE_FREQ_BASE  := 2.30        # GHz (base)
NODE_FREQ_MAX   := 3.80        # GHz (max turbo)
NODE_L1D        := 32          # KB per core
NODE_L1I        := 32          # KB per core
NODE_L2         := 1024        # KB per core
NODE_L3         := 36608       # KB per socket (36 MB)
NODE_MEM_GB     := 384         # total (≈96 GB per NUMA node)

# ---- ISA -------------------------------------------------------------------
NODE_ISA        := AVX-512      # AVX-512F/CD/DQ/BW/VL/VNNI
NODE_FLOP_CYCLE := 32           # 2×512-bit FMA units: 2×(512/64)×2=32 DP FLOPs/cycle
NODE_PEAK_CORE  := 74.56        # GFLOPS per core @ 2.33 GHz
NODE_PEAK_SOCKET:= 3578.88      # GFLOPS per socket
NODE_PEAK_TOTAL := 7157.76      # GFLOPS per node

# ---- BLAS 库路径 (module load 后可用) ---------------------------------------
# OpenBLAS 0.3.17
OPENBLAS_MODULE := openblas/0.3.17-ips18
OPENBLAS_LIB    := /public5/soft/openblas/0.3.17/lib
OPENBLAS_INC    := /public5/soft/openblas/0.3.17/include

# Intel oneAPI 2022.1 (含 MKL)
INTEL_MODULE    := intel/2022.1
MKLROOT_PATH    := /public5/soft/oneAPI/2022.1/mkl/latest
MKL_LIB         := $(MKLROOT_PATH)/lib/intel64
MKL_INC         := $(MKLROOT_PATH)/include

# 编译器 (icpx = Intel Clang-based C++ compiler)
ICC             := icpx
ICPC            := icpc

# ---- 线程/NUMA 绑定 baseline 设置 -------------------------------------------
# 线程数测试序列
BENCH_THREADS   := 1 2 4 8 16 24 48 96

# NUMA 绑定规则 (由 run_benchmark.sh 使用)
# 1-24 threads:  node 0 only
#    NUMA_BIND_1_24 := --cpunodebind=0 --membind=0
# 48 threads:    node 0-1, interleaved
#    NUMA_BIND_48  := --cpunodebind=0-1 --interleave=0-1
# 96 threads:    all nodes, interleaved
#    NUMA_BIND_96  := --cpunodebind=0-3 --interleave=all

# ---- OpenMP / MKL 线程环境变量 (公平对比 baseline) --------------------------
OMP_BASELINE := OMP_PROC_BIND=spread \
                OMP_PLACES=cores \
                OMP_NUM_THREADS=<N> \
                MKL_NUM_THREADS=<N> \
                OPENBLAS_NUM_THREADS=<N> \
                MKL_DYNAMIC=FALSE \
                KMP_AFFINITY=granularity=fine,compact,1,0

# ---- 计时 baseline ----------------------------------------------------------
BENCH_WARMUP      := 10       # warmup iterations
BENCH_RUNS        := 20       # timed iterations
BENCH_WARMUP_LARGE := 3       # warmup for large-memory shapes (S1, S8)
BENCH_RUNS_LARGE   := 5       # timed runs for large shapes
