#!/bin/bash
# ============================================================================
# check_env.sh — 超算环境探测脚本
#
# 用途：在超算登录节点或计算节点上运行，收集：
#   1. 可用 BLAS 库（OpenBLAS / MKL）的路径
#   2. CPU 型号、核数、频率、SIMD 支持
#   3. 理论峰值 GFLOPS（单核 & 全节点）
#
# 用法：
#   # 交互式（登录节点）
#   bash scripts/check_env.sh
#
#   # 提交到计算节点
#   sbatch scripts/check_env.sh
#
#   # 直接在计算节点运行
#   srun --pty bash scripts/check_env.sh
# ============================================================================

set -euo pipefail

# ---------- SLURM job header (ignored when run directly) -------------------
#SBATCH --job-name=tsmm_env_check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=logs/check_env_%j.out
#SBATCH --error= logs/check_env_%j.err

SEP="============================================================================"

echo "$SEP"
echo "  TSMM — HPC Environment Check"
echo "  Date: $(date)"
echo "  Host: $(hostname)"
echo "$SEP"

# ===========================================================================
# 1.  Module system detection
# ===========================================================================
echo ""
echo "--- 1. Module system ---"
echo ""

# Detect which module system is in use
if command -v module &>/dev/null; then
    # LMOD (Lua)
    echo "Module system: LMOD (Lmod)"
    echo ""
    echo "Available OpenBLAS modules:"
    module avail openblas 2>&1 | head -30 || echo "  (none found)"
    echo ""
    echo "Available MKL / oneAPI modules:"
    module avail mkl        2>&1 | head -20 || true
    module avail oneapi     2>&1 | head -20 || true
    module avail intel      2>&1 | head -20 || true
    echo ""
    echo "Whatis for openblas:"
    module whatis openblas 2>&1 | head -20 || echo "  (not available)"
    echo ""
    echo "Whatis for mkl:"
    module whatis mkl      2>&1 | head -20 || echo "  (not available)"

elif command -v module 2>/dev/null | grep -q . ; then
    # Also LMOD (Lmod), already handled above
    :
fi

if command -v modulecmd 2>/dev/null && ! command -v module 2>/dev/null; then
    echo "(old-style modulecmd — run 'modulecmd bash avail' to list)"
fi

if command -v spack 2>/dev/null; then
    echo ""
    echo "Spack: available"
    spack find openblas 2>/dev/null | head -10 || true
    spack find intel-mkl 2>/dev/null | head -10 || true
    spack find intel-oneapi-mkl 2>/dev/null | head -10 || true
fi

# ===========================================================================
# 2.  Search for BLAS libraries on disk
# ===========================================================================
echo ""
echo "$SEP"
echo "--- 2. BLAS library search (filesystem) ---"
echo ""

# --- Use module system to find paths (fast, no filesystem search) ---
# OpenBLAS
echo "=== OpenBLAS paths (via module show) ==="
if module is-loaded openblas 2>/dev/null; then
    echo "  openblas is loaded"
fi
OPENBLAS_MOD=$(module avail -t openblas 2>&1 | grep -i openblas | head -1 | xargs || echo "")
if [ -n "$OPENBLAS_MOD" ]; then
    echo "  Module:    $OPENBLAS_MOD"
    MOD_SHOW=$(module show "$OPENBLAS_MOD" 2>&1 || true)
    echo "$MOD_SHOW" | grep -iE 'PATH|LIB|INCLUDE|LD_' | head -10 || true
fi

echo ""

# Intel / MKL
echo "=== Intel MKL paths (via module show) ==="
INTEL_MOD=$(module avail -t intel 2>&1 | grep -E '^intel/' | sort -V | tail -1 | xargs || echo "")
if [ -n "$INTEL_MOD" ]; then
    echo "  Latest Intel module: $INTEL_MOD"
    MOD_SHOW=$(module show "$INTEL_MOD" 2>&1 || true)
    echo "$MOD_SHOW" | grep -iE 'PATH|MKL|LIB|INCLUDE|LD_' | head -15 || true
    # Try to load and check MKLROOT
    echo ""
    echo "  After 'module load $INTEL_MOD', MKLROOT would be checked."
fi
echo ""

echo "  (skipping filesystem find — module show paths above are sufficient)"

# pkg-config
echo ""
echo "pkg-config:"
for pkg in openblas mkl-dynamic-lp64-seq mkl-dynamic-lp64-iomp; do
    if pkg-config --exists "$pkg" 2>/dev/null; then
        echo "  $pkg:"
        echo "    cflags: $(pkg-config --cflags "$pkg" 2>/dev/null)"
        echo "    libs:   $(pkg-config --libs   "$pkg" 2>/dev/null)"
    fi
done

echo ""
echo "MKLROOT = ${MKLROOT:-not set}"

# ===========================================================================
# 3.  CPU / hardware topology
# ===========================================================================
echo ""
echo "$SEP"
echo "--- 3. CPU & topology ---"
echo ""

# --- lscpu (most useful) ---
if command -v lscpu &>/dev/null; then
    echo "=== lscpu ==="
    lscpu
    echo ""
fi

# --- /proc/cpuinfo summary ---
echo "=== /proc/cpuinfo summary ==="
CPU_MODEL=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo "unknown")
echo "  Model name:       $CPU_MODEL"

N_SOCKETS=$(grep -c 'physical id' /proc/cpuinfo 2>/dev/null || echo 1)
# Count unique physical IDs
N_SOCKETS=$(grep 'physical id' /proc/cpuinfo 2>/dev/null | sort -u | wc -l)
echo "  Sockets:          $N_SOCKETS"

N_CORES=$(grep -c '^processor' /proc/cpuinfo 2>/dev/null || echo "?")
echo "  Logical CPUs:     $N_CORES"

N_CORES_PER_SOCKET=$(grep 'cpu cores' /proc/cpuinfo 2>/dev/null | head -1 | awk -F: '{print $2}' | xargs || echo "?")
echo "  Cores per socket: $N_CORES_PER_SOCKET"

# Hyperthreading?
THREADS_PER_CORE=$(grep -m1 'siblings' /proc/cpuinfo 2>/dev/null | awk -F: '{print $2}' | xargs || echo "?")
if [ "$THREADS_PER_CORE" != "?" ] && [ "$N_CORES_PER_SOCKET" != "?" ]; then
    HT=$(( THREADS_PER_CORE / N_CORES_PER_SOCKET ))
    echo "  Threads/core:     $HT  ($([ $HT -gt 1 ] && echo 'HT ON' || echo 'HT OFF'))"
fi

echo ""

# --- NUMA ---
if command -v numactl &>/dev/null; then
    echo "=== numactl --hardware ==="
    numactl --hardware
    echo ""
fi

# --- GPU (optional) ---
if command -v nvidia-smi &>/dev/null; then
    echo "=== nvidia-smi (summary) ==="
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -5
    echo ""
fi

# ===========================================================================
# 4.  SIMD & ISA feature detection
# ===========================================================================
echo "$SEP"
echo "--- 4. SIMD / ISA features ---"
echo ""

if [ -f /proc/cpuinfo ]; then
    FLAGS=$(grep -m1 'flags' /proc/cpuinfo 2>/dev/null || echo "")

    echo "  AVX-512F:      $(echo "$FLAGS" | grep -q 'avx512f'      && echo 'YES' || echo 'no')"
    echo "  AVX-512CD:     $(echo "$FLAGS" | grep -q 'avx512cd'     && echo 'YES' || echo 'no')"
    echo "  AVX-512DQ:     $(echo "$FLAGS" | grep -q 'avx512dq'     && echo 'YES' || echo 'no')"
    echo "  AVX-512BW:     $(echo "$FLAGS" | grep -q 'avx512bw'     && echo 'YES' || echo 'no')"
    echo "  AVX-512VL:     $(echo "$FLAGS" | grep -q 'avx512vl'     && echo 'YES' || echo 'no')"
    echo "  AVX2:          $(echo "$FLAGS" | grep -q 'avx2'          && echo 'YES' || echo 'no')"
    echo "  AVX:           $(echo "$FLAGS" | grep -q 'avx '          && echo 'YES' || echo 'no')"
    echo "  FMA:           $(echo "$FLAGS" | grep -q 'fma'           && echo 'YES' || echo 'no')"
    echo "  SSE4.2:        $(echo "$FLAGS" | grep -q 'sse4_2'        && echo 'YES' || echo 'no')"

    # Count FMA units (heuristic from CPU model)
    # Intel Skylake-SP/Cascade Lake/Ice Lake: 2× AVX-512 FMA
    # Intel Sapphire Rapids: 2× AVX-512 FMA (but AVX-512 enabled in golden cove)
    # AMD Zen4: 1× AVX-512 FMA (double-pumped 256-bit)
    echo ""
    echo "  ISA heuristic based on CPU model:"
    case "$CPU_MODEL" in
        *"Platinum"*|*"Gold"*|*"Silver"*|*"Bronze"*)
            echo "    Intel Xeon Scalable — assuming 2× 512-bit FMA units"
            ;;
        *"Xeon"*"Phi"*)
            echo "    Intel Xeon Phi (KNL/KNM) — 2× 512-bit FMA"
            ;;
        *"Xeon"*)
            echo "    Intel Xeon — check generation"
            ;;
        *"EPYC"*|*"Ryzen"*)
            echo "    AMD EPYC/Ryzen — Zen4 has AVX-512 (1× 512-bit via 2×256-bit)"
            echo "    Earlier Zen have AVX2/FMA (2× 256-bit FMA)"
            ;;
        *"Kunpeng"*|*"TaiShan"*|*"AArch64"*|*"aarch64"*)
            echo "    ARM — NEON / SVE, peak FLOPs depend on SVE width"
            ;;
        *)
            echo "    (unknown model — see lscpu output above)"
            ;;
    esac
fi

# ===========================================================================
# 5.  Theoretical peak GFLOPS calculation
# ===========================================================================
echo ""
echo "$SEP"
echo "--- 5. Theoretical peak GFLOPS (double-precision) ---"
echo ""

# --- Parse CPU frequency ---
CPU_MHZ=""
if command -v lscpu &>/dev/null; then
    # Try to get max frequency
    CPU_MHZ=$(lscpu 2>/dev/null | grep -i 'max mhz\|CPU max MHz\|CPU MHz' | head -1 | awk -F: '{print $2}' | xargs | cut -d. -f1 || echo "")
fi
if [ -z "$CPU_MHZ" ] || [ "$CPU_MHZ" = "" ]; then
    CPU_MHZ=$(grep -m1 'cpu MHz' /proc/cpuinfo 2>/dev/null | awk -F: '{print $2}' | xargs | cut -d. -f1 || echo "2500")
fi
if [ -z "$CPU_MHZ" ] || [ "$CPU_MHZ" = "0" ]; then
    CPU_MHZ=2500  # fallback
fi

CPU_GHZ=$(awk "BEGIN { printf \"%.2f\", $CPU_MHZ / 1000 }")

echo "  CPU frequency:    ${CPU_MHZ} MHz  (${CPU_GHZ} GHz)"
echo "  Sockets:          $N_SOCKETS"
echo "  Cores per socket: $N_CORES_PER_SOCKET"

# --- FLOPs per cycle per core ---
# We determine this from ISA flags
HAS_AVX512F=$(grep -m1 'flags' /proc/cpuinfo 2>/dev/null | grep -q 'avx512f' && echo 1 || echo 0)
HAS_AVX2=$(grep -m1 'flags' /proc/cpuinfo 2>/dev/null | grep -q 'avx2' && echo 1 || echo 0)
HAS_FMA=$(grep -m1 'flags' /proc/cpuinfo 2>/dev/null | grep -q 'fma' && echo 1 || echo 0)

if [ "$HAS_AVX512F" = "1" ]; then
    # Assume 2× 512-bit FMA units for server CPUs
    # Each FMA: 512/64 = 8 doubles, FMA = 2 ops → 16 FLOPs/cycle per unit
    FLOP_PER_CYCLE=32   # 2 units × 8 doubles × 2 ops (FMA)
    ISA_LABEL="AVX-512 (2× 512-bit FMA)"
elif [ "$HAS_AVX2" = "1" ] && [ "$HAS_FMA" = "1" ]; then
    # 2× 256-bit FMA units
    # Each FMA: 256/64 = 4 doubles, FMA = 2 ops → 8 FLOPs/cycle per unit
    FLOP_PER_CYCLE=16   # 2 units × 4 doubles × 2 ops (FMA)
    ISA_LABEL="AVX2+FMA (2× 256-bit FMA)"
elif [ "$HAS_AVX2" = "1" ]; then
    # AVX2 add + mul (no FMA) → 2 ops × 4 doubles = 8, plus another unit → 16?
    # Safer: use 8
    FLOP_PER_CYCLE=8
    ISA_LABEL="AVX2 (no FMA)"
else
    # SSE: 2× 128-bit, 2 doubles each, add + mul → 4 FLOPs/cycle
    FLOP_PER_CYCLE=4
    ISA_LABEL="SSE2 (128-bit)"
fi

echo "  ISA:              $ISA_LABEL"
echo "  DP FLOPs/cycle:   $FLOP_PER_CYCLE (per core)"

# --- Total peak ---
# Per core
PEAK_PER_CORE_GFLOPS=$(awk "BEGIN { printf \"%.2f\", $CPU_GHZ * $FLOP_PER_CYCLE }")
echo ""
echo "  Peak per core:    ${PEAK_PER_CORE_GFLOPS} GFLOPS"

# Per socket
if [ "$N_CORES_PER_SOCKET" != "?" ] && [ -n "$N_CORES_PER_SOCKET" ]; then
    PEAK_PER_SOCKET=$(awk "BEGIN { printf \"%.2f\", $CPU_GHZ * $FLOP_PER_CYCLE * $N_CORES_PER_SOCKET }")
    echo "  Peak per socket:  ${PEAK_PER_SOCKET} GFLOPS  ($N_CORES_PER_SOCKET cores)"
else
    PEAK_PER_SOCKET=0
fi

# Whole node
if [ "$N_CORES_PER_SOCKET" != "?" ] && [ -n "$N_CORES_PER_SOCKET" ] && [ "$N_SOCKETS" != "?" ]; then
    TOTAL_CORES=$((N_CORES_PER_SOCKET * N_SOCKETS))
    PEAK_TOTAL=$(awk "BEGIN { printf \"%.2f\", $CPU_GHZ * $FLOP_PER_CYCLE * $TOTAL_CORES }")
    echo "  Peak per node:    ${PEAK_TOTAL} GFLOPS  ($TOTAL_CORES cores, $N_SOCKETS sockets)"
fi

echo ""
echo "  --- Efficiency targets for your TSMM ---"
echo "  Good serial:      > 70%  of peak-per-core"
echo "  Good threaded:    > 50%  of peak-per-node"
echo "  World-class:      > 85%  of peak"

# ===========================================================================
# 6.  Summary / short version for quick reference
# ===========================================================================
echo ""
echo "$SEP"
echo "--- SUMMARY ---"
echo ""
echo "  CPU: $CPU_MODEL"
echo "  Config: ${N_SOCKETS}S × ${N_CORES_PER_SOCKET}C  (total $TOTAL_CORES physical cores)"
echo "  Freq:  ${CPU_GHZ} GHz"
echo "  ISA:   $ISA_LABEL  → ${FLOP_PER_CYCLE} DP FLOPs/cycle/core"
echo "  Peak:  ${PEAK_PER_CORE_GFLOPS} GFLOPS/core  |  ${PEAK_TOTAL} GFLOPS/node"
echo ""
echo "  Recommended BLAS for this node:"
if [ "$HAS_AVX512F" = "1" ]; then
    echo "    → Intel MKL (best AVX-512 tuning) or OpenBLAS with USE_OPENMP=1 TARGET=SKYLAKEX"
elif [ "$HAS_AVX2" = "1" ]; then
    echo "    → OpenBLAS (TARGET=HASWELL) or Intel MKL"
else
    echo "    → OpenBLAS or system BLAS"
fi
echo ""
echo "  Run 'make blas BLAS=openblas' or 'make blas BLAS=mkl' after module load."
echo "$SEP"
