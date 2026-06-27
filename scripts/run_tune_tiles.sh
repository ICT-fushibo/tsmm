#!/bin/bash
# ============================================================================
# run_tune_tiles.sh — 单个 shape + kernel 的 Ti/Tj/Tk 网格搜索
#
# 环境变量:
#   SHAPE         shape index (1-8, default 1)
#   KERNEL        omp_s3_3d | omp_s3_mn | omp_s4_3d
#   THREADS       thread count (default 48)
#   BIN_SUFFIX    binary suffix (default _omp_s4)
#   NO_BUILD      skip make if set to 1
#
# 用法:
#   SHAPE=1 KERNEL=omp_s4_3d sbatch scripts/run_tune_tiles.sh
# ============================================================================
#SBATCH --job-name=tune_tile
#SBATCH --output=logs/tune_%j.out
#SBATCH --error=logs/tune_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0

set -euo pipefail

source /public5/soft/modules/module.sh
module purge
module load gcc/10.2.0

# --- parameters ---
SHAPE="${SHAPE:-1}"
KERN="${KERNEL:-omp_s4_3d}"
NTHREADS="${THREADS:-48}"
BIN_SUFFIX="${BIN_SUFFIX:-_omp_s4}"

BIN="./build/benchmark${BIN_SUFFIX}"

echo "=== Tile Tuning ==="
echo "Shape: $SHAPE  Kernel: $KERN  Threads: $NTHREADS"
echo "Node: $(hostname)  Date: $(date)"
echo ""

# build once
[ "${NO_BUILD:-0}" != "1" ] && { make BIN_SUFFIX="$BIN_SUFFIX" 2>&1 | tail -3; echo ""; }

# --- NUMA binding ---
if [ "$NTHREADS" -le 24 ]; then
    NUMA_CMD="numactl --cpunodebind=0 --membind=0"
elif [ "$NTHREADS" -le 48 ]; then
    NUMA_CMD="numactl --cpunodebind=0-1 --interleave=0-1"
else
    NUMA_CMD="numactl --cpunodebind=0-3 --interleave=all"
fi

export OMP_NUM_THREADS=$NTHREADS
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# --- tile search ranges (multiples of 8 for AVX-512 alignment) ---
# Ti choices: capped at m, but m can be small
# Tj choices: capped at n
# Tk choices: capped at k, not used for MN-only

TI_LIST="16 32 64 96 128"
TJ_LIST="64 128 192 256 384 512"
TK_LIST="64 128 256 384 512"

# CSV header
echo "kernel,shape,Ti,Tj,Tk,wall_ms,gflops"

best_gflops=0
best=""

for Ti in $TI_LIST; do
    for Tj in $TJ_LIST; do
        # --- Tk loop (skip for MN kernels) ---
        case "$KERN" in
            omp_s3_mn) TK_LOOP="0" ;;  # MN: Tk unused, use dummy 0
            *)         TK_LOOP="$TK_LIST" ;;
        esac

        for Tk in $TK_LOOP; do
            # Tile size sanity: Ti*Tj should be reasonable
            TILE_BYTES=$((Ti * Tj * 8))
            if [ "$TILE_BYTES" -gt $((2 * 1024 * 1024)) ]; then
                continue  # tile > 2MB, skip (won't fit L2+L1)
            fi

            # Build args
            if [ "$Tk" = "0" ]; then
                TK_ARG=""
            else
                TK_ARG="--tile-k $Tk"
            fi

            # Run benchmark (single shape, min of 3 runs)
            RESULT=$($NUMA_CMD $BIN \
                --kernel "$KERN" \
                --tile-m "$Ti" --tile-n "$Tj" $TK_ARG \
                --threads "$NTHREADS" \
                --runs 3 \
                --csv \
                --shape "$SHAPE" 2>/dev/null | grep "^${KERN}," | head -1)

            if [ -z "$RESULT" ]; then
                echo "SKIP Ti=$Ti Tj=$Tj Tk=$Tk"
                continue
            fi

            # Parse GFLOPS from CSV (8th field: kernel,shape,m,n,k,layout,wall_ms,gflops,...)
            GFLOPS=$(echo "$RESULT" | cut -d',' -f8)
            WALL=$(echo "$RESULT" | cut -d',' -f7)

            echo "$KERN,$SHAPE,$Ti,$Tj,${Tk:-0},$WALL,$GFLOPS"

            # Track best
            if awk "BEGIN { exit ($GFLOPS > $best_gflops) ? 0 : 1 }" 2>/dev/null; then
                best_gflops=$GFLOPS
                best="Ti=$Ti Tj=$Tj Tk=${Tk:-0} gflops=$GFLOPS wall_ms=$WALL"
            fi
        done
    done
done

echo ""
echo "=== Best: $best ==="
echo "=== Done at $(date) ==="
