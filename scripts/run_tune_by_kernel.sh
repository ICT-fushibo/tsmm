#!/bin/bash
# ============================================================================
# run_tune_by_kernel.sh — 单个 kernel 的 8 shape × 3 thread 网格搜索 (SLURM job)
#
# 环境变量:
#   KERNEL    omp_s3_3d | omp_s3_mn | omp_s4_3d
#
# 用法:
#   KERNEL=omp_s4_3d sbatch scripts/run_tune_by_kernel.sh
# ============================================================================
#SBATCH --job-name=tune_kern
#SBATCH --output=logs/tune_kern_%j.out
#SBATCH --error=logs/tune_kern_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0

set -euo pipefail

source /public5/soft/modules/module.sh
module purge
module load gcc/10.2.0

KERN="${KERNEL:-omp_s4_3d}"
BIN_SUFFIX="_omp_s4"

echo "=== Tile Tuning: kernel=$KERN ==="
echo "Node: $(hostname)  Date: $(date)"
echo ""

[ "${NO_BUILD:-0}" != "1" ] && make BIN_SUFFIX="$BIN_SUFFIX" 2>&1 | tail -2
echo ""

BIN="./build/benchmark${BIN_SUFFIX}"
SHAPES="1 2 3 4 5 6 7 8"
THREADS="24 48 96"

TI_LIST="16 32 64 96 128"
TJ_LIST="64 128 192 256 384 512"
TK_LIST="64 128 256 384 512"

export OMP_PROC_BIND=spread
export OMP_PLACES=cores

echo "kernel,shape,threads,Ti,Tj,Tk,wall_ms,gflops"
echo ""

for SH in $SHAPES; do
    for NT in $THREADS; do

        # --- NUMA ---
        if [ "$NT" -le 24 ]; then
            NUMA_CMD="numactl --cpunodebind=0 --membind=0"
        elif [ "$NT" -le 48 ]; then
            NUMA_CMD="numactl --cpunodebind=0-1 --interleave=0-1"
        else
            NUMA_CMD="numactl --cpunodebind=0-3 --interleave=all"
        fi
        export OMP_NUM_THREADS=$NT

        # --- Tk loop (skip for MN) ---
        case "$KERN" in
            omp_s3_mn) TK_LOOP="0" ;;
            *)         TK_LOOP="$TK_LIST" ;;
        esac

        best_gflops=0
        best_line=""

        for Ti in $TI_LIST; do
            for Tj in $TJ_LIST; do
                # tile sanely check (> 1MB, also skip if Ti > 2*m or Tj > 2*n won't help)
                TILE_KB=$((Ti * Tj * 8 / 1024))
                if [ "$TILE_KB" -gt 2048 ]; then continue; fi

                for Tk in $TK_LOOP; do
                    if [ "$Tk" = "0" ]; then
                        TK_ARG=""
                    else
                        TK_ARG="--tile-k $Tk"
                    fi

                    RESULT=$($NUMA_CMD $BIN \
                        --kernel "$KERN" \
                        --tile-m "$Ti" --tile-n "$Tj" $TK_ARG \
                        --threads "$NT" --runs 3 --csv \
                        --shape "$SH" 2>/dev/null | grep "^${KERN}," | head -1)

                    [ -z "$RESULT" ] && continue

                    GFLOPS=$(echo "$RESULT" | cut -d',' -f8)
                    WALL=$(echo "$RESULT" | cut -d',' -f7)
                    echo "$KERN,$SH,$NT,$Ti,$Tj,${Tk:-0},$WALL,$GFLOPS"

                    if awk "BEGIN { exit ($GFLOPS > $best_gflops) ? 0 : 1 }" 2>/dev/null; then
                        best_gflops=$GFLOPS
                        best_line="$KERN,$SH,$NT,$Ti,$Tj,${Tk:-0},$WALL,$GFLOPS"
                    fi
                done
            done
        done

        echo "BEST,$best_line"
        echo ""
    done
done

echo ""
echo "=== Done at $(date) ==="
