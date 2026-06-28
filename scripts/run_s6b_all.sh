#!/bin/bash
# ============================================================================
# run_s6b_all.sh — S6b (S2 full register-resident) 正确性 + bench + profile
# ============================================================================

echo "=== Pre-building (BIN_SUFFIX=_s6b) ==="
make BIN_SUFFIX=_s6b 2>&1 | tail -2
echo ""

# 1. Correctness (serial 1t + all thread counts OMP)
echo "  -> correctness (s6b, 1t + full thread sweep)"
sbatch --job-name="chk_s6b" \
       --output="logs/check_s6b_%j.out" --error="logs/check_s6b_%j.err" \
       --export=ALL,BIN_SUFFIX=_s6b,KERNEL=s6b \
       scripts/run_correctness_omp_all.sh

# 2. Benchmark (shape 2 only, all threads)
echo "  -> benchmark (S6b, S2 only, 10 threads sweep)"
sbatch --job-name="bm_s6b" \
       --output="logs/bench_s6b_%j.out" --error="logs/bench_s6b_%j.err" \
       --export=ALL,BIN_SUFFIX=_s6b,KERNEL=s6b,SHAPE=2,LAYOUT=row,NO_BUILD=1 \
       scripts/run_benchmark_s5.sh

# 3. Profile (shape 2 only, all threads, perf stat)
echo "  -> profile (S6b, S2 only, perf stat)"
sbatch --job-name="pf_s6b" \
       --output="logs/prof_s6b_%j.out" --error="logs/prof_s6b_%j.err" \
       --export=ALL,BIN_SUFFIX=_s6b,KERNEL=s6b,SHAPE=2,NO_BUILD=1 \
       scripts/run_profile_s5.sh

echo ""
echo "=== Submitted 3 jobs ==="
