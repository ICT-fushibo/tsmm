#!/bin/bash
echo "=== Submitting Step 4 profiling jobs (row-major only) ==="
for KERN in omp_s4_3d; do
    case "$KERN" in omp_s4_3d) KTAG="s4_3d" ;; omp_s4_mn) KTAG="s4_mn" ;; esac
    TAG="prof_${KTAG}_row"
    echo "  -> $TAG"
    sbatch --job-name="pf_${KTAG}_r" --output="logs/${TAG}_%j.out" --error="logs/${TAG}_%j.err" \
           --export=ALL,BIN_SUFFIX=_omp_s4,KERNEL="$KERN",LAYOUT="row",TAG="$TAG",NO_BUILD=1 \
           scripts/run_profile_omp.sh
done
echo "Submitted 2 jobs."
