#!/bin/bash
#
#SBATCH --job-name=real_bench_sweep
#SBATCH --partition=jang
#SBATCH --nodelist=compute147
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=real_bench_sweep_%j.out
#SBATCH --error=real_bench_sweep_%j.err

set -euo pipefail

module load bbc2/python3/python-3.13.7
module load cuda12.6/toolkit/12.6.2
unset PYTHONHOME PYTHONPATH

cd /varidata/research/projects/jang/TommyGoralski/flash_semicrf
source .venv/bin/activate

cd /varidata/research/projects/jang/TommyGoralski/flash_semicrf/flash-semicrf

export CUDA_VISIBLE_DEVICES=3

# Pull the sweep lists from the config generator output
CFGDIR=/varidata/research/projects/jang/TommyGoralski/flash_semicrf/benchmark_configs/human_from_stats_human_exonK
CMDFILE="${CFGDIR}/benchmark_command.txt"
test -f "$CMDFILE" || { echo "Missing benchmark command file: $CMDFILE" >&2; exit 2; }

#T_LIST=$(grep -oE -- '--T[[:space:]]+[^[:space:]]+' "$CMDFILE" | head -n1 | awk '{print $2}')
#K_LIST=$(grep -oE -- '--K[[:space:]]+[^[:space:]]+' "$CMDFILE" | head -n1 | awk '{print $2}')
# Override for reduced sweep
T_LIST="4450,75200"
K_LIST="200,430"

C_VAL=$(grep -oE -- '--C[[:space:]]+[^[:space:]]+' "$CMDFILE" | head -n1 | awk '{print $2}')

test -n "${T_LIST:-}" || { echo "Could not parse --T from $CMDFILE" >&2; exit 3; }
test -n "${K_LIST:-}" || { echo "Could not parse --K from $CMDFILE" >&2; exit 3; }
test -n "${C_VAL:-}"  || { echo "Could not parse --C from $CMDFILE" >&2; exit 3; }

# One sweep output directory (single benchmark_full.csv with MANY configs)
OUTDIR=/varidata/research/projects/jang/TommyGoralski/flash_semicrf/results/realistic/human_exonK_from_stats_sweep
mkdir -p "$OUTDIR"

echo "=== Sweep run ==="
echo "T_LIST=${T_LIST}"
echo "K_LIST=${K_LIST}"
echo "C=${C_VAL}"
echo "OUTDIR=${OUTDIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo

python ./benchmarks/benchmark_memory_analysis.py \
  --device cuda:0 \
  --T "${T_LIST}" \
  --K "${K_LIST}" \
  --C "${C_VAL}" \
  --B 4 \
  --repeats 3 \
  --output-dir "${OUTDIR}"

