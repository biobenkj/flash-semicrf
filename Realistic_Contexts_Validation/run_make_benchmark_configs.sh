#!/bin/bash
#
#SBATCH --job-name=make_real_bench_configs
#SBATCH --partition=jang
#SBATCH --nodelist=compute147
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=make_real_bench_configs_%j.out
#SBATCH --error=make_real_bench_configs_%j.err

set -euo pipefail

PROJ=/varidata/research/projects/jang/TommyGoralski/torch_semimarkov
SUMMARY=${PROJ}/stats_human/summary.csv
OUTDIR=${PROJ}/benchmark_configs/human_from_stats_human_exonK

# If your cluster uses modules, uncomment and adjust:
# module load python/3.10

# If you use a virtualenv or conda, activate it here:
# source ~/venvs/torch_semimarkov/bin/activate
# conda activate torch_semimarkov

cd "${PROJ}"
mkdir -p "${OUTDIR}"

python make_benchmark_configs_from_summary.py \
  --summary-csv "${SUMMARY}" \
  --K-source exon \
  --round \
  --C 6 \
  --out-dir "${OUTDIR}"
