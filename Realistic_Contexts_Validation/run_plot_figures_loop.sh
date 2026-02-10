#!/bin/bash
#
#SBATCH --job-name=plot_figs
#SBATCH --partition=jang
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=plot_figs_%j.out
#SBATCH --error=plot_figs_%j.err

set -euo pipefail

# Load Python (no CUDA needed)
module load bbc2/python3/python-3.13.7
unset PYTHONHOME PYTHONPATH

OUTER=/varidata/research/projects/jang/TommyGoralski/flash_semicrf
REPO=$OUTER/flash-semicrf

# UPDATED: point to the single sweep output directory (one benchmark_full.csv)
RESULTS=$OUTER/results/realistic/human_exonK_from_stats_sweep

cd "$OUTER"
source .venv/bin/activate

cd "$REPO"

csv="$RESULTS/benchmark_full.csv"
test -f "$csv" || { echo "Missing CSV: $csv" >&2; exit 2; }

fig_dir="$RESULTS/figures"
mkdir -p "$fig_dir"

echo "=== Plotting sweep results ==="
echo "Input:  $RESULTS"
echo "CSV:    $csv"
echo "Output: $fig_dir"

python ./benchmarks/plot_figures.py \
  --input-dir "$RESULTS" \
  --output-dir "$fig_dir" \
  --csv benchmark_full.csv

echo "All plotting complete."
