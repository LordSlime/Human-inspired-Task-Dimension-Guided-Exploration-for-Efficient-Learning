#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PACKAGE_ROOT"

python compute_final_metrics_cross_runs.py \
  --config cross_run_metric_config.json \
  --save-dir paper_figures \
  --num-runs 5 \
  --saved-std-ddof 0

python compute_final_metrics_clustering.py \
  --config clustering_metric_config.json \
  --save-dir paper_figures \
  --num-runs 5 \
  --saved-std-ddof 0

python plot_final.py
