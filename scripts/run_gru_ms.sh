#!/bin/bash
# Multi-seed gru_pn queue — run 3 seeds sequentially, prob-averaged downstream.
# Robust against inline-quoting issues: $s expands naturally in this file.
cd "$(dirname "$0")/.." || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dm2026-a3
for s in 17 23 41; do
  echo "=== starting seed $s ==="
  python -u -m src.models.train_gru --gpu --name pn_s$s --seed $s \
    --per-file-norm --concat-stats > logs/gru_pn_s$s.log 2>&1
done
touch logs/gru_ms_done.flag
echo "ALL SEEDS DONE"
