#!/usr/bin/env bash
# Run 2 — Phase 3 LightGBM with SMOTE on minority classes (full feature catalog).
# Output: oof/lgbm_full_smote_v1_*.{npy,json}, submissions/sub_lgbm_full_smote_v1.csv

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python}"
echo ">>> Run 2: full features + SMOTE"
$PY -m src.models.train_lgbm_full \
    --gpu \
    --cache-features \
    --smote \
    --name smote_v1
echo ">>> Run 2 done. CSV → submissions/sub_lgbm_full_smote_v1.csv"
