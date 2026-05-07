#!/usr/bin/env bash
# Sequential orchestration: Run 2 (SMOTE) → Run 3 (Optuna) → Phase-4 ablation sweep.
# Assumes Run 1 (sub_lgbm_full_v1) is already complete.
# Each step is idempotent and can be re-run safely.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python}"

if [ ! -f "oof/lgbm_full_v1_meta.json" ]; then
    echo "ERROR: oof/lgbm_full_v1_meta.json not found. Run 1 must complete first."
    echo "Run: python -m src.models.train_lgbm_full --gpu --cache-features --name v1"
    exit 1
fi

echo "============================================================"
echo " Run 2 — SMOTE on full feature catalog"
echo "============================================================"
bash scripts/run_smote.sh

echo
echo "============================================================"
echo " Run 3 — Optuna tune on the better base"
echo "============================================================"
bash scripts/run_tune.sh

echo
echo "============================================================"
echo " Phase 4 — feature-group ablation sweep"
echo "============================================================"
bash scripts/run_ablation.sh

echo
echo "============================================================"
echo " All Phase-3 / Phase-4 runs complete"
echo "============================================================"
$PY scripts/summarize_runs.py
