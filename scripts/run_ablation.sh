#!/usr/bin/env bash
# Phase 4 — feature-group ablation sweep.
# Trains an LGBM with the full catalog minus one group at a time,
# then writes reports/ablation_features.md (Phase 4 deliverable, feeds Report Q4).
#
# Skips groups that already have a sidecar JSON. Re-run safely.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python}"

# All ablatable groups. Quality is included for completeness; it's only 2
# features and effectively constant on this dataset (the Δ should be ~0).
GROUPS=(
    fft
    autocorr
    subwindow
    gravity
    jerk
    crossaxis
    zerocross
    per_file_norm
    magnitude
    basic_stats
    quality
)

for g in "${GROUPS[@]}"; do
    META="oof/lgbm_full_abl_no_${g}_meta.json"
    if [ -f "$META" ]; then
        echo ">>> Skip $g (already done — $META exists)"
        continue
    fi
    echo ">>> Ablating: --exclude $g"
    $PY -m src.models.train_lgbm_full \
        --gpu \
        --cache-features \
        --exclude "$g" \
        --name "abl_no_${g}"
done

echo ">>> Building Phase-4 ablation table..."
$PY scripts/summarize_ablation.py
echo ">>> Done. See reports/ablation_features.md"
