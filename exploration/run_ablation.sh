#!/usr/bin/env bash
# Phase 4 — feature-group ablation sweep.
# Trains an LGBM with the full catalog minus one group at a time,
# then writes reports/ablation_features.md (Phase 4 deliverable, feeds Report Q4).
#
# Skips groups that already have a sidecar JSON. Re-run safely.
#
# IMPORTANT: do not use a shell variable named `GROUPS` here — that's a
# bash built-in array (the current user's group IDs, e.g. 1002) and
# assignments to it are silently dropped. Use FEATURE_GROUPS instead.

set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python}"

# Hard-coded space-separated list (POSIX-portable; no bash arrays).
# Quality is included for completeness — only 2 features, expected Δ ≈ 0.
FEATURE_GROUPS="fft autocorr subwindow gravity jerk crossaxis zerocross per_file_norm magnitude basic_stats quality"

# Mirror of FEATURE_GROUPS dict in src/features/build.py — used for a
# defensive sanity check below.
KNOWN_GROUPS="basic_stats magnitude gravity jerk fft autocorr subwindow crossaxis zerocross quality per_file_norm"

is_known() {
    local needle="$1"
    local k          # IMPORTANT: local — must not clobber the outer loop's `g`.
    for k in $KNOWN_GROUPS; do
        if [ "$k" = "$needle" ]; then return 0; fi
    done
    return 1
}

echo ">>> Ablation sweep starting"
echo ">>> FEATURE_GROUPS to sweep: $FEATURE_GROUPS"
echo

failed_groups=""

for g in $FEATURE_GROUPS; do
    if ! is_known "$g"; then
        echo "!!! ERROR: '$g' is not in KNOWN_GROUPS — skipping." >&2
        failed_groups="$failed_groups $g"
        continue
    fi

    META="oof/lgbm_full_abl_no_${g}_meta.json"
    if [ -f "$META" ]; then
        echo ">>> Skip $g (already done — $META exists)"
        continue
    fi

    echo ">>> Ablating: --exclude $g"
    if ! $PY -m src.models.train_lgbm_full \
            --gpu \
            --cache-features \
            --exclude "$g" \
            --name "abl_no_${g}"; then
        echo "!!! Run for '$g' failed; continuing with next group." >&2
        failed_groups="$failed_groups $g"
    fi
done

echo
if [ -n "$failed_groups" ]; then
    echo "!!! Failed/skipped groups:$failed_groups"
fi

echo ">>> Building Phase-4 ablation table..."
$PY scripts/summarize_ablation.py
echo ">>> Done. See reports/ablation_features.md"
