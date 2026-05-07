#!/usr/bin/env bash
# Run 3 — Optuna tuning on whichever of (v1, smote_v1) had higher CV F1.
# Override the auto-pick by setting USE_SMOTE=yes|no in the environment.
# Override trial count via N_TRIALS (default 30).
# Output: oof/lgbm_full_tuned_*_*.{npy,json}, submissions/sub_lgbm_full_tuned_*.csv

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python}"
N_TRIALS="${N_TRIALS:-30}"

# Auto-detect best base unless USE_SMOTE is set
if [ -z "${USE_SMOTE:-}" ]; then
    USE_SMOTE=$($PY - <<'PY'
import json
from pathlib import Path
oof = Path("oof")
def cv(name):
    p = oof / f"lgbm_full_{name}_meta.json"
    if not p.exists(): return None
    return json.loads(p.read_text())["cv_f1_mean"]
v = cv("v1")
s = cv("smote_v1")
if s is not None and (v is None or s > v):
    print("yes")
else:
    print("no")
PY
)
fi

if [ "$USE_SMOTE" = "yes" ]; then
    NAME="tuned_smote"
    EXTRA="--smote"
else
    NAME="tuned_v1"
    EXTRA=""
fi

echo ">>> Run 3: Optuna tune ($N_TRIALS trials, USE_SMOTE=$USE_SMOTE, name=$NAME)"
$PY -m src.models.train_lgbm_full \
    --gpu \
    --cache-features \
    $EXTRA \
    --tune \
    --n-trials "$N_TRIALS" \
    --name "$NAME"
echo ">>> Run 3 done. CSV → submissions/sub_lgbm_full_${NAME}.csv"
