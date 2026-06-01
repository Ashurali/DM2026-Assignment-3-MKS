#!/usr/bin/env bash
# Sequential 3-seed queue for the v2 DG config (milder sqrt oversampling).
# One GPU -> one run at a time (shared box; ~1.9GB free). Detach with nohup.
cd "$(dirname "$0")/.." || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dm2026-a3
rm -f logs/dg_v2_queue.done
for s in 42 7 23; do
  echo "===== seed ${s} start $(date) ====="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u -m src.models.train_dg_cisc \
    --gpu --name v2s${s} --seed ${s} --sampler-power 0.5 --per-file-norm --concat-stats \
    > logs/dg_v2s${s}.log 2>&1
  echo "===== seed ${s} done $(date) rc=$? ====="
done
echo ALLDONE > logs/dg_v2_queue.done
