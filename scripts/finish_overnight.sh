#!/bin/bash
# Finish the interrupted overnight queue: resume the pure-gate Re=8000 leg from the
# last checkpoint (disk-full crash at ~iter 359), then the Re=2000 leg, then the
# Re=4000 base strength sweep.
#   nohup bash scripts/finish_overnight.sh > monitoring/finish_overnight.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=~/venv-ddpm/bin/python

echo "=== RESUME PUREGATE Re=8000 $(date -u +%H:%M:%S) ==="
$PY src/ddpo_ft/train_claude.py --re 8000 \
  --gt_override flow-data/generated/gen_fnons_re8000_kf_1024to256_20seq.npy \
  --train_seqs_override 0,1,2,3,4,5,6,7 \
  --stats base_results/regime_stats_re8000_measured_train.npz --scales_re 1000 \
  --grid_factor 4 --base_ddim_init --sampler ddim --policy_ddim_steps 50 --eta 1.0 \
  --chain_starts 100 75 --pde_weight 0.2 --pure_gate \
  --dose_weight 3.0 --dose_lowmid_weight 2.0 --dose_low_weight 2.0 \
  --sampling_temp 3.5 --kl_coef 0.01 --policy_ema 0.99 --clip_eps 0.2 \
  --n_outer 600 --lr 5e-5 --save_dir monitoring/ddpo_re8000_puregate_ckpts \
  --resume monitoring/ddpo_re8000_puregate_ckpts/ddpo_re1000_iter0319.pkl \
  || { echo "=== RESUME Re=8000 FAILED ==="; exit 1; }
echo "=== PUREGATE Re=8000 DONE $(date -u +%H:%M:%S) ==="

bash scripts/overnight_puregate.sh 2000 || exit 1
# gate4k strength sweep ON HOLD: the Re=4000 raw dataset was lost in the flow-data wipe
# (no GCS copy) - needs the saved-pool eval path or a regenerated dataset first
echo "=== FINISH-OVERNIGHT ALL DONE ==="
