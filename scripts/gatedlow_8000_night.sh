#!/bin/bash
# LAYERED Re=8000 fine-tune: the pure-gate terms PLUS the uniform per-shell push
# (spec 0.5, spec_highk 3.0 over [10,96), energy 0.1) - the recipe the pure-gate
# ablation showed is needed for within-band spectral shape. Note dose_low is now
# [1,7) (the banked layered Re=2000 model used [1,5)).
# Waits BY PID for queue v3 (pg4k evals, 2000 training, lambda sweep) to finish.
set -u
cd "$(dirname "$0")/.."
PY=~/venv-ddpm/bin/python
while kill -0 545563 2>/dev/null; do sleep 300; done

FLAGS=(--re 8000 --gt_override flow-data/generated/gen_fnons_re8000_kf_1024to256_20seq.npy
  --train_seqs_override 0,1,2,3,4,5,6,7
  --stats base_results/regime_stats_re8000_measured_train.npz --scales_re 1000
  --grid_factor 4 --base_ddim_init --sampler ddim --policy_ddim_steps 50 --eta 1.0
  --chain_starts 100 75 --highk_lo 10 --pde_weight 0.2
  --dose_weight 3.0 --dose_lowmid_weight 2.0 --dose_low_weight 2.0
  --sampling_temp 3.5 --kl_coef 0.01 --policy_ema 0.99 --clip_eps 0.2 --seed 0)

echo "=== GATEDLOW-8K SMOKE $(date -u +%H:%M:%S) ==="
$PY src/ddpo_ft/train_claude.py --smoke "${FLAGS[@]}" \
  --save_dir monitoring/ddpo_re8000_gatedlow_smoke \
  || { echo "=== GATEDLOW-8K SMOKE FAILED ==="; exit 1; }
rm -rf monitoring/ddpo_re8000_gatedlow_smoke
echo "=== GATEDLOW-8K FULL RUN $(date -u +%H:%M:%S) ==="
$PY src/ddpo_ft/train_claude.py "${FLAGS[@]}" --n_outer 600 --lr 5e-5 \
  --save_dir monitoring/ddpo_re8000_gatedlow_ckpts \
  || { echo "=== GATEDLOW-8K FULL RUN FAILED ==="; exit 1; }
echo "=== GATEDLOW-8K DONE $(date -u +%H:%M:%S) ==="
