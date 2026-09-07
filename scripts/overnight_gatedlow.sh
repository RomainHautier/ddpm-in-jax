#!/bin/bash
# Gated-dose fine-tunes with the LOW-BAND terms, three regimes sequentially.
#
# The newest gated formulation as a training reward: dose_mid [16,32) + dose_hi [32,96)
# (deadband 0.2, per-sample conditional targets, gt1k recipe) PLUS the v8 low-mid term
# [7,16) (weight 2) and the v9 large-scale anchor [1,5) (weight 2, deadband 0.05,
# target = each sample's own recon energy). Recipes per regime mirror the canonical
# fine-tunes: mt2k (temp 2.5, pde 1.0) and the repaired pdew02 runs at 4000/8000
# (temp 3.0/3.5, pde 0.2). Canonical generated-regime splits: train 0-7.
#
#   nohup bash scripts/overnight_gatedlow.sh > monitoring/gatedlow_driver.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=~/venv-ddpm/bin/python

run_regime () {
  local RE=$1 TEMP=$2 PDEW=$3
  local GT=flow-data/generated/gen_fnons_re${RE}_kf_1024to256_20seq.npy
  local STATS=base_results/regime_stats_re${RE}_measured_train.npz
  local DIR=monitoring/ddpo_re${RE}_gatedlow_ckpts
  local COMMON=(--re "$RE" --gt_override "$GT" --train_seqs_override 0,1,2,3,4,5,6,7
    --stats "$STATS" --scales_re 1000 --grid_factor 4 --base_ddim_init
    --sampler ddim --policy_ddim_steps 50 --eta 1.0 --chain_starts 100 75
    --highk_lo 10 --pde_weight "$PDEW"
    --dose_weight 3.0 --dose_lowmid_weight 2.0 --dose_low_weight 2.0
    --sampling_temp "$TEMP" --kl_coef 0.01 --policy_ema 0.99 --clip_eps 0.2 --seed 0)

  echo "=== GATEDLOW Re=$RE SMOKE $(date -u +%H:%M:%S) ==="
  if ! $PY src/ddpo_ft/train_claude.py --smoke "${COMMON[@]}" \
       --save_dir "monitoring/ddpo_re${RE}_gatedlow_smoke"; then
    echo "=== GATEDLOW Re=$RE SMOKE FAILED - aborting the sequence ==="
    return 1
  fi
  echo "=== GATEDLOW Re=$RE FULL RUN $(date -u +%H:%M:%S) ==="
  $PY src/ddpo_ft/train_claude.py "${COMMON[@]}" \
      --n_outer 600 --lr 5e-5 --save_dir "$DIR" \
      || { echo "=== GATEDLOW Re=$RE FULL RUN FAILED ==="; return 1; }
  echo "=== GATEDLOW Re=$RE DONE $(date -u +%H:%M:%S) ==="
}

run_regime 2000 2.5 1.0 || exit 1
run_regime 4000 3.0 0.2 || exit 1
run_regime 8000 3.5 0.2 || exit 1
echo "=== GATEDLOW ALL DONE ==="
