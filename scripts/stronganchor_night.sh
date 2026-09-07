#!/bin/bash
# STRONG-ANCHOR fine-tunes at Re=8000 and Re=4000: the pure-gate reward with the
# full-band per-sample-scaled spectrum anchor promoted to weight 3.0 (from the dial's
# 0.5) - one per-shell term owning spectral shape across [1,96), continuous pull to
# each sample's own level (no deadband), plus the gated dose terms and the pde guard.
# Targets the two pure-gate pathologies: within-band shape blindness (the k~35 bump)
# and the deadband's zero-force plateau.
# Waits BY PID for queue v5 (transfers, 2000 training, lambda sweep) to finish.
#   nohup bash scripts/stronganchor_night.sh > monitoring/stronganchor_night.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=~/venv-ddpm/bin/python
while kill -0 608240 2>/dev/null; do sleep 300; done

run_one () {
  local RE=$1 TEMP=$2
  local FLAGS=(--re "$RE"
    --gt_override "flow-data/generated/gen_fnons_re${RE}_kf_1024to256_20seq.npy"
    --train_seqs_override 0,1,2,3,4,5,6,7
    --stats "base_results/regime_stats_re${RE}_measured_train.npz" --scales_re 1000
    --grid_factor 4 --base_ddim_init --sampler ddim --policy_ddim_steps 50 --eta 1.0
    --chain_starts 100 75 --pde_weight 0.2 --pure_gate --anchor_weight 3.0
    --dose_weight 3.0 --dose_lowmid_weight 2.0 --dose_low_weight 2.0
    --sampling_temp "$TEMP" --kl_coef 0.01 --policy_ema 0.99 --clip_eps 0.2 --seed 0)
  echo "=== STRONGANCHOR Re=$RE SMOKE $(date -u +%H:%M:%S) ==="
  $PY src/ddpo_ft/train_claude.py --smoke "${FLAGS[@]}" \
    --save_dir "monitoring/ddpo_re${RE}_stronganchor_smoke" \
    || { echo "=== STRONGANCHOR Re=$RE SMOKE FAILED ==="; return 1; }
  rm -rf "monitoring/ddpo_re${RE}_stronganchor_smoke"
  echo "=== STRONGANCHOR Re=$RE FULL RUN $(date -u +%H:%M:%S) ==="
  $PY src/ddpo_ft/train_claude.py "${FLAGS[@]}" --n_outer 600 --lr 5e-5 \
    --save_dir "monitoring/ddpo_re${RE}_stronganchor_ckpts" \
    || { echo "=== STRONGANCHOR Re=$RE FULL RUN FAILED ==="; return 1; }
  echo "=== STRONGANCHOR Re=$RE DONE $(date -u +%H:%M:%S) ==="
}

run_one 8000 3.5 || exit 1
run_one 4000 3.0 || exit 1
echo "=== STRONGANCHOR ALL DONE ==="
