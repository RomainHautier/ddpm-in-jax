#!/bin/bash
# PURE-GATE fine-tunes, three regimes sequentially: the v8 dial's own structure as the
# training reward — 0.5 * per-sample-scaled spec anchor [1,96) + gated dose terms
# ([7,16) w2, [16,32) w3, [32,96) w3, [1,5) w2 db0.05) + pde guard. NO ungated ensemble
# push (spec/spec_highk/energy off), unlike the layered gt1k-style recipe that the
# ddpo_re2000_gatedlow run carries (kept as the A/B ablation).
#
# Waits for any running train_claude to exit before starting (the layered Re=2000 leg).
#
#   nohup bash scripts/overnight_puregate.sh > monitoring/puregate_driver.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=~/venv-ddpm/bin/python

while pgrep -f "[t]rain_claude.py" > /dev/null; do
  echo "waiting for the running training leg to finish... $(date -u +%H:%M:%S)"
  sleep 120
done

run_regime () {
  local RE=$1 TEMP=$2 PDEW=$3
  # Re=1000 lives in the 40-seed dataset with the canonical 20-27 train split; the
  # generated regimes use their own pools with train 0-7
  local GT=flow-data/generated/gen_fnons_re${RE}_kf_1024to256_20seq.npy SEQS=0,1,2,3,4,5,6,7
  if [ "$RE" = 1000 ]; then
    GT=flow-data/kf_2d_re1000_256_40seed.npy; SEQS=20,21,22,23,24,25,26,27
  fi
  local STATS=base_results/regime_stats_re${RE}_measured_train.npz
  local DIR=monitoring/ddpo_re${RE}_puregate_ckpts
  local COMMON=(--re "$RE" --gt_override "$GT" --train_seqs_override "$SEQS"
    --stats "$STATS" --scales_re 1000 --grid_factor 4 --base_ddim_init
    --sampler ddim --policy_ddim_steps 50 --eta 1.0 --chain_starts 100 75
    --pde_weight "$PDEW" --pure_gate
    --dose_weight 3.0 --dose_lowmid_weight 2.0 --dose_low_weight 2.0
    --sampling_temp "$TEMP" --kl_coef 0.01 --policy_ema 0.99 --clip_eps 0.2 --seed 0)

  echo "=== PUREGATE Re=$RE SMOKE $(date -u +%H:%M:%S) ==="
  if ! $PY src/ddpo_ft/train_claude.py --smoke "${COMMON[@]}" \
       --save_dir "monitoring/ddpo_re${RE}_puregate_smoke"; then
    echo "=== PUREGATE Re=$RE SMOKE FAILED - aborting the sequence ==="
    return 1
  fi
  rm -rf "monitoring/ddpo_re${RE}_puregate_smoke"
  echo "=== PUREGATE Re=$RE FULL RUN $(date -u +%H:%M:%S) ==="
  $PY src/ddpo_ft/train_claude.py "${COMMON[@]}" \
      --n_outer 600 --lr 5e-5 --save_dir "$DIR" \
      || { echo "=== PUREGATE Re=$RE FULL RUN FAILED ==="; return 1; }
  echo "=== PUREGATE Re=$RE DONE $(date -u +%H:%M:%S) ==="
}

# regimes come from the command line (default: the full ladder, home first); recipe
# lookup keeps temp and pde weight tied to the regime
recipe () {
  case $1 in
    1000) echo "2.0 1.0";; 2000) echo "2.5 1.0";;
    4000) echo "3.0 0.2";; 8000) echo "3.5 0.2";;
    *) echo "unknown regime $1" >&2; return 1;;
  esac
}
for RE in "${@:-1000 2000 4000 8000}"; do
  read -r TEMP PDEW <<< "$(recipe "$RE")" || exit 1
  run_regime "$RE" "$TEMP" "$PDEW" || exit 1
done
echo "=== PUREGATE ALL DONE ==="
