#!/bin/bash
# Strength sweep of the gated dial on the PLAIN BASE at Re=4000: does raising lambda
# close the fine-band plateau (tolerance-limited) or does it saturate (capability
# limit, as at Re=8000)? lambda=3 is the published base0|v7bandgate row; 6/12/25 land
# in isolated v7s{lam} tags. Canonical generated-regime test pool: seqs 12-19, 10/seq.
#
# Waits for the pure-gate training driver to finish before touching the TPU.
#   nohup bash scripts/gate4k_strength_sweep.sh > monitoring/gate4k_sweep.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

while pgrep -f "overnight_puregate.sh" > /dev/null || pgrep -f "[t]rain_claude.py" > /dev/null; do
  echo "waiting for training to finish... $(date -u +%H:%M:%S)"
  sleep 300
done

for LAM in 6 12 25; do
  echo "=== GATE4K lam=$LAM $(date -u +%H:%M:%S) ==="
  GATE_LAM=$LAM GATE_REGIMES=4000 GATE_MODELS=base0 \
  GATE_OOD_SEQS=12,13,14,15,16,17,18,19 GATE_OOD_PER=10 \
  ~/venv-ddpm/bin/python src/ddpo_ft/v7_strength.py \
    || { echo "=== GATE4K lam=$LAM FAILED ==="; exit 1; }
done
echo "=== GATE4K SWEEP DONE ==="
