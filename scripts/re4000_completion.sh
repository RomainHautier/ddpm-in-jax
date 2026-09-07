#!/bin/bash
# Re=4000 unblocked: complete the pg1k grid at 4000 (v8 gate + unguided rows) and run
# the parked base-model lambda strength sweep. Waits for the running chain
# (8000 training -> pg1k unguided evals -> 2000 training) to finish first.
set -u
cd "$(dirname "$0")/.."
CK="pg1k-0599:monitoring/ddpo_re1000_puregate_ckpts/pg1k_0599_ema.pkl"
CKF=monitoring/ddpo_re1000_puregate_ckpts/pg1k_0599_ema.pkl
while pgrep -f "pg1k_unguided_then_2000.sh" > /dev/null || pgrep -f "[t]rain_claude.py" > /dev/null; do
  sleep 300
done

echo "=== RE4K GATE EVAL $(date -u +%H:%M:%S) ==="
GATE_REGIMES=4000 GATE_MODELS=pg1k-0599,base0 GATE_CKPTS="$CK" \
  GATE_OOD_SEQS=12,13,14,15,16,17,18,19 GATE_OOD_PER=10 \
  ~/venv-ddpm/bin/python src/ddpo_ft/v8_lowband.py \
  || { echo "=== RE4K GATE EVAL FAILED ==="; exit 1; }

echo "=== RE4K PG1K UNGUIDED $(date -u +%H:%M:%S) ==="
MO_RE=4000 MO_LAMS=0 MO_PDE_W=0.2 MO_MODELS=pg1k-0599 MO_CKPT_FILE=$CKF \
  ~/venv-ddpm/bin/python -m src.ddpo_ft.matched_objective \
  || { echo "=== RE4K PG1K UNGUIDED FAILED ==="; exit 1; }

for LAM in 6 12 25; do
  echo "=== GATE4K lam=$LAM $(date -u +%H:%M:%S) ==="
  GATE_LAM=$LAM GATE_REGIMES=4000 GATE_MODELS=base0 \
  GATE_OOD_SEQS=12,13,14,15,16,17,18,19 GATE_OOD_PER=10 \
  ~/venv-ddpm/bin/python src/ddpo_ft/v7_strength.py \
    || { echo "=== GATE4K lam=$LAM FAILED ==="; exit 1; }
done
echo "=== RE4K COMPLETION ALL DONE ==="
