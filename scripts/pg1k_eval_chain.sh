#!/bin/bash
# pg1k-0599 (pure-gate Re=1000 fine-tune, EMA weights of iter 599) evaluation chain:
#   1. ONE test audit at Re=1000 through the v8 gate machinery (unguided + gated rows)
#   2. OOD transfer sweep at 2000/4000/6000/8000 (unguided + v7 gated dial, canonical
#      generated-regime test pool seqs 12-19 x 10)
#   3. resume the interrupted overnight queue (Re=8000 leg, Re=2000 leg, 4k lam sweep)
#   nohup bash scripts/pg1k_eval_chain.sh > monitoring/pg1k_eval.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
CK="pg1k-0599:monitoring/ddpo_re1000_puregate_ckpts/pg1k_0599_ema.pkl"

# ONE gate everywhere: v8 low-band, at home and across the ladder. base0 rides along
# in the OOD runs so the upward-transfer comparison rows carry the same gate too.
while pgrep -f "v8_lowband.py" > /dev/null; do
  echo "waiting for the running home audit... $(date -u +%H:%M:%S)"; sleep 60
done

echo "=== PG1K OOD SWEEP (v8 gate) $(date -u +%H:%M:%S) ==="
# Re=4000 dropped for now: its raw dataset was lost in the flow-data wipe and has no
# GCS copy (only the saved test-pool fields survive); 2000 rows are already in the store
GATE_REGIMES=6000,8000 GATE_MODELS=pg1k-0599,base0 GATE_CKPTS="$CK" \
  GATE_OOD_SEQS=12,13,14,15,16,17,18,19 GATE_OOD_PER=10 \
  ~/venv-ddpm/bin/python src/ddpo_ft/v8_lowband.py \
  || { echo "=== PG1K OOD SWEEP FAILED ==="; exit 1; }
echo "=== PG1K EVAL DONE $(date -u +%H:%M:%S) ==="

bash scripts/finish_overnight.sh
