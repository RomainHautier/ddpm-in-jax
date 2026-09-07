#!/bin/bash
# After the running Re=8000 leg: pg1k-0599 UNGUIDED rows (tag mop0.2_0, lam=0) at
# 1000/2000/6000/8000, then the Re=2000 pure-gate training leg.
set -u
cd "$(dirname "$0")/.."
while pgrep -f "[t]rain_claude.py" > /dev/null; do sleep 120; done
CKF=monitoring/ddpo_re1000_puregate_ckpts/pg1k_0599_ema.pkl
for R in 1000 2000 6000 8000; do
  echo "=== PG1K UNGUIDED Re=$R $(date -u +%H:%M:%S) ==="
  MO_RE=$R MO_LAMS=0 MO_PDE_W=0.2 MO_MODELS=pg1k-0599 MO_CKPT_FILE=$CKF \
    ~/venv-ddpm/bin/python -m src.ddpo_ft.matched_objective \
    || { echo "=== PG1K UNGUIDED Re=$R FAILED ==="; exit 1; }
done
echo "=== PG1K UNGUIDED DONE ==="
bash scripts/overnight_puregate.sh 2000
