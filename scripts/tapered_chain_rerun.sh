#!/bin/bash
# Tapered re-runs of the chain + dial cells (MO_TAPER=1, lam 3 only): pg8k at
# [160]@4000, [125]@2000, [100]@1000 and pg4k at [125]@2000, [100]@1000. Waits for
# the running strong-anchor eval block.
set -u
cd "$(dirname "$0")/.."
PY=~/venv-ddpm/bin/python
CK8=monitoring/ddpo_re8000_puregate_ckpts/pg8k_0499_ema.pkl
CK4=monitoring/ddpo_re4000_puregate_ckpts/pg4k_0299_ema.pkl
while pgrep -f "stronganchor_evals.sh" > /dev/null \
   || pgrep -f "src.ddpo_ft.matched_objective" > /dev/null \
   || pgrep -f "v8_lowband.py" > /dev/null; do sleep 60; done

run_chain () {  # run_chain NAME CKF RE STARTS
  echo "=== TAPCHAIN $1 Re=$3 S=[$4] $(date -u +%H:%M:%S) ==="
  MO_RE=$3 MO_STARTS=$4 MO_LAMS=3 MO_TAPER=1 MO_PDE_W=0.2 MO_MODELS=$1 MO_CKPT_FILE=$2 \
    $PY -m src.ddpo_ft.matched_objective || { echo "=== TAPCHAIN $1 Re=$3 FAILED ==="; exit 1; }
}
run_chain pg8k-0499 $CK8 4000 160
run_chain pg8k-0499 $CK8 2000 125
run_chain pg8k-0499 $CK8 1000 100
run_chain pg4k-0299 $CK4 2000 125
run_chain pg4k-0299 $CK4 1000 100
echo "=== TAPCHAIN ALL DONE ==="
