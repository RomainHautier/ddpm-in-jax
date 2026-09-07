#!/bin/bash
# The MISSING candidate: adapted chain + the GATED per-sample dial (v8), for both
# specialists at their downward rungs. Validation runs first (isolated 'v8val' tag,
# GT-free selection), then test runs (canonical pools, default tag).
set -u
cd "$(dirname "$0")/.."
PY=~/venv-ddpm/bin/python
CK8="pg8k-0499:monitoring/ddpo_re8000_puregate_ckpts/pg8k_0499_ema.pkl"
CK4="pg4k-0299:monitoring/ddpo_re4000_puregate_ckpts/pg4k_0299_ema.pkl"

run () {  # run MODELSPEC RE STARTS POOL(val|test)
  local NAME=${1%%:*} R=$2 ST=$3 POOL=$4
  local SEQS PER TAG EXTRA
  if [ "$POOL" = val ]; then
    TAG=v8val
    if [ "$R" = 1000 ]; then SEQS=28,29,30,31,32,33; PER=20; else SEQS=8,9,10,11; PER=10; fi
  else
    TAG=v8lowband
    if [ "$R" = 1000 ]; then SEQS=34,35,36,37,38,39; PER=20; else SEQS=12,13,14,15,16,17,18,19; PER=10; fi
  fi
  echo "=== CHAINGATE $NAME Re=$R S=[$ST] $POOL $(date -u +%H:%M:%S) ==="
  GATE_STARTS=$ST GATE_TAG=$TAG GATE_REGIMES=$R GATE_MODELS=$NAME GATE_CKPTS="$1" \
    GATE_OOD_SEQS=$SEQS GATE_OOD_PER=$PER GATE_SEQS_ALL=1 \
    $PY src/ddpo_ft/v8_lowband.py || { echo "=== CHAINGATE $NAME Re=$R $POOL FAILED ==="; exit 1; }
}

for POOL in val test; do
  run "$CK8" 4000 160 $POOL
  run "$CK8" 2000 125 $POOL
  run "$CK8" 1000 100 $POOL
  run "$CK4" 2000 125 $POOL
  run "$CK4" 1000 100 $POOL
done
echo "=== CHAINGATE ALL DONE ==="
