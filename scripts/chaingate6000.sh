#!/bin/bash
# Re=6000 chain + gated dial candidates for pg8k: [150,100] and [160], validation
# (isolated v8val tag) then test, for the GT-free selection against K3 + gate.
set -u
cd "$(dirname "$0")/.."
PY=~/venv-ddpm/bin/python
CK8="pg8k-0499:monitoring/ddpo_re8000_puregate_ckpts/pg8k_0499_ema.pkl"
run () {
  local ST=$1 POOL=$2 SEQS PER TAG
  if [ "$POOL" = val ]; then TAG=v8val; SEQS=8,9,10,11; PER=10
  else TAG=v8lowband; SEQS=12,13,14,15,16,17,18,19; PER=10; fi
  echo "=== CG6K S=[$ST] $POOL $(date -u +%H:%M:%S) ==="
  GATE_STARTS=$ST GATE_TAG=$TAG GATE_REGIMES=6000 GATE_MODELS=pg8k-0499 GATE_CKPTS="$CK8" \
    GATE_OOD_SEQS=$SEQS GATE_OOD_PER=$PER GATE_SEQS_ALL=1 \
    $PY src/ddpo_ft/v8_lowband.py || { echo "=== CG6K S=[$ST] $POOL FAILED ==="; exit 1; }
}
run 150,100 val
run 160 val
run 150,100 test
run 160 test
echo "=== CG6K ALL DONE ==="
