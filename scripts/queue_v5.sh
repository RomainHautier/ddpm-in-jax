#!/bin/bash
# Queue v5: COMPLETE 4k/8k transfer program before any training.
#  1. pg4k unguided at all five regimes (guarded; some already in store)
#  2. pg4k gated (v8) at all five regimes
#  3. pg8k unguided at 6000/4000/2000/1000 (the missing downward unguided rows)
#  4. chain-adapted downward transfers, published ladder configs, lam 0 and 3:
#     pg8k at [160]@4000, [125]@2000, [100]@1000; pg4k at [125]@2000, [100]@1000
#  5. Re=2000 pure-gate training, then the Re=4000 lambda sweep
set -u
cd "$(dirname "$0")/.."
PY=~/venv-ddpm/bin/python
BI=$(cat monitoring/ddpo_re8000_puregate_ckpts/BEST_ITER); printf -v BI4 '%04d' "$BI"
CKF8=monitoring/ddpo_re8000_puregate_ckpts/pg8k_${BI4}_ema.pkl
BJ=$(cat monitoring/ddpo_re4000_puregate_ckpts/BEST_ITER); printf -v BJ4 '%04d' "$BJ"
CKF4=monitoring/ddpo_re4000_puregate_ckpts/pg4k_${BJ4}_ema.pkl

has_row () {
  JAX_PLATFORMS=cpu $PY - "$1" "$2" "$3" <<'PYEOF'
import sys, numpy as np, os
R, m, tag = sys.argv[1], sys.argv[2], sys.argv[3]
p = 'base_results/re1000_audit.npz' if R == '1000' else f'base_results/regime_audit_re{R}.npz'
S = np.load(p, allow_pickle=True) if os.path.exists(p) else None
sys.exit(0 if S is not None and f'{R}|{m}|K3|{tag}||ret' in S.files else 1)
PYEOF
}

while pgrep -f "grade_puregate_re1k.py" > /dev/null \
   || pgrep -f "src.ddpo_ft.matched_objective" > /dev/null \
   || pgrep -f "v8_lowband.py" > /dev/null; do sleep 60; done

for R in 4000 1000 2000 6000 8000; do
  if has_row $R pg4k-$BJ4 mop0.2_0; then echo "=== PG4K UNGUIDED Re=$R already ==="; continue; fi
  echo "=== PG4K UNGUIDED Re=$R $(date -u +%H:%M:%S) ==="
  MO_RE=$R MO_LAMS=0 MO_PDE_W=0.2 MO_MODELS=pg4k-$BJ4 MO_CKPT_FILE=$CKF4 \
    $PY -m src.ddpo_ft.matched_objective || { echo "=== PG4K UNGUIDED Re=$R FAILED ==="; exit 1; }
done
echo "=== PG4K UNGUIDED DONE ==="

echo "=== PG4K GATED ALL $(date -u +%H:%M:%S) ==="
GATE_REGIMES=4000,1000,2000,6000,8000 GATE_MODELS=pg4k-$BJ4 GATE_CKPTS="pg4k-$BJ4:$CKF4" \
  GATE_OOD_SEQS=12,13,14,15,16,17,18,19 GATE_OOD_PER=10 \
  $PY src/ddpo_ft/v8_lowband.py || { echo "=== PG4K GATED FAILED ==="; exit 1; }
echo "=== PG4K EVAL DONE ==="

for R in 6000 4000 2000 1000; do
  if has_row $R pg8k-$BI4 mop0.2_0; then echo "=== PG8K UNGUIDED Re=$R already ==="; continue; fi
  echo "=== PG8K UNGUIDED Re=$R $(date -u +%H:%M:%S) ==="
  MO_RE=$R MO_LAMS=0 MO_PDE_W=0.2 MO_MODELS=pg8k-$BI4 MO_CKPT_FILE=$CKF8 \
    $PY -m src.ddpo_ft.matched_objective || { echo "=== PG8K UNGUIDED Re=$R FAILED ==="; exit 1; }
done
echo "=== PG8K UNGUIDED DONE ==="

chain_run () {  # chain_run MODEL CKPTFILE RE STARTS
  echo "=== CHAIN $1 Re=$3 S=[$4] $(date -u +%H:%M:%S) ==="
  MO_RE=$3 MO_STARTS=$4 MO_LAMS=0,3 MO_PDE_W=0.2 MO_MODELS=$1 MO_CKPT_FILE=$2 \
    $PY -m src.ddpo_ft.matched_objective || { echo "=== CHAIN $1 Re=$3 FAILED ==="; exit 1; }
}
chain_run pg8k-$BI4 $CKF8 4000 160
chain_run pg8k-$BI4 $CKF8 2000 125
chain_run pg8k-$BI4 $CKF8 1000 100
chain_run pg4k-$BJ4 $CKF4 2000 125
chain_run pg4k-$BJ4 $CKF4 1000 100
echo "=== CHAIN TRANSFERS DONE ==="

bash scripts/overnight_puregate.sh 2000 || exit 1
for LAM in 6 12 25; do
  echo "=== GATE4K lam=$LAM $(date -u +%H:%M:%S) ==="
  GATE_LAM=$LAM GATE_REGIMES=4000 GATE_MODELS=base0 \
  GATE_OOD_SEQS=12,13,14,15,16,17,18,19 GATE_OOD_PER=10 \
  $PY src/ddpo_ft/v7_strength.py || { echo "=== GATE4K lam=$LAM FAILED ==="; exit 1; }
done
echo "=== QUEUE V5 ALL DONE ==="
