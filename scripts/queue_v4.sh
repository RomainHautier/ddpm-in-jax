#!/bin/bash
# Queue v4: transfers first. After the running pg4k grading:
#  1. pg4k selection -> K3 evals (unguided + v8 gate, all five regimes)
#  2. CHAIN-ADAPTED downward transfers, published ladder configs, lam 0 and 3:
#     pg8k at [160]@4000, [125]@2000, [100]@1000; pg4k at [125]@2000, [100]@1000
#  3. Re=2000 pure-gate training, Re=4000 lambda sweep (deferred behind transfers)
set -u
cd "$(dirname "$0")/.."
PY=~/venv-ddpm/bin/python
BI=$(cat monitoring/ddpo_re8000_puregate_ckpts/BEST_ITER); printf -v BI4 '%04d' "$BI"
CKF8=monitoring/ddpo_re8000_puregate_ckpts/pg8k_${BI4}_ema.pkl

has_row () {
  JAX_PLATFORMS=cpu $PY - "$1" "$2" "$3" <<'PYEOF'
import sys, numpy as np, os
R, m, tag = sys.argv[1], sys.argv[2], sys.argv[3]
p = 'base_results/re1000_audit.npz' if R == '1000' else f'base_results/regime_audit_re{R}.npz'
S = np.load(p, allow_pickle=True) if os.path.exists(p) else None
sys.exit(0 if S is not None and any(f'{R}|{m}|' in k and f'|{tag}||ret' in k for k in (S.files if S is not None else [])) else 1)
PYEOF
}

while pgrep -f "grade_puregate_re1k.py" > /dev/null || pgrep -f "src.ddpo_ft.matched_objective" > /dev/null || pgrep -f "v8_lowband.py" > /dev/null; do sleep 60; done

# ---- pg4k selection + K3 evals ----
if [ ! -f monitoring/ddpo_re4000_puregate_ckpts/BEST_ITER ]; then
  echo "=== PG4K GRADE $(date -u +%H:%M:%S) ==="
  GRADE_RE=4000 GRADE_ITERS=99,199,299,399,499,599 \
    GRADE_BEST_OUT=monitoring/ddpo_re4000_puregate_ckpts/BEST_ITER \
    $PY src/ddpo_ft/grade_puregate_re1k.py || { echo "=== PG4K GRADE FAILED ==="; exit 1; }
fi
BJ=$(cat monitoring/ddpo_re4000_puregate_ckpts/BEST_ITER); printf -v BJ4 '%04d' "$BJ"
CKF4=monitoring/ddpo_re4000_puregate_ckpts/pg4k_${BJ4}_ema.pkl
[ -f "$CKF4" ] || $PY -c "
import pickle
ck = pickle.load(open('monitoring/ddpo_re4000_puregate_ckpts/ddpo_re1000_iter${BJ4}.pkl','rb'))
pickle.dump({'params': ck['ema_params'], 'iter': ck['iter']}, open('$CKF4','wb'))"
echo "=== PG4K SELECTED iter $BJ ==="

for R in 4000 1000 2000 6000 8000; do
  if has_row $R pg4k-$BJ4 mop0.2_0; then echo "=== PG4K UNGUIDED Re=$R already ==="; continue; fi
  echo "=== PG4K UNGUIDED Re=$R $(date -u +%H:%M:%S) ==="
  MO_RE=$R MO_LAMS=0 MO_PDE_W=0.2 MO_MODELS=pg4k-$BJ4 MO_CKPT_FILE=$CKF4 \
    $PY -m src.ddpo_ft.matched_objective || { echo "=== PG4K UNGUIDED Re=$R FAILED ==="; exit 1; }
done
echo "=== PG4K GATED ALL $(date -u +%H:%M:%S) ==="
GATE_REGIMES=4000,1000,2000,6000,8000 GATE_MODELS=pg4k-$BJ4 GATE_CKPTS="pg4k-$BJ4:$CKF4" \
  GATE_OOD_SEQS=12,13,14,15,16,17,18,19 GATE_OOD_PER=10 \
  $PY src/ddpo_ft/v8_lowband.py || { echo "=== PG4K GATED FAILED ==="; exit 1; }
echo "=== PG4K EVAL DONE ==="

# ---- chain-adapted downward transfers, published ladder configs ----
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
echo "=== QUEUE V4 ALL DONE ==="
