#!/bin/bash
# Queue v3: v2 + the pg4k full cross-regime sweep (guided and unguided, all five
# regimes) inserted behind the pg8k block, before the Re=2000 training leg.
# Every stage guards on existing store rows / artifacts so reruns are cheap.
set -u
cd "$(dirname "$0")/.."
CK1="pg1k-0599:monitoring/ddpo_re1000_puregate_ckpts/pg1k_0599_ema.pkl"
CKF1=monitoring/ddpo_re1000_puregate_ckpts/pg1k_0599_ema.pkl
PY=~/venv-ddpm/bin/python

has_row () {
  JAX_PLATFORMS=cpu $PY - "$1" "$2" "$3" <<'PYEOF'
import sys, numpy as np, os
R, m, tag = sys.argv[1], sys.argv[2], sys.argv[3]
p = 'base_results/re1000_audit.npz' if R == '1000' else f'base_results/regime_audit_re{R}.npz'
S = np.load(p, allow_pickle=True) if os.path.exists(p) else None
sys.exit(0 if S is not None and f'{R}|{m}|K3|{tag}||ret' in S.files else 1)
PYEOF
}

while pgrep -f "matched_objective|v8_lowband|grade_puregate|[t]rain_claude" > /dev/null; do sleep 60; done

for R in 1000 2000 4000 6000 8000; do
  if has_row $R pg1k-0599 mop0.2_0; then echo "=== PG1K UNGUIDED Re=$R already in store ==="; continue; fi
  echo "=== PG1K UNGUIDED Re=$R $(date -u +%H:%M:%S) ==="
  MO_RE=$R MO_LAMS=0 MO_PDE_W=0.2 MO_MODELS=pg1k-0599 MO_CKPT_FILE=$CKF1 \
    $PY -m src.ddpo_ft.matched_objective \
    || { echo "=== PG1K UNGUIDED Re=$R FAILED ==="; exit 1; }
done
echo "=== PG1K UNGUIDED DONE ==="

if ! has_row 4000 pg1k-0599 v8lowband; then
  echo "=== RE4K GATE EVAL $(date -u +%H:%M:%S) ==="
  GATE_REGIMES=4000 GATE_MODELS=pg1k-0599,base0 GATE_CKPTS="$CK1" \
    GATE_OOD_SEQS=12,13,14,15,16,17,18,19 GATE_OOD_PER=10 \
    $PY src/ddpo_ft/v8_lowband.py || { echo "=== RE4K GATE EVAL FAILED ==="; exit 1; }
fi
echo "=== RE4K GATE EVAL DONE ==="

# ---------------- pg8k ----------------
if [ ! -f monitoring/ddpo_re8000_puregate_ckpts/BEST_ITER ]; then
  echo "=== PG8K GRADE $(date -u +%H:%M:%S) ==="
  GRADE_RE=8000 GRADE_ITERS=199,299,399,499,599,699,799,859 \
    GRADE_BEST_OUT=monitoring/ddpo_re8000_puregate_ckpts/BEST_ITER \
    $PY src/ddpo_ft/grade_puregate_re1k.py || { echo "=== PG8K GRADE FAILED ==="; exit 1; }
fi
BI=$(cat monitoring/ddpo_re8000_puregate_ckpts/BEST_ITER); printf -v BI4 '%04d' "$BI"
CKF8=monitoring/ddpo_re8000_puregate_ckpts/pg8k_${BI4}_ema.pkl
[ -f "$CKF8" ] || $PY -c "
import pickle
ck = pickle.load(open('monitoring/ddpo_re8000_puregate_ckpts/ddpo_re1000_iter${BI4}.pkl','rb'))
pickle.dump({'params': ck['ema_params'], 'iter': ck['iter']}, open('$CKF8','wb'))"
echo "=== PG8K SELECTED iter $BI ==="

if ! has_row 8000 pg8k-$BI4 mop0.2_0; then
  echo "=== PG8K HOME UNGUIDED $(date -u +%H:%M:%S) ==="
  MO_RE=8000 MO_LAMS=0 MO_PDE_W=0.2 MO_MODELS=pg8k-$BI4 MO_CKPT_FILE=$CKF8 \
    $PY -m src.ddpo_ft.matched_objective || { echo "=== PG8K HOME UNGUIDED FAILED ==="; exit 1; }
fi
echo "=== PG8K GATED HOME+DOWN $(date -u +%H:%M:%S) ==="
GATE_REGIMES=8000,6000,4000,2000,1000 GATE_MODELS=pg8k-$BI4 GATE_CKPTS="pg8k-$BI4:$CKF8" \
  GATE_OOD_SEQS=12,13,14,15,16,17,18,19 GATE_OOD_PER=10 \
  $PY src/ddpo_ft/v8_lowband.py || { echo "=== PG8K GATED FAILED ==="; exit 1; }
echo "=== PG8K EVAL DONE ==="

# ---------------- pg4k ----------------
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
  if has_row $R pg4k-$BJ4 mop0.2_0; then echo "=== PG4K UNGUIDED Re=$R already in store ==="; continue; fi
  echo "=== PG4K UNGUIDED Re=$R $(date -u +%H:%M:%S) ==="
  MO_RE=$R MO_LAMS=0 MO_PDE_W=0.2 MO_MODELS=pg4k-$BJ4 MO_CKPT_FILE=$CKF4 \
    $PY -m src.ddpo_ft.matched_objective \
    || { echo "=== PG4K UNGUIDED Re=$R FAILED ==="; exit 1; }
done
echo "=== PG4K GATED ALL REGIMES $(date -u +%H:%M:%S) ==="
GATE_REGIMES=4000,1000,2000,6000,8000 GATE_MODELS=pg4k-$BJ4 GATE_CKPTS="pg4k-$BJ4:$CKF4" \
  GATE_OOD_SEQS=12,13,14,15,16,17,18,19 GATE_OOD_PER=10 \
  $PY src/ddpo_ft/v8_lowband.py || { echo "=== PG4K GATED FAILED ==="; exit 1; }
echo "=== PG4K EVAL DONE ==="

bash scripts/overnight_puregate.sh 2000 || exit 1

for LAM in 6 12 25; do
  echo "=== GATE4K lam=$LAM $(date -u +%H:%M:%S) ==="
  GATE_LAM=$LAM GATE_REGIMES=4000 GATE_MODELS=base0 \
  GATE_OOD_SEQS=12,13,14,15,16,17,18,19 GATE_OOD_PER=10 \
  $PY src/ddpo_ft/v7_strength.py || { echo "=== GATE4K lam=$LAM FAILED ==="; exit 1; }
done
echo "=== QUEUE ALL DONE ==="
