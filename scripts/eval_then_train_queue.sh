#!/bin/bash
# Consolidated queue, after the running Re=8000 pure-gate leg:
#  1. pg1k unguided rows at all five regimes
#  2. Re=4000 v8 gate eval (pg1k + base0)
#  3. pg8k ladder grading on Re=8000 validation -> best checkpoint -> EMA extraction
#  4. pg8k unguided rows at 8000 (home) and 6000/4000/2000/1000 (downward transfer)
#  5. pg8k + v8 gate at home (Re=8000)
#  6. Re=2000 pure-gate training leg
#  7. Re=4000 base lambda strength sweep
set -u
cd "$(dirname "$0")/.."
CK1="pg1k-0599:monitoring/ddpo_re1000_puregate_ckpts/pg1k_0599_ema.pkl"
CKF1=monitoring/ddpo_re1000_puregate_ckpts/pg1k_0599_ema.pkl
while pgrep -f "[t]rain_claude.py" > /dev/null; do sleep 120; done

for R in 1000 2000 4000 6000 8000; do
  echo "=== PG1K UNGUIDED Re=$R $(date -u +%H:%M:%S) ==="
  MO_RE=$R MO_LAMS=0 MO_PDE_W=0.2 MO_MODELS=pg1k-0599 MO_CKPT_FILE=$CKF1 \
    ~/venv-ddpm/bin/python -m src.ddpo_ft.matched_objective \
    || { echo "=== PG1K UNGUIDED Re=$R FAILED ==="; exit 1; }
done
echo "=== PG1K UNGUIDED DONE ==="

echo "=== RE4K GATE EVAL $(date -u +%H:%M:%S) ==="
GATE_REGIMES=4000 GATE_MODELS=pg1k-0599,base0 GATE_CKPTS="$CK1" \
  GATE_OOD_SEQS=12,13,14,15,16,17,18,19 GATE_OOD_PER=10 \
  ~/venv-ddpm/bin/python src/ddpo_ft/v8_lowband.py \
  || { echo "=== RE4K GATE EVAL FAILED ==="; exit 1; }
echo "=== RE4K GATE EVAL DONE ==="

echo "=== PG8K GRADE $(date -u +%H:%M:%S) ==="
GRADE_RE=8000 GRADE_ITERS=199,299,399,499,599,699,799,859 \
  GRADE_BEST_OUT=monitoring/ddpo_re8000_puregate_ckpts/BEST_ITER \
  ~/venv-ddpm/bin/python src/ddpo_ft/grade_puregate_re1k.py \
  || { echo "=== PG8K GRADE FAILED ==="; exit 1; }
BI=$(cat monitoring/ddpo_re8000_puregate_ckpts/BEST_ITER)
printf -v BI4 '%04d' "$BI"
~/venv-ddpm/bin/python -c "
import pickle
ck = pickle.load(open('monitoring/ddpo_re8000_puregate_ckpts/ddpo_re1000_iter${BI4}.pkl','rb'))
pickle.dump({'params': ck['ema_params'], 'iter': ck['iter']},
            open('monitoring/ddpo_re8000_puregate_ckpts/pg8k_${BI4}_ema.pkl','wb'))
print('pg8k EMA extracted, iter', ck['iter'])"
CKF8=monitoring/ddpo_re8000_puregate_ckpts/pg8k_${BI4}_ema.pkl
echo "=== PG8K SELECTED iter $BI ==="

for R in 8000 6000 4000 2000 1000; do
  echo "=== PG8K UNGUIDED Re=$R $(date -u +%H:%M:%S) ==="
  MO_RE=$R MO_LAMS=0 MO_PDE_W=0.2 MO_MODELS=pg8k-$BI4 MO_CKPT_FILE=$CKF8 \
    ~/venv-ddpm/bin/python -m src.ddpo_ft.matched_objective \
    || { echo "=== PG8K UNGUIDED Re=$R FAILED ==="; exit 1; }
done
echo "=== PG8K UNGUIDED DONE ==="

echo "=== PG8K HOME GATE $(date -u +%H:%M:%S) ==="
GATE_REGIMES=8000 GATE_MODELS=pg8k-$BI4 GATE_CKPTS="pg8k-$BI4:$CKF8" \
  GATE_OOD_SEQS=12,13,14,15,16,17,18,19 GATE_OOD_PER=10 \
  ~/venv-ddpm/bin/python src/ddpo_ft/v8_lowband.py \
  || { echo "=== PG8K HOME GATE FAILED ==="; exit 1; }
echo "=== PG8K EVAL DONE ==="

bash scripts/overnight_puregate.sh 2000 || exit 1

for LAM in 6 12 25; do
  echo "=== GATE4K lam=$LAM $(date -u +%H:%M:%S) ==="
  GATE_LAM=$LAM GATE_REGIMES=4000 GATE_MODELS=base0 \
  GATE_OOD_SEQS=12,13,14,15,16,17,18,19 GATE_OOD_PER=10 \
  ~/venv-ddpm/bin/python src/ddpo_ft/v7_strength.py \
    || { echo "=== GATE4K lam=$LAM FAILED ==="; exit 1; }
done
echo "=== QUEUE ALL DONE ==="
