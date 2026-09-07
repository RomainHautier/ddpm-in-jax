#!/bin/bash
# Full evaluation program for the strong-anchor specialists (sa8k, sa4k), after the
# night training finishes: ladder grading + selection on validation, then for each
# model the complete transfer matrix - unguided K3 at all five regimes, v8 gated at
# all five regimes, and the chain-adapted downward configs ([160]@4000 for sa8k,
# [125]@2000 and [100]@1000 for both) with lam 0 and 3.
#   nohup bash scripts/stronganchor_evals.sh > monitoring/stronganchor_evals.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=~/venv-ddpm/bin/python
while kill -0 608242 2>/dev/null; do sleep 300; done

select_model () {  # select_model RE NAMEPREFIX -> sets SEL_NAME SEL_CKF
  local RE=$1 PFX=$2
  local DIR=monitoring/ddpo_re${RE}_stronganchor_ckpts
  if [ ! -f "$DIR/BEST_ITER" ]; then
    echo "=== ${PFX} GRADE $(date -u +%H:%M:%S) ==="
    GRADE_RE=$RE GRADE_CKDIR=$DIR GRADE_ITERS=99,199,299,399,499,599 \
      GRADE_BEST_OUT=$DIR/BEST_ITER \
      $PY src/ddpo_ft/grade_puregate_re1k.py || { echo "=== ${PFX} GRADE FAILED ==="; return 1; }
  fi
  local BI; BI=$(cat $DIR/BEST_ITER); printf -v BI4 '%04d' "$BI"
  SEL_CKF=$DIR/${PFX}_${BI4}_ema.pkl
  [ -f "$SEL_CKF" ] || $PY -c "
import pickle
ck = pickle.load(open('$DIR/ddpo_re1000_iter${BI4}.pkl','rb'))
pickle.dump({'params': ck['ema_params'], 'iter': ck['iter']}, open('$SEL_CKF','wb'))"
  SEL_NAME=${PFX}-${BI4}
  echo "=== ${PFX} SELECTED iter $BI ==="
}

eval_model () {  # eval_model NAME CKF CHAINSPEC...
  local NAME=$1 CKF=$2; shift 2
  for R in 1000 2000 4000 6000 8000; do
    echo "=== $NAME UNGUIDED Re=$R $(date -u +%H:%M:%S) ==="
    MO_RE=$R MO_LAMS=0 MO_PDE_W=0.2 MO_MODELS=$NAME MO_CKPT_FILE=$CKF \
      $PY -m src.ddpo_ft.matched_objective || { echo "=== $NAME UNGUIDED Re=$R FAILED ==="; return 1; }
  done
  echo "=== $NAME GATED ALL $(date -u +%H:%M:%S) ==="
  GATE_REGIMES=1000,2000,4000,6000,8000 GATE_MODELS=$NAME GATE_CKPTS="$NAME:$CKF" \
    GATE_OOD_SEQS=12,13,14,15,16,17,18,19 GATE_OOD_PER=10 \
    $PY src/ddpo_ft/v8_lowband.py || { echo "=== $NAME GATED FAILED ==="; return 1; }
  for SPEC in "$@"; do
    local R=${SPEC%%:*} ST=${SPEC##*:}
    echo "=== $NAME CHAIN Re=$R S=[$ST] $(date -u +%H:%M:%S) ==="
    MO_RE=$R MO_STARTS=$ST MO_LAMS=0,3 MO_PDE_W=0.2 MO_MODELS=$NAME MO_CKPT_FILE=$CKF \
      $PY -m src.ddpo_ft.matched_objective || { echo "=== $NAME CHAIN Re=$R FAILED ==="; return 1; }
  done
  echo "=== $NAME EVALS DONE ==="
}

select_model 8000 sa8k || exit 1
N8=$SEL_NAME; C8=$SEL_CKF
select_model 4000 sa4k || exit 1
N4=$SEL_NAME; C4=$SEL_CKF

eval_model $N8 $C8 4000:160 2000:125 1000:100 || exit 1
eval_model $N4 $C4 2000:125 1000:100 || exit 1
echo "=== STRONGANCHOR EVALS ALL DONE ==="
