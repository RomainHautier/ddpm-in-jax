#!/bin/bash
# VALIDATION selection of the downward transfer configuration for the gate-reward
# specialists, dial included in the candidate set. For each model and rung, the
# published chain runs on the VALIDATION pool with lam 0 and 3 under the isolated
# 'vmo' tag family (test rows untouched); selection is argmin |1 - retention| per the
# chain ladder protocol, and the winner's existing TEST row is the one test shot.
#   nohup bash scripts/downxfer_valselect.sh > monitoring/downxfer_valselect.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=~/venv-ddpm/bin/python
CK8=monitoring/ddpo_re8000_puregate_ckpts/pg8k_0499_ema.pkl
CK4=monitoring/ddpo_re4000_puregate_ckpts/pg4k_0299_ema.pkl

run_val () {  # run_val NAME CKF RE STARTS
  local R=$3
  local SEQS PER
  if [ "$R" = 1000 ]; then SEQS=28,29,30,31,32,33; PER=20; else SEQS=8,9,10,11; PER=10; fi
  echo "=== VALSEL $1 Re=$R S=[$4] $(date -u +%H:%M:%S) ==="
  MO_RE=$R MO_STARTS=$4 MO_LAMS=0,3 MO_TAG=vmo MO_PDE_W=0.2 \
    MO_SEQS=$SEQS MO_PER=$PER MO_MODELS=$1 MO_CKPT_FILE=$2 \
    $PY -m src.ddpo_ft.matched_objective || { echo "=== VALSEL $1 Re=$R FAILED ==="; exit 1; }
}
run_val pg8k-0499 $CK8 4000 160
run_val pg8k-0499 $CK8 2000 125
run_val pg8k-0499 $CK8 1000 100
run_val pg4k-0299 $CK4 2000 125
run_val pg4k-0299 $CK4 1000 100
echo "=== VALSEL RUNS DONE ==="

JAX_PLATFORMS=cpu $PY - <<'PYEOF'
import numpy as np
print('validation selection (argmin |1-ret|), vmo tags:')
for m, rungs in (('pg8k-0499', [(4000,'K160'),(2000,'K125'),(1000,'K100')]),
                 ('pg4k-0299', [(2000,'K125'),(1000,'K100')])):
    for R, ch in rungs:
        p = 'base_results/re1000_audit.npz' if R == 1000 else f'base_results/regime_audit_re{R}.npz'
        S = np.load(p, allow_pickle=True)
        best, bv = None, 9e9
        for lam, lab in (('0','chain'), ('3','chain + dial')):
            pre = f'{R}|{m}|{ch}|vmop0.2_{lam}||'
            if pre+'ret' not in S.files: print(R, m, lam, 'MISSING'); continue
            ret = float(np.asarray(S[pre+'ret']))
            r = np.asarray(S[pre+'ps_ret_paired'])
            inb = np.mean(np.abs(r-1)<0.2)*100
            print(f'  {m} Re={R} {lab:<13} val ret {ret:.3f}  |1-ret| {abs(1-ret):.3f}  in-band {inb:3.0f}%')
            if abs(1-ret) < bv: bv, best = abs(1-ret), lab
        print(f'  -> SELECTED at Re={R}: {best}')
PYEOF
echo "=== VALSEL ALL DONE ==="
