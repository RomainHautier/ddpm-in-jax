#!/bin/bash
# Re=3000 rows for the pure-gate specialists (pg8k, pg4k): K3 unguided, v8 gate, and
# the published [150] chain (lam 0 and 3). Then render the four-regime chain-transfer
# figures (1000/3000/4000/8000) for both models and sync.
set -u
cd "$(dirname "$0")/.."
PY=~/venv-ddpm/bin/python
CK8=monitoring/ddpo_re8000_puregate_ckpts/pg8k_0499_ema.pkl
CK4=monitoring/ddpo_re4000_puregate_ckpts/pg4k_0299_ema.pkl
while pgrep -f "stronganchor_evals.sh" > /dev/null \
   || pgrep -f "src.ddpo_ft.matched_objective" > /dev/null \
   || pgrep -f "v8_lowband.py" > /dev/null; do sleep 60; done

for SPEC in "pg8k-0499:$CK8" "pg4k-0299:$CK4"; do
  NAME=${SPEC%%:*}; CKF=${SPEC##*:}
  echo "=== $NAME RE3000 K3 $(date -u +%H:%M:%S) ==="
  MO_RE=3000 MO_LAMS=0 MO_PDE_W=0.2 MO_MODELS=$NAME MO_CKPT_FILE=$CKF \
    $PY -m src.ddpo_ft.matched_objective || { echo "=== $NAME RE3000 K3 FAILED ==="; exit 1; }
  echo "=== $NAME RE3000 CHAIN150 $(date -u +%H:%M:%S) ==="
  MO_RE=3000 MO_STARTS=150 MO_LAMS=0,3 MO_PDE_W=0.2 MO_MODELS=$NAME MO_CKPT_FILE=$CKF \
    $PY -m src.ddpo_ft.matched_objective || { echo "=== $NAME RE3000 CHAIN FAILED ==="; exit 1; }
done
echo "=== RE3000 GATE $(date -u +%H:%M:%S) ==="
GATE_REGIMES=3000 GATE_MODELS=pg8k-0499,pg4k-0299 \
  GATE_CKPTS="pg8k-0499:$CK8,pg4k-0299:$CK4" \
  GATE_OOD_SEQS=12,13,14,15,16,17,18,19 GATE_OOD_PER=10 \
  $PY src/ddpo_ft/v8_lowband.py || { echo "=== RE3000 GATE FAILED ==="; exit 1; }
echo "=== RE3000 EVALS DONE ==="

JAX_PLATFORMS=cpu $PY plot_scripts/regime_spectra_dial.py --preset pg8kxfer --ratio \
  --regimes 1000,3000,4000,8000 --ncol 4 --tag chains
JAX_PLATFORMS=cpu $PY plot_scripts/regime_spectra_dial.py --preset pg4kxfer --ratio \
  --regimes 1000,3000,4000,8000 --ncol 4 --tag chains
mv -f docs/figs_overleaf/regime_ratio_dial_pg8kxfer_chains.pdf \
      docs/figs_overleaf/regime_ratio_dial_pg4kxfer_chains.pdf docs/figs_overleaf/gated/
gsutil -m -q cp docs/figs_overleaf/gated/regime_ratio_dial_pg8kxfer_chains.pdf \
  docs/figs_overleaf/gated/regime_ratio_dial_pg4kxfer_chains.pdf \
  gs://ddpm-thesis-rh/docs/figs_overleaf/gated/
echo "=== RE3000 FIGURES SYNCED ==="
