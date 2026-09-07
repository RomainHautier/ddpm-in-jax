#!/bin/bash
# v8-gate rows for the LEGACY specialists at home (r4kp02@4000, r8kp02@8000) so the
# guided panel of the reward-generation comparison carries ONE gate everywhere.
# Waits for the strong-anchor eval pipeline to finish.
set -u
cd "$(dirname "$0")/.."
PY=~/venv-ddpm/bin/python
while pgrep -f "stronganchor_evals.sh" > /dev/null \
   || pgrep -f "src.ddpo_ft.matched_objective" > /dev/null \
   || pgrep -f "v8_lowband.py" > /dev/null; do sleep 60; done

echo "=== LEGACY V8 HOME $(date -u +%H:%M:%S) ==="
GATE_REGIMES=4000 GATE_MODELS=r4kp02-0599 \
  GATE_CKPTS="r4kp02-0599:monitoring/ddpo_re4000_pdew02_ckpts/ddpo_re1000_iter0599.pkl" \
  GATE_OOD_SEQS=12,13,14,15,16,17,18,19 GATE_OOD_PER=10 \
  $PY src/ddpo_ft/v8_lowband.py || { echo "=== LEGACY V8 4000 FAILED ==="; exit 1; }
GATE_REGIMES=8000 GATE_MODELS=r8kp02-0599 \
  GATE_CKPTS="r8kp02-0599:monitoring/ddpo_re8000_pdew02_ckpts/ddpo_re1000_iter0599.pkl" \
  GATE_OOD_SEQS=12,13,14,15,16,17,18,19 GATE_OOD_PER=10 \
  $PY src/ddpo_ft/v8_lowband.py || { echo "=== LEGACY V8 8000 FAILED ==="; exit 1; }
echo "=== LEGACY V8 DONE ==="
JAX_PLATFORMS=cpu $PY plot_scripts/anchorcmp_2row.py
gsutil -q cp docs/figs_overleaf/gated/regime_ratio_anchorcmp_2row.pdf gs://ddpm-thesis-rh/docs/figs_overleaf/gated/
echo "=== ANCHORCMP FIGURE SYNCED ==="
