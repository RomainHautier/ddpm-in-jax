#!/usr/bin/env bash
# Wait until all 4 conditional models have their 8 reconstruction pkls AND the inference
# process has exited, then execute the comparison notebook (CPU-only, no TPU contention).
set -u
cd /home/rhautier/ddpm-jax
TAGS="grad_frozen60 grad_full60 field_frozen60 field_full60"
REC=monitoring/sequence_reconstructions
NB=base_results/learned_finetune_comparison.ipynb
JUP=/home/rhautier/venv-ddpm/bin/jupyter

echo "[analysis $(date -u +%H:%M:%S)] waiting for 32 pkls + inference to finish ..."
while true; do
  ready=1
  for t in $TAGS; do
    n=$(ls $REC/sequence_reconstruction_indist_re1000_${t}_seq3*.pkl 2>/dev/null | wc -l)
    [ "$n" -lt 8 ] && ready=0
  done
  if [ "$ready" -eq 1 ] && ! pgrep -f run_learned_inference_queue >/dev/null 2>&1; then break; fi
  sleep 30
done

echo "[analysis $(date -u +%H:%M:%S)] all 32 pkls present; executing notebook ..."
JAX_PLATFORMS=cpu "$JUP" nbconvert --to notebook --execute --inplace "$NB" \
  --ExecutePreprocessor.timeout=1800 2>&1
echo "[analysis $(date -u +%H:%M:%S)] NOTEBOOK DONE exit=$?"
