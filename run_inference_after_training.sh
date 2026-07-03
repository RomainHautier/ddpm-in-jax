#!/usr/bin/env bash
# Wait for the field-cond training queue to fully finish (TPU free + last checkpoint present),
# then run learned-guidance sequence inference over all conditional models.
set -u
cd /home/rhautier/ddpm-jax
LAST_CKPT="gs://ddpm-thesis-rh/checkpoints/ddpm/conditioned_field_cond_60ep/ckpt_epoch_0059.pkl"

echo "[wait $(date -u +%H:%M:%S)] waiting for training to finish (no train_ddpm proc + $LAST_CKPT) ..."
while pgrep -f "src.train_ddpm" >/dev/null 2>&1; do sleep 30; done
echo "[wait $(date -u +%H:%M:%S)] no train_ddpm process running."
until gcloud storage ls "$LAST_CKPT" >/dev/null 2>&1; do
  echo "[wait $(date -u +%H:%M:%S)] $LAST_CKPT not present yet..."; sleep 30
done
echo "[wait $(date -u +%H:%M:%S)] training complete; TPU free -> launching inference."
python3 run_learned_inference_queue.py
echo "[wait $(date -u +%H:%M:%S)] INFERENCE QUEUE COMPLETE (exit=$?)."
