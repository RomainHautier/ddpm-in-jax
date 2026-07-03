#!/usr/bin/env bash
# Sequential TPU queue: ENS field-conditioning at 60 epochs, both finetune scopes.
# Runs full-finetune first, then frozen. One JAX job at a time.
set -u
cd /home/rhautier/ddpm-jax

run () {
  local cfg="$1" log="$2" name="$3"
  echo "==== [$(date -u +%H:%M:%S)] START $name  ($cfg) ====" | tee -a "$log"
  python3 -m src.train_ddpm "$cfg" >> "$log" 2>&1
  local rc=$?
  echo "==== [$(date -u +%H:%M:%S)] END   $name  exit=$rc ====" | tee -a "$log"
  return $rc
}

run configs/config_field_full_finetune.yaml base_results/field_full_finetune_train.log "field-FULL-60ep"
run configs/config_field_cond.yaml          base_results/field_frozen_60ep_train.log   "field-FROZEN-60ep"

echo "==== [$(date -u +%H:%M:%S)] QUEUE DONE ===="
