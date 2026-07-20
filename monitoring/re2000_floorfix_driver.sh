#!/bin/bash
# Re=2000 with the CORRECTED PDE floor. The whole Re=2000 campaign ran with residual_ref=26.29
# while the blind Re-scaling estimate is 66.64 (measured 46-76) — i.e. the hinge was firing on
# physically-legitimate residual and the pde term leaned toward smoothness the entire time. The
# best model landed UNDER GT (ret 0.921, k*=84), consistent with that.
# Resumes the best checkpoint (temp2.5 @999) for +200 iters with the corrected floor, so the
# comparison is like-for-like against its own 0.921 / k*=84.
# Floor value is the BLIND estimate (66.64), not the measured one — keeps the run GT-free.
set -u; cd /home/rhautier/ddpm-jax; PY=/home/rhautier/venv-ddpm/bin/python; LOG=monitoring/ab_pdelocal
until grep -q "RE10000 COMPLETE" monitoring/re10000_driver.log 2>/dev/null; do sleep 60; done
sleep 20
echo "[floorfix $(date +%H:%M:%S)] Re=2000 +200 iters with residual_ref 26.29 -> 66.64"
JAX_PLATFORMS='' BASE_CKPT=/tmp/ema_ckpts/ema_base_0299.pkl $PY -m src.ddpo_ft.train_claude \
  --re 2000 --stats base_results/regime_stats_re2000_obsfit_gen12_floorfix.npz --scales_re 1000 \
  --grid_factor 4 --align_weight 2.0 --base_ddim_init --highk_lo 10 \
  --sampler ddim --policy_ddim_steps 50 --eta 1.0 --chain_starts 100 75 \
  --policy_ema 0.99 --sampling_temp 2.5 --lr 5e-5 \
  --resume monitoring/ddpo_re2000_gen12_temp25_long_ckpts/ddpo_re1000_iter0999.pkl \
  --n_outer 200 --eval_every 10 --save_every 50 \
  --save_dir monitoring/ddpo_re2000_floorfix_ckpts \
  > monitoring/ddpo_re2000_floorfix.log 2>&1
echo "[floorfix $(date +%H:%M:%S)] training done (exit $?) — deep eval iter1199 (CLEAN test seqs)"
JAX_PLATFORMS='' BASE_CKPT=/tmp/ema_ckpts/ema_base_0299.pkl $PY -m src.ddpo_ft.eval_guided_full \
  monitoring/ddpo_re2000_floorfix_ckpts/ddpo_re1000_iter1199.pkl \
  --re 2000 --base_ddim_init --sampler ddim --chain_starts 150,100,50 --policy_ddim_steps 86 \
  --eta 1.0 --ddim_stages --lam 3 --gt flow-data/kf_re2000_256_20seed.npy \
  --val 0,1,2,3,4,5,6,7 --test 12,13,14,15,16,17,18,19 --grid_factor 4 --n_per_seq 6 \
  > $LOG/eval_re2000_floorfix.log 2>&1
gsutil -m -q rsync -r monitoring/ddpo_re2000_floorfix_ckpts \
  gs://ddpm-thesis-rh/monitoring/ddpo_re2000_floorfix_ckpts 2>/dev/null
echo "[floorfix $(date +%H:%M:%S)] ===== FLOORFIX COMPLETE ====="
