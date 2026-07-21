# FROZEN CONFIRMATORY PROTOCOL — v1.0 (2026-07-23)

Purpose: one blind, non-adaptive run that converts the campaign's adaptively-developed numbers
into validated ones. Everything below is FROZEN before execution. Any deviation, rerun, or
post-hoc adjustment voids the confirmation and must be reported as such.

## 0. Status of this document
- Written BEFORE execution. The confirmatory data (Re=10000 sequences 25-39) has never been
  opened by any analysis, training, or eval in this project.
- One attempt. Results are reported whatever they are, including placement.

## 1. Data
- CONFIRMATORY TEST SET: `flow-data/kf_re10000_256_40seed.npy`, sequences 25-39 (15 seqs),
  6 frames/seq = 90 frames, grid-4x degradation. UNTOUCHED to date.
- Target-regime data usable during the run: LR (4x-subsampled) fields of TRAINING sequences
  20-23 only (the existing pool). Probe seq 24 (monitoring only; NOT in the test set).
- Reference regimes (full data allowed): Re=500, Re=1000.
- SECONDARY confirmATORY EVAL (eval-only): in-dist Re=1000 flagship re-scored on sequences
  37-39 ONLY (probe seq 36 is excluded — it overlaps the historical test set; known flaw).

## 2. Frozen calibrations (existing artifacts, already built blind)
- Spectral anchor: `base_results/regime_stats_re10000_blind.npz`
  (spectrum-model fit on Re=500+1000; k_d extrapolated with the MEASURED exponent q=0.526;
  UP-direction leave-one-regime-out bias correction; no grid cap; no tail generosity;
  enstrophy held at the Re=1000 value; quantiles by universal-shape scaling.)
- PDE floor: residual_ref = 5183 (contained in the npz; Re-scaling exponent b=2.705 measured
  on Re=500->1000 with proper consecutive-frame triplets; over-prediction is the safe direction
  for the one-sided hinge).
- NOTE: the pde hinge remains ONE-SIDED (the two-sided variant is unvalidated at hot rollouts —
  noise-saturated; see backlog 2026-07-21).

## 3. Frozen blind decision rules (previously made with GT; now fixed a priori)
- R3.1 TRAINING TEMPERATURE from the blind deficit:
    D = E_deployed_base[32,96) / E_anchor[32,96), where E_deployed_base is the spectrum of the
    base model's deep-cascade output on TRAINING-pool inputs (GT-free).
    Rule (piecewise, from the three calibration regimes):  D > 0.60 -> temp 2.0;
    0.15 < D <= 0.60 -> temp 2.5;  D <= 0.15 -> temp 3.5.
- R3.2 CHECKPOINT SELECTION (blind stopping): train n_outer = 600, save every 50. Select the
    checkpoint minimizing |E_deployed[10,96)/E_anchor[10,96) - 1|, deployed spectra computed on
    TRAINING-pool inputs. KNOWN LIMITATION, accepted a priori: the anchor grades ~1.18x GT, so
    this rule will select a checkpoint that overshoots true GT by roughly that margin.
- R3.3 PATCHWORK CROSSOVER: k_c = the smallest k at which the DDPO deployed spectrum exceeds
    the recon spectrum for 3 consecutive shells (both GT-free); fallback k_c = 8 if no crossing
    below k=16. Smooth 2-shell tanh crossover, as implemented.
- R3.4 INFERENCE TEMPERATURE: 0.30 (frozen constant).

## 4. Frozen training recipe (identical to the validated stack)
    BASE_CKPT=/tmp/ema_ckpts/ema_base_0299.pkl (EMA base, mu=0.9999)
    --re 10000 --stats base_results/regime_stats_re10000_blind.npz --scales_re 1000
    --grid_factor 4 --align_weight 2.0 --base_ddim_init --highk_lo 10
    --sampler ddim --policy_ddim_steps 50 --eta 1.0 --chain_starts 100 75
    --policy_ema 0.99 --lr 5e-5 --seed 0 --n_outer 600 --eval_every 10 --save_every 50
    --sampling_temp <R3.1>
  Fresh run from the base (no resume from campaign checkpoints — they embed adaptive history).

## 5. Frozen evaluation
- Deep eval: eval_guided_full, chain_starts 150,100,50, policy_ddim_steps 86, eta 1.0, lam 3,
  ddim_stages, grid_factor 4, val "20,21,22,23", test "25,...,39", n_per_seq 6.
  (val field is required by the tool; only TEST rows are reported.)
- Inference at temp 0.30 (R3.4) and the default 1.0 (both reported; 0.30 is the headline per R3.4).
- Patchwork applied with k_c from R3.3; hybrid metrics reported alongside.
- REPORTED: ret, |ret-1|, residual, MSE, placement, k*, low-k band [1,5) — for base / DDPO /
  hybrid. Placement is reported EVEN THOUGH the campaign predicts it will be poor (~0.6-0.7 and
  possibly below base): that prediction is part of what is being tested.

## 6. Success criteria (stated a priori)
- PRIMARY: DDPO (or hybrid) hi-k retention within [0.75, 1.30] on the untouched test set
  (i.e., the adaptive campaign's magnitude of gain survives non-adaptive replication;
  base expectation ~0.18).
- SECONDARY: hybrid low-k [1,5) >= 0.97; k* >= 55.
- The placement information-ceiling claim is CONFIRMED if placement <= base's, as predicted.
- Training-seed noise is +-0.04: no interpretation of margins below 0.08.

## 7. Execution checklist
1. Verify seqs 25-39 absent from every config/log used to date (grep).
2. Compute D (R3.1) -> record temp before training starts.
3. Train once. No probe-based interventions permitted (probe recorded, not acted on).
4. Select checkpoint by R3.2. Record the selection computation.
5. Eval per §5. Publish numbers unedited, with this document's version stamped.
