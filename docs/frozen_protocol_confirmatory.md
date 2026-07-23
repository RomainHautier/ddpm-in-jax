# FROZEN CONFIRMATORY PROTOCOL — v1.4 (2026-07-23)

CHANGELOG v1.1: R3.1 temperature buckets corrected (v1.0 derived them from GT-relative deficits;
recomputed against the BLIND anchor-relative deficit they misassigned in-dist and Re=10000).
Appendix A added: frozen Re=1000 in-dist baseline. No confirmatory data opened under either version.
CHANGELOG v1.3 (post Appendix-B failure): §2b added — FREEZE PROCEDURES, NOT ARTIFACTS, plus a
mandatory GT-free anchor-freshness pre-flight. Root cause of the Appendix-B primary failure
(ret 1.454 vs [0.80,1.20]): B2 froze the obs-fit anchor as an ARTIFACT built from the OLD data
generation's LR; the new generation's tail is 1.66x weaker, so training optimized toward a target
1.26x hotter (in E[32,96)) than the deployment regime's true spectrum. Rebuilding the anchor with
the SAME frozen procedure from the NEW file's train-pool LR grades 0.997x vs the new GT
(report-only diagnostic on the already-opened test seqs) — the procedure was correct; pinning its
output was the error.
CHANGELOG v1.4 (post B9): temporal-compatibility pre-flight added (LR lag-1 corr in [0.95,0.9985];
the 2026-07 40-seed generation is ~80x finer in time and fails it); dt32 stride-80 derivative
files defined; Appendix C records the dt32 gate result and the Re=10000 dt32 launch.

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

## 2b. FREEZE PROCEDURES, NOT ARTIFACTS (v1.3 — binding for all future runs)
- Any reward artifact that depends on TARGET-REGIME OBSERVATIONS (the obs-fit spectral anchor)
  is frozen as a CONSTRUCTION PROCEDURE, not as a file. Before every training run it MUST be
  rebuilt from the LR observations of the exact data generation being trained on, using
  `src/ddpo_ft/anchor_obsfit_builder.py` (the validated recipe, parameterized; nothing else about
  the recipe may change). Artifacts that depend only on REFERENCE regimes (Re=500/1000 spectrum
  fits, transfer function, LOO correction, the floor Re-scaling law b=2.705) remain frozen files.
- MANDATORY GT-FREE PRE-FLIGHT before any training launch: `verify_freshness(anchor, data, seqs)`
  compares the anchor's stored LR observation fingerprint against the band spectrum (k in [10,30))
  of the training LR about to be used. |ratio - 1| > 0.05 -> STALE: abort and rebuild. An anchor
  npz without a fingerprint is stale by definition. (The Appendix-B failure would have been caught
  here: old-LR/new-LR band ratio = 1.117, measurable before training, no GT needed.)
- Data-file identity (CRC/hash) of every input file is recorded at launch; a changed identity
  triggers the pre-flight even if the path is unchanged.

## 3. Frozen blind decision rules (previously made with GT; now fixed a priori)
- R3.1 TRAINING TEMPERATURE from the blind deficit:
    D = E_deployed_base[32,96) / E_anchor[32,96), where E_deployed_base is the spectrum of the
    base model's deep-cascade output on TRAINING-pool inputs (GT-free).
    Rule (piecewise; CORRECTED v1.1 — v1.0's buckets were derived from GT-relative deficits and
    misassigned two of the three calibration regimes when D is computed blind):
        D >= 0.35        -> temp 2.0    (calibration: in-dist D=0.40, optimum <=2.0)
        0.20 <= D < 0.35 -> temp 2.5    (calibration: Re=2000  D=0.25-0.30, optimum 2.5)
        D < 0.20         -> temp 3.5    (calibration: Re=10000 D=0.156, temp 2.5 plateaued, 3.5 broke it)
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


## Appendix A — Frozen Re=1000 IN-DIST BASELINE (v1.1)

The baseline finetune, runnable before any extrapolation. In-dist, so the anchor and floor are
MEASURED (blindness constraints apply only to OOD targets); the decision rules are still frozen.

A1. Data split (FIXES the historical probe/test overlap):
    train_seqs = [32, 33]   |   probe_seq = 34  (moved OUT of the test set; monitoring only)
    val = 34, 35            |   TEST = 36, 37, 38, 39  (probe no longer overlaps: fully reportable)
    Base DDPM trained on seqs 0-31; all of 32-39 unseen by it.
A2. Calibrations:
    anchor = base_results/regime_stats_re1000.npz (measured in-dist statistics)
    residual_ref = 17.455 (calibration-json value). RECONCILED 2026-07-23: the json and the
    session's re-measurement use the SAME operator and units; the differences (json 17.46 vs 10.22
    at Re=1000; 45.99 vs 75.87 at Re=2000) are pure SAMPLING VARIANCE of a heavy-tailed quantity
    across sequence subsets — reproducing the json's exact seq/frame selection yields its values
    to the digit (45.99 at Re=2000). Consequence: the floor carries ~+-50% intrinsic sampling
    uncertainty at Re>=1000; err-high doctrine applies, and the blind Re-scaling prediction (66.6)
    lies WITHIN the Re=2000 floor's own sampling spread [46, 76] — i.e. indistinguishable from a
    direct measurement. 17.455 frozen (lenient side, campaign continuity).
A3. Temperature by R3.1 (corrected): D = 0.472/1.194 = 0.40 -> temp 2.0.
    (Independently consistent with the GT-informed finding: temps <=2.0 preserve k*=95.)
A4. Recipe: identical to §4 with --re 1000, no --stats/--scales_re overrides, temp 2.0,
    n_outer 600, save_every 50, seed 0, fresh run from the EMA base.
A5. Selection & eval: R3.2 checkpoint rule (deployed-vs-anchor on training-pool inputs);
    deep eval per §5 with val "34,35", test "36,37,38,39"; inference temp 0.30 and 1.0 reported;
    patchwork per R3.3.
A6. A-priori expectations: ret in [0.85, 1.15]; k* >= 90; placement >= 0.85. Anything outside
    is reported as-is.

## Appendix B — Frozen Re=2000 CONFIRMATORY on the new 40-seed data (v1.2, 2026-07-23)

New data: flow-data/kf_re2000_256_40seed.npy (40 seqs x 320). VERIFIED before writing this
appendix: numerically unrelated to the old 20-seed file (corr ~ 0 at matched indices; new seq 0
matches no old sequence) — every sequence is VIRGIN w.r.t. the entire campaign.

B1. Split (fixed a priori):
    train_seqs = [0,1,2,3] (LR observations only enter training) | probe_seq = 4 (record-only)
    TEST = seqs 24-39 (16 seqs, 6 frames/seq = 96 frames), grid-4x. SEALED until the single eval.
    SEALED SPARE = seqs 5-23: untouched, reserved for future seed-repeat confirmatories.
B2. Calibrations (existing frozen artifacts, unchanged):
    anchor+floor = base_results/regime_stats_re2000_obsfit_floorfix.npz
    (accurate obs-fit anchor ~1.0x tail, GT-free; residual_ref = 66.64 blind Re-scaling — now known
    to lie within the floor's own sampling spread [46,76]).
B3. Temperature by R3.1 (corrected buckets), D computed BEFORE training on the B1 train-pool
    LR inputs. RECORDED AT LAUNCH (2026-07-23): D = 0.231 -> temp = 2.5.
B4. Recipe: §4 with --re 2000 --stats <B2> --scales_re 1000, temp <B3>, n_outer 600,
    save_every 50, seed 0, fresh from the EMA base. RE_CFG(2000) repointed to the 40-seed file
    and the B1 split for this run.
B5. Selection R3.2 (deployed-vs-anchor on train-pool inputs); eval per §5 on TEST 24-39 at
    inference temps 1.0 and 0.30 (R3.4 headline); patchwork per R3.3.
B6. A-priori expectations: PRIMARY ret(0.30) in [0.80, 1.20] (adaptive campaign: 0.980;
    training-seed noise +-0.04); SECONDARY k* >= 80, hybrid low-k [1,5) >= 0.97; placement
    reported as-is. One attempt; no reruns.

B7. OUTCOME (2026-07-23, seqs 24-39 opened once, graded as-is): PRIMARY FAILED —
    ret(0.30) = 1.454. Secondaries: k* = 95 PASS; hybrid low-k 0.983 PASS; placement 0.721 vs
    base 0.673 (above base for the first time). All blind rules (R3.1 D=0.231->temp 2.5, R3.2
    ckpt iter0599, R3.3 kc=5) executed correctly. Root cause: B2's frozen anchor ARTIFACT was
    built from OLD-generation LR; the reward drove deployed output to parity with a spectrum
    1.26x hotter than the new generation's GT in E[32,96). See §2b (v1.3) for the corrective.

B8. ATTEMPT #2 TERMS (v1.3-compliant; runs only on user approval):
    - Anchor: `base_results/regime_stats_re2000_obsfit_v2.npz` — rebuilt by the frozen §2b
      procedure from NEW-file train-pool LR (seqs 0-3; already-opened data, no new unsealing).
      Fingerprint stored; pre-flight ratio vs training LR = 1.000 OK. Report-only diagnostic
      against the burned test seqs 24-39: anchor/GT E[32,96) = 0.997 (old artifact: 1.261).
      residual_ref = 66.63 (unchanged law). Floor, recipe, R3.1-R3.4, itemps: all unchanged.
    - TEST = a fixed subset of the SEALED SPARE, seqs 8-23 (16 seqs x 6 frames = 96 frames),
      still never opened. Seqs 5-7 remain sealed spare. Same primary/secondary bars as B6.
    - Attempt #1's result stands and is reported alongside; attempt #2 is a test of the v1.3
      corrective, not a replacement of the record.
    - RECORDED AT LAUNCH (2026-07-22): R3.1 vs the v2 anchor D = 0.293 -> temp = 2.5 (same bucket
      as attempt #1; consistency: 0.231 x 1.261/0.997 = 0.292). Pre-flight ratio 1.000 OK.
      Training: 600 iters, seed 0, fresh EMA base -> monitoring/ddpo_re2000_frozen_confirmatory2_ckpts.
    - OUTCOME (2026-07-22, sealed seqs 8-23 opened once): PRIMARY FAILED — ret(0.30) = 1.380
      (itemp 1.0: 1.283). Secondaries: k* = 95 PASS, hybrid low-k 0.983 PASS; placement 0.713 vs
      base 0.637. v2 anchor improved ret 1.454 -> 1.380 and R3.2 now approaches the anchor from
      below (selected iter0449, ratio 0.870), but the bar was still missed. See B9 root cause.

B9. DATA-GENERATION TEMPORAL INCOMPATIBILITY (discovered post-B8; affects BOTH attempts):
    - The 2026-07 40-seed files (Re=500/2000/10000) are saved at ~80x finer frame spacing than the
      old generation and the Re=1000 base-training data: lag-1 frame corr 0.99999 vs required
      0.986-0.991; lag 80 of the new file matches lag 1 of the old. One 280-frame sequence spans
      ~3.5 old-convention frames (corr 0.89 end-to-end).
    - Consequences: (a) triplets are near-static -> conditioning off-distribution vs the base
      model in both attempts; (b) all dt=1/32-based residuals are in the wrong convention (the
      "GT residual 625" vs floor 66.6 is a dt artifact, not physics); (c) effective sample sizes
      collapse (96 test frames ~ 16 independent states). Per-frame SPATIAL metrics (ret, k*,
      placement, spectra, anchors) remain validly measured.
    - RETRACTION: the earlier claim that the Re=10000 confirmatory (§1-7) was unaffected is
      withdrawn — kf_re10000_256_40seed.npy is the same fine-dt generation.
    - v1.4 PRE-FLIGHT (GT-free, run at launch): temporal_compat_check() in
      src/ddpo_ft/anchor_obsfit_builder.py — LR lag-1 corr must lie in [0.95, 0.9985].
    - CORRECTIVE (user-directed): stride-80 subsample. kf_re2000_256_40seed_dt32.npy = frames
      {39,119,199,279} of every sequence (offset 39 keeps the skip-transients convention);
      lag-1 corr 0.9871 -> dt-compatible. Anchor v3 rebuilt from the dt32 train-pool LR by the
      frozen §2b procedure (fit identical to v2; fingerprint matches the dt32 file). Pre-flights:
      freshness 1.000 OK, temporal 0.9864 OK.
    - RELAUNCH RECORDED (2026-07-22): D = 0.330 vs v3 anchor on dt-correct conditioning -> temp
      2.5 (R3.1). Base probe hik_ret 0.394 on dt32 vs 0.337 on the raw file — dt-correct
      conditioning alone helps the base. Training: frozen recipe, 600 iters, seed 0, train pool =
      8 triplets -> monitoring/ddpo_re2000_dt32_ckpts. Post-training eval is REPORT-ONLY on the
      burned seqs 24-39 (32 frames, SEAL_NPER=2) — NOT a sealed confirmatory; only seqs 5-7 of
      the Re=2000 file remain sealed.

## Appendix C — dt32 REPORT-ONLY runs (2026-07-23; gate + Re=10000 extension)
C1. Re=2000 dt32 RESULT (report-only, burned seqs 24-39, 32 frames, SEAL_NPER=2):
    GT residual 53.3 — INSIDE the calibrated floor spread [46,76]; dt convention restored.
    base ret 0.380 / place 0.698 / k*31. DDPO itemp0.30: ret 1.166 (B6 primary bar MET),
    k*=95, place 0.750; hybrid kc=7 lowk 0.988 (bar MET). itemp1.0: ret 1.101.
    R3.2 selected iter0599 (ratio 0.850, whole trajectory below anchor); R3.3 kc=7.
    D=0.330 -> temp 2.5. Probe 0.394 -> 0.762. NOT a sealed confirmatory (data previously
    opened); it validates the dt32 + procedure-rebuilt-anchor stack end to end.
C2. GATE (user-directed 2026-07-23): C1 met the B6 bar -> extend the identical pipeline to
    Re=10000.
C3. Re=10000 dt32 LAUNCH RECORD (2026-07-23):
    - File: kf_re10000_256_40seed_dt32.npy (stride-80 frames {39,119,199,279} of the CRC-verified
      raw file IvvIXQ==); temporal pre-flight 0.9757 OK (on the Re-trend: 1000:0.991,
      2000:0.987, 10000:0.976).
    - Anchor: regime_stats_re10000_obsfit_dt32.npz — frozen §2b procedure on train-pool LR
      (seqs 20-23). kd fit 113.6 vs blind prior 112.7 (extrapolation law confirmed by target
      obs); residual_ref 5180 (law; old blind npz had 5183). Freshness 1.000 OK. vs the old
      blind-extrap anchor: [10,32) x1.263 (obs-band correction), [32,96) within ~6%.
    - R3.1: D = 0.163 -> temp 3.5 (deepest-deficit bucket). Recipe otherwise frozen: 600 iters,
      seed 0, lr 5e-5, K2 S=[100,75] ddim50 eta1, EMA base -> monitoring/ddpo_re10000_dt32_ckpts.
    - Eval: REPORT-ONLY on seqs 10-19 (historical non-sealed split; SEAL_RE=10000,
      SEAL_TRAIN=20-23, SEAL_NPER=2). CONFIRMATORY SEQS 25-39 REMAIN SEALED for a future
      one-shot §1-7 under this dt32 stack.
C4. Re=10000 dt32 RESULT (2026-07-23, report-only, seqs 10-19, 20 graded frames):
    Judged against the REGIME-APPROPRIATE a-priori bar (§6, set for Re=10000): PRIMARY
    ret in [0.75,1.30] -> ret(0.30) = 1.183 MET (itemp1.0: 1.128). SECONDARY k* >= 55 ->
    68/69 MET; hybrid low-k >= 0.97 -> 0.994 MET. (Against Appendix-B's Re=2000 bar of
    k*>=80 it would miss — stated for completeness; §6 is the pre-registered bar here.)
    - base: ret 0.242, k* 30, place 0.702. DDPO k* 30 -> 69 (effective resolution 2.3x).
    - PLACEMENT 0.641 < base 0.702: the §6 information-ceiling prediction (placement <= base
      at Re=10000) is CONFIRMED on this run — placement gains at Re=2000 (0.75 > 0.70) do
      not extend to Re=10000, exactly as pre-registered.
    - low-k 0.994-1.003: no low-k bleed at all this time (hence kc=2 and hybrid == DDPO).
    - FLOOR LAW LIMIT: measured GT residual 399.8 vs law-extrapolated residual_ref 5180 —
      the b=2.705 law OVER-predicts ~13x at Re=10000 (safe direction for the one-sided hinge;
      outputs sat at ~34, below both). Likely cause: the dataset is grid-limited above k~100,
      capping true residual growth. Flag for the fresh-repo spec: floor law needs a
      grid-resolution correction beyond Re~2000.
    - R3.2 selected iter0549 (ratio 0.790, approach-from-below, no overshoot); R3.3 kc=2.
    - Sealed seqs 25-39 remain UNTOUCHED; a one-shot §1-7 confirmatory under the dt32 stack
      (with §2 amended per v1.3/v1.4) is still available.
C5. SEQUENCE NON-INDEPENDENCE (discovered 2026-07-23 during report prep): the 40 sequences of the
    2026-07 generation are ORDERED SEGMENTS, not independent seeds — adjacent-index correlation
    ~0.6 at matched frames, decaying to the ~0.1 same-regime background over ~5-8 indices (old
    20-seed file: 0.095 = independent baseline). Implications, split-by-split:
    - Re=2000 dt32 grading (train 0-3 / test 24-39): CLEAN — min index distance 20, corr ~0.05.
    - Re=2000 attempt #2 (test 8-23): mild proximity for seqs 8-12 (d=5-12, corr 0.14-0.20);
      leakage direction would HELP, so the FAIL verdict is conservative and stands.
    - Re=10000 dt32 grading (train 20-23 / test 10-19): PARTIAL — test seqs 16-19 are adjacent
      (corr up to 0.64); seqs 10-15 are at background. Clean-subset re-grade (10-15 only) run and
      reported alongside; both numbers stand in the record.
    - Re=10000 SEALED confirmatory set 25-39: seqs 25-26 are within the correlation length of
      probe 24 / train 23. NOTED for the one-shot: grading may report 25-39 (as frozen) plus a
      27-39 clean-subset line; the split itself is NOT changed post-hoc.
    - Effective independent states per 16-seq test set: ~4-6 (correlation length ~3-4 indices),
      not 16. All quoted margins should be read against that.
