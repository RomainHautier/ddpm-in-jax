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

## §8 — v1.5 LOCKED OOD SETUP (user-directed corrections, 2026-07-23; supersedes R3.1/R3.4' notes)
8.1 REWARD EXTRAPOLATION: unchanged — the §2b frozen procedures (obs-fit anchor from deployment
    train-pool LR; floor by the b=2.705 law) are the ONLY target-regime inputs. Known a priori.
8.2 TRAINING TEMPERATURE = 2.5, SINGLE VALUE FOR ALL OOD DEPLOYMENTS. The R3.1 deficit->temp
    buckets are REMOVED from the deployment path: their calibration used GT-informed optima
    measured ON THE TARGET REGIMES (circular for a GT-free claim), and the "3.5 beats 2.5 at
    Re=10000" evidence came from the broken fine-dt run (near-static conditioning). 2.5 is the
    value validated on dt-correct data at the nearest extrapolation regime. D may still be
    COMPUTED and RECORDED at launch as a diagnostic — never acted on.
8.3 INFERENCE TEMPERATURE = 0.30, FROZEN. itemp 1.0 is reported as the untempered reference only;
    it is not a performance knob and is never promoted post-hoc. The R3.4' blind-rule idea is
    dropped (user decision).
8.4 TEST-SET INDEPENDENCE (consequence of C5): OOD eval sequences MUST be spaced >= 5 indices
    apart AND >= 5 indices from every train/probe sequence; grade ALL available frames of each
    chosen sequence (for dt32 files: both triplets). The cross-sequence correlation of the chosen
    test set is measured from LR at launch (GT-free) and recorded.
8.5 CONSEQUENCE FOR EXISTING RESULTS: Re=2000 dt32 already complies with 8.2 (temp 2.5); its
    grading is re-run on decorrelated picks {24,29,34,39} for the record. Re=10000 dt32 was run
    at temp 3.5 under the old rule -> RERUN at temp 2.5 required to validate the v1.5 lock;
    report-only eval on decorrelated non-sealed picks {0,5,10,15}. Sealed 25-39 untouched.
8.6 NEXT DATA DROP SPEC (for regeneration): truly independent seeds (distinct initial conditions,
    full spinup each), saved at dt=1/32 (or save-every-80th of the current fine step), >= 320
    frames per sequence. With that, full-length-sequence evaluation as in the original campaign.
8.7 §8.5 OUTCOME (2026-07-23): Re=2000 decorrelated re-grade PASSED (ret 1.160, k*95, place
    0.817>0.744 — headline bias-robust). Re=10000 RETRAIN AT LOCKED TEMP 2.5 FAILED the band:
    ret(0.30) = 0.498, k* = 41 (probe 0.143 vs 0.234 at temp 3.5). Paired evidence on dt-correct
    data, same recipe/seed: temp 2.5 -> 0.498 | temp 3.5 -> 1.183. The single-temp lock carries a
    real, now-measured cost at the deepest-deficit regime. DECISION PENDING (user): (a) keep
    single temp 2.5 and accept the Re=10000 result as-is; or (b) freeze a two-point deficit rule
    (D >= 0.20 -> 2.5; D < 0.20 -> 3.5) with Re=2000/Re=10000 explicitly marked as SPENT
    calibration regimes — non-circular only for FUTURE regimes and for the sealed 25-39 one-shot,
    which remains the sole untouched validation ground either way.
8.8 CLOSING THE TWO OPEN KNOBS (studies of 2026-07-23, both committed):
    (a) R3.3' HYBRID CUTOFF — ADOPTED. k_c = argmax of the locked training reward over the hybrid
        sweep on TRAIN-POOL outputs (GT-free). Validation: in-dist the landscape is flat (no wrong
        pick possible; blind 5 vs GT-tiebreak 3, indistinguishable); at Re=2000 the reward has an
        interior maximum at k_c=7 which equals the old crossing rule's pick AND attains the best
        low-k in the GT sweep (0.988; ret flat at 1.144 across all cutoffs). R3.3 (crossing) is
        retained as the fallback when the reward landscape is flat within noise.
    (b) R3.1' TEMPERATURE — the in-dist derivation is CLOSED OFF (negative result): the Re=1000
        ladder (1.5/1.75/2.0/2.5) is all-in-band on GT (0.877-1.023), so no failure exists to
        calibrate a threshold; healthy in-dist plateaus (0.657-0.682) sit BELOW healthy OOD ones
        (0.79-0.85), so no universal theta transfers; and plateau does not track fine quality
        (in-dist 2.5: highest plateau 0.682, worst ret 0.877, k* 86). What SURVIVES: the blind
        plateau is a reliable COARSE failure detector — the Re=10000 under-exploration stall reads
        0.60 vs the same regime's achievable 0.79, flat from iter ~50. The escalation rule
        (train 2.5; plateau < 0.70 at iter ~150 -> restart at 3.5) classifies all seven measured
        runs correctly but its threshold is calibrated on the OOD pair -> adopting it is §8.7
        option (b): Re=2000 + Re=10000 SPENT, sealed 25-39 the only validation ground.
8.9 GREEDY-POLICY SWEEP (2026-07-23, eval-only, both dt32 models, decorrelated picks, itemp
    0.30/0.15/0.05/0.0 at eta=1 — the true greedy limit; eta=0 is a different sampler, known bad):
    SATURATED in both regimes. Re=2000: ret 1.160->1.170, all other metrics flat to 3 digits.
    Re=10000 (temp-3.5 model, picks {0,5,10,15}): ret 1.073->1.080, flat otherwise — note 1.073 at
    itemp 0.30 is the best Re=10000 number yet, on the most independent picks (vs 1.183/1.168 on
    10-19: sequence-sampling variance at small effective n). CONCLUSION: R3.4 = 0.30 sits on the
    greedy plateau; residual exploration noise below 0.30 contributes nothing measurable; greedy
    (0.0) is an equivalent deployment choice, not an improvement. No protocol change.

## §8.10 — ONE-SHOT Re=10000: STAGED EXECUTION (user-directed, 2026-07-23)
- §8.7 RESOLVED (user): the ESCALATION RULE is adopted — train at 2.5; blind train-pool plateau
  < 0.70 -> escalate to 3.5 (fresh run). Re=2000 + Re=10000 are SPENT calibration regimes; the
  sealed one-shot is the rule's out-of-sample test.
- STAGED GRADING DIRECTIVE (user): Stage 1 = the deployment setup is assessed ONLY against the
  extrapolated anchor — no GT quantity computed or read anywhere, sealed 25-39 stays CLOSED.
  Stage 2 = the single sealed GT grading, executed only on the user's explicit go.
- RUN REUSE DECLARATION: the escalation rule is applied to the two existing frozen-recipe seed-0
  runs (ddpo_re10000_dt32_t25_ckpts, ddpo_re10000_dt32_ckpts). Both were executed fresh from the
  EMA base with all decisions blind; a live escalation with the same seed would replicate them.
  Their prior GT gradings on burned/non-sealed splits are part of the declared calibration spend.
- Stage-1 machinery: src/ddpo_ft/oneshot_stage1_re10000.py (escalation record, R3.2 blind
  selection, R3.3' reward-selected k_c, anchor-relative band assessment at itemp 0.30, all on
  train-pool inputs).
- STAGE 1 EXECUTED (2026-07-23, log monitoring/ab_pdelocal/oneshot_stage1.log):
    Escalation: temp-2.5 plateau 0.610 < 0.70 -> ESCALATED to 3.5 (rule fired as designed).
    R3.2 selected iter0549 (plateau 0.789, approach-from-below, no overshoot).
    R3.3' reward-selected k_c = 6 (supersedes crossing kc=2 per §8.8a adoption).
    Deployment vs anchor at itemp 0.30: plateau [10,96) = 0.798; bands [1,5) 0.998 (parity),
    [10,32) 0.760, [32,64) 1.146, [64,96) 0.223 (the far tail is the anchor's model-extrapolated
    region — deployed staying far below it is the expected safe side). PDE residual 36.3, below
    the blind floor (hinge inactive). No GT quantity computed; sealed 25-39 untouched.
- STAGE-2 CONFIG FROZEN (zero remaining degrees of freedom): model ddpo_re10000_dt32_ckpts/
    ddpo_re1000_iter0549.pkl, deep cascade [150,100,50] 86 steps lam3, itemp 0.30 (1.0 reference),
    hybrid tanh crossover k_c=6, TEST = sealed seqs 25-39 (primary, as frozen in §1) with a
    27-39 clean-subset line reported alongside (C5), bars per §6. Executes on user go only.
8.11 ESCALATION TIMING CORRECTED + RULE PROVENANCE AUDIT (2026-07-24): full plateau trajectories
    measured for the in-dist ladder (monitoring/indist_plateau_traj.json) show the healthy @1.5
    run sits at 0.61-0.65 until iter ~260 — an iter-150 escalation check WOULD MISFIRE in-dist.
    The rule is amended: R1 success = END-OF-RUN plateau >= theta = 0.65 (in-dist-derived: min
    healthy end-of-run 0.657); grey zone [0.62,0.66] resolved by the stall test (failed run flat
    from iter ~50; all healthy runs still rising). Stage-1's escalation record (end-of-run 0.610)
    is consistent with the amended rule. Full provenance audit of all deployment rules:
    docs/anchor_derived_rules.md (R1 in-dist / R2 in-dist criterion + anchor lever / R3
    anchor-only / R4 in-dist).

## §8.12 — STAGE 2 EXECUTED: THE SEALED ONE-SHOT VERDICT (2026-07-24, user-authorized)
Sealed seqs 25-39 opened ONCE (logs stage2_sealed_25_39.log / stage2_clean_27_39.log). Frozen
config as §8.10 (iter0549, itemp 0.30 headline, k_c=6 pre-registered, escalation rule applied
blind & out-of-sample). Graded as it fell, vs the §6 bars:
- PRIMARY ret(0.30): full sealed set 0.758 [bar 0.75-1.30 -> IN-BAND by 0.008]; C5 clean subset
  (27-39) 0.792 [IN-BAND by 0.042]. Both margins below the pre-registered 0.08 interpretation
  floor -> the BAND POSITION at the edge is not interpretable; the MAGNITUDE claim §6 tests is
  unambiguous: base 0.193/0.201 (predicted ~0.18) -> DDPO 3.9x, k* 32 -> 61/62.
- SECONDARY: hybrid lowk 0.992/0.993 PASS; k* 61/62 >= 55 PASS.
- PLACEMENT: 0.211/0.209 vs base 0.727/0.766 -> the §6 info-ceiling prediction CONFIRMED, and far
  more starkly than burned splits suggested (0.64 there): on virgin states the fine-structure
  placement is essentially uncorrelated. The prediction called the direction; the sealed set
  revealed the size.
- The sealed segment is HARDER than every previously seen split (GT residual 1400/1187 vs
  400-500; base ret 0.193 vs 0.23-0.24) — consistent with ret 0.76-0.79 vs ~1.07 on burned picks
  (regime-segment difficulty + sequence-sampling variance at effective n ~3-5).
- The clean subset grades HIGHER than the full set: the near-train seqs 25-26 depressed rather
  than inflated the result — no leakage story.
VERDICT: the one-shot PASSES all §6 criteria as pre-registered. The GT-free chain — obs-fit
anchor, escalation temperature (blind, out-of-sample), R3.2 selection, R3.3' cutoff, frozen
itemp — is validated end-to-end on never-touched data, with the placement information ceiling
confirmed as the honest limit of the method.

## 9. OOD INFERENCE SELECTION — PROVABLY GT-FREE (2026-08-07)

### 9.1 Why the previous claim was weak
`grid_downsample_degrade` and `coarse_spec` both open the full 256^2 ground-truth array and keep
every 4th pixel. The unused 15/16 of the field was loaded into memory on every call. Nothing used
it, but "we do not use ground truth" was a statement about code discipline that no auditor could
verify from the outside.

### 9.2 The fix: materialise the observation, then never read the fine field again
`src/ddpo_ft/materialize_observed.py` performs the ONE step that legitimately reads the fine field
and writes exactly what a 4x-coarse solver would have produced:
    flow-data/observed/re<R>_obs.npy   shape (n_seq, n_frames, 64, 64)   = 1/16 of the fine field
Everything downstream reads that file and nothing else. There is no ground truth present to leak.

EXACTNESS PROVEN, not asserted (--verify, run 2026-08-07 on Re=2000 seqs 0-7):
  - nearest-neighbour-filled 256^2 field: bit-identical to the old path (np.array_equal True)
  - coarse spectrum used by the anchor: max relative difference 0.00e+00
This is a provenance change with zero numerical effect.

### 9.3 Information provenance of every input to OOD inference selection
  TARGET-REGIME GROUND TRUTH ......... NEVER READ. Not by the anchor, the sweep, the scoring or
                                       the selection. After 9.2 it is not even opened.
  TARGET-REGIME OBSERVATION .......... re<R>_obs.npy — the 64^2 coarse field. The only target-
                                       regime information the pipeline ever sees.
  REFERENCE-REGIME GROUND TRUTH ...... Re=500 and Re=1000 only. Enters as FROZEN CONSTANTS fixed
                                       before deployment: the anchor's alpha/p priors, the kd and
                                       T laws, the inference temperature 0.30, and the set-point.
                                       Declared, not hidden — the method is GT-free AT THE TARGET,
                                       not GT-free everywhere.
  BASE MODEL ......................... EMA checkpoint trained on Re=1000 only.

### 9.4 The selection rule (frozen)
Do NOT freeze a cascade. Freeze the rule that picks one:
  1. Build the anchor from re<R>_obs.npy by the frozen procedure (v1.3), fingerprinted.
  2. Run every cascade config on those same observed inputs.
  3. Score each: blind = E_deployed[10,96) / E_anchor[10,96), on the anchor's own source pool.
  4. Select the config whose blind score is closest to the SET-POINT.
No target ground truth at any step. This is R5, measured 4/4 correct at regimes with a native model
and 11/12 on unseen regimes.

### 9.5 The set-point, and its one honest weakness
SET-POINT = 0.768: the blind score of a model that GROUND TRUTH confirmed was best, measured at
Re=1000 (iter 449, val retention 0.992; see anchor_derived_rules.md). It is reference-regime GT,
frozen a priori — legitimate, and declared.
WEAKNESS: n = 1. It is one number from one regime, and the blind signal's dynamic range (0.05) is
only ~2x its noise (+-0.02). Re=500 is a second reference with full GT that has NEVER been
finetuned; repeating the calibration there would say whether the set-point is a constant of the
method or drifts with regime. Until then, applying 0.768 at Re>=2000 is an ASSUMPTION, not a
measurement, and must be labelled as one in any result that depends on it.
