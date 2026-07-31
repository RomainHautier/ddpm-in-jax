# Anchor-derived deployment rules — provenance audit (2026-07-24)

Purpose: for every deployment rule, state exactly where it comes from — (a) Re=1000 runs (GT use
legitimate), (b) the extrapolated anchor alone (GT-free everywhere), or (c) OOD GT knowledge
(spent / hindsight — flagged). This is the requirements document for the fresh-repo PPO: every
rule below must be derivable without target-regime ground truth.

## The GT-free signal
plateau(t) = E_deployed[10,96) / E_anchor[10,96), deployed = deep-cascade output (itemp 1.0) on
TRAIN-POOL inputs, anchor = frozen §2b obs-fit extrapolation from the same pool's LR. Measurable
at any checkpoint of any run, in any regime, with zero GT.

## Measured trajectories (fig_plateau_trajectories.png; per-checkpoint noise ~±0.02)

| run | plateau trajectory | end value | GT verdict (where legitimate/burned) |
|---|---|---|---|
| Re=1000 @1.5 (iters 19–299) | 0.613 → 0.657, slow rise | 0.657 | in-band 0.893 |
| Re=1000 @1.75 (319–499) | 0.651 → 0.671 | 0.671 | in-band 0.987 |
| Re=1000 @2.0 (319–499) | 0.649 → 0.669 | 0.669 | in-band 1.023 |
| Re=1000 @2.5 (319–499) | 0.645 → 0.680 | 0.680 | in-band 0.877 (k* 86) |
| Re=2000 @2.5 (49–599) | 0.746 → 0.850, rising | 0.850 | in-band 1.160–1.166 (burned splits) |
| Re=10000 @3.5 (49–599) | 0.683 → 0.789, rising | 0.784–0.798 | in-band 1.073–1.183 (burned splits) |
| Re=10000 @2.5 (49–599) | 0.588 → 0.612, FLAT from iter ~50 | 0.612 | FAIL 0.498 (burned splits) |

## R1 — SUCCESS CRITERION [in-dist-derived]
A finetune has worked iff its END-OF-RUN plateau ≥ θ = 0.65, where θ := the minimum end-of-run
plateau across all healthy in-dist runs (0.657 at temp 1.5; every in-dist run above it graded
in-band 0.877–1.023). Applied to OOD it classifies all runs correctly (0.850 ✓, 0.784 ✓ vs
0.612 ✗). CAVEATS, stated honestly:
- The margin is thin: weakest healthy 0.657 vs failed 0.612–0.622 → treat [0.62, 0.66] as a grey
  zone requiring the stall test (below).
- The earlier working figure of 0.70 was an OOD-eyeballed number and would misclassify every
  healthy in-dist run; 0.65 is the in-dist-derived value. ("0.7 or above" ≈ right idea, wrong
  constant.)
- We wrote this recipe down after seeing the OOD outcomes (hindsight risk, declared). The recipe
  itself uses only in-dist data; sealed Stage 2 is the untainted test.

## R2 — TEMPERATURE ESCALATION [criterion in-dist; lever anchor-only; timing corrected]
If R1 fails — end-of-run plateau < θ, or the plateau is FLAT (< +0.03 over the final 300 iters)
inside the grey zone — retrain hotter (2.5 → 3.5, fresh run/optimizer).
- The LEVER's evidence is anchor-only: at Re=10000, escalation moved the plateau 0.612 → 0.789.
  No GT enters that comparison.
- The CRITERION is in-dist-derived (R1).
- What in-dist alone can NOT give: a reason to escalate — every in-dist temp succeeds, so the
  in-dist ladder is temp-invariant in plateau (0.657–0.680) and contains no failure. The
  escalation rule is therefore "in-dist criterion + anchor-observed lever", not GT-derived — but
  our CONFIDENCE that plateau ≥ θ ⇒ in-band OOD retention currently rests on burned-split GT
  gradings; that is precisely what sealed Stage 2 tests.
- TIMING CORRECTION (supersedes the "check at iter ~150" phrasing in §8.7b/§8.8): an early check
  misfires — the healthy in-dist @1.5 run reads 0.63 at iter ~150, inside failure territory. The
  check is END-OF-RUN (600 iters), with the stall signature (flat-from-iter-50) as corroboration:
  the failed run was flat for 550 iters; every healthy run was still rising.

## R3 — HYBRID LOW-K PATCH + CUTOFF [anchor-only]
The base-DDIM recon carries the low-k band; the hybrid restores it. k_c = argmax of the LOCKED
training reward over the hybrid sweep on train-pool outputs — the reward is computed against the
anchor, no GT. Validated: in-dist the landscape is flat (no wrong pick possible); at Re=2000 the
reward's interior maximum (k_c=7) coincides with the GT optimum and best low-k (0.988); at
Re=10000 it picked k_c=6 in the Stage-1 anchor-only assessment.

## R4 — INFERENCE TEMPERATURE [in-dist-derived]
0.30, one value for all regimes. Derived from the in-dist Re=1000 inference sweep (colder
monotonically better down to 0.30); the greedy sweep (0.30→0.15→0.05→0.0) then showed full
saturation at BOTH OOD regimes (±0.01, all metrics flat) — 0.30 sits on the greedy plateau.
itemp 1.0 is reported as the untempered reference only.

## Summary for the fresh-repo spec
Everything a deployment needs is: the frozen anchor procedure (§2b) + pre-flights (§2b/v1.4) +
the frozen recipe + R1–R4 above. GT enters only through Re=1000 calibration. The open validation:
sealed Re=10000 seqs 25–39 (Stage 2, frozen config, user-gated) tests the whole chain one-shot.
New Re=2000/10000 data on the user's drive: run the pre-flights first (temporal lag-1, anchor
fingerprint, seq-independence ≥5-index spacing), rebuild anchors per procedure, then R1–R4 apply
unchanged.

## R5 — INFERENCE-DEPTH SELECTION FOR CROSS-REGIME DEPLOYMENT [healthy-reference set-point]
(2026-07-24 backtest, src/ddpo_ft/backtest_lower_re.py + monitoring/depth_sweep_re10k_at_re2k.py)
Finetuned models carry their training regime's ENERGY DOSE in the weights: at full frozen depth,
the Re=2000 model overshoots Re=1000 by 1.52x; the Re=10000 model overshoots Re=1000 by 3.92x and
Re=2000 by 2.53x. Nothing structural is lost (k* stays 95, low-k intact; the in-dist model on
virgin seqs 0/16 reads 0.937/0.906 — first fully-virgin in-dist eval) — it is a dose problem.
CURE (GT-free): sweep the inference-depth ladder (K3x86 -> K2x50 -> K1[100]x20 -> K1[75]x20 ->
K1[50]x12) and pick the config whose blind anchor score lands at the regime's HEALTHY SET-POINT:
  theta(regime) := the blind reading of a verified-good native model on the same pool.
  Re=1000: 0.66-0.68 (four GT-verified runs + virgin pool, tight). Re=2000: 1.116 (native model).
  NOT 1.0 — the score's healthy value is offset by the anchor's own bias (true GT reads 0.78 on
  the Re=1000 anchor) times the universal mid-band shortfall (~0.88): 0.78 x 0.88 = 0.68.
  Targeting 1.0 selects the overshooting config every time (naive-parity rule FAILS).
Results of the blind picks: 2x model @Re1000 -> K2 (ret 0.870, place 0.867); 10x model @Re1000 ->
K1[100] (ret 1.059, place 0.845); 10x model @Re2000 -> K2 (ret 1.230, place 0.750 = native's).
Dose-depth law: hotter model -> shallower blind pick, monotone in every sweep. The coarse 5-rung
ladder lands within ~0.2 of parity; a finer ladder would tighten it. Depth selection REPAIRS dose
mismatch; it does not replace native finetuning (the set-point requires a healthy native reference).

### R5 CORRECTION (same day, user-caught): the set-point must not reference GT-verified models
As first written, theta(regime) = "reading of a VERIFIED-GOOD native model" — verification used GT,
unusable at a virgin regime. CORRECTED, fully-blind formulation, validated:
  set-point = (anchor/truth offset in the band) x (achievable healthy restoration)
  - factor 1 ~ 1 BY CONSTRUCTION for frozen obs-fit anchors: the obs band k[6,30] is fitted to
    the deployment LR itself and carries 92% of the score's energy. No GT needed.
  - factor 2 ~ 0.85 (in-dist-derived at Re=1000; consistent with healthy OOD readings 0.79-0.85).
  => deployable set-point band [0.80, 0.85], scored ON THE ANCHOR'S SOURCE POOL (the LR at hand).
  The Re=1000 "0.68" never enters deployment: that regime's measured-stats anchor is hot by 1/0.78
  (factor 1 = 0.78) — a different instrument, only readable because in-dist GT is legitimate.
BLIND CHECK (r5_blind_check.py): ood-re10000 @ re2000, source-pool scores 1.090/0.835/0.759/0.711/
0.681 for K3/K2/K1[100]/K1[75]/K1[50] -> band selects K2, which is the GT-best rung (1.230 vs
2.532 full-depth). Cross-check honesty: resolved-vs-LR parity (~1.0 all rungs) and residual-vs-
floor did NOT discriminate here (floor 66.6 loose at Re=2000) — they are guards against gross
failure, not selectors; the anchor score is the selector.

### R5 — cross-regime vs NATIVE benchmark (2026-07-24, ladder viz with native overlay)
After blind de-dosing, the foreign model lands close to the in-regime model on the same frames:
  @Re=1000: cross (Re10000 model, K1[100]) ret 1.059 / place 0.845 / lowk 0.994
            native (in-dist model, K3)     ret 0.937 / place 0.906 / lowk 0.978
  @Re=2000: cross (Re10000 model, K2)      ret 1.230 / place 0.750 / lowk 0.994
            native (Re2000 model, K3)      ret 1.160 / place 0.817 / lowk 0.921
Retention gap 0.07-0.12; cross-regime preserves low-k slightly BETTER (shallower chain disturbs the
large scales less). The native model's durable edge is PLACEMENT (+0.06 both regimes): energy AMOUNT
is transferable via inference dose, energy POSITION is what in-regime training buys and an inference
knob cannot recover. Figures: fig_ladder_{spec,pdf,fields}_{re1000,re2000}.

### R5 — OUT-OF-SAMPLE TEST ON UNSEEN REGIMES: THE ABSOLUTE SET-POINT IS REFUTED
(2026-07-29, src/ddpo_ft/crossregime_new.py; Re=1500 + Re=3000, new generator, no model ever
trained on them, no part in any calibration. Blind picks on the anchor's source pool, GT revealed
after. Measured band from Stage 0 @itemp0.30: [0.798, 0.858].)
SCORECARD (GT-best rung by |ret-1| vs the blind pick): 1 hit / 5.
  re1000@re1500 (up)   GT-best K4  (1.066, blind 0.873) | pick K4        1.066  HIT
  re2000@re1500 (down) GT-best K3  (1.068, blind 0.888) | pick K2        0.636  MISS
  re10000@re1500(down) GT-best K2  (1.165, blind 0.863) | pick K1[100]   0.746  MISS
  re2000@re3000 (up)   GT-best K3  (0.758, blind 0.690) | pick K4        1.332  MISS
  re10000@re3000(down) GT-best K2  (0.831, blind 0.681) | pick K3        1.689  MISS
WHAT SURVIVES (and is strong): the blind score RANKS the ladder correctly in all five, and the
parity reading is nearly MODEL-INDEPENDENT within a regime — 0.863/0.873/0.888 for three different
models at Re=1500; 0.681/0.690 for two at Re=3000.
WHAT FAILS: the parity reading is a REGIME constant, not a universal one. It tracks the anchor's
bias, measured here as anchor/GT over [10,96): Re=1500 1.088, Re=3000 1.419, Re=2000(old) 0.838.
RETRACTION: the R5-CORRECTION claim "factor 1 ~ 1 BY CONSTRUCTION for obs-fit anchors" is FALSE.
Measured in the FITTED band [6,30): 1.020 (re1500), 1.309 (re3000), 0.818 (re2000-old). The
obs-fit is prior-regularized (alpha/p/kd priors carry weight), so when a regime's true shape
departs from the reference-derived priors the fit is pulled off its own observations.
ALSO REFUTED: normalizing the score by the base recon's score does not cancel the bias
(parity/base = 1.22-1.26 at Re=1500 but ~1.45 at Re=3000).
NARROWED SCOPE: R5 is valid for dose-correcting a FOREIGN model inside a regime that already has a
healthy native reference (its parity reading is then measurable GT-free from that reference, and
transfers across models). It is NOT valid for a brand-new regime. Prior R5 validations stand —
they ran inside the calibration regimes — but the generalization claim does not.
SECONDARY: up-transfer is more strained than down-transfer (K4 buys retention but costs placement
0.734->0.630 and low-k 0.931->0.884; truncation costs almost nothing). The PDE floor cannot serve
as an alternative selector — at Re=3000 the law gives 199.5 vs measured residuals 15-25.
PATH FORWARD: make the anchor accurate in its own observation band (weaken the shape priors there,
or re-derive the coarse->fine transfer per data generation); a set-point only becomes portable if
the instrument stops drifting between regimes.

### R6 — CIRCULARITY AUDIT (user-caught, 2026-07-30)
The early-stop band [0.798, 0.858] = [re10000 healthy, re2000 healthy], both measured from those
regimes' own finished 600-iter runs. Consequences, split per run:
- Re=10000: the lower edge IS its own healthy value -> the criterion was derived from the run it
  was applied to. Its non-fire is CIRCULAR and is discarded as a result. (An earlier draft called
  it "structural"; that understated the problem.)
- Re=2000: scores 0.803->0.834 never approached 0.858, so the binding threshold was 0.798 — a
  FOREIGN constant (from re10000). The 42% saving therefore survives the circularity objection,
  but n=1, and R5's refutation shows cross-regime constants are exactly what fails to transfer.
  Read as suggestive, not validated.
NO VALIDATED BLIND STOPPING CONSTANT EXISTS. Options and their status:
  (a) threshold from another regime — refuted in general by R5 (anchor bias spans 0.82-1.31);
  (b) no threshold, shape/slope criterion ("stop when the score stops rising") — the naive version
      is ruled out by the data we have: re10000 was flat iters 199-349 (0.766->0.764) then climbed
      to 0.798, so a slope rule stops it prematurely.
WHAT IS SOUND: the mechanism — monitoring the DEPLOYMENT configuration in-loop, GT-free, and
proven non-invasive (R6 re10000 ckpts are BITWISE IDENTICAL to the original run's, which also
establishes seed-reproducibility of the pipeline).

### CROSS-REGIME TRANSFER — SUMMARY OF WHAT THE EVIDENCE SUPPORTS
1. A finetune stores two separable things: HOW MUCH fine-scale energy to inject (welded to the
   training regime) and WHAT STRUCTURE to inject (transfers: morphology, spectral shape, PDF
   tails, k*). Inference depth is a monotone, BIDIRECTIONAL knob on the amount (K4 adds, K1
   removes) with no retraining.
2. The knob cannot be set blind on a new regime: the correct target reads 0.87 (re1500), 0.685
   (re3000), 0.80-0.86 (calibration regimes), and the spread tracks unmeasurable anchor bias.
3. Even with the target known, native training keeps ~+0.06 placement over the de-dosed foreign
   model (retention comes within 0.07-0.12). Dose is transferable; POSITION is what in-regime
   training buys.
4. Up-transfer is the worse direction (adding dose costs placement 0.734->0.630 and low-k
   0.931->0.884; removing dose costs almost nothing).
=> Cross-regime reuse is a FALLBACK, not a substitute for native finetuning (which is GT-free and
   passed a sealed one-shot). It is worthwhile only where the dose can be fixed by an existing
   native model or a small GT sample. THE UNLOCK IS THE ANCHOR, not the selection rule: every
   failure traces to the obs-fit drifting 18% cold to 31% hot inside its own fitted band. Four
   untouched regimes (1500/3000/4000/5000) are available to test an improved anchor.

## R5 — DEFINITIVE TEST AT ROBUST n (2026-07-30, src/ddpo_ft/transfer_grade_636.py)
Design: 640 triplets across 16 sequences per regime (independent units are set by SEQUENCE count —
adjacent triplets are ~98% correlated, so 636-in-2-sequences would be 2 units, not 636). Errors are
bootstrap-by-sequence. Frozen config throughout (K3[150,100,50]x86, lam3, itemp 0.30): the
NO-RETUNING transfer baseline. Regimes 1500/3000/4000/5000 unseen by every model and calibration.

ret (+-seq-boot) | place | k* | blind        [band = 0.798-0.858]
Re=1500  base 0.338+-.003 .751 31 0.680 | indist 0.679+-.009 .877 85 0.740 | re2k 1.097+-.025 .739 94 0.855 ACCEPT | re10k 2.597+-.086 .681 95 1.068
Re=3000  base 0.269+-.002 .755 31 0.583 | indist 0.463+-.006 .844 43 0.618 | re2k 0.693+-.015 .734 63 0.704 | re10k 1.491+-.043 .670 76 0.866
Re=4000  base 0.250+-.001 .744 31 0.561 | indist 0.401+-.004 .823 38 0.587 | re2k 0.579+-.011 .716 55 0.662 | re10k 1.216+-.035 .651 70 0.808 ACCEPT
Re=5000  base 0.235+-.001 .746 31 0.615 | indist 0.365+-.004 .806 36 0.636 | re2k 0.524+-.007 .694 52 0.718 | re10k 1.075+-.023 .642 68 0.873

CORRECTION TO THE EARLIER REFUTATION: the "1 hit in 5" figure was NOISE-INFLATED and is withdrawn.
At robust n the band scores 3 of 4: correct ACCEPT at Re=1500 (ret 1.097) and Re=4000 (ret 1.216),
correct REJECT-ALL at Re=3000 (best available 31% under / 49% over — nothing in the [0.75,1.30] bar),
and one MISS at Re=5000 (rejects ret 1.075, essentially parity, for exceeding the ceiling by 0.015).
THE CONCLUSION NONETHELESS STANDS, on far cleaner evidence — the same Re=10000 model:
    at Re=3000: blind 0.866 -> true ret 1.491  (49% over)
    at Re=5000: blind 0.873 -> true ret 1.075  ( 7% over)
Nearly identical blind readings, wildly different truth; the 49%-over case reads LOWER than the
near-perfect one. The absolute value is NOT comparable across regimes and no threshold can separate
these. (Method note: I was one row from retracting a correct conclusion after three favourable
points arrived first — the small-n evidence was bad, the conclusion was right.)
WHAT IS ESTABLISHED: the blind score is a reliable RELATIVE instrument — monotone in dose within
every regime, 4 of 4, without exception. It RANKS; it does not CALIBRATE.
TRANSFER PERFORMANCE ITSELF (no retuning, frozen config):
- Nearby transfer works: Re=2000 model -> Re=1500 gives ret 1.097 (its own regime: 1.155).
- Far transfer fails on dose in the predicted direction and magnitude (re10k -> re1500: 2.597).
- Best available model is dose-matched to regime distance: re2k wins at 1500, re10k at 4000/5000.
- PLACEMENT degrades monotonically with dose in EVERY regime (e.g. Re=5000: .806 -> .694 -> .642),
  and only the UNDER-dosed in-dist model ever improves placement over the base recon. Energy parity
  and positional accuracy are in direct tension — a real trade-off, not a tuning artefact.

## ROOT CAUSE OF THE ANCHOR BIAS — IDENTIFIED AND FIXABLE (2026-07-30)
The obs-fit builder estimates the HR spectrum from LR by dividing by a transfer function
T(k) = coarse/fine, measured on the Re=500/1000 references and used as a FIXED CONSTANT
(T_used = 0.00414, band-averaged over k=6..30).
T IS RE-DEPENDENT (measured, band-averaged):
  Re=500 0.00407 | Re=1000 0.00420 | Re=1500 0.00427 | Re=3000 0.00462 | Re=4000 0.00461 | Re=5000 0.00470
i.e. +15% from Re=500 to Re=5000. Physical mechanism: at higher Re more energy sits above the
coarse-grid Nyquist, so 4x subsampling aliases more of it back into the resolved band, raising
coarse/fine. Using a fixed low-Re T UNDER-estimates T for hot targets, so Ct/T_used OVER-estimates
the true spectrum => HOT anchor, error growing with Re.
QUANTITATIVE ACCOUNTING (predicted inflation = T_true/T_used vs measured anchor/GT in [6,30)):
  Re=1500  predict +3.1%  | measured 1.032  -> FULLY explained
  Re=3000  predict +11.6% | measured 1.213  -> about HALF explained (rest: shape priors / LOO?)
  Re=4000  predict +11.3% | not yet measured
  Re=5000  predict +13.5% | not yet measured
  Re=2000 (old gen) ran COLD 0.818 — opposite sign; that data is 256-native like the references so
  T applies correctly there => a DIFFERENT error source (priors/LOO), still unidentified.
THE FIX (GT-free, no new data): extrapolate T in Re exactly as kd already is. Fitting log T vs log Re
on the REFERENCES ONLY gives T ~ Re^0.046, which predicts the measured T at all four unseen regimes
to within 0.2-4.4% (1500: 0.2%, 3000: 4.4%, 4000: 3.0%, 5000: 3.9%). This would remove essentially
all of Re=1500's bias and most of Re=3000's — the exact defect that makes the set-point
non-portable. It also shows the GENERATOR mismatch (1024->256 vs 256-native) is minor: the
reference-derived trend crosses generators to within 4%.
NEXT: implement T(Re) in anchor_obsfit_builder.py, rebuild all anchors, re-measure anchor/GT bias,
then re-run the blind selection to test whether the set-point becomes portable.

### T(Re) — IMPLEMENTED BUT PARKED FOR RESEARCH (user decision, 2026-07-30)
Status: the current selection/grading runs FINISH ON THE LEGACY FIXED-T ANCHORS. T(Re) is opt-in
via `T_RE_SCALING=1` (build(..., t_re_scaling=True)); the DEFAULT IS LEGACY and was verified to
reproduce the in-use anchors bit-for-bit, so every result to date stays reproducible.
Research artifacts already built with T(Re) on, for the eventual comparison:
  regime_stats_re{1500,3000,4000,5000}_obsfit_tre.npz   (T factors 1.036/1.069/1.084/1.095)
When picking this up, the affected chain is: rebuild anchors -> re-measure anchor/GT bias ->
RE-DERIVE the set-point band (it is defined by healthy models' blind readings, which move with the
anchor) -> re-run the blind selection + pick grading. Do not flip the default without that whole
chain, or the band and the scores will be on different instruments.
SECOND, INDEPENDENT DEFECT (not yet addressed): the dt32-based anchors are fitted on only 16 LR
samples (4 seqs x 4 frames) vs 136 for the new-generation anchors — enough noise for ~10% amplitude
error, and the likely cause of Re=2000's cold reading (0.870 +- 0.083) which T(Re) does NOT explain
(wrong sign). Fix available: the anchor is a purely SPATIAL statistic, so it can be built from the
RAW fine-dt file's LR using all 320 frames/sequence (1280 samples) — still training-sequence LR
only, no GT. Note the stored fingerprint would then reference the raw file; verify_freshness would
need the same source to compare against.
MEASURED anchor/GT bias in [6,30), 5 disjoint groups (for reference when this resumes):
  re1500 1.032+-0.048 | re2000(old) 0.870+-0.083 | re3000 1.213+-0.052 | re10000(old) 1.313+-0.132

### NEXT RESEARCH PHASE — ANCHOR ACCURACY ACROSS Re, TUNED WITHOUT TARGET GT (user directive)
CONSTRAINT: every correction must be derivable from (a) the REFERENCE regimes' full data and
(b) the TARGET's low-res observations. Target GT may be used to REPORT accuracy after the fact,
never to fit. (The T(Re) law already satisfies this: it is fitted on Re=500/1000 and extrapolated;
target GT was used only to verify it predicts T to 4.4%.)
KNOWN ERROR SOURCES, in order of measured size:
 1. T(Re) — the coarse->fine transfer is Re-dependent (+15% Re=500..5000). Fix implemented, opt-in.
    Explains re1500's bias fully, ~half of re3000/re10000.
 2. Residual bias at high Re beyond T (~10% at re3000, ~20% at re10000) — untested cause. Prime
    suspects: the alpha/p shape priors are held FIXED at the reference average while the true
    spectral shape drifts with Re (kd is already extrapolated; alpha/p are not), and the LOO
    correction is interpolated over only 3 band knots.
 3. Small-sample anchor fits for dt32-based anchors: 16 LR samples vs 136. Fix: build from the RAW
    fine-dt file's LR (the anchor is a purely SPATIAL statistic, so dt is irrelevant) -> 1280
    samples. Likely explains re2000's cold reading, which T(Re) cannot (wrong sign).
GT-FREE VALIDATION STRATEGY: leave-one-reference-out. With only two references (500, 1000) the LOO
is barely determined — the extrapolation exponents rest on a 2-point fit. GENERATING ADDITIONAL
REFERENCE REGIMES (e.g. Re=750, 1250, 2500 with the new generator) would let every extrapolated
quantity (T, kd, alpha, p) be fitted on >2 points and validated by holding one out, entirely
without touching any deployment target's GT. This is probably the highest-value next data spend.
NOTE ON THE SET-POINT: it is defined by healthy models' blind readings, so it MOVES whenever the
anchor changes. Any anchor improvement requires re-deriving the band and re-running the selection
studies, or band and scores end up on different instruments.

## THE THREE-STAGE ADAPTATION LOOP — COMPLETE RESULT (2026-07-30)
Stage 1 frozen config (no adaptation) | Stage 2 GT-free config choice from LR only | Stage 3 grade
the choice. All grading: 640 triplets across 16 sequences, bootstrap-by-sequence. Legacy anchors
(no T(Re), no alpha/p tuning). Bar = [0.75,1.30].

regime  model      frozen -> ADAPTED (config)        place frozen->adapted   k* frozen->adapted
Re=1500 in-dist    0.679 -> 1.109  (K3->K4)          .877 -> .836            85 -> 95
Re=1500 Re=2000    1.097 -> (kept K3)                .739                    94
Re=1500 Re=10000   2.597 -> 1.209  (K3->K2)          .681 -> .782            95 -> 95
Re=3000 in-dist    0.463 -> 0.712  (K3->K4)          .844 -> .804            43 -> 66
Re=3000 Re=2000    0.693 -> 1.203  (K3->K4)          .734 -> .647            63 -> 72
Re=3000 Re=10000   1.491 -> (kept K3)  <- THE MISS   .670                    76
Re=4000 in-dist    0.401 -> 0.604  (K3->K4)          .823 -> .782            38 -> 58
Re=4000 Re=2000    0.579 -> 0.980  (K3->K4)          .716 -> .631            55 -> 65
Re=4000 Re=10000   1.216 -> (kept K3)                .651                    70
Re=5000 in-dist    0.365 -> 0.544  (K3->K4)          .806 -> .764            36 -> 53
Re=5000 Re=2000    0.524 -> 0.878  (K3->K4)          .694 -> .604            52 -> 62
Re=5000 Re=10000   1.075 -> (kept K3)                .642                    68

SCORECARD: ALL 8 changed picks IMPROVED |ret-1| (8/8). Of the 4 kept, 3 were already correct
(1.097, 1.216, 1.075) and 1 was the miss (Re=3000 Re=10000 kept 1.491 when K2 gives 0.831).
=> 11 of 12 decisions correct or improving, using ONLY target low-res data.
KEY RESULTS:
- ONE MODEL COVERS A 3.3x Re RANGE: the Re=2000 model, blind-adapted, gives 1.097/1.203/0.980/0.878
  at Re=1500/3000/4000/5000 — ALL inside the bar. Un-adapted it fails the bar at 3 of 4.
- DOSE CORRECTION WORKS BOTH WAYS: 2.597 -> 1.209 (de-dose) and 0.579 -> 0.980 (up-dose).
- REACH LIMIT IS REAL AND PREDICTED: the in-dist model's ladder never reached the band at
  Re=3000/4000/5000, and indeed adaptation left it at 0.712/0.604/0.544 — below the bar. The blind
  signal correctly said "retrain, don't retune" BEFORE any GT was seen. A deeper chain buys a
  roughly constant +0.20..0.43 of retention, enough at 1.5x the training Re, insufficient beyond.
- PLACEMENT IS THE PRICE, and it is systematic: every K3->K4 up-dose cost 0.040-0.090 of placement
  (.041/.087/.040/.085/.090/.042) while the single K3->K2 de-dose GAINED 0.101. Placement tracks
  dose inversely, in both directions, in every regime.
- THE HYBRID REPAIRS THE LOW-K DAMAGE: up-dosing drops low-k to 0.878-0.884; the kc=6 patch
  restores it to 0.984-0.990 with retention and placement unchanged. It matters most exactly where
  the up-dose is largest.

## ANCHOR LAW VALIDATION BY LEAVE-ONE-REFERENCE-OUT (2026-07-30) — GT-FREE
Method: measure the four extrapolated quantities on 8 regimes with HR data (500, 1000, 1500, 2000,
3000, 4000, 5000, 10000), fit each law on 7, predict the 8th. A target's own data never enters its
own prediction, so this is the honest GT-free test. src/ddpo_ft/anchor_law_loo.py.
MEASURED TRUTH:
  Re      500   1000   1500   2000   3000   4000   5000  10000
  alpha  2.101  1.940  2.004  2.150  1.949  1.831  1.842  2.024   <- NO Re trend
  p      1.425  1.455  1.622  2.233  1.899  1.918  2.024  6.000   <- 6.0 = fit hit its bound
  kd      23.3   33.6   47.7   51.8   73.9   80.0   89.5   89.2   <- SATURATES ~90
  T     .00407 .00421 .00432 .00422 .00454 .00465 .00471 .00460
LOO ERRORS (extrapolate vs hold fixed at the reference mean):
  T      2.5% median  vs  6.5% fixed   -> EXTRAPOLATE (2.6x better). CONFIRMS the T(Re) fix.
  alpha  6.1% median  vs  4.0% fixed   -> DO NOT extrapolate. My hypothesis that alpha/p drift with
                                          Re is REFUTED: alpha scatters around 1.98 with no trend,
                                          so a power-law fit adds noise. The current fixed prior is
                                          the better choice.
  p     29.3% median  vs 24.5% fixed   -> DO NOT extrapolate; p is also badly determined (the
                                          Re=10000 fit saturated at its bound).
  kd    15.8% median, 53.8% at Re=10000 (power law, the current rule)
GRID CUTOFF (why kd saturates): the steepest spectral fall sits at k~99-101 in EVERY regime tested
(Re=1000, 5000, 10000) — regime-independent, so it is a property of the 256^2 grid, not physics.
Beyond Re~5000 the true dissipation scale is not representable in these datasets, so the fitted kd
pins at ~90 while a power law keeps climbing (predicting 141 at Re=10000 vs 89.2 measured). That
over-broad kd prior makes the anchor too wide in k => the Re=10000 anchor's 31% hot bias.
SATURATING kd LAW: kd = kd_max*(1-exp(-(Re/R0)^q)), fitted kd_max=92.3, R0=2047, q=1.05.
  LOO: median 5.9% vs 15.8% power law; at Re=10000 28.8% vs 53.8%; fitted on all 8 it predicts
  91.9 at Re=10000 (truth 89.2) where the power law gives 141.2. NOTE kd_max is a property of THIS
  data resolution (256^2), not of the physics — it would move with grid size.
EFFECT ON THE ANCHOR (both fixes on, 5 disjoint 4-seq groups on each regime's own test sequences):
  anchor/GT [6,30):  Re1500 1.035->0.976 | Re3000 1.220->1.122 | Re4000 1.260->1.149 | Re5000 1.153->1.048
  anchor/GT [10,96): Re1500 1.125->1.093 | Re3000 1.231->1.165 | Re4000 1.238->1.154 | Re5000 1.128->1.034
  8 of 8 improved. Mean |bias| in the fitted band HALVES: 16.7% -> 8.6%.
  BUT the CROSS-REGIME SPREAD — the thing that actually breaks set-point portability — narrows only
  from 0.225 to 0.173 (~23%). Residual +12-15% bias remains at Re=3000/4000 from an unidentified
  source. So: a real improvement, NOT a solution to portability.
STATUS: both fixes are OPT-IN (T_RE_SCALING=1, KD_SAT=1); legacy remains the default and all
existing results stand on it. Turning them on requires re-deriving the set-point band (it is defined
by healthy models' blind readings, which move with the anchor) and re-running the selection studies.

## ROBUSTNESS OF THE LOO VERDICTS — three of four do NOT survive (2026-07-31)
(src/ddpo_ft/anchor_law_robustness.py; prompted by the user flagging the Re=10000 data as suspect)

WHAT "VALIDATED" MEANT, STATED PLAINLY. The four quantities were measured from the TRUE high-res
spectra of 8 datasets — that is ground truth, of the REFERENCE regimes. The test holds one regime
out, so a target's own GT never enters its own prediction. "GT-free" here means GT-free OF THE
TARGET, which is what the anchor procedure has always assumed (references are GT-calibrated: 2 of
them before, 8 now). Two leaks, declared: (a) the saturating FORM for kd was chosen after seeing
all 8 kd values including held-out ones — LOO refits parameters, not the form, so 5.9% is
optimistic; (b) the 16.7%->8.6% bias table is measured against TARGET GT — it is after-the-fact
verification that the fix helped, not a signal used to build it.

CHALLENGE 1 — refit everything with Re=10000 dropped:
  quantity | with Re=10000        | without Re=10000      | verdict
  T        | LOO 2.5% vs fix 6.5% | LOO 1.4% vs fix 4.1%  | SURVIVES (exponent +0.051 -> +0.065)
  alpha    | 6.1% vs 4.0% "don't" | 3.0% vs 4.2% "do"     | FLIPS — the refutation was one point
  p        | 29.3% vs 24.5%"don't"| 6.1% vs 24.2% "do"    | FLIPS — and p(Re=10000)=6.000 PINNED
                                                            at its fit bound; a broken fit, which
                                                            should have been caught regardless.
  kd       | sat 5.9% < power 11.6%| power 6.3% < sat 7.9%| FLIPS — the saturation IS that point;
                                                            sat kd_max moves 92.3 -> 134.8.
  => Only T(Re) is robust to the Re=10000 data being bad. The alpha/p refutation and the kd
  saturation law both rest on it. RETRACTED accordingly.

CHALLENGE 2 — the grid-cutoff argument, tested independently on all 8 regimes:
  steepest log-log fall:  Re500 k=123(-19) | 1000 k=103(-14) | 1500 k=103(-12) | 2000 k=101(-31)
                          3000 k=103(-10) | 4000 k=103(-9.5) | 5000 k=103(-8.9) | 10000 k=101(-50)
  The cliff LOCATION is regime-independent at k~101-103 (Re=500's 123 is noise floor: E(120)/E(10)
  = 3e-7). So a representable CEILING near k~102 is real and any extrapolation must respect it —
  but the cliff exists at Re=500 too, where kd=23, so it is NOT evidence of saturation ONSET. It
  bounds kd; it does not date the bend. The kd saturation law remains unsupported.

CHALLENGE 3 — the real confound: the reference library MIXES TWO GENERATION PIPELINES.
  real sims (kf_*_seed: Re 500/1000/2000/10000) vs fnons 1024->256 generated (Re 1500/3000/4000/5000).
  The cliff DEPTH splits by family, not by Re: generated files fall shallowly (-9 to -12, energy at
  k=120 up to 2.3e-3 of E(10)); real sims fall steeply (-14 to -50). Re=10000's tail (2.5e-4) is
  WEAKER than Re=5000's (2.3e-3) despite 2x the Re — physically backwards, independent corroboration
  that the Re=10000 file is off.
  Family offsets on the 8-point law residuals:  T 1.04x | alpha 0.94x | kd 1.24x | p 0.66x
  i.e. the two tail-SENSITIVE parameters carry a large family offset and the two well-behaved ones
  do not. Within-family fits are far cleaner (in-sample median |err|):
      real: T 0.7%  kd 5.9%  p 12.5%   (exponents +0.039 / +0.446 / +0.511)
      gen:  T 0.1%  kd 2.0%  p  1.2%   (exponents +0.073 / +0.522 / +0.179)
  gen-only kd LOO (3 fit / 1 predict): 18.9 / 7.1 / 2.3 / 4.3% — a plain power law, no saturation
  needed to Re=5000. CONCLUSION: the apparent kd saturation is the joint artefact of one suspect
  regime and a 1.24x family offset that lifts the generated regimes' kd, flattening the top of the
  trend. Even T's exponent is family-contaminated (+0.039 real vs +0.073 gen vs +0.051 blended) —
  a ~40% uncertainty on the SIZE of the T correction, though its sign and existence hold in both.

REVISED STATUS:
  - T_RE_SCALING: keep as a real effect; treat the exponent as uncertain to ~40% until the
    reference library is single-pipeline. Still opt-in.
  - KD_SAT: DO NOT ADOPT. Unsupported once Re=10000 is removed or the family offset is modelled.
  - Before any of these laws is trusted: either regenerate the reference regimes through ONE
    pipeline, or refit with an explicit family indicator so the Re trend is not contaminated.
  - p's bound-pinned fit at Re=10000 must be excluded or the bound raised, independent of all above.
