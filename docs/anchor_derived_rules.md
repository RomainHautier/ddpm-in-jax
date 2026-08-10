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

## THE LOO STUDY ASSUMED AN INFORMATION SET WE DO NOT HAVE (2026-07-31, user-caught)
The deployed anchor has TWO references: Re=500 and Re=1000. Those are the only regimes whose GT we
are entitled to, and EVERY anchor behind EVERY result was extrapolated from those two points. The
leave-one-out study measured the four quantities at EIGHT regimes and fitted each law on seven.
Holding the target out stops self-prediction, but it silently assumes GT-calibrated references at
1500/2000/3000/4000/5000/10000 — which we do not own. It therefore answers "would extrapolation work
with a seven-regime library?", NOT "how good is the anchor we deployed?". Framing it as GT-free
validation was an error. COROLLARY, and the user's second point: the extrapolated anchor was never
judged at the time — we extrapolated from two regimes and rolled with it. That is a fact about every
result already produced.

WHAT CAN HONESTLY BE MEASURED — an AUDIT (target GT used only to score, never to tune):
run the real 2-reference procedure, then compare to the target's truth after the fact.
  target |  kd pred -> true    err |  T err  | alpha err | p err
  1500   |  41.6 -> 47.7    -12.9% |  -4.1%  |   +0.8%   | -11.2%
  2000   |  48.3 -> 51.8     -6.7% |  -1.9%  |   -6.0%   | -35.5%
  3000   |  59.8 -> 73.9    -19.0% |  -8.8%  |   +3.7%   | -24.2%
  4000   |  69.6 -> 80.0    -13.0% | -11.0%  |  +10.3%   | -24.9%
  5000   |  78.3 -> 89.5    -12.5% | -12.1%  |   +9.7%   | -28.9%
  10000  | 112.7 -> 89.2    +26.4% | -10.0%  |   -0.2%   | -76.0%
  median |err|:  kd 12.9%   T 9.4%   alpha 4.9%   p 26.9%
THE ERROR IS SYSTEMATIC: the 2-ref law UNDER-predicts kd, T and p at every OOD regime, same sign
every time. Only Re=10000 flips kd to +26% (grid ceiling). So the deployed anchors were consistently
TOO NARROW IN K and TOO LOW IN THE OBSERVED BAND. Useful as a known-direction bias; NOT retrofittable
into the procedure that produced the existing results.
CAVEATS: (a) for 1500/3000/4000/5000 the "truth" is itself fnons-generated data, so the audit's
reference is generator-dependent (see the family confound above); (b) the 8-regime laws are not a
free improvement — they are a proposal to BUY GT at more reference regimes, a cost decision. Until
paid, the deployed procedure stays 2-reference, and both T_RE_SCALING and KD_SAT stay OFF.

## CROSS-REGIME DECOMPOSITION — frozen config, 7 regimes x 4 models (2026-07-31)
src/ddpo_ft/crossregime_decomposition.py. 640 triplets/16 seqs on the generated regimes, 70/35 on
the dt32 files; bootstrap BY SEQUENCE. VALIDATION: all 28 cells reproduce the previously published
transfer-test and max-n numbers digit for digit (where batch size matches — see the noise caveat).

1. THE PDE RESIDUAL IS A ONE-SIDED DIAGNOSTIC, NOT A CORRECTNESS CHECK.
   It must be read against the GT's own residual, which rises steeply with Re (15.9 -> 877.8 from
   Re=1000 to 10000). Among UNDER-restored fields it is nearly blind: at Re=3000 the in-dist model
   (ret 0.463) and the Re=2000 model (ret 0.693) both read 0.11x GT; at Re=4000 (ret 0.401 vs 0.579)
   both read 0.07x; at Re=5000 (0.365 vs 0.524) both read 0.05x. Three independent confirmations.
   It only speaks once a model reaches/exceeds the correct dose (Re=10000 model at Re=1000: 2.02x).
2. MODELS OVER-FILL THE BOTTOM OF THE REWARD BAND WHILE THE TOP COLLAPSES.
   Re=3000, Re=10000 model, energy/GT by band: 24-32 1.06 | 32-48 1.60 | 48-64 1.72 | 64-80 0.74 |
   80-96 0.24 | 96-128 0.07. Retention reads 1.491 because 32-64 dominates the integral. So
   "over-restoring" and "4x too smooth in the PDE sense" are simultaneously true: the reward band
   stops at k=96 and the physics is decided above it.
3. THE PLACEMENT CEILING IS NOT ABOUT DOSE MAGNITUDE — CORRECTED.
   Previously attributed to DDPO "removing the base's hedge". But at Re=10000 EVERY model collapses,
   including the in-dist one which restores almost nothing: base ret 0.211 place 0.703 | in-dist
   0.309/0.327 | Re=2000 0.465/0.249 | Re=10000 0.941/0.221. At Re<=5000 the FIRST increment of
   restored energy RAISES placement (Re=5000: base .746 -> in-dist .806) before further dose lowers
   it. Correct statement: below the ceiling the first increment is placed correctly; above it, even
   the first increment is misplaced. Temporal pre-flight passes on all files (Re=10000 lag-1 0.980,
   in band), so frame spacing does not explain it. Confounded with that dataset's known weak tail.
4. THE FROZEN RESIDUAL Re-LAW IS WRONG AND SHOULD NOT BE QUOTED.
   residual_ref = 10.218*(Re/1000)^2.705 vs measured GT floor:
     Re     1000   1500   2000   3000   4000   5000   10000
     meas   15.9   42.5   52.8  170.6  307.2  449.8   877.8
     law    10.2   30.6   66.6  199.5  434.4  794.5  5180.4
     err    -36%   -28%   +26%   +17%   +41%   +77%   +490%
   EMPIRICAL exponent is 1.824, not 2.705 (prefactor 18.9 at Re=1000) — and even that leaves 25%
   scatter, because the true floor SATURATES for the same 256^2 grid reason that caps kd. This is
   independent corroboration of the grid ceiling from a quantity not involved in the kd fit.
   Impact is limited: residual_ref feeds a one-sided hinge, so over-prediction never binds.
5. SAMPLING-NOISE SENSITIVITY. The same model/frames/config gave resid 12.6 (bs=8) vs 14.9 (bs=16)
   — an 18% swing from the noise draw alone — while ret moved 0.7% and place 0.1%. Bootstrap-over-
   sequences does NOT capture this. Residual differences below ~20% are not findings.

## TWO DIAGNOSTICS CLOSED (2026-07-31, both refuting earlier claims of mine)
SAMPLING NOISE IS NOT THE ISSUE; BATCH SIZE IS. 5 seeds, same model/frames/config (Re=2000, 70
triplets): ret sd/mean 0.24%, place 0.23%, resid 0.43%. So my attribution of the 12.6-vs-14.9
residual gap to "noise draw" was WRONG. Direct test: bs=8 -> resid 12.27, bs=16 -> 14.50 (18%),
while ret moves 1.156->1.164 (0.7%); chunking the residual call makes NO difference (0.0000).
Cause is almost certainly XLA picking different layouts per batch shape, amplified by the residual's
derivative operators. RULE: residuals compare only at MATCHED BATCH SIZE. The whole decomposition is
bs=16 and internally consistent; maxn_regrade's 12.6 was bs=8.

THE "Re=2000 CALIBRATION FAILURE" WAS MY POOL BUG — RETRACTED. The blind score must be computed on
the ANCHOR'S OWN SOURCE SEQUENCES (fixed denominator => it does not cancel and is pool-dependent).
I scored it on the test pool. Corrected (src/ddpo_ft/blind_picks_source_pool.py):
  base recon @ Re=2000 source pool = 0.672 (NOT the 0.846 I reported) — well below the band.
  Re=2000  in-dist  -> K4 blind 0.822 IN BAND: ret 0.750 -> 1.213  (improved)
  Re=2000  Re=10000 -> K2 blind 0.835 IN BAND: ret 2.672 -> 1.282  (large correction)
  Re=10000 Re=2000  -> K4 blind 0.816 IN BAND: ret 0.465 -> 0.797  (BEST rung of six)
  Re=10000 in-dist  -> K4 blind 0.639 BELOW band: ret 0.309 -> 0.467 (correctly flagged out of reach)
4/4 correct or improving, bidirectional. The cross-regime non-calibration (0.866@Re3000 -> 1.491 true
vs 0.873@Re5000 -> 1.075 true) is unaffected: that comparison uses no pool choice.

## POTENTIAL WORK — ranked, with what each buys and what it costs (2026-07-31)

1. REGENERATE Re=2000 AND Re=10000 AT record-dt = 1/32.  [highest value]
   WHY: both models were trained on EIGHT triplets. The 2026-07 40-seed files record ~80x finer than
   the training convention, so the only dt-correct derivative is a stride-80 subsample yielding 2
   triplets/sequence (70 total, the evaluation ceiling too). A correct generation gives ~318
   triplets/sequence, i.e. ~1272 for a 4-sequence training pool — a 160x larger pool — and lifts the
   evaluation ceiling from 70 triplets to the same 640 the generated regimes enjoy.
   NOTE ON "dt=1/32 IS UNSTABLE": record-dt and solver dt are DIFFERENT things. record-dt=0.03125 is
   only the sampling interval and was ALREADY used for the Re=1500/3000/4000/5000 generations
   (gen_re4000.log/gen_re5000.log). The solver dt is separate and small (Re=4000 -> 2.5e-5,
   Re=5000 -> 2e-5, i.e. ~0.1/Re); the integrator is Crank-Nicolson on the viscous term with EXPLICIT
   advection, so the limit is CFL on the nonlinear term, not the recording rate. Stability probe in
   progress for Re=2000 (5e-5) and Re=10000 (1e-5 and 2e-5).
   COST: the Re=5000 run was 3h51m wall for 20 trajectories x 320 frames at 1024^2 on 4 devices.
   Re=10000 needs roughly half the timestep, so budget ~8h per 20-trajectory set.
   BLOCKER: user confirmation of the Re=10000 record-dt (measurement favours stride 100 over the 80
   currently used — suggestive at ~2.5x reference scatter, not conclusive).

2. SINGLE-PIPELINE REFERENCE LIBRARY.
   WHY: Re 500/1000/2000/10000 are direct simulations; Re 1500/3000/4000/5000 come from the
   1024->256 generator. The tail-sensitive fit parameters carry a generator offset (kd 1.24x,
   p 0.66x) while T and alpha do not, so ANY law fitted across the mixed set is confounded — this is
   what killed the kd saturation result. Regenerating the direct-sim regimes through the same
   pipeline (or adding a family term to the fits) is a precondition for trusting any extrapolation law.

3. MORE GT-CALIBRATED REFERENCE REGIMES — a cost decision, not a free win.
   The anchor extrapolates from TWO references. Improving it at all requires buying high-resolution
   ground truth at more regimes. The leave-one-out study showed the laws work well GIVEN seven
   references; we own two. Nothing about the anchor's accuracy improves until that is paid for.

4. SMALLER / DEFERRED:
   - The frozen residual Re-law is wrong (empirical exponent 1.824 vs 2.705, errors -36% to +490%);
     harmless where it is used (one-sided hinge) but should be refitted or removed.
   - Residuals are only comparable at MATCHED BATCH SIZE (bs8 12.27 vs bs16 14.50 on identical input).
     Pin the batch size in any harness that reports them.
   - Placement's abrupt collapse between Re=5000 and Re=10000 is established but its mechanism is
     confounded with that dataset's weak tail; item 1 would disambiguate it.

## THE GENERATOR CONFOUND, NOW MEASURED DIRECTLY (2026-07-31)
Regenerating Re=2000 through the fnons 1024->256 pipeline puts the SAME Reynolds number in BOTH
generation families, so the generator offset can be measured at fixed Re instead of inferred from
fit residuals across different regimes. Controlled comparison:
                          alpha      p     kd        T   cliff k  slope
  generated (1024->256)   1.953  1.684   56.0  0.00438      103   -11.4
  direct sim (40seed)     2.001  1.957   45.0  0.00422      101   -29.3
  ratio generated/direct  0.976  0.861  1.242    1.037
Compare the values inferred ACROSS regimes from the 8-point fit residuals: kd 1.24x, T 1.04x,
alpha 0.94x, p 0.66x. **kd and T match to within 1%.** The confound is therefore real, generator-
driven, and of the size previously estimated — it is not an artefact of the residual analysis. Any
law fitted across the mixed library carries a ~24% kd bias, which is precisely what corrupted the
kd saturation fit.
Also note the cliff LOCATION is identical (103 vs 101 — the 256^2 grid) while its DEPTH differs by
2.6x (-11.4 vs -29.3). Location is the grid; depth is the generator.
CONSEQUENCE FOR EXISTING Re=2000 RESULTS: the regenerated flow carries 3.0x the energy above k=64
and 8.4x above k=96 relative to the old 40seed file. It is a statistically different flow, so every
Re=2000 number in this project is tied to the file it was measured on and must be re-established on
the new data. The 1024-DNS generation resolves the small scales before downsampling, so it is the
more faithful of the two at high k.

## SINGLE-PIPELINE REFERENCE FAMILY — the generator confound is now REMOVED (2026-08-06)
Re=6000/7000/8000 generated (same fnons convention as 1500-5000 and the regenerated 2000/10000),
giving NINE regimes from ONE generator. Every file passed the temporal pre-flight (lag-1 0.953-0.971,
no monotone Re-trend — my prediction that it would fall below 0.95 at Re=8000 was WRONG, the scatter
between regimes is ~+-0.01) and every tail is monotone in Re. Cliff at k=103 in all nine (grid).
MEASURED (base_results/family_params_singlepipeline.npz):
  Re     1500   2000   3000   4000   5000   6000   7000   8000  10000
  alpha 2.004  1.953  1.949  1.831  1.842  1.829  1.812  1.794  1.758
  p     1.622  1.684  1.899  1.918  2.024  2.150  2.396  2.531  2.680
  kd     47.7   56.0   73.9   80.0   89.5   97.9  106.5  112.5  120.1
  T    .00432 .00438 .00454 .00465 .00471 .00478 .00487 .00491 .00500

1. kd DOES NOT SATURATE. A clean power law fits all nine to within +-6% (exponent 0.491; per-regime
   errors +3/+2/-6/-0/-0/-0/-1/-0/+4%). kd reaches 120.1 at Re=10000 — well ABOVE the 92 ceiling the
   old mixed-library fit produced, and above the k~103 cliff. The apparent saturation was ENTIRELY
   the generator confound plus the defective old Re=10000 file, exactly as the retraction concluded.
   The grid ceiling is real for the CLIFF LOCATION (k=103 everywhere) but does NOT cap the fitted kd.
2. THE 2-REFERENCE LAW's EXPONENT IS CLOSE: 0.526 (from Re=500/1000) vs 0.491 measured over nine
   regimes — 7% high, which is why it over-predicts at the top (112.7 vs a true ~120 at Re=10000 is
   now only -6%, not the +26% the defective old file implied).
3. alpha and p ARE NOT CONSTANT after all. Over a clean family both drift monotonically: alpha
   2.004 -> 1.758 (-12%), p 1.622 -> 2.680 (+65%). The earlier "no Re trend" verdict was measured on
   the mixed library where the generator offset (alpha 0.976x, p 0.861x) masked the trend. This does
   NOT resurrect the old extrapolation attempt - it means the priors should be refitted on the clean
   family before any further claim.
STATUS: this supersedes the mixed-library law analysis for the generated regimes. The 2-reference
DEPLOYMENT procedure is unchanged (we still own GT at Re=500/1000 only); what changed is that the
research-time reference library is now internally consistent.

## R6 SET-POINT RE-CALIBRATED NON-CIRCULARLY AT Re=1000 (2026-08-06)
The band [0.798, 0.858] was taken from the two OOD models' OWN blind readings — calibrated on the
models it judges. Replaced here by a reading from a model whose health is verified against real GT.
Run: 8 seqs (20-27, 2544 triplets, ~48 independent states) vs the old 2 seqs (~12 states); obs-fit
anchor (kd fit 33.9 vs measured 33.6); temp 2.0 from R3.1 blind (D=0.425); NO early stop, 600 iters.

VAL (seqs 32-35) RETENTION BY CHECKPOINT, and the blind score on the TRAIN pool:
  iter    49    99   149   199   249   299   349   399   449   499   549   599
  val ret .774  .866 .949  .912  .895  .905  .941  .900  .992 1.019 1.099 1.024
  place   .897  .891 .895  .893  .886  .863  .880  .875  .866  .874  .871  .879
  blind   .760  .750 .801  .786  .774  .775  .778  .754  .768  .782  .789  .785
  (base, no finetune: val ret 0.450, place 0.911, k*=35, blind 0.682)

1. SET-POINT = 0.768 (blind score at the GT optimum, iter 449, val ret 0.992).
   The inherited band [0.798, 0.858] does NOT contain it. Its lower edge 0.798 corresponds to
   iter 149 — val ret 0.949 and 300 iterations EARLIER than the GT optimum. So R6 as configured
   stops early. The whole blind trajectory spans only 0.750-0.801, i.e. the signal's DYNAMIC RANGE
   (0.05) is barely twice its noise (+-0.02) — a band of width 0.06 is nearly the entire range.
2. TRAIN AND VAL OPTIMA DIFFER: train 149 (ret 1.006) vs val 449 (ret 0.992), 300 iterations apart.
   Grading only on the training pool would have stopped at 149. This is why both were graded.
3. R3.2's BLIND PICK = iter 149, val ret 0.949 — NOT the GT optimum (449, 0.992). It is not a
   disaster (0.949 is inside any reasonable bar) but the rule systematically picks early.
4. TRAINING PAST THE OPTIMUM IS NOT FREE: placement decays 0.897 -> 0.863-0.879 while retention
   overshoots to 1.099 by iter 549. The base's placement is 0.911; every finetuned checkpoint is
   below it. Over-training trades structure for excess energy.
5. THE POOL WORKS: k* jumps 35 -> 95 by iter 49 and stays; low-k stays 0.96-0.99. Retention reaches
   0.992 vs the old in-dist run's 0.893-1.023 band across four temps. No sign the 48-state pool
   overfits in the damaging sense — but train/val optima differing by 300 iters shows the training
   pool alone cannot locate the optimum.
CONSEQUENCE FOR THE OOD RUNS: do NOT reuse [0.798, 0.858]. Either (a) target 0.768 with a tight
band, accepting that the blind signal's range is only ~2x its noise, or (b) drop early stopping,
run 600, and select post-hoc by R3.2 — which lands at 0.949 val retention, i.e. a known ~5% cost.

## SECOND SET-POINT MEASUREMENT AT Re=500 — IT IS NOT A CONSTANT (2026-08-07)
Re=500 is the only other regime whose GT we own, and it had never been finetuned. Same protocol as
the Re=1000 calibration (8 train seqs 0-7, obs-fit anchor, temp from R3.1 blind, no early stop, 600
iters, GT probe OFF). Splits train 0-7 | val 8-13 | test 14-19; inter-seq corr 0.096 (independent).

  R3.1: D = 0.7346 -> temp 2.0.  (Re=1000 0.425, Re=2000 0.326, Re=10000 0.191 — monotone in Re.)
  VAL RETENTION BY CHECKPOINT, with the blind score on the train pool:
    ckpt   base   49    99   149   199   249   299   349   399   449   499   549   599
    ret   0.739 0.707 0.677 0.660 0.717 0.630 0.651 0.660 0.704 0.744 0.692 0.684 0.659
    blind 0.806 0.815 0.824 0.821 0.839 0.832 0.826 0.818 0.837 0.850 0.831 0.825 0.825
    place 0.903 0.893 0.886 0.885 0.890 0.883 0.861 0.876 0.879 0.872 0.875 0.879 0.864

1. THE SET-POINT IS REGIME-DEPENDENT. Blind score at the GT optimum:
       Re=1000 -> 0.768        Re=500 -> 0.850
   A gap of 0.082, about 4x the blind signal's noise (+-0.02) and larger than the whole [0.798,
   0.858] band. A single frozen set-point applied across regimes is therefore WRONG, and the n=1
   value 0.768 must not be carried to the OOD runs as if it were a constant. Two points cannot give
   a law; they establish that a constant is not it.
2. AT Re=500, FINETUNING BARELY HELPS AND MOSTLY HURTS. The un-finetuned base already reads val ret
   0.739, k*=95 (FULL effective resolution), place 0.903, lowk 0.981 — because Re=500 is LESS
   turbulent than the base's training regime, so nothing in it is beyond the base's reach. The best
   checkpoint (449) reaches ret 0.744 — a 0.005 gain on retention — while LOSING placement
   (0.903 -> 0.872) and low-k (0.981 -> 0.949). 10 of 12 checkpoints are worse than the base on
   retention. Net: DDPO is not worth running at Re=500.
3. WHAT WENT RIGHT: R3.2's blind pick MATCHED the GT optimum exactly (both iter 449), and train and
   val optima AGREE (both 449) — unlike Re=1000 where they differed by 300 iterations. So the
   selection machinery works here; it is the fixed TARGET VALUE that does not transfer.
4. RETRACTED (mine, same day): from three early checkpoints I called a monotone decline "8 sigma,
   not noise" and an anti-correlation between blind score and truth. The 4th point broke it — the
   trajectory oscillates by ~0.05 with no trend. The durable statement is only that every finetuned
   checkpoint except 449 is worse than the base.
CAVEAT ON BOTH SET-POINTS: the Re=500 and Re=1000 anchors each use their own regime as one of the
two references (REFS = {500, 1000}), so both are partly self-referential and likely optimistic
relative to a genuine OOD anchor. Unavoidable with two GT regimes; declared, not hidden.

### RETRACTION (same day, before acting on it): the Re=500 set-point above is DEGENERATE
The set-point is DEFINED as the blind score of a model GROUND TRUTH says is healthy (ret ~ 1.0).
  Re=1000: base ret 0.450 -> best 0.992.  HEALTHY. 0.768 is a valid set-point.
  Re=500 : base ret 0.739 -> best 0.744.  NOT healthy (|ret-1| = 0.256). 0.850 is the blind score
           of an UNHEALTHY model, not a set-point.
Comparing 0.850 with 0.768 compares two different quantities. **"The set-point is regime-dependent"
is NOT established** and the previous entry is withdrawn on that point. What IS established stands:
at Re=500 the base already reaches k*=95 / ret 0.739 and 10 of 12 finetuned checkpoints are worse.

WHY Re=500 NEVER GETS HEALTHY — and it is the design flaw, not the regime:
every checkpoint was graded at ONE FIXED CASCADE (K3[150,100,50]x86). The cascade sets the ENERGY
DOSE; the checkpoint modulates it. At Re=500 that cascade delivers ~0.74 of the required energy
whatever the weights, so no checkpoint can be healthy and the set-point cannot be read. The
experiment varied the weaker variable and froze the dominant one.
CORRECT EXPERIMENT: sweep CASCADE x CHECKPOINT at Re=500, find whether any combination reaches
ret ~ 1.0, and read the blind score there. Only then is there a second set-point to compare — and
that sweep is also exactly the OOD inference-selection question, so it is needed either way.

## CASCADE x CHECKPOINT SWEEP — 84 cells at the two GT regimes (2026-08-07)
src/ddpo_ft/cascade_ckpt_sweep.py. 6 cascades x 7 models x 2 regimes; GT graded on held-out val
(Re=500 seqs 8-13, Re=1000 seqs 32-35), blind score on each anchor's own source pool. Test seqs
untouched. Motivation: every earlier grading FROZE the cascade and varied only the checkpoint, i.e.
froze the dominant variable and swept the weak one.

1. THE CASCADE DOMINATES THE CHECKPOINT.
     Re=500 : spread across cascades 0.296 vs across checkpoints 0.082  (3.6x)
     Re=1000: spread across cascades 1.019 vs across checkpoints 0.446  (2.3x)
   And the cascade is FREE at inference; a checkpoint costs a ~4 h training run. The project's
   effort has gone into blind CHECKPOINT selection (R3.2, R6, early stopping) when blind CASCADE
   selection is the larger term.
2. THE TWO REGIMES WANT OPPOSITE CASCADES — no single frozen cascade can serve both.
     Re=500  best = K4x110 (the DEEPEST rung) + ckpt449 -> ret 0.924
     Re=1000 best = K3x86               + ckpt449 -> ret 0.999
   At Re=1000 the K4 row overshoots on EVERY trained checkpoint (1.146 -> 1.937). At Re=500 K3 tops
   out at 0.750. This is exactly why the frozen K3x86 produced a degenerate Re=500 measurement.
3. THE SET-POINT IS REGIME-DEPENDENT — now measured like-for-like at each regime's best cell:
     Re=1000 blind 0.752 (at ret 0.999)      Re=500 blind 0.911 (at ret 0.924)
   Gap 0.159, ~8x the blind signal's noise. CONSEQUENCE, measured directly: applying Re=1000's
   set-point 0.752 at Re=500 selects K1-50x12|base -> true ret 0.536, i.e. the SHALLOWEST cascade
   and the worst usable cell, when 0.924 was available. A transferred absolute set-point is not
   merely imprecise, it is actively harmful.
   CAVEAT: Re=500's optimum sits at K4, the LADDER EDGE, and reaches only 0.924. The ladder may be
   truncated; a deeper rung might carry Re=500 to ~1.0 and move its blind reading. Until the optimum
   falls in the INTERIOR the 0.159 gap is strong but not airtight.
4. WHAT THE BLIND SCORE IS GOOD FOR: within a regime it RANKS well — correlation with true retention
   across all 42 cells is +0.985 (Re=1000) and +0.834 (Re=500). It is a good relative instrument and
   a bad absolute one, which is the same conclusion the OOD work reached, now on 84 controlled cells.
5. FINETUNING PAYS VERY DIFFERENTLY BY REGIME, and D predicts it before training:
     Re=500  (D=0.735): base beats every checkpoint on 4 of 6 rungs; best gain +0.040 over base.
     Re=1000 (D=0.425): K3 goes 0.452 (base) -> 0.999 (ckpt449).
   At regimes like Re=500 the right move may be "pick the cascade, skip the finetune" — 0.884 free.

## LADDER EXTENSION K5-K7 — the K4 optimum WAS a truncation artefact (2026-08-07)
src/ddpo_ft/cascade_deep_sweep.py. 3 deeper rungs x 7 models x 2 regimes = 42 cells, on top of the
84 already run. Total grid: 126 cells.

1. TRUNCATION CONFIRMED AND FIXED. Re=500 base retention is monotone in cascade depth and crosses
   1.0 between K5 and K6:
     K1-50 0.536 | K1-75 0.551 | K1-100 0.571 | K2 0.635 | K3 0.744 | K4 0.884 | K5 1.012 | K6 1.108 | K7 1.154
   The K4 "optimum" (0.924) sat on the boundary of the old ladder. The true interior optimum is K5.
2. AT Re=500, DDPO IS UNNECESSARY. The best cell on ALL THREE metrics is the UN-FINETUNED BASE at
   K5x140: ret 1.012, place 0.869, lowk 0.963. Every finetuned checkpoint at that rung is worse on
   all three. The right inference depth alone solves the regime; the 4 h training run buys nothing.
   (D=0.735 predicted this before training: the base already delivered 73% of the anchor's demand.)
3. RETENTION ALONE CANNOT DEFINE "HEALTHY" — and the set-point definition depended on it.
   At Re=500, 12 of 63 cells are within 10% of ret=1.0; they span placement 0.869 down to 0.575 and
   k* from 95 to 1. One cell reads ret 1.052 with k*=1 — total band energy correct, spectrum wrong
   at every wavenumber. Selecting on retention alone picks K6x170|0549 (place 0.737); a multi-metric
   criterion (ret within 10% AND max placement) picks K5x140|base (place 0.869).
   THE SET-POINT MOVES WITH THE CRITERION:
     Re=500 : 0.946 (retention-only)  vs  0.835 (multi-metric)
     Re=1000: 0.752 (retention-only)  vs  0.784 (multi-metric)
   Retention-only gap = 0.194; multi-metric gap = 0.051. The apparent regime-dependence is largely
   an artefact of an under-specified health criterion. With a multi-metric definition the two
   set-points are 0.835 and 0.784 — a gap of ~2.5x the noise, far from the 0.159 claimed earlier.
4. THE Re=1000 CONTROL BEHAVED AS PREDICTED: deeper rungs overshoot monotonically (K5 up to 3.0,
   K6 to 4.2, K7 to 5.35). Its optimum stays at K3, interior. The dose picture holds.
5. THE BLIND SCORE RANKS WELL OVER THE FULL GRID: r = +0.991 (Re=1000, 63 cells) and +0.858
   (Re=500). Confirms once more: good relative instrument, poor absolute one.

## PLACEMENT PROXY — a GT-free structural axis, tested on 42 re-run cells (2026-08-08)
src/ddpo_ft/placement_proxy_test.py. proxy = corr(model's local hi-k energy map, BASE
RECONSTRUCTION's map). The base recon exists at every regime from LR alone, so the proxy is
computable at OOD; true placement (corr with the GT map) grades it here and never feeds it.
42 cells re-run with the sweep's exact seeds: Re=500 x {K3,K5,K7}, Re=1000 x {K1-100,K3,K5}, all 7
models each. Reference context: raw-recon map vs truth = 0.695 (Re=500) / 0.698 (Re=1000).

VERDICT: Re=500 r=+0.959 (range 0.642-0.904) | Re=1000 r=+0.738 (range 0.716-0.896) | pooled +0.783.
READ HONESTLY:
- The proxy tracks true placement WELL across cascade rungs (the dose axis) — it cleanly separates
  K3-level structure (~0.88 true) from K7-level damage (~0.67 true) at both regimes.
- WITHIN a rung it is noisy: at Re=1000/K1-100 the proxy ordering partly inverts the true ordering
  (range 0.87-0.90 true vs 0.87-0.90 proxy but shuffled). The lower Re=1000 r is partly range
  restriction (no rung there craters placement) plus this within-rung noise.
- One systematic bias, expected and observed: SHALLOW cascades score high on the proxy partly
  because their output stays close to the recon (K1-100 proxies 0.87-0.90 exceed the 0.698
  "ceiling" — the model inherits the recon's structure rather than matching truth better).
  The proxy therefore REWARDS conservatism; it must be used as a GUARD (flag structural damage),
  never as a maximisation target, or it will always prefer the shallowest rung.
ROLE IN THE RULE: blind anchor score = dose axis (pick the rung nearest the set-point); proxy =
structure guard (among near-set-point candidates, reject cells whose proxy has collapsed relative
to the shallow rungs). Both GT-free at the target.
