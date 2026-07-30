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
