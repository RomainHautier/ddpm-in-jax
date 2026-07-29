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
