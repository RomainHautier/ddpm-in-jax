# DDPO — open backlog

Tracked follow-ups from the DDPO physics-reward investigation. See `ddpo_findings.md` (narrative),
`ddpo_report.pdf` (paper), and the results artifact for the completed work.

---

## 1. Make the PDE residual go DOWN toward the GT floor (priority)

**Goal:** get the finetuned model's NS residual to move *toward* the GT residual floor, instead of the
current flat-to-slightly-up behavior.

**State (2026-07-09).** At grid-4× the base reconstruction is already near-physical (mean |resid| ~3.8
vs GT floor ~1.5). DDPO's spectral reward *adds* high-k energy, which sharpens filaments and *raises* the
residual slightly (base 3.8 → DDPO 3.9; K=3 → 4.2). The recon residual field is bright and **speckled
everywhere**, whereas GT's residual is dark/sparse/clean on a few coherent filaments
(`monitoring/ab_pdelocal/viz_residual_k1k3.png`). **PDE-residual placement** stays ~0.36 — below the ~0.6
"useful" line — at every resolution reached; it is a fine-scale derivative field that aligns with GT far
more slowly than energy does. The `pde_local` reward (worst-region residual) was a controlled **negative**.

**Why it matters.** Energy/spectrum is solved at grid-4× (retention +0.21, k* 31→95, placement 0.81, MSE
flat), but physics-consistency at the finest scales is the one unsolved axis.

**KEY FINDING (2026-07-09, `diag_residual_landscape`).** GT is the PDE-residual **minimum**: GT residual is
lowest at full resolution (1.31) and *rises* under any smoothing (up to 4.78) — smoothing breaks the NS
balance. So **reducing residual tends TOWARD GT, not toward an over-smoothed solution** (the earlier
over-smoothing worry is ruled out; the reward premise is sound). The recon's excess residual (3.65 vs GT
1.31) is **high-k speckle**: the residual power spectrum shows base/DDPO above GT *only* at k>32, and
low-passing k>64 drops recon residual to ~2.1. Interpretation: DDPO adds high-k *energy* with high-k
*residual* — right amount, wrong dynamics (phase). GT has high-k energy AND low high-k residual because its
fine structure is physical.

**The reward to build:** a **spectral-residual term** that penalizes residual POWER at k>32 (the speckle) —
driving the added structure toward NS-consistent/physical (toward GT) — WITHOUT smoothing away the high-k
*energy* (a naive smoothness prior kills the retention DDPO worked for and gets stuck at residual ~2.1).
The two must be separated: keep the enstrophy spectrum, kill the residual spectrum at high k.

**Other ideas (untested):**
- **Two-sided / stronger pde term** so the reward pushes residual *toward* the floor (current hinge only
  penalizes *above* the floor → tolerates the speckle). Now known safe since GT is the residual minimum.
- **Gentler multi-phase (K=2) / lower sampling temperature** — K=3 over-sharpens and raises residual.
- **Denser input** (>11% sampling) where PDE placement finally climbs — the information-limited fallback.

---

## 2. Multi-phase K=2 / lower-temp test

K=3 helps placement but over-amplifies in-distribution (harmful) and is useful only OOD (capacity-starved).
Test K=2 (S=[100,50]) or lower temperature to capture the placement/sharpness benefit without the amplitude
overshoot — turning the "K=3 backfires" caveat into a usable in-dist lever. (`eval_multiphase`,
`diag_multiphase_spatial` already parametrized.)

## 3. Conditioned-model ("learned head") comparison

All results use the **unconditional** base + SDEdit + DDPO. Compare against a model with a learned
conditioning head (`config_field_cond.yaml`, `check_conditioning_training.ipynb`) at grid-4×: does an
explicit conditioning path beat SDEdit for reconstruction, and does DDPO still add on top?

## 4. Pure zero-lower-regime extrapolation

The Re=2000 extrapolation uses anchors from {Re=500, 1000}. Push further: extrapolate to a regime with
*no* bracketing lower-Re data, or extrapolate the `align` reference (currently the Re=1000 value) too.

## 5. Loose ends
- Re=1000 grid-4× enstrophy-vs-k spectrum with the input curve (`eval_multiphase --re 1000 --grid_factor 4`)
  — only the Re=2000 one was generated.
- Resolution sweep at intermediate densities (2048, 3072 pts) to pin where placement crosses 0.6 more
  precisely (currently bracketed to ~3000).

---

## Final guided-eval matrix (2026-07-10) — base/finetuned x unguided/guided, both regimes

Full val+test at grid-4x (`eval_guided_full`). x0-guidance lambda=3.

**Re=1000** (GT residual 1.06): base 0.396/3.36 -> DDPO 0.607/3.44 (retention +0.21, k* 31->95,
residual +0.08). Guidance lambda=3: residual -3.6% on both (DDPO 3.44->3.32), retention/MSE/placement
flat. Best = DDPO+lam3: ret 0.597, resid 3.32 (~=base), k* 95.

**Re=2000 OOD** (extrapolated anchor, zero target GT; GT residual 3.01): base 0.275/3.81 -> DDPO
0.350/4.07 (retention +0.075, placement 0.838->0.851). Guidance lambda=3: residual -7% on both
(DDPO 4.07->3.77 < base 3.81), everything else flat. GENERALIZES OOD (analytic physics).

**Two additive levers:** DDPO = energy (retention/k*/placement), x0-guidance(lam=3) = residual
(-3.5..7%, free, OOD-safe). Combined DDPO+lam3 beats base on every moving metric at both regimes.
CEILING: residual still 3x GT in-dist (3.3 vs 1.06), 1.25x OOD (3.8 vs 3.01) — nudge not closure;
temporal-consistency gap needs a structural fix (temporally-coupled training), not reward/guidance.

---

## Base-DDIM-init finetune (2026-07-13) — start DDPO from the base reconstruction, not raw low-res

**Setup.** `--base_ddim_init`: input pool (and probe) pre-denoised ONCE by the FROZEN base via
deterministic DDIM (SDEdit t=100 -> 20 steps, eta=0, single chain, Hu et al-style); the policy then
noises/denoises THAT. Eval applies the same transform (`eval_guided_full --base_ddim_init`).
Re=1000: 300 iters, standard config. Re=2000: 200 iters, fully GT-free (extrap anchor + extrap
residual floor 26.29, scales_re=1000, align 2.0 = Re=1000 value).

**Training probes** (seq-local, optimistic): Re=1000 0.421 -> 0.661 final (peak 0.669, passed the
raw-init final 0.607 at iter 69 — ~3x faster). Re=2000 0.290 -> 0.381.

**Full val+test matrix at matched iter0199** (vs raw-init baseline rows):
- Re=1000: DDPO ret 0.585 (vs 0.607 — PAR, probe gain didn't survive averaging), placement
  **0.862 vs 0.825** (+0.04, also on the base row), MSE/resid par (0.0170/3.49).
- Re=2000 OOD: DDPO ret **0.361 vs 0.350**, placement **0.868 vs 0.851**, resid 4.01 vs 4.07,
  MSE par. +lam3: ret 0.354, resid **3.74** (< base-unguided 3.84), placement 0.865.

**Verdict:** the durable win is PLACEMENT (+0.02..0.04 both regimes, both model rows) plus a small
OOD retention/residual edge — the base-DDIM reconstruction is a cleaner spatial prior, so added
energy lands more accurately. In-dist retention is par at matched iters (train probe overstated it).
Best OOD combo so far: ddiminit DDPO + lam3. Cost: one extra 20-step DDIM pass per input.

**t_start=50 intervention (2026-07-13) — REFUTES the "preserve start-state signal" implication.**
Renoise study said t=50 keeps 93% of the DDIM recon's placement signal (vs 59% at t=100) -> ran the
t=50 ddim-init finetune (300 iters, 21.5s/iter vs 43). Result: probe 0.411 -> 0.507 (vs t=100's
0.661); reward plateaued ~-10.4 vs -7.9, spec_highk STUCK at ~1.2 (t=100 drove it to 0.45).
`diag_group_hik_std` (frozen base, groups of 8, seqs 32/36/38): within-group CV of E(k>=32) scales
~linearly with t_start (1.14% @50, 2.13% @75, 3.04% @100) — at t=50 rollout groups are near-clones
in the reward band -> no advantage signal (USER HYPOTHESIS 2 CONFIRMED). temp 2.5 does NOT fix it:
absolute spread doubles but CV flat — the group shifts TOGETHER (hot-sampling speckle), no
differential signal (HYPOTHESIS 1 REFUTED). Eval matrix (t=50 recipe): DDPO ret 0.481, place 0.818,
MSE 0.0156 — placement is WORSE than t=100's 0.862 despite the 93% signal survival, and even base
placement rises with chain depth (recon clean 0.745 -> base@t50 0.823 -> base@t100 0.862): final
placement is CHAIN-GENERATED (denoising manifold work), not start-inherited. Start-state signal
survival is the wrong optimization target. VERDICT: keep t_start=100; t=75 unpromising (trades
exploration for signal that doesn't matter); t=50 only as a conservative low-MSE point (0.0156).

**K-chain DDIM-policy training (2026-07-13) — THE RESIDUAL FINALLY MOVES.** Policy = stochastic-DDIM
(eta=1, ~50-step budget) K=2 cascade S=[100->75], renoise between chains theta-free
(`build_ddim_rollout`); 300 iters from raw-LR and from ddim-init inputs. Rewards reach -4.1/-3.6 (vs
-7.9 best DDPM-policy). Matched eval (`eval_guided_full --sampler ddim --chain_starts 100,75`):
**PDE residual 3.3-3.5 -> 1.61-1.73 (GT floor 1.06)** — most of it from the inference process itself
(base rows too: coarse chain + final deterministic x0-prediction = conditional mean, suppresses the
residual-carrying sampled texture; caveat: k* 95 -> 36-40, some fine-scale sample realism traded).
DDPO adds ret +0.11-0.12 at record placement (0.883-0.888). Raw-LR ~ ddim-init under chain training
(first chain does the reconstruction work). NEW BEST (physicality-first): k2-chain ddim-init DDPO
+lam3 = ret 0.528 / resid 1.62 / MSE 0.0167 / place 0.883 at half inference compute. Detail-first
alternative: gentle cascade (below). Next: K=3 [100,50,30] chain training; retention-weighted reward
to close the k* gap.

**Re=2000 GT-FREE K=2 chain training (2026-07-14).** Same K=2 [100,75] stoch-DDIM chain finetuning,
fully extrapolated reward (extrap anchor+floor, scales_re=1000, align=Re1000 value), raw + ddiminit
variants, 200 iters. Probes: raw 0.290->0.352, ddiminit 0.300->0.372 (peak 0.381). Matched eval:
ddiminit DDPO ret 0.351 ~= prev OOD best 0.361 at equal placement (0.869) and MSE, HALF the inference
compute; resid 2.44 (+lam3 2.24). CAVEAT: OOD residual lands BELOW the 3.01 GT floor (2.1-2.4) —
partially under-turbulence (ret ~0.35 -> intrinsically smaller residual terms), not super-physicality;
below the floor, resid-vs-floor stops being a physicality metric. OOD chain-base gains almost nothing
from the recipe alone (0.275->0.290) — the DDPO gain is genuine finetuning. ddim-init keeps a small
OOD edge (0.351 vs 0.339; weaker base recon -> pre-pass less redundant). Report sections 11-12 have
fully annotated config tables (input/finetuning/K/inference/guidance per row).

**Gentle decreasing cascade S=[100,50,30] (2026-07-13, user-proposed, `--s_multi`) — best in-dist
DDPM-policy config.** Deep exploration once then two conservative refinements: ret 0.675 (NO
overshoot vs 1.19 for [150,100,50]), placement 0.877 (DDPO record; base rows 0.886-0.888), +lam3
resid 3.39. OOD balanced middle (0.401/0.863). Deep-once-then-refine beats deep-repeatedly.
Also: single-chain t_start at inference = enhancement dial (t=30 ~= raw DDIM recon, t=100 = full).

**K-refinement on ddim-init inputs (2026-07-13,** `--base_ddim_init --k3`**, K=2 = end of chain 2 of
the K=3 cascade, same noise).** Re=1000: base K1/K2/K3 ret .407/.453/.465, place .862/.889/.890
(HIGHEST measured); DDPO K1/K2/K3 ret .585/1.123/1.194 (OVERSHOOT in-dist, tempered vs raw-init's
1.333), place .862/.858/.853, MSE .0170/.0236/.0248. Re=2000 OOD: DDPO K1/K2/K3 ret
.361/.592/.614 (refinement nearly DOUBLES OOD retention), place .868/.834/.822, MSE .0270/.0328/.0340;
+lam3 trims resid 4.35->3.89 (~= base K1). PATTERN: most of the refinement effect arrives at K=2;
K=3 adds ~+.02 ret for -.01 place / +.001 MSE everywhere -> K=2 is the efficient point (confirms
backlog #2 hypothesis). Refinement amplifies DDPO's energy lever: harmful in-dist (overshoot),
strong OOD. Best OOD detail config now: ddiminit DDPO K2 + lam3 (ret .576, resid 3.89, place .823).

**Deep-cascade unlock (2026-07-14) — the comparability eval that dissolved the trade.** The K=2-trained
models evaluated on the DDPM results' noise levels: stoch-DDIM 3-chain [150,100,50] (86 steps = training
density; t=150 off-recipe), per-chain rows (`--ddim_stages`). Re=1000 k2dd: 0.555 -> 0.664 -> 0.696
(k*=95, vs 39 on the training schedule!), +lam3 resid 1.70 at place 0.883 — retention ABOVE gentle
cascade (0.668) at HALF its residual (1.70 vs 3.39). OOD (GT-free): 0.359 -> 0.419 -> 0.436 (+lam3
resid 2.38, place 0.837), prev best 0.361. Cost: ~6% MSE. FINAL BEST (both regimes): k2-chain ddim-init
DDPO + deep cascade [150,100,50] + lam3. Diff-maps figure (deepchain_diff_maps.png): all configs still
under-carry E_hik at GT hotspots (0.41-0.65 of GT); deep cascade fills most with least misplaced excess.
Report sections 13 (tables + montage + spectra + diff maps).

**Band A/B + degradation A/B + per-band limits (2026-07-14).** Config-verified pair (grid4x, ddim-init,
K2-chain; only spec_highk band differs). (1) DEGRADATION DOMINATES: grid4x training beats the accidental
random-1024 training by +0.17 ret at same recipe (0.854 vs 0.682 deep-K3+lam3) — per-checkpoint
config.json provenance now mandatory (commit 5052f2a). (2) Band k>=10: small free win both regimes
(+0.02 ret in-dist; +0.008 ret/-0.05 resid OOD) — KEEP — but per-band E/E_GT shows it did NOT fill the
mid band: [10,20) 0.833->0.849, [20,32) 0.698->0.718 (+2pt against a 28pt deficit). (3) THE LIMIT,
quantified: k=32 is the grid-4x input Nyquist. Above it (free synthesis) DDPO+chains fully close the
gap (base 0.32->1.01 at [64,96), 0.43->0.77 at [32,64)); below it the input pins the field and reward
shaping is nearly powerless (base 0.62->0.72 at [20,32), deficit 28% remains). NEW BEST both regimes:
hk10 deep-K3+lam3 — in-dist ret 0.873/resid 1.78/place 0.861/k*95; OOD GT-free 0.441/2.33/0.849.
Open frontier: sub-Nyquist [10,32) deficit (mid-t chain? input-consistency term?) — report section 14.

**Depth sweep + hybrid (2026-07-14, report s15).** hk10 single chain stride-5, t=500..50, full
val+test: sub-Nyquist k[20,32) FOLDS at t~350 (0.63->0.98) — manifold travel works — but bands mature
at different depths (k>=64 @ t~200, 32-64 @ t~250) so the tail overshoots 1.5-2.1x at the mid-band's
depth, MSE 2.5x (0.038), placement 0.774 (peak 0.872 @ t=150). Band-staggered HYBRID (base deep
D=250-450 -> hk10 [100,75]) REFUTED: pure travel does NOT fill mid-band (base deep k[20,32)
0.58-0.62 < ref 0.731 — the sweep's filling was the AMPLIFIER traveling); hybrid strictly worse
(D=350: MSE +83%, ret -0.14, place -0.04 vs ref). Residual robust 1.6-1.8 everywhere. Frontier:
overshoot control during deep amplified travel (spectral clamp/guidance on deep chain, or
depth-trained model). Commit: --stride in make_kchain_ddim_sampler.

**Spectral brake (2026-07-14, report s16) — FULL-BAND SPECTRUM MATCH.** One-sided hinge guidance
above the reward anchor on k[32,96) (`make_spec_brake_grad`, applied on x0_hat like lam-guidance;
anchor = the reward's own regime stat, OOD-safe via extrap anchors). hk10 deep t=350 + mu~1e3-4e3
(robust 3x plateau; unstable >1e4; calibrate mu on an ABOVE-anchor field — below-anchor gives zero
grad): ALL bands land 0.91-0.99 of GT simultaneously (tail 2.11->0.92 while mid-band fill survives
0.96->0.93 — tail/mid amplification are SEPARABLE). Brake also trims MSE 0.038->0.035, resid
2.20->1.99. Operating menu: fidelity-first (deep cascade: MSE 0.0195/place 0.86/midband 0.73) vs
spectrum-complete (t=350+brake: bands 0.91-0.99/resid 2.0/MSE 0.035/place 0.71). Depth decorrelation
remains the fidelity wall -> frontier: observed-pixel consistency guidance during deep chains.

**Disentangle + anchor cost + vs-input contrast (2026-07-14/15, report s17).** (1) GUIDANCE ALONE =
ZERO: base rows bit-identical +-brake OOD (below-anchor -> ReLU gate closed); all energy from
finetuning. (2) CFG-MATCHED Re=2000 GT-free run (align 0, 300 it, k>=10 — twin of the Re=1000 recipe;
probe peak 0.386, record): deep expression 0.564 vs cross-regime 0.871 -> THE ANCHOR IS THE GT-FREE
COST (extrap tail too conservative), not align/iters/band. (3) Missing cell: ft-Re1000 OOD deep
cascade K3 = 0.521 (+lam3 0.518/2.29/0.830) vs GT-free 0.443 same recipe -> NEW OOD BEST = transfer
the measured-anchor in-dist model; fix extrap anchor high-k as training-side alternative. (4) Brake
normalization bug fixed (batch-mean -> per-sample sum; validated mu 3657@B16 == per-sample ~228;
contrast-script divergence caught it). (5) vs-input contrasts: in-dist spectrum-complete trades 2x
frame MSE for the spectral fill; OOD the deep chain IMPROVES E_hik placement over the input
(0.737 -> 0.784/0.789). Push-notification-on-completion workflow in use.

**Step-resolved chain diagnostics (2026-07-15, report s18).** x0_hat metrics at EVERY step of the two
best configs + per-chain incremental maps. Cascade: chain 1 does ALL durable work (placement
0.846->0.867, resid 5->1.9 attractor, mid-band written in first ~15 deep steps then frozen); final
low-t steps of every chain OVER-SMOOTH the tail (k64-96 peaks 0.78 mid-chain -> 0.62 at chain end);
renoises reset the tail up (0.62->1.1) at +0.001 MSE each; chains 2-3 = tail managers only. Braked
deep: best MSE occurs EARLY (t~320), brake-vs-amplifier spiky tug-of-war converges below t~100;
residual attractor ~1.9-2.0 self-restores after every perturbation. Increments maps: every chain adds
energy at the SAME filament sites (placement-respecting amplification). ACTIONABLE: shorten chains
2-3 / single t~50 finisher (compute saving); stop chains at t~10-20 instead of 1 to keep tail near
its mid-chain peak (one-line experiment); mid-band = deep-travel-only (4th line of evidence).

**BaratiLab comparison (2026-07-15, report s19).** Their baseline preds (pred_it2, banked pkl) vs ours
on THEIR benchmark (random-1024 = 1/64 = their 32x32 paper case, irregular masks, seqs 36-39, 320
frames, their inputs+reference). WE WIN EVERY AGGREGATE: MSE 0.1438 vs 0.1578 (-9%), resid 2.00 vs
2.52 (GT floor 1.12), ret 0.553 vs 0.468, place 0.414 vs 0.343 (best rows: deep cascade +lam3).
Nuances: their iterative-refinement sampler already physicality-preserving (their baseline resid 2.5
vs OUR base 4.4 — their procedure anticipated chain-inference; ours still beats it); band profiles
differ (they hold mid-band 0.63-0.82, we rebuild tail 0.72-0.78 vs their 0.44; our k* 24 vs 35);
cross-degradation: grid4x-trained hk10 edges task-matched k2dd even on random-1024. NOT compared:
their physics-informed conditional variant (never exported — still the open head-to-head).

**Baseline vs baseline (2026-07-15, report s20).** Our base under THEIR exact cascade [150,100,50] on
their inputs (320 frames): per-frame win rates MSE 100% ours (0.1425 vs 0.1578, -9.7%), residual 0%
ours (3.67 vs 2.52 — distributions don't even overlap). Two implementations of the "same" recipe land
on opposite trade points: ours = conservative/averaged (best MSE+placement 0.395), theirs = textured/
physical (ret 0.468, k*35, resid 2.5 — their sampler avoids the t->1 oversmoothing that costs our
cascade retention 0.396->0.361 across stages, cf s18). Vorticity PDFs near-identical in core, both
thin extreme tails. The finetuned hk10 deep+lam3 (s19) breaks the dichotomy: beats their baseline on
ALL FOUR aggregates. Worth borrowing: their per-stage sampler detail that avoids end-of-chain
smoothing (aligns with s18's "stop chains at t~10-20" candidate).

**Their sampler decoded (2026-07-15, s20 corrected).** BaratiLab's shipped recons sampler (verified
from main_v1 source): per iteration it: depth = t*0.7^it (default t=400 -> 400/280/196; README t=240),
steps = r*0.7^it (default 20/14/9) COARSE DETERMINISTIC DDIM (eta=0), stride ~20 -> finest visited
t~20 then single x0 jump = EARLY-STOP by construction — independently validates s18's stop-at-t~10-20
candidate (their cascade avoids our stage-wise retention decay this way). Banked pred_it2's exact
(t,r) unrecorded (provenance lesson applies to them too). TODO next: (a) early-stop endpoint in OUR
cascades (one-line); (b) run our base + our finetuned under their true ladder (400/280/196 coarse
eta=0) — cheap (43 steps total); (c) s20 table = both models under OUR nominal [150,100,50], labeled
correctly now.

**Their ladder on our models (2026-07-15, report s20b).** Faithful replica (400/280/196, eta=0,
20/14/9 steps, early-stop t~20, raw inputs) on our checkpoints. (1) hk10 under THEIR sampler = BEST
SYSTEM on their benchmark: ret 0.803 (vs their 0.468), resid 1.72-1.79 (GT floor 1.12; 99.7% of
frames better), place 0.428 (vs 0.343), MSE tie 0.158 — their sampler + our finetuning COMPOSE
(deep ladder = physics; early-stop preserves what the amplifier builds). (2) Our base under their
ladder = physicality extreme (resid 1.31, 100% frame-wise; MSE 0.1453, 94%) but spectrally collapsed
(ret 0.207) — identical procedure, very different eps-conservatism between the two trained DDPMs
(caveat: banked pred_it2's t unknown, 240 vs 400). (3) lam3 adds ~nothing near floor. TODO: t=240/r=30
variant to disambiguate their banked args; early-stop endpoint in our own cascades still pending.

**Ladder follow-ups (2026-07-15, report s21).** (1) README args t=240/r=30: our base ret 0.307 (vs
their 0.468) — neither arg set reproduces their signature -> MODEL-level texture difference is
genuine (their DDPM retains more under identical procedures). hk10 gains a depth dial on their
benchmark: t=240 -> MSE 0.1446 beats their 0.1578 at ret 0.57; t=400 -> ret 0.80 at MSE parity.
(2) EARLY-STOP DOES NOT TRANSPLANT to our cascade: t_floor 15/25 lifts ret +0.02/+0.04 but entirely
as tail overshoot (1.145->1.368) + resid 1.76->1.99, mid-band unmoved. MECHANISTIC CLOSURE: the t->1
"over-smoothing" (s18) is a NATURAL BRAKE for an amplified model (pulls overshooting tail toward 1);
the same mechanism is a loss for their conservative model. Keep t_floor=1 for our cascade. Their
trick is right for their sampler, wrong for ours — now with the exact reason.

**EMA hypothesis + probe (2026-07-15, report.md).** Root cause of base conservatism identified:
BaratiLab trains with EMA (mu=0.9999, ema.py; inference loads states[-1] = EMA weights,
rs256_guided_diffusion.py:299); our train_ddpm.py has NO EMA. Post-hoc average of last 6/11 epoch
snapshots (GCS series ep0009-0299): every band UP at zero cost — ret 0.209->0.246 (their ladder,
+18% rel, ~15% of gap to their ckpt) and 0.385->0.400 (our K1) — DIRECTIONAL CONFIRMATION with the
crudest surrogate. NEXT: add optax-style EMA to src/train_ddpm.py and retrain base (mu=0.9999);
then re-evaluate DDPO stack on EMA base (amplifier gain may need less; sub-Nyquist band the metric
to watch). Handoff doc: report.md (repo root + gs://ddpm-thesis-rh/report.md).

**Conditional model scored + baseline args corrected (2026-07-16, report s22).** Their conditional
(as-released t240/r30/s1, w=0, smoothed input; reference bit-identical, md5 verified): ret 0.101 /
resid 1.79 / MSE 0.1386 / place 0.394 — extreme conservative; beats their baseline MSE 100% / resid
99.7% of frames but spectrum nearly empty. Their run report revealed the June baseline's TRUE args
(t400/r150/s3, NOT r=20 defaults) -> correction run: our base at exact args = ret 0.289 / resid 1.64
/ MSE 0.1288 / place 0.464 (step count explained ~30% of the s21 gap; model-level texture difference
0.289-vs-0.468 PERSISTS under matched-everything — EMA hypothesis intact and sharpened). Bonus: our
base at true args = best MSE/resid/place of ANY config on their benchmark. Pending: their run B
(budget-matched conditional, t400/r150/s3).

**EMA arc executed: base retrained + DDPO restacked (2026-07-17).** Base retrained 300ep with
EMA mu=0.9999 (train_ddpm.py shadow update; ckpts carry named ema_params/ema_rate; run config.json
provenance; gs://.../checkpoints/ddpm/base_ema_mu9999_300ep/). A/B: retrain lifts base everywhere
(their ladder 0.209->0.240, our K1 0.385->0.401, their-bench true-args 0.289->0.314, bands k>=10
all up, MSE/resid flat); EMA shadow ~= online at convergence (slightly cleaner resid 3.47 vs 3.67,
slightly lower ret) — EMA is real but SMALL at 300ep; remaining gap to their 0.468 lives elsewhere.
DDPO hk10 recipe rerun on EMA base (BASE_CKPT override): probe base 0.438 (old 0.420), plateau
0.60-0.67, final 0.662 (old-run trajectory beaten throughout). EMA-stack evals: their-bench deep+l3
ret 0.553->0.629, their-ladder 0.803->0.858; our grid4x deep+l3 ret 0.872 / place 0.891. NEW ISSUE:
tail k[64,96) now OVERSHOOTS GT (1.11-1.21) and resid worsens where overshoot largest (2.26 ladder)
— brake retune (higher lambda or hinge on [64,96)) is the obvious next lever.

**Policy-EMA option added to DDPO trainer (2026-07-17, OPTION ONLY, default off).**
train_claude.py --policy_ema MU (e.g. 0.99 ~ 100-outer-iter horizon): per-OUTER-iteration shadow of
policy weights, motivated by the 0.56-0.67 GT-probe plateau noise (checkpoint quality = eval-draw
luck). Shadow is passive (does not alter training); checkpoints then carry named ema_params/
ema_rate; save_ckpt payload unchanged when off. Unit-tested (payload keys + update math). Natural
first test: the armed Re=2000 GT-free run (monitoring/ema_re2000_driver.sh) — one run yields both
online and shadow checkpoints to compare.

**Full EMA-stack matrix + old-stack comparability + brake sweep (2026-07-17, artifact
816fb598).** Old stack re-measured under identical protocol/seeds (old flagship reproduced to the
digit — 0.553/2.00/0.1438). EMA advantage GROWS with chain depth (their bench: +0.039 K1 -> +0.078
K3; grid4x: +0.019 -> +0.053); flagship deep+l3: their bench 0.553->0.629, grid4x 0.819->0.872
(place 0.862->0.891). OOD Re=2000 cross-regime NEW BEST 0.518->0.555 (place 0.830->0.861). Dense
true-args ladder overshoots on BOTH stacks (old 1.120 / EMA 1.169) — amplifier property, not EMA.
BRAKE SWEEP on the 86-step cascade (mu 15..230): hinge SATURATES immediately (mu=15 ~= mu=230; the
cascade re-amplifies each step, hinge clamps band to anchor — anchor is the binding parameter, not
mu). In-dist verdict: flagship stays UNBRAKED (costs 0.06-0.08 ret for 0.03-0.05 resid; the
[32,96) hinge drags the still-under-GT [32,64) band too). Brake still right for dense ladders
(their ladder tail 1.208->0.732 at mu=150) and GT-free OOD. SUPERSEDES the earlier "brake retune"
note: the sharper lever is a [64,96)-ONLY hinge (or slightly raised anchor), NOT a mu retune.
Report artifact: claude.ai/code/artifact/816fb598-398e-4fa4-ad42-7fdf8b86b921. Armed, not
launched: monitoring/ema_re2000_driver.sh (GT-free Re=2000 on EMA base, --policy_ema 0.99).

**Dedicated GT-free Re=2000 on EMA base: ANCHOR-CAPPED (2026-07-17).** Run healthy (reward -11.1->
-8.7, gstd 0.60-0.72 no collapse, |A| 0.83) yet deep-eval K3+l3 ret 0.427 vs old dedicated 0.441 —
identical cap, base swap irrelevant. Cross-regime EMA-ft(Re1000) remains OOD best at 0.555. ROOT
CAUSE: extrapolated anchor under-predicts the true Re=2000 tail and the spec reward is a DISTANCE
(penalizes above-anchor too) -> training pulls the policy TO the anchor. NOT a signal problem.
NEXT OOD LEVER: raise the extrapolated anchor's tail (better spectral extrapolation), then retrain.
POLICY-EMA FIRST DATA (same ckpt, shadow mu=0.99 vs online, K3+l3): 0.405/2.03/0.0302/0.876 vs
0.427/2.26/0.0319/0.867 — shrinkage signature; shadow = cleaner (resid -0.23, MSE -5%, place up)
for -0.022 ret. Use shadow when quality>retention (OOD), online in-dist. Free choice per ckpt.

**GT-free anchor FIXED via observation-constrained fit (2026-07-17, src/ddpo_ft/anchor_obs_fit.py).**
Root cause of the old extrap's tail hole decomposed: k_d ~ Re^0.5 scaling law is wrong here
(empirical exponent grows: 0.56 on 500->1000, 0.71 realized on 1000->2000); slope/amplitude were
fine. FIX (uses ONLY 4x-subsampled target samples): measure observation transfer T(k) on known
regimes -> fit spectrum model to T-corrected observed band k[6,30] with alpha/p priors + k_d
scaling-law prior (empirical exponent) -> two-sided leave-one-regime-out validation (500<->1000)
gives bias correction + error bar (both directions recover k_d to <1 shell). RESULT vs true Re2000
(report-only): bands 1.01/0.97/1.06 (was 0.91/0.82/0.81). Variants saved:
base_results/regime_stats_re2000_obsfit.npz (accurate) and _obsfit_gen12.npz (x1.2 tail generosity
= in-dist anchor profile 1.08/1.12/1.16 that trains well). PREDICTION (falsifiable): dedicated
GT-free finetune with --stats regime_stats_re2000_obsfit_gen12.npz beats cross-regime 0.555
(old anchor capped it at 0.427). Retrain not yet launched — user decides.

**BREAKTHROUGH: sampling temperature is the OOD throttle (2026-07-18).** Dedicated GT-free Re=2000
finetuning now BEATS cross-regime transfer for the first time: temp=2.0 resumed from gen12 iter0399
(+200 iters) -> deep K3+lam3 ret **0.630** / resid 2.39 / MSE 0.0310 / place 0.846 / **k*=75**,
vs its own start 0.485 (k*=41), vs null expectation ~0.513, vs cross-regime champion 0.555.
Probe climbed 0.381->0.467 (+0.086/200 iters = ~3x the standard +0.028 rate), never destabilized.
LEVER RANKING from the full campaign (all measured, same protocol/eval):
  temp 1.5->2.0    : +0.145  <-- the throttle (exploration)
  iterations       : +0.028 / 200 iters (linear, no plateau, anchor-independent)
  anchor 0.78->1.2 : +0.024 at 400 iters (weak ~7% pass-through; real but modest)
  clip 0.2->0.35   : NULL (probe peaks early 0.416 then decays; clamp was never binding)
  lr 5e-5->1.5e-4  : DIVERGES (probe 0.381->0.318->0.296, below base, in 20 iters)
MECHANISM: gstd FELL with temp (0.736->0.577) while reward ROSE (-13.50->-12.36) — so it is NOT
"more signal spread"; it is exploration: wider reverse-step noise reaches trajectories the policy
never sampled, and those gains compound. Consistent with clip-null (clamp not binding) + lr-divergence
(gradients too noisy for big steps): the binder was never step size, it was trajectory coverage.
COSTS: placement 0.871->0.846, resid 2.29->2.39, MSE 0.0301->0.0310 (amplifier trade, acceptable).
NOTE probe_x0 uses a deterministic temp=1 eval rollout (ppo_claude.py:437), so probe gains are
genuine policy improvements, NOT sampler artifacts — verified before claiming the result.
NEXT: temp 2.5 (find the optimum in the proven direction); temp2.0 + lr 7.5e-5 (1.5x, in the
cleaner-gradient regime, early-kill on 2 consecutive probe drops); n_inner/group_size as
noise-reducing alternatives to raising lr.

**temp 2.5 -> ret 0.799 on OOD Re=2000 (2026-07-18).** Continuing the temperature direction from
temp2.0 iter0599: deep K3+lam3 **0.799** / resid 2.55 / MSE 0.0327 / place 0.813 / **k*=82**.
Progression base 0.297 -> gen12@400 0.485 -> temp2.0@599 0.630 -> temp2.5@799 0.799 (GT = 1.000).
REFRAME (important, corrects earlier wording): rising PDE residual is NOT a cost — GT's own residual
is 3.01 and an over-smoothed field has a LOW one (base 1.89). So 1.89->2.29->2.39->2.55 is
convergence TOWARD the physical truth. Real costs are MSE (0.0301->0.0327, expected for an
amplifier: statistically-right texture is not pixel-exact) and PLACEMENT (0.871->0.846->0.813,
falling monotonically with temp). PLACEMENT IS THE STOPPING CRITERION for this lever — retention
climbs much faster than placement falls so the trade is still strongly favourable, but far enough
up the temperature axis is just well-calibrated noise. temp 3.0 queued with that as the read-out.

**temp 2.5 DOMINATES temp 3.0 at equal compute; OOD now 0.921, in-dist 0.967 (2026-07-19).**
Controlled head-to-head from the SAME temp2.5 iter0799 checkpoint, both +200 iters to 999:
  temp 3.0 : ret 0.870 / resid 2.62 / MSE 0.0390 / place 0.798 / k*=71
  temp 2.5 : ret 0.921 / resid 2.60 / MSE 0.0356 / place 0.807 / k*=84   <-- dominates on ALL FOUR
So 3.0 is not merely "past the optimum with costs" — it is strictly worse. temp 2.5 is THE operating
point. Iterations at 2.5 still compound hard (+0.122 in 200 iters: 0.799 -> 0.921 vs GT 1.000).
IN-DIST temp 2.5 also works (Re=1000 flagship +200 iters): ret 0.967 / resid 1.66 / MSE 0.0197 /
place 0.847 / k*=86 vs base 0.472 — near spectral parity with GT. CAVEAT: in-dist the residual now
OVERSHOOTS (1.66 vs GT 1.06) and placement falls 0.900->0.847, i.e. in-dist we have moved from
filling a deficit to adding excess roughness — the in-dist optimum is probably BELOW 2.5 (the model
was already at ~87% retention pre-temp, vs OOD's 30%, so far less deficit to fill). Matched-protocol
eval of the standard-temp flagship queued to make the in-dist delta exact (0.967 came from
eval_guided_full; the 0.872 figure came from a different frame set — not apples-to-apples yet).
OPEN: seed repeats for variance (every number in this campaign is a single run); in-dist temp sweep
below 2.5 (try 1.75/2.0); kl_coef / n_inner / group_size still untested.

**INFERENCE-temp is a free lever; in-dist optimum is BELOW 2.5 (2026-07-19).**
(1) INFERENCE-temp sweep on FIXED weights (best OOD model, temp2.5-trained, no retraining):
      0.70 -> ret 0.934 / resid 2.55 / MSE 0.0350 / place 0.801 / k*=86
      0.85 -> 0.921        1.00 -> 0.906 (default)   1.20 -> 0.885   1.40 -> 0.864
    MONOTONE: colder inference is strictly better on ALL FIVE metrics. +0.028 retention for free.
    Mechanism: the model was TRAINED under temp 2.5 so its score fn compensates for heavy noise;
    at inference you want the amplifying correction WITHOUT the stochastic dispersion. Training-time
    exploration and inference-time stochasticity are opposite needs. eta=0 limit queued.
(2) EVAL-SEED NOISE MEASURED: the sweep's temp=1.0 row (0.906) is the same model/settings as the
    earlier deep eval (0.921) — only the sampling seed differs => eval noise ~ +-0.015. ANY margin
    below ~0.02 in this campaign is inside the noise floor. CORRECTION: the temp2.5-vs-3.0
    "dominates on all four" claim should read "dominates on ret/MSE/k*, TIES on placement"
    (0.807 vs 0.798 is within noise).
(3) IN-DIST temp sweep downward — 2.5 was PAST OPTIMUM, as predicted:
      standard (1.5): ret 0.917 / resid 1.86 / place 0.885 / k*=95
      temp 1.75     : ret 1.043 / resid 1.68 / place 0.877 / k*=95   <-- keeps SHAPE fidelity
      temp 2.5      : ret 0.967 / resid 1.66 / place 0.847 / k*=86   <-- shape degrades
    Retention now OVERSHOOTS (1.043 > GT 1.000); 1.0 is perfect, not maximal, so score by |ret-1|.
    temp 1.75 wins on the metrics that catch MISALLOCATED energy (k*, placement) => better in-dist
    operating point. Earlier "0.967 = near spectral parity" was wrong twice over (baseline was 0.917
    not 0.872, and a LOWER temp is closer to correct).
OPEN: in-dist temp 2.0 + OOD seed repeat (running); cold inference sweep 0.3-0.7 + eta=0 (queued);
kl_coef / n_inner / group_size still untested.

**SEED REPEAT FORCES A CORRECTION; cold inference optimum; eta=0 is NOT the temp->0 limit (2026-07-19).**
(1) SEED VARIANCE MEASURED (the most important result here). OOD temp2.5 @999, identical config,
    seed 1234 vs original: ret 0.842 vs 0.921 => TRAINING-SEED SPREAD ~0.079 (~ +-0.04), 5x the
    +-0.015 eval-seed noise. CONSEQUENCE: temp3.0's 0.870 falls BETWEEN the two temp2.5 seeds, so
    "temp 2.5 dominates temp 3.0" is NOT SUPPORTED at n=1 per config. RETRACTED. The large effects
    survive comfortably (temp 1.5->2.5: 0.485 -> 0.88+-0.04), but every fine-grained margin in this
    campaign (<0.08) needs n>=3 before it can be claimed.
(2) IN-DIST temp sweep is NON-MONOTONIC in retention once variance is admitted:
      1.5 -> 0.917 (k*95) | 1.75 -> 1.043 (k*95) | 2.0 -> 1.116 (k*95) | 2.5 -> 0.967 (k*86)
    Retention cannot separate 1.75/2.0/2.5 given +-0.04. The metric that DOES separate cleanly is
    k*: 95 for temps <=2.0, 86 at 2.5. DEFENSIBLE CLAIM: temp <=2.0 preserves spectral shape, 2.5
    degrades it. NOT defensible: "1.75 is the in-dist optimum" (earlier entry over-claimed).
(3) COLD INFERENCE SWEEP (fixed weights, no retraining) keeps improving as temp falls:
      0.30 -> 0.958 (k*87) | 0.45 -> 0.951 | 0.60 -> 0.942 | 0.70 -> 0.934 | 1.00 -> 0.906
    BEST OOD CONFIG = temp2.5-trained model @ INFERENCE TEMP 0.30 -> ret 0.958, k*=87, free.
(4) eta=0 IS NOT THE temp->0 LIMIT: eta=0 gives ret 0.551 / k*=59 (vs 0.958 at temp 0.30). With
    eta=1 the DDIM mean uses sqrt(1-ab_n-sigma^2) and temp scales ONLY the added noise; eta=0
    changes the MEAN term itself. Different samplers, 0.4 retention apart.
NEXT (properly powered): n>=3 seeds per config for any comparison with margin <0.08; kl_coef /
n_inner / group_size still untested; inference temp below 0.30 not yet probed.

**temp vs eta clarified; original advantage-spread rationale CORRECTED in code (2026-07-19).**
temp is ONE parameter used with OPPOSITE objectives: training wants it WIDE (exploration: reach
unvisited trajectories), inference wants it NARROW (fresh noise is white and later steps partially
smooth it, so each renoise/denoise cycle costs fine structure). Applied consistently to rollouts AND
log-probs, so temp REDEFINES pi_theta — deploying a temp2.5-trained policy at temp 0.30 is a
DELIBERATE train/test mismatch (means learned assuming heavy noise follows, then not supplying it),
and it WINS (+0.05). eta is NOT the same knob: it enters BOTH sigma and the mean via
sqrt(1-ab_n-sigma^2), so eta=0 is variance-PRESERVING/deterministic while eta=1 + small temp is
variance-DEFICIENT -> different trajectories, 0.4 retention apart. DOCSTRING FIX: ppo_claude.py
claimed temp>1 works by widening K-sample spread for a "stronger advantage signal"; measured gstd
FELL 0.736->0.577 while reward ROSE — spread is a symptom, not the driver. Corrected in code.

**PDE floor (residual_ref): how to set it blind, and two ways that FAIL (2026-07-20).**
The pde term is a HINGE: max(0, ln(mean R^2) - ln(residual_ref))^2. residual_ref is the GT
discretization floor. Too LOW => fires on physically-correct samples and swamps the spectral terms
(pde scale ~0.39 amplifies it ~2.6x); too HIGH => hinge merely goes slack. ERR HIGH.
MEASURED mean(R^2) on proper consecutive-frame triplets (reproduces the calibration json):
  Re= 500 -> 1.57 (json 1.56) | Re=1000 -> 10.2 (json 17.5) | Re=2000 -> 75.9 (json 46.0)
  Re=10000 -> 2156.5  [report-only]
METHOD A (used): Re-scaling from Re=500/1000, exponent b=2.705 -> residual_ref(10000)=5183.
  2x validation: predicts 66.6 vs measured 46.0 = 1.45x OVER. At 10x: 5183 vs 2156 = 2.4x OVER.
  Over-prediction is the SAFE direction for a hinge. THIS IS WHAT THE RUN USES.
METHOD B (tested, FAILS): evaluate the residual on the target's own LOW-RES observations (no GT
  needed — the LR fields are real data) and map LR->HR using the known regimes. The HR/LR ratio is
  NOT stable: 0.023 / 0.044 / 0.122 / 1.391 for Re=500/1000/2000/10000 (60x drift). Calibrated on
  500+1000 it predicts 20.6 at Re=2000 (measured 75.9, 0.27x) and 51.4 at Re=10000 (true 2156,
  42x TOO LOW — the dangerous direction). Reason: LR residual is dominated by coarse-grid
  discretization error, nearly Re-independent, while HR residual explodes with fine structure.
METHOD C (FATAL, was nearly shipped): fall back to the Re=1000 json entry (17.46) when the target
  regime has no calibration row. That is ~123x too low for Re=10000 -> constant max-strength
  penalty on every sample incl. GT-quality ones, dominating the reward with a smoothness push.
ALSO: the Re=2000 GT-free runs used residual_ref=26.29 from the extrap npz vs a MEASURED 45.99 —
i.e. the floor was ~1.75x too low and the pde term was over-strict all along. Those results were
obtained DESPITE a pde term fighting high-k energy addition, not because of a tuned one.

**METHOD D (tested, FAILS): PDE floor from the BASE-DDIM RECON of the LR input.** Idea: the recon
is at 256^2 (right resolution for the operator), carries correct large-scale structure, and uses no
target GT — so evaluate mean(R^2) on it and map recon->GT using the known regimes. MEASURED
recon/GT: 3.72 / 0.64 / 0.21 / 0.09 at Re=500/1000/2000/10000 — a 40x drift. Calibrated on 500+1000
it predicts 95 at Re=10000 vs a true 2297 (24x UNDER, the dangerous direction).
STRUCTURAL REASON (rules out the whole family): the recon is smoother than GT, and HOW MUCH
smoother is precisely what varies with Re — the base over-produces structure at Re=500 (3.72x) and
progressively under-produces as Re rises (0.09x at 10000). That deficit-vs-Re curve IS the problem
DDPO exists to fix, so any estimator built from the model's own output carries a bias coupled to
the failure it is meant to calibrate. You cannot use the model's output to measure the target it is
failing to reach. Same reason METHOD B (raw LR) fails.
BLIND-FLOOR SCOREBOARD at Re=10000 (true 2156-2297): Re-scaling 5183 = 2.4x OVER (safe, IN USE);
raw-LR 51 = 42x under; base-recon 95 = 24x under. Prefer over-estimation: a hinge that is too high
merely goes slack, one that is too low fires on physical samples and drives over-smoothing.

**Re=10000 RESULT (2026-07-20): blind anchor works, temperature rule confirmed, placement is the ceiling.**
Setup: anchor blind from Re=500+1000 ONLY (grades 1.18x truth); clean disjoint split (train 20-23,
probe 24, test 10-19); PDE floor 5183 (blind Re-scaling). Base recovers just 9.4% of hi-k -> the
most starved regime yet (Re=2000 30%, in-dist 92%).
TEMPERATURE RULE CONFIRMED AS A PREDICTION: temp 2.5 PLATEAUED (reward flat -13.48, spec_hik stuck
2.49 across 100 iters, gstd healthy 0.485 = under-exploration not signal collapse). Our rule says
the optimum tracks DISTANCE FROM TARGET, so a 9.4% deficit should want >2.5. Resumed at temp 3.5 ->
plateau broke immediately (probe 0.135 -> 0.24, spec_hik 2.49 -> 0.22). Falsifiable: temp 3.0 was
WORSE at Re=2000, so "hotter is better" would have been wrong; the optimum MOVES with the deficit.
CHECKPOINT SWEEP (K3+lam3, clean test seqs): base 0.184/place 0.711/k*25 ->
  249: 0.792/0.659/67 | 299: 0.768/0.659/66 | 349: 0.957/0.633/69 |
  399: 0.983/0.626/68 <-- BEST |ret-1|=0.017 | 449: 1.035/0.589/70 | 499: 1.072/0.599/69
CEILING: NO checkpoint gets both. Placement falls monotonically as retention climbs and even the
best-placement checkpoint (0.659) is BELOW the base (0.711). Contrast Re=2000, where the best model
held placement within 0.06 of base (0.807 vs 0.868) at 0.92 retention. Second diagnostic: model
residual 4.25 vs GT 28.78 — 7x SMOOTHER dynamically while carrying comparable spectral energy =
right amount of energy, wrong places, too clean.
INTERPRETATION: from a 9.4% deficit the model SYNTHESISES most of the fine structure rather than
refining what the input constrains, and a 4x-downsampled input does not determine where k~70
structures belong. INFORMATION LIMIT OF THE OBSERVATION, not a tuning failure. This marks where the
method stops buying physical fidelity — retention keeps improving, placement does not follow.
CAVEAT on the dataset: Re=10000's spectrum is anomalous above k~100 (LESS energy than Re=2000,
physically impossible) -> grid-limited. A good score means matching THIS SIMULATION, not recovering
true Re=10000 turbulence. Not detectable GT-free with any method we have.
Report: claude.ai/code/artifact/816fb598-398e-4fa4-ad42-7fdf8b86b921 (v18)

**Why the two-sided pde was a NULL at temp 3.5, and the anneal redesign (2026-07-21).**
MEASURED: K2 rollouts at temp 3.5 have residual^2 = 5377 (units: make_residual_loss on normalized
triplets) vs the deployed temp-1 cascade's ~40 and GT's 1610. The exploration noise contributes
~99% of the rollout residual -> the pde term's measurement channel is SATURATED at the training
operating point: it cannot distinguish a dynamically-rough model from a smooth one, whatever the
hinge shape or floor value (c_pde = 0.001-0.004 with the two-sided term active). GENERAL PRINCIPLE:
the reward only ever evaluates the TRAINING operating point, which sits 2.5 temperature units from
deployment — every reward term sees noise-inflated statistics, not just pde.
HOT-INFERENCE DEPLOYMENT: tested and NULL — sampling the fixed weights hotter raises deployed
residual^2 only 40 -> 104 at itemp 2.5 (GT 1610, still 15x short) while ret/place/k* all degrade.
The smoothness is in the weights, not the sampler.
REDESIGN (running): TWO-PHASE ANNEAL. Hot phase (temp 3.5) found spectral parity; cold phase
resumes best ckpt at temp 1.5 with --pde_two_sided, where the rollout noise floor is low enough
for the pde term to finally see the model's own smoothness (expected c_pde ~18 vs 0.001).
Refinement, not search. Then Re=2000 floorfix (re-queued; old waiter was stuck on a dead marker).

**BLIND-PROXY STUDY: GT-free stats CANNOT rank structure fidelity (2026-07-22).** Tested four
GT-free instruments (ensemble agreement K=4, temporal coherence, input-gradient align, |ln R^2/floor|)
against GT placement across 3-model spreads at Re=2000 AND Re=10000, plus a planted noise impostor.
RESULT, replicated in both regimes: agree & tempo ANTI-correlate with placement (finetuning makes
added structure more input-deterministic -> consistently WRONG; self-consistency measures conviction,
not correctness). align tracks but with 0.001-0.003 dynamic range (no sensitivity). rfloor catches
crude noise at Re=2000 (6.32 vs 1.2) but at Re=10000 PREFERS the impostor (1.87) over real models
(~4.85): noise pushes residual up toward a high blind floor and looks "more physical" than genuinely
structured-but-smooth outputs. CAVEATS: 3-model spreads, single seed, crude impostor (temporally
coherent by construction bug — tempo untested vs impostor; noise scale 7-14x too strong).
CONSEQUENCE: OOD quality claims rest ENTIRELY on frozen-recipe validation from truth-available
regimes; per-instance blind QA = gross-noise rejection at best, unreliable at high floors. The
placement-requires-GT objection (user, 2026-07-21) is now an empirical result, demonstrated twice.
ALSO Re=2000 FLOORFIX RESULT: probe 0.56->0.62, deep ret 0.908->1.085 (k* 83->89, resid 2.68->2.73
toward GT 3.16, place 0.808->0.789). Mechanism confirmed (old floor suppressed fine structure) but
OVERSHOOT: with the constraint released the model converges to its actual target = gen12 anchor at
1.2x GT -> deployed 1.09. The over-low floor was an ACCIDENTAL REGULARIZER cancelling the
over-generous anchor. TESTABLE NEXT: corrected floor + ACCURATE obsfit anchor (not gen12) should
land ~1.0 with the k*=89 shape gain and no overshoot.

**PRINCIPLED CONFIG CONFIRMS THE PREDICTION (2026-07-22): accurate anchor + corrected floor -> parity.**
Three-way from the SAME iter0999 checkpoint, +200 iters each, clean test seqs, K3+lam3:
  parent      (anchor 1.2x WRONG, floor 26.29 WRONG): ret 0.908 |d|=0.092 place 0.808 k*83
  floorfix    (anchor 1.2x WRONG, floor 66.64 right): ret 1.085 |d|=0.085 place 0.789 k*89
  PRINCIPLED  (anchor 1.0x right, floor 66.64 right): ret 0.980 |d|=0.020 place 0.802 k*86  <-- BEST
Predicted ~1.0 / k*~89 / no overshoot; measured 0.980 / 86 / none. 4x closer to parity, placement
recovered, MSE back to parent level. 0.980-vs-1.085 gap (0.105) >> +-0.04 noise; same-parent design
makes attribution clean. CALIBRATION STORY CLOSED: anchor = target, floor = constraint, temperature
= reach; all three blind-estimable from Re=500/1000; set all three right and the system lands on
parity. The old flagship's number was two miscalibrations cancelling.

**LOW-k LOSS + SPECTRAL PATCHWORK (2026-07-22, user's idea, works).** The finetuned cascade loses
large-scale energy (k=1 at 0.855 of GT vs recon's 0.996; crossover at k~5, DDPO better above).
Cause: energy reward weight 0.1 and k=1 is one shell of 95 in the spec log-distance — a 15% k=1
loss is nearly free to the reward. FIX = GT-free POST-PROCESSING, no retraining: Fourier hybrid
X = ifft[fft(recon)*(k<kc) + fft(ddpo)*(k>=kc)], smooth 2-shell crossover. At kc=8: low-k [1,5)
0.955->0.989 (recon level), ret/place/k* UNCHANGED (0.959/0.791/86), MSE 0.0363->0.0342 (-6%;
kc=16 -> 0.0325). Principle: use each source where the information lives — large scales are
input-determined (recon is anchored), fine scales are synthesized (DDPO). Caveat: hybrid residual
rises 17->28 (seam cost; still below GT ~76). Retires the boost-energy-weight retraining option.

**FLOOR "DISCREPANCY" RECONCILED (2026-07-23): pure sampling variance, no methodological bug.**
Reproducing the calibration notebook's exact seq/frame selection (stat_seqs[:4], t0=100:292:24)
through the session's pipeline yields the json values to the digit (Re=2000: 45.99). The json vs
re-measurement gaps (17.46-vs-10.22 at 1000, 45.99-vs-75.87 at 2000, agreement at 500) are the
heavy-tailed floor's sequence-subset variance, growing with Re, ~+-50% at Re>=1000. IMPLICATIONS:
(1) never precision-tune the floor below ~50%; measure on all available seqs and round up.
(2) REVISED CLAIM: the blind Re-scaling floor 66.64 is INSIDE the Re=2000 floor's sampling spread
[46,76] — previously reported as "1.45x over"; it is actually indistinguishable from a direct
measurement, a STRONGER validation of the Re-scaling method than claimed.

**FROZEN Re=2000 CONFIRMATORY — SEALED VERDICT (2026-07-23): PRIMARY BAR FAILED, and the failure
is the protocol working.** Single sanctioned run per Appendix B (blind D=0.231->temp 2.5, R3.2
selected iter0599, R3.3 k_c=5 — both blind rules fired sensibly). Sealed test (virgin seqs 24-39):
  base:              ret 0.369  place 0.673  k*30
  DDPO  itemp 0.30:  ret 1.454  place 0.721  k*95   <- PRIMARY FAIL (bar [0.80,1.20])
  hybrid itemp 0.30: ret 1.454  lowk 0.983  MSE 0.0434  resid 42.8 (GT 99.1)  k*95
Secondary bars PASSED (k*>=80; hybrid lowk>=0.97); placement ABOVE base (0.721 vs 0.673) — a first.
ROOT CAUSE (post-grading diagnosis): the NEW data generation has a 1.66x weaker hi-k tail than the
old 20-seed file; the frozen anchor (built from OLD-generation LR) grades 1.56x above the NEW GT.
The pipeline did its job perfectly — drove deployed output to anchor parity — against a stale
target. Anchor/old-GT = 0.94 (fine); anchor/new-GT = 1.56 (stale).
TWO LESSONS, both protocol-level:
(1) FREEZE PROCEDURES, NOT ARTIFACTS: any artifact that depends on target observations (the
    obs-fit anchor) must be REBUILT from the deployment data at hand, not pinned from an earlier
    data generation.
(2) A GT-FREE PRE-FLIGHT EXISTS AND WOULD HAVE CAUGHT IT: LR spectra old-vs-new differ by 1.12x
    in the observable band k[10,30) — a drift check on the observations alone, before any training.
Also real and positive: the machinery generalized (blind temp/selection/k_c all fired correctly;
k* 30->95; placement +0.05 over base on virgin data; hybrid restored lowk 0.983 and moved residual
toward GT). What failed is one number, for a reason now measured and fixable blind.

## 2026-07-23 (later) — v1.3: freeze PROCEDURES, not artifacts (post-mortem corrective, VALIDATED)
- Root cause of the Appendix-B primary fail (ret 1.454): B2 froze the obs-fit anchor as an
  ARTIFACT built from OLD-generation LR. New generation's tail is 1.66x weaker; the stale anchor
  grades 1.261x vs the new GT in E[32,96) — training drove deployed output to parity with the
  wrong spectrum. The user's directive: when the dataset changes and LR inputs exist, anchors are
  recomputed from them.
- Corrective implemented AND validated (report-only, on the already-burned test seqs 24-39):
  - `src/ddpo_ft/anchor_obsfit_builder.py`: the frozen construction PROCEDURE, parameterized by
    (data file, Re, LR seqs). Rebuilt from NEW train-pool LR (seqs 0-3): anchor/GT E[32,96) =
    0.997 vs stale 1.261 — procedure-rebuild alone closes the entire staleness gap. kd fit 48.7
    vs prior 48.3; residual_ref 66.63 by the unchanged law.
  - Anchor npz now stores an OBS FINGERPRINT (LR band spectrum + source path/seqs).
    `verify_freshness()` = GT-free pre-flight: fingerprint vs training-LR band ratio in k[10,30),
    tol 5%. Old/new LR ratio 1.117 would have been caught BEFORE training. No fingerprint = stale.
  - `train_claude.py` runs the pre-flight at launch and ABORTS on stale/fingerprint-less obs-fit
    anchors (tested: stale aborts, v2 passes at ratio 1.000).
  - Protocol v1.3: §2b (binding rule + pre-flight), B7 (outcome record), B8 (attempt #2 terms:
    v2 anchor, sealed seqs 8-23 as fresh test, 5-7 spare; attempt #1 stands in the record).
- Attempt #2 and the Re=10000 confirmatory (§1-7, unaffected — reference-only anchor, file CRC
  unchanged) both await user go.

## 2026-07-22 — attempt #2 verdict; fine-dt data generation discovered; dt32 corrective relaunch
- ATTEMPT #2 (sealed 8-23): PRIMARY FAIL ret(0.30)=1.380 (itemp1.0: 1.283); k*=95 PASS, hybrid
  lowk 0.983 PASS, placement 0.713 > base 0.637. v2 anchor closed part of the gap (1.454->1.380),
  R3.2 approached the anchor from below (iter0449, 0.870).
- ROOT CAUSE (bigger than the anchor): the whole 2026-07 40-seed generation (Re=500/2000/10000)
  is saved ~80x finer in time (lag-1 corr 0.99999 vs 0.986-0.991; lag-80 == old lag-1; one seq
  spans ~3.5 old frames). Conditioning near-static in BOTH attempts; dt-based residuals in wrong
  convention (GT "625" = artifact); test sets ~16 independent states. Spatial metrics stand.
  RETRACTED: "Re=10000 protocol unaffected".
- v1.4 temporal pre-flight: temporal_compat_check() (LR lag-1 corr in [0.95,0.9985]) — flags all
  three new files GT-free. Trainer-side hard abort was offered and declined; check runs at launch
  via the driver instead.
- CORRECTIVE (user-directed): kf_re2000_256_40seed_dt32.npy = stride-80 frames {39,119,199,279}
  (lag-1 0.9871). Anchor v3 = frozen procedure on dt32 train LR (fit == v2, fingerprinted).
  D=0.330 -> temp 2.5. Base probe 0.394 on dt32 vs 0.337 raw. 600 iters running ->
  ddpo_re2000_dt32_ckpts; then REPORT-ONLY eval on burned 24-39 (5-7 stay sealed).

## 2026-07-23 — dt32 Re=2000 CLEARS THE BAR; Re=10000 dt32 launched (gate passed)
- Re=2000 dt32 report-only eval: ret(0.30)=1.166 IN-BAR (first bar-clearing result on the new
  generation), k*=95, hybrid lowk 0.988, placement 0.750 > base 0.698, GT residual 53.3 back
  inside [46,76]. Probe 0.394->0.762. The failure chain is closed: stale anchor (v1.3) + wrong
  dt (B9/dt32) were the whole story; the frozen recipe itself generalizes GT-free.
- Re=10000: file re-staged from GCS (CRC IvvIXQ== verified), dt32 strided (lag-1 0.9757 OK),
  obs-fit anchor built (kd 113.6 ~ prior 112.7; residual_ref 5180), D=0.163 -> temp 3.5,
  600 iters running -> ddpo_re10000_dt32_ckpts; report-only eval on 10-19; 25-39 SEALED.
- Seal-eval script now parameterized: SEAL_GT/ANCHOR/CKDIR/TEST/NPER/RE/TRAIN.

## 2026-07-23 (later) — Re=10000 dt32: all §6 regime bars MET (report-only)
- ret(0.30)=1.183 in [0.75,1.30]; k* 30->69 (>=55 bar); hybrid lowk 0.994; placement 0.641 <
  base 0.702 = §6 info-ceiling prediction CONFIRMED. No low-k bleed (kc=2, hybrid==DDPO).
- Floor law over-predicts 13x at Re=10000 (5180 vs measured 400; grid-limited data likely) —
  safe direction, no training impact; needs grid correction in the fresh-repo spec.
- Full corrected stack (dt32 + procedure-rebuilt anchors + frozen recipe) now works GT-free at
  BOTH extrapolation regimes. Sealed 25-39 still available for a one-shot confirmatory.

## 2026-07-23 — 40-seed sequences are ORDERED SEGMENTS, not independent seeds
- Adjacent-seq corr ~0.6 (matched frames), background ~0.1 by index distance 5-8; old 20-seed
  file = 0.095 baseline. Re=2000 dt32 grading clean (d>=20); Re=10000 dt32 test 16-19 adjacent to
  train 20-23 (max 0.64) -> clean-subset re-grade on 10-15; sealed 25-26 within corr length of
  probe/train (noted in C5, split unchanged). Effective test n ~4-6 states per 16 seqs.

## 2026-07-23 — protocol v1.5 (§8): user-directed lock corrections
- SINGLE training temp 2.5 for all OOD (R3.1 buckets removed from deployment path — their
  calibration used GT-informed optima ON the target regimes = circular; "3.5 at Re=10000" evidence
  came from the broken fine-dt run). D recorded as diagnostic only.
- itemp 0.30 stays frozen headline; 1.0 = reference only; R3.4' idea dropped.
- Test-set independence rule (§8.4): picks spaced >=5 indices, >=5 from train/probe, ALL frames
  graded; LR cross-corr recorded at launch. Measured: re2000 picks {24,29,34,39} corr 0.102 (clean);
  re10000 picks {0,5,10,15} corr 0.218 (best available non-sealed; background 0.1).
- Validation chain LAUNCHED: (A) re2000 decorrelated re-grade; (B) re10000 RETRAIN at temp 2.5;
  (C) re10000 decorrelated eval. Sealed 25-39 untouched. Next-data-drop spec in §8.6.

## 2026-07-23 — v1.5 validation verdict: single temp 2.5 fails at Re=10000 (0.498 vs 3.5's 1.183)
- Re=2000 decorrelated re-grade: 1.160 — headline robust. Re=10000 locked-temp retrain: ret 0.498,
  k*41 — the exploration deficit at D=0.163 is real; paired, dt-correct, same-seed evidence.
- Decision pending (§8.7): single-temp honesty vs frozen two-point deficit rule with spent
  calibration regimes. Sealed 25-39 remains the only untouched validation ground.

## 2026-07-23 — kc rule adopted; temp-rule in-dist derivation closed off (§8.8)
- R3.3' reward-selected k_c VALIDATED (in-dist flat/safe; Re=2000 interior max at 7 = crossing
  rule's pick, best lowk 0.988). Adopted with R3.3 as flat-landscape fallback.
- In-dist ladder: plateaus 0.657/0.673/0.672/0.682 for temps 1.5/1.75/2.0/2.5, GT ret
  0.893/0.987/1.023/0.877 — no in-dist failure to calibrate on; plateau scale regime-dependent;
  fine plateau != fine quality. Blind plateau = coarse under-exploration detector only
  (Re=10000: 0.60 vs achievable 0.79). Escalation rule remains §8.7(b) — OOD-calibrated.

## 2026-07-23 — greedy-policy sweep (§8.9): itemp curve saturated below 0.30
- Both dt32 models, decorrelated picks: ret drifts +0.01 from 0.30 -> 0.0, all else flat. R3.4=0.30
  is on the greedy plateau. Bonus: Re=10000 temp-3.5 model on the cleanest picks {0,5,10,15} =
  1.073 — best Re=10000 number, in-band with margin.

## 2026-07-23 — one-shot Stage 1 (anchor-only) PASSED; Stage-2 config frozen
- Escalation fired (0.610<0.70 -> 3.5); R3.2 iter0549 (0.789); R3.3' kc=6; deployment vs anchor:
  plateau 0.798, low-k parity 0.998, [32,64) 1.146, tail 0.223 (anchor's extrapolated region,
  safe side); residual 36.3 << blind floor. NO GT touched; sealed 25-39 closed. Stage 2 = single
  sealed grading, frozen config, awaits explicit user go.

## 2026-07-24 — plateau trajectories + rule provenance audit (docs/anchor_derived_rules.md)
- In-dist trajectories computed (4 temps): healthy end-of-run 0.657-0.680; @1.5 crosses 0.65 only
  at iter ~260 -> iter-150 escalation checks misfire; rule amended to END-OF-RUN theta=0.65
  (in-dist-derived) + stall test (§8.11). OOD: Re2000 0.746->0.850 rising; Re10000@3.5
  0.683->0.789 rising; Re10000@2.5 FLAT 0.59-0.61 from iter 50 (the failure signature).
- Provenance: R1 success criterion in-dist; R2 escalation lever anchor-only (0.612->0.789, no
  GT); R3 kc anchor-only; R4 itemp in-dist. GT-free chain complete; sealed Stage 2 = the test.

## 2026-07-24 — SEALED ONE-SHOT PASSED (§8.12)
- ret(0.30): 0.758 sealed full / 0.792 clean subset — in-band (edge margin below noise floor,
  stated); base 0.193->3.9x, k* 32->61/62, lowk 0.992 PASS. Placement 0.21 vs base 0.73 =
  info-ceiling CONFIRMED (much starker than burned splits). Sealed segment measurably harder.
  GT-free chain validated one-shot; escalation rule passed out-of-sample.

## 2026-07-24 — placement decomposition (sealed set): hedge vs commitment
- Base map = 18% of GT hi-k energy, input-locked shadow -> corr 0.73 (amplitude-blind metric rewards
  the hedge). DDPO carries the same shadow (corr 0.40-0.49 vs base map at all scales) + 71% committed
  energy at input-undetermined positions -> corr 0.21; heavy smoothing recovers only to 0.32 (coarse
  arrangement unconstrained = true info ceiling, not jitter). Nothing destroyed: base recon exists in
  the cascade; sample-averaging recovers hedged maps. fig_sealed_placement_decomp.png in report.

## 2026-07-24 — lower-Re backtest + R5 inference-depth rule (dose-depth compensation)
- OOD models carry their training dose: 1.52x/3.92x/2.53x overshoot at full depth on milder
  regimes. In-dist model on virgin seqs 0/16: 0.937/0.906 (first fully-virgin in-dist eval).
- R5: blind depth selection at the regime's healthy set-point (theta = verified-good native
  reading; 0.68 in-dist ruler decomposed as 0.78 anchor-bias x 0.88 mid-band shortfall). Blind
  picks: K2->0.870 (2x@1000), K1[100]->1.059 (10x@1000), K2->1.230 (10x@2000). Naive parity
  target selects the overshoot every time. docs/anchor_derived_rules.md R5.

## 2026-07-24 — R5 corrected (user-caught GT leak in the set-point) + fully-blind validation
- Set-point reformulated: obs-fit factor-1~1 by construction x in-dist factor-2 0.85 -> band
  [0.80,0.85] on the anchor's source pool. Blind check @re2000: selects K2 (GT-best, 1.230) with
  zero GT. Guards (resolved-vs-LR, resid-vs-floor) non-discriminative here; anchor score selects.

## 2026-07-30 — overnight: R6 verdict, R5 refuted out-of-sample, Re=4000/5000 generated
- R6 Re=2000: early stop FIRED at iter 349 (0.827->0.834 in band). Graded on the same decorrelated
  picks as the 600-iter original: ret 1.138 vs 1.160 (dRet 0.022, inside +-0.04 noise), |ret-1|
  0.138 vs 0.160 (marginally BETTER), place 0.812 vs 0.817, lowk/k* identical. => 42% training
  compute saved for indistinguishable quality.
- R6 Re=10000: stop never fired — STRUCTURAL, not a failure: the band's lower edge (0.798) was
  measured from this very regime's healthy model, so its run can only enter the band at the end
  (touched 0.798 at iter 549, closed 0.794). A pooled band makes each regime stop at the WEAKEST
  regime's healthy level -> asymmetric savings. Its ckpts are BITWISE IDENTICAL to the original
  run's => the anchor monitor is provably read-only and the pipeline is seed-reproducible.
- BUGS: (1) post-loop save relabelled early-stopped runs as iter0599 (fixed 14c930e);
  (2) the overnight quarantine step then moved re10000's LEGITIMATE iter0599 out of its dir —
  restored and verified (iter field 599, params != iter0549). Quarantine must be conditional.
- NEW DATA: gen_fnons_re4000/re5000_kf_1024to256_20seq.npy (20x320). Pre-flights PASS: lag-1
  0.973/0.978 (dt-compatible, no stride needed); seq independence max 0.218/0.233; std 5.10/4.96.
  Four untouched regimes now available (1500, 3000, 4000, 5000) — enough to test whether a
  better-behaved anchor makes the set-point portable.

## 2026-07-30 — MAX-N RE-GRADE (user-demanded robustness): all OOD claims hold or improve
Basis: 35 sequences x 2 triplets = 70 triplets (the dt32 file CEILING — see limitation below).
Errors are bootstrap-BY-SEQUENCE (the honest unit; the 2 triplets per sequence overlap in 2 of 3
frames and are NOT independent).
Re=2000            ret            lowk   place  k*   blind
  base recon       0.385 +-0.007  0.989  0.666  31   0.846
  DDPO 600-iter    1.155 +-0.021  0.922  0.744  95   1.042   (hybrid kc=7 -> lowk 0.988)
  DDPO R6 (349it)  1.148 +-0.020  0.945  0.748  95   1.021   (hybrid kc=7 -> lowk 0.988)
Re=10000
  base recon       0.210 +-0.005  0.989  0.703  30   0.557
  DDPO 600-iter    0.934 +-0.044  0.998  0.221  65   0.771   (hybrid kc=6 -> lowk 0.990)
VERDICTS:
- R6 EARLY STOP CONFIRMED ROBUST: 349 vs 600 iters, dret = 0.007 vs SE 0.020 -> indistinguishable;
  R6 slightly BETTER on low-k bleed (0.945 vs 0.922). 42% compute saving is real. Note this
  comparison is independent of the set-point circularity — it only says iter349 deploys like iter599.
- Re=2000 retention robust: 1.155 at n=70 vs 1.160 (n=8) / 1.166 (n=32). REVISED: placement is
  0.744, not the 0.817 from the 8-triplet pool; still above base (0.666).
- Re=10000 retention 0.934 +-0.044 at max n. The sealed 0.758 was a HARD-SEGMENT reading (gap
  0.176 = 4x SE, so real difficulty, not noise). Sealed remains the valid one-shot; 0.934 is the
  better typical-performance estimate. Both in the §6 bar.
- Re=10000 PLACEMENT COLLAPSE IS THE ROBUST RESULT: 0.221 at 35 seqs vs base 0.703. The
  reassuring 0.640 from near-train picks {0,5,10,15} was TRAIN-PROXIMITY INFLATION. Info ceiling
  confirmed at 35 independent units.
- eval-sampling noise is SMALL on ret (ratio to GT over the same frames self-normalizes: +-0.005
  to +-0.044) but LARGE on the blind anchor score (fixed denominator: base recon reads 0.846 at
  n=70 vs 0.909 at n=8). Every selection rule depends on the noisy quantity — relevant to R5/R6.
LIMITATION (hard): dt32 files hold 2 triplets/sequence, so 70 is the ceiling; 636 triplets is
IMPOSSIBLE on this data. Also: TRAINING pool was 8 triplets (vs 636 in-dist, 1272 on the wrong-dt
data) — see the regeneration case.
OPEN (needs user): Re=10000 record-dt/stride (my extrapolation from the new dt=1/32 generations
favours stride 100 over the 80 I used, but only at ~2.5x the reference scatter); Re=10000 timestep
for regeneration. Test design for 636-triplet runs SETTLED: spread across ALL sequences (independent
units are set by sequence count, not triplet count — 636 triplets in 2 sequences = 2 units).
