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
