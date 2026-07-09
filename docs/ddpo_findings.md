# DDPO physics-reward finetuning — honest findings (Re=1000, in-distribution)

**Bottom line.** DDPO with the calibrated physics reward measurably improves the **spectral statistics
it optimizes** (high-k energy retention, per-scale spectrum, effective resolution, PDE residual) but
does **not** improve the **pointwise reconstruction** of the flow (MSE/RMS error ≈ flat, vorticity
PDF slightly worse). The gains are real but live entirely in *phase-blind, energy-per-scale* metrics.
The reconstruction error to ground truth barely moves because the reward cannot fix **phase**
(where the small scales are), and phase is underdetermined by the sparse input. This is a genuine
win for turbulence *statistics* / spectral fidelity, and **not** a win for pixel-accurate
reconstruction.

---

## 1. What was run

- **Base:** unconditional Re=1000 DDPM (`ckpt_epoch_0299`), sparse-reconstruction task
  (BaratiLab `sparse_nnfill`, 1024 points), **single denoising chain, SDEdit `t_start=100`**
  (NOT the 3-stage `S=[150,100,50]` procedure — an open follow-up).
- **DDPO:** `src/ddpo_ft/` — 4-chip pmap data-parallel PPO, per-input advantage, KL-to-base,
  `sampling_temp=1.5`. Reward = calibrated components with reweight for the "spec-vs-pde plateau":
  `spec_highk`=3.0 (dominant), `spec`=0.5, `energy`=0.1, **`w1`=0 (not trained)**, `pde`=1.0 with a
  one-sided hinge. Best models: continuation (`lr=5e-5`) and a stabilized fresh run (`lr=1.5e-5`).

## 2. Results — val (32–35) + test (36–39), DDPO vs base

| metric | base | DDPO | Δ | what it measures |
|:--|--:|--:|--:|:--|
| **hi-k retention** `R(k≥32)` | 0.64 (test) | 0.72 | **+0.08** | high-k **energy ratio** vs GT (phase-blind) |
| effective resolution `k*` | 41 | 48 | **+7** | finest scale reconstructed (energy) |
| PDE residual | 62.8 | 52.1 | **−10.7** | NS consistency (toward GT floor ~12) |
| **MSE** (recon vs GT) | 0.1585 | 0.1575 | **−0.001 (~<1%)** | **pointwise** fidelity |
| **RMS(field − GT)** | 2.14 | 2.13 | **−0.4%** | **pointwise** fidelity |
| vorticity PDF `W1` | 0.229 | 0.251 | **+0.022 (worse)** | pointwise value distribution (untrained) |
| total enstrophy retention | 0.90 | 0.90 | ~0 | total energy (peak-dominated) |

**The split is the whole story:** every *spectral / energy-per-scale* metric improves clearly; every
*pointwise* metric is flat-to-slightly-worse.

## 3. The residual analysis (the decisive check)

Comparing `base − GT` vs `DDPO − GT` (the reconstruction errors) directly:

- **RMS error: 2.14 → 2.13 (−0.4%)** — the two error fields are visually near-identical.
- **Error-reduction map** `|base−GT| − |DDPO−GT|`: essentially blank; **~50% of pixels "improved"**
  (a coin flip). DDPO lowers the error in half the pixels and raises it in the other half.
- **`corr(DDPO−base, GT−base) = +0.11`** — what DDPO *adds* is only weakly co-located with the
  *true* missing structure. The additions are the right *character* (fine filaments, concentrated on
  shear layers) but in the wrong *places*.

Figures: `monitoring/ddpo_ckpts/eval_spectrum.png` (spectrum — the "win"),
`viz_diff2.png` (residual/error maps — the flat reconstruction), `viz_vortpdf.png` (PDF).

## 4. Why — phase is the bottleneck, and a spectral reward can't fix it

The reward `d_spec_highk` and the retention metric are **phase-blind**: they score how much high-k
*energy* is present, integrated over shells, not *where* it is (Fourier amplitude, not phase). DDPO
therefore fills the energy deficit with plausible, correctly-shaped, but **mislocated** fine
structure — which:

- raises the spectrum and hi-k retention (energy is now present at the right scales),
- leaves the pointwise error unchanged (the structure is in the wrong positions, so it doesn't
  cancel `base − GT`),
- slightly perturbs the vorticity PDF (untrained; extra energy shifts the marginal off GT).

The exact fine-scale **phase is underdetermined by the sparse observation** — 1024 scattered points
cannot fix where each filament sits — so **no marginal-spectrum reward can recover it.** Matching the
spectrum is *necessary but not sufficient* for reconstruction (Parseval applies to the *error*
spectrum, which the reward deliberately does not use — see `docs/ddpo_reward_math.md` §2).

## 4b. The knob proves it — spectral-vs-pointwise tradeoff (checkpoint sweep, test set)

Sweeping training strength (gentler `lr=5e-5` continuation vs harder fresh `lr=1.5e-5` run to iter 199):

| model | hi-k retention Δ | MSE Δ | residual Δ | `k*` Δ |
|:--|--:|--:|--:|--:|
| continuation (gentle) | +0.078 | **−0.001** | −10.7 | +7 |
| overnight iter119 | +0.115 | +0.0018 | −7.8 | +15 |
| overnight iter159 | +0.125 | +0.0012 | −9.7 | +18 |
| overnight iter199 (hard) | **+0.148** | **+0.0011** | −9.5 | **+22** |

Pushing the spectral reward harder monotonically **raises hi-k retention and `k*`** while **worsening
MSE**. The gentle model is the only one that lowers MSE at all; the harder models buy ~2× the spectral
gain by *increasing* pointwise error. This is the phase mechanism made a controllable knob: extra
high-k energy, added in the wrong places, improves the (phase-blind) spectrum and *degrades* the field
match. Choose the operating point by goal — spectral fidelity (hard) vs reconstruction accuracy (gentle).

## 5. Honest correction to earlier framing

Initial write-ups called this "works decisively (MSE down, retention up)." That over-weighted the
*sign* of the MSE change over its *magnitude*: MSE improved by **<1%** — statistically real,
practically flat. The headline hi-k retention "+0.08" is a **phase-blind energy ratio in the
<2%-of-total-enstrophy tail** — impressive-sounding but a small part of the field. The residual /
error-map view (§3) is the honest arbiter and shows the reconstruction is essentially unchanged.

## 6. Implications / next levers

- **If the goal is turbulence statistics** (energy spectrum, high-k content, `k*`, spectral fidelity
  for e.g. sub-grid modeling): this is a legitimate, measurable improvement — use it.
- **If the goal is pointwise reconstruction accuracy**: the spectral reward is the wrong lever. Options:
  1. **Phase/position-aware objective** — keep paired MSE in the mix, or add a two-point
     correlation / structure-function term (sees spatial arrangement, not just the marginal spectrum).
  2. **Richer observations** — the sparse input caps recoverable phase; denser sampling raises the ceiling.
  3. **Re-add `w1`** (weight >0) if the vorticity PDF matters — it is currently untrained and regressed.
- **Untested and worth doing:** evaluate the best model under the **official 3-stage recon**
  (the finetuning + eval used a single chain); the OOD regimes (Re=500/2000) where the *spectral*
  deficit is largest and this reward has the most leverage.

## 7. What is solid

The **pipeline and method are validated end-to-end**: reward calibration → 4-chip DDPO → a real,
reproducible improvement in the trained (spectral) objective, on held-out test data, without reward
hacking (physics stayed clean). The limitation is not a bug — it is the correct, expected behavior of
a phase-blind reward on a phase-underdetermined task, now measured rather than assumed.

---

# Part II — OOD extrapolation & the placement experiments

## 8. OOD: DDPO transfers to Re=2000 (measured anchor)

Finetuning the Re=1000 base **toward Re=2000** (sparse inputs from degraded Re=2000 sequences;
reward anchored to Re=2000 regime stats; no paired supervision anywhere). Full 20-sequence eval
(val 0–9 / test 10–19, indistinguishable to ±0.001):

| metric | base | DDPO | Δ |
|:--|--:|--:|--:|
| hi-k retention | 0.266 | 0.332 | **+0.067** |
| PDE residual | 83.0 | 75.9 | **−7.2 (−9%)** |
| MSE | 0.1905 | 0.1913 | flat |
| `k*` | 28 | 28.5 | ~0 |

Same shape as in-distribution: spectral gain real and transfers OOD, pointwise flat (phase), and the
physics term *improves* (−9% residual). The ceiling is capacity: base has 73% of Re=2000 hi-k energy
missing; DDPO closes ~1/12 of the deficit.

## 9. **Headline: extrapolated anchors work — zero target-regime data needed**

The Re=2000 anchor was rebuilt from **{Re=500, Re=1000} only** (`base_results/build_extrap_anchor.py`
→ `regime_stats_re2000_extrap.npz`): spectrum via the fitted cascade model with the dissipation
scale shifted by k_d ∼ Re^½ (hi-k `d_spec` to measured = **0.038** vs 1.27 for naively reusing the
Re=1000 anchor); enstrophy held (saturated, drag-controlled); PDF via universal shape × √enstrophy;
residual floor = owned Re=1000 floor × extrapolated hi-k enstrophy ratio. Controlled A/B — identical
seed/lr/length, only the anchor differs:

| | measured anchor | **extrapolated anchor** |
|:--|--:|--:|
| retention Δ (live probe, final) | +0.067 | **+0.075** |
| retention Δ (full eval, val) | +0.066 | **+0.073** |
| MSE | flat | flat |
| residual Δ | −7.2 | −3.6 |

The extrapolated-anchor run **tracked the measured one for all 200 iterations and finished ahead**.
Conclusion: DDPO finetuning toward an unseen regime requires **no high-res target data** — the
anchors are predictable from lower-Re simulations, as the feasibility study (commit 1a91dbf)
projected. The residual difference is the retention↔residual tradeoff (lower extrapolated floor →
slightly more energy added, slightly less residual cleanup), not a deficiency.

## 10. Localizer diagnostics — where corrections belong, and what is GT-free

Built to design a spatially-targeted reward (`viz_pde`, `viz_energy`, `diag_residual_locality`):

- **PDE residual field** (GT-free): corr +0.21 with the true error; corr with the **GT's own
  residual floor only +0.16** and 3.4× its magnitude → the model's residual is its *own* artifact,
  so penalizing it moves *toward* GT (legitimate GT-free target). Extremely local: correlation
  length ≈ 3 px, worst-10% of pixels hold 77% of total residual². High-shear regions are ×2.25
  residual-enriched but pixelwise corr(strain, R²) is only 0.07 — shear is a soft prior, not a mask.
- **Local hi-k energy map** (needs GT for the deficit): recon's own map conflates correct detail
  with error → not a clean GT-free target. Its **placement corr** `E_hik(recon)↔E_hik(GT)` = **0.44**
  for the base — the spatial-phase number that every subsequent experiment tried to move.

## 11. Placement experiment 1 — `pde_local` (worst-region residual penalty): **negative**

`make_pde_local`: magnitude-weighted mean residual² over the worst-`frac` P×P regions (patch=2
matches the 3 px residual scale). Controlled A/B vs plain DDPO (same seed/config, 100 iters, Re=1000):

| (val) | pde_local | plain DDPO |
|:--|--:|--:|
| retention Δ | +0.071 | **+0.096** |
| residual Δ | −9.5 | −9.0 |
| `k*` Δ | +8 | **+14** |
| placement | 0.436 | 0.440 |

**No extra residual cleanup** (the whole-field `pde` term already does it all), **no placement
change**, and it *cost* retention/resolution. Removing wrong structure does not teach the model
where missing structure belongs. Dead end, kept in the codebase as a documented negative.

## 12. Placement experiment 2 — strain-based rewards (`diag_strain` gated the design)

Premise: filaments are created by large-scale strain, and the strain field **is** resolved
(corr(σ_recon, σ_GT) = **0.82**). Two candidates diagnosed before building:

- **Strain-placement** (push energy into strain regions): **killed at diagnostic** — the recon's
  strain predicts GT's energy locations at only 0.26, *worse* than the recon's own energy already
  does (0.44, negative headroom), and the recon is already as strain-consistent as GT (0.33 vs 0.31).
- **Strain-orientation** (Batchelor filament coherence): **live signal** — GT's ∇ω is strongly
  organized relative to the local strain eigenframe (alignment stat **0.289**; isotropic = 0.5)
  while the recon sits at **0.394**: the energy DDPO adds is orientation-random speckle.

## 13. Placement experiment 3 — `align` reward (Run C): **marginal positive, keep it**

`make_alignment_distance`: d = ((a(x) − 0.289)/0.105)², a(x) = |∇ω|²-weighted cos²(∠(∇ω, compressive
strain axis)) — GT-free at reward time (reference measured once from owned Re=1000 GT). Run C, same
seed/config as the A/B (w=2.0):

| (val) | pde_local (A) | plain (B) | **align (C)** |
|:--|--:|--:|--:|
| retention Δ | +0.071 | +0.096 | **+0.098** |
| `k*` Δ | +8 | +14 | **+20** |
| residual Δ | −9.5 | −9.0 | −8.8 |
| placement | 0.436 (−.004) | 0.440 (−.001) | **0.446 (+.005)** |
| orientation stat | — | 0.389 | **0.386** (GT 0.289) |

The **only** reward to move placement in the positive direction, best `k*`, tied-best retention,
no cost — but small: it moved its own orientation target just ~7% of the gap (policy-gradient
exploration is isotropic, so better-organized samples are rare → weak advantage signal).
**Keep `align` in the reward (free improvement); do not expect it to break the ceiling.**

## 14. The ceiling, now established on three controlled experiments

Residual targeting (negative), strain-placement (killed at diagnostic), orientation coherence
(~+1%): DDPO robustly fixes **statistics** (spectrum, retention, k*, residual — and can do so OOD
from extrapolated anchors), nudges **structure** marginally, and cannot fix **position** — placement
corr is frozen at ≈0.44 because fine-scale phase is not in the sparse input. Any further placement
gain must come from **more information** (denser observations, temporal context, measurement-
consistency enforcement at the sensor locations), not from a smarter GT-free reward.

A note on the "retrain with energy+PDE loss" idea: a *deterministic* physics loss regresses to the
posterior mean and over-smooths exactly like MSE — the generative sampler is what buys sharpness.
If retraining, stay generative; but the placement cap is an information limit and will persist.
