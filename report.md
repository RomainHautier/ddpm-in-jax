# Why our base DDPM is spectrally conservative — the EMA hypothesis

**Handoff document** (2026-07-15). Context for another Claude instance / collaborator. Full
experimental narrative lives in the results artifact (§1–§21) and `docs/ddpo_backlog.md`; this file
isolates one finding and its likely root cause.

---

## 1. The measured behavioral gap

Two nominally identical DDPMs — same U-Net architecture, same Kolmogorov-flow dataset
(`kf_2d_re1000_256_40seed.npy`), same train split — behave very differently under **identical
inference procedures** (BaratiLab's shipped ladder sampler, replicated from source; their sparse-
reconstruction benchmark, 320 test frames, scored against their reference):

| model, same sampler (ladder t=400/r=20) | hi-k retention | PDE residual | MSE | placement |
|---|---|---|---|---|
| BaratiLab's released checkpoint (their pred_it2*) | 0.468 | 2.52 | 0.1578 | 0.343 |
| our `ckpt_epoch_0299` | 0.207 | 1.31 | 0.1453 | 0.461 |
| ours, ladder t=240/r=30 (their README args) | 0.307 | 1.77 | 0.1388 | 0.397 |

\* their banked run's exact (t, r) args are unrecorded; both plausible arg sets were tested on our
side — neither closes the texture gap.

**Signature:** our base is far more *conservative* — better MSE (94–100% of frames), better
placement, near-floor residual, but a spectrally collapsed output (retention 0.21–0.36 vs their
0.47; k* 11–22 vs 35). Their model commits to fine-scale texture; ours shrinks it away. The gap is
model-level, not sampler-level (same code path produced both rows).

Downstream relevance: our DDPO finetunes inherit this base. They *compensate* (the reward re-adds
the missing energy — the finetuned system beats their full system on every aggregate, report §19,
§20b), but the base's conservatism is plausibly the root of the one residual weakness: sub-Nyquist
mid-band (k 10–32) conservation, where their base holds more of the input's structure (report §14,
§21).

## 2. Root cause candidate: EMA weights (they use them; we don't)

**Their side (verified in source, repo `BaratiLab/Diffusion-based-Fluid-Super-resolution`, branch
`main_v1`):**

- Training config `train_ddpm/configs/km_re1000_rs256.yml:24-25`:
  ```yaml
  ema_rate: 0.9999
  ema: True
  ```
- EMA implementation `train_ddpm/models/ema.py` (`EMAHelper`): a shadow copy of every parameter,
  updated each optimizer step as
  `shadow = mu * shadow + (1 - mu) * param`  (mu = 0.9999).
- Training loop `train_ddpm/runners/diffusion_tub.py:134,200,210`: helper registered, updated every
  step, and its state dict **appended as the LAST element** of the saved checkpoint list.
- **Inference** `runners/rs256_guided_diffusion.py:299`:
  ```python
  model.load_state_dict(torch.load(self.config.model.ckpt_path)[-1])
  ```
  `[-1]` = the EMA weights. **All their published reconstructions run on the EMA model.**

**Our side:** `src/train_ddpm.py` contains no EMA (no shadow weights, no averaging of any kind).
`checkpoints/ddpm/ckpt_epoch_0299.pkl` is the raw online weights at the moment training stopped.

## 3. What EMA is and why it matters here (primer)

**Mechanism.** Keep a second, passive copy of the model weights θ̄ ("shadow"). After every
optimizer step: θ̄ ← μ·θ̄ + (1−μ)·θ. Train with θ as usual; **evaluate/sample with θ̄**. With
μ = 0.9999 the shadow is an exponentially-weighted average over roughly 1/(1−μ) = 10,000 recent
steps. Cost: one extra parameter copy in memory during training, zero cost at inference.

**Why diffusion models specifically need it.** SGD leaves the online weights θ jittering around the
loss basin; each step is the last minibatch's noise. A diffusion model's sampler output is an
iterated composition of hundreds of ε-predictions, so per-step weight noise compounds. More
precisely: the model's clean estimate x̂₀ is a posterior-mean estimate, and where the learned score
is noisy/uncertain the estimate behaves like a Wiener filter — it *shrinks the modes it is unsure
of*. Fine-scale (high-k) modes have the lowest SNR in training, so weight noise expresses itself as
**systematic over-smoothing of fine detail** — exactly the conservative signature in §1. EMA
averages away the SGD jitter, yielding an effectively sharper, more committed score at high k.
Empirically EMA is one of the largest single levers on diffusion sample quality (see refs), which is
why every canonical DDPM implementation ships it on by default.

**References:**
- Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models*, NeurIPS 2020, arXiv:2006.11239 —
  the canonical DDPM; trains with EMA rate 0.9999 (their Appendix B). Their released checkpoints
  are EMA weights — the convention BaratiLab inherits.
- **Song & Ermon, *Improved Techniques for Training Score-Based Generative Models*, NeurIPS 2020,
  arXiv:2006.09011 — the primary reference to read**: Section 3 ("technique 4") is a dedicated
  analysis of EMA for score models, with ablations showing large FID gains and stabilized sampling.
- Nichol & Dhariwal, *Improved DDPM*, ICML 2021, arXiv:2102.09672 — EMA as standard practice.
- Karras et al., *Analyzing and Improving the Training Dynamics of Diffusion Models*, 2023/24,
  arXiv:2312.02696 — modern deep-dive; introduces **post-hoc EMA** (reconstructing arbitrary EMA
  lengths from saved snapshots) — directly relevant to our situation below.
- (Background on weight averaging generally: Izmailov et al., *Averaging Weights Leads to Wider
  Optima*, UAI 2018, arXiv:1803.05407.)

## 4. What we can do about it (ordered by cost)

1. **Post-hoc weight averaging — testable TODAY, no retraining.** We saved epoch snapshots every 10
   epochs; the full series `ckpt_epoch_0009 … 0299` is at
   `gs://ddpm-thesis-rh/checkpoints/ddpm/` (only 0299 is on local disk). Averaging the last N
   snapshots (e.g. 0249–0299, uniform or exponentially weighted) is a coarse EMA surrogate (à la
   SWA / Karras post-hoc EMA, though our 10-epoch snapshot spacing is far coarser than per-step
   EMA — treat it as a cheap directional probe, not the real thing). **Verification protocol:** drop
   the averaged weights in as `base_params` and rerun the §20/§20b base rows (script:
   `scratchpad/their_ladder.py` / `baseline_assessment.py` pattern) — watch retention/k* on their
   benchmark and the mid-band (k 10–32) ratios on ours. If retention moves 0.21 → 0.3+ the
   hypothesis is confirmed cheaply.
2. **Add EMA to `src/train_ddpm.py` and retrain the base** (μ = 0.9999 to match; trivial to
   implement in JAX/optax — `optax.ema` or a manual shadow pytree). This is the real fix. Also fixes
   the snapshot-retention policy: save both online and EMA states (like their `states` list) and
   keep the per-epoch series.
3. **Re-run the DDPO stack on an EMA base.** Every finetune inherits the base; if the base stops
   shrinking high-k, (a) the finetuned amplifier may need less gain (retrain or just re-evaluate
   existing recipes), (b) the sub-Nyquist mid-band conservation gap (§14's k=32 Nyquist wall, §21's
   mid-band deficit vs their model) is the metric most likely to move — it is exactly
   "input-texture the model declines to keep".

## 5. Open questions for discussion

- Is 10-epoch snapshot spacing sufficient for a meaningful post-hoc average, or does the coarse
  spacing wash out (Karras post-hoc EMA assumes dense snapshots)?
- Should the EMA retrain also revisit training length? (Their released kmflow checkpoint's step
  count is unknown; our 300 epochs may also simply be shorter.)
- After an EMA base exists: do the DDPO recipes (K2-chain, deep cascade, brake) need re-tuning, or
  do they transfer? (Hypothesis: brake μ and reward scales transfer; the amplifier's learned gain
  may be excessive on a sharper base.)

**Repo pointers recap** — theirs: `train_ddpm/models/ema.py`, `train_ddpm/runners/diffusion_tub.py:134/200/210`,
`train_ddpm/configs/km_re1000_rs256.yml:24`, `runners/rs256_guided_diffusion.py:299`. Ours (no EMA):
`src/train_ddpm.py`; single local checkpoint `checkpoints/ddpm/ckpt_epoch_0299.pkl`; full series on GCS.

---

## 6. RESULT: the post-hoc averaging probe (run 2026-07-15)

Option 1 from §4 was executed — uniform average of the last 6 (ep 249–299) and last 11 (ep 199–299)
snapshots, evaluated against raw `ckpt_epoch_0299`:

| config | ret | resid | MSE | place | k[10,20) | k[20,32) | k[32,64) | k[64,96) |
|---|---|---|---|---|---|---|---|---|
| **their ladder, their benchmark** | | | | | | | | |
| BaratiLab pred_it2 (target) | 0.468 | 2.52 | 0.1578 | 0.343 | 0.82 | 0.63 | 0.47 | 0.44 |
| raw 0299 | 0.209 | 1.31 | 0.1454 | 0.460 | 0.42 | 0.26 | 0.21 | 0.17 |
| avg6 | **0.246** | 1.28 | 0.1468 | 0.445 | 0.45 | 0.29 | 0.25 | 0.20 |
| avg11 | 0.245 | 1.34 | 0.1476 | 0.445 | 0.46 | 0.30 | 0.25 | 0.20 |
| **our grid-4× benchmark, base K1** | | | | | | | | |
| raw 0299 | 0.385 | 3.36 | 0.0159 | 0.819 | 0.86 | 0.63 | 0.39 | 0.27 |
| avg6 | **0.400** | 3.38 | **0.0157** | 0.819 | 0.86 | 0.64 | 0.41 | 0.29 |

**Reading.** Every spectral band rises under weight averaging, on both benchmarks, at zero cost
(MSE flat-to-better, residual flat, placement ~flat): retention +18% relative on their ladder
(0.209 → 0.246), +0.015 on our K1 row. The wider window (avg11) plateaus rather than hurts. This is
the *coarsest conceivable* EMA surrogate — snapshots 10 epochs apart, uniform weights — and it still
closes ~15% of the texture gap to their checkpoint. **Directional confirmation of the EMA
hypothesis**: a true per-step EMA (μ=0.9999) during training is well-motivated as the real fix;
expected effect substantially larger than the probe. Figure:
`monitoring/ab_pdelocal/ema_probe_spectra.png`; log: `.../ema_probe.log`; probe script:
(session scratchpad) `ema_probe.py`. Averaged weights are trivially rebuildable from the GCS series.

## 7. Update (2026-07-16): exact-args correction sharpens the hypothesis

The conditional run report revealed the banked baseline's true args (t400/r150/step3). Rerunning our
base at exactly those args: ret 0.289 (vs 0.209 at the r=20 defaults) — finer stepping explains ~30%
of the previously-reported gap, but the remaining model-level difference (0.289 vs their 0.468, now
under matched-everything) stands, and remains EMA-shaped: pure high-k texture, while our base wins
MSE (0.1288, best of any config on their benchmark), residual (1.64) and placement (0.464). Their
physics-conditioned model (as released) scored ret 0.101 / resid 1.79 / MSE 0.1386 — conservative
extreme; no threat to the finetuned stack. See results artifact §22.

## §8 EMA arc — outcome (2026-07-17)

Retrain executed (300 ep, mu=0.9999): retrain lifts the base everywhere (their ladder 0.209->0.240,
K1 0.385->0.401, true-args 0.289->0.314); EMA shadow ~= online at convergence (cleaner resid, hair
lower ret) — the EMA mechanism is real but small at full convergence; the remaining gap to their
conditional (0.468 ret at matched args) lives elsewhere. DDPO restack on the EMA base: probe floor
0.420->0.438, final 0.662; flagship deep+l3 0.553->0.629 (their bench) / 0.819->0.872 (grid-4x);
OOD Re=2000 cross-regime 0.518->0.555. New EMA-era failure mode: tail k[64,96) overshoot (up to
1.77x under their dense ladder); hinge-brake sweep shows saturation on cascades (anchor-binding,
mu-insensitive) -> in-dist flagship runs unbraked; next lever = [64,96)-only hinge. Full tables:
claude.ai/code/artifact/816fb598-398e-4fa4-ad42-7fdf8b86b921
