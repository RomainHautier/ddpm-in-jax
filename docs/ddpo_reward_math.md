# DDPO reward components — mathematical specification

Exact definitions of the four reward families in `src/rewards.py`, as implemented, with the
reasoning behind each choice flagged so it can be challenged. Calibration evidence:
`base_results/reward_calibration.ipynb`; spectral background: `docs/energy_spectrum_metric.md`.

---

## 0. Setup and conventions

**Sample format.** Every reward maps a *normalized vorticity triplet* to a per-sample scalar:

```
x ∈ R^(N×N×3),  N = 256,   x = (ω − μ)/σ,   μ = 0, σ = 4.7988
```

σ, μ are the model's training stats (Re=1000, first-32-seq split) and are used for **all** regimes
(repo convention — the model only ever sees this normalization). Channels are consecutive frames
`(ω_{t−1}, ω_t, ω_{t+1})`; statistical rewards use the **middle frame** `x_mid = x[...,1]` only;
the PDE residual uses all three. Rewards are evaluated on the final denoised sample x₀ only —
DDPO needs scalars, not gradients (everything is jax-differentiable anyway).

**Spectral conventions** (identical to `lambda_sweep_re1000.spectrum_fn`): unnormalized `fft2`,
integer wavenumbers `k = fftfreq(N, 1/N) ∈ {0,…,127,−128,…,−1}` on the torus `(0,2π)²`, radial
shell index `κ(k) = round(|k|)`. The **enstrophy (vorticity-power) spectrum** of the normalized
middle frame is the shell sum

```
E(k) = Σ_{κ(k')=k} |x̂_mid(k')|² ,      k = 0 … N/2−1 = 127
```

Parseval (numpy convention `Σ_k |x̂|² = N² Σ_x x²`) gives `Σ_k E(k) = N⁴ · mean_x(x_mid²)`, so
E(k) is total (normalized) enstrophy decomposed by scale. The KE spectrum is the same data
reweighted by `1/(2k²)`; we use the enstrophy convention because it weights exactly the high-k
band where over-smoothing lives (see energy_spectrum_metric.md §2).

**Convention: distances, not rewards.** Each component is a distance `d(x) ≥ 0`, lower = better;
the combined DDPO reward negates them (§6).

---

## 1. Regime anchors (`compute_regime_stats`, rewards.py:61)

All rewards are **reference-free w.r.t. paired GT**: they are anchored to *regime-level statistics*
computed once from any M frames of the target regime (train split, or a generated run at the
target Re). From frames {x_m}:

```
spec_ref(k)      = (1/M) Σ_m E_m(k)                      (arithmetic mean — kept, not used)
log_spec_ref(k)  = (1/M) Σ_m log E_m(k)                  (geometric-mean anchor — used)
enstrophy_ref    = (1/M) Σ_m mean_x(x_m²)
quantiles_ref(j) = (1/M) Σ_m F_m^{-1}(q_j),   q_j = j/(Q−1),  j = 0…Q−1,  Q = 257
```

where `F_m^{-1}` is frame m's empirical quantile function (np.quantile, linear interpolation).

**Why the geometric mean for the spectrum anchor** (calibration finding): per-shell frame spectra
are approximately log-normally distributed, so the arithmetic mean lies *above* the typical
frame's spectrum. With an arithmetic anchor, a *slightly blurred* frame scored better than GT
(blur σ=0.5 gave d_spec = 0.09 vs GT floor 0.24) — i.e. the reward paid the model to smooth. The
geometric mean is the minimizer of the expected squared log-distance, putting the typical frame
at ≈ the floor. Challenge point: a median anchor or per-shell variance weighting (Mahalanobis in
log space) would be the next refinements.

---

## 2. Spectrum matching — `d_spec`, `d_spec_highk` (rewards.py:106)

```
d_spec(x) = (1/|B|) Σ_{k∈B} [ log ẽ(k) − log_spec_ref(k) ]²

ẽ(k) = max( E(k), ε·exp(log_spec_ref(k)) ),   ε = 10⁻⁶  (relative floor)
B    = {1,…,95}   for d_spec        (full resolved cascade)
B    = {32,…,95}  for d_spec_highk  (the deficit band, k ≥ 32 ≫ forcing k_f = 4)
```

- **Log-space**: E(k) falls ~3 decades across the band; a linear metric would be dominated by the
  first few shells and blind at high k — reproducing exactly the failure of L2. In log space every
  decade of the cascade counts equally: this is a *per-scale relative energy error*.
- **Band**: k=0 excluded (mean mode, ≈0 on normalized data); k ≥ 96 excluded because GT energy →
  0 there and log-ratios become noise (energy_spectrum_metric.md §5).
- **Relative floor ε**: a hard-zeroed shell contributes a bounded `(ln 10⁻⁶)² ≈ 191` instead of an
  epsilon-driven blowup; keeps hard low-pass corruption on the same scale as everything else.
- **What it cannot see** (by construction, proven by the calibration scramble probe): phases. Any
  field with the reference spectrum — including spectrum-matched noise — has d_spec at the GT
  floor. Matching marginal spectra is necessary, not sufficient (Parseval argument in
  energy_spectrum_metric.md §3 applies to the *error* spectrum, which this deliberately is not).

Relation to the tracked metric: hi-k retention `R(k≥32) = Σ_{k≥32}E_recon / Σ_{k≥32}E_GT` is a
GT-paired linear band ratio; d_spec_highk is its reference-free log-space counterpart. Calibration:
rank correlation within-input ≈ −0.7, model-level −1.0.

---

## 3. Energy conservation — `d_energy` (rewards.py:128)

```
d_energy(x) = [ ln( mean_x(x_mid²) ) − ln( enstrophy_ref ) ]²
```

Total normalized enstrophy vs the regime mean, as a squared log-ratio (symmetric: over- and
under-energized both penalized; a ratio of e^±1 costs 1). By Parseval this is the k-integral of
the spectrum, so it is *scale-blind* — it pins the overall level while d_spec fixes the
distribution across scales. Deliberately redundant with d_spec's band integral, except that
d_energy also sees k=0 and k≥96 and weighs shells by energy rather than per-shell equally.

Note this is **enstrophy** conservation, not kinetic-energy conservation. The KE analogue would be
`Σ_k E(k)/(2k²)` (low-k weighted). Enstrophy was chosen for loss-alignment and high-k sensitivity;
challenge point if you want the conserved-quantity argument to be about KE in the inverse-cascade
range instead. Calibration: weak sensor at Re=1000 (recon/floor 1.8×), monotone only on the noise
ladder — kept as a cheap guard (weight 0.25).

---

## 4. Vorticity distribution — `d_w1` (rewards.py:141)

1-Wasserstein distance between the sample's pointwise vorticity marginal and the regime reference,
via the quantile representation

```
W₁(F, G) = ∫₀¹ |F⁻¹(q) − G⁻¹(q)| dq
```

discretized at the Q = 257 evenly spaced ranks:

```
d_w1(x) = (1/Q) Σ_j | x_sorted[ r_j ] − quantiles_ref(j) |,   r_j = round( j·(N²−1)/(Q−1) )
```

(x_mid's N² pixels sorted ascending; sample side uses nearest-rank order statistics, reference
side linear-interpolated quantiles — a minor asymmetry, negligible at N² = 65536 ≫ Q.)

Units: normalized vorticity. Sensitive to PDF shape including the tails (extreme-vorticity
filaments that over-smoothing clips), blind to *where* values sit spatially and to phases.
Calibration: the weakest sensor — recons sit at 0.8× the GT floor (models don't measurably distort
the marginal PDF on this task), only noise-monotone. Kept as a guard (weight 0.25), candidate to
drop. Challenge point: replacing/augmenting with W₁ on *velocity increments* or |∇ω| would target
intermittency directly, which the pointwise marginal barely sees.

---

## 5. PDE residual — `d_pde`, `d_pde_lr` (rewards.py:157; residual: physics_guidance.py:38)

De-normalize `w = σx + μ`, evaluate the Kolmogorov-flow vorticity equation residual at the middle
frame (faithful port of BaratiLab's `voriticity_residual`, same operators as the data generator):

```
R(w) = ∂_t ω + (u·∇)ω − (1/Re) Δω + 0.1·ω + 4 cos(4y)

∂_t ω ≈ (w₃ − w₁)/(2Δt),  Δt = 1/32           (central FD across the triplet — only FD term)
u = (∂_y ψ, −∂_x ψ),  Δψ = −ω                  (spectral Poisson solve, streamfunction velocity)
∇, Δ spectral on (0,2π)², integer wavenumbers; NO 2/3 dealiasing (matches reference impl.)
```

The last two terms are the kf_2d source terms: linear drag `+0.1ω` and forcing `f = −4cos(4y)`
(k_f = 4) moved to the left side. **Assumption to challenge:** the *target* regime shares this
forcing and drag; Re is the only parameter varied (`make_ns_residual(re=target)`).

```
d_pde(x)    = mean_x R(w)²                                    (raw, physical units)
d_pde_lr(x) = [ ln d_pde(x) − ln r_ref ]²                     (log-ratio to the GT floor)
```

**The GT floor r_ref is not zero** — GT data has `d_pde` ≈ 1.56 / 17.46 / 45.99 at Re = 500 /
1000 / 2000 (measured, `regime_stats` cell), from O(Δt²) temporal truncation plus
spectral-truncation/aliasing of the unresolved scales. Two consequences, both found in
calibration:

1. Raw `d_pde` mildly *rewards blurring*: mild smoothing filters the GT's own truncation noise
   (residual drops 25 → 17 at blur σ≈1) — a reward-hacking direction.
2. Therefore DDPO uses `d_pde_lr`: the optimum is "as NS-consistent as the data itself", and
   departing the floor in *either* direction (dirtier, or suspiciously cleaner than physics
   allows) is penalized.

This is the only component that sees **dynamics/phases**: under spectrum-matched noise (all phases
randomized, spectrum bit-exact) d_pde_lr = 146× its floor while d_spec/d_energy/d_w1 all sit at
1.0×. Known wart: r_ref varies by sequence at Re=2000 (~32–46), making the anchor noisy there —
flagged in the tracker to fix before Re=2000 DDPO.

---

## 6. Combined reward and DDPO usage (`make_ddpo_reward`, rewards.py:181)

```
r(x) = − Σ_i  w_i · d_i(x) / s_i
```

- **Scales s_i**: std of d_i over the *step-0 recon population* at that regime (what the frozen
  model produces before any finetuning), from `base_results/reward_calibration.json`. Purely a
  units choice so the weights are comparable knobs; any affine slack is absorbed by advantage
  normalization.
- **Weights** (starting point, to revise after first runs):
  `spec 0.5, spec_highk 1.0, energy 0.25, w1 0.25, pde(=pde_lr mode) 1.0`
  — spec_highk carries the objective, spec anchors the rest of the cascade, pde_lr polices
  physics, energy/w1 guard gross drift.
- **Advantages must be per-input.** For K samples {x^(1..K)} of the same conditioning input:
  `A^(j) = (r^(j) − mean_j r)/std_j r`. This is not optional: all anchors are regime-level, so
  each input carries a systematic offset (its own GT's deviation from the regime mean — anchor
  variance). That offset is constant across samples of one input and cancels in A; pooled
  normalization would leave it as dominant noise (measured: pooled per-frame rho vs hi-k retention
  ≈ 0.1, within-input ≈ −0.7).

---

## 7. Summary of deliberate choices (attack surface)

| choice | alternative not taken | why |
|:--|:--|:--|
| enstrophy spectrum | KE spectrum (×1/2k²) | weight the deficit band; loss-aligned |
| geometric-mean spectral anchor | arithmetic mean | typical frame ≈ floor; kills the blur exploit |
| squared log-ratio per shell | linear ratio / L2 on spectra | equal weight per cascade decade |
| band k ∈ [1,96) | full [0,128) | k=0 trivial; k≥96 GT-energy noise |
| pde as log-ratio to GT floor | raw residual → 0 | floor ≠ 0; raw rewards smoothing |
| W1 on pointwise ω marginal | increments / |∇ω| PDFs | simplicity; intermittency not covered |
| σ = 4.7988 for all regimes | per-regime normalization | model convention; anchors absorb it |
| middle frame for statistics | all 3 frames | matches spectrum_fn / metric conventions |
| per-input advantage baseline | pooled batch normalization | regime anchors ⇒ per-input offsets |
