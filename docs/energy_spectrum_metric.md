# Energy-spectrum metric — reference

A precise account of the spectral diagnostics used in this project (`spectrum_fn` in
`lambda_sweep_re1000.py`, the `energy_spectrum` helpers in the analysis notebooks). It is the
right way to measure whether a super-resolution reconstruction has the *correct energy at each
scale* — the failure mode (over-smoothing) that L2 is blind to.

> **Note on BaratiLab.** Their `main_v1` repo does **not** compute an energy spectrum anywhere —
> it logs only an L2 (RMSE) loss and the PDE residual. The spectrum figures in Shu et al. were made
> offline and not committed. So this note grounds the metric in its *definition*, not their code.

---

## 1. Setup and the discrete Fourier transform

A 2D vorticity field `ω(x, y)` lives on the torus `(0, 2π)²`, sampled on an `N×N` grid (`N=256`).
Its discrete Fourier transform (numpy's unnormalized `fft2`) is

```
ω̂(k) = Σ_x ω(x) · exp(−i k·x),     k = (k_x, k_y),  integer wavenumbers
```

We use integer wavenumbers via `fftfreq(N, d=1/N)` → `k ∈ {0, 1, …, N/2−1, −N/2, …, −1}`. The scalar
wavenumber of a mode is its radius `|k| = √(k_x² + k_y²)`.

---

## 2. Two spectra: vorticity power vs kinetic energy

Both are radial profiles `E(k)` obtained by summing a per-mode spectral density over the shell
`k ≤ |k'| < k+1`. They differ only in the density.

### (a) Vorticity-power (enstrophy) spectrum — what we use
```
E_Ω(k) = Σ_{k ≤ |k'| < k+1} |ω̂(k')|²
```
Total enstrophy `½∫ω² dx = ½ Σ_k |ω̂|²` (up to the DFT normalization), so `|ω̂(k)|²` is the
enstrophy density. This is the **loss-aligned** convention (see §3).

### (b) Kinetic-energy spectrum
In 2D, velocity comes from the streamfunction (`−∇²ψ = ω`, `u = ∂_yψ`, `v = −∂_xψ`), so in Fourier
`û = i k_y ω̂/|k|²`, `v̂ = −i k_x ω̂/|k|²`, giving `|û|² + |v̂|² = |ω̂|² / |k|²`. Hence
```
E_KE(k) = Σ_{shell} |ω̂(k')|² / (2 |k'|²)        # = E_Ω(k) / (2k²) per shell
```
**They are the same data weighted differently by `1/(2k²)`:** KE emphasises the large scales (low
`k`); vorticity power emphasises the small scales (high `k`). For a *smoothing* diagnostic the
vorticity-power spectrum is more sensitive (it puts weight exactly where over-smoothing bites), and
it is the one tied to the training loss. Energy/retention *ratios* are identical under either
convention (the `1/2k²` cancels), so the choice only matters for the shape of the curve, not the
retention numbers.

---

## 3. Why the vorticity-power spectrum is the loss decomposed by scale (Parseval)

For an `N×N` field, Parseval's theorem (numpy convention) is `Σ_x |f(x)|² = (1/N²) Σ_k |f̂(k)|²`.
Take the reconstruction error `e = ω_recon − ω_GT` on the **normalized** field. The MSE the model is
trained/scored on is

```
MSE = (1/N²) Σ_x e(x)²  =  (1/N⁴) Σ_k |ê(k)|²  =  (1/N⁴) Σ_k E_err(k)
```

So **MSE is, up to a constant, the sum over wavenumber of the error's vorticity-power spectrum.**
The spectrum is literally the MSE *decomposed by scale*. This is why:
- a spectral deficit at high `k` is an MSE contribution L2 cannot "see" as anything but a small number
  (high-`k` modes are few and low-amplitude), yet it is exactly the turbulent fine structure;
- penalising the spectrum (or the PDE residual) supplies the high-`k` incentive L2 lacks.

**Important caveat.** Retention (§4) compares the spectra of `recon` and `GT` *separately*, not the
error spectrum. Matching `E_recon(k) = E_GT(k)` is **necessary but not sufficient** for low error —
the phases can still be wrong (right amount of energy, wrong place). But `E_recon(k) < E_GT(k)`
*does* prove missing energy (over-smoothing), and `>` proves spurious energy (noise). So retention is
a clean one-sided diagnostic of spectral fidelity.

---

## 4. The derived metrics

Let `E_recon(k)`, `E_GT(k)`, `E_input(k)` be the (averaged over frames/sequences) spectra.

- **Per-scale retention** `r(k) = E_recon(k) / E_GT(k)`. `=1` ideal, `<1` over-smoothed, `>1` over-energised.
- **High-k energy retention** (the headline OOD number):
  ```
  R(k ≥ k₀) = Σ_{k≥k₀} E_recon(k) / Σ_{k≥k₀} E_GT(k)
  ```
  with `k₀ = 32` here (the fine-scale band well above the `k=4` forcing). Difficulty-free and
  GT-absolute, unlike the input-relative "advantage".
- **Effective resolution `k*`**: the smallest `k` where `E_recon(k) < ½ E_GT(k)` — "the model
  reconstructs structure down to scale `k*`". A compact scalar for "how blurred".

---

## 5. What the code computes

`spectrum_fn` (in `lambda_sweep_re1000.py`) and the notebook `energy_spectrum` helper compute the
**vorticity-power spectrum of the normalized middle-frame** of each triplet:

```python
P   = |fft2(ω_norm)|²                       # (B, N, N) spectral power, ω_norm = (ω−mean)/std
KR  = round(√(k_x² + k_y²))                 # integer shell index per mode
E   = segment_sum(P.ravel(), KR.ravel())    # Σ over each shell  ->  E_Ω(k), length N/2
```

Conventions baked in: normalized field (so the spectrum is in the same units as the L2 loss);
integer `fftfreq` wavenumbers on `(0,2π)²`; `round` shell binning; `k = 0..127`. Ratios at `k ≳ 96`
are unreliable (GT energy → 0 there, so the denominator is tiny).

---

## 6. Physical context (2D Kolmogorov flow)

Forcing at `k = 4` sets the injection scale. 2D turbulence has a **dual cascade**: an inverse energy
cascade toward low `k` (`k < 4`) and a forward **enstrophy cascade** toward high `k` (`k > 4`), the
latter with a steep `E_KE(k) ∼ k⁻³`-ish falloff. The interesting reconstruction question is whether
the model reproduces the enstrophy-cascade tail (`k ≳ 32`) or rolls off too early — i.e. the high-k
retention. The sparse NN-fill *input* has the opposite pathology: blocky nearest-neighbour edges
inject **spurious** high-`k` power (input retention ≫ 1), which a good reconstruction must remove
without overshooting below the GT level.

---

## 7. Practical reading guide

- **Spectrum overlay (log-log):** `GT`, `input`, `recon`. Look at where the recon curve departs from
  GT. Below GT at high `k` = over-smoothing; above = spurious speckle.
- **Per-scale retention `r(k)`:** the same information flattened around `1.0`; easiest for comparing
  methods (e.g. λ values) scale-by-scale.
- **`R(k≥32)` and `k*`:** the two scalars to report for OOD-ability.
- **Pair with the error/MSE**, since retention alone ignores phase. A method that raises `R(k≥32)`
  toward 1 *and* lowers MSE is genuinely recovering structure; one that raises `R` while MSE worsens
  is adding wrong-phase high-k energy.
