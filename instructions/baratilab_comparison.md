# Comparing our JAX DDPM against BaratiLab's baseline + physics-guided model

Goal: run the **BaratiLab** PyTorch repo
(`https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution`, branch `main_v1`)
on the **same test sequences** we use, export its reconstructions, and score both models with
the **same metric in the same (physical) units**. Our model is a JAX/Flax reimplementation of
the *baseline* (unconditional) DDPM in that repo; the comparison is genuinely apples-to-apples
because both use the identical dataset, split, and U-Net architecture.

This file is meant to be handed to a Claude instance running on a **GPU machine** (e.g. UCL),
since the repo is PyTorch/CUDA and cannot run on our TPU VM.

---

## 0. Why this can't run on the TPU VM
- BaratiLab is **PyTorch + CUDA**, no JAX/TPU support. Our VM has **no GPU** (`nvidia-smi` fails;
  `~/venv-ddpm` has jax/flax, no torch). CPU fallback works but is impractically slow.
- Run BaratiLab on a **CUDA GPU** (Colab T4/A100 or a GCP `g2`/`a2` VM). Our TPU stays the home
  of the JAX model only.

---

## 1. Datasets (identical to ours)
Both repos use the **same two files** (Kolmogorov flow, Re=1000, 256×256 vorticity):

| Role | File | Figshare download |
|---|---|---|
| Clean HR ground truth | `kf_2d_re1000_256_40seed.npy`  shape `(40, 320, 256, 256)` | https://figshare.com/ndownloader/files/39181919 |
| Sparse/degraded input | `kmflow_sampled_data_irregnew.npz` (keys `u3232`, `idx_lst (40,1024)`) | https://figshare.com/ndownloader/files/39214622 |

- Test set in BaratiLab = `[-4:]` → **last 4 sequences = indices 36, 37, 38, 39** — exactly our test set.
- Triplets: 3 consecutive frames stacked as channels → 318 frames/seq. **BaratiLab is channel-FIRST
  `(3,256,256)`; ours is channel-LAST `(256,256,3)`** → transpose when comparing.

### How the low-res / sparse input is built (answering "where is this in our code")
- The `irregnew` input is **irregular random sparse sampling at 1/64 density** (1024 of 65536 pixels),
  with a **fixed mask per sequence** stored in `idx_lst (40,1024)`, followed by a **nearest-neighbour fill**.
- Our `src/utils.py` already implements this exact family of operations:
  - `sparsify_input(im, cfg)` — pick `res/sparsity_ratio` random pixels (per-seed), zero the rest, then `nn_fill`.
  - `nn_fill(sparse_array, mask)` — nearest-neighbour fill via `scipy.ndimage.distance_transform_edt`.
  - `lr_input(im, cfg)` — the *other* task: regular-grid downsample + bicubic/bilinear upsample (super-resolution).
  - Difference vs the npz: our config uses `sparsity_ratio: 16` (1/16) with per-seed random masks; the npz
    is 1/64 with one fixed mask per sequence (`idx_lst`). Same algorithm, different parameters.

### ⚠️ `u3232` vs `u3232_nearest` (the #1 correctness risk)
- BaratiLab's sparse-reconstruction config reads npz key **`u3232_nearest`** (the nearest-FILLED field).
- Our npz only has **`u3232`** (raw sparse), and our `sequence_inference.py` currently feeds `u3232` directly.
- **Both models must receive the SAME input.** Two options:
  1. Build `u3232_nearest` ourselves from `u3232` + `idx_lst` via nearest fill (we have `nn_fill`), and feed
     that to BOTH models; **or**
  2. Confirm whether the figshare npz already contains a `u3232_nearest` key (re-download and check `.files`).
- Decide this before trusting any numbers. (TODO: we will pre-build `u3232_nearest` on the TPU side and ship it.)

---

## 2. Normalization (measured on our data)
Each model must use **its own training normalization** to run inference; the **comparison metric** is then
computed in **common physical units** (de-normalize both before scoring).

Measured stats of `kf_2d_re1000_256_40seed.npy` (mean≈0 everywhere):

| Split | std |
|---|---|
| first 32 seqs (our train) | **4.7988** |
| first 36 seqs (train+val = BaratiLab `[:-4]`) | 4.7870 |
| all 40 | 4.7846 |

Key points:
- Our 32-seq std (4.799) and their 36-seq std (4.787) differ by only **0.25%** → negligible; the
  train/val-split difference does **not** meaningfully affect normalization. So "we used train only, they
  used train+val" is not a problem for a fair comparison.
- **Our hardcoded inference value `std=5.0157` is wrong** for the full 32-seq training set — it came from an
  old subsetted config (`seeds_subset=8, timesteps_subset=50`). TODO on the TPU side: confirm what the
  300-epoch checkpoint actually trained with and, if it was full data, switch inference to `std≈4.799`.
- BaratiLab recomputes mean/std from its reference data at runtime (`np.mean/np.std(ref_data[:-4])`), and also
  ships `pretrained_weights/km256_stats.npz` — let it use its own; we use ours.

---

## 3. BaratiLab setup (on the GPU machine)
```bash
git clone https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution
cd Diffusion-based-Fluid-Super-resolution        # branch main_v1
# env: Python 3.8, PyTorch (CUDA). On modern GPUs relax the pinned torch 1.7/CUDA 10.1.
pip install torch torchvision numpy tqdm einops matplotlib tensorboard

mkdir -p data pretrained_weights
# data
curl -L -o data/kf_2d_re1000_256_40seed.npy   https://figshare.com/ndownloader/files/39181919
curl -L -o data/kmflow_sampled_data_irregnew.npz https://figshare.com/ndownloader/files/39214622
# checkpoints
curl -L -o pretrained_weights/baseline_ckpt.pth  https://figshare.com/ndownloader/files/40320733   # unconditional baseline
curl -L -o pretrained_weights/physics_ckpt.pth   https://figshare.com/ndownloader/files/39184073   # physics-informed (conditional)
```
(If we ship a subset of seqs 36–39 + a prebuilt `u3232_nearest`, place those in `data/` instead and point
the config's `data_dir` / `sample_data_dir` / `data_kw` at them.)

### Configs and what the flags mean
- `configs/kmflow_re1000_rs256.yml` — 8× super-resolution task. Ships with `sampling.lambda_ = 0.0`
  → **physics guidance OFF (pure baseline)**.
- `configs/kmflow_re1000_rs256_sparse_recons.yml` — the **irregular sparse** task that matches our `irregnew`
  input (`data_kw: u3232_nearest`). Use THIS for our comparison.
- CLI: `python main.py --config <cfg> --seed 1234 --sample_step 1 --t 240 --r 30`
  - `--t 240` = forward-noise start level (of 1000).
  - `--r 30` = reverse/denoising steps per iteration.
  - `--sample_step 1` = number of refinement iterations.
  - Physics guidance is toggled by `sampling.lambda_` in the YAML (no CLI flag): `lambda_ = 0` → off;
    `lambda_ > 0` → on (adds the FFT vorticity-residual gradient during sampling).

### Runs to produce (for the sparse-reconstruction config)
1. **Baseline**: `baseline_ckpt.pth`, `lambda_ = 0`.
2. **Physics-guided**: either `baseline_ckpt.pth` with `lambda_ > 0` (sampling-time guidance), and/or the
   `physics_ckpt.pth` conditional model. Capture whichever matches the paper's headline; ideally both.

Keep `--seed 1234 --t 240 --r 30 --sample_step 1` consistent across runs.

---

## 4. Outputs to export
The runner saves per sample folder (`runners/rs256_guided_diffusion.py`):
- `reference_arr.npy` — their ground truth (should be numerically identical to our `kf_2d[36:40]` triplets).
- `input_arr.npy` — the (sparse/filled) input.
- `sample_arr_run_{r}_it{it}.npy` — the reconstruction (export the final iteration).
- the logged `l2_loss_all` per-frame errors.

Export, per model variant: `sample_arr` (final), `reference_arr`, and per-frame `l2_loss`. Note channel-first
`(N,3,256,256)` and check whether saved arrays are normalized or already inverse-scaled (the `l2_loss` is
computed on `scaler.inverse(x)`).

---

## 5. Comparison (back on the TPU VM, pure numpy — no torch)
Our results live in `monitoring/sequence_reconstructions/sequence_reconstruction_seq{36..39}.pkl`:
`{metadata{...mean_mse...}, frames:[{frame_idx, input, ground_truth, final, mse}]}` — all channel-last,
normalized units, 318 frames/seq.

Our metric (`src/sequence_inference.py`): `((gt - pred)**2).mean()` over the full 256×256×3 triplet, normalized.
BaratiLab metric: `l2_loss(x,y) = ((x-y)**2).mean((-1,-2)).sqrt().mean()` (per-frame RMSE, averaged), physical units.

Steps:
1. Load both models' `(pred, gt)` for seqs 36–39. Transpose BaratiLab to channel-last.
2. Put both in the **same units** — pick physical (de-normalize ours: `x*std + mean`) OR normalized
   (normalize theirs). Apply the same choice to both.
3. **Sanity check:** assert BaratiLab `reference_arr` == our `ground_truth` (de-normalized) for seqs 36–39,
   frame-for-frame — confirms alignment before trusting anything.
4. Compute BOTH metrics on BOTH models: our MSE and their L2/RMSE (and optionally true relative L2:
   divide by `((gt**2).mean((-1,-2))).sqrt()`).
5. Plots:
   - per-frame metric vs frame index, one line per model, faceted by seq 36–39;
   - mean-metric bar chart per seq + overall (both metrics, all model variants);
   - qualitative panels for a few frames: `input | GT | ours | baratilab_baseline | baratilab_physics | error maps`
     (shared colormap/scale);
   - optional: energy spectrum / vorticity comparison.

---

## 6. Generating data at OTHER Reynolds numbers

**The BaratiLab repo does NOT contain a data-generation solver** — it only downloads pre-computed
data and ships a PDE *residual* + an FNO *neural network*. Do not confuse these:

- **FNO `residual`** (`runners/rs256_guided_diffusion.py` → `voriticity_residual`): evaluates how far an
  existing field is from satisfying the PDE (time derivative = finite difference over given frames).
  It is a *checker* for physics-guidance, **not** a forward integrator. Re=1000 and forcing `-4cos(4y)`
  are hardcoded.
- **FNO `network`** (`models/diffusion_new.py` → `SpectralConv2d_fast`, `FNO2d`): a *learned* Fourier
  Neural Operator used as a denoiser/surrogate inside the diffusion model. It is a trained network, not a
  numerical solver — it cannot produce validated ground truth, and it was trained at Re=1000.
- ⚠️ "Li et al." is cited for **two different things**: the FNO *architecture* (in `diffusion_new.py`) and
  the pseudo-spectral *solver* that GENERATED the data (NOT in the repo). The solver is the one you need.

The real solver is Zongyi Li's FNO data generator
(`data_generation/navier_stokes/ns_2d.py` + `random_fields.py`, mirror:
https://github.com/scaomath/fourier_neural_operator/tree/master/data_generation/navier_stokes ),
which uses the removed pre-1.8 `torch.rfft` API and a different forcing.

### Confirmed generation parameters (from Shu et al. 2211.14680, Li et al. FNO/PINO, Kochkov 2102.01010)
| Quantity | Value | Source |
|---|---|---|
| Equation | 2D vorticity NS, periodic `(0,2π)²` | Shu et al. (verbatim) |
| Re / viscosity | 1000 / ν=1e-3 | Shu et al. |
| Forcing | `f = −4cos(4x₂) − 0.1ω` (Kolmogorov + drag) | Shu et al.; Kochkov `curl(sin(4y)x̂)=−4cos(4y)` |
| GRF IC | `N(0, 7^(3/2)(−Δ+49I)^(−5/2))` (τ=7, α=2.5) | Shu et al. |
| **DNS grid** | **2048², downsampled 8× to 256²** | Shu et al. |
| Sequences / frames | 40 seqs × 320 frames, Δt=1/32 s, T=10 s | Shu et al. |
| Time scheme / dt | Crank–Nicolson (linear implicit) + explicit nonlinear; dt≈1e-4 | FNO code defaults |
| Dealiasing | 2/3 rule | FNO code |

### ⚠️ Critical correction — the data is recorded in STEADY STATE (with spin-up), not from t=0
The papers don't state a spin-up, and `320×1/32=10s` led to an initial guess of "record from t=0".
**That is wrong.** Measured directly from `kf_2d`: frame-0 already has the **k=4 forcing enstrophy peak**
and **std 4.55** (≈ the equilibrated mid-trajectory std 4.64) — nothing like a smooth GRF (std~1.5).
So a long **spin-up to statistical steady state is required before recording** (drag timescale 1/0.1=10 →
use ~40, matching Kochkov's burn-in=40). The GRF IC is only a seed; its amplitude is forgotten after spin-up.

**Our solver:** `data_generation/generate_kmflow.py` — pseudo-spectral (FFT + Crank–Nicolson) on `(0,2π)²`,
Kolmogorov forcing + drag, GRF ICs, tunable `--re`, with `--spinup` (default 40) and `--downsample-to`
(spectral truncation, verified to preserve large-scale std). Output `(n_samples, record_steps, res, res)`.

Exact reproduction (heavy — 2048² DNS, large GPU):
```bash
python data_generation/generate_kmflow.py --re 1000 --n-samples 40 --res 2048 --downsample-to 256 \
    --record-steps 320 --record-dt 0.03125 --dt 1e-4 --spinup 40.0 --seed 0 \
    --out kf_2d_re1000_256_40seed_REGEN.npy
```
Cheaper approximate (native 256², under-resolved DNS but right statistics):
```bash
python data_generation/generate_kmflow.py --re 1000 --n-samples 40 --res 256 \
    --record-steps 320 --record-dt 0.03125 --dt 1e-3 --spinup 40.0 --out kf_re1000_256.npy
```
Other Reynolds numbers — change `--re` (reduce `--dt` if unstable at higher Re):
```bash
python data_generation/generate_kmflow.py --re 4000 --res 2048 --downsample-to 256 --dt 5e-5 --out kf_re4000.npy
```

### TPU-native alternative — `google/jax-cfd` (Kochkov)
Pure JAX (runs on the TPU, no PyTorch/GPU): `jax_cfd.spectral.equations.ForcedNavierStokes2D`
(k=4, drag=0.1) + `crank_nicolson_rk4`, CFL=0.5. Authoritative & validated, but archived (may need a
jax-version-compatible env) and its default regime uses burn-in=40 — configure spin-up + record to match.
**Validation:** `data_generation/validate_kmflow.py` (pure numpy/matplotlib — runs on the TPU VM, no GPU/torch)
statistically compares a generated `.npy` against reference full-res data. It reports the std ratio and an
energy-spectrum agreement metric, and plots vorticity PDF, energy & enstrophy spectra, and per-frame std
(stationarity). Generated trajectories use random ICs so they never match frame-by-frame — only statistics do.
```bash
python data_generation/validate_kmflow.py \
    --generated kf_2d_re1000_256_40seed_REGEN.npy \
    --reference flow-data/kf_2d_re1000_256_40seed.npy \
    --out monitoring/validation_re1000.png
```
Targets for a good Re=1000 regen: **std ratio ≈ 1.0**, **energy-spectrum |log10 ratio| < ~0.2**, an
**enstrophy peak at k=4** (the forcing wavenumber), and an overlapping vorticity PDF.

**Caveats:** Shu et al. never published their exact generation script, so `dt`, `T`, spinup, and the precise
drag integration here are inferred from the FNO solver + their residual definition — the Re=1000 validation
above is how you confirm those choices. Generation requires a GPU (slow on CPU; will not run on the TPU VM).

---

## 7. Open TODOs to finalize before the GPU run
- [ ] Build/ship `u3232_nearest` (nearest-fill of `u3232` via `idx_lst`) so inputs match — or verify the
      figshare npz already has the key.
- [ ] Confirm the 300-epoch checkpoint's training normalization; fix our inference `std` (4.799 vs 5.0157) if needed.
- [ ] Decide which physics-guided variant to compare (sampling-time `lambda_>0` vs conditional `physics_ckpt.pth`).
- [ ] Ship a subset of seqs 36–39 from both data files to keep the GPU-side download small.
