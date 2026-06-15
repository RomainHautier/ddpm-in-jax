# DDPM in JAX — Fluid Flow Reconstruction

A Denoising Diffusion Probabilistic Model (DDPM) trained on 2D Kolmogorov flow simulations (Re=1000, 256×256) with two goals:

1. **Baseline (plain DDPM)** — characterise how well the model reconstructs a flow field when starting from a partially-noised version of the ground truth, as a function of the denoising horizon.
2. **Sparse reconstruction** — given a spatially degraded input (low-resolution bicubic upsampling or sparse NN-filled field), iteratively refine it with the diffusion model to recover a high-fidelity flow field.

The approach is inspired by [*A physics-informed diffusion model for high-fidelity flow field reconstruction*](https://www.sciencedirect.com/science/article/pii/S0021999123000670) (Shu et al., 2023), which uses SDEdit-style conditioning — noising a degraded observation to timestep *t* and then denoising — as a data-driven super-resolution/reconstruction mechanism without task-specific retraining.

---

## Architecture

U-Net with 4 resolution levels (256→128→64→32):

| Component | Detail |
|---|---|
| ResNet blocks | GroupNorm → SiLU → circular Conv → time injection |
| Time conditioning | Sinusoidal embeddings injected into every ResNet block |
| Self-attention | Bottleneck (32×32) only |
| Skip connections | Encoder → decoder at each resolution |
| Channel widths | `ch=64`, multipliers `(1,1,1,2)` → 64/64/64/128 |

Circular convolutions are used throughout to respect the periodic boundary conditions of the Kolmogorov flow.

---

## Data

**Dataset:** `kf_2d_re1000_256_40seed.npy` — 40 independent flow trajectories at Re=1000, each of shape `[T, 256, 256]`, stored on GCS.

**Training samples:** triplets of consecutive frames stacked as `(256, 256, 3)` — the model sees three consecutive vorticity snapshots as channels, learning the temporal structure of the flow.

**Split:** 32 sequences train / 4 validation / 4 test (80/10/10%).

---

## Repo Structure

```
src/
  models/model.py         # U-Net architecture + DDPM class (forward process, sample)
  train_ddpm.py           # Training loop + data pipeline
  inference.py            # Sparse reconstruction inference (Task 1: LR, Task 2: sparse)
  plain_ddpm_inference.py # Baseline reconstruction from partially-noised GT
  utils.py                # GCS I/O, checkpointing, monitoring utilities
configs/
  config.yaml             # Model and training hyperparameters
  inference_config.yaml   # Inference settings (n_samples, S schedule, task)
monitoring/
  base_results/           # Plain DDPM baseline results
  sparse_reconstructions/ # Sparse/LR reconstruction results and notebooks
checkpoints/              # Saved model weights (gitignored)
```

---

## Setup

**TPU / GPU (recommended):**
```bash
conda create -n ddpm_jax python=3.12
conda activate ddpm_jax
pip install "jax[cuda12]==0.7.2" jaxlib==0.7.2 flax==0.11.2 optax==0.2.7 \
    numpy==2.0.2 tensorflow==2.19.0 matplotlib==3.10.0 tqdm gcsfs pyyaml
```

**UCL cluster (CentOS 7, no AVX, Python 3.12):**
```bash
pip install --only-binary=:all: numpy==1.26.4 ml-dtypes==0.3.2
pip install --only-binary=:all: jax==0.4.30 jaxlib==0.4.30
pip install --force-reinstall --no-deps flax==0.8.1
pip install optax tensorflow-cpu==2.17.0 gcsfs pyyaml tqdm matplotlib
```

---

## Training

```bash
python3 -m src.train_ddpm
```

All hyperparameters live in `configs/config.yaml`. Checkpoints are saved every `save_every_n_epochs` epochs. Loss curves are uploaded to GCS automatically.

---

## Inference

### Task 0 — Baseline: partial denoising from ground truth

Measures how well the model can reconstruct a flow field when starting from a partially-noised version of the ground truth, at several noise horizons *S* ∈ {10, 50, 100, 200, …, 800}.

```bash
python3 -m src.plain_ddpm_inference
```

Results are saved to `base_results/plain_ddpm_reconstructions_baseline.pkl` and uploaded to GCS. Inspect with `monitoring/base_results/ddpm_results.ipynb`.

**Key result:** reconstruction MSE as a function of denoising start *t*. At low *t* the model starts close to the GT and barely changes it; at high *t* significant information is lost to noise and recovery degrades.

![MSE vs denoising start t](monitoring/base_results/mse_vs_t.png)

---

### Task 1 — Low-resolution → high-resolution reconstruction

A high-resolution (256×256) flow field is downsampled to a low resolution (LR32 or LR64), bicubic-upsampled back to 256×256, then iteratively refined by the diffusion model:

1. Add noise to the LR input up to timestep *S*
2. Denoise back to *t=0*
3. Feed the output back as the new conditioning input for *K* iterations

```bash
python3 -m src.inference   # task=1 in inference_config.yaml
```

Results: `monitoring/sparse_reconstructions/reconstructions_task1_lr{32,64}.pkl`

**Key finding:** LR32 inputs (more aggressively downsampled) produce lower MSE than LR64 in early iterations. This is consistent with the SDEdit mechanism — a stronger degradation gives the model more freedom to hallucinate high-frequency structure aligned with the training distribution, whereas LR64 retains more of the original signal and constrains the denoising trajectory more tightly.

---

### Task 2 — Sparse → dense reconstruction

A high-resolution field is sparsified (retaining 1/16 or 1/64 of pixels), NN-filled to create a smooth dense initialisation, then refined with the same iterative SDEdit procedure as Task 1.

```bash
python3 -m src.inference   # task=2 in inference_config.yaml
```

Results: `monitoring/sparse_reconstructions/reconstructions_task2_sratio{16,64}.pkl`

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `monitoring/base_results/ddpm_results.ipynb` | Baseline reconstruction grids + MSE vs *t* curve |
| `monitoring/sparse_reconstructions/reconstructions.ipynb` | PCA trajectory plot, side-by-side comparisons, MSE progression across iterations |

---

## References

- Ho et al. (2020) — [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Meng et al. (2021) — [SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073)
- Shu et al. (2023) — [A physics-informed diffusion model for high-fidelity flow field reconstruction](https://www.sciencedirect.com/science/article/pii/S0021999123000670)
