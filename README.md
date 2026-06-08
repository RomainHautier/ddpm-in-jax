# DDPM in JAX — 2D Turbulence Flow Generation

A Denoising Diffusion Probabilistic Model (DDPM) trained on 2D Kolmogorov flow simulations (Re=1000, 256×256) to learn and generate fluid dynamics sequences. Built with JAX/Flax for the model and TensorFlow for the data pipeline, trained on Google Cloud (UCL cluster).

## Architecture

U-Net with 4 resolution levels (256→128→64→32):
- **ResNet blocks** (`DDPMResnet`): GroupNorm + SiLU activations + circular convolutions (periodic boundary conditions)
- **Sinusoidal time embeddings** injected into every ResNet block
- **Self-attention** at the bottleneck (32×32) only
- **Skip connections** from encoder to decoder
- Channel multipliers `(1,1,1,2)` with base `ch=64` → widths 64, 64, 64, 128

## Data

Dataset: `kf_2d_re1000_256_40seed.npy`, shape `[40 seeds, T timesteps, 256, 256]`, stored on GCS.

Training samples are triplets of consecutive frames stacked as `(256, 256, 3)` — the model sees 3 consecutive flow snapshots as channels.

## Repo Structure

```
model.py          # U-Net architecture + DDPM wrapper class
train_ddpm.py     # Training loop
utils.py          # GCS I/O and monitoring utilities
config.yaml       # All hyperparameters
checkpoints/      # Saved model weights (gitignored)
flow-data/        # Dataset (gitignored)
monitoring/       # Loss plots saved to GCS (gitignored)
```

## Setup

```bash
conda create -n ddpm_jax python=3.12
conda activate ddpm_jax
pip install "jax[cuda12]==0.7.2" jaxlib==0.7.2 flax==0.11.2 optax==0.2.7 \
    orbax-checkpoint==0.11.33 numpy==2.0.2 tensorflow==2.19.0 \
    matplotlib==3.10.0 tqdm==4.67.0 gcsfs pyyaml
```

For CPU-only (no GPU):
```bash
pip install jax==0.7.2  # without cuda12 extra
```

## Training

All hyperparameters are in `config.yaml`. Checkpoints are saved every 10 epochs to `checkpoints/ddpm/` and loss plots are uploaded to the GCS monitoring bucket.

## Inference

Inference works on CPU (slow, ~minutes per sample) or GPU (fast). Checkpoints are `.pkl` files loadable with `pickle` or `orbax-checkpoint`.
