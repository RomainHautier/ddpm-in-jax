"""Evaluate a trained DDPM: test loss, reconstruction grid, and generation.

Usage:
  python generate_and_eval.py
  python generate_and_eval.py --ckpt gs://thesis-project-bucket-rh-ucl/checkpoints/ddpm/ckpt_epoch_0299.pkl
  python generate_and_eval.py --no_gcs --skip_generation
"""
import argparse
import functools
import json
import io
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import gcsfs
from flax.serialization import from_bytes
from tqdm import tqdm

from train_ddpm import (
    Unet, calc_alpha, forward_process,
    load_npy_from_gcs, load_and_split, build_samples, save_plot_to_gcs,
    GCS_BUCKET, MONITORING_DIR, CHECKPOINT_DIR, DATA_PATH,
)


def get_fs():
    return gcsfs.GCSFileSystem(project=GCS_BUCKET)


def load_latest_checkpoint_path():
    """Find the latest checkpoint on GCS and return its gs:// path."""
    fs = get_fs()
    files = sorted(fs.ls(CHECKPOINT_DIR.replace("gs://", "")))
    pkl_files = [f for f in files if f.endswith('.pkl')]
    if not pkl_files:
        raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}")
    path = f"gs://{pkl_files[-1]}"
    print(f"Latest checkpoint: {path}")
    return path


def load_params(ckpt_path, model):
    """Load params from a GCS checkpoint."""
    fs = get_fs()
    with fs.open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)

    key = jax.random.PRNGKey(0)
    dummy_x = jnp.ones((1, 256, 256, 3))
    dummy_t = jnp.ones((1,), dtype=jnp.int32)
    variables = model.init({"params": key, "dropout": key}, dummy_x, dummy_t)
    params = from_bytes(variables['params'], ckpt['params'])
    epoch = ckpt['epoch']
    train_losses = ckpt.get('train_losses', [])
    val_losses = ckpt.get('val_losses', [])
    print(f"Loaded checkpoint from epoch {epoch}")
    return params, epoch, train_losses, val_losses


# ── JIT'd single denoise step ────────────────────────────────────────────

def make_denoise_step(model):
    """Create a JIT-compiled single denoising step.

    Compiles once, then every subsequent call is fast.
    """
    @jax.jit
    def _denoise_step(params, x_t, t_batch, alpha, alpha_bar, beta, key):
        eps_pred = model.apply({'params': params}, x_t, t_batch, train=False)

        # All samples in the batch share the same t, so index with t_batch[0]
        t_val = t_batch[0]
        alpha_t = alpha[t_val]
        alpha_bar_t = alpha_bar[t_val]
        beta_t = beta[t_val]

        coef1 = 1.0 / jnp.sqrt(alpha_t)
        coef2 = beta_t / jnp.sqrt(1.0 - alpha_bar_t)
        mean = coef1 * (x_t - coef2 * eps_pred)

        z = jax.random.normal(key, x_t.shape)
        sigma = jnp.sqrt(beta_t)
        # If t > 0 add noise, else just return mean
        x_next = jnp.where(t_val > 0, mean + sigma * z, mean)
        return x_next

    return _denoise_step


# ── Reverse diffusion (sampling) ──────────────────────────────────────────

def ddpm_sample(denoise_step, params, shape, T, beta_schedule, key):
    """Algorithm 2: Sampling. x_T ~ N(0,I), denoise to x_0."""
    alpha = 1 - beta_schedule
    alpha_bar = jnp.cumprod(alpha)

    key, subkey = jax.random.split(key)
    x_t = jax.random.normal(subkey, shape)

    for t_val in tqdm(range(T - 1, -1, -1), desc="Sampling"):
        t_batch = jnp.full((shape[0],), t_val, dtype=jnp.int32)
        key, subkey = jax.random.split(key)
        x_t = denoise_step(params, x_t, t_batch, alpha, alpha_bar,
                           beta_schedule, subkey)

    return x_t


def reconstruct(denoise_step, params, x_0, t_noise, beta_schedule, key,
                show_progress=True):
    """Noise x_0 to timestep t_noise, then denoise back to x_0."""
    alpha = 1 - beta_schedule
    alpha_bar = jnp.cumprod(alpha)

    key, subkey = jax.random.split(key)
    eps = jax.random.normal(subkey, x_0.shape)
    t = jnp.full((x_0.shape[0],), t_noise, dtype=jnp.int32)
    x_t = forward_process(x_0, t, alpha_bar, eps)

    iterator = range(t_noise, -1, -1)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Denoise t={t_noise}", leave=False)

    for t_val in iterator:
        t_batch = jnp.full((x_0.shape[0],), t_val, dtype=jnp.int32)
        key, subkey = jax.random.split(key)
        x_t = denoise_step(params, x_t, t_batch, alpha, alpha_bar,
                           beta_schedule, subkey)

    return x_t


# ── Test loss ─────────────────────────────────────────────────────────────

def compute_test_loss(model, params, test_samples, T, beta_schedule, key,
                      batch_size=8):
    """Compute noise-prediction MSE over the full test set (random t per sample)."""
    alpha_bar = jnp.cumprod(1 - beta_schedule)

    @jax.jit
    def _eval_batch(params, batch, t, eps):
        noised = forward_process(batch, t, alpha_bar, eps)
        eps_pred = model.apply({'params': params}, noised, t, train=False)
        return jnp.mean((eps - eps_pred) ** 2)

    losses = []
    n = len(test_samples)

    for i in tqdm(range(0, n, batch_size), desc="Test loss"):
        batch = jnp.array(test_samples[i:i + batch_size])
        bs = batch.shape[0]
        key, k_t, k_eps = jax.random.split(key, 3)
        t = jax.random.randint(k_t, (bs,), 0, T)
        eps = jax.random.normal(k_eps, batch.shape)
        loss = float(_eval_batch(params, batch, t, eps))
        losses.append(loss)

    return float(np.mean(losses)), float(np.std(losses))


# ── Visualization ─────────────────────────────────────────────────────────

def plot_reconstruction_grid(denoise_step, params, samples, sample_indices,
                             t_values, beta_schedule, mean, std, key,
                             save_gcs=True):
    """Grid: rows = test samples, cols = [Original, t1, t2, ...].

    Each cell shows the reconstructed image; the original column is first.
    MSE is annotated on each reconstructed cell.
    """
    n_samples = len(sample_indices)
    n_t = len(t_values)
    ch = 0  # channel to display

    fig, axes = plt.subplots(n_samples, n_t + 1,
                             figsize=(3.5 * (n_t + 1), 3.5 * n_samples))
    if n_samples == 1:
        axes = axes[None, :]

    per_sample_metrics = []

    for row, idx in enumerate(sample_indices):
        x_0 = jnp.array(samples[idx:idx + 1])
        orig_img = np.array(x_0[0]) * std + mean
        vmin = orig_img[:, :, ch].min()
        vmax = orig_img[:, :, ch].max()

        # Original
        axes[row, 0].imshow(orig_img[:, :, ch], cmap='RdBu_r',
                            vmin=vmin, vmax=vmax)
        if row == 0:
            axes[row, 0].set_title('Original', fontsize=11, fontweight='bold')
        axes[row, 0].set_ylabel(f'Sample {idx}', fontsize=10)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        sample_row_metrics = {}
        for col, t_noise in enumerate(t_values):
            print(f"  Sample {row+1}/{n_samples}, t={t_noise}")
            key, k_recon = jax.random.split(key)
            recon = reconstruct(denoise_step, params, x_0, t_noise,
                                beta_schedule, k_recon)
            recon_img = np.array(recon[0]) * std + mean
            mse = float(jnp.mean((x_0[0] - recon[0]) ** 2))
            mae = float(jnp.mean(jnp.abs(x_0[0] - recon[0])))
            sample_row_metrics[int(t_noise)] = {"mse": mse, "mae": mae}

            axes[row, col + 1].imshow(recon_img[:, :, ch], cmap='RdBu_r',
                                      vmin=vmin, vmax=vmax)
            if row == 0:
                axes[row, col + 1].set_title(f't={t_noise}', fontsize=11,
                                             fontweight='bold')
            axes[row, col + 1].set_xlabel(f'MSE={mse:.4f}', fontsize=8)
            axes[row, col + 1].set_xticks([])
            axes[row, col + 1].set_yticks([])

        per_sample_metrics.append({"sample_idx": int(idx),
                                   "metrics": sample_row_metrics})

    fig.suptitle('Reconstruction from different noise levels',
                 fontsize=14, y=1.01)
    plt.tight_layout()

    if save_gcs:
        save_plot_to_gcs(fig, 'reconstruction_grid.png')
    else:
        fig.savefig('reconstruction_grid.png', dpi=150, bbox_inches='tight')
        print("Saved reconstruction_grid.png locally")
    plt.close(fig)

    return per_sample_metrics


def plot_generated_samples(samples, mean, std, save_gcs=True):
    """Plot a grid of generated samples."""
    n = min(len(samples), 16)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for i in range(n):
        img = np.array(samples[i]) * std + mean
        axes[i].imshow(img[:, :, 0], cmap='RdBu_r')
        axes[i].set_title(f'Sample {i}')
        axes[i].axis('off')

    for i in range(n, len(axes)):
        axes[i].axis('off')

    fig.suptitle(f'Generated Samples ({n})', fontsize=14)
    plt.tight_layout()

    if save_gcs:
        save_plot_to_gcs(fig, 'generated_samples.png')
    else:
        fig.savefig('generated_samples.png', dpi=150, bbox_inches='tight')
        print("Saved generated_samples.png locally")
    plt.close(fig)


def plot_metrics_summary(recon_metrics, save_gcs=True):
    """Bar chart of MSE and MAE vs noise level t."""
    t_vals = sorted(recon_metrics.keys())
    mses = [recon_metrics[t]["mse_mean"] for t in t_vals]
    mse_stds = [recon_metrics[t]["mse_std"] for t in t_vals]
    maes = [recon_metrics[t]["mae_mean"] for t in t_vals]
    mae_stds = [recon_metrics[t]["mae_std"] for t in t_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(t_vals))
    ax1.bar(x, mses, yerr=mse_stds, capsize=4, color='steelblue', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(t) for t in t_vals])
    ax1.set_xlabel('Noise level t')
    ax1.set_ylabel('MSE')
    ax1.set_title('Reconstruction MSE vs noise level')

    ax2.bar(x, maes, yerr=mae_stds, capsize=4, color='coral', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(t) for t in t_vals])
    ax2.set_xlabel('Noise level t')
    ax2.set_ylabel('MAE')
    ax2.set_title('Reconstruction MAE vs noise level')

    plt.tight_layout()

    if save_gcs:
        save_plot_to_gcs(fig, 'reconstruction_metrics.png')
    else:
        fig.savefig('reconstruction_metrics.png', dpi=150, bbox_inches='tight')
        print("Saved reconstruction_metrics.png locally")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DDPM")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="GCS path to checkpoint (default: latest)")
    parser.add_argument("--n_gen", type=int, default=4,
                        help="Number of samples to generate from noise")
    parser.add_argument("--T", type=int, default=1000,
                        help="Number of diffusion timesteps")
    parser.add_argument("--t_noise", type=int, nargs='+',
                        default=[50, 100, 200, 400, 600, 800],
                        help="Timesteps for reconstruction evaluation")
    parser.add_argument("--n_grid_samples", type=int, default=5,
                        help="Number of test samples in the reconstruction grid")
    parser.add_argument("--no_gcs", action="store_true",
                        help="Save plots locally instead of GCS")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip slow generation from pure noise")
    parser.add_argument("--output_json", type=str, default="eval_results.json",
                        help="Path for the JSON metrics report")
    args = parser.parse_args()

    # Setup
    T = args.T
    beta_schedule = jnp.linspace(0.0001, 0.02, T)
    model = Unet(ch=64, ch_mult=(1, 1, 1, 2), dropout_p=0.0)

    # ── Load checkpoint ──
    if args.ckpt is None:
        args.ckpt = load_latest_checkpoint_path()

    params, epoch, train_losses, val_losses = load_params(args.ckpt, model)

    # ── Create JIT'd denoise step (compiled once, reused everywhere) ──
    denoise_step = make_denoise_step(model)
    print("Warming up JIT (first denoise call)...")
    _warmup_x = jnp.ones((1, 256, 256, 3))
    _warmup_t = jnp.ones((1,), dtype=jnp.int32)
    _warmup_key = jax.random.PRNGKey(0)
    _alpha = 1 - beta_schedule
    _alpha_bar = jnp.cumprod(_alpha)
    _ = denoise_step(params, _warmup_x, _warmup_t, _alpha, _alpha_bar,
                     beta_schedule, _warmup_key)
    print("JIT warmup done.")

    # ── Load data with SAME split as training (seed=1, 70/10/20) ──
    num_devices = len(jax.devices())
    train_samples, val_samples, test_samples, mean, std, _ = load_and_split(
        batch_size=8, num_devices=num_devices)
    print(f"Data normalization: mean={mean:.4f}, std={std:.4f}")
    print(f"Test samples: {test_samples.shape}")

    key = jax.random.PRNGKey(42)

    # Collect all metrics into a dict
    report = {
        "checkpoint": args.ckpt,
        "epoch": int(epoch),
        "final_train_loss": float(train_losses[-1]) if train_losses else None,
        "final_val_loss": float(val_losses[-1]) if val_losses else None,
        "T": T,
        "t_noise_levels": args.t_noise,
        "n_test_samples": len(test_samples),
        "mean": float(mean),
        "std": float(std),
    }

    # ── 1. Test loss (noise prediction MSE, random t over full test set) ──
    print("\n=== Test Loss (full test set) ===")
    key, subkey = jax.random.split(key)
    test_mean, test_std = compute_test_loss(
        model, params, test_samples, T, beta_schedule, subkey)
    print(f"Test noise-prediction MSE: {test_mean:.6f} +/- {test_std:.6f}")
    report["test_noise_mse_mean"] = test_mean
    report["test_noise_mse_std"] = test_std

    # ── 2. Reconstruction grid + metrics (5 random test samples) ──
    print("\n=== Reconstruction Grid ===")
    key, subkey = jax.random.split(key)
    n_grid = min(args.n_grid_samples, len(test_samples))
    grid_indices = np.array(
        jax.random.choice(subkey, len(test_samples), (n_grid,), replace=False))
    print(f"Grid sample indices: {grid_indices}")

    key, subkey = jax.random.split(key)
    grid_metrics = plot_reconstruction_grid(
        denoise_step, params, test_samples, grid_indices, args.t_noise,
        beta_schedule, mean, std, subkey, save_gcs=not args.no_gcs)
    report["grid_sample_metrics"] = grid_metrics

    # Aggregate reconstruction metrics from the grid samples
    recon_metrics = {}
    for t_noise in args.t_noise:
        mses = [s["metrics"][t_noise]["mse"] for s in grid_metrics]
        maes = [s["metrics"][t_noise]["mae"] for s in grid_metrics]
        recon_metrics[int(t_noise)] = {
            "mse_mean": float(np.mean(mses)),
            "mse_std": float(np.std(mses)),
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes)),
            "n_samples": len(mses),
        }
    report["reconstruction_metrics"] = recon_metrics

    plot_metrics_summary(recon_metrics, save_gcs=not args.no_gcs)

    # ── 3. Generate from pure noise ──
    if not args.skip_generation:
        print(f"\n=== Generation ({args.n_gen} samples) ===")
        key, subkey = jax.random.split(key)
        generated = ddpm_sample(
            denoise_step, params, (args.n_gen, 256, 256, 3),
            T, beta_schedule, subkey)
        plot_generated_samples(generated, mean, std, save_gcs=not args.no_gcs)
        report["n_generated"] = args.n_gen
        print("Generated samples plotted.")
    else:
        report["n_generated"] = 0
        print("\nSkipping generation (--skip_generation).")

    # ── Save JSON report ──
    with open(args.output_json, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nMetrics saved to {args.output_json}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Checkpoint:         epoch {epoch}")
    print(f"Test noise MSE:     {test_mean:.6f} +/- {test_std:.6f}")
    if train_losses:
        print(f"Final train loss:   {train_losses[-1]:.6f}")
    if val_losses:
        print(f"Final val loss:     {val_losses[-1]:.6f}")
    print(f"\nReconstruction ({n_grid} samples):")
    print(f"  {'t':>6s}  {'MSE':>12s}  {'MAE':>12s}")
    print(f"  {'---':>6s}  {'---':>12s}  {'---':>12s}")
    for t_noise in sorted(recon_metrics.keys()):
        m = recon_metrics[t_noise]
        print(f"  {t_noise:6d}  {m['mse_mean']:8.6f} +/- {m['mse_std']:.4f}  "
              f"{m['mae_mean']:8.6f} +/- {m['mae_std']:.4f}")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
