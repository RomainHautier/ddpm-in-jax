import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.train_ddpm import load_dataset
from src.utils import sparsify_input

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plot_sparsification(cfg, inference_cfg, n_images=10, channel=1):
    """Plot original / sparse mask / NN-filled result for each seed.

    channel: which of the 3 vorticity frames to display (0=t-1, 1=t, 2=t+1)
    """
    _, _, test_ds, mean, std = load_dataset(cfg, max_test_samples=n_images)

    seeds = inference_cfg["nn_fill"]["seeds"]
    n_seeds = len(seeds)

    for im_idx, batch in enumerate(test_ds):
        im = jnp.array(batch).squeeze(0)           # (H, W, C)
        sparse_inputs = sparsify_input(im, inference_cfg)

        # cols: original + (sparse mask + nn-filled) per seed
        n_cols = 1 + 2 * n_seeds
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
        fig.suptitle(f"Image {im_idx} — channel {channel} (vorticity frame {channel})", fontsize=12)

        vmin = float(im[:, :, channel].min())
        vmax = float(im[:, :, channel].max())

        axes[0].imshow(np.array(im[:, :, channel]), cmap="RdBu", vmin=vmin, vmax=vmax)
        axes[0].set_title("Ground truth")
        axes[0].axis("off")

        for s_idx, (seed, filled) in enumerate(zip(seeds, sparse_inputs)):
            # sparse mask: True where pixels are kept
            res = im.shape[0] ** 2
            ds_res = int(res / inference_cfg["nn_fill"]["sparsity_ratio"])

            import jax
            key = jax.random.key(seed)
            _, subkey = jax.random.split(key)
            sparse_indices = jax.random.randint(subkey, shape=(ds_res,), minval=0, maxval=res - 1)
            mask = jnp.zeros(res, dtype=bool).at[sparse_indices].set(True).reshape(im.shape[0], im.shape[1])

            sparse_vis = np.full(im.shape[:2], np.nan)
            sparse_vis[np.array(mask)] = np.array(im[:, :, channel])[np.array(mask)]

            col = 1 + s_idx * 2

            axes[col].imshow(sparse_vis, cmap="RdBu", vmin=vmin, vmax=vmax)
            axes[col].set_title(f"Sparse (seed {seed})\n{100/inference_cfg["nn_fill"]['sparsity_ratio']:.1f}% kept")
            axes[col].axis("off")

            axes[col + 1].imshow(np.array(filled[:, :, channel]), cmap="RdBu", vmin=vmin, vmax=vmax)
            axes[col + 1].set_title(f"NN-filled (seed {seed})")
            axes[col + 1].axis("off")

        plt.tight_layout()
        save_dir = os.path.join(BASE_DIR, "monitoring", "sparsity_check")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"sparsification_check_im{im_idx:02d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {save_path}")


if __name__ == "__main__":
    with open(os.path.join(BASE_DIR, "configs", "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(BASE_DIR, "configs", "inference_config.yaml")) as f:
        inference_cfg = yaml.safe_load(f)

    plot_sparsification(cfg, inference_cfg, n_images=10, channel=1)
