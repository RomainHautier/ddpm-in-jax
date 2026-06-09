import io
import os

import gcsfs
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
from scipy.ndimage import distance_transform_edt
import pickle

GCS_BUCKET = "thesis-project-bucket-rh-ucl"
MONITORING_DIR = f"gs://{GCS_BUCKET}/monitoring"


def get_fs():
    """Return an authenticated GCSFileSystem."""
    return gcsfs.GCSFileSystem(project=GCS_BUCKET)


def load_npy_from_gcs(gcs_path: str) -> np.ndarray:
    """
    Load a .npy file directly from GCS into memory.
    gcs_path: full gs:// path, e.g. 'gs://bucket/folder/file.npy'
    """
    fs = get_fs()
    with fs.open(gcs_path, "rb") as f:
        data = np.load(f)
    print(f"Loaded {gcs_path} — shape: {data.shape}")
    return data


def save_plot_to_gcs(fig, filename: str):
    """
    Save a matplotlib figure to GCS monitoring folder.
    filename: e.g. 'loss_epoch_010.png'
    """
    fs = get_fs()
    gcs_path = f"{MONITORING_DIR}/{filename}"
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    with fs.open(gcs_path, "wb") as f:
        f.write(buf.read())
    print(f"Saved plot to {gcs_path}")


def plot_losses(train_losses, val_losses, epoch, save_to_gcs=True):
    """
    Plot train and val losses. Displays inline and optionally saves to GCS.
    Called every epoch from the training loop.
    """
    from IPython.display import clear_output

    clear_output(wait=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="train", color="steelblue")
    ax.plot(val_losses, label="val", color="coral")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"DDPM Training — Epoch {epoch}")
    ax.legend()
    plt.tight_layout()
    plt.show()

    if save_to_gcs and len(train_losses) > 0:
        save_plot_to_gcs(fig, f"loss_epoch_{epoch:04d}.png")

    plt.close(fig)

    # ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(params, opt_state, epoch, cfg):
    ckpt_dir = cfg["checkpointing"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"ckpt_epoch_{epoch:04d}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"params": params, "opt_state": opt_state, "epoch": epoch}, f)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(path):
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    return ckpt["params"], ckpt["opt_state"], ckpt["epoch"]


def save_final_loss_plot(train_losses, val_losses):
    """Save the final loss curve to GCS at end of training."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="train", color="steelblue")
    ax.plot(val_losses, label="val", color="coral")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("DDPM Training — Final")
    ax.legend()
    plt.tight_layout()
    save_plot_to_gcs(fig, "loss_final.png")
    plt.close(fig)
    print("Final loss plot saved to GCS.")


def nn_fill(sparse_array, mask):
    
    filled_array = sparse_array.copy()
    unknown = ~mask
    _, indices = distance_transform_edt(np.array(unknown), return_indices=True)
    filled_array = filled_array.at[unknown].set(sparse_array[indices[0][unknown] ,indices[1][unknown]])
    
    return filled_array


def sparsify_input(im, cfg):
    """ This function takes a high fidelity sample, randomly takes
    out pixels from the image to respect the defined sparsity percentage
    and reconstructs the missing pixels using NN methods.
    """
    
    seeds = cfg["nn_fill"]["seeds"]
    res = im.shape[0]**2
    ds_res = int(res/cfg["nn_fill"]["sparsity_ratio"])
    seeded_samples = []

    for seed in seeds:
        key = jax.random.key(seed)
        key, subkey = jax.random.split(key)

        sparse_indices = jax.random.randint(subkey, shape = (ds_res,), minval = 0, maxval = res-1)
        mask = jnp.zeros(res, dtype=bool).at[sparse_indices].set(True).reshape(im.shape[0], im.shape[1])
        im_flat = im.reshape(-1, im.shape[-1])
        sparse_array = jnp.zeros_like(im_flat).at[sparse_indices].set(im_flat[sparse_indices])
        sparse_sample = nn_fill(sparse_array.reshape(im.shape), mask)
        seeded_samples.append(sparse_sample)
    
    return seeded_samples


