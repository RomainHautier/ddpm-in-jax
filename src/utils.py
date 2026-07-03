import io
import os

import gcsfs
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
from scipy.ndimage import distance_transform_edt
import pickle

GCS_BUCKET = "ddpm-thesis-rh"
GCS_PROJECT = "csml-thesis"
MONITORING_DIR = f"gs://{GCS_BUCKET}/monitoring"

# ---------------------------------------------------------------------------
# Loading & Saving Data
# ---------------------------------------------------------------------------

def get_fs():
    """Return an authenticated GCSFileSystem."""
    return gcsfs.GCSFileSystem(project=GCS_PROJECT)


def load_npy_from_gcs(gcs_path: str) -> np.ndarray:
    """
    Load a .npy file from GCS, caching it locally under flow-data/.
    gcs_path: full gs:// path, e.g. 'gs://bucket/folder/file.npy'
    """
    import subprocess
    local_cache = os.path.join("flow-data", os.path.basename(gcs_path))
    os.makedirs("flow-data", exist_ok=True)
    if not os.path.exists(local_cache):
        print(f"Downloading {gcs_path} → {local_cache} …")
        subprocess.run(["gcloud", "storage", "cp", gcs_path, local_cache], check=True)
    else:
        print(f"Using cached {local_cache}")
    data = np.load(local_cache)
    print(f"Loaded {local_cache} — shape: {data.shape}")
    return data


def save_plot_to_gcs(fig, filename: str, subdir: str = ""):
    """
    Save a matplotlib figure to GCS monitoring folder.
    filename: e.g. 'loss_epoch_010.png'
    subdir: optional run-specific subfolder under monitoring/ (e.g. 'conditioned_frozen_base')
            so different runs don't overwrite each other's plots.
    """
    fs = get_fs()
    gcs_path = f"{MONITORING_DIR}/{subdir}/{filename}" if subdir else f"{MONITORING_DIR}/{filename}"
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    with fs.open(gcs_path, "wb") as f:
        f.write(buf.read())
    print(f"Saved plot to {gcs_path}")


def save_results_to_gcs(results, filename: str):
    """Upload a pickle results dict to the GCS monitoring/sparse_reconstructions/ folder."""
    import subprocess, tempfile
    local_path = os.path.join("monitoring", "sparse_reconstructions", filename)
    gcs_path = f"{MONITORING_DIR}/sparse_reconstructions/{filename}"
    # file is already saved locally by inference.py — just upload it
    if not os.path.exists(local_path):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as t:
            local_path = t.name
        with open(local_path, "wb") as f:
            pickle.dump(results, f)
    subprocess.run(["gcloud", "storage", "cp", local_path, gcs_path], check=True)
    print(f"Saved results to {gcs_path}")


def save_final_loss_plot(train_losses, val_losses, subdir: str = ""):
    """Save the final loss curve to GCS at end of training."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="train", color="steelblue")
    ax.plot(val_losses, label="val", color="coral")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("DDPM Training — Final")
    ax.legend()
    plt.tight_layout()
    save_plot_to_gcs(fig, "loss_final.png", subdir=subdir)
    plt.close(fig)
    print("Final loss plot saved to GCS.")

# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------


def plot_losses(train_losses, val_losses, epoch, save_to_gcs=True, subdir: str = ""):
    """
    Plot train and val losses. Displays inline and optionally saves to GCS.
    Called every epoch from the training loop.
    """
    try:
        from IPython.display import clear_output

        clear_output(wait=True)
    except ImportError:
        pass  # running headless (e.g. on the TPU VM), no notebook to clear

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="train", color="steelblue")
    ax.plot(val_losses, label="val", color="coral")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"DDPM Training — Epoch {epoch}")
    ax.legend()
    plt.tight_layout()

    if save_to_gcs and len(train_losses) > 0:
        save_plot_to_gcs(fig, f"loss_epoch_{epoch:04d}.png", subdir=subdir)

    plt.close(fig)


def save_residual_plot(res_cond, res_uncond, epoch, subdir: str = ""):
    """Plot the PDE-residual probe (one-step x0, conditioned vs unconditioned) over epochs.
    The noise-MSE stays flat on a frozen base; THIS is where the adapter's progress shows."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(res_uncond, label="unconditioned", color="gray")
    ax.plot(res_cond, label="conditioned", color="crimson")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PDE residual (one-step x0)")
    ax.set_title(f"Conditioning residual probe — Epoch {epoch}")
    ax.legend()
    plt.tight_layout()
    save_plot_to_gcs(fig, f"residual_epoch_{epoch:04d}.png", subdir=subdir)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(params, opt_state, epoch, cfg, subdir: str = ""):
    ckpt_dir = cfg["checkpointing"]["checkpoint_dir"]
    if subdir:
        ckpt_dir = f"{ckpt_dir.rstrip('/')}/{subdir}"
    filename = f"ckpt_epoch_{epoch:04d}.pkl"
    payload = {"params": params, "opt_state": opt_state, "epoch": epoch}

    if ckpt_dir.startswith("gs://"):
        fs = get_fs()
        path = f"{ckpt_dir.rstrip('/')}/{filename}"
        buf = io.BytesIO()
        pickle.dump(payload, buf)
        buf.seek(0)
        with fs.open(path, "wb") as f:
            f.write(buf.read())
    else:
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, filename)
        with open(path, "wb") as f:
            pickle.dump(payload, f)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(path):
    import flax.serialization
    if path.startswith("gs://"):
        import subprocess, tempfile
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as t:
            tmp = t.name
        subprocess.run(["gcloud", "storage", "cp", path, tmp], check=True)
        with open(tmp, "rb") as f:
            ckpt = pickle.load(f)
        os.unlink(tmp)
    else:
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
    params_raw = ckpt["params"]
    if isinstance(params_raw, bytes):
        params = flax.serialization.msgpack_restore(params_raw)
    else:
        params = params_raw
    return params, ckpt.get("opt_state"), ckpt["epoch"]


# ---------------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------------

def nn_fill(sparse_array, mask):
    
    filled_array = sparse_array.copy()
    unknown = ~mask
    _, indices = distance_transform_edt(np.array(unknown), return_indices=True)
    filled_array = filled_array.at[unknown].set(sparse_array[indices[0][unknown] ,indices[1][unknown]])
    
    return filled_array


def lr_input(im, cfg):
    """Downsample to a regular LR grid then upsample back to HR resolution.
    Implements the super-resolution (Task 1) conditioning input.
    method: 'bicubic' or 'bilinear' (set in inference_config.yaml under super_resolution)
    """
    sr_cfg = cfg["super_resolution"]
    lr_res = sr_cfg["lr_res"]
    method = sr_cfg.get("upsample_method", "bicubic")
    stride = im.shape[0] // lr_res
    lr = im[::stride, ::stride, :]
    return [jax.image.resize(lr, im.shape, method=method)]


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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mse(a, b):
    sq_diff = (a-b)**2
    return [jnp.mean(sq_diff), sq_diff]
