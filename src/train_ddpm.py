import functools
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import yaml
from tqdm import tqdm

from models.model import DDPM
from src.utils import load_npy_from_gcs, plot_losses, save_final_loss_plot

tf.config.experimental.set_visible_devices([], "GPU")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def build_samples(data, idx_list, mean, std):
    timesteps = np.arange(data.shape[1] - 2)
    samples = []
    for idx in idx_list:
        for t in timesteps:
            frame0 = (data[idx, t] - mean) / std
            frame1 = (data[idx, t + 1] - mean) / std
            frame2 = (data[idx, t + 2] - mean) / std
            frame = np.stack([frame0, frame1, frame2], axis=-1)
            samples.append(frame)
    return np.array(samples, dtype=np.float32)


def make_tf_ds(samples, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(samples)
    ds = ds.shuffle(10000)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def load_dataset(cfg):
    data_path = cfg["data"]["data_path"]
    batch_size = cfg["training"]["batch_size"]
    train_ratio = cfg["training"]["train_ratio"]
    seeds_sub = cfg["data"]["seeds_subset"]
    time_sub = cfg["data"]["timesteps_subset"]

    if data_path.startswith("gs://"):
        data = load_npy_from_gcs(data_path)
    else:
        data = np.load(data_path)

    data = data[:seeds_sub, :time_sub, :, :]

    np.random.seed(cfg["training"]["seed"])
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)

    n_train = int(train_ratio * len(indices))
    n_val = int(0.1 * len(indices))

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_data = data[train_idx]
    mean = np.mean(train_data)
    std = np.std(train_data)

    train_ds = make_tf_ds(build_samples(data, train_idx, mean, std), batch_size)
    val_ds = make_tf_ds(build_samples(data, val_idx, mean, std), batch_size)
    test_ds = make_tf_ds(build_samples(data, test_idx, mean, std), batch_size)

    return train_ds, val_ds, test_ds, mean, std


# ---------------------------------------------------------------------------
# Train / eval step
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(1, 5))
def train_step(params, model, ims, t, key, alpha_bar, train=True):
    def loss_fn(params):
        eps = jax.random.normal(key, ims.shape)
        noised = (
            jnp.sqrt(alpha_bar[t])[:, None, None, None] * ims
            + jnp.sqrt(1 - alpha_bar[t])[:, None, None, None] * eps
        )
        if train:
            eps_pred = model.apply({"params": params}, noised, t, train=True, rngs={"dropout": key})
        else:
            eps_pred = model.apply({"params": params}, noised, t, train=False)
        return jnp.mean((eps - eps_pred) ** 2)

    if train:
        loss, grads = jax.value_and_grad(loss_fn)(params)
    else:
        loss = loss_fn(params)
        grads = None

    return loss, grads


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


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(cfg):
    key = jax.random.PRNGKey(cfg["training"]["seed"])

    ddpm = DDPM(cfg)
    model = ddpm.unet
    alpha_bar = ddpm.alpha_bar
    T = cfg["diffusion"]["T"]
    batch_size = cfg["training"]["batch_size"]
    n_epochs = cfg["training"]["n_epochs"]
    save_every = cfg["checkpointing"]["save_every_n_epochs"]

    # Initialise model
    key, init_key, dropout_key = jax.random.split(key, 3)
    dummy_x = jnp.ones(
        (1, cfg["data"]["image_size"], cfg["data"]["image_size"], cfg["model"]["in_ch"])
    )
    dummy_t = jnp.ones((1,), dtype=jnp.int32)
    params = model.init({"params": init_key, "dropout": dropout_key}, dummy_x, dummy_t)["params"]

    # Optimiser
    optimizer = optax.adam(learning_rate=cfg["training"]["learning_rate"])
    opt_state = optimizer.init(params)

    # Data
    train_ds, val_ds, _, mean, std = load_dataset(cfg)
    print(f"Dataset loaded — mean={mean:.4f}, std={std:.4f}")

    train_losses, val_losses = [], []

    for epoch in tqdm(range(n_epochs), desc="Epochs"):
        # --- training ---
        epoch_train = []
        for ims in train_ds:
            ims = jnp.array(ims.numpy())
            key, t_key, noise_key = jax.random.split(key, 3)
            t = jax.random.randint(t_key, (batch_size,), minval=0, maxval=T)
            loss, grads = train_step(params, model, ims, t, noise_key, alpha_bar, train=True)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            epoch_train.append(float(loss))

        # --- validation ---
        epoch_val = []
        for ims in val_ds:
            ims = jnp.array(ims.numpy())
            key, t_key, noise_key = jax.random.split(key, 3)
            t = jax.random.randint(t_key, (batch_size,), minval=0, maxval=T)
            loss, _ = train_step(params, model, ims, t, noise_key, alpha_bar, train=False)
            epoch_val.append(float(loss))

        train_losses.append(float(jnp.mean(jnp.array(epoch_train))))
        val_losses.append(float(jnp.mean(jnp.array(epoch_val))))

        tqdm.write(f"Epoch {epoch:04d} — train: {train_losses[-1]:.4f}  val: {val_losses[-1]:.4f}")

        plot_losses(train_losses, val_losses, epoch, save_to_gcs=True)

        if (epoch + 1) % save_every == 0:
            save_checkpoint(params, opt_state, epoch, cfg)

    save_final_loss_plot(train_losses, val_losses)
    return params, train_losses, val_losses


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    params, train_losses, val_losses = train(cfg)
