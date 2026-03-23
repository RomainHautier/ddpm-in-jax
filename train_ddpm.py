import os
import io
import functools
import pickle
import numpy as np
import matplotlib.pyplot as plt
import gcsfs

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from flax import linen as nn
from flax.serialization import to_bytes, from_bytes
import optax
from tqdm import tqdm

# ── GCS config ──────────────────────────────────────────────────────────────
GCS_BUCKET = "thesis-project-bucket-rh-ucl"
MONITORING_DIR = f"gs://{GCS_BUCKET}/monitoring"
DATA_PATH = f"gs://{GCS_BUCKET}/flow-data/kf_2d_re1000_256_40seed (1).npy"


def get_fs():
    """Return an authenticated GCSFileSystem."""
    return gcsfs.GCSFileSystem(project=GCS_BUCKET)


def load_npy_from_gcs(gcs_path: str) -> np.ndarray:
    fs = get_fs()
    with fs.open(gcs_path, 'rb') as f:
        data = np.load(f)
    print(f"Loaded {gcs_path} — shape: {data.shape}")
    return data


def save_plot_to_gcs(fig, filename: str):
    fs = get_fs()
    gcs_path = f"{MONITORING_DIR}/{filename}"
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    with fs.open(gcs_path, 'wb') as f:
        f.write(buf.read())
    print(f"Saved plot to {gcs_path}")


CHECKPOINT_DIR = f"gs://{GCS_BUCKET}/checkpoints/ddpm"


def save_checkpoint(params, opt_state, epoch, train_losses, val_losses):
    """Save model checkpoint to GCS."""
    fs = get_fs()
    ckpt = {
        'params': to_bytes(params),
        'opt_state': to_bytes(opt_state),
        'epoch': epoch,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }
    gcs_path = f"{CHECKPOINT_DIR}/ckpt_epoch_{epoch:04d}.pkl"
    with fs.open(gcs_path, 'wb') as f:
        pickle.dump(ckpt, f)
    print(f"Checkpoint saved to {gcs_path}")


def load_checkpoint(params_template, opt_state_template, path=None):
    """Load latest (or specific) checkpoint from GCS. Returns None if none found."""
    fs = get_fs()
    if path is None:
        try:
            files = sorted(fs.ls(CHECKPOINT_DIR.replace("gs://", "")))
            if not files:
                return None
            path = f"gs://{files[-1]}"
        except FileNotFoundError:
            return None
    with fs.open(path, 'rb') as f:
        ckpt = pickle.load(f)
    params = from_bytes(params_template, ckpt['params'])
    opt_state = from_bytes(opt_state_template, ckpt['opt_state'])
    print(f"Resumed from {path} (epoch {ckpt['epoch']})")
    return params, opt_state, ckpt['epoch'], ckpt['train_losses'], ckpt['val_losses']


def plot_losses(train_losses, val_losses, epoch, save_to_gcs=True):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label='train', color='steelblue')
    ax.plot(val_losses,   label='val',   color='coral')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'DDPM Training — Epoch {epoch}')
    ax.legend()
    plt.tight_layout()
    if save_to_gcs and len(train_losses) > 0:
        save_plot_to_gcs(fig, f'loss_epoch_{epoch:04d}.png')
    plt.close(fig)


# ── Model ───────────────────────────────────────────────────────────────────

def sinusoidal_time_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(half) / half)
    args = t[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


class DDPMResnet(nn.Module):
    filters: int = 160
    dropout_p: float = 0.5

    @nn.compact
    def __call__(self, x, time_embed, train=True):
        residual = x
        h = nn.GroupNorm(8)(x)
        h = jax.nn.silu(h)
        h = nn.Conv(features=self.filters, kernel_size=(3, 3), padding='CIRCULAR')(h)
        t = nn.Dense(self.filters)(jax.nn.silu(time_embed))
        h = h + t[:, None, None, :]
        h = nn.GroupNorm(8)(h)
        h = jax.nn.silu(h)
        h = nn.Dropout(rate=self.dropout_p)(h, deterministic=not train)
        h = nn.Conv(features=self.filters, kernel_size=(3, 3), padding='CIRCULAR')(h)
        if x.shape[-1] != self.filters:
            residual = nn.Conv(self.filters, (1, 1), padding='CIRCULAR')(residual)
        return h + residual


class SelfAttention(nn.Module):
    num_groups: int = 32

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        h = nn.GroupNorm(self.num_groups)(x)
        h_flat = h.reshape(B, H * W, C)
        q = nn.Dense(C)(h_flat)
        k = nn.Dense(C)(h_flat)
        v = nn.Dense(C)(h_flat)
        scaling = jnp.sqrt(C)
        attn = jax.nn.softmax(q @ k.transpose(0, 2, 1) / scaling, axis=-1)
        out = attn @ v
        out = nn.Dense(C)(out)
        out = out.reshape(B, H, W, C)
        return out + x


class Unet(nn.Module):
    ch: int = 64
    ch_mult: tuple = (1, 1, 1, 2)
    out_ch: int = 3
    in_ch: int = 3
    n_resnet_blocks: int = 1
    dropout_p: float = 0.0
    freq_dim: int = 128

    @nn.compact
    def __call__(self, x, t, train=True):
        ch = self.ch
        temp_ch = ch * 4

        time_embed = sinusoidal_time_embedding(t, dim=self.freq_dim)
        time_embed = nn.Dense(temp_ch)(time_embed)
        time_embed = jax.nn.silu(time_embed)
        time_embed = nn.Dense(temp_ch)(time_embed)

        h = nn.Conv(ch, kernel_size=(3, 3), padding='CIRCULAR')(x)
        hs = [h]

        for _ in range(self.n_resnet_blocks):
            h = DDPMResnet(ch * self.ch_mult[0], self.dropout_p)(h, time_embed=time_embed, train=train)
            hs.append(h)

        h = nn.Conv(ch * self.ch_mult[0], kernel_size=(3, 3), strides=(2, 2), padding=((0, 1), (0, 1)))(h)
        hs.append(h)

        for _ in range(self.n_resnet_blocks):
            h = DDPMResnet(ch * self.ch_mult[1], self.dropout_p)(h, time_embed=time_embed, train=train)
            hs.append(h)

        h = nn.Conv(ch * self.ch_mult[1], kernel_size=(3, 3), strides=(2, 2), padding=((0, 1), (0, 1)))(h)
        hs.append(h)

        for _ in range(self.n_resnet_blocks):
            h = DDPMResnet(ch * self.ch_mult[2], self.dropout_p)(h, time_embed=time_embed, train=train)
            hs.append(h)

        h = nn.Conv(ch * self.ch_mult[2], kernel_size=(3, 3), strides=(2, 2), padding=((0, 1), (0, 1)))(h)
        hs.append(h)

        for _ in range(self.n_resnet_blocks):
            h = DDPMResnet(ch * self.ch_mult[3], self.dropout_p)(h, time_embed=time_embed, train=train)
            hs.append(h)

        # bottleneck
        h = DDPMResnet(ch * self.ch_mult[-1], self.dropout_p)(h, time_embed=time_embed, train=train)
        h = SelfAttention(num_groups=8)(h)
        h = DDPMResnet(ch * self.ch_mult[-1], self.dropout_p)(h, time_embed=time_embed, train=train)

        # decoder
        for _ in range(self.n_resnet_blocks + 1):
            h = DDPMResnet(ch * self.ch_mult[3], self.dropout_p)(
                jnp.concatenate([h, hs.pop()], axis=-1), time_embed=time_embed, train=train)
        B, H, W, C = h.shape
        h = jax.image.resize(h, (B, H * 2, W * 2, C), method='nearest')
        h = nn.Conv(ch * self.ch_mult[3], kernel_size=(3, 3), padding='CIRCULAR')(h)

        for _ in range(self.n_resnet_blocks + 1):
            h = DDPMResnet(ch * self.ch_mult[2], self.dropout_p)(
                jnp.concatenate([h, hs.pop()], axis=-1), time_embed=time_embed, train=train)
        B, H, W, C = h.shape
        h = jax.image.resize(h, (B, H * 2, W * 2, C), method='nearest')
        h = nn.Conv(ch * self.ch_mult[2], kernel_size=(3, 3), padding='CIRCULAR')(h)

        for _ in range(self.n_resnet_blocks + 1):
            h = DDPMResnet(ch * self.ch_mult[1], self.dropout_p)(
                jnp.concatenate([h, hs.pop()], axis=-1), time_embed=time_embed, train=train)
        B, H, W, C = h.shape
        h = jax.image.resize(h, (B, H * 2, W * 2, C), method='nearest')
        h = nn.Conv(ch * self.ch_mult[1], kernel_size=(3, 3), padding='CIRCULAR')(h)

        for _ in range(self.n_resnet_blocks + 1):
            h = DDPMResnet(ch * self.ch_mult[0], self.dropout_p)(
                jnp.concatenate([h, hs.pop()], axis=-1), time_embed=time_embed, train=train)

        h = nn.GroupNorm(8)(h)
        h = jax.nn.silu(h)
        return nn.Conv(self.out_ch, kernel_size=(3, 3), padding='CIRCULAR')(h)


# ── Diffusion helpers ───────────────────────────────────────────────────────

def calc_alpha(beta):
    return jnp.cumprod(1 - beta)


def forward_process(ims, t, alpha, eps):
    return (jnp.sqrt(alpha[t])[:, None, None, None] * ims
            + jnp.sqrt(1 - alpha[t])[:, None, None, None] * eps)


# ── Sharding setup ──────────────────────────────────────────────────────────

def setup_sharding():
    """Create a 1D mesh over all available devices, sharding along batch."""
    devices = jax.devices()
    num_devices = len(devices)
    print(f"Found {num_devices} device(s): {devices}")
    mesh = Mesh(np.array(devices), axis_names=('batch',))
    data_sharding = NamedSharding(mesh, P('batch'))
    replicated = NamedSharding(mesh, P())
    return mesh, data_sharding, replicated, num_devices


def shard_batch(x, data_sharding):
    """Place a batch array onto the mesh, split along dim 0."""
    return jax.device_put(x, data_sharding)


# ── Data ────────────────────────────────────────────────────────────────────

def build_samples(data, idx_list, mean, std):
    timesteps = np.arange(data.shape[1] - 2)
    samples = []
    for idx in idx_list:
        for t in timesteps:
            frame0 = (data[idx, t]   - mean) / std
            frame1 = (data[idx, t+1] - mean) / std
            frame2 = (data[idx, t+2] - mean) / std
            samples.append(np.stack([frame0, frame1, frame2], axis=-1))
    return np.array(samples, dtype=np.float32)


def load_and_split(batch_size, num_devices, train_ratio=0.7):
    """Load dataset from GCS, split into train/val/test numpy arrays."""
    data = load_npy_from_gcs(DATA_PATH)

    np.random.seed(1)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)

    n = len(indices)
    n_val   = max(1, int(0.1 * n))
    n_test  = max(1, int(0.2 * n))
    n_train = n - n_val - n_test

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    train_data = data[train_idx]
    mean = np.mean(train_data)
    std  = np.std(train_data)

    train_samples = build_samples(data, train_idx, mean, std)
    val_samples   = build_samples(data, val_idx,   mean, std)
    test_samples  = build_samples(data, test_idx,  mean, std)

    # ensure batch size is divisible by num_devices
    effective_bs = batch_size
    if effective_bs % num_devices != 0:
        effective_bs = (batch_size // num_devices) * num_devices
        print(f"Adjusted batch size to {effective_bs} (divisible by {num_devices} devices)")

    return train_samples, val_samples, test_samples, mean, std, effective_bs


def make_batches(samples, batch_size, rng):
    """Yield shuffled batches as jnp arrays (no tf.data dependency)."""
    n = len(samples)
    perm = rng.permutation(n)
    for i in range(0, n - batch_size + 1, batch_size):
        yield jnp.array(samples[perm[i:i + batch_size]])


# ── Training ────────────────────────────────────────────────────────────────

def create_train_step(model, optimizer, alpha_bar):
    """Return a jitted, sharded train step function."""

    def _train_step(params, opt_state, ims, t, key):
        def loss_fn(params):
            eps = jax.random.normal(key, ims.shape)
            noised = forward_process(ims, t, alpha_bar, eps)
            eps_pred = model.apply(
                {'params': params}, noised, t, train=True,
                rngs={'dropout': key})
            return jnp.mean((eps - eps_pred) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    return _train_step


def create_eval_step(model, alpha_bar):
    """Return a jitted, sharded eval step function."""

    def _eval_step(params, ims, t, key):
        eps = jax.random.normal(key, ims.shape)
        noised = forward_process(ims, t, alpha_bar, eps)
        eps_pred = model.apply({'params': params}, noised, t, train=False)
        return jnp.mean((eps - eps_pred) ** 2)

    return _eval_step


def train(n_epochs=300, batch_size=32, lr=2e-4, T=1000, dropout_p=0.1,
          ckpt_every=10, resume=True):
    # ── sharding ──
    mesh, data_sharding, replicated, num_devices = setup_sharding()

    # ── diffusion schedule ──
    beta_schedule = jnp.linspace(0.0001, 0.02, T)
    alpha_bar = calc_alpha(beta_schedule)

    # ── model init ──
    key = jax.random.PRNGKey(0)
    model = Unet(ch=64, ch_mult=(1, 1, 1, 2), dropout_p=dropout_p)
    key, init_key, dropout_key = jax.random.split(key, 3)
    dummy_x = jnp.ones((1, 256, 256, 3))
    dummy_t = jnp.ones((1,), dtype=jnp.int32)
    variables = model.init({"params": init_key, "dropout": dropout_key}, dummy_x, dummy_t)
    params = variables['params']

    # ── optimizer ──
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    # ── try resuming from checkpoint ──
    start_epoch = 0
    train_losses, val_losses = [], []

    if resume:
        ckpt = load_checkpoint(params, opt_state)
        if ckpt is not None:
            params, opt_state, start_epoch, train_losses, val_losses = ckpt
            start_epoch += 1  # resume from next epoch

    # replicate params/opt_state across devices
    params = jax.device_put(params, replicated)
    opt_state = jax.device_put(opt_state, replicated)

    # ── data ──
    train_samples, val_samples, _, mean, std, batch_size = load_and_split(
        batch_size, num_devices)
    print(f"Train samples: {train_samples.shape}, Val samples: {val_samples.shape}")

    # ── jitted steps (inside mesh context) ──
    raw_train_step = create_train_step(model, optimizer, alpha_bar)
    raw_eval_step = create_eval_step(model, alpha_bar)

    @jax.jit
    def train_step(params, opt_state, ims, t, key):
        return raw_train_step(params, opt_state, ims, t, key)

    @jax.jit
    def eval_step(params, ims, t, key):
        return raw_eval_step(params, ims, t, key)

    # ── training loop ──
    np_rng = np.random.default_rng(42)

    with mesh:
        for epoch in tqdm(range(start_epoch, n_epochs), desc='Epochs',
                          initial=start_epoch, total=n_epochs):
            # — train —
            epoch_loss = []
            for ims in make_batches(train_samples, batch_size, np_rng):
                ims = shard_batch(ims, data_sharding)
                key, time_key, noise_key = jax.random.split(key, 3)
                t = jax.random.randint(time_key, (batch_size,), minval=0, maxval=T)
                t = shard_batch(t, data_sharding)

                params, opt_state, loss = train_step(
                    params, opt_state, ims, t, noise_key)
                epoch_loss.append(float(loss))

            train_losses.append(np.mean(epoch_loss))

            # — val —
            epoch_val = []
            for ims in make_batches(val_samples, batch_size, np_rng):
                ims = shard_batch(ims, data_sharding)
                key, time_key, noise_key = jax.random.split(key, 3)
                t = jax.random.randint(time_key, (batch_size,), minval=0, maxval=T)
                t = shard_batch(t, data_sharding)

                loss = eval_step(params, ims, t, noise_key)
                epoch_val.append(float(loss))

            val_losses.append(np.mean(epoch_val) if epoch_val else 0.0)

            # — logging —
            print(f"Epoch {epoch:4d} | train {train_losses[-1]:.4f} | val {val_losses[-1]:.4f}")

            # — checkpoint & plot every ckpt_every epochs —
            if (epoch + 1) % ckpt_every == 0:
                # pull params to host for serialization
                host_params = jax.device_get(params)
                host_opt_state = jax.device_get(opt_state)
                save_checkpoint(host_params, host_opt_state, epoch,
                                train_losses, val_losses)
                plot_losses(train_losses, val_losses, epoch)

    # final checkpoint + plot
    host_params = jax.device_get(params)
    host_opt_state = jax.device_get(opt_state)
    save_checkpoint(host_params, host_opt_state, n_epochs - 1,
                    train_losses, val_losses)
    plot_losses(train_losses, val_losses, n_epochs, save_to_gcs=True)
    print("Training complete.")
    return params, train_losses, val_losses


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume from latest checkpoint')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--ckpt_every', type=int, default=10)
    args = parser.parse_args()

    params, train_losses, val_losses = train(
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        ckpt_every=args.ckpt_every, resume=args.resume)
