import os
import pickle
from functools import partial


import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import yaml
from flax.traverse_util import path_aware_map, flatten_dict
from jax.sharding import NamedSharding, PartitionSpec as P
from tqdm import tqdm

from src.models.model import DDPM, ConditionalUnet
from src.utils import load_npy_from_gcs, plot_losses, save_final_loss_plot, save_checkpoint, load_checkpoint, save_residual_plot
from src.physics_guidance import make_cond_func, make_residual_loss
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


def build_params(model, cfg, key):
    key, init_key, dropout_key = jax.random.split(key, 3)
    dummy_x = jnp.ones(
        (1, cfg["data"]["image_size"], cfg["data"]["image_size"], cfg["model"]["in_ch"])
    )
    dummy_t = jnp.ones((1,), dtype=jnp.int32)
    cond = cfg['conditioning']['train']['enabled']
    cond_kw = {"condRes": jnp.ones_like(dummy_x)} if cond else {}
    params = model.init({"params": init_key, "dropout": dropout_key}, dummy_x, dummy_t, **cond_kw)["params"]

    if cond and cfg['conditioning']['train']["pretrained_ckpt"] is not None:
        pretrained_params, _, _ = load_checkpoint(cfg['conditioning']['train']["pretrained_ckpt"])
        # overwriting the dummy parameters by those of the pretrained model
        init_flat = flatten_dict(params)
        pretrained_flat = flatten_dict(pretrained_params)
        for path, w in pretrained_flat.items():
            assert path in init_flat
            assert init_flat[path].shape  == w.shape

        extra_modules = {path[0] for path in set(init_flat) - set(pretrained_flat)}
        assert extra_modules <= {"cond_in", "cond_hidden", "cond_combine"}, f"unexpected extra modules {extra_modules}"
        
        
        params = {**params, **pretrained_params}
    
    return params




def params_opti_mapping(path, _leaf):
    name = '/'.join(str(k) for k in path)
    return "train" if "cond_" in name else "freeze"

def build_optimizer(cfg, params):
    lr = cfg['training']['learning_rate']
    
    if cfg['conditioning']['train']['enabled'] and cfg['conditioning']['train']['freeze_base']:
        labels = path_aware_map(params_opti_mapping, params)

        return optax.multi_transform({"train": optax.adam(lr), "freeze": optax.set_to_zero()}, labels)
    
    return optax.adam(learning_rate=cfg["training"]["learning_rate"])


def make_tf_ds(samples, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(samples)
    if shuffle:
        ds = ds.shuffle(10000)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def load_dataset(cfg, max_test_samples=None, n_devices=None):
    data_path = cfg["data"]["data_path"]
    if n_devices is not None:
        batch_size = n_devices
    elif max_test_samples is not None:
        batch_size = 1
    else:
        batch_size = cfg["training"]["batch_size"]
    n_train_seqs = cfg["data"]["n_train_seqs"]  # 32 (80%)
    n_val_seqs   = cfg["data"]["n_val_seqs"]    # 4  (10%)
    n_test_seqs  = cfg["data"]["n_test_seqs"]   # 4  (10%)

    if data_path.startswith("gs://"):
        data = load_npy_from_gcs(data_path)
    else:
        data = np.load(data_path)

    train_idx = np.arange(n_train_seqs)
    val_idx   = np.arange(n_train_seqs, n_train_seqs + n_val_seqs)
    test_idx  = np.arange(n_train_seqs + n_val_seqs, n_train_seqs + n_val_seqs + n_test_seqs)

    if max_test_samples:
        test_idx = test_idx[:max_test_samples]

    mean = np.mean(data[train_idx])
    std  = np.std(data[train_idx])

    n_triplets = data.shape[1] - 2
    print(f"Dataset split — train: {len(train_idx)} seqs, val: {len(val_idx)} seqs, test: {len(test_idx)} seqs")
    print(f"Triplets per seq: {n_triplets}  |  mean={mean:.4f}, std={std:.4f}")

    train_ds = make_tf_ds(build_samples(data, train_idx, mean, std), batch_size, shuffle=True)
    val_ds   = make_tf_ds(build_samples(data, val_idx,   mean, std), batch_size, shuffle=False)
    test_ds  = make_tf_ds(build_samples(data, test_idx,  mean, std), batch_size, shuffle=False)

    return train_ds, val_ds, test_ds, mean, std


# ---------------------------------------------------------------------------
# Train / eval step
# ---------------------------------------------------------------------------


def make_steps(model, optimizer, alpha_bar, cond_func):
    """Build jitted train/eval steps closing over the (static) model, optimizer
    and noise schedule. Batches are sharded across all devices, so JAX's GSPMD
    runs the forward/backward pass data-parallel over every TPU chip and
    all-reduces the gradients automatically."""

    def _noised(params, ims, t, key):
        eps = jax.random.normal(key, ims.shape)
        noised = (
            jnp.sqrt(alpha_bar[t])[:, None, None, None] * ims
            + jnp.sqrt(1 - alpha_bar[t])[:, None, None, None] * eps
        )
        return eps, noised

    @partial(jax.jit, static_argnums=(3,))
    def train_step(params, opt_state, ims, condition, t, key):
        def loss_fn(params):
            dropout_key, noise_key = jax.random.split(key)
            eps, noised = _noised(params, ims, t, noise_key)
            condRes = cond_func(noised) if condition else None
            eps_pred = model.apply(
                {"params": params}, noised, t, train=True, rngs={"dropout": dropout_key}, condRes=condRes
            )
            return jnp.mean((eps - eps_pred) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def eval_step(params, ims, t, key):
        eps, noised = _noised(params, ims, t, key)
        eps_pred = model.apply({"params": params}, noised, t, train=False)
        return jnp.mean((eps - eps_pred) ** 2)

    return train_step, eval_step



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
    # run-specific monitoring subfolder so plots don't overwrite other runs'
    run_name = cfg.get("monitoring", {}).get("run_name", "")

    # Initialise model
    params = build_params(model, cfg, key)


    if cfg['conditioning']['train']['enabled']:
        assert isinstance(ddpm.unet, ConditionalUnet)
    
        dum_x = jnp.ones((1, cfg['data']['image_size'], cfg['data']['image_size'], cfg['model']['in_ch']))
        tt = jnp.ones((1,), dtype = jnp.int32)
        out_none = model.apply({"params": params}, dum_x, tt, train=False, condRes = None)
        out_cond = model.apply({"params": params}, dum_x, tt, train=False, condRes = jnp.ones_like(dum_x))
        assert jnp.allclose(out_cond, out_none, atol = 1e-5)
    
    # Optimiser -> INSERT CONDITIONAL OPTI INIT
    optimizer = build_optimizer(cfg, params)
    opt_state = optimizer.init(params)
    

    # Data parallelism: replicate the model across every chip and shard each
    # batch along its leading (batch) axis. The loss is a single mean over the
    # *global* batch of `batch_size`, so the gradient GSPMD all-reduces is
    # exactly the batch-of-`batch_size` gradient — identical to single-device
    # training, just split 4 ways. No manual gradient accumulation needed.
    n_devices = jax.device_count()
    if batch_size % n_devices != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by the {n_devices} "
            "available devices for data-parallel training."
        )
    mesh = jax.make_mesh((n_devices,), ("data",))
    data_sharding = NamedSharding(mesh, P("data"))
    repl_sharding = NamedSharding(mesh, P())
    params = jax.device_put(params, repl_sharding)
    opt_state = jax.device_put(opt_state, repl_sharding)
    print(
        f"Data-parallel over {n_devices} devices "
        f"(global batch {batch_size}, {batch_size // n_devices} samples/device)."
    )

    

    # Data
    train_ds, val_ds, _, mean, std = load_dataset(cfg)
    print(f"Dataset loaded — mean={mean:.4f}, std={std:.4f}")
    
    # init function to calculate conditional physics guidance
    n = cfg['data']['image_size']
    re = cfg['conditioning']['train']['re']
    cond_proba = cfg['conditioning']['train']['proba']
    cond_signal = cfg['conditioning']['train'].get('cond_signal', 'gradient')
    cond_func = make_cond_func(cond_signal, n=n, re=re, std=std, mean=mean)
    
    # Make steps
    train_step, eval_step = make_steps(model, optimizer, alpha_bar, cond_func)

    # Residual probe: one-step x0 PDE-residual (conditioned vs unconditioned) on a FIXED val
    # batch each epoch. The noise-MSE stays flat on a frozen base, so this is what actually
    # shows the adapter learning. Fixed noise/timestep => comparable across epochs.
    enabled = cfg['conditioning']['train']['enabled']
    res_cond_hist, res_uncond_hist = [], []
    if enabled:
        res_loss_fn = make_residual_loss(n=n, re=re, std=std, mean=mean)
        t_res = 200
        res_key = jax.random.PRNGKey(cfg["training"]["seed"] + 1)
        fixed_val = jax.device_put(jnp.asarray(next(iter(val_ds)).numpy()), data_sharding)

        @jax.jit
        def residual_probe(params):
            abar = alpha_bar[t_res]
            eps = jax.random.normal(res_key, fixed_val.shape)
            noised = jnp.sqrt(abar) * fixed_val + jnp.sqrt(1 - abar) * eps
            tt = jnp.full((fixed_val.shape[0],), t_res)
            x0hat = lambda ep: (noised - jnp.sqrt(1 - abar) * ep) / jnp.sqrt(abar)
            eps_c = model.apply({"params": params}, noised, tt, train=False, condRes=cond_func(noised))
            eps_u = model.apply({"params": params}, noised, tt, train=False)
            return res_loss_fn(x0hat(eps_c)).mean(), res_loss_fn(x0hat(eps_u)).mean()

    train_losses, val_losses = [], []

    for epoch in tqdm(range(n_epochs), desc="Epochs"):
        # --- training ---
        epoch_train = []
        counter = 0
        params_saved = params

        for ims in train_ds:
            ims = jax.device_put(jnp.asarray(ims.numpy()), data_sharding)
            key, t_key, noise_key = jax.random.split(key, 3)
            t = jax.random.randint(t_key, (batch_size,), minval=0, maxval=T)
            t = jax.device_put(t, data_sharding)
            # classifier-free dropout decided host-side: True (conditional) ~ 1 - proba
            condition = bool(np.random.rand() >= cond_proba)
            params, opt_state, loss = train_step(params, opt_state, ims, condition, t, noise_key)
            epoch_train.append(float(loss))
            
            counter += 1
            if counter == 1 and cfg['conditioning']['train']['freeze_base']:
                maxabs = lambda a, b: float(jnp.abs(a-b).max())
                assert maxabs(params_saved["Conv_0"]["kernel"], params["Conv_0"]["kernel"]) == 0.0
                assert maxabs(params_saved["cond_combine"]["kernel"], params["cond_combine"]["kernel"]) > 0.0


            

        # --- validation ---
        epoch_val = []
        for ims in val_ds:
            ims = jax.device_put(jnp.asarray(ims.numpy()), data_sharding)
            key, t_key, noise_key = jax.random.split(key, 3)
            t = jax.random.randint(t_key, (batch_size,), minval=0, maxval=T)
            t = jax.device_put(t, data_sharding)
            loss = eval_step(params, ims, t, noise_key)
            epoch_val.append(float(loss))

        train_losses.append(float(jnp.mean(jnp.array(epoch_train))))
        val_losses.append(float(jnp.mean(jnp.array(epoch_val))))

        tqdm.write(f"Epoch {epoch:04d} — train: {train_losses[-1]:.4f}  val: {val_losses[-1]:.4f}")

        plot_losses(train_losses, val_losses, epoch, save_to_gcs=True, subdir=run_name)

        if enabled:
            r_c, r_u = residual_probe(params)
            res_cond_hist.append(float(r_c))
            res_uncond_hist.append(float(r_u))
            save_residual_plot(res_cond_hist, res_uncond_hist, epoch, subdir=run_name)
            tqdm.write(f"           residual  cond={res_cond_hist[-1]:.3e}  uncond={res_uncond_hist[-1]:.3e}")

        if (epoch + 1) % save_every == 0:
            save_checkpoint(params, opt_state, epoch, cfg, subdir=run_name)

    save_final_loss_plot(train_losses, val_losses, subdir=run_name)
    return params, train_losses, val_losses


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    print(f"Loading config: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    params, train_losses, val_losses = train(cfg)
