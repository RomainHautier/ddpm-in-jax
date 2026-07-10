import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from jax.sharding import NamedSharding, PartitionSpec as P

from src.models.model import DDPM
from src.physics_guidance import make_dx_func, make_cond_func
from src.utils import load_checkpoint, save_results_to_gcs


def load_sequence(path, seq_idx):
    """Load sequence `seq_idx` from a .npy or .npz dataset (gs:// or local),
    returning the raw (n_frames, H, W) scalar field. Caches gs:// files locally."""
    local = path
    if path.startswith("gs://") or not os.path.exists(path):
        local = os.path.join("flow-data", os.path.basename(path))
        if not os.path.exists(local):
            import subprocess

            os.makedirs("flow-data", exist_ok=True)
            print(f"Downloading {path} -> {local} ...", flush=True)
            subprocess.run(["gcloud", "storage", "cp", path, local], check=True)
    if local.endswith(".npz"):
        arr = np.load(local)["u3232"]  # (n_seqs, n_frames, H, W)
    else:
        arr = np.load(local, mmap_mode="r")  # avoid loading the whole .npy
    return np.asarray(arr[seq_idx])


def sparse_nnfill_degrade(seq, seq_idx, npz_path="flow-data/kmflow_idx_lst.npz"):
    """BaratiLab-style degradation of a clean (n_frames, H, W) sequence: keep the 1024
    sampled pixels (npz `idx_lst` mask for this sequence) and nearest-neighbour-fill the
    rest. Reproduces how `kmflow_sampled_data_irregnew` was built (verified)."""
    from scipy.ndimage import distance_transform_edt

    idx = np.load(npz_path)["idx_lst"][seq_idx]
    H, W = seq.shape[-2], seq.shape[-1]
    mask = np.zeros(H * W, bool)
    mask[idx] = True
    mask = mask.reshape(H, W)
    _, ind = distance_transform_edt(~mask, return_indices=True)
    return np.stack([f[ind[0], ind[1]] for f in seq])


def grid_downsample_degrade(seq, factor):
    """Deterministic regular-grid degradation of a clean (n_frames, H, W) sequence: keep every
    `factor`-th pixel in each dim (a clean, reproducible sampling anchor vs the random-collocation
    mask), then nearest-neighbour-fill. factor 4 -> 64x64=4096 pts on a 256 grid, etc. Field stays
    full-resolution (256x256) — `factor` sets observation sparsity, not tensor size."""
    from scipy.ndimage import distance_transform_edt

    H, W = seq.shape[-2], seq.shape[-1]
    mask = np.zeros((H, W), bool)
    mask[::factor, ::factor] = True
    _, ind = distance_transform_edt(~mask, return_indices=True)
    return np.stack([f[ind[0], ind[1]] for f in seq])


def build_triplets(seq, mean, std):
    """Normalize with the model's training stats and stack 3 consecutive
    frames as channels -> (n_frames - 2, H, W, 3), matching train_ddpm.build_samples."""
    seq = (seq.astype(np.float32) - mean) / std
    n = seq.shape[0] - 2
    triplets = np.stack([seq[0:n], seq[1 : n + 1], seq[2 : n + 2]], axis=-1)
    return triplets  # (n, H, W, 3)


def make_batched_sampler(ddpm, data_sharding, config, cond_func=None, dx_func=None, guidance_t_max=None, log_every=10):
    """Build a data-parallel batched DDPM sampler. A batch of frames is sharded
    along its leading axis, so the U-Net forward pass at each denoising step runs
    across every TPU chip at once (GSPMD). Mirrors DDPM.sample but for a batch.

    If dx_func is given, applies sampling-time physics guidance, faithfully replicating
    BaratiLab's ddpm_steps: dx = dx_func(x) is computed on the field BEFORE the denoise
    step and subtracted from the sample AFTER it (sample = mean + noise - dx). dx_func
    bakes in lambda and Re (src.physics_guidance.make_dx_func). No retraining.

    If a `probe` callable is passed to sample_batch, it is evaluated on the current field
    every log_every steps (and at the first/last step); sample_batch then returns (xT, log)
    with log a list of (t, probe(xT)). Use it to trace residual/MSE/spectrum vs denoising step
    (works for the baseline dx_func=None too). probe is bound per chunk (e.g. to that chunk's GT)."""
    model = ddpm.unet
    alpha_bar = ddpm.alpha_bar
    beta = ddpm.beta_schedule
    linear_guidance = config[1]['sequence_diffusion']['guidance_lambda']
    learned_guidance = config[0]['conditioning']['inference']['enabled']

    if learned_guidance:
        cond_strength = config[0]['conditioning']['inference']['cond_strength']
    else:
        cond_strength = 0

    @jax.jit
    def denoise_step(params, xT, t, z, condRes=None):
        b = xT.shape[0]

        eps_pred = model.apply({"params": params}, xT, jnp.full((b,), t), train=False)

        if condRes is not None:
            eps_pred_cond = model.apply({"params": params}, xT, jnp.full((b,), t), train=False, condRes=condRes)
            eps_pred = (1 + cond_strength)*(eps_pred_cond) - cond_strength*eps_pred
        
        alpha_t = 1.0 - beta[t]
        abar_t = alpha_bar[t]
        return (1.0 / jnp.sqrt(alpha_t)) * (
            xT - (1.0 - alpha_t) / jnp.sqrt(1.0 - abar_t) * eps_pred
        ) + jnp.sqrt(beta[t]) * z

    def sample_batch(params, x_g, key, t_start, probe=None):
        """x_g: (B, H, W, C) sharded along batch. Noise to level t_start, denoise back."""
        key, init_key = jax.random.split(key)
        keys = jax.random.split(key, t_start + 1)
        eps = jax.device_put(jax.random.normal(init_key, x_g.shape), data_sharding)
        xT = jnp.sqrt(alpha_bar[t_start]) * x_g + jnp.sqrt(1.0 - alpha_bar[t_start]) * eps
        log = [] if probe is not None else None
        
        for t in range(t_start, 0, -1):
            if t % 50 == 0 or t == t_start:
                print(f"      denoising t={t}/{t_start}", flush=True)
            if t > 1:
                z = jax.device_put(jax.random.normal(keys[t], xT.shape), data_sharding)
            else:
                z = jax.device_put(jnp.zeros(xT.shape, xT.dtype), data_sharding)
            # Two orthogonal physics signals, both read the pre-step field xT:
            #   LEARNED: condRes = cond_func(xT) fed to the model (signal set by cond_signal:
            #            gradient -> make_dx_func, field -> make_field_func). Applied INSIDE denoise_step.
            #   LINEAR : dx = dx_func(xT) (always the residual gradient) subtracted AFTER the step
            #            (BaratiLab ddpm_steps). They compose; either can be off.
            condRes = cond_func(xT) if learned_guidance else None
            dx = dx_func(xT) if linear_guidance > 0.0 else None

            xT = denoise_step(params, xT, t, z, condRes=condRes)

            if linear_guidance > 0.0 and (guidance_t_max is None or t <= guidance_t_max ):
                xT = xT - linear_guidance*dx
            
            if probe is not None and (t == t_start or t == 1 or t % log_every == 0):
                log.append((t, probe(xT)))                     # metrics of the field after the step
        return (xT, log) if probe is not None else xT

    return sample_batch


def reconstruct_sequence(cfgs, sampler, params, seq_idx, B, data_sharding, base_key, max_frames=None):
    """Reconstruct every triplet-frame of one sequence by running the K-step /
    S-step refinement, batched and sharded across all chips."""
    sd = cfgs[1]["sequence_diffusion"]
    K, S = sd["K"], sd["S"]
    mean, std = sd["mean"], sd["std"]
    store_iterations = sd.get("store_iterations", False)
    gt_path = sd["gt_data_path"]

    gt_seq = load_sequence(gt_path, seq_idx)
    if sd.get("degrade_input") == "sparse_nnfill":
        # generated/clean flow has no pre-made degraded file: build the BaratiLab input
        # (1024-pt sparse sample at the npz masks + NN-fill) from the clean GT.
        input_seq = sparse_nnfill_degrade(gt_seq, seq_idx)
    else:
        input_seq = load_sequence(cfgs[0]["data"]["data_path"], seq_idx)
    input_triplets = build_triplets(input_seq, mean, std)
    gt_triplets = build_triplets(gt_seq, mean, std)
    assert input_triplets.shape == gt_triplets.shape, (
        f"input/gt mismatch: {input_triplets.shape} vs {gt_triplets.shape}"
    )
    if max_frames is not None:
        input_triplets = input_triplets[:max_frames]
        gt_triplets = gt_triplets[:max_frames]
    n_frames = input_triplets.shape[0]

    # Pad to a multiple of B so every chunk has the same shardable shape
    # (so the jitted step compiles once); padded frames are sliced off afterwards.
    pad = (-n_frames) % B
    inp_padded = (
        np.concatenate([input_triplets, np.zeros((pad,) + input_triplets.shape[1:], np.float32)], axis=0)
        if pad
        else input_triplets
    )
    n_chunks = inp_padded.shape[0] // B
    print(f"  {n_frames} frames -> {n_chunks} chunks of {B} (pad={pad})", flush=True)

    finals = []
    iters_acc = [[] for _ in range(K)] if store_iterations else None
    for ci in range(n_chunks):
        print(f"  chunk {ci + 1}/{n_chunks}", flush=True)
        x_g = jax.device_put(jnp.asarray(inp_padded[ci * B : (ci + 1) * B]), data_sharding)
        for j in range(K):
            key = jax.random.fold_in(base_key, seq_idx * 100000 + ci * K + j)
            print(f"    iteration {j + 1}/{K} (S={S[j]} steps)", flush=True)
            x_g = sampler(params, x_g, key, S[j])
            if store_iterations:
                iters_acc[j].append(np.asarray(x_g, dtype=np.float32))
        finals.append(np.asarray(x_g, dtype=np.float32))

    finals = np.concatenate(finals, axis=0)[:n_frames]
    if store_iterations:
        iters_acc = [np.concatenate(a, axis=0)[:n_frames] for a in iters_acc]

    mse_per = ((gt_triplets - finals) ** 2).reshape(n_frames, -1).mean(axis=1)
    frames = []
    for f in range(n_frames):
        entry = {
            "frame_idx": f,
            # input/ground_truth are derivable from gt_path (gt = load_sequence; input =
            # sparse_nnfill(gt)); drop them to keep pkls ~3x smaller (avoids filling the disk).
            "final": finals[f],
            "mse": float(mse_per[f]),
        }
        if store_iterations:
            entry["iterations"] = [iters_acc[j][f] for j in range(K)]
        frames.append(entry)
    return frames


def run_sequence_inference(cfgs, max_frames=None):
    print(f"JAX devices: {jax.devices()}", flush=True)

    sd = cfgs[1]["sequence_diffusion"]
    ckpt_path = sd["checkpoint"]
    learned_guidance = cfgs[0]['conditioning']['inference']['enabled']
    linear_guidance = sd.get('guidance_lambda', 0)

    K, S = sd["K"], sd["S"]
    mean, std = sd["mean"], sd["std"]
    assert K == len(S), "K and S must have the same length in sequence_diffusion"

    n_devices = jax.device_count()
    B = sd.get("batch_size", 16)
    assert B % n_devices == 0, f"batch_size {B} must be divisible by {n_devices} devices"

    print(f"Loading checkpoint: {ckpt_path}", flush=True)
    ddpm = DDPM(cfgs[0])
    params, _, ckpt_epoch = load_checkpoint(ckpt_path)
    print(f"Loaded checkpoint epoch {ckpt_epoch}", flush=True)

    # Data parallelism: replicate params, shard each frame-batch across all chips.
    mesh = jax.make_mesh((n_devices,), ("data",))
    data_sharding = NamedSharding(mesh, P("data"))
    repl_sharding = NamedSharding(mesh, P())
    params = jax.device_put(params, repl_sharding)

    # Two orthogonal physics signals, built here and passed to the sampler:
    #  - LEARNED (condRes): matches how the adapter was TRAINED — cond_signal 'gradient'
    #    (make_dx_func, residual-min direction) or 'field' (make_field_func, raw residual field,
    #    ENS-style). Evaluated at the conditioning Re (conditioning.inference.re). MUST match the
    #    checkpoint's training signal or the model sees an input it never trained on.
    #  - LINEAR (dx): BaratiLab "Linear" variant — dx = grad_w(mean residual^2)/std at guidance_re,
    #    subtracted after each step, scaled by guidance_lambda (0 = off). Always the gradient.
    n = cfgs[0]["data"]['image_size']
    cond_func = None
    dx_func = None

    if learned_guidance:
        cond_re = float(cfgs[0]["conditioning"]['inference'].get("re", 1000.0))
        cond_signal = cfgs[0]['conditioning']['train'].get('cond_signal', 'gradient')
        cond_func = make_cond_func(cond_signal, n=n, re=cond_re, std=std, mean=mean)
        print(f"Learned guidance ON — cond_signal={cond_signal}, cond_re={cond_re}", flush=True)

    if linear_guidance > 0.0:
        gre = float(sd.get("guidance_re") or 1000.0)
        dx_func = make_dx_func(n=n, re=gre, std=std, mean=mean)
        print(f"Linear guidance ON — lambda={linear_guidance}, guidance_re={gre}", flush=True)

    sampler = make_batched_sampler(ddpm=ddpm, data_sharding=data_sharding, config=cfgs,
                                   cond_func=cond_func, dx_func=dx_func, guidance_t_max=sd.get("guidance_t_max"))

    base_key = jax.random.key(cfgs[1]["inference_seed"])
    gt_path = sd["gt_data_path"]
    input_path = "sparse_nnfill(gt)" if sd.get("degrade_input") == "sparse_nnfill" else cfgs[0]["data"]["data_path"]

    # Which sequences to process. test_set=True -> the last n_test_seqs (the test
    # split, matching train_ddpm.load_dataset); else the single seq_idx.
    d = cfgs[0]["data"]
    total_seqs = d["n_train_seqs"] + d["n_val_seqs"] + d["n_test_seqs"]
    if sd.get("seq_idxs") is not None:
        seq_idxs = list(sd["seq_idxs"])            # explicit list (e.g. simulated seqs [0, 1])
    elif sd.get("test_set", False):
        n_test = d["n_test_seqs"]
        seq_idxs = list(range(total_seqs - n_test, total_seqs))
    else:
        si = sd["seq_idx"]
        seq_idxs = [si if si >= 0 else total_seqs + si]

    print(
        f"Reconstructing sequences {seq_idxs} | batch={B} ({B // n_devices}/chip), "
        f"K={K}, S={S}, input={os.path.basename(input_path)}, gt={os.path.basename(gt_path)}",
        flush=True,
    )

    save_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "monitoring",
        "sequence_reconstructions",
    )
    os.makedirs(save_dir, exist_ok=True)

    for seq_idx in seq_idxs:
        print(f"\n===== Sequence {seq_idx} =====", flush=True)
        frames = reconstruct_sequence(
            cfgs, sampler, params, seq_idx, B, data_sharding, base_key, max_frames
        )
        mean_mse = float(np.mean([fr["mse"] for fr in frames]))
        results = {
            "metadata": {
                "task": "sequence_diffusion",
                "checkpoint": ckpt_path,
                "checkpoint_epoch": ckpt_epoch,
                "seq_idx": seq_idx,
                "input_path": input_path,
                "gt_path": gt_path,
                "K": K,
                "S": S,
                "mean": mean,
                "std": std,
                "inference_seed": cfgs[1]["inference_seed"],
                "n_frames": len(frames),
                "mean_mse": mean_mse,
            },
            "frames": frames,
        }
        filename = f"sequence_reconstruction_{sd.get('out_tag', '')}seq{seq_idx}.pkl"
        local_path = os.path.join(save_dir, filename)
        with open(local_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Sequence {seq_idx}: mean MSE={mean_mse:.5f} -> saved {local_path}", flush=True)
        save_results_to_gcs(results, filename)


if __name__ == "__main__":
    with open("configs/inference_config.yaml", "r") as f:
        inference_cfg = yaml.safe_load(f)

    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    run_sequence_inference([cfg, inference_cfg])
