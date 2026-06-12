import jax
import jax.numpy as jnp
import numpy as np
import yaml
import os, pickle

from src.models.model import DDPM
from src.train_ddpm import load_dataset
from src.utils import load_checkpoint, mse
from jax.sharding import NamedSharding, PartitionSpec as P


def run_inference(cfgs):
    """This function performs ddpm inference for a range of noising timesteps."""

    print(f"JAX devices: {jax.devices()}", flush=True)

    # Load the model
    ckpt_path = cfgs[1]["plain_diffusion"]["checkpoint"]
    print(f"Loading checkpoint: {ckpt_path}", flush=True)
    ddpm = DDPM(cfgs[0])
    params, _, ckpt_epoch = load_checkpoint(ckpt_path)
    print(f"Loaded checkpoint epoch {ckpt_epoch}", flush=True)

    n_devices = jax.device_count()
    mesh = jax.make_mesh((n_devices,), ("data",))
    repl_sharding = NamedSharding(mesh, P())
    data_sharding = NamedSharding(mesh, P("data"))
    params = jax.device_put(params, repl_sharding)

    n_test_samples = cfgs[1]["plain_diffusion"]["n_samples"]
    S = cfgs[1]["plain_diffusion"]["S"]
    _, _, test_ds, mean, std = load_dataset(cfgs[0], max_test_samples=n_test_samples, n_devices=n_devices)

    base_key = jax.random.key(cfgs[1]["inference_seed"])

    task = "plain_diffusion"
    
    results = {
        "metadata": {
            "task": task,
            "checkpoint": ckpt_path,
            "checkpoint_epoch": ckpt_epoch,
            "S": S,
            "inference_seed": cfgs[1]["inference_seed"],
            "n_samples": n_test_samples,
            "train_config": cfgs[0],
        },
        "samples": [],
    }

    print(f"Running inference (task {task}): {n_test_samples} images, S={S}", flush=True)

    for im_idx, batch in enumerate(test_ds):
        if im_idx * n_devices >= n_test_samples:
            break
        print(f"\n[Image {im_idx + 1}/{n_test_samples}]", flush=True)
        
        ims = jax.device_put(jnp.array(batch), data_sharding)

        actual_B = ims.shape[0]

        batch_entries = []
        for b in range(actual_B):
            global_idx = im_idx * n_devices + b
            batch_entries.append({
            "image_idx": global_idx,
            "ground_truth": np.array(ims[b]),
            "denoising_timesteps": {s: None for s in S},
            "mse": {s: None for s in S}

        })
        
            
        for s_idx, s in enumerate(S):
            noise_key = jax.random.fold_in(base_key, im_idx * len(S) + s_idx)
            print(f"(S={s} denoising steps)", flush=True)
            x0 = ddpm.sample(params=params, dims=ims.shape, key=noise_key, x_g=ims, t_start=s)
            for b in range(actual_B):
                mean_se, per_pixel_se = mse(ims[b], x0[b])
                batch_entries[b]["denoising_timesteps"][s] = np.array(x0[b])
                batch_entries[b]["mse"][s] = (float(mean_se), np.array(per_pixel_se))

        results["samples"].extend(batch_entries)

    
    filename = f"plain_ddpm_reconstructions_baseline.pkl"

    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "base_results")
    os.makedirs(save_dir, exist_ok=True)
    local_path = os.path.join(save_dir, filename)
    with open(local_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results locally to {local_path}", flush=True)

    import subprocess
    gcs_path = f"gs://ddpm-thesis-rh/base_results/{filename}"
    subprocess.run(["gcloud", "storage", "cp", local_path, gcs_path], check=True)
    print(f"Saved to {gcs_path}", flush=True)
        

if __name__ == "__main__":
    
    with open("configs/inference_config.yaml", "r") as f:
        inference_cfg = yaml.safe_load(f)
    
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    cfgs = [cfg, inference_cfg]

    run_inference(cfgs)