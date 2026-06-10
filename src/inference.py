import os

import jax
import jax.numpy as jnp
import numpy as np
import pickle as pkl
import yaml

from src.models.model import DDPM
from src.train_ddpm import load_dataset
from src.utils import load_checkpoint, sparsify_input


def run_inference(cfgs):
    """This function performs low-to-high resolution flow reconstruction
    using a pretrained U-Net backbone diffusion model. An high res input
    is sparsified, sparse pixels are filled using NN methods to obtain
    the low res flow and fed to the model to perform inference."""

    print(f"JAX devices: {jax.devices()}", flush=True)

    # Load the model
    ckpt_path = cfgs[1]["sparse_diffusion"]["checkpoint"]
    print(f"Loading checkpoint: {ckpt_path}", flush=True)
    ddpm = DDPM(cfgs[0])
    params, _, ckpt_epoch = load_checkpoint(ckpt_path)
    print(f"Loaded checkpoint epoch {ckpt_epoch}", flush=True)

    n_test_samples = cfgs[1]["sparse_diffusion"]["n_samples"]
    K = cfgs[1]["sparse_diffusion"]["K"]
    S = cfgs[1]["sparse_diffusion"]["S"]

    assert K == len(S), "K and S must have the same length in inference_config.yaml"

    _, _, test_ds, mean, std = load_dataset(cfgs[0], max_test_samples=n_test_samples)

    noise_key = jax.random.key(cfgs[1]["inference_seed"])

    results = {
        "metadata": {
            "checkpoint": ckpt_path,
            "checkpoint_epoch": ckpt_epoch,
            "sparsity_ratio": cfgs[1]["sparsity_ratio"],
            "K": K,
            "S": S,
            "inference_seed": cfgs[1]["inference_seed"],
            "n_samples": n_test_samples,
        },
        "samples": [],
    }

    n_seeds = len(cfgs[1]["nn_fill"]["seeds"])
    print(f"Running inference: {n_test_samples} images, {n_seeds} seeds, K={K}, S={S}", flush=True)

    for im_idx, batch in enumerate(test_ds):
        print(f"\n[Image {im_idx + 1}/{n_test_samples}]", flush=True)
        im = jnp.array(batch).squeeze(0)

        sparse_inputs = sparsify_input(im, cfgs[1])

        sample_entry = {
            "image_idx": im_idx,
            "ground_truth": np.array(im),
            "seeds": {},
        }

        for seed_idx, x_g_init in enumerate(sparse_inputs):
            print(f"  seed {seed_idx + 1}/{n_seeds}", flush=True)
            x_g = x_g_init
            seed_entry = {
                "sparse_input": np.array(x_g_init),
                "iterations": [],
            }

            for j in range(K):
                print(f"  iteration {j + 1}/{K} (S={S[j]} denoising steps)", flush=True)
                x0 = ddpm.sample(params=params, dims=x_g.shape, key=noise_key, x_g=x_g, t_start=S[j])
                x_g = x0
                seed_entry["iterations"].append(np.array(x0))

            seed_entry["final"] = seed_entry["iterations"][-1]
            sample_entry["seeds"][seed_idx] = seed_entry

        results["samples"].append(sample_entry)

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(BASE_DIR, "monitoring", "sparse_reconstructions")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"reconstructions_epoch_{ckpt_epoch:04d}.pkl")
    with open(save_path, "wb") as f:
        pkl.dump(results, f)
    print(f"Saved reconstructions to {save_path}")
        

if __name__ == "__main__":
    
    with open("configs/inference_config.yaml", "r") as f:
        inference_cfg = yaml.safe_load(f)
    
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    cfgs = [cfg, inference_cfg]

    run_inference(cfgs)