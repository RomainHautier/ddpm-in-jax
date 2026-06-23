import jax
import jax.numpy as jnp
import numpy as np
import yaml

from src.models.model import build_diffusion
from src.train_ddpm import load_dataset
from src.utils import load_checkpoint, sparsify_input, lr_input, save_results_to_gcs


def run_inference(cfgs):
    """This function performs low-to-high resolution flow reconstruction
    using a pretrained U-Net backbone diffusion model. An high res input
    is sparsified, sparse pixels are filled using NN methods to obtain
    the low res flow and fed to the model to perform inference."""

    print(f"JAX devices: {jax.devices()}", flush=True)

    # Load the model
    ckpt_path = cfgs[1]["sparse_diffusion"]["checkpoint"]
    print(f"Loading checkpoint: {ckpt_path}", flush=True)
    ddpm = build_diffusion(cfgs[0])
    params, _, ckpt_epoch = load_checkpoint(ckpt_path)
    print(f"Loaded checkpoint epoch {ckpt_epoch}", flush=True)

    n_test_samples = cfgs[1]["sparse_diffusion"]["n_samples"]
    K = cfgs[1]["sparse_diffusion"]["K"]
    S = cfgs[1]["sparse_diffusion"]["S"]

    assert K == len(S), "K and S must have the same length in inference_config.yaml"

    _, _, test_ds, mean, std = load_dataset(cfgs[0], n_devices=1)
    test_ds = test_ds.shuffle(buffer_size=10000, seed=cfgs[1]["inference_seed"], reshuffle_each_iteration=False).take(n_test_samples)

    base_key = jax.random.key(cfgs[1]["inference_seed"])

    task = cfgs[1]["super_resolution"]["task"]

    if task == 1:
        task_meta = {"lr_res": cfgs[1]["super_resolution"]["lr_res"], "upsample_method": cfgs[1]["super_resolution"]["upsample_method"]}
    else:
        task_meta = {"sparsity_ratio": cfgs[1]["nn_fill"]["sparsity_ratio"], "seeds": cfgs[1]["nn_fill"]["seeds"]}

    results = {
        "metadata": {
            "task": task,
            "checkpoint": ckpt_path,
            "checkpoint_epoch": ckpt_epoch,
            "K": K,
            "S": S,
            "inference_seed": cfgs[1]["inference_seed"],
            "n_samples": n_test_samples,
            **task_meta,
        },
        "samples": [],
    }

    print(f"Running inference (task {task}): {n_test_samples} images, K={K}, S={S}", flush=True)

    for im_idx, batch in enumerate(test_ds):
        if im_idx >= n_test_samples:
            break
        print(f"\n[Image {im_idx + 1}/{n_test_samples}]", flush=True)
        im = jnp.array(batch).squeeze(0)
        
        if cfgs[1]["super_resolution"]["task"] == 1:
            sparse_inputs = lr_input(im, cfg=cfgs[1])
        else:
            sparse_inputs = sparsify_input(im, cfgs[1])

        sample_entry = {
            "image_idx": im_idx,
            "ground_truth": np.array(im),
            "seeds": {},
        }

        n_inputs = len(sparse_inputs)
        for seed_idx, x_g_init in enumerate(sparse_inputs):
            print(f"  input {seed_idx + 1}/{n_inputs}", flush=True)
            x_g = x_g_init
            seed_entry = {
                "sparse_input": np.array(x_g_init),
                "iterations": [],
            }
            
            for j in range(K):
                noise_key = jax.random.fold_in(base_key, im_idx * n_inputs * K + seed_idx * K + j)
                print(f"  iteration {j + 1}/{K} (S={S[j]} denoising steps)", flush=True)
                x0 = ddpm.sample(params=params, dims=x_g.shape, key=noise_key, x_g=x_g, t_start=S[j])
                x_g = x0
                seed_entry["iterations"].append(np.array(x0))

            seed_entry["final"] = seed_entry["iterations"][-1]
            sample_entry["seeds"][seed_idx] = seed_entry

        results["samples"].append(sample_entry)

    import os, pickle
    if task == 1:
        filename = f"reconstructions_task1_lr{cfgs[1]['super_resolution']['lr_res']}.pkl"
    else:
        filename = f"reconstructions_task2_sratio{cfgs[1]['nn_fill']['sparsity_ratio']}.pkl"
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "monitoring", "sparse_reconstructions")
    os.makedirs(save_dir, exist_ok=True)
    local_path = os.path.join(save_dir, filename)
    with open(local_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results locally to {local_path}", flush=True)

    save_results_to_gcs(results, filename)
        

if __name__ == "__main__":
    
    with open("configs/inference_config.yaml", "r") as f:
        inference_cfg = yaml.safe_load(f)
    
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    cfgs = [cfg, inference_cfg]

    run_inference(cfgs)
