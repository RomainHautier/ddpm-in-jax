"""Run the low-res (sparse + NN-fill) sequence inference on the 2 simulated jax-cfd
sequences, to test whether the simulated flow is in-distribution for the model.

Mirrors the test-set task exactly:
  - degraded INPUT = sparse-sample each clean simulated frame at the npz's 1024-point
    masks (idx_lst) + nearest-neighbour fill (verified to reproduce kmflow_sampled_data_irregnew),
  - GROUND TRUTH = the clean simulated flow,
  - same checkpoint, K/S schedule, and normalization (std=4.7988) as the test-set run.

Outputs: monitoring/sequence_reconstructions/sequence_reconstruction_sim_seq{0,1}.pkl
Run with the JAX/TPU venv:  source ~/venv-ddpm/bin/activate && python -m src.run_sim_inference
"""
import os

import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt

from src.sequence_inference import run_sequence_inference

SIM_CLEAN = "data_generation/jaxcfd_re1000_1024to256_2seq.npy"
SIM_INPUT = "data_generation/sim_lowres_input.npy"
NPZ = "flow-data/kmflow_sampled_data_irregnew.npz"


def build_sparse_nnfill_input():
    """Sparse-sample (1024 pts, npz masks) + NN-fill each clean simulated frame."""
    clean = np.load(SIM_CLEAN)  # (n_seq, n_t, H, W), physical units
    idx_lst = np.load(NPZ)["idx_lst"]  # (40, 1024) sampling masks
    n_seq, n_t, H, W = clean.shape
    out = np.empty_like(clean)
    for s in range(n_seq):
        mask = np.zeros(H * W, bool)
        mask[idx_lst[s]] = True
        mask = mask.reshape(H, W)
        _, ind = distance_transform_edt(~mask, return_indices=True)  # same mask all frames
        for t in range(n_t):
            out[s, t] = clean[s, t][ind[0], ind[1]]
        print(f"  built degraded input seq {s} ({mask.sum()} sampled pts)", flush=True)
    np.save(SIM_INPUT, out.astype(np.float32))
    print(f"saved {SIM_INPUT} {out.shape}", flush=True)


def main():
    if not os.path.exists(SIM_INPUT):
        print("Building sparse + NN-fill degraded input from the simulated clean flow ...", flush=True)
        build_sparse_nnfill_input()
    else:
        print(f"Using existing {SIM_INPUT}", flush=True)

    with open("configs/inference_config.yaml") as f:
        inf = yaml.safe_load(f)
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    cfg["data"]["data_path"] = SIM_INPUT            # model input = degraded simulated
    sd = inf["sequence_diffusion"]
    sd["gt_data_path"] = SIM_CLEAN                  # ground truth = clean simulated
    sd["seq_idxs"] = [0, 1]                         # the 2 simulated sequences
    sd["out_tag"] = "sim_"                          # -> sequence_reconstruction_sim_seq{0,1}.pkl
    run_sequence_inference([cfg, inf])


if __name__ == "__main__":
    main()
