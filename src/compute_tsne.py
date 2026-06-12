"""
Compute t-SNE embedding over val+test GT frames and reconstruction trajectories.
Saves a .pkl that the notebook loads — no need to re-run t-SNE every time.

Usage (from repo root):
    python monitoring/sparse_reconstructions/compute_tsne.py [--local-data PATH]

GPU path : cuML (RAPIDS) is used when available — install via
               conda install -c rapidsai -c conda-forge cuml cuda-version=12
CPU fallback: sklearn TSNE with PCA pre-reduction to 50 dims for speed.
"""

import argparse
import os
import pickle
import sys

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--local-data", default=None,
                    help="Path to local .npy data file; overrides config data_path")
parser.add_argument("--config",     default="configs/config.yaml")
parser.add_argument("--results-dir", default="monitoring/sparse_reconstructions")
parser.add_argument("--result-files", nargs="+",
                    default=["reconstructions_task1_lr32.pkl",
                             "reconstructions_task1_lr64.pkl"])
parser.add_argument("--out",  default="monitoring/sparse_reconstructions/tsne_embedding.pkl")
parser.add_argument("--perplexity", type=float, default=30)
parser.add_argument("--n-iter",     type=int,   default=1000)
parser.add_argument("--max-bg",     type=int,   default=300,
                    help="Max val/test frames to include as background (per split)")
parser.add_argument("--pca-dims",   type=int,   default=50,
                    help="PCA pre-reduction dims before t-SNE (0 = skip)")
parser.add_argument("--show-n",     type=int,   default=5,
                    help="Number of example reconstructions per config to embed")
parser.add_argument("--seed",       type=int,   default=42)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

with open(args.config) as f:
    cfg = yaml.safe_load(f)

data_cfg  = cfg["data"]
data_path = args.local_data or data_cfg["data_path"]

print(f"Loading data from {data_path} …", flush=True)
if data_path.startswith("gs://"):
    import subprocess
    local_cache = os.path.join("flow-data", os.path.basename(data_path))
    os.makedirs("flow-data", exist_ok=True)
    if not os.path.exists(local_cache):
        print(f"  downloading to {local_cache} …", flush=True)
        subprocess.run(["gcloud", "storage", "cp", data_path, local_cache], check=True)
    else:
        print(f"  using cached {local_cache}", flush=True)
    data = np.load(local_cache)
else:
    data = np.load(data_path)

print(f"  data shape: {data.shape}", flush=True)

n_train = data_cfg.get("n_train_seqs", 32)
n_val   = data_cfg.get("n_val_seqs",   4)
n_test  = data_cfg.get("n_test_seqs",  4)
val_idx  = np.arange(n_train, n_train + n_val)
test_idx = np.arange(n_train + n_val, n_train + n_val + n_test)

mean = float(np.mean(data[:n_train]))
std  = float(np.std(data[:n_train]))


def build_frames(data, idx_list):
    frames = []
    for idx in idx_list:
        for t in range(data.shape[1] - 2):
            f = np.stack([
                (data[idx, t]   - mean) / std,
                (data[idx, t+1] - mean) / std,
                (data[idx, t+2] - mean) / std,
            ], axis=-1)
            frames.append(f)
    return np.array(frames, dtype=np.float32)


val_frames  = build_frames(data, val_idx)
test_frames = build_frames(data, test_idx)
print(f"  val frames: {val_frames.shape},  test frames: {test_frames.shape}", flush=True)

# ---------------------------------------------------------------------------
# Load reconstruction results
# ---------------------------------------------------------------------------

all_results = {}
for fname in args.result_files:
    key = fname.replace("reconstructions_", "").replace(".pkl", "")
    fpath = os.path.join(args.results_dir, fname)
    with open(fpath, "rb") as f:
        all_results[key] = pickle.load(f)
    print(f"  loaded {key}", flush=True)

recon_vecs_flat = []
recon_meta = []  # (cfg_key, n_pts)

for cfg_key, res in all_results.items():
    for sample in res["samples"][:args.show_n]:
        for seed_entry in sample["seeds"].values():
            pts = ([seed_entry["sparse_input"].flatten()]
                   + [it.flatten() for it in seed_entry["iterations"]])
            recon_vecs_flat.extend(pts)
            recon_meta.append((cfg_key, len(pts)))

# ---------------------------------------------------------------------------
# Subsample background frames
# ---------------------------------------------------------------------------

rng = np.random.default_rng(args.seed)
val_sub  = val_frames.reshape(len(val_frames),  -1)[
    rng.choice(len(val_frames),  min(args.max_bg, len(val_frames)),  replace=False)]
test_sub = test_frames.reshape(len(test_frames), -1)[
    rng.choice(len(test_frames), min(args.max_bg, len(test_frames)), replace=False)]

all_vecs   = np.concatenate([val_sub, test_sub, np.array(recon_vecs_flat)])
n_val_sub  = len(val_sub)
n_test_sub = len(test_sub)

print(f"\nTotal points for t-SNE: {len(all_vecs)} "
      f"({n_val_sub} val, {n_test_sub} test, {len(recon_vecs_flat)} recon)", flush=True)
print(f"Feature dim: {all_vecs.shape[1]}", flush=True)

# ---------------------------------------------------------------------------
# PCA pre-reduction
# ---------------------------------------------------------------------------

if args.pca_dims > 0 and args.pca_dims < all_vecs.shape[1]:
    print(f"\nPCA pre-reduction: {all_vecs.shape[1]} → {args.pca_dims} dims …", flush=True)
    try:
        from cuml.decomposition import PCA as cuPCA
        pca = cuPCA(n_components=args.pca_dims, random_state=args.seed)
        reduced = np.array(pca.fit_transform(all_vecs))
        print("  using cuML PCA (GPU)", flush=True)
    except ImportError:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=args.pca_dims, random_state=args.seed)
        reduced = pca.fit_transform(all_vecs)
        print("  using sklearn PCA (CPU)", flush=True)
    print(f"  explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%", flush=True)
else:
    reduced = all_vecs
    print("Skipping PCA pre-reduction.", flush=True)

# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------

print(f"\nFitting t-SNE (perplexity={args.perplexity}, n_iter={args.n_iter}) …", flush=True)
try:
    from cuml.manifold import TSNE as cuTSNE
    tsne = cuTSNE(n_components=2, perplexity=args.perplexity,
                  n_iter=args.n_iter, random_state=args.seed, verbose=True)
    embedded = np.array(tsne.fit_transform(reduced))
    print("  used cuML TSNE (GPU)", flush=True)
except ImportError:
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=args.perplexity,
                max_iter=args.n_iter, random_state=args.seed,
                init="pca", verbose=1)
    embedded = tsne.fit_transform(reduced)
    print("  used sklearn TSNE (CPU)", flush=True)

print("Done.", flush=True)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

result = {
    "embedded":     embedded,
    "n_val_sub":    n_val_sub,
    "n_test_sub":   n_test_sub,
    "recon_meta":   recon_meta,
    "params": {
        "perplexity": args.perplexity,
        "n_iter":     args.n_iter,
        "pca_dims":   args.pca_dims,
        "max_bg":     args.max_bg,
        "seed":       args.seed,
    },
}

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "wb") as f:
    pickle.dump(result, f)
print(f"\nSaved embedding to {args.out}", flush=True)
