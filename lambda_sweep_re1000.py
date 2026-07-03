"""Lambda sweep with FULL-SEQUENCE inference + per-step residual / MSE / energy-spectrum logging.

Env-parameterized so the same script runs the generated Re=1000 case and the OOD Re cases:
  SWEEP_DATASET = generated | real        (default generated)
  SWEEP_RE      = residual/data Reynolds number (default 1000; set 500/2000 for OOD)
  SWEEP_GEN_GT  = path to the generated GT .npy (default the Re=1000 regen)
  SWEEP_LAMBDAS = comma list (default "0,0.2,0.8")

Stores MIDDLE-CHANNEL reconstructions for the FULL sequences (disk-friendly), plus GT/input middle
channels and the per-step trajectories (chunk 0). All metrics (spectra, retention, MSE, PDF, energy
balance) are computed in base_results/lambda_sweep_metrics.ipynb. Same inference seed across lambda.
"""
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from jax.sharding import NamedSharding, PartitionSpec as P

from src.models.model import DDPM
from src.utils import load_checkpoint
from src.sequence_inference import (build_triplets, make_batched_sampler,
                                    load_sequence, sparse_nnfill_degrade)
from src.physics_guidance import make_dx_func, make_residual_loss

# ----------------------------- configuration (env-overridable) -----------------------------
DATASET = os.environ.get("SWEEP_DATASET", "generated")              # generated (our regen) | real (u3232)
RE = float(os.environ.get("SWEEP_RE", "1000"))                      # residual Re = the data's Re
GEN_GT_NPY = os.environ.get("SWEEP_GEN_GT", "flow-data/kf_re1000_256_40seed_regen.npy")
LAMBDAS = [float(x) for x in os.environ.get("SWEEP_LAMBDAS", "0,0.2,0.8").split(",")]
SEQ_IDXS = [int(x) for x in os.environ.get("SWEEP_SEQS", "36,37,38,39").split(",")]  # OOD: 20-seed -> use 16-19
MAX_FRAMES = None                      # None = FULL sequences (all ~318 triplets)
K, S = 3, [150, 100, 50]               # SDEdit K-iteration schedule
MEAN, STD = 0.0, 4.7988                # model normalization
B = 16
CKPT = "gs://ddpm-thesis-rh/checkpoints/ddpm/ckpt_epoch_0299.pkl"
SEED = 42
REAL_INPUT_NPZ = "flow-data/kmflow_sampled_data_irregnew.npz"      # u3232 sparse + idx_lst
TAG = "real_re1000" if DATASET == "real" else f"gen_re{int(RE)}"
OUTDIR = f"base_results/lambda_sweep_{TAG}"
os.makedirs(OUTDIR, exist_ok=True)
print(f"DATASET={DATASET} RE={RE:g} LAMBDAS={LAMBDAS} full_seq -> {OUTDIR}", flush=True)

cfg = yaml.safe_load(open("configs/config.yaml"))
ddpm = DDPM(cfg)
params, _, epoch = load_checkpoint(CKPT)
print(f"checkpoint epoch {epoch} | devices={jax.device_count()}", flush=True)
mesh = jax.make_mesh((jax.device_count(),), ("data",))
data_sharding = NamedSharding(mesh, P("data"))
params = jax.device_put(params, NamedSharding(mesh, P()))
base_key = jax.random.key(SEED)

# ----------------------------- data: input + GT triplets (lambda-independent) -----------------------------
print(f"loading {DATASET} inputs + GT (full sequences) ...", flush=True)
inputs, gt_tris = {}, {}
if DATASET == "real":
    _u = np.load(REAL_INPUT_NPZ)["u3232"]
    for s in SEQ_IDXS:
        inputs[s] = build_triplets(np.asarray(_u[s], np.float32), MEAN, STD)[:MAX_FRAMES]
    del _u
    _graw = np.load(f"{OUTDIR}/gt_full.npy")            # full real GT frames (must be pre-extracted)
    for i, s in enumerate(SEQ_IDXS):
        gt_tris[s] = build_triplets(_graw[i], MEAN, STD)[:MAX_FRAMES]
else:  # generated: GT = regen npy; input = sparse 1024-pt + NN-fill (idx_lst from kmflow npz)
    for s in SEQ_IDXS:
        _gts = load_sequence(GEN_GT_NPY, s)             # (320,256,256) full generated GT (mmap)
        inputs[s] = build_triplets(sparse_nnfill_degrade(_gts, s, npz_path="flow-data/kmflow_idx_lst.npz"), MEAN, STD)[:MAX_FRAMES]
        gt_tris[s] = build_triplets(_gts, MEAN, STD)[:MAX_FRAMES]

# Save GT and input middle-channels (full sequences) for the notebook (lambda-independent).
np.save(f"{OUTDIR}/gt_mid.npy", np.stack([np.asarray(gt_tris[s][..., 1], np.float32) for s in SEQ_IDXS]))
np.save(f"{OUTDIR}/inp_mid.npy", np.stack([np.asarray(inputs[s][..., 1], np.float32) for s in SEQ_IDXS]))
print(f"  triplets/seq = {gt_tris[SEQ_IDXS[0]].shape[0]}", flush=True)

# ----------------------------- per-sample probes (residual at the data's Re) -----------------------------
residual_fn = make_residual_loss(re=RE, std=STD, mean=MEAN)
_kf = jnp.fft.fftfreq(256, 1.0 / 256)
_KR = jnp.round(jnp.sqrt(_kf[:, None] ** 2 + _kf[None, :] ** 2)).astype(jnp.int32).ravel()
KMAX = 128


@jax.jit
def spectrum_fn(x):
    Pw = jnp.abs(jnp.fft.fft2(x[..., 1], axes=(-2, -1))) ** 2
    Pw = Pw.reshape(Pw.shape[0], -1)
    E = jax.vmap(lambda p: jax.ops.segment_sum(p, _KR, num_segments=256))(Pw)
    return E[:, :KMAX]


@jax.jit
def mse_fn(x, gt):
    return jnp.mean((x - gt) ** 2, axis=(1, 2, 3))


def reconstruct(sampler, inp_tri, gt_tri, seq_idx):
    """Full-sequence SDEdit reconstruction. Returns (recon[n,256,256,3], records); records logged
    over the FIRST chunk only: [(stage_j, t, resid[B], mse[B], spec[B,KMAX])]."""
    n = inp_tri.shape[0]
    pad = (-n) % B
    arr = np.concatenate([inp_tri, np.zeros((pad,) + inp_tri.shape[1:], np.float32)], 0) if pad else inp_tri
    gt0 = jax.device_put(jnp.asarray(gt_tri[:B]), data_sharding)

    def probe(xT):
        return (np.asarray(residual_fn(xT)), np.asarray(mse_fn(xT, gt0)), np.asarray(spectrum_fn(xT)))

    outs, records = [], []
    for ci in range(arr.shape[0] // B):
        x = jax.device_put(jnp.asarray(arr[ci * B:(ci + 1) * B]), data_sharding)
        for j in range(K):
            key = jax.random.fold_in(base_key, seq_idx * 100000 + ci * K + j)
            out = sampler(params, x, key, S[j], probe=(probe if ci == 0 else None))
            x, log = out if ci == 0 else (out, None)
            if ci == 0:
                records.extend((j, int(t), m[0], m[1], m[2]) for (t, m) in log)
        outs.append(np.asarray(x, np.float32))
    return np.concatenate(outs, 0)[:n], records


json.dump({"dataset": DATASET, "re": RE, "lambdas": LAMBDAS, "seq_idxs": SEQ_IDXS,
           "max_frames": MAX_FRAMES, "K": K, "S": S, "mean": MEAN, "std": STD, "seed": SEED,
           "gen_gt": GEN_GT_NPY, "checkpoint_epoch": int(epoch)},
          open(f"{OUTDIR}/meta.json", "w"), indent=2)

for lam in LAMBDAS:
    dx_func = make_dx_func(re=RE, std=STD, mean=MEAN, lam=lam) if lam > 0 else None
    sampler = make_batched_sampler(ddpm, data_sharding, dx_func, None, log_every=10)
    print(f"\n===== lambda={lam:g} ({'guided' if dx_func else 'baseline'}) =====", flush=True)
    seqs_r, stages_r, ts_r, resid_r, mse_r, spec_r = [], [], [], [], [], []
    for s in SEQ_IDXS:
        t0 = time.time()
        recon, records = reconstruct(sampler, inputs[s], gt_tris[s], s)
        np.save(f"{OUTDIR}/recon_mid_lam{lam:g}_seq{s}.npy", recon[..., 1].astype(np.float32))
        for (j, t, resid, mse, spec) in records:
            seqs_r.append(s); stages_r.append(j); ts_r.append(t)
            resid_r.append(resid); mse_r.append(mse); spec_r.append(spec)
        print(f"  lam={lam:g} seq{s}: {recon.shape[0]} frames finite={bool(np.isfinite(recon).all())} "
              f"mean|x|={np.abs(recon).mean():.3f}  [{time.time()-t0:.0f}s]", flush=True)
    np.savez(f"{OUTDIR}/traj_lam{lam:g}.npz",
             seqs=np.array(seqs_r), stages=np.array(stages_r), ts=np.array(ts_r),
             resid=np.stack(resid_r), mse=np.stack(mse_r), spec=np.stack(spec_r), S=np.array(S))
    print(f"  saved trajectory -> traj_lam{lam:g}.npz", flush=True)

print("\nSWEEP DONE", flush=True)
