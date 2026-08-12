"""MEASURED (ground-truth) regime statistics for the oracle multi-regime fine-tune.

For each regime, the reward target is built from the TRUE fine-resolution spectra of that
regime's TRAINING sequences - exactly as the original Re=1000 fine-tune used measured statistics.
THIS IS AN ORACLE EXPERIMENT: it deliberately uses target-regime ground truth and is therefore
NOT DEPLOYABLE; it exists to separate 'can one model learn input-dependent dose?' from 'are the
extrapolated anchors accurate enough?'. Validation/test sequences are untouched.

Writes base_results/regime_stats_re{R}_measured_train.npz with the exact keys the Reward class
reads (spec_ref, log_spec_ref, enstrophy_ref, quantiles_ref, residual_ref).
"""
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax.numpy as jnp, jax
from anchor_obsfit_builder import fine_spec
from src.rewards import make_residual_loss
from src.sequence_inference import build_triplets, load_sequence

SIG = 4.7988
GEN = 'flow-data/generated/gen_fnons_re{}_kf_1024to256_20seq.npy'
REGIMES = {1000: ('flow-data/kf_2d_re1000_256_40seed.npy', list(range(20, 28)))}
for R in (1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000):
    REGIMES[R] = (GEN.format(R), list(range(0, 8)))

for R, (gt, seqs) in REGIMES.items():
    E = fine_spec(gt, seqs)                                   # true spectrum, training seqs only
    a = np.load(gt, mmap_mode='r')
    frames = np.concatenate([np.asarray(a[s, ::40], np.float32) / SIG for s in seqs[:4]])
    ens = float((frames ** 2).mean())
    q = np.quantile(frames.ravel(), np.linspace(0, 1, 257))
    resid_fn = jax.jit(make_residual_loss(n=256, re=float(R), std=SIG, mean=0.0))
    trip = build_triplets(load_sequence(gt, seqs[0]), 0.0, SIG)[::30][:8]
    r_ref = float(np.asarray(resid_fn(jnp.asarray(trip))).mean())
    np.savez(f'base_results/regime_stats_re{R}_measured_train.npz',
             spec_ref=E.astype(np.float32), log_spec_ref=np.log(E + 1e-30).astype(np.float32),
             enstrophy_ref=np.float64(ens), quantiles_ref=q.astype(np.float64),
             residual_ref=np.float32(r_ref))
    print(f"Re={R}: spec_ref[32:96]={E[32:96].sum():.3e}  resid_ref={r_ref:.1f}  "
          f"ens={ens:.3f}", flush=True)
print("MEASURED STATS COMPLETE", flush=True)
