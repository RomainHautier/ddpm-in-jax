"""N=1 ANCHOR (user 2026-08-23): the empirical arm of the anchor-budget study — can a SINGLE
high-res triplet (3 consecutive frames) provide a good-enough training target? Builds
base_results/regime_stats_re2000_n1.npz with the exact schema/conventions of the measured
anchors (fine_spec's shell binning, /SIG normalization), from ONE triplet of train seq 0.
Control: rs2k-549 (full measured anchor, same recipe).
"""
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp
from anchor_obsfit_builder import _KR, N, KMAX
from src.rewards import make_residual_loss
from src.sequence_inference import build_triplets, load_sequence

SIG = 4.7988
GT = 'flow-data/generated/gen_fnons_re2000_kf_1024to256_20seq.npy'
T0 = 150                                     # mid-sequence, one triplet: frames 150,151,152
a = np.load(GT, mmap_mode='r')
frames = np.asarray(a[0, T0:T0 + 3], np.float32) / SIG
S = [np.bincount(_KR, (np.abs(np.fft.fft2(f)) ** 2).ravel(), minlength=N)[:KMAX]
     for f in frames]
E = np.asarray(S).mean(0)
ens = float((frames ** 2).mean())
q = np.quantile(frames.ravel(), np.linspace(0, 1, 257))
resid_fn = jax.jit(make_residual_loss(n=256, re=2000.0, std=SIG, mean=0.0))
trip = build_triplets(load_sequence(GT, 0), 0.0, SIG)[T0:T0 + 1]
r_ref = float(np.asarray(resid_fn(jnp.asarray(trip))).mean())
np.savez('base_results/regime_stats_re2000_n1.npz',
         spec_ref=E.astype(np.float32), log_spec_ref=np.log(E + 1e-30).astype(np.float32),
         enstrophy_ref=np.float64(ens), quantiles_ref=q.astype(np.float64),
         residual_ref=np.float32(r_ref))
ref = np.load('base_results/regime_stats_re2000_measured_train.npz')['spec_ref']
print(f"n1 anchor written; [32,96) vs full anchor: {E[32:96].sum()/ref[32:96].sum():.3f}, "
      f"[1,5): {E[1:5].sum()/ref[1:5].sum():.3f}", flush=True)
