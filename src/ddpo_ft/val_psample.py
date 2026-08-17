"""VALIDATION GATE for the data-parallel sampler: regrade ONE already-graded cell with
pbatched and require agreement within the eval-noise floor. Exit 0 = safe to use PSAMPLE=1
for the unattended chain; exit 1 = fall back to serial.

Cell: Re=2000 | mr (unconditioned multi-regime, final ckpt) @ K2x50 on the standard val pool
(seqs 8-19, per=10) — stored ret came from the serial helper; tolerance 0.04 covers the
documented eval-seed noise floor (~±0.015) plus the resharded noise draws.
"""
import os, sys, glob, pickle
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp
from diag_guided_residual import make_kchain_ddim_sampler
from ppo_claude import build_ddim_denoiser
from train_claude import build_base_ddpm
from src.rewards import make_spectrum_fn
from src.physics_guidance import make_dx_func
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence
from psample import pbatched

MEAN, SIG, N, HIK0 = 0.0, 4.7988, 256, 32
R, TOL = 2000, 0.04
stored = np.load('base_results/multiregime_grade.npz', allow_pickle=True)
ref = float(stored[f'{R}|mr|K2x50||ret'])
print(f"stored serial ret for {R}|mr|K2x50: {ref:.4f}  (tolerance ±{TOL})", flush=True)

mr = pickle.load(open(sorted(glob.glob(
    'monitoring/ddpo_multiregime_gt_ckpts/ddpo_multi_iter*.pkl'))[-1], 'rb'))['params']
ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))

xg, xl = [], []
for s in range(8, 20):
    q = load_sequence('flow-data/generated/gen_fnons_re2000_kf_1024to256_20seq.npy', s)
    g = build_triplets(q, MEAN, SIG)
    l = build_triplets(grid_downsample_degrade(q, 4), MEAN, SIG)
    i2 = np.linspace(0, len(g) - 1, 10).astype(int)
    xg.append(g[i2]); xl.append(l[i2])
xg, xl = np.concatenate(xg), np.concatenate(xl)
E_gt = np.asarray(spec_fn(jnp.asarray(xg))).mean(0)

recon = pbatched(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
    jax.random.fold_in(kk, 1), xb.shape)), xl, 500)
dx = make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN)
smp = make_kchain_ddim_sampler(ddpm.unet, ab, [100, 75], 50, dx, 3.0, temp=0.30)
sa, s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
y = pbatched(lambda xb, kk: smp(mr, sa * xb + s1 * jax.random.normal(
    jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), recon, 700)
E = np.asarray(spec_fn(jnp.asarray(y))).mean(0)
ret = float(E[HIK0:96].sum() / E_gt[HIK0:96].sum())
diff = abs(ret - ref)
print(f"pbatched ret: {ret:.4f}   |diff| = {diff:.4f}", flush=True)
if diff > TOL:
    print("PSAMPLE VALIDATION FAILED — use serial", flush=True)
    sys.exit(1)
print("PSAMPLE VALIDATION OK", flush=True)
