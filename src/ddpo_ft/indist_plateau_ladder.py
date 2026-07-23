"""R3.1' calibration source — the IN-DIST temperature ladder (Re=1000, GT legitimate).

For each archived Re=1000 run (temps 1.5 / 1.75 / 2.0 / 2.5, same hk10 EMA recipe):
  (a) BLIND plateau: deployed[10,96)/anchor[10,96) on TRAIN-POOL inputs (seqs 32-33) — the same
      GT-free quantity R3.2 uses, i.e. what a deployment can always measure;
  (b) GT retention E[32,96)/GT on in-dist eval picks (seqs 37-39; probe 36 excluded), deep cascade
      + lam3 at itemp 0.30 (frozen §5 operating point).
If the in-dist mapping plateau->GT brackets a threshold below which GT quality leaves the band,
that threshold is derived WITHOUT touching any OOD regime — the three OOD runs (0.85/1.166,
0.79/1.183, 0.60/0.498) then become out-of-sample validation of the escalation rule instead of
its calibration. Caveat (documented): the in-dist anchor is measured (≈GT) while OOD anchors are
obs-fit extrapolations with tail bias; plateau comparisons inherit that difference.
"""
import os, sys, pickle
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp
from diag_guided_residual import make_kchain_ddim_sampler
from ppo_claude import build_ddim_denoiser
from train_claude import build_base_ddpm
from src.rewards import make_spectrum_fn
from src.physics_guidance import make_dx_func
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence
from eval_ddpo import eff_resolution

MEAN, SIG, N = 0.0, 4.7988, 256
GT_PATH = 'flow-data/kf_2d_re1000_256_40seed.npy'
ANCHOR = np.load('base_results/regime_stats_re1000.npz')['spec_ref']
RUNS = [
    (1.50, 'monitoring/ddpo_re1000_k2_s100-75_ddim50_ddiminit_hk10_emabase_ckpts/ddpo_re1000_iter0299.pkl'),
    (1.75, 'monitoring/ddpo_re1000_temp175_ckpts/ddpo_re1000_iter0499.pkl'),
    (2.00, 'monitoring/ddpo_re1000_temp20_ckpts/ddpo_re1000_iter0499.pkl'),
    (2.50, 'monitoring/ddpo_re1000_hk10_emabase_temp25_ckpts/ddpo_re1000_iter0499.pkl'),
]
ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
dx = make_dx_func(n=N, re=1000.0, std=SIG, mean=MEAN)
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
sat, s1t = float(jnp.sqrt(ab[150])), float(jnp.sqrt(1.0 - ab[150]))
deep = make_kchain_ddim_sampler(ddpm.unet, ab, [150, 100, 50], 86, dx, 3.0, temp=0.30)

def batched(fn, xin, seed, bs=8):
    k = jax.random.PRNGKey(seed); o = []
    for i in range(0, len(xin), bs):
        o.append(np.asarray(fn(jnp.asarray(xin[i:i+bs]), jax.random.fold_in(k, i))))
    return np.concatenate(o)

def pool(seqs, n_per):
    xg, xl = [], []
    for s in seqs:
        seq = load_sequence(GT_PATH, s)
        g = build_triplets(seq, MEAN, SIG); l = build_triplets(grid_downsample_degrade(seq, 4), MEAN, SIG)
        i2 = np.linspace(0, len(g) - 1, n_per).astype(int); xg.append(g[i2]); xl.append(l[i2])
    return np.concatenate(xg), np.concatenate(xl)

_, xl_tr = pool([32, 33], 6)
recon_tr = batched(lambda xb, kk: ddim20(base_params, _sa*xb + _s1*jax.random.normal(jax.random.fold_in(kk, 1), xb.shape)), xl_tr, 500)
xg_te, xl_te = pool([37, 38, 39], 6)
recon_te = batched(lambda xb, kk: ddim20(base_params, _sa*xb + _s1*jax.random.normal(jax.random.fold_in(kk, 1), xb.shape)), xl_te, 900)
E_gt = np.asarray(spec_fn(jnp.asarray(xg_te))).mean(0)

print("IN-DIST LADDER (Re=1000): blind plateau vs GT retention, frozen deep-cascade itemp 0.30", flush=True)
print(f"{'temp':>5} | {'blind deployed/anchor [10,96)':>30} | {'GT ret [32,96)':>14} | {'k*':>4}", flush=True)
for temp, ck in RUNS:
    P = pickle.load(open(ck, 'rb'))['params']
    y_tr = batched(lambda xb, kk: deep(P, sat*xb + s1t*jax.random.normal(jax.random.fold_in(kk, 1), xb.shape),
                                       jax.random.fold_in(kk, 2)), recon_tr, 700)
    E_tr = np.asarray(spec_fn(jnp.asarray(y_tr))).mean(0)
    plateau = float(E_tr[10:96].sum() / ANCHOR[10:96].sum())
    y_te = batched(lambda xb, kk: deep(P, sat*xb + s1t*jax.random.normal(jax.random.fold_in(kk, 1), xb.shape),
                                       jax.random.fold_in(kk, 2)), recon_te, 900)
    E_te = np.asarray(spec_fn(jnp.asarray(y_te))).mean(0)
    ret = float(E_te[32:96].sum() / E_gt[32:96].sum())
    print(f"{temp:5.2f} | {plateau:30.3f} | {ret:14.3f} | {eff_resolution(E_te, E_gt):>4}", flush=True)
print("done", flush=True)
