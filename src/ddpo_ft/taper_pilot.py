"""TAPERED BAND EDGE PILOT (user 2026-08-23): the k=96 cliff in every ratio plot is partly
the hard edge of the dose band (full pull at k=95, nothing at 96). Test a tapered second
term: weight 1 in [32,96), smooth cosine roll-off to 0 across [96,120) — pulls gently into
the distrusted near-Nyquist zone instead of stopping at a wall. re2k-149 @ K3, Re 5000/8000,
ls in {8,16}, hard vs tapered. Keys '{R}|re2k-149|TP{edge}ls{v}' in steering_pilot.npz.
"""
import os, sys, pickle
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp
from functools import partial
from diag_guided_residual import make_kchain_ddim_sampler
from ppo_claude import build_ddim_denoiser
from train_claude import build_base_ddpm
from viz_energy import local_hik_energy
from src.rewards import make_spectrum_fn, make_residual_loss, make_spectrum_distance
from src.physics_guidance import make_dx_func
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence
from eval_ddpo import eff_resolution
from psample import pbatched

MEAN, SIG, N, HIK0 = 0.0, 4.7988, 256, 32
STARTS, STEPS = [150, 100, 50], 86
GEN = 'flow-data/generated/gen_fnons_re{}_kf_1024to256_20seq.npy'
P = pickle.load(open('monitoring/ddpo_re2000_newpool_ckpts/ddpo_re1000_iter0149.pkl', 'rb'))['params']
ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
sa3, s13 = float(jnp.sqrt(ab[STARTS[0]])), float(jnp.sqrt(1.0 - ab[STARTS[0]]))
B16 = partial(pbatched, per_dev=16)

OUT = {}
OUTP = 'base_results/steering_pilot.npz'
if os.path.exists(OUTP):
    old = np.load(OUTP, allow_pickle=True); OUT = {k: old[k] for k in old.files}

for R in (5000, 8000):
    d = np.load(f'base_results/regime_stats_re{R}_measured_train.npz')
    stats = {k: d[k] for k in d.files}
    lref = jnp.asarray(stats['log_spec_ref'])
    dx_pde = make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN)
    d1 = make_spectrum_distance(stats['spec_ref'], kband=(1, 96), n=N,
                                log_ref=stats.get('log_spec_ref'))

    # hard-edge second term (the production dial) and the tapered variant
    def make_hi(weights_hi):
        w = jnp.asarray(weights_hi)          # over shells [32, 120)
        def dist(x):
            lE = jnp.log(spec_fn(x)[:, 32:120] + 1e-12)
            return (w * (lE - lref[32:120]) ** 2).mean(axis=1).sum()
        return jax.grad(dist)
    w_hard = np.zeros(88); w_hard[:64] = 1.0                       # [32,96) only
    w_tap = np.ones(88); w_tap[64:] = 0.5 * (1 + np.cos(np.pi * np.arange(24) / 24))
    D2 = {'hard': make_hi(w_hard), 'taper': make_hi(w_tap)}

    xg, xl = [], []
    for sq in range(8, 20):
        q = load_sequence(GEN.format(R), sq)
        g = build_triplets(q, MEAN, SIG)
        l = build_triplets(grid_downsample_degrade(q, 4), MEAN, SIG)
        i2 = np.linspace(0, len(g) - 1, 10).astype(int)
        xg.append(g[i2]); xl.append(l[i2])
    xg, xl = np.concatenate(xg), np.concatenate(xl)
    resid_fn = jax.jit(make_residual_loss(n=N, re=float(R), std=SIG, mean=0.0))
    E_gt = np.asarray(spec_fn(jnp.asarray(xg))).mean(0)
    Ehg = local_hik_energy(xg[..., 1] * SIG, HIK0, 6.0)
    rg = float(np.concatenate([np.asarray(resid_fn(jnp.asarray(xg[i:i + 32]))).ravel()
                               for i in range(0, len(xg), 32)]).mean())
    recon = B16(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape)), xl, 500)
    print(f"\n=== Re={R}: taper pilot ===", flush=True)
    for edge, dhi in D2.items():
        for ls in (8.0, 16.0):
            key = f'{R}|re2k-149|TP{edge}ls{ls:g}'
            if f'{key}||ret' in OUT:
                continue
            def dx(x, _ls=ls, _dhi=dhi):
                g1 = jax.grad(lambda y: jnp.sum(d1(y)))(x)
                return dx_pde(x) + (_ls / 3.0) * (0.5 * g1 + 3.0 * _dhi(x))
            smp = make_kchain_ddim_sampler(ddpm.unet, ab, STARTS, STEPS, dx, 3.0, temp=0.30)
            y = B16(lambda xb, kk: smp(P, sa3 * xb + s13 * jax.random.normal(
                jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), recon, 700)
            E = np.asarray(spec_fn(jnp.asarray(y))).mean(0)
            ry = float(np.concatenate([np.asarray(resid_fn(jnp.asarray(y[i:i + 32]))).ravel()
                                       for i in range(0, len(y), 32)]).mean())
            Eh = local_hik_energy(y[..., 1] * SIG, HIK0, 6.0)
            vals = dict(ret=E[HIK0:96].sum() / E_gt[HIK0:96].sum(),
                        tail=E[96:120].sum() / E_gt[96:120].sum(),
                        place=np.corrcoef(Eh.ravel(), Ehg.ravel())[0, 1],
                        lowk=E[1:5].sum() / E_gt[1:5].sum(), kstar=eff_resolution(E, E_gt),
                        resid_ratio=ry / rg)
            for f, vv in vals.items():
                OUT[f'{key}||{f}'] = np.float32(vv)
            OUT[f'{key}||E'] = E.astype(np.float32)
            np.savez(OUTP, **OUT)
            print(f"  {edge:<6} ls={ls:g}  ret={vals['ret']:.3f} TAIL[96,120)={vals['tail']:.3f} "
                  f"place={vals['place']:.3f} k*={vals['kstar']:.0f} resid={vals['resid_ratio']:.2f}",
                  flush=True)
    jax.clear_caches()
print("\nTAPER PILOT COMPLETE", flush=True)
