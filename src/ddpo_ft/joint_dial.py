"""JOINT-DIAL GRID (user 2026-08-19): do the spectral and residual gradients interact, or
compose independently? Small factorial: ls x lp on re2k-149 at Re=5000/8000, K3x86.
Existing single-dial cells cover the axes; this fills the corners:
(ls,lp) in {(2,0),(2,12),(8,0),(8,12)}. Keys '{R}|{nm}|ls{a}lp{b}' in steering_pilot.npz.
Read with the axes cells: lp swept at ls=0 ('lp{b}'), ls swept at lp=3 ('ls{a}').
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
COMBOS = [(2.0, 0.0), (2.0, 12.0), (8.0, 0.0), (8.0, 12.0)]
GEN = 'flow-data/generated/gen_fnons_re{}_kf_1024to256_20seq.npy'
REGIMES = {5000: dict(gt=GEN.format(5000), seqs=list(range(8, 20)), per=10,
                      anchor='base_results/regime_stats_re5000_obsfit_gen.npz'),
           8000: dict(gt=GEN.format(8000), seqs=list(range(8, 20)), per=10,
                      anchor='base_results/regime_stats_re8000_obsfit_gen.npz')}
NM = 're2k-149'
P = pickle.load(open('monitoring/ddpo_re2000_newpool_ckpts/ddpo_re1000_iter0149.pkl', 'rb'))['params']
ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
sa3, s13 = float(jnp.sqrt(ab[STARTS[0]])), float(jnp.sqrt(1.0 - ab[STARTS[0]]))
B16 = partial(pbatched, per_dev=16)


def make_anchor_dose_dx(stats):
    lref = stats.get('log_spec_ref')
    d1 = make_spectrum_distance(stats['spec_ref'], kband=(1, 96), n=N, log_ref=lref)
    d2 = make_spectrum_distance(stats['spec_ref'], kband=(32, 96), n=N, log_ref=lref)
    def loss(x):
        return jnp.sum(0.5 * d1(x) + 3.0 * d2(x))
    return jax.jit(jax.grad(loss))


OUT = {}
OUTP = 'base_results/steering_pilot.npz'
if os.path.exists(OUTP):
    old = np.load(OUTP, allow_pickle=True); OUT = {k: old[k] for k in old.files}

for R, c in REGIMES.items():
    d = np.load(c['anchor'])
    stats = {k: d[k] for k in d.files}
    dx_pde = make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN)
    dx_anchor = make_anchor_dose_dx(stats)
    xg, xl = [], []
    for s in c['seqs']:
        q = load_sequence(c['gt'], s)
        g = build_triplets(q, MEAN, SIG)
        l = build_triplets(grid_downsample_degrade(q, 4), MEAN, SIG)
        i2 = np.linspace(0, len(g) - 1, c['per']).astype(int)
        xg.append(g[i2]); xl.append(l[i2])
    xg, xl = np.concatenate(xg), np.concatenate(xl)
    resid_fn = jax.jit(make_residual_loss(n=N, re=float(R), std=SIG, mean=0.0))
    E_gt = np.asarray(spec_fn(jnp.asarray(xg))).mean(0)
    Ehg = local_hik_energy(xg[..., 1] * SIG, HIK0, 6.0)
    rg = float(np.concatenate([np.asarray(resid_fn(jnp.asarray(xg[i:i + 32]))).ravel()
                               for i in range(0, len(xg), 32)]).mean())
    recon = B16(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape)), xl, 500)
    print(f"\n=== Re={R}: joint dial grid ===", flush=True)
    for ls, lp in COMBOS:
        key = f'{R}|{NM}|ls{ls:g}lp{lp:g}'
        if f'{key}||ret' in OUT:
            continue
        # total guidance applied as lam=3 * dx_comb: 3*dx_comb = lp*dx_pde + ls*dx_anchor
        dx_comb = jax.jit(lambda x, _ls=ls, _lp=lp: (_lp / 3.0) * dx_pde(x)
                          + (_ls / 3.0) * dx_anchor(x))
        smp = make_kchain_ddim_sampler(ddpm.unet, ab, STARTS, STEPS, dx_comb, 3.0, temp=0.30)
        y = B16(lambda xb, kk: smp(P, sa3 * xb + s13 * jax.random.normal(
            jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), recon, 700)
        E = np.asarray(spec_fn(jnp.asarray(y))).mean(0)
        ry = float(np.concatenate([np.asarray(resid_fn(jnp.asarray(y[i:i + 32]))).ravel()
                                   for i in range(0, len(y), 32)]).mean())
        Eh = local_hik_energy(y[..., 1] * SIG, HIK0, 6.0)
        vals = dict(ret=E[HIK0:96].sum() / E_gt[HIK0:96].sum(),
                    place=np.corrcoef(Eh.ravel(), Ehg.ravel())[0, 1],
                    lowk=E[1:5].sum() / E_gt[1:5].sum(), kstar=eff_resolution(E, E_gt),
                    resid_ratio=ry / rg,
                    mse=np.mean((y[..., 1] - xg[..., 1]) ** 2) * SIG ** 2)
        for f, vv in vals.items():
            OUT[f'{key}||{f}'] = np.float32(vv)
        OUT[f'{key}||E'] = E.astype(np.float32)
        np.savez(OUTP, **OUT)
        print(f"  ls={ls:g} lp={lp:g}  ret={vals['ret']:.3f} place={vals['place']:.3f} "
              f"resid={vals['resid_ratio']:.2f}xGT mse={vals['mse']:.2f}", flush=True)
print("\nJOINT DIAL COMPLETE", flush=True)
