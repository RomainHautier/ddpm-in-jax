"""RESIDUAL-DIAL PILOT (user 2026-08-18): the companion to the spectral steering dial — sweep
the PDE-residual guidance strength itself. Every deployed cell ever graded used lam_pde=3,
frozen; it has never been dose-swept on the current models. Same three frozen models, same
regimes, pools, seeds and battery as the spectral pilot, spectral dial OFF:
lam_pde in {0, 1, 3, 6, 12} — 0 = no guidance at all, 3 = the deployment convention (should
reproduce the spectral pilot's ls=0 rows within sampling noise, a built-in consistency check).

Keys 'lp{val}' merge into the same steering_pilot.npz so one page renders both dials.
"""
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp, pickle
from functools import partial
from diag_guided_residual import make_kchain_ddim_sampler
from ppo_claude import build_ddim_denoiser
from train_claude import build_base_ddpm
from viz_energy import local_hik_energy
from src.rewards import make_spectrum_fn, make_residual_loss
from src.physics_guidance import make_dx_func
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence
from eval_ddpo import eff_resolution
from psample import pbatched

MEAN, SIG, N, HIK0 = 0.0, 4.7988, 256, 32
STARTS, STEPS = [150, 100, 50], 86
LP = [0.0, 1.0, 3.0, 6.0, 12.0]
GEN = 'flow-data/generated/gen_fnons_re{}_kf_1024to256_20seq.npy'
REGIMES = {2000: dict(gt=GEN.format(2000), anchor='base_results/regime_stats_re2000_obsfit_newgen.npz'),
           5000: dict(gt=GEN.format(5000), anchor='base_results/regime_stats_re5000_obsfit_gen.npz'),
           8000: dict(gt=GEN.format(8000), anchor='base_results/regime_stats_re8000_obsfit_gen.npz')}
R1K = 'monitoring/ddpo_re1000_newpool_ckpts/ddpo_re1000_iter0149.pkl'
R2K = 'monitoring/ddpo_re2000_newpool_ckpts/ddpo_re1000_iter0149.pkl'
ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
MODELS = {'base': base_params,
          'r1k-149': pickle.load(open(R1K, 'rb'))['params'],
          're2k-149': pickle.load(open(R2K, 'rb'))['params']}
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
sa3, s13 = float(jnp.sqrt(ab[STARTS[0]])), float(jnp.sqrt(1.0 - ab[STARTS[0]]))
B16 = partial(pbatched, per_dev=16)


def pool(gt, seqs, n_per, with_gt=False):
    xg, xl = [], []
    for s in seqs:
        q = load_sequence(gt, s)
        l = build_triplets(grid_downsample_degrade(q, 4), MEAN, SIG)
        i2 = np.linspace(0, len(l) - 1, n_per).astype(int)
        xl.append(l[i2])
        if with_gt:
            xg.append(build_triplets(q, MEAN, SIG)[i2])
    return (np.concatenate(xg) if with_gt else None), np.concatenate(xl)


OUT = {}
OUTP = 'base_results/steering_pilot.npz'
if os.path.exists(OUTP):
    old = np.load(OUTP, allow_pickle=True); OUT = {k: old[k] for k in old.files}

for R, c in REGIMES.items():
    d = np.load(c['anchor'])
    A = d['spec_ref']
    src_seqs = eval(d['obs_source'].item().decode().split('|seqs=')[1])
    dx_pde = make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN)
    _, xl_src = pool(c['gt'], src_seqs, 8)
    xg, xl_val = pool(c['gt'], list(range(8, 20)), 10, with_gt=True)
    resid_fn = jax.jit(make_residual_loss(n=N, re=float(R), std=SIG, mean=0.0))
    E_gt = np.asarray(spec_fn(jnp.asarray(xg))).mean(0)
    Ehg = local_hik_energy(xg[..., 1] * SIG, HIK0, 6.0)
    rg = float(np.concatenate([np.asarray(resid_fn(jnp.asarray(xg[i:i + 32]))).ravel()
                               for i in range(0, len(xg), 32)]).mean())
    rc_src = B16(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape)), xl_src, 500)
    rc_val = B16(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape)), xl_val, 500)
    print(f"\n=== Re={R}: residual-dial sweep @ K3x86 ===", flush=True)
    for lp in LP:
        smp = make_kchain_ddim_sampler(ddpm.unet, ab, STARTS, STEPS, dx_pde, lp, temp=0.30)
        for nm, P in MODELS.items():
            key = f'{R}|{nm}|lp{lp:g}'
            if f'{key}||ret' in OUT:
                continue
            yb = B16(lambda xb, kk: smp(P, sa3 * xb + s13 * jax.random.normal(
                jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), rc_src, 700)
            blind = float(np.asarray(spec_fn(jnp.asarray(yb))).mean(0)[10:96].sum() / A[10:96].sum())
            y = B16(lambda xb, kk: smp(P, sa3 * xb + s13 * jax.random.normal(
                jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), rc_val, 700)
            E = np.asarray(spec_fn(jnp.asarray(y))).mean(0)
            ry = float(np.concatenate([np.asarray(resid_fn(jnp.asarray(y[i:i + 32]))).ravel()
                                       for i in range(0, len(y), 32)]).mean())
            Eh = local_hik_energy(y[..., 1] * SIG, HIK0, 6.0)
            vals = dict(ret=E[HIK0:96].sum() / E_gt[HIK0:96].sum(),
                        place=np.corrcoef(Eh.ravel(), Ehg.ravel())[0, 1],
                        lowk=E[1:5].sum() / E_gt[1:5].sum(), kstar=eff_resolution(E, E_gt),
                        resid_ratio=ry / rg, blind_src=blind,
                        mse=np.mean((y[..., 1] - xg[..., 1]) ** 2) * SIG ** 2)
            for f, v in vals.items():
                OUT[f'{key}||{f}'] = np.float32(v)
            OUT[f'{key}||E'] = E.astype(np.float32)
            np.savez(OUTP, **OUT)
            print(f"  {nm:<9} lp={lp:<4g} blind(src)={blind:.3f}  ret={vals['ret']:.3f}  "
                  f"place={vals['place']:.3f}  lowk={vals['lowk']:.3f}  "
                  f"resid={vals['resid_ratio']:.2f}xGT  mse={vals['mse']:.2f}  k*={vals['kstar']}",
                  flush=True)
print("\nPDE DIAL PILOT COMPLETE", flush=True)
