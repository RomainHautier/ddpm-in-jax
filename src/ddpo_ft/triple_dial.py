"""TRIPLE DIAL (user 2026-08-21): all three gradients at once — measured-target spectral dose
(ls) + placement consistency vs own input (mu) + physics polish (lp=3 fixed). The question:
does mu recover the spectral dial's placement cost while keeping its retention gain?
Grid: re2k-149 @ K3x86, (ls, mu) in {(8,0),(8,3),(8,10),(16,3),(16,10)}, Re 5000/8000.
Keys '{R}|re2k-149|K3ls{a}mu{b}' in steering_pilot.npz. ((8,0)/(16,0) axes exist in
steering_measured as ls8/ls16.)
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
COMBOS = [(8.0, 3.0), (8.0, 10.0), (16.0, 3.0), (16.0, 10.0), (8.0, 0.0)]
GEN = 'flow-data/generated/gen_fnons_re{}_kf_1024to256_20seq.npy'
REGIMES = {5000: dict(gt=GEN.format(5000), anchor='base_results/regime_stats_re5000_measured_train.npz'),
           8000: dict(gt=GEN.format(8000), anchor='base_results/regime_stats_re8000_measured_train.npz')}
P = pickle.load(open('monitoring/ddpo_re2000_newpool_ckpts/ddpo_re1000_iter0149.pkl', 'rb'))['params']
ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
sa3, s13 = float(jnp.sqrt(ab[STARTS[0]])), float(jnp.sqrt(1.0 - ab[STARTS[0]]))
B16 = partial(pbatched, per_dev=16)

fy = np.fft.fftfreq(N) * N
kmag = np.sqrt(fy[:, None] ** 2 + fy[None, :] ** 2)
gsm = jnp.asarray(np.exp(-2.0 * (np.pi * 6.0) ** 2 *
                         ((fy[:, None] / N) ** 2 + (fy[None, :] / N) ** 2)))
PB = [jnp.asarray(((kmag >= lo) & (kmag < hi)).astype(np.float32)) for lo, hi in
      [(16, 32), (32, 64)]]


def nmaps(w):
    F = jnp.fft.fft2(w)
    out = []
    for m in PB:
        bp = jnp.real(jnp.fft.ifft2(F * m))
        e = jnp.real(jnp.fft.ifft2(jnp.fft.fft2(bp ** 2) * gsm))
        out.append(e / (jnp.mean(e, axis=(-2, -1), keepdims=True) + 1e-12))
    return out


def make_place_dx(refs):
    refs = [jax.lax.stop_gradient(r) for r in refs]
    def loss(x):
        ms = nmaps(x[..., 1] * SIG)
        return sum(jnp.sum((m - r) ** 2) for m, r in zip(ms, refs)) / (N * N)
    return jax.grad(loss)


def make_anchor_dose_dx(stats):
    lref = stats.get('log_spec_ref')
    d1 = make_spectrum_distance(stats['spec_ref'], kband=(1, 96), n=N, log_ref=lref)
    d2 = make_spectrum_distance(stats['spec_ref'], kband=(32, 96), n=N, log_ref=lref)
    def loss(x):
        return jnp.sum(0.5 * d1(x) + 3.0 * d2(x))
    return jax.jit(jax.grad(loss))


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
    stats = {k: d[k] for k in d.files}
    dx_pde = make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN)
    dx_anchor = make_anchor_dose_dx(stats)
    xg, xl = pool(c['gt'], list(range(8, 20)), 10, with_gt=True)
    resid_fn = jax.jit(make_residual_loss(n=N, re=float(R), std=SIG, mean=0.0))
    E_gt = np.asarray(spec_fn(jnp.asarray(xg))).mean(0)
    Ehg = local_hik_energy(xg[..., 1] * SIG, HIK0, 6.0)
    rg = float(np.concatenate([np.asarray(resid_fn(jnp.asarray(xg[i:i + 32]))).ravel()
                               for i in range(0, len(xg), 32)]).mean())
    recon = B16(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape)), xl, 500)
    print(f"\n=== Re={R}: triple dial ===", flush=True)
    for ls, mu in COMBOS:
        key = f'{R}|re2k-149|K3ls{ls:g}mu{mu:g}'
        if f'{key}||ret' in OUT:
            continue
        def run_chunk(xb_lr, xb_rc, kk, _ls=ls, _mu=mu):
            terms = [dx_pde]
            if _mu > 0:
                pd = make_place_dx(nmaps(jnp.asarray(xb_lr[..., 1]) * SIG))
            def dx(x):
                g = dx_pde(x) + (_ls / 3.0) * dx_anchor(x)
                if _mu > 0:
                    g = g + (_mu / 3.0) * pd(x)
                return g
            smp = make_kchain_ddim_sampler(ddpm.unet, ab, STARTS, STEPS, dx, 3.0, temp=0.30)
            return smp(P, sa3 * jnp.asarray(xb_rc) + s13 * jax.random.normal(
                jax.random.fold_in(kk, 1), xb_rc.shape), jax.random.fold_in(kk, 2))
        ys, k0 = [], jax.random.PRNGKey(700)
        B = 64
        for i in range(0, len(recon), B):
            xb_l, xb_r = xl[i:i + B], recon[i:i + B]
            pad = B - len(xb_r)
            if pad:
                xb_l = np.concatenate([xb_l, np.repeat(xb_l[-1:], pad, 0)])
                xb_r = np.concatenate([xb_r, np.repeat(xb_r[-1:], pad, 0)])
            out = np.asarray(run_chunk(xb_l, xb_r, jax.random.fold_in(k0, i)))
            ys.append(out[:B - pad] if pad else out)
        y = np.concatenate(ys)
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
        print(f"  ls={ls:g} mu={mu:g}  ret={vals['ret']:.3f} place={vals['place']:.3f} "
              f"resid={vals['resid_ratio']:.2f}xGT mse={vals['mse']:.2f}", flush=True)
print("\nTRIPLE DIAL COMPLETE", flush=True)
