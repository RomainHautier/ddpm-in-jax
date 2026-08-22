"""THE DEFINITIVE STEERING GRID (user 2026-08-22): two fine-tuned models x five sampling
strategies x ALL NINE regimes, fixed K3x86 — the quantification of what each guidance method
buys, everywhere. Measured (Track-A) targets throughout; the two-sided dose gradient pushes
down where a model runs hot and up where it runs cold, so lower regimes are covered by the
same machinery. Strategies: none (lam=0) / residual (lp=3) / reward-dose (ls=8) /
placement (mu=3) / all three (lp3+ls8+mu3). Models: r1k-449, re2k-149, and rs8kkl-799 (the KL-loosened measured-anchor Re=8000 specialist - tests whether its fine-tune worked in-distribution once steering tops up the under-dose, and how it transfers DOWN under each strategy).
Per cell: full battery + spectrum (per-band retention derivable). Keys
'{R}|{model}|SG{strategy}' in base_results/steering_full_grid.npz. Resume-aware.
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
REGIMES = {
    1000: dict(gt='flow-data/kf_2d_re1000_256_40seed.npy', seqs=list(range(4, 20)), per=8,
               anchor='base_results/regime_stats_re1000_measured_train.npz'),
    **{R: dict(gt=GEN.format(R), seqs=list(range(8, 20)), per=10,
               anchor=f'base_results/regime_stats_re{R}_measured_train.npz')
       for R in (1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000)},
}
STRATS = {'none': (0.0, 0.0, 0.0), 'residual': (3.0, 0.0, 0.0), 'reward': (0.0, 8.0, 0.0),
          'placement': (0.0, 0.0, 3.0), 'all3': (3.0, 8.0, 3.0)}
R1K = 'monitoring/ddpo_re1000_newpool_ckpts/ddpo_re1000_iter0449.pkl'
R2K = 'monitoring/ddpo_re2000_newpool_ckpts/ddpo_re1000_iter0149.pkl'
R8K = 'monitoring/ddpo_re8000_rs_kl3_ckpts/ddpo_re1000_iter0799.pkl'
PRK = 'monitoring/ddpo_re2000_placereward_ckpts/ddpo_re1000_iter0549.pkl'
ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
MODELS = {'r1k-449': pickle.load(open(R1K, 'rb'))['params'],
          're2k-149': pickle.load(open(R2K, 'rb'))['params'],
          'rs8kkl-799': pickle.load(open(R8K, 'rb'))['params'],
          'pr2k-549': pickle.load(open(PRK, 'rb'))['params']}
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
sa3, s13 = float(jnp.sqrt(ab[STARTS[0]])), float(jnp.sqrt(1.0 - ab[STARTS[0]]))
B16 = partial(pbatched, per_dev=16)

fy = np.fft.fftfreq(N) * N
km = np.sqrt(fy[:, None] ** 2 + fy[None, :] ** 2)
gsm = jnp.asarray(np.exp(-2.0 * (np.pi * 6.0) ** 2 *
                         ((fy[:, None] / N) ** 2 + (fy[None, :] / N) ** 2)))
PB = [jnp.asarray(((km >= lo) & (km < hi)).astype(np.float32)) for lo, hi in
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


OUT = {}
OUTP = 'base_results/steering_full_grid.npz'
if os.path.exists(OUTP):
    old = np.load(OUTP, allow_pickle=True); OUT = {k: old[k] for k in old.files}

for R, c in REGIMES.items():
    if all(f'{R}|{nm}|SG{sg}||ret' in OUT for nm in MODELS for sg in STRATS):
        continue
    d = np.load(c['anchor'])
    stats = {k: d[k] for k in d.files}
    dx_pde = make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN)
    dx_dose = make_anchor_dose_dx(stats)
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
    OUT[f'{R}|GT||E'] = E_gt.astype(np.float32)
    print(f"\n=== Re={R}: full strategy grid ===", flush=True)
    # 4-chip parallel strategy loop (v1 ran single-chip at ~11 min/cell): recon+LR packed as
    # 6 channels so the pmapped fn can build per-sample placement refs from its own chunk.
    xpack = np.concatenate([np.asarray(recon), xl], axis=-1)

    def ploss(x, refs):
        ms = nmaps(x[..., 1] * SIG)
        return sum(jnp.sum((m - jax.lax.stop_gradient(r)) ** 2)
                   for m, r in zip(ms, refs)) / (N * N)
    pgrad = jax.grad(ploss)
    for sg, (lp, ls, mu) in STRATS.items():
        # sampler built OUTSIDE the pmap trace (its build-time float() constants must stay
        # concrete); per-chunk placement refs enter through the aux ARGUMENT.
        def dx(x, aux, _lp=lp, _ls=ls, _mu=mu):
            g = (_lp / 3.0) * dx_pde(x) + (_ls / 3.0) * dx_dose(x)
            if _mu > 0:
                g = g + (_mu / 3.0) * pgrad(x, aux)
            return g
        lam = 3.0 if (lp + ls + mu) > 0 else 0.0
        smp = make_kchain_ddim_sampler(ddpm.unet, ab, STARTS, STEPS, dx, lam,
                                       temp=0.30, jit=False, aux_dx=True)
        for nm, P in MODELS.items():
            key = f'{R}|{nm}|SG{sg}'
            if f'{key}||ret' in OUT:
                continue
            def run6(xb6, kk, _P=P, _smp=smp):
                rc, lr = xb6[..., :3], xb6[..., 3:]
                refs = tuple(nmaps(lr[..., 1] * SIG))
                return _smp(_P, sa3 * rc + s13 * jax.random.normal(
                    jax.random.fold_in(kk, 1), rc.shape), jax.random.fold_in(kk, 2),
                    aux=refs)
            y = B16(run6, xpack, 700)
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
            print(f"  {nm:<9} {sg:<10} ret={vals['ret']:.3f} place={vals['place']:.3f} "
                  f"resid={vals['resid_ratio']:.2f}xGT lowk={vals['lowk']:.3f}", flush=True)
    jax.clear_caches()
print("\nSTEERING FULL GRID COMPLETE", flush=True)
