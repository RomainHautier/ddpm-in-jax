"""REWARD-STEERED SAMPLING PILOT (user's proposal 2026-08-18): differentiate the retention-
matching reward and add its gradient to the sampling-time guidance — can steering supply the
DOSE that physics guidance never did, without any fine-tuning?

Mechanics: the kchain sampler already applies lam*dx(x0_hat) at the clean estimate inside every
step. We compose dx_total = dx_pde + (ls/lam)*dx_anchor, where dx_anchor is the gradient of the
reward's spectral distances to the regime's EXTRAPOLATED anchor (GT-free, deployable):
0.5*d_spec[1,96) + 3.0*d_spec_highk[32,96), the same weights the reward uses. The gradient is
taken through a SUM over the batch, so per-sample steering strength is batch-size independent
(make_dx_func's batch-mean convention makes lam depend on the sampler's batch — matched here by
running per_dev=16, the historical serial batch).

Sweep: three models x ls in {0, 0.5, 1, 2, 4} x Re=2000/5000/8000, all at the K3x86 cascade:
- base (no fine-tune): steering as a REPLACEMENT for fine-tuning;
- r1k-149 (fine-tuned for Re=1000, transported): steering as a TOP-UP — can ls make K3
  sufficient where this model undershoots and currently needs K5?
- re2k-149 (fine-tuned for Re=2000, the current best traveler): steering on the deployed pick.
Every cell: blind score on the anchor's source pool (could the rule tune ls blind?) + the full
battery on the standard val pool (did it actually work?).

Readout: ls=0 is the current deployment (base ret ~0.31/0.23/0.21 at K3). If retention climbs
toward 1 with ls while placement/residual stay sane, sampling-time dose replaces the depth
ladder with a continuous knob; if the tail fills with misplaced energy (placement collapses,
residual blows past GT), the fine-tunes stay necessary and we learn why.
"""
import os, sys
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
STARTS, STEPS = [150, 100, 50], 86          # K3x86
LS = [0.0, 0.5, 1.0, 2.0, 4.0]
GEN = 'flow-data/generated/gen_fnons_re{}_kf_1024to256_20seq.npy'
REGIMES = {2000: dict(gt=GEN.format(2000), anchor='base_results/regime_stats_re2000_obsfit_newgen.npz'),
           5000: dict(gt=GEN.format(5000), anchor='base_results/regime_stats_re5000_obsfit_gen.npz'),
           8000: dict(gt=GEN.format(8000), anchor='base_results/regime_stats_re8000_obsfit_gen.npz')}

import pickle
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
B16 = partial(pbatched, per_dev=16)         # match the historical serial batch -> same lam_eff


def make_anchor_dose_dx(stats):
    lref = stats.get('log_spec_ref')
    d1 = make_spectrum_distance(stats['spec_ref'], kband=(1, 96), n=N, log_ref=lref)
    d2 = make_spectrum_distance(stats['spec_ref'], kband=(32, 96), n=N, log_ref=lref)
    def loss(x):
        return jnp.sum(0.5 * d1(x) + 3.0 * d2(x))    # SUM over batch: per-sample grad, B-independent
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
    A = stats['spec_ref']
    src_seqs = eval(d['obs_source'].item().decode().split('|seqs=')[1])
    dx_pde = make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN)
    dx_anchor = make_anchor_dose_dx(stats)
    # pools
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
    print(f"\n=== Re={R}: steering sweep @ K3x86 ===", flush=True)
    for ls in LS:
        dx_comb = (dx_pde if ls == 0.0 else
                   jax.jit(lambda x, _l=ls: dx_pde(x) + (_l / 3.0) * dx_anchor(x)))
        smp = make_kchain_ddim_sampler(ddpm.unet, ab, STARTS, STEPS, dx_comb, 3.0, temp=0.30)
        for nm, P in MODELS.items():
            key = f'{R}|{nm}|ls{ls:g}'
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
            print(f"  {nm:<9} ls={ls:<4g} blind(src)={blind:.3f}  ret={vals['ret']:.3f}  "
                  f"place={vals['place']:.3f}  lowk={vals['lowk']:.3f}  "
                  f"resid={vals['resid_ratio']:.2f}xGT  mse={vals['mse']:.2f}  k*={vals['kstar']}",
                  flush=True)
print("\nSTEERING PILOT COMPLETE", flush=True)
