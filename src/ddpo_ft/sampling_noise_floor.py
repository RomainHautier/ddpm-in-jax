"""How much of a metric difference is just the sampling noise draw?

Found 2026-07-31: the SAME model on the SAME frames with the SAME config gave residual 12.6 in one
harness and 14.9 in another — an 18% swing — while retention moved 0.7% and placement 0.1%. The only
difference was the batch size, which changes which noise keys are folded in. Bootstrap-over-sequences
error bars do not capture this at all: they resample frames, not noise draws.

This measures it directly: one regime, one model, one config, N independent sampling seeds. The
spread across seeds IS the noise floor, and any model-to-model difference smaller than it is not a
finding. Runs on Re=2000 (70 triplets, the dt32 ceiling) so it is cheap.
"""
import os, sys, pickle
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp
from diag_guided_residual import make_kchain_ddim_sampler
from ppo_claude import build_ddim_denoiser
from train_claude import build_base_ddpm
from viz_energy import local_hik_energy
from src.rewards import make_spectrum_fn, make_residual_loss
from src.physics_guidance import make_dx_func
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence
from eval_ddpo import eff_resolution

MEAN, SIG, N, HIK0 = 0.0, 4.7988, 256, 32
RE = 2000.0
GT = 'flow-data/kf_re2000_256_40seed_dt32.npy'
SEQS = [s for s in range(40) if s > 4]
CK = 'monitoring/ddpo_re2000_dt32_ckpts/ddpo_re1000_iter0599.pkl'
SEEDS = [700, 1700, 2700, 3700, 4700]

ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
sat, s1t = float(jnp.sqrt(ab[150])), float(jnp.sqrt(1.0 - ab[150]))
P = pickle.load(open(CK, 'rb'))['params']


def batched(fn, x, seed, bs=16):
    k = jax.random.PRNGKey(seed); o = []
    for i in range(0, len(x), bs):
        o.append(np.asarray(fn(jnp.asarray(x[i:i + bs]), jax.random.fold_in(k, i))))
    return np.concatenate(o)


xg, xl = [], []
for s in SEQS:
    seq = load_sequence(GT, s)
    xg.append(build_triplets(seq, MEAN, SIG))
    xl.append(build_triplets(grid_downsample_degrade(seq, 4), MEAN, SIG))
xg, xl = np.concatenate(xg), np.concatenate(xl)
print(f"Re={int(RE)}: {len(SEQS)} sequences, {len(xg)} triplets, {len(SEEDS)} sampling seeds",
      flush=True)

resid_fn = jax.jit(make_residual_loss(n=N, re=RE, std=SIG, mean=0.0))
E_gt = np.asarray(spec_fn(jnp.asarray(xg))).mean(0)
Ehg = local_hik_energy(xg[..., 1] * SIG, HIK0, 6.0)
rg = np.concatenate([np.asarray(resid_fn(jnp.asarray(xg[i:i + 32]))).ravel()
                     for i in range(0, len(xg), 32)]).mean()
dx = make_dx_func(n=N, re=RE, std=SIG, mean=MEAN)
smp = make_kchain_ddim_sampler(ddpm.unet, ab, [150, 100, 50], 86, dx, 3.0, temp=0.30)

rows = []
for sd in SEEDS:
    recon = batched(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape)), xl, sd - 200)
    y = batched(lambda xb, kk: smp(P, sat * xb + s1t * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), recon, sd)
    E = np.asarray(spec_fn(jnp.asarray(y))).mean(0)
    ry = np.concatenate([np.asarray(resid_fn(jnp.asarray(y[i:i + 32]))).ravel()
                         for i in range(0, len(y), 32)]).mean()
    Eh = local_hik_energy(y[..., 1] * SIG, HIK0, 6.0)
    rows.append(dict(seed=sd, ret=float(E[HIK0:96].sum() / E_gt[HIK0:96].sum()),
                     place=float(np.corrcoef(Eh.ravel(), Ehg.ravel())[0, 1]),
                     resid=float(ry), resid_ratio=float(ry / rg),
                     lowk=float(E[1:5].sum() / E_gt[1:5].sum()), kstar=eff_resolution(E, E_gt)))
    r = rows[-1]
    print(f"  seed {sd}: ret={r['ret']:.3f}  place={r['place']:.3f}  resid={r['resid']:.1f} "
          f"({r['resid_ratio']:.2f}x GT)  lowk={r['lowk']:.3f}  k*={r['kstar']}", flush=True)

print("\nSAMPLING-NOISE FLOOR (spread across seeds, same model/frames/config):")
for f in ('ret', 'place', 'resid', 'resid_ratio', 'lowk'):
    v = np.array([r[f] for r in rows])
    print(f"  {f:<12} mean {v.mean():8.3f}   sd {v.std():7.4f}   "
          f"range {v.max()-v.min():7.4f}   sd/mean {v.std()/abs(v.mean()):6.2%}")
np.savez('base_results/sampling_noise_floor.npz',
         **{f: np.array([r[f] for r in rows]) for f in rows[0]}, resid_gt=rg)
print("\nNOISE FLOOR COMPLETE", flush=True)
