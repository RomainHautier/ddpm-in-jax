"""BAND-RESOLVED PLACEMENT + SPECTRUM DISTRIBUTIONS at the two ground-truth-statistics regimes
(user 2026-08-19): where in the spectrum does structural correlation survive, per model and
depth, and what is the sample-to-sample spread of the spectra? Re=1000 and Re=2000, the Track-A
card checkpoints, K3/K4/K5. Same pools/seeds as the cards, so every number is comparable.

Per cell:
- band placement: corr(local band-energy map of the sample, same map of the paired GT frame),
  bands [1,5),[5,16),[16,32),[32,64),[64,96) — the decay curve of structure with wavenumber.
  (Placement is a sample-paired AUDIT metric; never used for selection.)
- spectrum distribution: per-sample E(k) percentiles (10/50/90) over the val pool.
Output: base_results/bandplace_re1k2k.npz
"""
import os, sys, pickle
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp
from functools import partial
from diag_guided_residual import make_kchain_ddim_sampler
from ppo_claude import build_ddim_denoiser
from train_claude import build_base_ddpm
from src.rewards import make_spectrum_fn
from src.physics_guidance import make_dx_func
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence
from psample import pbatched

MEAN, SIG, N = 0.0, 4.7988, 256
BANDS = [(1, 5), (5, 16), (16, 32), (32, 64), (64, 96)]
LADDER = {'K3x86': ([150, 100, 50], 86), 'K4x110': ([200, 150, 100, 50], 110),
          'K5x140': ([250, 200, 150, 100, 50], 140)}
REGIMES = {1000: dict(gt='flow-data/kf_2d_re1000_256_40seed.npy', seqs=list(range(4, 20)), per=8),
           2000: dict(gt='flow-data/generated/gen_fnons_re2000_kf_1024to256_20seq.npy',
                      seqs=list(range(8, 20)), per=10)}
R1K = 'monitoring/ddpo_re1000_newpool_ckpts/ddpo_re1000_iter0449.pkl'
R2K = 'monitoring/ddpo_re2000_newpool_ckpts/ddpo_re1000_iter0599.pkl'
ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
MODELS = {'base': base_params,
          'r1k-449': pickle.load(open(R1K, 'rb'))['params'],
          're2k-599': pickle.load(open(R2K, 'rb'))['params']}
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
B16 = partial(pbatched, per_dev=16)

fy = np.fft.fftfreq(N) * N
kmag = np.sqrt(fy[:, None] ** 2 + fy[None, :] ** 2)
gsm = np.exp(-2.0 * (np.pi * 6.0) ** 2 * ((fy[:, None] / N) ** 2 + (fy[None, :] / N) ** 2))
MASKS = [( (kmag >= lo) & (kmag < hi) ).astype(np.float32) for lo, hi in BANDS]


def band_maps(w):
    """w (B,H,W) physical vorticity -> (nbands, B, H, W) smoothed local band-energy maps."""
    F = np.fft.fft2(w)
    out = []
    for m in MASKS:
        bp = np.real(np.fft.ifft2(F * m))
        out.append(np.real(np.fft.ifft2(np.fft.fft2(bp ** 2) * gsm)))
    return np.stack(out)


OUT = {}
OUTP = 'base_results/bandplace_re1k2k.npz'
if os.path.exists(OUTP):
    old = np.load(OUTP, allow_pickle=True); OUT = {k: old[k] for k in old.files}

for R, c in REGIMES.items():
    xg, xl = [], []
    for s in c['seqs']:
        q = load_sequence(c['gt'], s)
        g = build_triplets(q, MEAN, SIG)
        l = build_triplets(grid_downsample_degrade(q, 4), MEAN, SIG)
        i2 = np.linspace(0, len(g) - 1, c['per']).astype(int)
        xg.append(g[i2]); xl.append(l[i2])
    xg, xl = np.concatenate(xg), np.concatenate(xl)
    dx = make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN)
    gt_maps = band_maps(xg[..., 1] * SIG)
    Eg_all = np.asarray(spec_fn(jnp.asarray(xg)))
    for p in (10, 50, 90):
        OUT[f'{R}|GT||Ep{p}'] = np.percentile(Eg_all, p, axis=0).astype(np.float32)
    recon = B16(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape)), xl, 500)
    # many-frame LR/recon reference rows (user diagnostic 2026-08-19): how much placement the
    # observation itself carries per band, and how much the recon inherits from it
    lr_maps = band_maps(xl[..., 1] * SIG)
    rc_maps = band_maps(recon[..., 1] * SIG)
    for tag, (a, b) in {'LRvsGT': (lr_maps, gt_maps), 'reconvsGT': (rc_maps, gt_maps),
                        'reconvsLR': (rc_maps, lr_maps)}.items():
        bp = np.array([np.corrcoef(a[i].ravel(), b[i].ravel())[0, 1]
                       for i in range(len(BANDS))], np.float32)
        OUT[f'{R}|{tag}||bp'] = bp
        print(f"  {tag:<10} bands " + " ".join(f"{x:.3f}" for x in bp), flush=True)
    np.savez(OUTP, **OUT)
    print(f"\n=== Re={R}: {len(xg)} triplets ===", flush=True)
    for cname, (starts, steps) in LADDER.items():
        smp = make_kchain_ddim_sampler(ddpm.unet, ab, starts, steps, dx, 3.0, temp=0.30)
        sa, s1 = float(jnp.sqrt(ab[starts[0]])), float(jnp.sqrt(1.0 - ab[starts[0]]))
        for nm, P in MODELS.items():
            key = f'{R}|{nm}|{cname}'
            if f'{key}||bp' in OUT:
                continue
            y = B16(lambda xb, kk: smp(P, sa * xb + s1 * jax.random.normal(
                jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), recon, 700)
            ym = band_maps(y[..., 1] * SIG)
            bp = np.array([np.corrcoef(ym[b].ravel(), gt_maps[b].ravel())[0, 1]
                           for b in range(len(BANDS))], np.float32)
            Ey = np.asarray(spec_fn(jnp.asarray(y)))
            OUT[f'{key}||bp'] = bp
            for p in (10, 50, 90):
                OUT[f'{key}||Ep{p}'] = np.percentile(Ey, p, axis=0).astype(np.float32)
            np.savez(OUTP, **OUT)
            print(f"  {nm:<9} @ {cname}  bands " +
                  " ".join(f"{b:.3f}" for b in bp), flush=True)
print("\nBANDPLACE COMPLETE", flush=True)
