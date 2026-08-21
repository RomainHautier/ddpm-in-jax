"""SAMPLING-TRAJECTORY EMBEDDINGS (user 2026-08-21): watch guided sampling travel through
representation space. Objects: GT fine snapshots, coarse-input recons, and the cascade's
per-chain stage outputs (K3 -> 3 waypoints per sample) with the measured-target spectral dial
OFF (ls=0) and ON (ls=16). Two embeddings of the SAME states:
 (a) log-spectrum space: 126-d log E(k), PCA(2) fit on GT+recon;
 (b) UNet bottleneck: flax capture_intermediates -> mid SelfAttention output (B,32,32,128),
     spatial mean -> 128-d, PCA(2) fit on GT+recon embeddings. Bottleneck probed at t=1.
Model re2k-149, THREE regimes (2000 home / 5000 / 8000) x five variants, 60 val samples each — prediction to test: at home the dose gradient ~0, so steered and unsteered trajectories should nearly coincide; divergence grows with regime distance. Output: base_results/trajectory_embed.npz + figure.
"""
import os, sys, pickle
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp
from functools import partial
from diag_guided_residual import make_kchain_ddim_sampler
from ppo_claude import build_ddim_denoiser
from train_claude import build_base_ddpm
from src.rewards import make_spectrum_fn, make_spectrum_distance
from src.physics_guidance import make_dx_func
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence
from psample import pbatched

MEAN, SIG, N = 0.0, 4.7988, 256
STARTS, STEPS = [150, 100, 50], 86
GEN = 'flow-data/generated/gen_fnons_re{}_kf_1024to256_20seq.npy'
REGIMES = {2000: 'base_results/regime_stats_re2000_measured_train.npz',
           5000: 'base_results/regime_stats_re5000_measured_train.npz',
           8000: 'base_results/regime_stats_re8000_measured_train.npz'}
P = pickle.load(open('monitoring/ddpo_re2000_newpool_ckpts/ddpo_re1000_iter0149.pkl', 'rb'))['params']
ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
sa3, s13 = float(jnp.sqrt(ab[STARTS[0]])), float(jnp.sqrt(1.0 - ab[STARTS[0]]))
B16 = partial(pbatched, per_dev=16)

SAVE, FSAVE = {}, {}
for R in REGIMES:
    print(f'=== Re={R} ===', flush=True)
    xg, xl = [], []
    for sq in range(8, 14):
        q = load_sequence(GEN.format(R), sq)
        g = build_triplets(q, MEAN, SIG)
        l = build_triplets(grid_downsample_degrade(q, 4), MEAN, SIG)
        i2 = np.linspace(0, len(g) - 1, 10).astype(int)
        xg.append(g[i2]); xl.append(l[i2])
    xg, xl = np.concatenate(xg), np.concatenate(xl)
    recon = B16(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape)), xl, 500)

    d = np.load(REGIMES[R]); stats = {k: d[k] for k in d.files}
    lref = stats.get('log_spec_ref')
    d1 = make_spectrum_distance(stats['spec_ref'], kband=(1, 96), n=N, log_ref=lref)
    d2 = make_spectrum_distance(stats['spec_ref'], kband=(32, 96), n=N, log_ref=lref)
    def dose_loss(x): return jnp.sum(0.5 * d1(x) + 3.0 * d2(x))
    dx_anchor = jax.jit(jax.grad(dose_loss))
    dx_pde = make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN)

    # five guidance variants, IDENTICAL random keys throughout (same folds, same shapes):
    # none / residual-only / placement-only / reward(dose)-only / all three.
    fy2 = np.fft.fftfreq(N) * N
    km2 = np.sqrt(fy2[:, None] ** 2 + fy2[None, :] ** 2)
    gs2 = jnp.asarray(np.exp(-2.0 * (np.pi * 6.0) ** 2 *
                             ((fy2[:, None] / N) ** 2 + (fy2[None, :] / N) ** 2)))
    PB2 = [jnp.asarray(((km2 >= lo) & (km2 < hi)).astype(np.float32)) for lo, hi in
           [(16, 32), (32, 64)]]

    def nmaps2(w):
        F = jnp.fft.fft2(w)
        out = []
        for m in PB2:
            bp = jnp.real(jnp.fft.ifft2(F * m))
            e = jnp.real(jnp.fft.ifft2(jnp.fft.fft2(bp ** 2) * gs2))
            out.append(e / (jnp.mean(e, axis=(-2, -1), keepdims=True) + 1e-12))
        return out

    def make_place_dx2(refs):
        refs = [jax.lax.stop_gradient(r) for r in refs]
        def loss(x):
            ms = nmaps2(x[..., 1] * SIG)
            return sum(jnp.sum((m - r) ** 2) for m, r in zip(ms, refs)) / (N * N)
        return jax.grad(loss)

    VARIANTS = {          # (use_pde, ls, mu); sampler lam=3, dx composed as (lp/3)pde + (ls/3)dose + (mu/3)place
        'none':      (0.0, 0.0, 0.0),
        'residual':  (3.0, 0.0, 0.0),
        'placement': (0.0, 0.0, 3.0),
        'reward':    (0.0, 16.0, 0.0),
        'all3':      (3.0, 16.0, 3.0),
    }
    STAGES = {}
    for tag, (lp, ls, mu) in VARIANTS.items():
        outs = []
        k0 = jax.random.PRNGKey(700)
        for i in range(0, len(recon), 64):
            xb = recon[i:i + 64]
            xb_l = xl[i:i + 64]
            pd = make_place_dx2(nmaps2(jnp.asarray(xb_l[..., 1]) * SIG)) if mu > 0 else None
            def dx(x, _lp=lp, _ls=ls, _mu=mu, _pd=pd):
                g = (_lp / 3.0) * dx_pde(x) + (_ls / 3.0) * dx_anchor(x)
                if _pd is not None:
                    g = g + (_mu / 3.0) * _pd(x)
                return g
            lam = 3.0 if (lp + ls + mu) > 0 else 0.0
            smp = make_kchain_ddim_sampler(ddpm.unet, ab, STARTS, STEPS, dx, lam, temp=0.30,
                                           return_stages=True)
            st = smp(P, sa3 * jnp.asarray(xb) + s13 * jax.random.normal(
                jax.random.fold_in(jax.random.fold_in(k0, i), 1), xb.shape),
                jax.random.fold_in(jax.random.fold_in(k0, i), 2))
            outs.append([np.asarray(x) for x in st])
        STAGES[tag] = [np.concatenate([o[j] for o in outs]) for j in range(len(STARTS))]
        print(f"{tag}: stages {STAGES[tag][0].shape}", flush=True)

    # ---- embedding (a): log-spectra ----
    def logspec(x): return np.log(np.asarray(spec_fn(jnp.asarray(x)))[:, 1:127] + 1e-12)
    OBJ = {'GT': xg, 'LR': xl, 'recon': recon,
           **{f'{tag}_s{j+1}': STAGES[tag][j] for tag in STAGES for j in range(3)}}
    LS_ = {k: logspec(v) for k, v in OBJ.items()}
    fitA = np.concatenate([LS_['GT'], LS_['recon']])
    muA = fitA.mean(0); U = np.linalg.svd(fitA - muA, full_matrices=False)[2][:2]
    EMB_A = {k: (v - muA) @ U.T for k, v in LS_.items()}

    # ---- embedding (b): UNet bottleneck at t=1, spatial-mean, PCA ----
    def bottleneck(x):
        out = []
        for i in range(0, len(x), 32):
            _, st = ddpm.unet.apply({'params': base_params}, jnp.asarray(x[i:i + 32]),
                                    jnp.full((min(32, len(x) - i),), 1, jnp.int32), train=False,
                                    capture_intermediates=True, mutable=['intermediates'])
            inter = st['intermediates']
            key = [k for k in inter if 'SelfAttention' in k][0]
            z = np.asarray(inter[key]['__call__'][0])
            out.append(z.mean(axis=(1, 2)))
        return np.concatenate(out)
    BN = {k: bottleneck(v) for k, v in OBJ.items()}
    fitB = np.concatenate([BN['GT'], BN['recon']])
    muB = fitB.mean(0); Ub = np.linalg.svd(fitB - muB, full_matrices=False)[2][:2]
    EMB_B = {k: (v - muB) @ Ub.T for k, v in BN.items()}

    # field renders: same two held-out samples under every variant (identical noise keys)
    FLDS = {'GT': (xg[:2, ..., 1] * SIG), 'LR': (xl[:2, ..., 1] * SIG),
            'recon': (np.asarray(recon[:2, ..., 1]) * SIG)}
    for tag in STAGES:
        FLDS[tag] = STAGES[tag][-1][:2, ..., 1] * SIG        # final-stage x0 of the cascade
    FSAVE.update(
                        **{f'{R}|{k}': v.astype(np.float32) for k, v in FLDS.items()})
    SAVE.update(
             **{f'{R}|A|{k}': v for k, v in EMB_A.items()},
             **{f'{R}|B|{k}': v for k, v in EMB_B.items()})

np.savez('base_results/trajectory_embed.npz', **SAVE)
np.savez_compressed('base_results/trajectory_fields.npz', **FSAVE)
print("TRAJECTORY EMBED COMPLETE", flush=True)
