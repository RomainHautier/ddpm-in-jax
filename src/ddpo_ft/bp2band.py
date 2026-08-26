"""2-BAND DIAL COMPARATOR (the missing bp row) (user 2026-08-23): the base-DDIM recon step's per-band
placement effect vs the raw LR, at all nine regimes — does the [64,96) loss grow with Re?
Plus per-band energy ratios (LR and recon vs GT). Clone of the bandplace_re1k2k reference-row
machinery. Keys '{R}|{LRvsGT,reconvsGT,reconvsLR}||bp' and '{R}|{LR,recon}||Eb' in
base_results/reconcost_allre.npz.

SECOND HALF — the fix test: placement dial v2 with the raw LR's [64,96) map as a THIRD
reference band, vs the production 2-band dial; re2k-149 @ K3, mu=3, Re 5000/8000. Cells
report the standard battery PLUS the output's per-band placement (does [64,96) recover?).
Keys '{R}|re2k-149|P3mu3||...' (+ '||bp') alongside the grid's SGplacement (2-band) cells.
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
BANDS = [(1, 5), (5, 16), (16, 32), (32, 64), (64, 96)]
GEN = 'flow-data/generated/gen_fnons_re{}_kf_1024to256_20seq.npy'
REGIMES = {R: dict(gt=GEN.format(R), seqs=list(range(8, 20)), per=10)
           for R in (5000, 8000)}
ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
STARTS, STEPS = [150, 100, 50], 86
sa3, s13 = float(jnp.sqrt(ab[STARTS[0]])), float(jnp.sqrt(1.0 - ab[STARTS[0]]))
B16 = partial(pbatched, per_dev=16)
P2K = pickle.load(open('monitoring/ddpo_re2000_newpool_ckpts/ddpo_re1000_iter0149.pkl', 'rb'))['params']

fy = np.fft.fftfreq(N) * N
kmag = np.sqrt(fy[:, None] ** 2 + fy[None, :] ** 2)
gsm = np.exp(-2.0 * (np.pi * 6.0) ** 2 * ((fy[:, None] / N) ** 2 + (fy[None, :] / N) ** 2))
MASKS = [((kmag >= lo) & (kmag < hi)).astype(np.float32) for lo, hi in BANDS]
gsm_j = jnp.asarray(gsm)
MASKS_J = [jnp.asarray(m) for m in MASKS]


def band_maps(w):
    F = np.fft.fft2(w)
    out = []
    for m in MASKS:
        bp = np.real(np.fft.ifft2(F * m))
        out.append(np.real(np.fft.ifft2(np.fft.fft2(bp ** 2) * gsm)))
    return np.stack(out)


def bp_corr(a, b):
    return np.array([np.corrcoef(a[i].ravel(), b[i].ravel())[0, 1]
                     for i in range(len(BANDS))], np.float32)


def nmaps3(w, idxs):
    """differentiable normalized band-energy maps for the dial (jnp), bands by index."""
    F = jnp.fft.fft2(w)
    out = []
    for i in idxs:
        bp = jnp.real(jnp.fft.ifft2(F * MASKS_J[i]))
        e = jnp.real(jnp.fft.ifft2(jnp.fft.fft2(bp ** 2) * gsm_j))
        out.append(e / (jnp.mean(e, axis=(-2, -1), keepdims=True) + 1e-12))
    return out


OUT = {}
OUTP = 'base_results/reconcost_allre.npz'
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
    recon = B16(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape)), xl, 500)
    Eg = np.asarray(spec_fn(jnp.asarray(xg))).mean(0)
    # ---- dial v2 test at 5000/8000 ----
    if f'{R}|re2k-149|P2mu3||ret' not in OUT:
        stats = dict(np.load(f'base_results/regime_stats_re{R}_measured_train.npz'))
        dx_pde = make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN)
        resid_fn = jax.jit(make_residual_loss(n=N, re=float(R), std=SIG, mean=0.0))
        Ehg = local_hik_energy(xg[..., 1] * SIG, HIK0, 6.0)
        rg = float(np.concatenate([np.asarray(resid_fn(jnp.asarray(xg[i:i + 32]))).ravel()
                                   for i in range(0, len(xg), 32)]).mean())
        gt_maps = band_maps(xg[..., 1] * SIG)
        xpack = np.concatenate([np.asarray(recon), xl], axis=-1)
        IDX3 = [2, 3]                             # PRODUCTION 2-band refs

        def ploss(x, refs):
            ms = nmaps3(x[..., 1] * SIG, IDX3)
            return sum(jnp.sum((m - jax.lax.stop_gradient(r)) ** 2)
                       for m, r in zip(ms, refs)) / (N * N)
        pgrad = jax.grad(ploss)

        def dx3(x, aux):
            return dx_pde(x) + 1.0 * pgrad(x, aux)          # mu=3 convention: lam*(mu/3)=mu
        smp = make_kchain_ddim_sampler(ddpm.unet, ab, STARTS, STEPS, dx3, 3.0, temp=0.30,
                                       jit=False, aux_dx=True)

        def run6(xb6, kk):
            rc, lr = xb6[..., :3], xb6[..., 3:]
            refs = tuple(nmaps3(lr[..., 1] * SIG, IDX3))
            return smp(P2K, sa3 * rc + s13 * jax.random.normal(
                jax.random.fold_in(kk, 1), rc.shape), jax.random.fold_in(kk, 2), aux=refs)
        y = B16(run6, xpack, 700)
        E = np.asarray(spec_fn(jnp.asarray(y))).mean(0)
        ry = float(np.concatenate([np.asarray(resid_fn(jnp.asarray(y[i:i + 32]))).ravel()
                                   for i in range(0, len(y), 32)]).mean())
        Eh = local_hik_energy(np.asarray(y)[..., 1] * SIG, HIK0, 6.0)
        key = f'{R}|re2k-149|P2mu3'
        vals = dict(ret=E[HIK0:96].sum() / Eg[HIK0:96].sum(),
                    place=np.corrcoef(Eh.ravel(), Ehg.ravel())[0, 1],
                    lowk=E[1:5].sum() / Eg[1:5].sum(), kstar=eff_resolution(E, Eg),
                    resid_ratio=ry / rg)
        for f, vv in vals.items():
            OUT[f'{key}||{f}'] = np.float32(vv)
        OUT[f'{key}||bp'] = bp_corr(band_maps(np.asarray(y)[..., 1] * SIG), gt_maps)
        OUT[f'{key}||E'] = E.astype(np.float32)
        np.savez(OUTP, **OUT)
        print(f"  P2mu3 (2-band dial): ret={vals['ret']:.3f} place={vals['place']:.3f} "
              f"bp " + " ".join(f"{x:.3f}" for x in OUT[f'{key}||bp']), flush=True)
    jax.clear_caches()
print("BP2BAND COMPLETE", flush=True)
