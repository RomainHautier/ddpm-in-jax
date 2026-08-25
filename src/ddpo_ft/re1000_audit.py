"""RE=1000 FULL AUDIT (user 2026-08-25) on sequences NO model ever trained on: 34-39 of the
40-seed file (base pretrain 0-31, base val 32-35 with r1k fine-tuned on 32-33, base test 36-39).
Rows: raw LR, base-DDIM recon (t=100/20), then models x strategies at K3, plus the base's
unsteered depth ladder K2/K4/K5. Per cell: per-band retention (5 bands), per-band placement,
aggregate hi-k placement, MSE, residual ratio, lowk, spectrum. Keys '1000|{row}||{field}' in
base_results/re1000_audit.npz. Resume-aware; models read from MODELS env-extension
(EXTRA_MODELS name:path[:key]) so pr1k joins once trained.
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

MEAN, SIG, N, HIK0, R = 0.0, 4.7988, 256, 32, 1000
GT = 'flow-data/kf_2d_re1000_256_40seed.npy'
SEQS, PER = [34, 35, 36, 37, 38, 39], 20
BANDS = [(1, 5), (5, 16), (16, 32), (32, 64), (64, 96)]
LADDER = {'K2x50': ([100, 75], 50), 'K3x86': ([150, 100, 50], 86),
          'K4x110': ([200, 150, 100, 50], 110), 'K5x140': ([250, 200, 150, 100, 50], 140)}
STRATS = {'none': (0, 0, 0, 1), 'residual': (3, 0, 0, 1), 'reward': (0, 8, 0, 1),
          'placement': (0, 0, 3, 1), 'all3': (3, 8, 3, 1),
          'rewardv2': (0, 8, 0, 2), 'all3v2': (3, 8, 3, 2)}
ddpm, base_params, _ = build_base_ddpm(); ab = ddpm.alpha_bar
MODELS = {'base0': base_params,
          'r1k-449': pickle.load(open('monitoring/ddpo_re1000_newpool_ckpts/ddpo_re1000_iter0449.pkl', 'rb'))['params'],
          'st1k-599': pickle.load(open('monitoring/ddpo_re1000_steeredtrain_ckpts/ddpo_re1000_iter0599.pkl', 'rb'))['params']}
for _spec in filter(None, os.environ.get('EXTRA_MODELS', '').split(',')):
    _p = _spec.split(':'); MODELS[_p[0]] = pickle.load(open(_p[1], 'rb'))[_p[2] if len(_p) > 2 else 'params']
spec_fn = make_spectrum_fn(N)
ddim20 = build_ddim_denoiser(ddpm.unet, ab, 100, 20)
_sa, _s1 = float(jnp.sqrt(ab[100])), float(jnp.sqrt(1.0 - ab[100]))
B16 = partial(pbatched, per_dev=16)

fy = np.fft.fftfreq(N) * N
kmag = np.sqrt(fy[:, None] ** 2 + fy[None, :] ** 2)
gsm = np.exp(-2.0 * (np.pi * 6.0) ** 2 * ((fy[:, None] / N) ** 2 + (fy[None, :] / N) ** 2))
MASKS = [((kmag >= lo) & (kmag < hi)).astype(np.float32) for lo, hi in BANDS]
gsm_j = jnp.asarray(gsm); MASKS_J = [jnp.asarray(m) for m in MASKS]


def band_maps(w):
    F = np.fft.fft2(w); out = []
    for m in MASKS:
        bp = np.real(np.fft.ifft2(F * m))
        out.append(np.real(np.fft.ifft2(np.fft.fft2(bp ** 2) * gsm)))
    return np.stack(out)


def nmaps(w, idxs=(2, 3)):
    F = jnp.fft.fft2(w); out = []
    for i in idxs:
        bp = jnp.real(jnp.fft.ifft2(F * MASKS_J[i]))
        e = jnp.real(jnp.fft.ifft2(jnp.fft.fft2(bp ** 2) * gsm_j))
        out.append(e / (jnp.mean(e, axis=(-2, -1), keepdims=True) + 1e-12))
    return out


d = np.load('base_results/regime_stats_re1000_measured_train.npz'); stats = {k: d[k] for k in d.files}
lref = stats.get('log_spec_ref')
d1 = make_spectrum_distance(stats['spec_ref'], kband=(1, 96), n=N, log_ref=lref)
d2 = make_spectrum_distance(stats['spec_ref'], kband=(32, 96), n=N, log_ref=lref)
dm = make_spectrum_distance(stats['spec_ref'], kband=(16, 32), n=N, log_ref=lref)
dose1 = jax.jit(jax.grad(lambda x: jnp.sum(0.5 * d1(x) + 3.0 * d2(x))))
dose2 = jax.jit(jax.grad(lambda x: jnp.sum(0.5 * d1(x) + 3.0 * dm(x) + 3.0 * d2(x))))
dx_pde = make_dx_func(n=N, re=float(R), std=SIG, mean=MEAN)
resid_fn = jax.jit(make_residual_loss(n=N, re=float(R), std=SIG, mean=0.0))

xg, xl = [], []
for s in SEQS:
    q = load_sequence(GT, s)
    g = build_triplets(q, MEAN, SIG); l = build_triplets(grid_downsample_degrade(q, 4), MEAN, SIG)
    i2 = np.linspace(0, len(g) - 1, PER).astype(int)
    xg.append(g[i2]); xl.append(l[i2])
xg, xl = np.concatenate(xg), np.concatenate(xl)
E_gt = np.asarray(spec_fn(jnp.asarray(xg))).mean(0)
Ehg = local_hik_energy(xg[..., 1] * SIG, HIK0, 6.0)
gt_maps = band_maps(xg[..., 1] * SIG)
rg = float(np.concatenate([np.asarray(resid_fn(jnp.asarray(xg[i:i + 32]))).ravel()
                           for i in range(0, len(xg), 32)]).mean())
recon = np.asarray(B16(lambda xb, kk: ddim20(base_params, _sa * xb + _s1 * jax.random.normal(
    jax.random.fold_in(kk, 1), xb.shape)), xl, 500))

OUT = {}
OUTP = 'base_results/re1000_audit.npz'
if os.path.exists(OUTP):
    old = np.load(OUTP, allow_pickle=True); OUT = {k: old[k] for k in old.files}
OUT['1000|GT||E'] = E_gt.astype(np.float32)
FDIR = 'base_results/fields/re1000'; os.makedirs(FDIR, exist_ok=True)
def save_fields(name, y):
    """persist the actual reconstructions (float16, compressed) so any later analysis on the
    held-out samples runs offline from stored data - no re-inference (user 2026-08-25)"""
    np.savez_compressed(f"{FDIR}/{name.replace('|', '__')}.npz", x=np.asarray(y, np.float16))
if not os.path.exists(f'{FDIR}/GT.npz'):
    save_fields('GT', xg); save_fields('LR', xl); save_fields('recon', recon)
    np.savez(f'{FDIR}/index.npz', seqs=np.array(SEQS), per=PER, note='seqs 34-39 of kf_2d_re1000_256_40seed; middle frame = channel 1; multiply by SIG=4.7988 for physical vorticity')


FORCE = set(filter(None, os.environ.get('FORCE_ROWS', '').split(',')))   # prefixes to regrade
Eg_s = np.asarray(spec_fn(jnp.asarray(xg)))                                    # per-sample GT spectra
OUT['1000|GT||psEb'] = np.stack([Eg_s[:, lo:hi].sum(1) for lo, hi in BANDS], 1).astype(np.float32)


def grade(y, row):
    y = np.asarray(y)
    if row not in ('LR', 'recon'): save_fields(row, y)
    Es = np.asarray(spec_fn(jnp.asarray(y)))                                   # per-sample spectra
    E = Es.mean(0)
    # per-sample validation statistics (user 2026-08-25): paired and ensemble-relative
    OUT[f'1000|{row}||psEb'] = np.stack([Es[:, lo:hi].sum(1) for lo, hi in BANDS], 1).astype(np.float32)
    OUT[f'1000|{row}||ps_ret_paired'] = (Es[:, HIK0:96].sum(1) / Eg_s[:, HIK0:96].sum(1)).astype(np.float32)
    OUT[f'1000|{row}||ps_ret_ens'] = (Es[:, HIK0:96].sum(1) / E_gt[HIK0:96].sum()).astype(np.float32)
    _Eh = local_hik_energy(y[..., 1] * SIG, HIK0, 6.0)
    OUT[f'1000|{row}||ps_place'] = np.array([np.corrcoef(_Eh[i].ravel(), Ehg[i].ravel())[0, 1]
                                             for i in range(len(y))], np.float32)
    OUT[f'1000|{row}||ps_mse'] = (np.mean((y[..., 1] - xg[..., 1]) ** 2, axis=(1, 2)) * SIG ** 2).astype(np.float32)
    OUT[f'1000|{row}||ps_resid'] = np.concatenate([np.asarray(resid_fn(jnp.asarray(y[i:i + 32]))).ravel()
                                                   for i in range(0, len(y), 32)]).astype(np.float32) / rg
    ry = float(np.concatenate([np.asarray(resid_fn(jnp.asarray(y[i:i + 32]))).ravel()
                               for i in range(0, len(y), 32)]).mean())
    Eh = local_hik_energy(y[..., 1] * SIG, HIK0, 6.0)
    ym = band_maps(y[..., 1] * SIG)
    vals = dict(ret=E[HIK0:96].sum() / E_gt[HIK0:96].sum(),
                place=np.corrcoef(Eh.ravel(), Ehg.ravel())[0, 1],
                lowk=E[1:5].sum() / E_gt[1:5].sum(), kstar=eff_resolution(E, E_gt),
                resid_ratio=ry / rg, mse=np.mean((y[..., 1] - xg[..., 1]) ** 2) * SIG ** 2)
    for f, vv in vals.items(): OUT[f'1000|{row}||{f}'] = np.float32(vv)
    OUT[f'1000|{row}||E'] = E.astype(np.float32)
    OUT[f'1000|{row}||Eb'] = np.array([E[lo:hi].sum() / E_gt[lo:hi].sum() for lo, hi in BANDS], np.float32)
    OUT[f'1000|{row}||bp'] = np.array([np.corrcoef(ym[i].ravel(), gt_maps[i].ravel())[0, 1]
                                       for i in range(5)], np.float32)
    np.savez(OUTP, **OUT)
    print(f"  {row:<26} ret={vals['ret']:.3f} place={vals['place']:.3f} mse={vals['mse']:.2f} "
          f"resid={vals['resid_ratio']:.2f} Eb=" + " ".join(f"{x:.2f}" for x in OUT[f'1000|{row}||Eb']), flush=True)


for row, arr in (('LR', xl), ('recon', recon)):
    if f'1000|{row}||ret' not in OUT or any(row.startswith(f) for f in FORCE): grade(arr, row)

xpack = np.concatenate([recon, xl], axis=-1)
for nm, P in MODELS.items():
    for sg, (lp, ls, mu, ver) in STRATS.items():
        row = f'{nm}|K3|{sg}'
        if f'1000|{row}||ret' in OUT and not any(row.startswith(f) for f in FORCE): continue
        dose = dose2 if ver == 2 else dose1
        def ploss(x, refs):
            ms = nmaps(x[..., 1] * SIG)
            return sum(jnp.sum((m - jax.lax.stop_gradient(r)) ** 2) for m, r in zip(ms, refs)) / (N * N)
        pgrad = jax.grad(ploss)
        def dx(x, aux, _lp=lp, _ls=ls, _mu=mu, _dose=dose):
            g = (_lp / 3.0) * dx_pde(x) + (_ls / 3.0) * _dose(x)
            if _mu > 0: g = g + (_mu / 3.0) * pgrad(x, aux)
            return g
        lam = 3.0 if (lp + ls + mu) > 0 else 0.0
        starts, steps = LADDER['K3x86']
        sa3, s13 = float(jnp.sqrt(ab[starts[0]])), float(jnp.sqrt(1.0 - ab[starts[0]]))
        smp = make_kchain_ddim_sampler(ddpm.unet, ab, starts, steps, dx, lam, temp=0.30,
                                       jit=False, aux_dx=True)
        def run6(xb6, kk, _P=P, _smp=smp, _sa=sa3, _s1=s13):
            rc, lr = xb6[..., :3], xb6[..., 3:]
            return _smp(_P, _sa * rc + _s1 * jax.random.normal(jax.random.fold_in(kk, 1), rc.shape),
                        jax.random.fold_in(kk, 2), aux=tuple(nmaps(lr[..., 1] * SIG)))
        grade(B16(run6, xpack, 700), row)
    jax.clear_caches()
# base depth ladder, unsteered (lam=0)
for cfg in ('K2x50', 'K4x110', 'K5x140'):
    row = f'base0|{cfg}|none'
    if f'1000|{row}||ret' in OUT and not any(row.startswith(f) for f in FORCE): continue
    starts, steps = LADDER[cfg]
    sa, s1 = float(jnp.sqrt(ab[starts[0]])), float(jnp.sqrt(1.0 - ab[starts[0]]))
    smp = make_kchain_ddim_sampler(ddpm.unet, ab, starts, steps, dx_pde, 0.0, temp=0.30)
    grade(B16(lambda xb, kk: smp(base_params, sa * xb + s1 * jax.random.normal(
        jax.random.fold_in(kk, 1), xb.shape), jax.random.fold_in(kk, 2)), recon, 700), row)
    jax.clear_caches()
print("RE1000 AUDIT COMPLETE", flush=True)
