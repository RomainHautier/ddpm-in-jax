"""Backfill missing metrics into an audit store from the STORED FIELDS - no inference, no TPU.
Any row whose reconstructions were saved (base_results/fields/re{R}/) gets the full metric set:
per-band retention (Eb), per-band placement (bp), aggregate placement, and the per-sample arrays
(ps_ret_paired, ps_ret_ens, ps_place, ps_mse, psEb). Rows graded by older scripts (e.g. the
calibrated-onset dial) wrote a reduced set; this makes every row comparable.

  AUDIT_RE=1000 python -m src.ddpo_ft.backfill_from_fields
"""
import os, sys, glob
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp
from viz_energy import local_hik_energy
from src.rewards import make_spectrum_fn, make_residual_loss

SIG, N, HIK0 = 4.7988, 256, 32
R = os.environ.get('AUDIT_RE', '1000')
STORE = 'base_results/re1000_audit.npz' if R == '1000' else f'base_results/regime_audit_re{R}.npz'
FDIR = f'base_results/fields/re{R}'
BANDS = [(1, 5), (5, 16), (16, 32), (32, 64), (64, 96)]
spec_fn = make_spectrum_fn(N)
fy = np.fft.fftfreq(N) * N
kmag = np.sqrt(fy[:, None] ** 2 + fy[None, :] ** 2)
gsm = np.exp(-2.0 * (np.pi * 6.0) ** 2 * ((fy[:, None] / N) ** 2 + (fy[None, :] / N) ** 2))
MASKS = [((kmag >= lo) & (kmag < hi)).astype(np.float32) for lo, hi in BANDS]


def band_maps(w):
    F = np.fft.fft2(w); out = []
    for m in MASKS:
        bp = np.real(np.fft.ifft2(F * m))
        out.append(np.real(np.fft.ifft2(np.fft.fft2(bp ** 2) * gsm)))
    return np.stack(out)


S = {k: v for k, v in np.load(STORE, allow_pickle=True).items()}
xg = np.load(f'{FDIR}/GT.npz')['x'].astype(np.float32)
E_gt_s = np.asarray(spec_fn(jnp.asarray(xg))); E_gt = E_gt_s.mean(0)
Ehg = local_hik_energy(xg[..., 1] * SIG, HIK0, 6.0)
# per-FRAME references. The spectrum helper reads index 1 of a triplet, so feeding it a
# triplet made of three copies of frame f returns frame f's spectrum with identical binning.
_as_mid = lambda x, f: np.repeat(x[..., f:f + 1], 3, axis=-1)
E_gt_f = np.stack([np.asarray(spec_fn(jnp.asarray(_as_mid(xg, f)))) for f in range(3)], 1)
Ehg_f = np.stack([local_hik_energy(xg[..., f] * SIG, HIK0, 6.0) for f in range(3)], 1)
gt_maps = band_maps(xg[..., 1] * SIG)
resid_fn = jax.jit(make_residual_loss(n=N, re=float(R), std=SIG, mean=0.0))
rg = float(np.concatenate([np.asarray(resid_fn(jnp.asarray(xg[i:i + 16]))).ravel()
                           for i in range(0, len(xg), 16)]).mean())
S[f'{R}|GT||psEb'] = np.stack([E_gt_s[:, lo:hi].sum(1) for lo, hi in BANDS], 1).astype(np.float32)
n_fixed = 0
for f in sorted(glob.glob(f'{FDIR}/*.npz')):
    name = os.path.basename(f)[:-4]
    if name in ('GT', 'LR', 'index'): continue
    row = name.replace('__', '|')
    need = [k for k in ('bp', 'Eb', 'ps_ret_paired', 'ps_place', 'ps_mse', 'psEb', 'place_norm', 'ps_bp', 'ps3_ret', 'ps3_place', 'pst_ret', 'pst_place')
            if f'{R}|{row}||{k}' not in S]
    if not need: continue
    y = np.load(f)['x'].astype(np.float32)
    E_s = np.asarray(spec_fn(jnp.asarray(y))); E = E_s.mean(0)
    ym = band_maps(y[..., 1] * SIG)
    Eh = local_hik_energy(y[..., 1] * SIG, HIK0, 6.0)
    # PER-FRAME metrics on all three frames, then the per-triplet average of them. Every
    # other metric here uses the middle frame only; these cover the whole triplet.
    E_y_f = np.stack([np.asarray(spec_fn(jnp.asarray(_as_mid(y, f)))) for f in range(3)], 1)
    Eh_f = np.stack([local_hik_energy(y[..., f] * SIG, HIK0, 6.0) for f in range(3)], 1)
    ps3_ret = (E_y_f[:, :, HIK0:96].sum(-1) / E_gt_f[:, :, HIK0:96].sum(-1))     # (n, 3)
    ps3_place = np.array([[np.corrcoef(Eh_f[i, f].ravel(), Ehg_f[i, f].ravel())[0, 1]
                           for f in range(3)] for i in range(len(y))])           # (n, 3)
    S[f'{R}|{row}||Eb'] = np.array([E[lo:hi].sum() / E_gt[lo:hi].sum() for lo, hi in BANDS], np.float32)
    S[f'{R}|{row}||bp'] = np.array([np.corrcoef(ym[i].ravel(), gt_maps[i].ravel())[0, 1]
                                    for i in range(5)], np.float32)
    # PER-TRIPLET per-band placement: one correlation per reconstructed triplet per band. The
    # pooled version above correlates the whole stack at once, so a triplet given the wrong TOTAL
    # energy drags the correlation down even when its energy is perfectly placed - it measures
    # dose error, not placement. Aggregate this one instead.
    S[f'{R}|{row}||ps_bp'] = np.array(
        [[np.corrcoef(ym[b, i].ravel(), gt_maps[b, i].ravel())[0, 1] for b in range(5)]
         for i in range(len(y))], np.float32)
    S[f'{R}|{row}||place'] = np.float32(np.corrcoef(Eh.ravel(), Ehg.ravel())[0, 1])
    # placement with each frame's amplitude divided out: isolates SPATIAL placement from the
    # frame-to-frame dose error that the globally-normalised pooled metric folds in.
    _n = lambda E: E / E.mean(axis=(1, 2), keepdims=True)
    S[f'{R}|{row}||place_norm'] = np.float32(np.corrcoef(_n(Eh).ravel(), _n(Ehg).ravel())[0, 1])
    S[f'{R}|{row}||psEb'] = np.stack([E_s[:, lo:hi].sum(1) for lo, hi in BANDS], 1).astype(np.float32)
    S[f'{R}|{row}||ps_ret_paired'] = (E_s[:, HIK0:96].sum(1) / E_gt_s[:, HIK0:96].sum(1)).astype(np.float32)
    S[f'{R}|{row}||ps_ret_ens'] = (E_s[:, HIK0:96].sum(1) / E_gt[HIK0:96].sum()).astype(np.float32)
    S[f'{R}|{row}||ps_place'] = np.array([np.corrcoef(Eh[i].ravel(), Ehg[i].ravel())[0, 1]
                                          for i in range(len(y))], np.float32)
    S[f'{R}|{row}||ps3_ret'] = ps3_ret.astype(np.float32)
    S[f'{R}|{row}||ps3_place'] = ps3_place.astype(np.float32)
    S[f'{R}|{row}||pst_ret'] = ps3_ret.mean(1).astype(np.float32)     # per-triplet average
    S[f'{R}|{row}||pst_place'] = ps3_place.mean(1).astype(np.float32)
    S[f'{R}|{row}||ps_mse'] = (np.mean((y[..., 1] - xg[..., 1]) ** 2, axis=(1, 2)) * SIG ** 2).astype(np.float32)
    S[f'{R}|{row}||ps_resid'] = (np.concatenate([np.asarray(resid_fn(jnp.asarray(y[i:i + 16]))).ravel()
                                                 for i in range(0, len(y), 16)]) / rg).astype(np.float32)
    n_fixed += 1
    print(f"  {row:<28} filled {len(need)} groups | per-frame ret "
          + "/".join(f"{np.median(ps3_ret[:, f]):.2f}" for f in range(3))
          + "  place " + "/".join(f"{np.median(ps3_place[:, f]):.3f}" for f in range(3)), flush=True)
np.savez(STORE, **S)
print(f"BACKFILL COMPLETE: {n_fixed} rows completed in {STORE}", flush=True)
