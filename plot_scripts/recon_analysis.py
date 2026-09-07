"""Reconstruction analysis at one OOD regime, in a single figure.

Top row: one representative sample (median GT fine-band energy) reconstructed by the base model,
the Re=1000 finetune, and the regime's own finetune, beside the ground truth.
Bottom row: how the ensemble energy deficit is addressed (ratio to GT per shell) and where the
energy lands (per-triplet placement correlation per shell), with the LR input and deterministic
base reconstruction as floors.

  JAX_PLATFORMS=cpu python plot_scripts/recon_analysis.py --re 2000
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import style

SIG, N = 4.7988, 256
HOME = {2000: ('mt2k-0599', 'Re=2000 finetune'), 8000: ('r8kp02-0599', 'Re=8000 finetune')}
CONFIG = dict(dk=4, kmax=100, sigma=6.0, dpi=170,
              outdir='docs/figs_overleaf', pngdir='plotting/figs')


def load(R, name):
    p = f'base_results/fields/re{R}/{name}.npz'
    return np.load(p)['x'].astype(np.float32) if os.path.exists(p) else None


def align(R, x, n):
    if x is None or len(x) == n: return x
    S = np.load(f'base_results/regime_audit_re{R}.npz', allow_pickle=True)
    m = np.asarray(S[f'{R}|EVAL||seq']) >= 12
    assert m.sum() == n, f'mask {m.sum()} != fields {n}'
    return x[m]


fy = np.fft.fftfreq(N) * N
KMAG = np.sqrt(fy[:, None] ** 2 + fy[None, :] ** 2)


def shell_maps(w, edges, gsm):
    F = np.fft.fft2(w)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        bp = np.real(np.fft.ifft2(F * ((KMAG >= lo) & (KMAG < hi)).astype(np.float32)))
        out.append(np.real(np.fft.ifft2(np.fft.fft2(bp ** 2) * gsm)))
    return np.stack(out)


def spectra(x):
    import jax.numpy as jnp
    from src.rewards import make_spectrum_fn
    sf = make_spectrum_fn(N)
    return np.concatenate([np.asarray(sf(jnp.asarray(x[i:i + 32]))) for i in range(0, len(x), 32)])


def make(R, c):
    style.apply(style.BASE_FS, c['dpi'])
    hm, hlab = HOME[R]
    rows = [('base0__K3__mop0.2_0', 'base', style.MODEL_COLOR['base0']),
            ('mt1k-0499__K3__mop0.2_0', 'Re=1000 finetune', style.MODEL_COLOR['mt1k-0499']),
            (f'{hm}__K3__mop0.2_0', hlab, style.MODEL_COLOR[hm])]
    fields = {lab: load(R, n) for n, lab, _ in rows}
    n = len(next(iter(fields.values())))
    gt = align(R, load(R, 'GT'), n); lr = align(R, load(R, 'LR'), n)
    rc = align(R, load(R, 'recon'), n)
    _c = np.median([np.corrcoef(gt[j, ..., 1].ravel(), fields['base'][j, ..., 1].ravel())[0, 1]
                    for j in range(12)])
    assert _c > 0.5, f'GT misaligned (corr {_c:.2f})'
    print(f'  Re={R}: {n} triplets, alignment corr {_c:.3f}')

    fig = plt.figure(figsize=(13.4, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1])
    Eg = spectra(gt)
    i = int(np.argsort(Eg[:, 32:96].sum(1))[n // 2])
    v = float(np.percentile(np.abs(gt[i, ..., 1] * SIG), 99.5))
    for j, (lab, f) in enumerate([('ground truth', gt)] + [(l, fields[l]) for _, l, _ in rows]):
        a = fig.add_subplot(gs[0, j])
        a.imshow(f[i, ..., 1] * SIG, cmap='RdBu_r', vmin=-v, vmax=v, interpolation='nearest')
        a.set_title(lab, fontsize=style.TITLE_FS); a.set_xticks([]); a.set_yticks([])

    kk = np.arange(1, 97)
    a = fig.add_subplot(gs[1, :2]); style.shade_bands(a)
    a.axhline(1, color=style.GT_COLOR, lw=1.5, zorder=6)
    a.semilogx(kk, spectra(lr).mean(0)[1:97] / Eg.mean(0)[1:97], ':', color='#b8399e',
               lw=1.7, label='LR input')
    for nname, lab, col in rows:
        a.semilogx(kk, spectra(fields[lab]).mean(0)[1:97] / Eg.mean(0)[1:97], '-', color=col,
                   lw=1.8, label=lab)
    a.set(yscale='log', ylim=(0.02, 2.3), xlabel='wavenumber $k$',
          ylabel='$E(k)/E_{\\mathrm{GT}}(k)$')
    a.set_title('the energy deficit, and how far each model closes it', fontsize=style.TITLE_FS)
    a.legend(fontsize=style.LEG_FS - 1, loc='lower left')

    edges = np.arange(1, c['kmax'] + 1, c['dk']); cent = edges[:-1] + c['dk'] / 2
    gsm = np.exp(-2.0 * (np.pi * c['sigma']) ** 2 * ((fy[:, None] / N) ** 2 + (fy[None, :] / N) ** 2))
    G = shell_maps(gt[..., 1] * SIG, edges, gsm)
    a = fig.add_subplot(gs[1, 2:]); style.shade_bands(a)
    for w, lab, col, ls in ([(lr[..., 1] * SIG, 'LR input', '#b8399e', ':'),
                             (rc[..., 1] * SIG, 'base recon (input)', '#d4770a', ':')]
                            + [(fields[l][..., 1] * SIG, l, col, '-') for _, l, col in rows]):
        Y = shell_maps(w, edges, gsm)
        med = np.median([[np.corrcoef(Y[b, j].ravel(), G[b, j].ravel())[0, 1]
                          for j in range(n)] for b in range(len(cent))], axis=1)
        a.plot(cent, med, ls, color=col, lw=1.8, label=lab)
    a.set(ylim=(0, 1.02), xlabel='wavenumber $k$', ylabel='placement, per triplet (median)')
    a.set_title('and whether it lands in the right place', fontsize=style.TITLE_FS)
    a.axvline(32, color='#5d645d', lw=1, ls='--')
    a.legend(fontsize=style.LEG_FS - 1, loc='lower left')
    fig.suptitle(f'Reconstruction at Re={R}: one sample, the ensemble deficit, and placement '
                 f'(all models unguided, triplet {i})', fontsize=style.SUP_FS)
    for d, ext in ((c['outdir'], 'pdf'), (c['pngdir'], 'png')):
        os.makedirs(d, exist_ok=True)
        p = f'{d}/recon_analysis_re{R}.{ext}'; fig.savefig(p); print('  written', p)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--re', type=int, default=2000)
    a = ap.parse_args(); make(a.re, dict(CONFIG))
