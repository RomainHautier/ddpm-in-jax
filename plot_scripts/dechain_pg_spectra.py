"""Inference configurations of a gate-reward specialist, manifold view beside the
corresponding spectra, in the dechain_manifold_spectra format. One row per target
regime. Left, single-sample endpoints of the tested configurations on the spectral
manifold (median test sample; per-config copy located by field correlation). Right,
the pool mean spectrum ratios of the same configurations in matching colors. The
selected configuration per rung is black edged on both sides.

  MODEL=pg8k JAX_PLATFORMS=cpu python plot_scripts/dechain_pg_spectra.py
  MODEL=pg4k JAX_PLATFORMS=cpu python plot_scripts/dechain_pg_spectra.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import style

WHICH = os.environ.get('MODEL', 'pg8k')
M = {'pg8k': 'pg8k-0499', 'pg4k': 'pg4k-0299'}[WHICH]
BASIS_REGS = [1000, 2000, 4000, 8000]
TGT_COL = '#b8399e'
KMAX = 127
SIG = 4.7988
CCOL = {'[150,100,50]': '#4a4e57', '[150,100,50] + gate': '#0f9e78',
        '[160]': '#8a5cd6', '[125]': '#d4770a', '[100]': '#c2402f',
        '[125] + ens. dial': '#e8a94f', '[100] + ens. dial': '#e8a94f',
        '[160] + ens. dial': '#e8a94f',
        '[125] + gated dial': '#1ba3c6', '[100] + gated dial': '#1ba3c6',
        '[160] + gated dial': '#1ba3c6'}
CHAIN = {1000: 'K100', 2000: 'K125', 4000: 'K160', 6000: 'K160'}
# best in-regime configuration per target rung, drawn as the reference to beat
BESTREF = {1000: ('base0', 'base + gate'), 2000: ('pg1k-0599', 'pg1k + gate'),
           4000: ('pg4k-0299', 'pg4k + gate'), 6000: ('pg4k-0299', 'pg4k + gate'),
           8000: ('pg4k-0299', 'pg4k + gate')}
if WHICH == 'pg8k':
    BESTREF[1000] = ('pg1k-0599', 'pg1k + gate')
BREF_COL = '#28658a'
# per target regime: list of (label, chain token, tag)
if WHICH == 'pg8k':
    TARGETS = [1000, 2000, 4000, 6000, 8000]
    SELECTED = {1000: '[100] + gated dial', 2000: '[125] + gated dial',
                4000: '[160] + gated dial', 6000: '[160] + gated dial',
                8000: '[150,100,50] + gate'}
else:
    TARGETS = [1000, 2000, 4000]
    SELECTED = {1000: '[100] + gated dial', 2000: '[125] + gated dial',
                4000: '[150,100,50] + gate'}


def cfgs(R):
    out = [('[150,100,50]', 'K3', 'mop0.2_0'), ('[150,100,50] + gate', 'K3', 'v8lowband')]
    if R in CHAIN:
        ch = CHAIN[R]
        lab = f'[{ch[1:]}]'
        out += [(lab, ch, 'mop0.2_0'), (f'{lab} + gated dial', ch, 'v8lowband')]
    return out


def box(ax):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color('black'); sp.set_linewidth(1.0)


def store(R):
    return ('base_results/re1000_audit.npz' if R == 1000
            else f'base_results/regime_audit_re{R}.npz')


def main():
    import jax.numpy as jnp
    from src.rewards import make_spectrum_fn
    style.apply(style.BASE_FS, 150)
    sf = make_spectrum_fn(256)

    def spec(x):
        return np.log(np.maximum(np.concatenate(
            [np.asarray(sf(jnp.asarray(x[i:i + 32]))) for i in range(0, len(x), 32)]
        )[:, 1:96], 1e-20))

    gts = {R: spec(np.load(f'base_results/fields/re{R}/GT.npz')['x'].astype(np.float32))
           for R in BASIS_REGS}
    X = np.concatenate(list(gts.values())); mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P = lambda A: (A - mu) @ Vt[:2].T

    k = np.arange(1, KMAX + 1)
    fig, axes = plt.subplots(len(TARGETS), 2, figsize=(14.2, 4.5 * len(TARGETS)),
                             gridspec_kw=dict(width_ratios=[1.45, 1]), squeeze=False)
    for ri, R in enumerate(TARGETS):
        axm, axs = axes[ri]
        S = np.load(store(R), allow_pickle=True); F = set(S.files)
        Eg = np.asarray(S[f'{R}|GT||E'])[1:KMAX + 1]
        Fd = f'base_results/fields/re{R}'
        gt = np.load(f'{Fd}/GT.npz')['x'].astype(np.float32)
        # background clouds
        for Rb in BASIS_REGS:
            if Rb == R: continue
            v = P(gts[Rb])
            axm.scatter(v[:, 0], v[:, 1], s=12, color=style.REGIME_COLOR[Rb],
                        alpha=.16, linewidths=0, zorder=2)
        v = P(gts[R] if R in gts else spec(gt))
        axm.scatter(v[:, 0], v[:, 1], s=20, color=TGT_COL, alpha=.65, linewidths=0,
                    zorder=3, label='target regime manifold')
        # the row's sample and its base recon
        span = np.arange(len(gt) - 80, len(gt)) if len(gt) > 80 else np.arange(len(gt))
        e = (np.abs(np.fft.rfft2(gt[span, ..., 1]))[:, 32:96, :] ** 2).sum((1, 2))
        g = span[int(np.argsort(e)[len(e) // 2])]
        pg = P(spec(gt[g:g + 1]))[0]
        axm.scatter(*pg, marker='*', s=320, color=TGT_COL,
                    zorder=9, edgecolors='black', linewidths=.8,
                    label='ground truth sample')
        rcf = f'{Fd}/recon.npz'
        if os.path.exists(rcf):
            rc = np.load(rcf)['x'].astype(np.float32)
            prc = P(spec(rc[g:g + 1]))[0]
            axm.scatter(*prc, marker='s', s=70, color='#d4770a', zorder=8,
                        edgecolors='black', linewidths=.7,
                        label='base DDIM reconstruction')
        flat = (gt[g, ..., 1] - gt[g, ..., 1].mean()).ravel()
        pts = []
        for lab, ch, tag in cfgs(R):
            f = f'{Fd}/{M}__{ch}__{tag}.npz'
            key = f'{R}|{M}|{ch}|{tag}||E'
            col = CCOL[lab]
            if os.path.exists(f):
                x = np.load(f)['x'].astype(np.float32)
                wv = x[..., 1].reshape(len(x), -1)
                wv = wv - wv.mean(1, keepdims=True)
                j = int(np.argmax(wv @ flat))
                p_ = P(spec(x[j:j + 1]))[0]
                pts.append(p_)
                sel = SELECTED.get(R) == lab
                axm.scatter(*p_, marker='D', s=120 if sel else 85, color=col,
                            zorder=7 if sel else 6,
                            edgecolors='black' if sel else 'white',
                            linewidths=1.2 if sel else .6,
                            label=lab + (' (selected)' if sel else ''))
            if key in F:
                E = np.asarray(S[key])[1:KMAX + 1]
                sel = SELECTED.get(R) == lab
                axs.semilogx(k, E / Eg, '-', color=col, lw=2.3 if sel else 1.7,
                             label=lab + (' (selected)' if sel else ''),
                             zorder=5 if sel else 4,
                             path_effects=None)
        bm, blab = BESTREF.get(R, (None, None))
        if bm and bm != M:
            bf = f'{Fd}/{bm}__K3__v8lowband.npz'
            bkey = f'{R}|{bm}|K3|v8lowband||E'
            _p4 = bm == 'pg4k-0299'
            _mk, _col = ('v', '#5e35b1') if _p4 else ('o', BREF_COL)
            if os.path.exists(bf):
                x = np.load(bf)['x'].astype(np.float32)
                wv = x[..., 1].reshape(len(x), -1)
                wv = wv - wv.mean(1, keepdims=True)
                j = int(np.argmax(wv @ flat))
                p_ = P(spec(x[j:j + 1]))[0]
                axm.scatter(*p_, marker=_mk, s=95, color=_col, zorder=7,
                            edgecolors='black', linewidths=.9, label=blab)
            if bkey in F:
                E = np.asarray(S[bkey])[1:KMAX + 1]
                axs.semilogx(k, E / Eg, '--', color=_col, lw=1.7,
                             label=blab, zorder=5)
        # the pg4k specialist's own BEST configuration in the pg8k figure, where it
        # differs from the reference already drawn (its dechained configs at 1000/2000)
        if WHICH == 'pg8k' and R in (1000, 2000):
            p4ch = {1000: 'K100', 2000: 'K125'}[R]
            p4lab = f'pg4k [{p4ch[1:]}] + gated dial'
            p4f = f'{Fd}/pg4k-0299__{p4ch}__v8lowband.npz'
            p4key = f'{R}|pg4k-0299|{p4ch}|v8lowband||E'
            if os.path.exists(p4f):
                x = np.load(p4f)['x'].astype(np.float32)
                wv = x[..., 1].reshape(len(x), -1)
                wv = wv - wv.mean(1, keepdims=True)
                j = int(np.argmax(wv @ flat))
                p_ = P(spec(x[j:j + 1]))[0]
                axm.scatter(*p_, marker='v', s=95, color='#5e35b1', zorder=7,
                            edgecolors='black', linewidths=.9, label=p4lab)
            if p4key in F:
                E = np.asarray(S[p4key])[1:KMAX + 1]
                axs.semilogx(k, E / Eg, '-.', color='#5e35b1', lw=1.7,
                             label=p4lab, zorder=5)
        if len(pts) > 1:
            pts = np.array(pts)
            o = np.argsort(pts[:, 0])
            axm.plot(pts[o, 0], pts[o, 1], ls=':', color='#5d645d', lw=1.1, alpha=.7,
                     zorder=4)
        style.shade_bands(axs)
        axs.axhline(1, color=style.GT_COLOR, lw=1.3, zorder=6)
        axs.set(yscale='log', ylim=(0.05, 6.5))
        axs.set_yticks([0.1, 0.3, 1, 2, 4, 6]); axs.set_yticklabels(['0.1', '0.3', '1', '2', '4', '6'])
        axs.yaxis.set_minor_formatter(plt.NullFormatter())
        axs.axvline(96, color='#9aa198', lw=.8, ls=':', zorder=1)
        axs.legend(fontsize=style.LEG_FS - 2, loc='lower left', framealpha=.85)
        axm.set_ylabel(f'Re = {R}\nPC2', fontsize=style.TITLE_FS - 1)
        axm.grid(alpha=.2); box(axm); box(axs)
        axm.legend(fontsize=style.LEG_FS - 2,
                   loc='lower right' if R == 1000 else 'lower left', framealpha=.9)
        if ri == len(TARGETS) - 1:
            axm.set_xlabel('PC1'); axs.set_xlabel('wavenumber $k$')
        if ri == 0:
            axm.set_title('manifold endpoints, one median test sample',
                          fontsize=style.TITLE_FS)
            axs.set_title('pool $E(k)\\,/\\,E_{\\mathrm{GT}}(k)$',
                          fontsize=style.TITLE_FS)
    fig.tight_layout(rect=(0, 0, 1, .98))
    for d_, ext in (('docs/figs_overleaf/gated', 'pdf'), ('plotting/figs', 'png')):
        os.makedirs(d_, exist_ok=True)
        p_ = f'{d_}/dechain_{WHICH}_spectra.{ext}'; fig.savefig(p_)
        print('written', p_)


if __name__ == '__main__':
    main()
