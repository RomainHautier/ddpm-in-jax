"""The pure-gate specialists on the spectral manifold, one row per regime (8000 down
to 1000), manifold clouds beside the matching mean spectrum ratios. Per row: the
Re=8000 specialist unguided and gated, the Re=4000 specialist gated; the Re=1000 row
adds the home fine-tune with and without the gate.

  JAX_PLATFORMS=cpu python plot_scripts/manifold_pg_rows.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import style

BASIS_REGS = [1000, 2000, 4000, 8000]
ROWS = [1000, 2000, 4000, 6000, 8000]
TGT_COL = '#b8399e'
KMAX = 127
# per regime: (model, chain, tag, label, colour)
CFG = {R: [('pg8k-0499', 'K3', 'mop0.2_0',  'pg8k, unguided', '#b79ce8'),
           ('pg8k-0499', 'K3', 'v8lowband', 'pg8k + gate',    '#8a5cd6'),
           ('pg4k-0299', 'K3', 'v8lowband', 'pg4k + gate',    '#c22f4f')]
       for R in ROWS}
CFG[1000] += [('pg1k-0599', 'K3', 'mop0.2_0',  'pg1k, unguided', '#7fb3d3'),
              ('pg1k-0599', 'K3', 'v8lowband', 'pg1k + gate',    '#28658a')]
CFG[4000] = [('pg8k-0499', 'K3', 'mop0.2_0',  'pg8k, unguided', '#b79ce8'),
             ('pg8k-0499', 'K3', 'v8lowband', 'pg8k + gate',    '#8a5cd6'),
             ('pg4k-0299', 'K3', 'mop0.2_0',  'pg4k, unguided', '#e8a3bd'),
             ('pg4k-0299', 'K3', 'v8lowband', 'pg4k + gate',    '#c22f4f'),
             ('pg1k-0599', 'K3', 'v8lowband', 'pg1k + gate',    '#28658a')]


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
           for R in sorted(set(BASIS_REGS) | set(ROWS))}
    X = np.concatenate(list(gts.values())); mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P = lambda A: (A - mu) @ Vt[:2].T

    k = np.arange(1, KMAX + 1)
    fig, axes = plt.subplots(len(ROWS), 2, figsize=(13.6, 4.8 * len(ROWS)),
                             gridspec_kw=dict(width_ratios=[1.35, 1]),
                             constrained_layout=True)
    for ri, R in enumerate(ROWS):
        axm, axs = axes[ri]
        S = np.load(store(R), allow_pickle=True); F = set(S.files)
        Eg = np.asarray(S[f'{R}|GT||E'])[1:KMAX + 1]
        for Rb in BASIS_REGS:
            if Rb == R: continue
            v = P(gts[Rb])
            axm.scatter(v[:, 0], v[:, 1], s=14, color=style.REGIME_COLOR[Rb],
                        alpha=.22, linewidths=0, zorder=2)
        v = P(gts[R])
        axm.scatter(v[:, 0], v[:, 1], s=24, color=style.REGIME_COLOR.get(R, '#b8399e'),
                    alpha=.75, linewidths=0, zorder=3, label=f'Re={R} truth')
        style.shade_bands(axs)
        axs.axhline(1, color=style.GT_COLOR, lw=1.3, zorder=6)
        for m, ch, tag, lab, col in CFG[R]:
            f = f'base_results/fields/re{R}/{m}__{ch}__{tag}.npz'
            if os.path.exists(f):
                v = P(spec(np.load(f)['x'].astype(np.float32)))
                gated = 'v8lowband' in tag
                axm.scatter(v[:, 0], v[:, 1], s=42 if gated else 30, marker='D',
                            color=col, alpha=.9 if gated else .55,
                            linewidths=.7, edgecolors='black' if gated else 'white',
                            zorder=6 if gated else 5, label=lab)
            key = f'{R}|{m}|{ch}|{tag}||E'
            if key in F:
                E = np.asarray(S[key])[1:KMAX + 1]
                axs.semilogx(k, E / Eg, '-', color=col, lw=1.8, label=lab, zorder=4)
        axs.set(yscale='log', ylim=(0.015, 2.6))
        axs.set_yticks([0.03, 0.1, 0.3, 1, 2])
        axs.set_yticklabels(['0.03', '0.1', '0.3', '1', '2'])
        axs.yaxis.set_minor_formatter(plt.NullFormatter())
        axs.axvline(96, color='#9aa198', lw=.8, ls=':', zorder=1)
        axm.set_ylabel(f'Re = {R}\nPC2', fontsize=style.TITLE_FS - 1)
        axm.grid(alpha=.2)
        axm.legend(fontsize=style.LEG_FS - 2, loc='upper left', framealpha=.85)
        if ri == len(ROWS) - 1:
            axm.set_xlabel('PC1'); axs.set_xlabel('wavenumber $k$')
        if ri == 0:
            axm.set_title('spectral manifold', fontsize=style.TITLE_FS)
            axs.set_title('$E(k)\\,/\\,E_{\\mathrm{GT}}(k)$', fontsize=style.TITLE_FS)
    fig.suptitle('Pure-gate specialists across the ladder, manifold clouds and spectra',
                 fontsize=style.SUP_FS)
    for d_, ext in (('docs/figs_overleaf/gated', 'pdf'), ('plotting/figs', 'png')):
        os.makedirs(d_, exist_ok=True)
        p_ = f'{d_}/manifold_pg_rows.{ext}'; fig.savefig(p_)
        print('written', p_)


if __name__ == '__main__':
    main()
