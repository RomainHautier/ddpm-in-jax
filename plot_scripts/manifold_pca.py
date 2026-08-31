"""Where do the regimes sit, and where do the models land?

Each sample is embedded by its log energy spectrum (k=1..96) - the coordinate the whole study is
about - and projected onto the first two principal components of the GROUND TRUTH samples pooled
over the regimes shown. Ground truth forms the reference manifold; model generations are projected
into the same basis, so the distance from a regime's truth cloud is readable.

  JAX_PLATFORMS=cpu python plot_scripts/manifold_pca.py [--regimes 1000,2000,8000]
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import style

KLO, KHI = 1, 96
CONFIG = dict(regimes=[1000, 2000, 8000], fig_w=11.4, fig_h=5.0,
              fontsize=style.BASE_FS, dpi=150,
              outdir='docs/figs_overleaf', pngdir='plotting/figs')
GEN = [('base0', 'mop0.2_0', 'base', 'o'),
       ('mt1k-0499', 'mop0.2_0', 'Re=1000 fine-tune', 's')]


def spectra(R, key):
    """per-sample spectra from the stored fields (n, 96)"""
    p = f'base_results/fields/re{R}/{key}.npz'
    if not os.path.exists(p): return None
    import jax.numpy as jnp
    from src.rewards import make_spectrum_fn
    x = np.load(p)['x'].astype(np.float32)
    sf = make_spectrum_fn(256)
    out = np.concatenate([np.asarray(sf(jnp.asarray(x[i:i + 32]))) for i in range(0, len(x), 32)])
    return np.log(np.maximum(out[:, KLO:KHI], 1e-20))


def make(c):
    style.apply(c['fontsize'], c['dpi'])
    regs = c['regimes']
    gts = {R: spectra(R, 'GT') for R in regs}
    gts = {R: v for R, v in gts.items() if v is not None}
    X = np.concatenate([gts[R] for R in gts])
    mu = X.mean(0); Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    proj = lambda A: (A - mu) @ Vt[:2].T
    fig, ax = plt.subplots(1, 2, figsize=(c['fig_w'], c['fig_h']), constrained_layout=True)

    a = ax[0]
    for R in gts:
        p = proj(gts[R])
        a.scatter(p[:, 0], p[:, 1], s=16, color=style.REGIME_COLOR.get(R, '#5d645d'),
                  alpha=.65, lw=0, label=f'Re={R} ground truth')
    for m, sg, lab, mk in GEN:
        for R in gts:
            sp = spectra(R, f'{m}__K3__{sg}')
            if sp is None: continue
            p = proj(sp)
            a.scatter(p[:, 0], p[:, 1], s=20, marker=mk, facecolors='none',
                      edgecolors=style.MODEL_COLOR[m], linewidths=.8, alpha=.8,
                      label=f'{lab} generations' if R == regs[0] else None)
    a.set_xlabel(f'PC1  ({var[0]*100:.0f}% of variance)')
    a.set_ylabel(f'PC2  ({var[1]*100:.0f}%)')
    a.set_title('spectral manifold: ground truth (filled) and generations (open)',
                fontsize=style.TITLE_FS)
    a.legend(fontsize=style.LEG_FS - 1.5, loc='best')

    # ---- distance from each regime's truth cloud, in the full 95-d spectral space ----
    b = ax[1]
    cent = {R: gts[R].mean(0) for R in gts}
    spread = {R: np.median(np.linalg.norm(gts[R] - cent[R], axis=1)) for R in gts}
    labs, vals, cols = [], [], []
    for m, sg, lab, mk in GEN:
        for R in gts:
            sp = spectra(R, f'{m}__K3__{sg}')
            if sp is None: continue
            d = np.median(np.linalg.norm(sp - cent[R], axis=1)) / spread[R]
            labs.append(f'{lab}\nat Re={R}'); vals.append(d); cols.append(style.MODEL_COLOR[m])
    b.bar(range(len(vals)), vals, color=cols, alpha=.9)
    b.axhline(1, color=style.GT_COLOR, lw=1.6)
    b.text(len(vals) - .4, 1.04, "truth's own spread", fontsize=style.LEG_FS - 1.5,
           ha='right', color='#5d645d')
    b.set_xticks(range(len(labs))); b.set_xticklabels(labs, fontsize=style.LEG_FS - 2)
    b.set_ylabel('distance to that regime\'s truth centroid\n(in units of the truth\'s own spread)')
    b.set_title('how far off-manifold the generations are', fontsize=style.TITLE_FS)
    fig.suptitle('Regimes and generations in log-spectral coordinates', fontsize=style.SUP_FS)
    for d_, ext in ((c['outdir'], 'pdf'), (c['pngdir'], 'png')):
        os.makedirs(d_, exist_ok=True)
        q = f'{d_}/manifold_pca.{ext}'; fig.savefig(q); print('  written', q)
    print(f'\n  variance explained: PC1 {var[0]*100:.1f}%  PC2 {var[1]*100:.1f}%  '
          f'PC3 {var[2]*100:.1f}%')
    for l, v in zip(labs, vals): print(f'   {l.replace(chr(10), " "):34s} {v:6.2f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--regimes'); a = ap.parse_args()
    c = dict(CONFIG)
    if a.regimes: c['regimes'] = [int(x) for x in a.regimes.split(',')]
    make(c)
