"""Where do the standard reconstructions start, regime by regime?

Projects the full base-DDIM reconstruction pool of each regime into the canonical
four-regime log-spectral basis, beside the truth clouds. The question on display:
do the chain starting points depend on the regime of the input, or do all standard
reconstructions begin from the same neighbourhood of the manifold?

  JAX_PLATFORMS=cpu python plot_scripts/manifold_recon_starts.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import style

REGS = [1000, 2000, 4000, 8000]


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
           for R in REGS}
    X = np.concatenate(list(gts.values())); mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P = lambda A: (A - mu) @ Vt[:2].T

    fig, ax = plt.subplots(figsize=(10.6, 5.6), constrained_layout=True)
    for R in REGS:
        v = P(gts[R])
        ax.scatter(v[:, 0], v[:, 1], s=16, color=style.REGIME_COLOR[R], alpha=.35,
                   linewidths=0, zorder=2, label=f'Re={R} truth')
    for R in REGS:
        rc = P(spec(np.load(f'base_results/fields/re{R}/recon.npz')['x']
                    .astype(np.float32)))
        col = style.vivid(style.REGIME_COLOR[R])
        ax.scatter(rc[:, 0], rc[:, 1], s=30, marker='s', facecolors='none',
                   edgecolors=col, linewidths=1.1, alpha=.8, zorder=4,
                   label=f'Re={R} standard recon')
        g = P(gts[R])
        print(f'Re={R}: recon PC1 median {np.median(rc[:,0]):6.2f} '
              f'(10-90% {np.percentile(rc[:,0],10):.1f}..{np.percentile(rc[:,0],90):.1f})   '
              f'GT median {np.median(g[:,0]):6.2f}   gap {np.median(g[:,0])-np.median(rc[:,0]):.1f}')
    ax.set_xlabel('PC1  (log-spectral shape, Reynolds axis)')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=style.LEG_FS - 2, ncol=2, loc='lower left', framealpha=.92)
    ax.set_title('Chain starting points: the standard reconstruction pool of each regime '
                 'on the manifold', fontsize=style.TITLE_FS)
    ax.grid(alpha=.25)
    for d_, ext in (('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')):
        p_ = f'{d_}/manifold_recon_starts.{ext}'; fig.savefig(p_); print('written', p_)


if __name__ == '__main__':
    main()
