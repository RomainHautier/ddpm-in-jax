"""The transfer on the manifold: one panel per high-regime specialist, its K3 unguided
reconstruction cloud at every embedded target regime, over the truth clouds.

Recon clouds are triangles in the saturated color of their TARGET regime; the truth
clouds are dots in the same (unsaturated) colors, so each triangle cloud should be read
against the dot cloud of its own color. Overshoot appears as a triangle cloud right of
its dots; the top-octave deficit as a cloud left of them.

  JAX_PLATFORMS=cpu python plot_scripts/manifold_transfer_clouds.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import style

REGS = [1000, 2000, 4000, 8000]
MODELS = [('r4kp02-0599', 'Re=4000 fine-tune'), ('r8kp02-0599', 'Re=8000 fine-tune')]


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

    fig, axes = plt.subplots(2, 1, figsize=(10.6, 9.4), sharex=True, sharey=True)
    for ax, (m, lab) in zip(axes, MODELS):
        for R in REGS:
            v = P(gts[R])
            ax.scatter(v[:, 0], v[:, 1], s=15, color=style.REGIME_COLOR[R], alpha=.30,
                       linewidths=0, zorder=2,
                       label=f'Re={R} truth' if m == MODELS[0][0] else None)
        for R in REGS:
            rc = P(spec(np.load(f'base_results/fields/re{R}/{m}__K3__mop0.2_0.npz')['x']
                        .astype(np.float32)))
            col = style.vivid(style.REGIME_COLOR[R])
            ax.scatter(rc[:, 0], rc[:, 1], s=34, marker='^', color=col, alpha=.65,
                       edgecolors='white', linewidths=.4, zorder=4,
                       label=f'recons of Re={R}' if m == MODELS[0][0] else None)
            g = P(gts[R])
            print(f'{m} at Re={R}: recon PC1 median {np.median(rc[:,0]):6.2f}  '
                  f'truth {np.median(g[:,0]):6.2f}  offset {np.median(rc[:,0])-np.median(g[:,0]):+.1f}')
        ax.text(.008, .95, lab, transform=ax.transAxes, fontsize=11, va='top',
                fontweight='bold')
        ax.set_ylabel('PC2'); ax.grid(alpha=.25)
    axes[-1].set_xlabel('PC1  (log-spectral shape, Reynolds axis)')
    axes[0].legend(fontsize=style.LEG_FS - 2.5, ncol=2, loc='lower left', framealpha=.92)
    fig.suptitle('The specialists carried across the ladder, on the manifold '
                 '(K3 unguided)')
    fig.tight_layout()
    for d_, ext in (('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')):
        p_ = f'{d_}/manifold_transfer_clouds.{ext}'; fig.savefig(p_); print('written', p_)


if __name__ == '__main__':
    main()
