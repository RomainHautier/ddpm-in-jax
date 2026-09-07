"""The manifold, and where the Re=8000 reconstruction POPULATIONS actually sit.

Four regimes' ground truths form the reference manifold (log-spectral PCA); the reconstruction
clouds of every configuration evaluated at Re=8000 are projected into it. The march toward the
target cloud, and where it stops, is the figure.

  JAX_PLATFORMS=cpu python plot_scripts/manifold_recon_clouds.py
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import style

REGS = [1000, 2000, 4000, 8000]
RECONS = [('base0__K3__mop0.2_0', 'base', style.MODEL_COLOR['base0'], 'o'),
          ('mt1k-0499__K3__mop0.2_0', 'Re=1000 ft', style.MODEL_COLOR['mt1k-0499'], 's'),
          ('r8kp02-0599__K3__mop0.2_0', 'Re=8000 ft (in-regime)', style.MODEL_COLOR['r8kp02-0599'], 'D'),
          ('r8kp02-0599__K3__v7bandgate', 'Re=8000 ft + gate', '#0f9e78', '^')]


def main():
    import jax.numpy as jnp
    from src.rewards import make_spectrum_fn
    style.apply(style.BASE_FS, 150)
    sf = make_spectrum_fn(256)
    spec = lambda x: np.log(np.maximum(np.concatenate(
        [np.asarray(sf(jnp.asarray(x[i:i+32]))) for i in range(0, len(x), 32)])[:, 1:96], 1e-20))
    gts = {R: spec(np.load(f'base_results/fields/re{R}/GT.npz')['x'].astype(np.float32)) for R in REGS}
    X = np.concatenate(list(gts.values())); mu = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    P = lambda A: (A - mu) @ Vt[:2].T

    fig, ax = plt.subplots(figsize=(11.8, 7.6), constrained_layout=True)
    for R, g in gts.items():
        v = P(g)
        ax.scatter(v[:, 0], v[:, 1], s=26, color=style.REGIME_COLOR[R], alpha=.55, lw=0,
                   label=f'Re={R} ground truth')
    for name, lab, col, mk in RECONS:
        x = np.load(f'base_results/fields/re8000/{name}.npz')['x'].astype(np.float32)
        v = P(spec(x))
        ax.scatter(v[:, 0], v[:, 1], s=34, marker=mk, color=col, alpha=.60,
                   edgecolors='white', linewidths=.4,
                   label=f'{lab} — reconstructions of Re=8000')
    ax.set_xlabel(f'PC1  ({var[0]*100:.0f}% of variance — the Reynolds axis)')
    ax.set_ylabel(f'PC2  ({var[1]*100:.0f}%)')
    ax.set_title('Every configuration reconstructing Re=8000, on the manifold of the four '
                 'regimes', fontsize=style.TITLE_FS)
    ax.legend(fontsize=style.LEG_FS - 1, loc='lower left', framealpha=.9)
    for d_, ext in (('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')):
        p_ = f'{d_}/manifold_recon_clouds.{ext}'; fig.savefig(p_); print('  written', p_)


if __name__ == '__main__':
    argparse.ArgumentParser().parse_args(); main()
