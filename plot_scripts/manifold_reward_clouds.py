"""Sample clouds of the legacy and pure-gate specialists on the spectral manifold, at
their home regimes. Same basis as every manifold figure: log spectra over shells
[1,96), PCA from the pooled ground truth of Re 1000/2000/4000/8000.

  JAX_PLATFORMS=cpu python plot_scripts/manifold_reward_clouds.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import style

BASIS_REGS = [1000, 2000, 4000, 8000]
PANELS = {4000: [('r4kp02-0599', 'legacy reward', '#0f9e78'),
                 ('pg4k-0299',   'pure gate',     '#8a5cd6')],
          8000: [('r8kp02-0599', 'legacy reward', '#0f9e78'),
                 ('pg8k-0499',   'pure gate',     '#8a5cd6')]}
TGT_COL = '#b8399e'


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

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4), constrained_layout=True)
    for ax, (R, rows) in zip(axes, PANELS.items()):
        for Rb in BASIS_REGS:
            v = P(gts[Rb])
            if Rb == R:
                ax.scatter(v[:, 0], v[:, 1], s=22, color=TGT_COL, alpha=.6,
                           linewidths=0, zorder=3, label=f'Re={R} truth')
            else:
                ax.scatter(v[:, 0], v[:, 1], s=10, color=style.REGIME_COLOR[Rb],
                           alpha=.14, linewidths=0, zorder=2)
        for m, lab, col in rows:
            f = f'base_results/fields/re{R}/{m}__K3__mop0.2_0.npz'
            if not os.path.exists(f):
                print('missing', f); continue
            v = P(spec(np.load(f)['x'].astype(np.float32)))
            ax.scatter(v[:, 0], v[:, 1], s=26, marker='D', color=col, alpha=.8,
                       linewidths=.4, edgecolors='white', zorder=5,
                       label=f'{lab}, unguided')
        ax.set_title(f'Re = {R} specialists at home', fontsize=style.TITLE_FS)
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2' if R == 4000 else '')
        ax.grid(alpha=.2)
        ax.legend(fontsize=style.LEG_FS - 1, loc='upper left', framealpha=.9)
    fig.suptitle('Reward generations on the spectral manifold, home regimes',
                 fontsize=style.SUP_FS)
    for d_, ext in (('docs/figs_overleaf/gated', 'pdf'), ('plotting/figs', 'png')):
        os.makedirs(d_, exist_ok=True)
        p_ = f'{d_}/manifold_reward_clouds.{ext}'; fig.savefig(p_)
        print('written', p_)


if __name__ == '__main__':
    main()
