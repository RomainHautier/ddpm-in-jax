"""Adaptive inference on the manifold: the Re=8000 specialist's reconstruction CLOUDS
under different chain configs, one panel per downward target regime.

The cloud companion of offmanifold_dechain: instead of one sample's trajectory, the full
evaluation pool under each inference config. Dropping passes and reducing the renoise
depth slides the whole cloud left along the Reynolds axis until it covers the target
truth. Clouds from validation pools are marked (val); the rest are test pools. The
per-regime selected chain is black-edged.

  JAX_PLATFORMS=cpu python plot_scripts/manifold_chain_clouds.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import style

BASIS_REGS = [1000, 2000, 4000, 8000]
M = 'r8kp02-0599'
CCOL = {'[150,100,50]': '#4a4e57', '[150,100]': '#5b8bd0', '[150]': '#2ba58f',
        '[125]': '#d4770a', '[120]': '#d4a017', '[100]': '#c2402f'}
# per target: (label, file token, selected?)
PANELS = {
    1000: [('[150,100,50]', 'K3__mop0.2_0', False), ('[150,100]', 'K150-100__fcp0.2_0', False),
           ('[150]', 'K150__fcp0.2_0', False), ('[100]', 'K100__ccp0.2_0', True)],
    1500: [('[150,100,50]', 'K3__mop0.2_0', False), ('[100]', 'K100__clvp0.2_0', False),
           ('[120]', 'K120__cltp0.2_0', True)],
    2000: [('[150,100,50]', 'K3__mop0.2_0', False), ('[150]', 'K150__clvp0.2_0', False),
           ('[120]', 'K120__cltp0.2_0', False), ('[125]', 'K125__cltp0.2_0', True)],
    3000: [('[150,100,50]', 'K3__mop0.2_0', False), ('[150,100]', 'K150-100__clvp0.2_0', False),
           ('[120]', 'K120__clvp0.2_0', False), ('[150]', 'K150__cltp0.2_0', True)],
}


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
    tgt_gts = dict(gts)
    for R in PANELS:
        if R not in tgt_gts:
            tgt_gts[R] = spec(np.load(f'base_results/fields/re{R}/GT.npz')['x']
                              .astype(np.float32))

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(12.5, 10.4),
                             sharex=True, sharey=True)
    for ax, (R, cfgs) in zip(axes, PANELS.items()):
        for Rb in BASIS_REGS:
            if Rb == R: continue
            v = P(gts[Rb])
            ax.scatter(v[:, 0], v[:, 1], s=14, color=style.REGIME_COLOR[Rb],
                       alpha=.22, linewidths=0, zorder=2)
        v = P(tgt_gts[R])
        ax.scatter(v[:, 0], v[:, 1], s=24, color=style.vivid(style.REGIME_COLOR[R]),
                   alpha=.75, linewidths=0, zorder=3)
        for lab, tok, sel in cfgs:
            f = f'base_results/fields/re{R}/{M}__{tok}.npz'
            if not os.path.exists(f):
                print(f'  MISSING {f}'); continue
            rc = P(spec(np.load(f)['x'].astype(np.float32)))
            val = 'clv' in tok
            ax.scatter(rc[:, 0], rc[:, 1], s=40 if sel else 30, marker='^',
                       color=CCOL[lab], alpha=.75 if sel else .6,
                       edgecolors='black' if sel else 'white',
                       linewidths=.7 if sel else .4, zorder=6 if sel else 5)
            g = P(tgt_gts[R])
            print(f'Re={R} {lab:12s}{"(val)" if val else "     "} '
                  f'PC1 median {np.median(rc[:,0]):6.2f}  truth {np.median(g[:,0]):6.2f}')
        ax.set_ylabel('PC2')
        sel_lab = next(l for l, _, s in cfgs if s)
        ax.text(.008, .93, f'target Re={R}   selected {sel_lab}',
                transform=ax.transAxes, fontsize=11, va='top', fontweight='bold')
        ax.set_ylim(-5.8, 5.6)
        ax.grid(alpha=.25)
    axes[-1].set_xlabel('PC1  (log-spectral shape, Reynolds axis)')
    handles = [plt.Line2D([], [], ls='none', marker='^', color=c, ms=8, label=l)
               for l, c in CCOL.items()]
    handles.append(plt.Line2D([], [], ls='none', marker='^', color='#9aa198', ms=8,
                              markeredgecolor='black', label='selected chain'))
    fig.legend(handles=handles, loc='upper center', ncol=7, fontsize=8.6,
               bbox_to_anchor=(.5, .972), framealpha=.92)
    fig.suptitle('One model, many chains: the Re=8000 specialist\'s reconstruction '
                 'clouds under each inference config', y=.995)
    fig.tight_layout(rect=(0, 0, 1, .95))
    for d_, ext in (('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')):
        p_ = f'{d_}/manifold_chain_clouds.{ext}'; fig.savefig(p_); print('written', p_)


if __name__ == '__main__':
    main()
