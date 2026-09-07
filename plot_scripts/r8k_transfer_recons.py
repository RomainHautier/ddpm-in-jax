"""Appendix figure: per sample reconstructions of the downward transferred Re=8000 model.

Four regimes down the rows, three samples per regime across the columns, each shown as a
ground truth and reconstruction pair. The reconstruction uses the regime's selected
inference configuration, the shortened chains below home and the full K3 chain at home.
Samples are picked at the 25th, 50th and 75th percentile of ground truth fine band
energy so the per sample dose tracking is visible.

  JAX_PLATFORMS=cpu python plot_scripts/r8k_transfer_recons.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import style

M = 'r8kp02-0599'
SEL = {1000: ('K100__ccp0.2_0', 'chain [100]'), 2000: ('K125__cltp0.2_0', 'chain [125]'),
       4000: ('K160__cltp0.2_0', 'chain [160]'), 6000: ('K3__mop0.2_0', 'K3'),
       8000: ('K3__mop0.2_0', 'K3 (home)')}
PCTS = [25, 50, 75]


def main():
    import jax.numpy as jnp
    from src.rewards import make_spectrum_fn
    style.apply(style.BASE_FS, 170)
    sf = make_spectrum_fn(256)

    fig, axes = plt.subplots(len(SEL), 6, figsize=(3.0 * 6, 3.15 * len(SEL)),
                             constrained_layout=True)
    for ri, (R, (tok, lab)) in enumerate(SEL.items()):
        gt = np.load(f'base_results/fields/re{R}/GT.npz')['x'].astype(np.float32)
        rc = np.load(f'base_results/fields/re{R}/{M}__{tok}.npz')['x'].astype(np.float32)
        if len(gt) != len(rc):
            S = np.load(f'base_results/regime_audit_re{R}.npz', allow_pickle=True)
            msk = np.asarray(S[f'{R}|EVAL||seq']) >= 12
            assert msk.sum() == len(rc), f'mask {msk.sum()} vs fields {len(rc)} at {R}'
            gt = gt[msk]
        E = np.concatenate([np.asarray(sf(jnp.asarray(gt[i:i + 32])))
                            for i in range(0, len(gt), 32)])[:, 32:].sum(1)
        order = np.argsort(E)
        idx = [int(order[int(len(E) * p / 100)]) for p in PCTS]
        for ci, (p, i) in enumerate(zip(PCTS, idx)):
            v = float(np.percentile(np.abs(gt[i, ..., 1]), 99.5))
            for half, img in ((0, gt[i, ..., 1]), (1, rc[i, ..., 1])):
                ax = axes[ri][2 * ci + half]
                ax.imshow(img, cmap='RdBu_r', vmin=-v, vmax=v)
                ax.set_xticks([]); ax.set_yticks([])
                if ri == 0:
                    ax.set_title(('ground truth' if half == 0 else 'reconstruction')
                                 + f'\n({p}th pctile sample)', fontsize=style.TITLE_FS - 2)
            if ci == 0:
                axes[ri][0].set_ylabel(f'Re={R}\n{lab}', fontsize=style.TITLE_FS - 1)
    fig.suptitle('The Re=8000 fine tune across the ladder under the selected inference '
                 'configuration per regime, three samples per regime (vorticity)',
                 fontsize=style.SUP_FS)
    for d_, ext in (('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')):
        p_ = f'{d_}/r8k_transfer_recons.{ext}'; fig.savefig(p_); print('written', p_)


if __name__ == '__main__':
    main()
