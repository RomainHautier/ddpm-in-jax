"""Does the low-resolution input carry the per-sample energy ordering of the truth?

Three scatters over the held-out Re=1000 pool. Left, LR total energy against GT total,
the full-band ordering. Middle, the LR's highest resolved band [16,32) against the GT
fine band [32,96), the ordering the per-sample dose must recover. Right, the base
reconstruction's fine band against the same target, the learned digest the v7 predictor
uses. Annotations carry Pearson on logs and Spearman rank correlation.

  JAX_PLATFORMS=cpu python plot_scripts/lr_hr_ordering.py
"""
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'plot_scripts'); sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, style
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


REGS = [int(x) for x in os.environ.get('LRHR_REGS', '1000,2000,4000,8000').split(',')]


def main():
    import jax.numpy as jnp
    from src.rewards import make_spectrum_fn
    style.apply(style.BASE_FS, 150)
    sf = make_spectrum_fn(256)
    def spec(x):
        return np.concatenate([np.asarray(sf(jnp.asarray(x[i:i + 32])))
                               for i in range(0, len(x), 32)])
    fig, axes = plt.subplots(len(REGS), 3, figsize=(13.8, 4.3 * len(REGS)),
                             constrained_layout=True, squeeze=False)
    for ri, R in enumerate(REGS):
        F = f'base_results/fields/re{R}'
        EL = spec(np.load(f'{F}/LR.npz')['x'].astype(np.float32))
        EG = spec(np.load(f'{F}/GT.npz')['x'].astype(np.float32))
        ER = spec(np.load(f'{F}/recon.npz')['x'].astype(np.float32))
        assert len(EL) == len(EG) == len(ER), f'pool mismatch at Re={R}'
        panels = [
            ('LR total $[1,32)$ vs GT total', EL[:, 1:32].sum(1), EG[:, 1:96].sum(1),
             'LR total energy', 'GT total energy', '#28658a'),
            ('LR $[16,32)$ vs GT $[32,96)$', EL[:, 16:32].sum(1), EG[:, 32:96].sum(1),
             'LR band $[16,32)$ energy', 'GT band $[32,96)$ energy', '#d4a017'),
            ('base recon $[32,96)$ vs GT $[32,96)$', EG[:, 32:96].sum(1),
             ER[:, 32:96].sum(1), 'GT band $[32,96)$ energy',
             'recon band $[32,96)$ energy', '#c22f4f'),
        ]
        for ci, (ttl, x, y, xl, yl, col) in enumerate(panels):
            ax = axes[ri][ci]
            r = np.corrcoef(np.log(x), np.log(y))[0, 1]
            sp = spearmanr(x, y).correlation
            b, a = np.polyfit(np.log(x), np.log(y), 1)
            note = f'pearson(log) {r:.3f}\nspearman {sp:.3f}'
            if ci < 2:
                ax.scatter(x, y, s=14, color=col, lw=0, alpha=.8)
                xs = np.array([x.min(), x.max()])
                ax.plot(xs, np.exp(a) * xs ** b, '-', color=style.GT_COLOR, lw=1.2,
                        zorder=3)
            else:
                # raw recon cloud (below the identity, the deficit) and the same cloud
                # lifted by the ONE pool constant: the level is all the recon lacks
                C = np.exp(np.mean(np.log(x / y)))
                inb = np.mean(np.abs(y * C / x - 1) < .2) * 100
                xs = np.array([min(x.min(), y.min()) * .8, x.max() * 1.4])
                ax.plot(xs, xs, '-', color=style.GT_COLOR, lw=1.3, zorder=2)
                ax.scatter(x, y, s=18, marker='o', facecolors='none',
                           edgecolors='#9aa198', lw=.9, zorder=3, label='raw')
                ax.scatter(x, y * C, s=18, marker='o', facecolors='none',
                           edgecolors=col, lw=.9, zorder=4,
                           label=f'$\\times\\,{C:.2f}$ shift')
                ax.legend(fontsize=style.LEG_FS - 2, loc='lower right', framealpha=.9)
                note += (f'\nmean ratio {np.mean(y / x):.2f}, log-slope {b:.2f}'
                         f'\n{inb:.0f}% within $\\pm20\\%$ shifted')
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlabel(xl if ri == len(REGS) - 1 else '')
            ax.set_ylabel(f'Re={R}\n{yl}' if ci == 0 else yl)
            if ri == 0: ax.set_title(ttl, fontsize=style.TITLE_FS - 1)
            ax.text(.03, .95, note, transform=ax.transAxes, fontsize=style.LEG_FS - 1,
                    va='top', bbox=dict(fc='white', ec='none', alpha=.8))
            ax.grid(alpha=.25)
    where = 'at every regime' if len(REGS) > 1 else f'Re={REGS[0]}'
    fig.suptitle('The per-sample energy ordering of the truth is readable from the '
                 f'low-resolution input, {where}', fontsize=style.SUP_FS)
    sfx = '' if len(REGS) > 1 else f'_re{REGS[0]}'
    for d_, ext in (('docs/figs_overleaf/gated', 'pdf'), ('plotting/figs', 'png')):
        os.makedirs(d_, exist_ok=True)
        p_ = f'{d_}/lr_hr_ordering{sfx}.{ext}'; fig.savefig(p_); print('written', p_)


if __name__ == '__main__':
    main()
