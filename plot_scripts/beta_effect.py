"""What the variance-matching exponent beta does to the per-sample target predictor.

Two regimes by two predictor variants. The plain ratio target inherits the recon
observable's dispersion; the beta-corrected target compresses the log deviations by
beta = std(log GT band) / std(log obs), matching the true per-sample spread. At home
beta is near one and nothing changes; out of regime the recon over-disperses and the
correction pulls exaggerated targets back toward the truth.

  JAX_PLATFORMS=cpu python plot_scripts/beta_effect.py
"""
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'plot_scripts'); sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, style
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

REGS = [1000, 8000]


def main():
    import jax.numpy as jnp
    from src.rewards import make_spectrum_fn
    style.apply(style.BASE_FS, 150)
    sf = make_spectrum_fn(256)
    def spec(x):
        return np.concatenate([np.asarray(sf(jnp.asarray(x[i:i + 32])))
                               for i in range(0, len(x), 32)])
    fig, axes = plt.subplots(len(REGS), 2, figsize=(11.6, 4.6 * len(REGS)),
                             constrained_layout=True)
    for ri, R in enumerate(REGS):
        F = f'base_results/fields/re{R}'
        EG = spec(np.load(f'{F}/GT.npz')['x'].astype(np.float32))
        ER = spec(np.load(f'{F}/recon.npz')['x'].astype(np.float32))
        tgt = EG[:, 32:96].sum(1)
        obs = ER[:, 32:96].sum(1)
        beta = float(np.std(np.log(tgt)) / np.std(np.log(obs)))
        preds = [('plain ratio  $s_i = \\mathrm{obs}_i/C_R$',
                  tgt.mean() * (obs / obs.mean())),
                 (f'$\\beta$-corrected  $s_i = (\\mathrm{{obs}}_i/C_R)^{{\\beta}}$,  '
                  f'$\\beta={beta:.2f}$',
                  tgt.mean() * (obs / obs.mean()) ** beta)]
        for ci, (lab, pred) in enumerate(preds):
            ax = axes[ri][ci]
            err = np.abs(pred / tgt - 1)
            lo, hi = tgt.min() * .7, tgt.max() * 1.4
            xs = np.array([lo, hi])
            ax.fill_between(xs, xs * .8, xs * 1.2, color='#efe9da', zorder=1)
            ax.plot(xs, xs, '-', color=style.GT_COLOR, lw=1.3, zorder=2)
            ax.scatter(tgt, pred, s=16, color='#c22f4f' if ci == 0 else '#0f9e78',
                       lw=0, alpha=.8, zorder=3)
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlim(lo, hi); ax.set_ylim(lo * .6, hi * 1.3)
            ax.set_xlabel("sample's true $E_{GT}[32,96)$" if ri == len(REGS) - 1 else '')
            ax.set_ylabel(f'Re={R}\npredicted target' if ci == 0 else 'predicted target')
            ax.set_title(lab, fontsize=style.TITLE_FS - 1)
            ax.text(.03, .95, f'median |err| {np.median(err)*100:.1f}%\n'
                    f'within 20%: {np.mean(err<.2)*100:.0f}%\n'
                    f'spearman {spearmanr(pred, tgt).correlation:.3f}',
                    transform=ax.transAxes, fontsize=style.LEG_FS - 1, va='top',
                    bbox=dict(fc='white', ec='none', alpha=.85))
            ax.grid(alpha=.25)
    fig.suptitle('The variance-matching exponent: predicted per-sample targets with and '
                 'without $\\beta$', fontsize=style.SUP_FS)
    for d_, ext in (('docs/figs_overleaf/gated', 'pdf'), ('plotting/figs', 'png')):
        os.makedirs(d_, exist_ok=True)
        p_ = f'{d_}/beta_effect.{ext}'; fig.savefig(p_); print('written', p_)


if __name__ == '__main__':
    main()
