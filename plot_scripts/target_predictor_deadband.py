"""The per-sample target predictor against the truth, and what each deadband tolerates.

Left, predicted fine-band target against each sample's true fine-band energy, with the
10 and 20 percent corridors. A sample outside a corridor is one the gate would actively
push toward a mispredicted level at that deadband. Right, the error distribution with
the shares each deadband leaves unprotected.

  JAX_PLATFORMS=cpu python plot_scripts/target_predictor_deadband.py
"""
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'plot_scripts'); sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, style
import matplotlib.pyplot as plt


def main():
    import jax.numpy as jnp
    from src.rewards import make_spectrum_fn
    style.apply(style.BASE_FS, 150)
    sf = make_spectrum_fn(256)
    def spec(x):
        return np.concatenate([np.asarray(sf(jnp.asarray(x[i:i + 32])))
                               for i in range(0, len(x), 32)])
    gt = np.load('base_results/fields/re1000/GT.npz')['x'].astype(np.float32)
    rc = np.load('base_results/fields/re1000/recon.npz')['x'].astype(np.float32)
    EG, ER = spec(gt), spec(rc)
    tgt = EG[:, 32:96].sum(1)
    obs = ER[:, 32:96].sum(1)
    pred = tgt.mean() * (obs / obs.mean())
    err = pred / tgt - 1

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.6), constrained_layout=True)
    lo, hi = tgt.min() * .8, tgt.max() * 1.3
    xs = np.array([lo, hi])
    ax.fill_between(xs, xs * .8, xs * 1.2, color='#efe9da', zorder=1,
                    label=r'$\pm20\%$ deadband')
    ax.fill_between(xs, xs * .9, xs * 1.1, color='#ddd3ba', zorder=2,
                    label=r'$\pm10\%$ deadband')
    ax.plot(xs, xs, '-', color=style.GT_COLOR, lw=1.4, zorder=3)
    out20 = np.abs(err) >= .2
    out10 = (np.abs(err) >= .1) & ~out20
    ax.scatter(tgt[~(out10 | out20)], pred[~(out10 | out20)], s=18, color='#28658a',
               lw=0, zorder=4, label=f'within 10%  ({np.mean(np.abs(err)<.1)*100:.0f}%)')
    ax.scatter(tgt[out10], pred[out10], s=26, color='#d4a017', lw=0, zorder=5,
               label=f'10 to 20%  ({out10.mean()*100:.0f}%)')
    ax.scatter(tgt[out20], pred[out20], s=34, color='#c22f4f', lw=0, zorder=6,
               label=f'beyond 20%  ({out20.mean()*100:.0f}%)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lo, hi); ax.set_ylim(lo * .7, hi * 1.2)
    ax.set_xlabel("sample's true fine-band energy $E_{GT}[32,96)$")
    ax.set_ylabel('predicted per-sample target')
    ax.set_title('the target predictor against the truth', fontsize=style.TITLE_FS)
    ax.legend(fontsize=style.LEG_FS - 2, loc='upper left', framealpha=.9)

    ax2.hist(np.abs(err) * 100, bins=np.arange(0, 62, 2.5), color='#28658a', alpha=.85)
    for v, lab, col in ((10, 'deadband 10%', '#d4a017'), (20, 'deadband 20%', '#c22f4f')):
        ax2.axvline(v, color=col, lw=1.8, ls='--')
        ax2.text(v + .8, ax2.get_ylim()[1] * .9, lab, color=col,
                 fontsize=style.LEG_FS - 1)
    ax2.set_xlabel('|predicted target / true  $-$  1|  (%)')
    ax2.set_ylabel('samples')
    ax2.set_title(f'error distribution  (median '
                  f'{np.median(np.abs(err))*100:.1f}%)', fontsize=style.TITLE_FS)
    fig.suptitle('What each gate deadband tolerates: samples outside the corridor are '
                 'pushed toward a mispredicted target', fontsize=style.SUP_FS)
    for d_, ext in (('docs/figs_overleaf/gated', 'pdf'), ('plotting/figs', 'png')):
        os.makedirs(d_, exist_ok=True)
        p_ = f'{d_}/target_predictor_deadband.{ext}'; fig.savefig(p_); print('written', p_)


if __name__ == '__main__':
    main()
