"""Energy spectra of the regime-appropriate inference family: the best configuration of
the Re=8000 specialist at five regimes, against the default K3 chain and the base model.

Curves come from the stored mean spectra of the same evaluation rows the tables carry.

  JAX_PLATFORMS=cpu python plot_scripts/bestconfig_spectra.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib.pyplot as plt
import style

M = 'r8kp02-0599'
KMAX = 127
BEST = {1000: ('K100', 'ccp0.2_0', 'chain [100]'),
        2000: ('K125', 'cltp0.2_0', 'chain [125]'),
        4000: ('K160', 'cltp0.2_3', 'chain [160] + dial'),
        6000: ('K3', 'v7bandgate', 'K3 + gate'),
        8000: ('K3', 'v7bandgate', 'K3 + gate')}


def store(R):
    return ('base_results/re1000_audit.npz' if R == 1000
            else f'base_results/regime_audit_re{R}.npz')


def main():
    style.apply(style.BASE_FS, 150)
    k = np.arange(1, KMAX + 1)
    for absolute in (False, True):
        fig, axes = plt.subplots(2, 3, figsize=(4.2 * 3, 3.3 * 2),
                                 constrained_layout=True, sharey=not absolute)
        for i, (R, (ck, sg, lab)) in enumerate(BEST.items()):
            ax = axes[i // 3][i % 3]
            S = np.load(store(R), allow_pickle=True); F = set(S.files)
            Eg = np.asarray(S[f'{R}|GT||E'])[1:KMAX + 1]
            style.shade_bands(ax)
            curves = [('base', 'base0', 'K3', 'mop0.2_0', '#9aa198', '--'),
                      ('K3 unguided', M, 'K3', 'mop0.2_0', '#b9a3e3', '-'),
                      (lab, M, ck, sg, '#8a5cd6', '-')]
            if absolute:
                ax.loglog(k, Eg, '-', color=style.GT_COLOR, lw=style.GT_LW,
                          label='ground truth', zorder=6)
            else:
                ax.axhline(1, color=style.GT_COLOR, lw=1.3, zorder=6)
            for clab, m, cck, csg, col, ls in curves:
                key = f'{R}|{m}|{cck}|{csg}||E'
                if key not in F:
                    print(f'  missing {key}'); continue
                E = np.asarray(S[key])[1:KMAX + 1]
                (ax.loglog if absolute else ax.semilogx)(
                    k, E if absolute else E / Eg, ls, color=col, lw=1.8,
                    label=clab, zorder=4)
            if not absolute:
                ax.set(yscale='log', ylim=(0.015, 4.2))
                ax.set_yticks([0.03, 0.1, 0.3, 1, 2, 4])
                ax.set_yticklabels(['0.03', '0.1', '0.3', '1', '2', '4'])
            ax.axvline(96, color='#9aa198', lw=.8, ls=':', zorder=1)
            ax.set_title(f'Re = {R}   best: {lab}', fontsize=style.TITLE_FS - 1)
            if i // 3 == 1: ax.set_xlabel('wavenumber $k$')
            if i % 3 == 0:
                ax.set_ylabel('$E(k)$' if absolute
                              else '$E(k)\\,/\\,E_{\\mathrm{GT}}(k)$')
            if i == 0:
                ax.legend(fontsize=style.LEG_FS - 1.5, loc='lower left')
        axes[1][2].axis('off')
        fig.suptitle('The best inference configuration per regime, one model '
                     '(the Re=8000 specialist)', fontsize=style.SUP_FS)
        sfx = 'spectra' if absolute else 'ratio'
        for d_, ext in (('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')):
            p_ = f'{d_}/bestconfig_{sfx}.{ext}'; fig.savefig(p_); print('written', p_)


if __name__ == '__main__':
    main()
