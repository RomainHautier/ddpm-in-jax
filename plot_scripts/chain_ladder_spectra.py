"""The chain ladder in spectral form: r8kp02 carried down to Re=1000, dose set by the chain.

  JAX_PLATFORMS=cpu python plot_scripts/chain_ladder_spectra.py
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib.pyplot as plt
import style

S = np.load('base_results/re1000_audit.npz', allow_pickle=True)
# cold -> hot: less generation -> more. K3/K100 are TEST rows, the rest validation (~1-2% pool
# offset in the GT denominator, flagged in the caption).
ROWS = [('K50', 'fcp0.2_0', '[50]', '#28658a'),
        ('K100', 'ccp0.2_0', '[100]   (the selected rung)', '#0f9e78'),
        ('K150', 'fcp0.2_0', '[150]', '#d4770a'),
        ('K150-100', 'fcp0.2_0', '[150,100]', '#e08a9c'),
        ('K3', 'mop0.2_0', 'K3 [150,100,50]  (standard)', '#c22f4f')]


def make():
    style.apply()
    Eg = np.asarray(S['1000|GT||E'])
    k = np.arange(1, 128)
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.4), constrained_layout=True)
    for a in ax: style.shade_bands(a)
    ax[0].loglog(k, Eg[1:128], '-', color=style.GT_COLOR, lw=style.GT_LW, label='ground truth', zorder=6)
    ax[1].axhline(1, color=style.GT_COLOR, lw=1.5, zorder=6)
    for ck, sg, lab, col in ROWS:
        key = f'1000|r8kp02-0599|{ck}|{sg}||E'
        if key not in S.files: continue
        E = np.asarray(S[key])
        ax[0].loglog(k, E[1:128], '-', color=col, lw=1.7, label=lab)
        ax[1].semilogx(k, E[1:128] / Eg[1:128], '-', color=col, lw=1.7)
    for a in ax:
        a.axvline(96, color='#9aa198', lw=0.8, ls=':')
        a.set_xlabel('wavenumber $k$')
    ax[0].set_ylabel('$E(k)$'); ax[0].legend(fontsize=style.LEG_FS, loc='lower left')
    ax[0].set_title('mean vorticity spectrum', fontsize=style.TITLE_FS)
    ax[1].set(yscale='log', ylim=(0.25, 6))
    ax[1].set_yticks([0.25, 0.5, 1, 2, 4]); ax[1].set_yticklabels(['0.25', '0.5', '1', '2', '4'])
    ax[1].set_ylabel('$E(k)/E_{\\mathrm{GT}}(k)$')
    ax[1].set_title('ratio to ground truth', fontsize=style.TITLE_FS)
    fig.suptitle('The Re=8000 specialist at Re=1000, unguided: each chain rung is a dose',
                 fontsize=style.SUP_FS)
    for d, ext in (('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')):
        p = f'{d}/chain_ladder_spectra.{ext}'; fig.savefig(p); print('  written', p)


if __name__ == '__main__':
    argparse.ArgumentParser().parse_args(); make()
