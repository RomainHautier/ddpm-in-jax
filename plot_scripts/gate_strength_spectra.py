"""Ratio spectra of the gate strength sweep: mt1k carried to Re=8000 under lambda
3/6/12/25, against the unguided model and the Re=8000 fine-tune as the reference of
what weights achieve.

  JAX_PLATFORMS=cpu python plot_scripts/gate_strength_spectra.py
"""
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'plot_scripts')
import numpy as np, style
import matplotlib.pyplot as plt

def main():
    style.apply(style.BASE_FS, 150)
    S = np.load('base_results/regime_audit_re8000.npz', allow_pickle=True); F = set(S.files)
    Eg = np.asarray(S['8000|GT||E'])[1:128]
    k = np.arange(1, 128)
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    style.shade_bands(ax)
    ax.axhline(1, color=style.GT_COLOR, lw=1.4, zorder=6, label='ground truth')
    curves = [('8000|mt1k-0499|K3|mop0.2_0||E', 'Re=1000 ft, unguided', '#9aa198', '--'),
              ('8000|mt1k-0499|K3|v7s3||E', '$+$ gate, $\\lambda=3$', '#c7dbe8', '-'),
              ('8000|mt1k-0499|K3|v7s6||E', '$+$ gate, $\\lambda=6$', '#7fb0cc', '-'),
              ('8000|mt1k-0499|K3|v7s12||E', '$+$ gate, $\\lambda=12$', '#3d80ab', '-'),
              ('8000|mt1k-0499|K3|v7s25||E', '$+$ gate, $\\lambda=25$', '#1b5578', '-'),
              ('8000|r8kp02-0599|K3|mop0.2_0||E', 'Re=8000 ft, unguided', '#8a5cd6', '-')]
    for key, lab, col, ls in curves:
        if key not in F: print('missing', key); continue
        E = np.asarray(S[key])[1:128]
        ax.semilogx(k, E / Eg, ls, color=col, lw=1.9, label=lab, zorder=4)
    ax.set(yscale='log', ylim=(0.03, 2.4))
    ax.set_yticks([0.03, 0.1, 0.3, 1, 2]); ax.set_yticklabels(['0.03', '0.1', '0.3', '1', '2'])
    ax.axvline(96, color='#9aa198', lw=.8, ls=':', zorder=1)
    ax.set_xlabel('wavenumber $k$'); ax.set_ylabel('$E(k)\\,/\\,E_{\\mathrm{GT}}(k)$')
    ax.legend(fontsize=style.LEG_FS - 1, loc='lower left', framealpha=.9)
    ax.set_title('Gate strength sweep at Re=8000: the Re=1000 fine-tune under '
                 'increasing $\\lambda$', fontsize=style.TITLE_FS)
    for d_, ext in (('docs/figs_overleaf/gated', 'pdf'), ('plotting/figs', 'png')):
        os.makedirs(d_, exist_ok=True)
        p_ = f'{d_}/gate_strength_spectra.{ext}'; fig.savefig(p_); print('written', p_)

if __name__ == '__main__':
    main()
