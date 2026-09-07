"""Dial against gate at Re=1000, one panel, colour encodes the model.

Base model in blue, fine-tune in red; linestyle encodes the treatment (dotted =
unguided, dashed = tapered dial, solid = per-band per-sample gate), all as energy
spectrum ratios to ground truth.

  JAX_PLATFORMS=cpu python plot_scripts/dial_vs_gate_re1000.py
"""
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'plot_scripts')
import numpy as np, style
import matplotlib.pyplot as plt

MODELS = [('base', 'base0', '#28658a'), ('fine-tune', 'mt1k-0499', '#c22f4f')]
DIALS = [('mop0.2_0', 'unguided', ':'),
         ('tapp0.2_3', '$+$ dial', '--'),
         ('v8lowband', '$+$ gate', '-')]


def main():
    style.apply(style.BASE_FS, 150)
    S = np.load('base_results/re1000_audit.npz', allow_pickle=True); F = set(S.files)
    Eg = np.asarray(S['1000|GT||E'])[1:128]
    k = np.arange(1, 128)
    fig, ax = plt.subplots(figsize=(7.4, 4.9), constrained_layout=True)
    style.shade_bands(ax)
    ax.axhline(1, color=style.GT_COLOR, lw=1.4, zorder=6, label='ground truth')
    for mlab, m, col in MODELS:
        for sg, dlab, ls in DIALS:
            key = f'1000|{m}|K3|{sg}||E'
            if key not in F: continue
            E = np.asarray(S[key])[1:128]
            ax.semilogx(k, E / Eg, ls, color=col, lw=1.9,
                        label=f'{mlab}, {dlab}' if sg == 'mop0.2_0' else f'{mlab} {dlab}',
                        zorder=4)
    ax.set(yscale='log', ylim=(0.25, 1.6))
    ax.set_yticks([0.3, 0.5, 0.7, 1, 1.3])
    ax.set_yticklabels(['0.3', '0.5', '0.7', '1', '1.3'])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.axvline(96, color='#9aa198', lw=.8, ls=':', zorder=1)
    ax.set_xlabel('wavenumber $k$')
    ax.set_ylabel('$E(k)\\,/\\,E_{\\mathrm{GT}}(k)$')
    ax.grid(alpha=.25)
    ax.legend(fontsize=style.LEG_FS, loc='lower left', framealpha=.9, ncol=2)
    ax.set_title('Tapered dial against the per-band per-sample gate, Re=1000',
                 fontsize=style.TITLE_FS)
    for d_, ext in (('docs/figs_overleaf/gated', 'pdf'), ('plotting/figs', 'png')):
        os.makedirs(d_, exist_ok=True)
        p_ = f'{d_}/dial_vs_gate_re1000.{ext}'; fig.savefig(p_); print('written', p_)


if __name__ == '__main__':
    main()
