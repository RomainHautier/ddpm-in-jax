"""Reward-generation comparison at the specialists' home regimes, two rows: unguided
above, v8-gated below. Columns Re=4000 and Re=8000; same colour per reward generation.

  JAX_PLATFORMS=cpu python plot_scripts/anchorcmp_2row.py
"""
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'plot_scripts')
import numpy as np, style
import matplotlib.pyplot as plt

MODELS = {4000: [('r4kp02-0599', 'legacy reward (per-shell push)', '#0f9e78'),
                 ('pg4k-0299',   'pure gate (anchor 0.5)',         '#8a5cd6'),
                 ('sa4k-0399',   'strong anchor (3.0)',            '#c22f4f')],
          8000: [('r8kp02-0599', 'legacy reward (per-shell push)', '#0f9e78'),
                 ('pg8k-0499',   'pure gate (anchor 0.5)',         '#8a5cd6'),
                 ('sa8k-0599',   'strong anchor (3.0)',            '#c22f4f')]}
ROWS = [('mop0.2_0', 'unguided'), ('v8lowband', 'with the gated dial')]


def main():
    style.apply(style.BASE_FS, 150)
    k = np.arange(1, 128)
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.6), constrained_layout=True,
                             sharey=True, squeeze=False)
    for ci, R in enumerate((4000, 8000)):
        S = np.load(f'base_results/regime_audit_re{R}.npz', allow_pickle=True)
        F = set(S.files)
        Eg = np.asarray(S[f'{R}|GT||E'])[1:128]
        for ri, (tag, rlab) in enumerate(ROWS):
            ax = axes[ri][ci]
            style.shade_bands(ax)
            ax.axhline(1, color=style.GT_COLOR, lw=1.4, zorder=6, label='ground truth')
            kb = f'{R}|base0|K3|mop0.2_0||E'
            if kb in F:
                ax.semilogx(k, np.asarray(S[kb])[1:128] / Eg, '--', color='#9aa198',
                            lw=1.6, label='base, unguided', zorder=3)
            for m, lab, col in MODELS[R]:
                if tag == 'v8lowband' and m.startswith('r'):
                    continue          # legacy models stay unguided-only, per user
                key = f'{R}|{m}|K3|{tag}||E'
                if key not in F: continue
                ax.semilogx(k, np.asarray(S[key])[1:128] / Eg, '-', color=col, lw=1.8,
                            label=lab, zorder=4)
            ax.set(yscale='log', ylim=(0.015, 2.6))
            ax.set_yticks([0.03, 0.1, 0.3, 1, 2])
            ax.set_yticklabels(['0.03', '0.1', '0.3', '1', '2'])
            ax.yaxis.set_minor_formatter(plt.NullFormatter())
            ax.axvline(96, color='#9aa198', lw=.8, ls=':', zorder=1)
            if ri == 0: ax.set_title(f'Re = {R}', fontsize=style.TITLE_FS)
            if ri == 1: ax.set_xlabel('wavenumber $k$')
            if ci == 0:
                ax.set_ylabel(f'{rlab}\n$E(k)\\,/\\,E_{{\\mathrm{{GT}}}}(k)$')
            ax.grid(alpha=.25)
    axes[0][0].legend(fontsize=style.LEG_FS - 1, loc='lower left', framealpha=.9)
    fig.suptitle('Reward generations at their home regimes, unguided and gated',
                 fontsize=style.SUP_FS)
    for d_, ext in (('docs/figs_overleaf/gated', 'pdf'), ('plotting/figs', 'png')):
        os.makedirs(d_, exist_ok=True)
        p_ = f'{d_}/regime_ratio_anchorcmp_2row.{ext}'; fig.savefig(p_)
        print('written', p_)


if __name__ == '__main__':
    main()
