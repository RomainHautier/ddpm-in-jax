"""Per-band energy retention against Reynolds number: one panel per wavenumber band, four
configurations (base and matched fine-tune, each with and without the matched dial).

  JAX_PLATFORMS=cpu python plot_scripts/perband_vs_re.py
Edit CONFIG / CURVES to restyle.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib.pyplot as plt
import style

CONFIG = dict(regimes=[1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
              panel_w=3.0, panel_h=2.9, fontsize=style.BASE_FS, dpi=150,
              outdir='docs/figs_overleaf', pngdir='plotting/figs')
CURVES = [('base0', 'mop0.2_0', 'base, unguided', '#28658a', '--', 'o'),
          ('base0', 'tapp0.2_3', 'base + dial', '#28658a', '-', 'o'),
          ('mt1k-0499', 'mop0.2_0', 'fine-tune, unguided', '#c22f4f', '--', 's'),
          ('mt1k-0499', 'tapp0.2_3', 'fine-tune + dial', '#c22f4f', '-', 's')]


def make(c):
    style.apply(c['fontsize'], c['dpi'])
    nb = len(style.BAND_LABELS)
    fig, ax = plt.subplots(1, nb, figsize=(c['panel_w'] * nb, c['panel_h']),
                           constrained_layout=True, sharey=True)
    for m, sg, lab, col, ls, mk in CURVES:
        for b in range(nb):
            xs, ys = [], []
            for R in c['regimes']:
                p = ('base_results/re1000_audit.npz' if R == 1000
                     else f'base_results/regime_audit_re{R}.npz')
                if not os.path.exists(p): continue
                S = np.load(p, allow_pickle=True)
                k = f'{R}|{m}|K3|{sg}||Eb'
                if k not in S.files: continue
                xs.append(R); ys.append(float(np.asarray(S[k])[b]))
            if xs:
                # base+dial is drawn wider and under: in the top band the two dialled curves
                # coincide almost exactly, and a same-width line would hide one completely.
                wide = (m == 'base0' and sg.endswith('_3'))
                ax[b].plot(xs, ys, ls, color=col, lw=3.2 if wide else 1.7, marker=mk,
                           ms=4.6 if wide else 3.4, alpha=0.55 if wide else 1.0,
                           zorder=2 if wide else 3, label=lab if b == 0 else None)
    for b in range(nb):
        ax[b].axhline(1, color=style.GT_COLOR, lw=1.3, zorder=5)
        style.re_axis(ax[b]); ax[b].set_xlabel('Reynolds number')
        lab = style.BAND_LABELS[b]
        note = '  (below coarse Nyquist)' if b <= 2 else '  (above)'
        ax[b].set_title(f'$k \\in$ {lab}', fontsize=style.TITLE_FS)
        ax[b].text(.5, .02, note.strip(), transform=ax[b].transAxes, ha='center', va='bottom',
                   fontsize=style.LEG_FS - 2, color='#9aa198')
        ax[b].set_ylim(0, 1.32)
    ax[0].set_ylabel('$E_{\\mathrm{band}}\\,/\\,E_{\\mathrm{GT,band}}$')
    ax[0].legend(fontsize=style.LEG_FS, loc='lower left')
    fig.suptitle('Energy retention band by band, across every regime', fontsize=style.SUP_FS)
    for d, ext in ((c['outdir'], 'pdf'), (c['pngdir'], 'png')):
        os.makedirs(d, exist_ok=True)
        q = f"{d}/perband_vs_re{('_' + c['tag']) if c.get('tag') else ''}.{ext}"
        fig.savefig(q); print('  written', q)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--regimes')
    ap.add_argument('--tag', default='')
    a = ap.parse_args(); c = dict(CONFIG)
    if a.regimes: c['regimes'] = [int(x) for x in a.regimes.split(',')]
    c['tag'] = a.tag
    make(c)
