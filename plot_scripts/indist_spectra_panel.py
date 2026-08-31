"""Re=1000, in distribution: mean vorticity spectrum, ratio to ground truth, and per-band energy
retention, side by side for the four configurations (base and matched fine-tune, each with and
without the matched dial).

Colour encodes the MODEL, linestyle/hatch encodes the DIAL.

  JAX_PLATFORMS=cpu python plot_scripts/indist_spectra_panel.py
  ... --re 2000            # same panel at another regime
Edit CONFIG / CURVES below to restyle.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib.pyplot as plt
import style

CONFIG = dict(re=1000, kmax=127, ratio_ylim=(0.25, 1.6),
              panel_w=4.6, panel_h=3.6, fontsize=style.BASE_FS, dpi=150,
              outdir='docs/figs_overleaf', pngdir='plotting/figs')

# (model key, dial tag, label, colour, linestyle, hatch)
# linestyle: dashed = unguided, dotted = dial with the HARD band edge, solid = dial with the
# tapered edge. The tapered dial is the current version; the hard-edge rows are kept alongside so
# the band-edge artifact it removes is visible rather than asserted.
PRESETS = {
    # the dial comparison: hard band edge vs tapered, on both models
    'dial': [
        ('base0',     'mop0.2_0', 'base, unguided',              '#28658a', '--', '///'),
        ('base0',     'mop0.2_3', 'base + dial (hard edge)',     '#7fb0cc', ':',  '\\\\'),
        ('base0',     'tapp0.2_3', 'base + dial (tapered)',      '#28658a', '-',  None),
        ('mt1k-0499', 'mop0.2_0', 'fine-tune, unguided',         '#c22f4f', '--', '///'),
        ('mt1k-0499', 'mop0.2_3', 'fine-tune + dial (hard edge)','#e08a9c', ':',  '\\\\'),
        ('mt1k-0499', 'tapp0.2_3', 'fine-tune + dial (tapered)', '#c22f4f', '-',  None),
    ],
    # the fine-tune comparison: what the weights alone do, no guidance at sampling time
    'finetunes': [
        ('base0',     'mop0.2_0', 'base, unguided',                  '#9aa198', '--', '///'),
        ('mt1k-0499', 'mop0.2_0', 'matched fine-tune',               '#c22f4f', '-',  '\\\\'),
        ('gt1k-0099', 'mop0.2_0', 'GATED-dose fine-tune',            '#0f9e78', '-',  None),
        ('base0',     'v7bandgate', 'base + gated dial (reference)', '#28658a', ':',  '...'),
    ],
}
CURVES = PRESETS['dial']


def make(c, CURVES=None):
    CURVES = CURVES if CURVES is not None else globals()['CURVES']
    style.apply(c['fontsize'], c['dpi'])
    R = c['re']
    p = ('base_results/re1000_audit.npz' if R == 1000
         else f'base_results/regime_audit_re{R}.npz')
    S = np.load(p, allow_pickle=True); F = set(S.files)
    k = np.arange(1, c['kmax'] + 1)
    Eg = np.asarray(S[f'{R}|GT||E'])[1:c['kmax'] + 1]
    fig, ax = plt.subplots(1, 3, figsize=(c['panel_w'] * 3, c['panel_h']), constrained_layout=True)

    # ---- panels 1 and 2: spectrum and ratio ----
    style.shade_bands(ax[0]); style.shade_bands(ax[1])
    ax[0].loglog(k, Eg, '-', color=style.GT_COLOR, lw=style.GT_LW, label='ground truth', zorder=6)
    ax[1].axhline(1, color=style.GT_COLOR, lw=1.3, zorder=6)
    present = []
    for m, sg, lab, col, ls, _ in CURVES:
        key = f'{R}|{m}|K3|{sg}||E'
        if key not in F: continue
        E = np.asarray(S[key])[1:c['kmax'] + 1]
        ax[0].loglog(k, E, ls, color=col, lw=1.7, label=lab, zorder=4)
        ax[1].semilogx(k, E / Eg, ls, color=col, lw=1.7, zorder=4)
        present.append((m, sg, lab, col, ls, _))
    for a in ax[:2]:
        a.axvline(96, color='#9aa198', lw=0.8, ls=':', zorder=1)
        a.set_xlabel('wavenumber $k$')
    # The ratio panel stops at the reward band edge. Beyond k=96 no term acts, so the curves are
    # uncontrolled - and on a log axis k=96..127 is a few pixels wide, so they crush together into
    # something that reads as a rendering fault. The absolute panel keeps the full range.
    ax[1].set_xlim(1, 96)
    ax[0].axvspan(96, c['kmax'], color='#efeeea', zorder=0)
    ax[0].set_ylabel('$E(k)$'); ax[0].set_title('mean vorticity spectrum', fontsize=style.TITLE_FS)
    ax[1].set(ylim=c['ratio_ylim'])
    ax[1].set_ylabel('$E(k)\\,/\\,E_{\\mathrm{GT}}(k)$')
    ax[1].set_title('ratio to ground truth', fontsize=style.TITLE_FS)
    ax[0].legend(fontsize=style.LEG_FS, loc='lower left')

    # ---- panel 3: per-band retention ----
    nb = len(style.BAND_LABELS); x = np.arange(nb); w = 0.8 / max(len(present), 1)
    ax[2].axhline(1, color=style.GT_COLOR, lw=1.3, zorder=5)
    for i, (m, sg, lab, col, ls, hatch) in enumerate(present):
        key = f'{R}|{m}|K3|{sg}||Eb'
        if key not in F: continue
        Eb = np.asarray(S[key])[:nb]
        ax[2].bar(x + (i - (len(present) - 1) / 2) * w, Eb, w * 0.92, label=lab,
                  color=col, alpha=0.45 if hatch else 0.9, hatch=hatch,
                  edgecolor=col, linewidth=0.9, zorder=3)
    ax[2].set_xticks(x); ax[2].set_xticklabels(style.BAND_LABELS)
    ax[2].set_xlabel('wavenumber band')
    ax[2].set_ylabel('$E_{\\mathrm{band}}\\,/\\,E_{\\mathrm{GT,band}}$')
    ax[2].set_title('energy retention per band', fontsize=style.TITLE_FS)
    ax[2].set_ylim(0, 1.32); ax[2].grid(axis='x', visible=False)
    # no legend here: panel 1 already carries the same colour/label key, and six
    # entries in this panel sit on top of the bars.

    fig.suptitle(c.get('title') or f'In distribution (Re={R})', fontsize=style.SUP_FS)
    name = f"indist_spectra_panel_re{R}{c.get('suffix','')}"
    for d, ext in ((c['outdir'], 'pdf'), (c['pngdir'], 'png')):
        os.makedirs(d, exist_ok=True)
        q = f'{d}/{name}.{ext}'; fig.savefig(q); print('  written', q)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--re', type=int)
    ap.add_argument('--preset', default='dial', choices=sorted(PRESETS))
    ap.add_argument('--suffix', default='')
    a = ap.parse_args(); c = dict(CONFIG)
    if a.re: c['re'] = a.re
    CURVES = PRESETS[a.preset]
    c['suffix'] = a.suffix or ('' if a.preset == 'dial' else f'_{a.preset}')
    c['title'] = {
        'dial': f'In distribution (Re={c["re"]}) — base and matched fine-tune, unguided and with '
                'the matched dial (hard band edge vs tapered)',
        'finetunes': f'In distribution (Re={c["re"]}) — what the WEIGHTS alone do: the matched '
                     'fine-tune against the gated-dose fine-tune, no guidance at sampling time',
    }[a.preset]
    make(c, CURVES)
