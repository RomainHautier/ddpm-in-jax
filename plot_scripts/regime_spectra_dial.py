"""Mean energy spectra and ratio to ground truth, every regime, for the base model and the
matched Re=1000 fine-tune, each with and without the matched dial.

Colour encodes the MODEL, linestyle encodes the DIAL (dashed = unguided, solid = dialled), so a
pair of curves in one colour shows what the dial did to that model at that regime.

  JAX_PLATFORMS=cpu python plot_scripts/regime_spectra_dial.py            # both figures
  ... --absolute        only the raw E(k) grid
  ... --ratio           only the ratio grid
  ... --regimes 1000,2000,8000 --out docs/figs_overleaf/foo.pdf
Edit CONFIG below to restyle.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib.pyplot as plt
import style

CONFIG = dict(
    regimes=[1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
    kmax=127, ratio_ylim=(0.015, 2.6),
    panel_w=4.2, panel_h=3.3, fontsize=style.BASE_FS, dpi=150,
    outdir='docs/figs_overleaf', pngdir='plotting/figs',
)

# (model key, dial tag, label, colour, linestyle)
PRESETS = {
    'matched': [
        ('base0',     'mop0.2_0',  'base, unguided',      '#28658a', '--'),
        ('base0',     'tapp0.2_3', 'base + dial',         '#28658a', '-'),
        ('mt1k-0499', 'mop0.2_0',  'fine-tune, unguided', '#c22f4f', '--'),
        ('mt1k-0499', 'tapp0.2_3', 'fine-tune + dial',    '#c22f4f', '-'),
    ],
    # the GATED dial optimises a DIFFERENT objective, so it gets its own figure rather than a
    # fifth curve on the matched comparison
    # in-distribution ability: each panel shows the model TRAINED THERE, against the Re=1000
    # fine-tune carried in and the base+dial
    'home': [
        ('base0',     'mop0.2_0',  'base, unguided',                   '#28658a', '--'),
        ('base0',     'tapp0.2_3', 'base + dial',                      '#28658a', '-'),
        ('mt1k-0499', 'mop0.2_0',  'Re=1000 fine-tune, unguided',      '#c22f4f', '--'),
        ('mt1k-0499', 'tapp0.2_3', 'Re=1000 fine-tune + dial',         '#c22f4f', '-'),
        ('@home',     'mop0.2_0',  "the regime's OWN fine-tune",       '#0f9e78', '-'),
        ('@fix',      'mop0.2_0',  'Re=8000 fine-tune, repaired',      '#8a5cd6', '-'),
    ],
    'gate': [
        ('base0',     'mop0.2_0',   'base, unguided',         '#9aa198', '--'),
        ('base0',     'v7bandgate', 'base + gated dial',      '#0f9e78', '-'),
        ('mt1k-0499', 'mop0.2_0',   'fine-tune, unguided',    '#d8a0aa', '--'),
        ('mt1k-0499', 'v7bandgate', 'fine-tune + gated dial', '#28658a', '-'),
    ],
}
CURVES = PRESETS['matched']


# A regime's OWN fine-tune, where one exists. '@home' resolves per panel, so a single curve can
# mean "the model trained here" rather than one fixed checkpoint carried everywhere.
HOME = {2000: 'mt2k-0599', 8000: 'mt8k-0549'}   # 1000 omitted: mt1k is already the
                                               # red curve, and is at home there
HOME_FIX = {8000: 'r8kp02-0599'}       # the repaired Re=8000 run (pde weight 0.2)


def resolve(model, R):
    if model == '@home': return HOME.get(R)
    if model == '@fix': return HOME_FIX.get(R)
    return model


def store(R):
    return ('base_results/re1000_audit.npz' if R == 1000
            else f'base_results/regime_audit_re{R}.npz')


def make(c, absolute, CURVES=None):
    CURVES = CURVES if CURVES is not None else PRESETS['matched']
    style.apply(c['fontsize'], c['dpi'])
    regs = [R for R in c['regimes'] if os.path.exists(store(R))]
    k = np.arange(1, c['kmax'] + 1)
    ncol = c.get('ncol') or (3 if len(regs) > 4 else min(len(regs), 2))
    nrow = int(np.ceil(len(regs) / ncol))
    ph = c['panel_h'] * (1.12 if nrow == 1 else 1.0)   # a single row reads squat otherwise
    fig, axes = plt.subplots(nrow, ncol, figsize=(c['panel_w'] * ncol, ph * nrow),
                             constrained_layout=True, squeeze=False, sharey=not absolute)
    drawn = []
    for i, R in enumerate(regs):
        ax = axes[i // ncol][i % ncol]
        S = np.load(store(R), allow_pickle=True); F = set(S.files)
        Eg = np.asarray(S[f'{R}|GT||E'])[1:c['kmax'] + 1]
        style.shade_bands(ax)
        if absolute:
            ax.loglog(k, Eg, '-', color=style.GT_COLOR, lw=style.GT_LW,
                      label='ground truth', zorder=6)
        else:
            ax.axhline(1, color=style.GT_COLOR, lw=1.3, ls='-', zorder=6)
        for m0, sg, lab, col, ls in CURVES:
            m = resolve(m0, R)
            if m is None: continue
            key = f'{R}|{m}|K3|{sg}||E'
            if key not in F: continue
            E = np.asarray(S[key])[1:c['kmax'] + 1]
            (ax.loglog if absolute else ax.semilogx)(
                k, E if absolute else E / Eg, ls, color=col, lw=1.7, label=lab, zorder=4)
            if lab not in drawn: drawn.append(lab)
        if not absolute:
            ax.set(yscale='log', ylim=c['ratio_ylim'])
            ax.set_yticks([0.03, 0.1, 0.3, 1, 2])
            ax.set_yticklabels(['0.03', '0.1', '0.3', '1', '2'])
        ax.axvline(96, color='#9aa198', lw=0.8, ls=':', zorder=1)
        ax.set_title(f'Re = {R}', fontsize=style.TITLE_FS)
        if i // ncol == nrow - 1: ax.set_xlabel('wavenumber $k$')
        if i % ncol == 0:
            ax.set_ylabel('$E(k)$' if absolute else '$E(k)\\,/\\,E_{\\mathrm{GT}}(k)$')
    for j in range(len(regs), nrow * ncol): axes[j // ncol][j % ncol].axis('off')
    h = ([plt.Line2D([], [], color=style.GT_COLOR, lw=style.GT_LW, label='ground truth')]
         if absolute else [plt.Line2D([], [], color=style.GT_COLOR, lw=1.3, label='ground truth')])
    h += [plt.Line2D([], [], color=col, ls=ls, lw=1.7, label=lab)
          for _, _, lab, col, ls in CURVES if lab in drawn]
    axes[0][0].legend(handles=h, fontsize=style.LEG_FS,
                      loc='lower left' if absolute else 'lower center')
    what = 'Mean vorticity spectrum' if absolute else 'Ratio to ground truth'
    where = 'every regime' if len(regs) >= 9 else ', '.join(f'Re={r}' for r in regs)
    fig.suptitle(f'{what}, {where}', fontsize=style.SUP_FS)
    name = ('regime_spectra_dial' if absolute else 'regime_ratio_dial') + c.get('suffix','')
    for d, ext in ((c['outdir'], 'pdf'), (c['pngdir'], 'png')):
        os.makedirs(d, exist_ok=True)
        p = f'{d}/{name}.{ext}'; fig.savefig(p); print('  written', p)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--regimes'); ap.add_argument('--absolute', action='store_true')
    ap.add_argument('--ratio', action='store_true')
    ap.add_argument('--preset', default='matched', choices=sorted(PRESETS))
    ap.add_argument('--ncol', type=int, help='panels per row (default 3 when 9 regimes)')
    ap.add_argument('--tag', default='', help='filename suffix so a regime SUBSET does not '
                                              'overwrite the full nine-regime figure')
    a = ap.parse_args()
    c = dict(CONFIG)
    CURVES = PRESETS[a.preset]
    c['suffix'] = ('' if a.preset == 'matched' else f'_{a.preset}') + (f'_{a.tag}' if a.tag else '')
    if a.regimes: c['regimes'] = [int(x) for x in a.regimes.split(',')]
    if a.ncol: c['ncol'] = a.ncol
    if a.absolute or not a.ratio: make(c, True, CURVES)
    if a.ratio or not a.absolute: make(c, False, CURVES)
