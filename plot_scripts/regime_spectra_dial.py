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
    # the lower half of the ladder: what the regime's own model does, against the two
    # high-regime specialists carried down, all unguided K3
    'lowxfer': [
        ('base0',       'mop0.2_0', 'base',          '#9aa198', '--'),
        ('r8kp02-0599', 'mop0.2_0', 'Re=8000 ft',    '#8a5cd6', '-'),
        ('r4kp02-0599', 'mop0.2_0', 'Re=4000 ft',    '#1ba3c6', '-'),
        ('@home',       'mop0.2_0', 'in-regime ft',  '#0f9e78', '-'),
    ],
    # transfer, spectra form: the two extreme specialists carried everywhere, against the
    # gated in-regime model where one exists
    'xfer': [
        ('base0',       'mop0.2_0',   'base',                '#9aa198', '--'),
        ('mt1k-0499',   'mop0.2_0',   'Re=1000 ft',          '#c22f4f', '-'),
        ('r8kp02-0599', 'mop0.2_0',   'Re=8000 ft',          '#8a5cd6', '-'),
        ('@home',       'v7bandgate', 'in-regime ft + gate', '#0f9e78', '-'),
    ],
    # the Re=4000 specialist everywhere, base for reference, dial variant
    'r4k': [
        ('base0',       'mop0.2_0',   'base',                 '#9aa198', '--'),
        ('r4kp02-0599', 'mop0.2_0',   'Re=4000 ft, unguided', '#1ba3c6', '-'),
        ('r4kp02-0599', 'tapp0.2_3',  'Re=4000 ft + dial',    '#8a5cd6', '-'),
        ('r4kp02-0599', 'v7bandgate', 'Re=4000 ft + gate',    '#0f9e78', '-'),
    ],
    # the Re=8000 specialist everywhere, with the base for reference and the gate variant
    'r8k': [
        ('base0',       'mop0.2_0',   'base',                 '#9aa198', '--'),
        ('r8kp02-0599', 'mop0.2_0',   'Re=8000 ft, unguided', '#8a5cd6', '-'),
        ('r8kp02-0599', 'v7bandgate', 'Re=8000 ft + gate',    '#0f9e78', '-'),
    ],
    # the pure-gate Re=1000 fine-tune carried upward under the v8 gate, against the
    # plain base under the SAME gate
    'pg1k': [
        ('base0',     'mop0.2_0',  'base, unguided',           '#9aa198', '--'),
        ('pg1k-0599', 'mop0.2_0',  'pure-gate ft, unguided',   '#d8a0aa', '--'),
        ('base0',     'v8lowband', 'base + gate',              '#0f9e78', '-'),
        ('pg1k-0599', 'v8lowband', 'pure-gate ft + gate',      '#c22f4f', '-'),
        ('@spec',     'mop0.2_0',  'in-regime pure-gate ft, unguided', '#b79ce8', '--'),
        ('@spec',     'v8lowband', 'in-regime pure-gate ft + gate', '#8a5cd6', '-'),
        ('@up4',      'v8lowband', 'pg4k + gate (carried up)', '#d4770a', '-'),
    ],
    # three reward generations at each specialist's HOME regime, unguided: where the
    # spectral shape comes from
    'anchorcmp': [
        ('base0', 'mop0.2_0', 'base, unguided',                    '#9aa198', '--'),
        ('@leg',  'mop0.2_0', 'legacy reward (per-shell push)',    '#0f9e78', '-'),
        ('@pg',   'mop0.2_0', 'pure gate (anchor 0.5)',            '#8a5cd6', '-'),
        ('@sa',   'mop0.2_0', 'strong anchor (3.0)',               '#c22f4f', '-'),
    ],
    # the Re=8000 pure-gate specialist carried DOWN: fixed K3 (unguided + gate), the
    # chain-adapted config, against the best-known setting at each regime
    'pg8kxfer': [
        ('pg8k-0499', 'mop0.2_0',        'pg8k, unguided K3',       '#b79ce8', '--'),
        ('pg8k-0499', 'v8lowband',       'pg8k + gate, K3',         '#8a5cd6', '-'),
        ('pg8k-0499', '@chain:mop0.2_0', 'pg8k, adapted chain',     '#d4650f', '-'),
        ('pg8k-0499', '@chain:mop0.2_3', 'pg8k, chain + dial',      '#e8a94f', '--'),
        ('@best8',    'v8lowband',       'best in-regime setting',  '#0f9e78', '-'),
    ],
    # the Re=4000 pure-gate specialist both ways, same comparison
    'pg4kxfer': [
        ('pg4k-0299', 'mop0.2_0',        'pg4k, unguided K3',       '#e8a3bd', '--'),
        ('pg4k-0299', 'v8lowband',       'pg4k + gate, K3',         '#c22f4f', '-'),
        ('pg4k-0299', '@chain:mop0.2_0', 'pg4k, adapted chain',     '#d4650f', '-'),
        ('pg4k-0299', '@chain:mop0.2_3', 'pg4k, chain + dial',      '#e8a94f', '--'),
        ('base0',     'v8lowband',       'base + gate',             '#0f9e78', '-'),
        ('pg1k-0599', 'v8lowband',       'pg1k + gate',             '#28658a', '-'),
    ],
    # the plain BASE carried upward by the gated dial alone: every regime above home
    'upgate': [
        ('base0', 'mop0.2_0',   'base, unguided',    '#9aa198', '--'),
        ('base0', 'v7bandgate', 'base + gated dial', '#0f9e78', '-'),
    ],
    # Re=1000 canonical gate view: the low-band (v8) gate is THE gate at Re=1000
    'gate1k': [
        ('base0',     'mop0.2_0',  'base, unguided',         '#9aa198', '--'),
        ('base0',     'v8lowband', 'base + gated dial',      '#0f9e78', '-'),
        ('mt1k-0499', 'mop0.2_0',  'fine-tune, unguided',    '#d8a0aa', '--'),
        ('mt1k-0499', 'v8lowband', 'fine-tune + gated dial', '#28658a', '-'),
        ('gt1k-0199', 'tapp0.2_0', 'gated-dose fine-tune (weights)', '#b8399e', '-'),
    ],
    # ablation: the same per-band tolerance gate but steering to the ENSEMBLE target,
    # every sample pushed to the same dose
    'ensgate1k': [
        ('base0',     'mop0.2_0',  'base, unguided',            '#9aa198', '--'),
        ('base0',     'v7ensgate', 'base + ensemble gate',      '#0f9e78', '-'),
        ('mt1k-0499', 'mop0.2_0',  'fine-tune, unguided',       '#d8a0aa', '--'),
        ('mt1k-0499', 'v7ensgate', 'fine-tune + ensemble gate', '#28658a', '-'),
    ],
    'gate': [
        ('base0',     'mop0.2_0',   'base, unguided',         '#9aa198', '--'),
        ('base0',     'v7bandgate', 'base + gated dial',      '#0f9e78', '-'),
        ('mt1k-0499', 'mop0.2_0',   'fine-tune, unguided',    '#d8a0aa', '--'),
        ('mt1k-0499', 'v7bandgate', 'fine-tune + gated dial', '#28658a', '-'),
        ('gt1k-0199', 'tapp0.2_0',  'gated-dose fine-tune (weights)', '#b8399e', '-'),
    ],
}
CURVES = PRESETS['matched']


# A regime's OWN fine-tune, where one exists. '@home' resolves per panel, so a single curve can
# mean "the model trained here" rather than one fixed checkpoint carried everywhere.
HOME = {1000: 'mt1k-0499', 2000: 'mt2k-0599', 4000: 'r4kp02-0599',
        8000: 'r8kp02-0599'}   # 8000 is the REPAIRED run; the matched mt8k is excluded
HOME_FIX = {6000: 'r8kp02-0599',       # carried in as a transfer at 6000
            8000: 'r8kp02-0599'}       # the repaired Re=8000 run (pde weight 0.2)


SPEC = {4000: 'pg4k-0299', 8000: 'pg8k-0499'}
UP4 = {6000: 'pg4k-0299', 8000: 'pg4k-0299'}   # the Re=4000 specialist carried UP, gated   # the pure-gate specialist, drawn ONLY at its home regime
# reward-generation comparison at each specialist's home regime
LEGACY = {4000: 'r4kp02-0599', 8000: 'r8kp02-0599'}
PUREG = {4000: 'pg4k-0299', 8000: 'pg8k-0499'}
STRONG = {4000: 'sa4k-0399', 8000: 'sa8k-0599'}
# the published chain-ladder selections, for '@chain:' tagged curves
CHAIN_SEL = {1000: 'K100', 2000: 'K125', 3000: 'K150', 4000: 'K160'}
# best-known configuration per regime, EXCLUDING the model each figure studies
BEST8 = {1000: 'base0', 2000: 'pg1k-0599', 4000: 'pg4k-0299'}          # for the pg8k figure
BEST4 = {1000: 'base0', 2000: 'pg1k-0599', 6000: 'pg8k-0499', 8000: 'pg8k-0499'}  # for pg4k


def resolve(model, R):
    if model == '@home': return HOME.get(R)
    if model == '@fix': return HOME_FIX.get(R)
    if model == '@spec': return SPEC.get(R)
    if model == '@up4': return UP4.get(R)
    if model == '@best8': return BEST8.get(R)
    if model == '@best4': return BEST4.get(R)
    if model == '@leg': return LEGACY.get(R)
    if model == '@pg': return PUREG.get(R)
    if model == '@sa': return STRONG.get(R)
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
            ch = 'K3'
            if sg.startswith('@chain:'):        # per-regime selected chain config
                ch = CHAIN_SEL.get(R)
                if ch is None: continue
                sg = sg.split(':', 1)[1]
            key = f'{R}|{m}|{ch}|{sg}||E'
            if key not in F: continue
            E = np.asarray(S[key])[1:c['kmax'] + 1]
            (ax.loglog if absolute else ax.semilogx)(
                k, E if absolute else E / Eg, ls, color=col, lw=1.7, label=lab, zorder=4)
            if lab not in drawn: drawn.append(lab)
            if m0 in ('@best8', '@best4'):     # name the per-panel best configuration
                _nice = {'base0': 'base', 'pg1k-0599': 'pg1k', 'pg4k-0299': 'pg4k',
                         'pg8k-0499': 'pg8k'}.get(m, m)
                ax.text(.03, .05, f'best = {_nice} + gate', transform=ax.transAxes,
                        color=col, fontsize=style.LEG_FS - 1.5,
                        bbox=dict(fc='white', ec='none', alpha=.75))
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
    ap.add_argument('--ymax', type=float, help='upper ratio-panel limit (default 2.6)')
    ap.add_argument('--tag', default='', help='filename suffix so a regime SUBSET does not '
                                              'overwrite the full nine-regime figure')
    a = ap.parse_args()
    c = dict(CONFIG)
    CURVES = PRESETS[a.preset]
    c['suffix'] = ('' if a.preset == 'matched' else f'_{a.preset}') + (f'_{a.tag}' if a.tag else '')
    if a.regimes: c['regimes'] = [int(x) for x in a.regimes.split(',')]
    if a.ncol: c['ncol'] = a.ncol
    if a.ymax: c['ratio_ylim'] = (c['ratio_ylim'][0], a.ymax)
    if a.absolute or not a.ratio: make(c, True, CURVES)
    if a.ratio or not a.absolute: make(c, False, CURVES)
