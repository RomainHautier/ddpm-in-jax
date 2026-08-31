"""Per-triplet dose against each triplet's OWN ground-truth energy, every regime, 3x3.

Answers the question the ensemble spectra cannot: does a configuration put the right amount of
energy into EACH reconstruction, or only into the population mean? A configuration on the diagonal
tracks every triplet; a horizontal cloud gives every triplet the same dose regardless of its truth.

  JAX_PLATFORMS=cpu python plot_scripts/regime_pertriplet_gate.py [--preset gate|matched] [--band 32,96]
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib.pyplot as plt
import style

CONFIG = dict(regimes=[1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
              panel_w=3.6, panel_h=3.3, fontsize=style.BASE_FS, dpi=150,
              outdir='docs/figs_overleaf', pngdir='plotting/figs')
PRESETS = {
    # every model, unguided (solid marker) and with the TAPERED dial at lambda=3 (open marker),
    # the same dial for every model. Colours come from style.MODEL_COLOR.
    'allmodels': [('base0', 'mop0.2_0', 'base', style.MODEL_COLOR['base0'], 'o'),
                  ('base0', 'mop0.2_3', 'base + dial', style.MODEL_COLOR['base0'], 'o'),
                  ('mt1k-0499', 'mop0.2_0', 'Re=1000 ft', style.MODEL_COLOR['mt1k-0499'], 's'),
                  ('mt1k-0499', 'mop0.2_3', 'Re=1000 ft + dial', style.MODEL_COLOR['mt1k-0499'], 's'),
                  ('mt2k-0599', 'mop0.2_0', 'Re=2000 ft', style.MODEL_COLOR['mt2k-0599'], '^'),
                  ('mt2k-0599', 'tapp0.2_3', 'Re=2000 ft + dial', style.MODEL_COLOR['mt2k-0599'], '^'),
                  ('r8kp02-0599', 'mop0.2_0', 'Re=8000 ft', style.MODEL_COLOR['r8kp02-0599'], 'D'),
                  ('r8kp02-0599', 'tapp0.2_3', 'Re=8000 ft + dial', style.MODEL_COLOR['r8kp02-0599'], 'D')],
    'gate': [('base0', 'mop0.2_0', 'base, unguided', '#9aa198', 'o'),
             ('base0', 'v7bandgate', 'base + gated dial', '#0f9e78', '^'),
             ('mt1k-0499', 'v7bandgate', 'fine-tune + gated dial', '#28658a', 's')],
    # colour = model, filled/open marker = dialled/unguided, as in the other Re=1000 figures
    'matched': [('base0', 'mop0.2_0', 'base, unguided', '#7fb0cc', 'o'),
                ('base0', 'tapp0.2_3', 'base + dial', '#28658a', '^'),
                ('mt1k-0499', 'mop0.2_0', 'fine-tune, unguided', '#e08a9c', 's'),
                ('mt1k-0499', 'tapp0.2_3', 'fine-tune + dial', '#c22f4f', 'v')],
}
EDGES = [1, 5, 16, 32, 64, 96]


def store(R):
    return np.load('base_results/re1000_audit.npz' if R == 1000
                   else f'base_results/regime_audit_re{R}.npz', allow_pickle=True)


def make(c, curves, cols, band):
    style.apply(c['fontsize'], c['dpi'])
    regs = [R for R in c['regimes'] if os.path.exists(
        'base_results/re1000_audit.npz' if R == 1000 else f'base_results/regime_audit_re{R}.npz')]
    ncol = c.get('ncol', 3); nrow = int(np.ceil(len(regs) / ncol))
    if len(regs) == 1: c = {**c, 'panel_w': 6.4, 'panel_h': 5.6}
    fig, axes = plt.subplots(nrow, ncol, figsize=(c['panel_w'] * ncol, c['panel_h'] * nrow),
                             constrained_layout=True, squeeze=False)
    for i, R in enumerate(regs):
        ax = axes[i // ncol][i % ncol]
        S = store(R); F = set(S.files)
        # The GT array is the FULL audit pool (seqs 8-19 OOD, 120 triplets); the matched/gate rows
        # are the TEST pool only (12-19, 80). matched_objective keeps the same linspace frames, so
        # the 80 are exactly the seq>=12 subset of the 120 - mask the GT rather than assume.
        T_all = np.asarray(S[f'{R}|GT||psEb'])[:, cols].sum(1)
        sq = np.asarray(S[f'{R}|EVAL||seq']) if f'{R}|EVAL||seq' in F else None
        mask = None
        if sq is not None and len(sq) == len(T_all):
            mask = sq >= 12 if R != 1000 else np.ones(len(sq), bool)
        T = T_all
        lo, hi = T_all.min() * .7, T_all.max() * 1.4
        ax.fill_between([lo, hi], [lo * .8, hi * .8], [lo * 1.2, hi * 1.2],
                        color='k', alpha=.07, lw=0)
        ax.plot([lo, hi], [lo, hi], color=style.GT_COLOR, lw=1.8, zorder=5)
        stats = []
        for m, sg, lab, col, mk in curves:
            k = f'{R}|{m}|K3|{sg}||psEb'
            if k not in F: continue
            P = np.asarray(S[k])[:, cols].sum(1)
            T = T_all
            if len(P) != len(T):
                assert mask is not None and mask.sum() == len(P), (
                    f'cannot align {len(P)} rows to {len(T)} GT at Re={R}')
                T = T_all[mask]
            sl = np.polyfit(np.log(T), np.log(P), 1)[0]
            ib = np.mean(np.abs(P / T - 1) < .2) * 100
            dialled = not sg.endswith('_0')
            kw = (dict(facecolors='none', edgecolors=col, linewidths=.85, alpha=.75)
                  if dialled else dict(color=col, alpha=.5, lw=0))
            lg = f'{lab} — {sl:.2f}, {ib:.0f}%' if len(curves) > 6 else lab
            ax.scatter(T, P, s=13 if dialled else 10, marker=mk,
                       label=lg if i == 0 else None, **kw)
            stats.append((f'{sl:.2f}   {ib:3.0f}%', col))
        ax.set(xscale='log', yscale='log', xlim=(lo, hi), ylim=(lo * .35, hi * 2.2))
        # slope and in-band change with regime, so each panel carries its own
        n_st = len(stats)
        if len(curves) > 6: stats = []          # already in the legend; a 10-row box hides the data
        if stats: ax.add_patch(plt.Rectangle((0.545, 0.035), 0.44, 0.062 * (n_st + 1) + 0.02,
                                   transform=ax.transAxes, fc='white', ec='#d8d6cd',
                                   lw=.7, alpha=.88, zorder=8))
        ytop = 0.035 + 0.062 * (n_st + 1) - 0.012
        if stats: ax.text(0.965, ytop, 'slope  in-band', transform=ax.transAxes, ha='right', va='top',
                fontsize=style.LEG_FS - 1.5, color='#5d645d', zorder=9)
        for j, (txt, col) in enumerate(stats):
            ax.text(0.965, ytop - 0.062 * (j + 1), txt, transform=ax.transAxes, ha='right',
                    va='top', fontsize=style.LEG_FS - 1, color=col, family='monospace', zorder=9)
        if len(regs) > 1:            # a single-regime figure says it in the suptitle already
            ax.set_title(f'Re = {R}', fontsize=style.TITLE_FS)
        if i // ncol == nrow - 1: ax.set_xlabel(f"that triplet's own GT energy, {band}")
        if i % ncol == 0: ax.set_ylabel(f'reconstructed energy, {band}')
    for j in range(len(regs), nrow * ncol): axes[j // ncol][j % ncol].axis('off')
    axes[0][0].legend(fontsize=style.LEG_FS - 1.5, loc='upper left', ncol=1,
                      title='slope, in-band' if len(curves) > 6 else None,
                      title_fontsize=style.LEG_FS - 2)
    where = 'every regime' if len(regs) >= 9 else ', '.join(f'Re={r}' for r in regs)
    if len(regs) > 1:
        fig.suptitle(f'Per-triplet dose over {band}, {where}  —  filled = unguided, '
                     'open = dialled', fontsize=style.SUP_FS)
    else:
        fig.suptitle(f'Per-triplet dose over {band} at {where}\n'
                     'filled = unguided, open = dialled', fontsize=style.TITLE_FS)
    return fig


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', default='gate', choices=sorted(PRESETS))
    ap.add_argument('--band', default='32,96')
    ap.add_argument('--regimes'); ap.add_argument('--ncol', type=int)
    ap.add_argument('--tag', default='', help='filename suffix so a regime subset does not '
                                              'overwrite the nine-regime figure')
    a = ap.parse_args(); c = dict(CONFIG)
    if a.regimes: c['regimes'] = [int(x) for x in a.regimes.split(',')]
    if a.ncol: c['ncol'] = a.ncol
    blo, bhi = (int(x) for x in a.band.split(','))
    cols = [i for i in range(5) if EDGES[i] >= blo and EDGES[i + 1] <= bhi]
    assert cols, f'band {a.band} does not align with {EDGES}'
    band = f'[{blo},{bhi})'
    fig = make(c, PRESETS[a.preset], cols, band)
    suf = '' if (blo, bhi) == (32, 96) else f'_k{blo}'
    name = f'regime_pertriplet_{a.preset}{suf}' + (f'_{a.tag}' if a.tag else '')
    for d, ext in ((c['outdir'], 'pdf'), (c['pngdir'], 'png')):
        os.makedirs(d, exist_ok=True)
        p = f'{d}/{name}.{ext}'; fig.savefig(p); print('  written', p)
