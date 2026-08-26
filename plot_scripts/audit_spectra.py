"""Energy spectra for a SINGLE-REGIME audit store (default: the Re=1000 audit).
Subplots: absolute E(k) with ground truth | ratio E/E_GT | per-band retention bars.

Edit the CONFIG block or pass CLI flags (flags override CONFIG). Row keys in the store:
  'LR', 'recon', '{model}|{cfg}|{strategy}'   e.g. 'base0|K3|rewardv2'
Wildcards allowed: 'base0|K3|*' (all strategies), 'base0|*|none' (depth ladder).
"""
import argparse, fnmatch, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib.pyplot as plt
import style

CONFIG = dict(
    npz='base_results/re1000_audit.npz', regime='1000',
    rows=['LR', 'recon', 'base0|K3|none', 'base0|K3|reward', 'base0|K3|rewardv2', 'base0|K3|all3v2'],
    panels=['abs', 'ratio', 'bands'],       # any subset / order
    kmax=127, ratio_ylim=(0.08, 4), bands_ylim=(0, 1.6),
    panel_w=5.0, panel_h=4.6, fontsize=10, dpi=140, legend_cols=1,
    styles={'LR': '--', 'recon': '--'},      # per-row line style overrides
    title='', out='monitoring/figs/audit_spectra.pdf',
)


def label(row):
    if row == 'LR': return 'raw LR'
    if row == 'recon': return 'base-DDIM recon'
    m, cfg, sg = row.split('|')
    return f"{style.MODEL.get(m, m)} {cfg} {style.STRATEGY.get(sg, sg)}"


def expand_rows(store, regime, patterns):
    have = sorted({k.split('||')[0].split('|', 1)[1] for k in store.files
                   if k.startswith(f'{regime}|') and k.endswith('||E') and '|GT' not in k})
    rows = []
    for pat in patterns:
        rows += [h for h in have if fnmatch.fnmatchcase(h, pat) and h not in rows]
    return rows


def make_figure(c):
    style.apply(c['fontsize'], c['dpi'])
    A = np.load(c['npz'], allow_pickle=True); R = c['regime']
    rows = expand_rows(A, R, c['rows'])
    Eg = np.asarray(A[f'{R}|GT||E']); k = np.arange(1, c['kmax'] + 1)
    fig, axes = plt.subplots(1, len(c['panels']), figsize=(c['panel_w'] * len(c['panels']), c['panel_h']),
                             constrained_layout=True, squeeze=False)
    for ax, pn in zip(axes[0], c['panels']):
        if pn in ('abs', 'ratio'): style.shade_bands(ax)
        if pn == 'abs': ax.loglog(k, Eg[1:c['kmax'] + 1], '-', color=style.GT_COLOR, lw=style.GT_LW, label='GROUND TRUTH', zorder=9)
        if pn == 'ratio': ax.axhline(1, color=style.GT_COLOR, lw=2.2, ls='--', zorder=6, label='GROUND TRUTH (=1)')
        for i, r in enumerate(rows):
            E = np.asarray(A[f'{R}|{r}||E']); col = style.PALETTE[i % len(style.PALETTE)]
            ls = c['styles'].get(r, '-')
            if pn == 'abs':
                ax.loglog(k, E[1:c['kmax'] + 1], ls, color=col, lw=1.7, label=label(r))
            elif pn == 'ratio':
                ax.semilogx(k, E[1:c['kmax'] + 1] / Eg[1:c['kmax'] + 1], ls, color=col, lw=1.7, label=label(r))
            elif pn == 'bands':
                eb = np.asarray(A[f'{R}|{r}||Eb']); w = 0.8 / len(rows)
                ax.bar(np.arange(5) + (i - len(rows) / 2 + 0.5) * w, np.clip(eb, 0, 50), w, color=col, label=label(r))
        if pn == 'abs':
            ax.set(xlabel='wavenumber k', ylabel='E(k)', title='energy spectrum (absolute)')
        elif pn == 'ratio':
            ax.set(yscale='log', ylim=c['ratio_ylim'], xlabel='wavenumber k',
                   ylabel='E(k) / E$_{GT}$(k)', title='ratio to ground truth')
        elif pn == 'bands':
            ax.axhline(1, color=style.INK, lw=1.1, ls='--')
            ax.set_xticks(range(5)); ax.set_xticklabels(style.BAND_LABELS)
            ax.set(ylim=c['bands_ylim'], ylabel='band retention E/E$_{GT}$', title='per-band retention')
    axes[0][0].legend(fontsize=c['fontsize'] - 2.5, ncol=c['legend_cols'], loc='lower left')
    if c['title']: fig.suptitle(c['title'], fontsize=c['fontsize'] + 2.5)
    fig.savefig(c['out']); plt.close(fig)
    print('written', c['out'], '| rows:', rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows'); ap.add_argument('--panels'); ap.add_argument('--out')
    ap.add_argument('--title'); ap.add_argument('--npz'); ap.add_argument('--regime')
    a = ap.parse_args()
    c = dict(CONFIG)
    if a.rows: c['rows'] = a.rows.split(',')
    if a.panels: c['panels'] = a.panels.split(',')
    for f in ('out', 'title', 'npz', 'regime'):
        if getattr(a, f) is not None: c[f] = getattr(a, f)
    make_figure(c)
