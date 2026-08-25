"""Thesis figure script (Re=1000 audit store, or any single-regime audit npz with the same
keys): energy spectra as subplots — absolute E(k) with ground truth, the ratio E/E_GT, and
optionally per-band retention bars — for any selection of rows.

  python -m src.ddpo_ft.plot_audit_spectra --rows "LR,recon,base0|K3|none,base0|K3|reward,base0|K3|all3v2" \
      --panels abs,ratio,bands --out monitoring/figs/my_spectra.png
Row names are the audit keys: 'LR', 'recon', '{model}|{cfg}|{strategy}'; wildcards allowed
('base0|K3|*' = all strategies of the base at K3, 'base0|*|none' = the depth ladder).
Adjust: --panels (any of abs,ratio,bands), --kmax, --bands-ylim, --ratio-ylim, --figw/--figh,
--fontsize, --dpi, --title, --legend-cols, --styles (override line styles per row: 'row=--').
"""
import argparse, os, fnmatch
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BANDS = [(1, 5), (5, 16), (16, 32), (32, 64), (64, 96)]
BL = ['[1,5)', '[5,16)', '[16,32)', '[32,64)', '[64,96)']
INK = '#232823'
PALETTE = ['#5d645d', '#2a78d6', '#7a4bd0', '#0f9e78', '#d4770a', '#b8399e', '#c22f4f',
           '#8a5cd6', '#28658a', '#9aa198']
STRAT_LAB = {'none': 'unguided', 'residual': 'residual dial', 'reward': 'dose dial v1',
             'placement': 'placement dial', 'all3': 'all three (v1)', 'rewardv2': 'dose dial v2',
             'all3v2': 'all three (v2)'}
MLAB = {'base0': 'base', 'r1k-449': 'fine-tuned Re=1000', 'st1k-599': 'steered-trained Re=1000',
        'pr1k-549': 'placement-reward Re=1000'}


def label(row):
    if row == 'LR': return 'raw LR'
    if row == 'recon': return 'base-DDIM recon'
    m, cfg, sg = row.split('|')
    return f"{MLAB.get(m, m)} {cfg} {STRAT_LAB.get(sg, sg)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='base_results/re1000_audit.npz')
    ap.add_argument('--regime', default='1000')
    ap.add_argument('--rows', default='LR,recon,base0|K3|none,base0|K3|reward,base0|K3|all3v2')
    ap.add_argument('--panels', default='abs,ratio,bands')
    ap.add_argument('--kmax', type=int, default=127)
    ap.add_argument('--ratio-ylim', default='0.08,4')
    ap.add_argument('--bands-ylim', default='0,1.6')
    ap.add_argument('--figw', type=float, default=5.0, help='width per panel')
    ap.add_argument('--figh', type=float, default=4.6)
    ap.add_argument('--fontsize', type=float, default=10)
    ap.add_argument('--dpi', type=int, default=140)
    ap.add_argument('--legend-cols', type=int, default=1)
    ap.add_argument('--styles', default='', help="per-row style overrides, e.g. 'LR=--,recon=:'")
    ap.add_argument('--title', default='')
    ap.add_argument('--out', default='monitoring/figs/audit_spectra.png')
    a = ap.parse_args()
    plt.rcParams.update({'font.size': a.fontsize, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'savefig.dpi': a.dpi, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.color': '#d8d6cd', 'grid.linewidth': 0.5,
                         'legend.frameon': False})
    A = np.load(a.npz, allow_pickle=True)
    R = a.regime
    all_rows = sorted({k.split('||')[0].split('|', 1)[1] for k in A.files if k.endswith('||E') and '|GT' not in k})
    rows = []
    for pat in a.rows.split(','):
        hits = [r for r in all_rows if fnmatch.fnmatchcase(r, pat)]
        rows += [h for h in hits if h not in rows]
    styles = dict(kv.split('=') for kv in a.styles.split(',') if '=' in kv)
    Eg = np.asarray(A[f'{R}|GT||E']); k = np.arange(1, a.kmax + 1)
    panels = a.panels.split(',')
    fig, axes = plt.subplots(1, len(panels), figsize=(a.figw * len(panels), a.figh),
                             constrained_layout=True, squeeze=False)
    axes = axes[0]
    for pi, pn in enumerate(panels):
        ax = axes[pi]
        if pn in ('abs', 'ratio'):
            ax.axvspan(16, 32, color='#f8e8d0', zorder=0); ax.axvspan(32, 96, color='#e3edf3', zorder=0)
        if pn == 'abs':
            ax.loglog(k, Eg[1:a.kmax + 1], '-', color=INK, lw=2.6, label='ground truth')
        if pn == 'ratio':
            ax.axhline(1, color=INK, lw=1.3, ls='--', zorder=6)
        for i, r in enumerate(rows):
            E = np.asarray(A[f'{R}|{r}||E']); c = PALETTE[i % len(PALETTE)]
            sty = styles.get(r, '--' if r in ('LR', 'recon') else '-')
            if pn == 'abs':
                ax.loglog(k, E[1:a.kmax + 1], sty, color=c, lw=1.7, label=label(r))
            elif pn == 'ratio':
                ax.semilogx(k, E[1:a.kmax + 1] / Eg[1:a.kmax + 1], sty, color=c, lw=1.7, label=label(r))
            elif pn == 'bands':
                eb = np.asarray(A[f'{R}|{r}||Eb'])
                w = 0.8 / len(rows)
                ax.bar(np.arange(5) + (i - len(rows) / 2 + 0.5) * w, np.clip(eb, 0, 50), w, color=c, label=label(r))
        if pn == 'abs':
            ax.set(xlabel='wavenumber k', ylabel='E(k)', title='energy spectrum (absolute)')
        elif pn == 'ratio':
            lo, hi = (float(x) for x in a.ratio_ylim.split(','))
            ax.set(yscale='log', ylim=(lo, hi), xlabel='wavenumber k', ylabel='E(k) / E$_{GT}$(k)',
                   title='ratio to ground truth')
        elif pn == 'bands':
            lo, hi = (float(x) for x in a.bands_ylim.split(','))
            ax.axhline(1, color=INK, lw=1.1, ls='--')
            ax.set_xticks(range(5)); ax.set_xticklabels(BL); ax.set(ylim=(lo, hi), ylabel='band retention E/E$_{GT}$',
                                                                     title='per-band retention')
    axes[0].legend(fontsize=a.fontsize - 2.5, ncol=a.legend_cols, loc='lower left')
    if a.title: fig.suptitle(a.title, fontsize=a.fontsize + 2.5)
    fig.savefig(a.out); plt.close(fig)
    print('written', a.out, 'rows:', rows)


if __name__ == '__main__':
    main()
