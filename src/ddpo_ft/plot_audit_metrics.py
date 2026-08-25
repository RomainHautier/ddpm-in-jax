"""Thesis figure script (single-regime audit store): metric subplots for any selection of rows —
retention, placement, MSE, PDE residual, low-k, any k-band — as bars with the per-sample
distribution overlaid as dots when the store carries ps_* arrays.

  python -m src.ddpo_ft.plot_audit_metrics --rows "recon,base0|K3|*" \
      --metrics ret,place,mse,resid --out monitoring/figs/base_dials.png
Metrics: ret, place, mse, resid (ratio to GT floor), lowk, kstar, b{i} (band i in 0..4 of
[1,5),[5,16),[16,32),[32,64),[64,96)), pl{i} (per-band placement, i in 2..4).
Adjust: --persample (overlay dots; default on when available), --figw/--figh, --fontsize, --dpi,
--rotate (tick label angle), --logy (comma list), --title.
"""
import argparse, os, sys, fnmatch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_audit_spectra import label, PALETTE, INK, BL

LAB = {'ret': 'retention [32,96)', 'place': 'placement', 'mse': 'MSE', 'resid': 'PDE residual / GT floor',
       'lowk': 'large-scale energy / GT', 'kstar': 'effective resolution k*'}
PS = {'ret': 'ps_ret_paired', 'place': 'ps_place', 'mse': 'ps_mse', 'resid': 'ps_resid'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='base_results/re1000_audit.npz')
    ap.add_argument('--regime', default='1000')
    ap.add_argument('--rows', default='recon,base0|K3|*')
    ap.add_argument('--metrics', default='ret,place,mse,resid')
    ap.add_argument('--persample', default='auto', choices=['auto', 'on', 'off'])
    ap.add_argument('--figw', type=float, default=4.2)
    ap.add_argument('--figh', type=float, default=4.4)
    ap.add_argument('--fontsize', type=float, default=10)
    ap.add_argument('--dpi', type=int, default=140)
    ap.add_argument('--rotate', type=float, default=55)
    ap.add_argument('--logy', default='resid')
    ap.add_argument('--title', default='')
    ap.add_argument('--out', default='monitoring/figs/audit_metrics.png')
    a = ap.parse_args()
    plt.rcParams.update({'font.size': a.fontsize, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'savefig.dpi': a.dpi, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.color': '#d8d6cd', 'grid.linewidth': 0.5,
                         'legend.frameon': False})
    A = np.load(a.npz, allow_pickle=True); R = a.regime
    all_rows = sorted({k.split('||')[0].split('|', 1)[1] for k in A.files if k.endswith('||ret')})
    rows = []
    for pat in a.rows.split(','):
        rows += [h for h in all_rows if fnmatch.fnmatchcase(h, pat) and h not in rows]
    mets = a.metrics.split(','); logy = set(filter(None, a.logy.split(',')))

    def val(r, m):
        if m.startswith('b') and m[1:].isdigit(): return float(np.asarray(A[f'{R}|{r}||Eb'])[int(m[1:])])
        if m.startswith('pl') and m[2:].isdigit(): return float(np.asarray(A[f'{R}|{r}||bp'])[int(m[2:])])
        return float(A[f'{R}|{r}||{"resid_ratio" if m == "resid" else m}'])

    fig, axes = plt.subplots(1, len(mets), figsize=(a.figw * len(mets), a.figh), constrained_layout=True,
                             squeeze=False)
    rng = np.random.default_rng(0)
    for ax, m in zip(axes[0], mets):
        for i, r in enumerate(rows):
            c = PALETTE[i % len(PALETTE)]
            ax.bar(i, val(r, m), 0.7, color=c, alpha=0.85)
            key = f'{R}|{r}||{PS.get(m, "")}'
            if a.persample != 'off' and PS.get(m) and key in A.files:
                ps = np.asarray(A[key])
                ax.scatter(np.full(len(ps), i) + rng.uniform(-0.22, 0.22, len(ps)), ps, s=6, color=INK, alpha=0.35)
        if m in ('ret', 'resid', 'lowk') or (m.startswith('b') and m[1:].isdigit()):
            ax.axhline(1, color=INK, lw=1.1, ls='--')
        if m in logy: ax.set_yscale('log')
        t = LAB.get(m, f'band retention {BL[int(m[1:])]}' if m.startswith('b') and m[1:].isdigit()
                    else f'placement {BL[int(m[2:])]}' if (m.startswith('pl') and m[2:].isdigit()) else m)
        ax.set_title(t)
        ax.set_xticks(range(len(rows))); ax.set_xticklabels([label(r) for r in rows], rotation=a.rotate,
                                                            ha='right', fontsize=a.fontsize - 2.5)
    if a.title: fig.suptitle(a.title, fontsize=a.fontsize + 2.5)
    fig.savefig(a.out); plt.close(fig)
    print('written', a.out, 'rows:', rows)


if __name__ == '__main__':
    main()
