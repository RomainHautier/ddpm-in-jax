"""Metric bars for a SINGLE-REGIME audit store: retention, placement, MSE, PDE residual, low-k,
any k-band ('b2' = [16,32) retention) or per-band placement ('pl4' = [64,96)). Per-sample
distributions are overlaid as dots automatically when the store carries them.

Edit CONFIG or pass --rows/--metrics/--out. Same row keys and wildcards as audit_spectra.py.
"""
import argparse, fnmatch, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib.pyplot as plt
import style
from audit_spectra import label

CONFIG = dict(
    npz='base_results/re1000_audit.npz', regime='1000',
    rows=['recon', 'base0|K3|none', 'base0|K3|residual', 'base0|K3|reward', 'base0|K3|placement',
          'base0|K3|all3', 'base0|K3|rewardv2', 'base0|K3|all3v2'],
    metrics=['ret', 'place', 'mse', 'resid'],
    logy={'resid'}, persample=True,
    panel_w=4.2, panel_h=4.4, fontsize=10, dpi=140, rotate=55,
    title='', out='monitoring/figs/audit_metrics.png',
)
LAB = {'ret': 'retention [32,96)', 'place': 'placement', 'mse': 'MSE', 'resid': 'PDE residual / GT floor',
       'lowk': 'large-scale energy / GT', 'kstar': 'effective resolution k*'}
PS = {'ret': 'ps_ret_paired', 'place': 'ps_place', 'mse': 'ps_mse', 'resid': 'ps_resid'}


def value(A, R, r, m):
    if m.startswith('b') and m[1:].isdigit(): return float(np.asarray(A[f'{R}|{r}||Eb'])[int(m[1:])])
    if m.startswith('pl') and m[2:].isdigit(): return float(np.asarray(A[f'{R}|{r}||bp'])[int(m[2:])])
    return float(A[f'{R}|{r}||{"resid_ratio" if m == "resid" else m}'])


def title_of(m):
    if m.startswith('b') and m[1:].isdigit(): return f'band retention {style.BAND_LABELS[int(m[1:])]}'
    if m.startswith('pl') and m[2:].isdigit(): return f'placement {style.BAND_LABELS[int(m[2:])]}'
    return LAB.get(m, m)


def make_figure(c):
    style.apply(c['fontsize'], c['dpi'])
    A = np.load(c['npz'], allow_pickle=True); R = c['regime']
    have = sorted({k.split('||')[0].split('|', 1)[1] for k in A.files if k.startswith(f'{R}|') and k.endswith('||ret')})
    rows = []
    for pat in c['rows']:
        rows += [h for h in have if fnmatch.fnmatchcase(h, pat) and h not in rows]
    fig, axes = plt.subplots(1, len(c['metrics']), figsize=(c['panel_w'] * len(c['metrics']), c['panel_h']),
                             constrained_layout=True, squeeze=False)
    rng = np.random.default_rng(0)
    for ax, m in zip(axes[0], c['metrics']):
        for i, r in enumerate(rows):
            col = style.PALETTE[i % len(style.PALETTE)]
            ax.bar(i, value(A, R, r, m), 0.7, color=col, alpha=0.85)
            key = f'{R}|{r}||{PS.get(m, "")}'
            if c['persample'] and PS.get(m) and key in A.files:
                ps = np.asarray(A[key])
                ax.scatter(np.full(len(ps), i) + rng.uniform(-0.22, 0.22, len(ps)), ps, s=6,
                           color=style.INK, alpha=0.35)
        if m in ('ret', 'resid', 'lowk') or (m.startswith('b') and m[1:].isdigit()):
            ax.axhline(1, color=style.INK, lw=1.1, ls='--')
        if m in c['logy']: ax.set_yscale('log')
        ax.set_title(title_of(m))
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([label(r) for r in rows], rotation=c['rotate'], ha='right', fontsize=c['fontsize'] - 2.5)
    if c['title']: fig.suptitle(c['title'], fontsize=c['fontsize'] + 2.5)
    fig.savefig(c['out']); plt.close(fig)
    print('written', c['out'], '| rows:', rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows'); ap.add_argument('--metrics'); ap.add_argument('--out'); ap.add_argument('--title')
    ap.add_argument('--npz'); ap.add_argument('--regime')
    a = ap.parse_args(); c = dict(CONFIG)
    if a.rows: c['rows'] = a.rows.split(',')
    if a.metrics: c['metrics'] = a.metrics.split(',')
    for f in ('out', 'title', 'npz', 'regime'):
        if getattr(a, f) is not None: c[f] = getattr(a, f)
    make_figure(c)
