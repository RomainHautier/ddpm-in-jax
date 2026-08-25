"""Metrics vs Reynolds number from the CROSS-REGIME strategy grid (base_results/steering_full_grid.npz):
one row per model, one column per metric, one colored curve per sampling strategy. Error bands
appear automatically where seed-repeat cells exist.

Cell keys in the store: '{Re}|{model}|SG{strategy}||{field}', E = spectrum.
Metrics: ret, place, resid, mse, lowk, kstar, b{lo}{hi} (band retention, e.g. b1632, b3264, b6496).
Edit CONFIG or pass --models/--metrics/--strategies/--out.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib.pyplot as plt
import style

CONFIG = dict(
    npz='base_results/steering_full_grid.npz',
    models=['re2k-149'],
    strategies=['none', 'residual', 'reward', 'placement', 'all3', 'rewardv2'],
    metrics=['b1632', 'b3264', 'b6496', 'place', 'resid', 'mse'],
    regimes=[1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
    ylim={'place': (0.55, 0.92), 'resid': (0.03, 3.0)},
    logy={'resid', 'b1632', 'b3264', 'b6496', 'ret'},
    panel_w=3.1, panel_h=2.9, fontsize=9.5, dpi=150, legend_panel=0,
    title='', out='monitoring/figs/grid_panel.pdf',
)
SCOL = {'none': '#5d645d', 'residual': '#2a78d6', 'reward': '#7a4bd0', 'placement': '#0f9e78',
        'all3': '#d4770a', 'rewardv2': '#b8399e', 'all3v2': '#c22f4f'}
LAB = {'place': 'placement', 'resid': 'PDE residual / GT floor', 'mse': 'MSE',
       'ret': 'retention [32,96)', 'lowk': 'large-scale energy / GT', 'kstar': 'k*'}


def metric(G, R, m, sg, met, suf=''):
    if met.startswith('b') and met[1:].isdigit():
        lo, hi = int(met[1:3]), int(met[3:])
        k = f'{R}|{m}|SG{sg}{suf}||E'
        if k not in G.files: return np.nan
        return np.asarray(G[k])[lo:hi].sum() / np.asarray(G[f'{R}|GT||E'])[lo:hi].sum()
    k = f'{R}|{m}|SG{sg}{suf}||{"resid_ratio" if met == "resid" else met}'
    return float(G[k]) if k in G.files else np.nan


def make_figure(c):
    style.apply(c['fontsize'], c['dpi'])
    G = np.load(c['npz'], allow_pickle=True)
    M, S, T, R = c['models'], c['strategies'], c['metrics'], c['regimes']
    fig, axes = plt.subplots(len(M), len(T), figsize=(c['panel_w'] * len(T), c['panel_h'] * len(M)),
                             constrained_layout=True, squeeze=False)
    for mi, m in enumerate(M):
        for ci, met in enumerate(T):
            ax = axes[mi][ci]
            if met.startswith('b') or met in ('ret', 'resid', 'lowk'):
                ax.axhline(1, color=style.INK, lw=1.1, ls='--', zorder=6)
            for sg in S:
                vals = [[metric(G, r, m, sg, met, sf) for sf in ('', '|s701', '|s702')] for r in R]
                mean = [np.nanmean(v) if not np.isnan(v[0]) else np.nan for v in vals]
                ax.plot(R, mean, '-', color=SCOL.get(sg, style.INK), lw=1.9, marker='o', ms=3.8,
                        mec='white', mew=0.7,
                        label=style.STRATEGY.get(sg, sg) if (mi * len(T) + ci) == c['legend_panel'] else None)
                if any(not np.isnan(v[1]) for v in vals):
                    ax.fill_between(R, [np.nanmin(v) for v in vals], [np.nanmax(v) for v in vals],
                                    color=SCOL.get(sg, style.INK), alpha=0.18, lw=0)
            style.re_axis(ax)
            if met in c['logy']: ax.set_yscale('log')
            if met in c['ylim']: ax.set_ylim(*c['ylim'][met])
            if mi == 0: ax.set_title(LAB.get(met, f'band retention [{met[1:3]},{met[3:]})' if met.startswith('b') else met))
            if mi == len(M) - 1: ax.set_xlabel('Re')
            if ci == 0: ax.set_ylabel(style.MODEL.get(m, m))
    lp = c['legend_panel']; axes[lp // len(T)][lp % len(T)].legend(fontsize=c['fontsize'] - 1.7)
    if c['title']: fig.suptitle(c['title'], fontsize=c['fontsize'] + 3)
    fig.savefig(c['out']); plt.close(fig); print('written', c['out'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for f in ('models', 'strategies', 'metrics'): ap.add_argument('--' + f)
    ap.add_argument('--out'); ap.add_argument('--title')
    a = ap.parse_args(); c = dict(CONFIG)
    for f in ('models', 'strategies', 'metrics'):
        if getattr(a, f): c[f] = getattr(a, f).split(',')
    for f in ('out', 'title'):
        if getattr(a, f) is not None: c[f] = getattr(a, f)
    make_figure(c)
