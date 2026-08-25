"""Per-regime energy spectra from the CROSS-REGIME grid for ONE model: 3x3 panels (one per
regime), one curve per strategy, placement values inset. Ratio to ground truth by default;
set absolute=True for raw E(k) with the ground-truth curve.
Edit CONFIG or pass --model/--strategies/--absolute/--out.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib.pyplot as plt
import style
from grid_panel import SCOL

CONFIG = dict(
    npz='base_results/steering_full_grid.npz', model='st8k-599',
    strategies=['none', 'residual', 'reward', 'placement', 'all3', 'rewardv2'],
    regimes=[1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
    absolute=False, kmax=127, ratio_ylim=(0.04, 5),
    panel_w=4.4, panel_h=3.5, fontsize=9.5, dpi=150,
    title='', out='monitoring/figs/grid_spectra.pdf',
)


def make_figure(c):
    style.apply(c['fontsize'], c['dpi'])
    G = np.load(c['npz'], allow_pickle=True); k = np.arange(1, c['kmax'] + 1)
    n = len(c['regimes']); ncol = 3 if n > 4 else min(n, 2); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(c['panel_w'] * ncol, c['panel_h'] * nrow),
                             constrained_layout=True, squeeze=False, sharey=not c['absolute'])
    for i, R in enumerate(c['regimes']):
        ax = axes[i // ncol][i % ncol]; Eg = np.asarray(G[f'{R}|GT||E'])
        if c['absolute']: ax.loglog(k, Eg[1:c['kmax'] + 1], '-', color=style.INK, lw=2.2, label='ground truth')
        else: ax.axhline(1, color=style.INK, lw=1.3, ls='--', zorder=6)
        style.shade_bands(ax)
        inset = []
        for sg in c['strategies']:
            key = f'{R}|{c["model"]}|SG{sg}||E'
            if key not in G.files: continue
            E = np.asarray(G[key]); pl = float(G[f'{R}|{c["model"]}|SG{sg}||place'])
            y = E[1:c['kmax'] + 1] if c['absolute'] else E[1:c['kmax'] + 1] / Eg[1:c['kmax'] + 1]
            (ax.loglog if c['absolute'] else ax.semilogx)(k, y, '-', color=SCOL.get(sg, style.INK), lw=1.8)
            inset.append(f"{style.STRATEGY.get(sg, sg)}: place {pl:.2f}")
        if not c['absolute']: ax.set(yscale='log', ylim=c['ratio_ylim'])
        ax.text(0.02, 0.02, "\n".join(inset), transform=ax.transAxes, fontsize=c['fontsize'] - 2.3, va='bottom',
                bbox=dict(fc='white', ec='#d8d6cd', alpha=0.85, boxstyle='round,pad=0.3'))
        ax.set_title(f'Re = {R}')
        if i // ncol == nrow - 1: ax.set_xlabel('wavenumber k')
        if i % ncol == 0: ax.set_ylabel('E(k)' if c['absolute'] else 'E(k) / E$_{GT}$(k)')
    for j in range(n, nrow * ncol): axes[j // ncol][j % ncol].axis('off')
    handles = [plt.Line2D([], [], color=SCOL.get(s, style.INK), lw=2.2, label=style.STRATEGY.get(s, s)) for s in c['strategies']]
    axes[0][0].legend(handles=handles, fontsize=c['fontsize'] - 1.5, loc='upper right')
    fig.suptitle(c['title'] or f"{style.MODEL.get(c['model'], c['model'])} — energy per strategy"
                 f" ({'absolute' if c['absolute'] else 'ratio to GT'})", fontsize=c['fontsize'] + 2.5)
    fig.savefig(c['out']); plt.close(fig); print('written', c['out'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model'); ap.add_argument('--strategies'); ap.add_argument('--out'); ap.add_argument('--title')
    ap.add_argument('--absolute', action='store_true')
    a = ap.parse_args(); c = dict(CONFIG)
    if a.model: c['model'] = a.model
    if a.strategies: c['strategies'] = a.strategies.split(',')
    if a.absolute: c['absolute'] = True
    for f in ('out', 'title'):
        if getattr(a, f) is not None: c[f] = getattr(a, f)
    make_figure(c)
