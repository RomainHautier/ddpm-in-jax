"""Reusable thesis figure: per-regime energy-spectrum ratio E(k)/E_GT(k) for the sampling
strategies of one model, placement values annotated per curve. Reads steering_full_grid.npz.

  python -m src.ddpo_ft.plot_strategy_spectra --model rs8kkl-799 --out monitoring/figs/spec_rs8k.png
Options: --regimes, --strategies, --absolute (plot E(k) and GT instead of the ratio),
--kmax, --panel-w/h, --dpi, --fontsize.
"""
import argparse, os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VCOL = {'none': '#5d645d', 'residual': '#2a78d6', 'reward': '#7a4bd0',
        'placement': '#0f9e78', 'all3': '#d4770a'}
VLAB = {'none': 'unguided', 'residual': 'residual', 'reward': 'reward', 'placement': 'placem.',
        'all3': 'all three'}
MLAB = {'r1k-449': 'fine-tuned Re=1000', 're2k-149': 'fine-tuned Re=2000',
        'pr2k-549': 'placement-anchored Re=2000', 'rs8kkl-799': 'fine-tuned Re=8000 (KL)',
        'base0': 'base model (no fine-tune)', 'st8k-599': 'steered-trained Re=8000'}
INK = '#232823'

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='rs8kkl-799')
ap.add_argument('--strategies', default='none,residual,reward,placement,all3')
ap.add_argument('--regimes', default='1000,1500,2000,3000,4000,5000,6000,7000,8000')
ap.add_argument('--absolute', action='store_true')
ap.add_argument('--kmax', type=int, default=127)
ap.add_argument('--panel-w', type=float, default=4.4)
ap.add_argument('--panel-h', type=float, default=3.5)
ap.add_argument('--dpi', type=int, default=150)
ap.add_argument('--fontsize', type=float, default=9.5)
ap.add_argument('--npz', default='base_results/steering_full_grid.npz')
ap.add_argument('--out', default='monitoring/figs/strategy_spectra.png')
a = ap.parse_args()
plt.rcParams.update({'font.size': a.fontsize, 'figure.facecolor': 'white',
                     'axes.facecolor': 'white', 'savefig.dpi': a.dpi,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#d8d6cd', 'grid.linewidth': 0.5,
                     'legend.frameon': False})
G = np.load(a.npz, allow_pickle=True)
regs = [int(r) for r in a.regimes.split(',')]
strats = a.strategies.split(',')
k = np.arange(1, a.kmax + 1)
n = len(regs)
ncol = 3 if n > 4 else min(n, 2)
nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(a.panel_w * ncol, a.panel_h * nrow),
                         constrained_layout=True, squeeze=False, sharey=not a.absolute)
for i, R in enumerate(regs):
    ax = axes[i // ncol][i % ncol]
    Eg = np.asarray(G[f'{R}|GT||E'])
    if a.absolute:
        ax.loglog(k, Eg[1:a.kmax + 1], '-', color=INK, lw=2.2, label='ground truth')
    else:
        ax.axhline(1, color=INK, lw=1.3, ls='--', zorder=6)
        ax.axvspan(32, 96, color='#e3edf3', zorder=0)
    entries = []
    for sg in strats:
        key = f'{R}|{a.model}|SG{sg}||E'
        if key not in G.files: continue
        E = np.asarray(G[key])
        pl = float(G[f'{R}|{a.model}|SG{sg}||place'])
        y = E[1:a.kmax + 1] if a.absolute else E[1:a.kmax + 1] / Eg[1:a.kmax + 1]
        (ax.loglog if a.absolute else ax.semilogx)(k, y, '-', color=VCOL.get(sg, INK), lw=1.8)
        entries.append((sg, pl))
    if not a.absolute: ax.set_yscale('log'); ax.set_ylim(0.04, 5)
    leg = "\n".join(f"{VLAB.get(sg, sg)}: place {pl:.2f}" for sg, pl in entries)
    ax.text(0.02, 0.02, leg, transform=ax.transAxes, fontsize=a.fontsize - 2.1,
            va='bottom', ha='left',
            bbox=dict(fc='white', ec='#d8d6cd', alpha=0.85, boxstyle='round,pad=0.3'))
    for sg, _ in entries:
        pass
    ax.set_title(f'Re = {R}', fontsize=a.fontsize + 0.5)
    if i // ncol == nrow - 1: ax.set_xlabel('wavenumber k')
    if i % ncol == 0:
        ax.set_ylabel('E(k)' if a.absolute else 'E(k) / E$_{GT}$(k)')
for j in range(n, nrow * ncol):
    axes[j // ncol][j % ncol].axis('off')
handles = [plt.Line2D([], [], color=VCOL[s], lw=2.2, label=VLAB.get(s, s)) for s in strats]
axes[0][0].legend(handles=handles, fontsize=a.fontsize - 1.5, loc='upper right')
fig.suptitle(f'{MLAB.get(a.model, a.model)} — energy distribution per strategy '
             f'({"absolute spectra" if a.absolute else "ratio to ground truth; shaded = [32,96) band"}), '
             'placement values inset', fontsize=a.fontsize + 2.5)
fig.savefig(a.out); plt.close(fig)
print('written', a.out)
