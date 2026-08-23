"""Reusable thesis figure: one model's strategy panel across regimes (or several stacked).
Everything adjustable from the CLI; reads base_results/steering_full_grid.npz. If seed-repeat
keys (|s701, |s702) exist for a cell, draws mean with min-max error bars.

  python -m src.ddpo_ft.plot_strategy_panel --models re2k-149 \
      --metrics b1632,b3264,b6496,place,resid,mse --out monitoring/figs/panel_re2k.png
  # base model appears as 'base0' once its grid block has run.

Metrics: b{lo}{hi} = band retention E[lo,hi)/GT; place; resid (ratio to GT floor); mse; ret
(the standard [32,96) band); lowk. --ylim metric=lo,hi overrides; --linear metric,... forces
linear y where log is default.
"""
import argparse, os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VCOL = {'none': '#5d645d', 'residual': '#2a78d6', 'reward': '#7a4bd0',
        'placement': '#0f9e78', 'all3': '#d4770a'}
VLAB = {'none': 'unguided', 'residual': 'residual', 'reward': 'reward (dose)',
        'placement': 'placement', 'all3': 'all three'}
MLAB = {'r1k-449': 'fine-tuned Re=1000', 're2k-149': 'fine-tuned Re=2000',
        'pr2k-549': 'placement-anchored Re=2000', 'rs8kkl-799': 'fine-tuned Re=8000 (KL)',
        'base0': 'base model (no fine-tune)', 'st8k-599': 'steered-trained Re=8000'}
DEF_YL = {'place': (0.55, 0.92), 'mse': None, 'resid': (0.03, 3.0)}
LOGY_DEFAULT = {'resid'} | {f'b{a}{b}' for a, b in ((16, 32), (32, 64), (64, 96))} | {'ret'}
INK = '#232823'

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='re2k-149')
    ap.add_argument('--strategies', default='none,residual,reward,placement,all3')
    ap.add_argument('--metrics', default='b1632,b3264,b6496,place,resid,mse')
    ap.add_argument('--regimes', default='1000,1500,2000,3000,4000,5000,6000,7000,8000')
    ap.add_argument('--ylim', action='append', default=[], help='metric=lo,hi (repeatable)')
    ap.add_argument('--linear', default='', help='comma list of metrics to force linear y')
    ap.add_argument('--panel-w', type=float, default=3.1)
    ap.add_argument('--panel-h', type=float, default=2.9)
    ap.add_argument('--dpi', type=int, default=150)
    ap.add_argument('--ms', type=float, default=3.8)
    ap.add_argument('--lw', type=float, default=1.9)
    ap.add_argument('--fontsize', type=float, default=9.5)
    ap.add_argument('--legend-panel', type=int, default=0)
    ap.add_argument('--title', default='')
    ap.add_argument('--npz', default='base_results/steering_full_grid.npz')
    ap.add_argument('--out', default='monitoring/figs/strategy_panel.png')
    return ap.parse_args()

def main():
    a = parse()
    plt.rcParams.update({'font.size': a.fontsize, 'figure.facecolor': 'white',
                         'axes.facecolor': 'white', 'savefig.dpi': a.dpi,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.color': '#d8d6cd', 'grid.linewidth': 0.5,
                         'legend.frameon': False})
    G = np.load(a.npz, allow_pickle=True)
    models = a.models.split(',')
    strats = a.strategies.split(',')
    mets = a.metrics.split(',')
    regs = [int(r) for r in a.regimes.split(',')]
    linear = set(filter(None, a.linear.split(',')))
    yl = dict(DEF_YL)
    for spec in a.ylim:
        k, v = spec.split('=')
        yl[k] = tuple(float(x) for x in v.split(','))

    def cell(R, m, sg, f, suf=''):
        k = f'{R}|{m}|SG{sg}{suf}||{f}'
        return float(G[k]) if k in G.files else np.nan

    def metric(R, m, sg, met, suf=''):
        if met.startswith('b'):
            lo, hi = int(met[1:3]), int(met[3:])
            k = f'{R}|{m}|SG{sg}{suf}||E'
            if k not in G.files: return np.nan
            E, Eg = np.asarray(G[k]), np.asarray(G[f'{R}|GT||E'])
            return E[lo:hi].sum() / Eg[lo:hi].sum()
        return cell(R, m, sg, {'resid': 'resid_ratio'}.get(met, met), suf)

    LABEL = {'place': 'placement', 'resid': 'PDE residual / GT floor', 'mse': 'MSE',
             'ret': 'retention [32,96)', 'lowk': 'large-scale energy / GT'}
    fig, axes = plt.subplots(len(models), len(mets),
                             figsize=(a.panel_w * len(mets), a.panel_h * len(models)),
                             constrained_layout=True, squeeze=False)
    for mi, m in enumerate(models):
        for ci, met in enumerate(mets):
            ax = axes[mi][ci]
            lab = LABEL.get(met, f'band retention [{met[1:3]},{met[3:]})')
            if met.startswith('b') or met in ('ret', 'resid', 'lowk'):
                ax.axhline(1, color=INK, lw=1.1, ls='--', zorder=6)
            for sg in strats:
                vals = [[metric(R, m, sg, met, sf) for sf in ('', '|s701', '|s702')]
                        for R in regs]
                mean = [np.nanmean(v) if not np.isnan(v[0]) else np.nan for v in vals]
                has_seeds = any(not np.isnan(v[1]) for v in vals)
                ax.plot(regs, mean, '-', color=VCOL.get(sg, INK), lw=a.lw, marker='o',
                        ms=a.ms, mec='white', mew=0.7,
                        label=VLAB.get(sg, sg) if (mi * len(mets) + ci) == a.legend_panel else None)
                if has_seeds:
                    lo = [np.nanmin(v) for v in vals]; hi = [np.nanmax(v) for v in vals]
                    ax.fill_between(regs, lo, hi, color=VCOL.get(sg, INK), alpha=0.18, lw=0)
            ax.set(xscale='log')
            if met in LOGY_DEFAULT and met not in linear:
                ax.set_yscale('log')
            if yl.get(met): ax.set_ylim(*yl[met])
            ax.set_xticks([1000, 2000, 4000, 8000])
            ax.set_xticklabels(['1k', '2k', '4k', '8k'], fontsize=a.fontsize - 1.2)
            ax.minorticks_off()
            if mi == 0: ax.set_title(lab, fontsize=a.fontsize + 0.5)
            if mi == len(models) - 1: ax.set_xlabel('Re')
            if ci == 0: ax.set_ylabel(MLAB.get(m, m), fontsize=a.fontsize)
    axes[a.legend_panel // len(mets)][a.legend_panel % len(mets)].legend(fontsize=a.fontsize - 1.7)
    if a.title: fig.suptitle(a.title, fontsize=a.fontsize + 3)
    fig.savefig(a.out); plt.close(fig)
    print('written', a.out)

if __name__ == '__main__':
    main()
