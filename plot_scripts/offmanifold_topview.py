"""Top-down (PC1 x PC2) view of the off-manifold travel, one wide panel per target regime.

Reads the trajectory dump written by offmanifold_travel.py so no TPU work is repeated.
Each panel shows the four truth clouds, the shared low-res starting point, the standard
reconstruction, the target's ground-truth sample, and the eta=1 chain of each model with
its renoising events numbered. The stack shares the PC1 axis so travel distances are
directly comparable across regimes.
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import style

DUMP = os.environ.get(
    'OFFMANIFOLD_DUMP',
    '/tmp/claude-2001/-home-rhautier-ddpm-jax/24ff525a-722f-4c10-b5d9-663bc835ccb1/'
    'scratchpad/offmanifold_data.json')

MODEL_LABEL = {'base': 'base', 'mt1k-0499': 'Re=1000 ft', 'mt2k-0599': 'Re=2000 ft',
               'r4kp02-0599': 'Re=4000 ft', 'r8kp02-0599': 'Re=8000 ft'}



def main(only=None):
    style.apply()
    d = json.load(open(DUMP))
    clouds = {int(k): np.array(v) for k, v in d['clouds'].items()}
    nstep = d['nstep']
    regimes = [only] if only else [1000, 2000, 4000, 8000]

    fig, axes = plt.subplots(len(regimes), 1,
                             figsize=(10.6, 4.9) if only else (12.5, 10.4),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[:, 0]
    for ax, R in zip(axes, regimes):
        t = d['targets'][str(R)]
        for Rb, v in clouds.items():
            tgt = Rb == R
            col = style.vivid(style.REGIME_COLOR[Rb]) if tgt else style.REGIME_COLOR[Rb]
            ax.scatter(v[:, 0], v[:, 1], s=24 if tgt else 14,
                       color=col, alpha=.75 if tgt else .22,
                       linewidths=0, zorder=3 if tgt else 2)
        rc0 = np.array(t['rc'])
        for n, tr in t['trajs'].items():
            tr = np.array(tr)
            col = style.MODEL_COLOR.get({'base': 'base0'}.get(n, n), '#232823')
            # the move the first renoise-and-denoise makes, from the shared recon start
            ax.plot([rc0[0], tr[0, 0]], [rc0[1], tr[0, 1]], ls=':', lw=1.2,
                    color=col, alpha=.7, zorder=4)
            ax.plot(tr[:, 0], tr[:, 1], color=col, lw=1.5, alpha=.85, zorder=5)
            ev = tr[::nstep]  # first state of each pass = renoising event
            ax.scatter(ev[:, 0], ev[:, 1], s=120, color=col, zorder=6,
                       edgecolors='white', linewidths=.8)
            for j, p in enumerate(ev):
                ax.text(p[0], p[1], str(j + 1), color='white', fontsize=7,
                        ha='center', va='center', zorder=7, fontweight='bold')
            ax.scatter(*tr[-1, :2], marker='D', s=55, color=col, zorder=7,
                       edgecolors='black', linewidths=.8)
        lr = np.array(t['lr']); rc = np.array(t['rc']); gt = np.array(t['gt'])
        ax.scatter(*lr[:2], marker='v', s=90, color='#232823', zorder=8,
                   edgecolors='white', linewidths=.7)
        ax.scatter(*rc[:2], marker='s', s=70, color='#d4770a', zorder=8,
                   edgecolors='white', linewidths=.7)
        ax.scatter(*gt[:2], marker='*', s=320, color=style.vivid(style.REGIME_COLOR[R]), zorder=9,
                   edgecolors='black', linewidths=.9)
        ax.set_ylabel('PC2')
        ax.text(.008, .93, f'target Re={R}', transform=ax.transAxes,
                fontsize=11, va='top', fontweight='bold')
        ax.set_ylim(-5.8, 4.4)
        ax.grid(alpha=.25)
    axes[-1].set_xlabel('PC1  (log-spectral shape, Reynolds axis)')

    handles = [plt.Line2D([], [], color=style.MODEL_COLOR[{'base': 'base0'}.get(n, n)],
                          lw=2, label=MODEL_LABEL[n]) for n in
               ['base', 'mt1k-0499', 'mt2k-0599', 'r4kp02-0599', 'r8kp02-0599']]
    handles += [
        plt.Line2D([], [], ls='none', marker='v', color='#232823', ms=8,
                   label='low-res start'),
        plt.Line2D([], [], ls='none', marker='s', color='#d4770a', ms=8,
                   label='standard recon'),
        plt.Line2D([], [], ls='none', marker='*', color='#777777', ms=13,
                   markeredgecolor='black', label='ground-truth sample'),
        plt.Line2D([], [], ls='none', marker='o', color='#777777', ms=9,
                   markeredgecolor='white', label='renoising event (numbered)'),
        plt.Line2D([], [], ls='none', marker='D', color='#777777', ms=7,
                   markeredgecolor='black', label='chain endpoint'),
        plt.Line2D([], [], ls=':', color='#777777', lw=1.4,
                   label='recon to first estimate'),
    ]
    if only:
        fig.legend(handles=handles, loc='upper center', ncol=5, fontsize=8.2,
                   bbox_to_anchor=(.5, .985), framealpha=.92)
        fig.suptitle('')
        fig.tight_layout(rect=(0, 0, 1, .90))
    else:
        fig.legend(handles=handles, loc='upper center', ncol=5, fontsize=9,
                   bbox_to_anchor=(.5, .965), framealpha=.92)
        fig.suptitle('Off-manifold travel, top view: chains from the standard '
                     'reconstruction toward each regime', y=.995)
        fig.tight_layout(rect=(0, 0, 1, .935))
    sfx = f'_re{only}' if only else ''
    for d_, ext in [('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')]:
        os.makedirs(d_, exist_ok=True)
        p_ = f'{d_}/offmanifold_topview{sfx}.{ext}'
        fig.savefig(p_)
        print('written', p_)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', type=int, help='render a single target-regime panel')
    main(ap.parse_args().only)
