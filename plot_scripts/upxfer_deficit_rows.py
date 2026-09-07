"""The full-grid transfer figure in DEFICIT form: same layout as upxfer_recon_rows
(one shared sample per regime row, fixed column per model, each cell under that
model's best configuration for the rung) but each cell shows the local fine-scale energy
deficit against the ground truth (red = model short of truth, blue = excess). The
first column shows the ground truth's own local fine-scale energy map.

  JAX_PLATFORMS=cpu python plot_scripts/upxfer_deficit_rows.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np
import matplotlib.pyplot as plt
import style
from viz_energy import local_hik_energy

SIG = 4.7988
FS = 20
KCUT, SIGMA = 32, 6.0
COLS = [('base0',     'Base Model - No Guidance'),
        ('pg1k-0599', 'Re = 1000 GR-FT'),
        ('pg4k-0299', 'Re = 4000 GR-FT'),
        ('pg8k-0499', 'Re = 8000 GR-FT')]
# the base column shows the UNGUIDED chain (dial off); fine-tunes are gated
TAG = {'base0': 'mop0.2_0'}
ROWS = [
    (1000, ['base0', 'pg1k-0599', 'pg4k-0299', 'pg8k-0499']),
    (2000, ['base0', 'pg1k-0599', 'pg4k-0299', 'pg8k-0499']),
    (4000, ['base0', 'pg4k-0299', 'pg1k-0599', 'pg8k-0499']),
    (6000, ['base0', 'pg4k-0299', 'pg1k-0599', 'pg8k-0499']),
    (8000, ['base0', 'pg8k-0499', 'pg4k-0299', 'pg1k-0599']),
]
CHCFG = {('pg8k-0499', 1000): 'K100', ('pg8k-0499', 2000): 'K125',
         ('pg8k-0499', 4000): 'K160', ('pg8k-0499', 6000): 'K160',
         ('pg4k-0299', 1000): 'K100', ('pg4k-0299', 2000): 'K125'}
NCOL = 1 + len(COLS)


def box(ax):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color('black'); sp.set_linewidth(1.2)


def main():
    style.apply(style.BASE_FS, 150)
    fig, axes = plt.subplots(len(ROWS), NCOL, figsize=(3.5 * NCOL, 3.7 * len(ROWS)),
                             constrained_layout=True, squeeze=False)
    titled = set()
    for ri, (R, models) in enumerate(ROWS):
        F = f'base_results/fields/re{R}'
        gt = np.load(f'{F}/GT.npz')['x'].astype(np.float32)
        span = np.arange(len(gt) - 80, len(gt)) if len(gt) > 80 else np.arange(len(gt))
        e = (np.abs(np.fft.rfft2(gt[span, ..., 1]))[:, 32:96, :] ** 2).sum((1, 2))
        g = span[int(np.argsort(e)[len(e) // 2])]
        wg = gt[g, ..., 1] * SIG
        Eg = local_hik_energy(wg[None], KCUT, SIGMA)[0]
        ax = axes[ri][0]
        ax.imshow(Eg, cmap='magma')
        if ri == 0:
            ax.set_title(f'GT local energy $k\\geq{KCUT}$', fontsize=FS - 2)
        ax.set_ylabel(f'Re = {R}', fontsize=FS)
        ax.set_xticks([]); ax.set_yticks([]); box(ax)
        flat = wg.ravel() - wg.mean()
        # symmetric deficit scale per row, from all model deficits
        defs = []
        for m in models:
            ch = CHCFG.get((m, R), 'K3')
            x = np.load(f'{F}/{m}__{ch}__{TAG.get(m, "v8lowband")}.npz')['x'].astype(np.float32)
            wv = x[..., 1].reshape(len(x), -1) * SIG
            wv = wv - wv.mean(1, keepdims=True)
            j = int(np.argmax(wv @ flat))
            wm = x[j, ..., 1] * SIG
            defs.append((m, Eg - local_hik_energy(wm[None], KCUT, SIGMA)[0]))
        vd = np.percentile(np.abs(np.concatenate([d.ravel() for _, d in defs])), 99.5)
        dmap = dict(defs)
        for ci, (m, lab) in enumerate(COLS):
            ax = axes[ri][1 + ci]
            if m not in dmap:
                ax.axis('off'); continue
            d = dmap[m]
            ax.imshow(d, cmap='RdBu_r', vmin=-vd, vmax=vd)
            if m not in titled:
                ax.set_title(lab, fontsize=FS)
                titled.add(m)
            ax.set_xticks([]); ax.set_yticks([]); box(ax)
    for d_, ext in (('docs/figs_overleaf/gated', 'pdf'), ('plotting/figs', 'png')):
        os.makedirs(d_, exist_ok=True)
        p_ = f'{d_}/upxfer_deficit_rows.{ext}'; fig.savefig(p_)
        print('written', p_)


if __name__ == '__main__':
    main()
