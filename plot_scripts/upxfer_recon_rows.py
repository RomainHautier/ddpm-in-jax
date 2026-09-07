"""Reconstruction panels for the upward transfer story, one row per regime, one
shared sample per row: ground truth at the left, then the in-regime gated model and
the transported gated models.

  Re=1000: base + gate | pg1k + gate
  Re=4000: pg4k + gate | pg1k + gate
  Re=8000: pg8k + gate | pg4k + gate | pg1k + gate

Model pools at the generated regimes are the 80-sample test subset while the GT pool
carries 120 rows, so each model panel locates its own copy of the row's sample by
vorticity correlation instead of assuming index alignment.

  JAX_PLATFORMS=cpu python plot_scripts/upxfer_recon_rows.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib.pyplot as plt
import style

SIG = 4.7988
FS = 20          # ONE font size for every label in this figure
# fixed column per model; a panel appears only where that model was evaluated, and
# the column title prints once, on that model's topmost reconstruction
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
# below-home cells use the model's SELECTED downward configuration (chain + gated
# dial); everything else is the K3 + gate configuration
CHCFG = {('pg8k-0499', 1000): 'K100', ('pg8k-0499', 2000): 'K125',
         ('pg8k-0499', 4000): 'K160', ('pg8k-0499', 6000): 'K160',
         ('pg4k-0299', 1000): 'K100', ('pg4k-0299', 2000): 'K125'}
NCOL = 1 + len(COLS)


def w(x, i):
    return x[i, ..., 1] * SIG


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
        wg = w(gt, g)
        vw = np.percentile(np.abs(wg), 99.5)
        ax = axes[ri][0]
        ax.imshow(wg, cmap='RdBu_r', vmin=-vw, vmax=vw)
        if ri == 0:
            ax.set_title('Ground Truth', fontsize=FS)
        ax.set_ylabel(f'Re = {R}', fontsize=FS)
        ax.set_xticks([]); ax.set_yticks([]); box(ax)
        flat = wg.ravel() - wg.mean()
        for ci, (m, lab) in enumerate(COLS):
            ax = axes[ri][1 + ci]
            if m not in models:
                ax.axis('off'); continue
            ch = CHCFG.get((m, R), 'K3')
            x = np.load(f'{F}/{m}__{ch}__{TAG.get(m, "v8lowband")}.npz')['x'].astype(np.float32)
            wv = x[..., 1].reshape(len(x), -1) * SIG
            wv = wv - wv.mean(1, keepdims=True)
            j = int(np.argmax(wv @ flat))
            ax.imshow(w(x, j), cmap='RdBu_r', vmin=-vw, vmax=vw)
            if m not in titled:
                ax.set_title(lab, fontsize=FS)
                titled.add(m)
            ax.set_xticks([]); ax.set_yticks([]); box(ax)
    for d_, ext in (('docs/figs_overleaf/gated', 'pdf'), ('plotting/figs', 'png')):
        os.makedirs(d_, exist_ok=True)
        p_ = f'{d_}/upxfer_recon_rows.{ext}'; fig.savefig(p_)
        print('written', p_)


if __name__ == '__main__':
    main()
