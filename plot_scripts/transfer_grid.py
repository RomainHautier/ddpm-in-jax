"""Transfer made visual: the same median triplet at four regimes, under the two extreme
specialists carried across the ladder and the in-regime model with the per-band gate.

Rows are regimes (1000/2000/4000/8000); columns are ground truth, the base model, the
Re=1000 fine-tune (transfer), the Re=8000 fine-tune (transfer), and the model trained at
that regime with the v7 per-band gate. Complements sample_row.py, which deliberately
excludes transfer; here the transfer IS the subject: the Re=1000 model is too smooth
above its regime, the Re=8000 model over-etches below its own, the gated home model
holds the middle.

  JAX_PLATFORMS=cpu python plot_scripts/transfer_grid.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd())
import matplotlib.pyplot as plt
import numpy as np
import style

REGS = [1000, 2000, 4000, 8000]
HOME = {1000: ('mt1k-0499', 'Re=1000 ft'), 2000: ('mt2k-0599', 'Re=2000 ft'),
        4000: ('r4kp02-0599', 'Re=4000 ft'), 8000: ('r8kp02-0599', 'Re=8000 ft')}


def load(R, name):
    p = f'base_results/fields/re{R}/{name}.npz'
    return np.load(p)['x'].astype(np.float32) if os.path.exists(p) else None


def align_gt(R, gt, n):
    if len(gt) == n:
        return gt
    S = np.load('base_results/re1000_audit.npz' if R == 1000
                else f'base_results/regime_audit_re{R}.npz', allow_pickle=True)
    sq = np.asarray(S[f'{R}|EVAL||seq']); m = sq >= 12
    assert m.sum() == n, f'mask gives {m.sum()}, fields have {n}'
    return gt[m]


def main():
    import jax.numpy as jnp
    from src.rewards import make_spectrum_fn
    style.apply(style.BASE_FS, 170)
    sf = make_spectrum_fn(256)

    ncol = 5
    fig, axes = plt.subplots(len(REGS), ncol, figsize=(3.0 * ncol, 3.15 * len(REGS)),
                             constrained_layout=True)
    for ri, R in enumerate(REGS):
        gt = load(R, 'GT')
        hm, hlab = HOME[R]
        gate = load(R, f'{hm}__K3__v7bandgate')
        gate_lab = f'{hlab} + gate\n(trained here)'
        if gate is None:
            gate, gate_lab = load(R, f'{hm}__K3__mop0.2_0'), f'{hlab} unguided\n(trained here)'
        cols = [('ground truth', None),
                ('base', load(R, 'base0__K3__mop0.2_0')),
                ('Re=1000 ft', load(R, 'mt1k-0499__K3__mop0.2_0')),
                ('Re=8000 ft', load(R, 'r8kp02-0599__K3__mop0.2_0')),
                (gate_lab, gate)]
        n = cols[1][1].shape[0]
        gt = align_gt(R, gt, n)
        # median fine-band triplet, as in sample_row
        E = np.concatenate([np.asarray(sf(jnp.asarray(gt[i:i + 32])))
                            for i in range(0, len(gt), 32)])[:, 32:].sum(1)
        i = int(np.argmin(np.abs(E - np.median(E))))
        v = float(np.percentile(np.abs(gt[i, ..., 1]), 99.5))
        for ci, (lab, f) in enumerate(cols):
            ax = axes[ri][ci]
            img = gt[i, ..., 1] if f is None else f[i, ..., 1]
            ax.imshow(img, cmap='RdBu_r', vmin=-v, vmax=v)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0 or 'trained here' in lab or lab == 'ground truth':
                ax.set_title(lab if ri == 0 or ci == ncol - 1 else '',
                             fontsize=style.TITLE_FS - 1)
            if ci == 0:
                ax.set_ylabel(f'Re={R}', fontsize=style.TITLE_FS)
    fig.suptitle('The two extreme specialists carried across the ladder, against the gated '
                 'in-regime model (median test triplet per regime, vorticity)',
                 fontsize=style.SUP_FS)
    for d_, ext in [('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')]:
        os.makedirs(d_, exist_ok=True)
        p_ = f'{d_}/transfer_grid.{ext}'
        fig.savefig(p_)
        print('written', p_)


if __name__ == '__main__':
    main()
