"""Reconstructions of the downward transfer: one Re=1000 sample under the Re=8000 specialist,
raw (K3) and chain-corrected ([100]), beside the home specialist. Test-pool rows only, aligned.

  JAX_PLATFORMS=cpu python plot_scripts/chain_row.py [--idx N]
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np
import matplotlib.pyplot as plt
import style
from viz_energy import local_hik_energy

SIG = 4.7988
PANELS = [('GT', None, 'ground truth'),
          ('mt1k-0499__K3__mop0.2_0', None, 'Re=1000 finetune\n(home specialist, K3)'),
          ('r8kp02-0599__K3__mop0.2_0', None, 'Re=8000 finetune, K3\n(raw downward transfer)'),
          ('r8kp02-0599__K100__ccp0.2_0', None, 'Re=8000 finetune, [100] chain\n(dose set by the sampler)')]


def make(idx):
    style.apply(style.BASE_FS, 170)
    F = 'base_results/fields/re1000'
    xs = {n: np.load(f'{F}/{n}.npz')['x'].astype(np.float32) for n, _, _ in PANELS}
    n = min(len(v) for v in xs.values())
    assert all(len(v) == n for v in xs.values()), 'pool mismatch'
    gt = xs['GT']
    from src.rewards import make_spectrum_fn
    import jax.numpy as jnp
    E = np.asarray(make_spectrum_fn(256)(jnp.asarray(gt)))[:, 32:96].sum(1)
    i = idx if idx is not None else int(np.argsort(E)[len(E) // 2])
    _c = np.median([np.corrcoef(gt[j, ..., 1].ravel(),
                                xs['r8kp02-0599__K100__ccp0.2_0'][j, ..., 1].ravel())[0, 1]
                    for j in range(12)])
    assert _c > 0.5, f'misaligned ({_c:.2f})'
    w = gt[i, ..., 1] * SIG
    v = float(np.percentile(np.abs(w), 99.5))
    fig, axes = plt.subplots(2, 4, figsize=(11.6, 6.3), constrained_layout=True)
    Eg = local_hik_energy(w, 32, 6.0)
    defs = []
    for a, (name, _, lab) in zip(axes[0], PANELS):
        f = xs[name][i, ..., 1] * SIG
        a.imshow(f, cmap='RdBu_r', vmin=-v, vmax=v, interpolation='nearest')
        a.set_title(lab, fontsize=style.LEG_FS + 1)
        a.set_xticks([]); a.set_yticks([])
        if name != 'GT': defs.append(Eg - local_hik_energy(f, 32, 6.0))
    dv = float(np.percentile(np.abs(np.stack(defs)), 99.0))
    im0 = axes[1, 0].imshow(Eg, cmap='magma', vmin=0, vmax=float(np.percentile(Eg, 99.5)),
                            interpolation='nearest')
    axes[1, 0].set_title('GT fine-scale energy', fontsize=style.LEG_FS + 1)
    for a, d in zip(axes[1, 1:], defs):
        imd = a.imshow(d, cmap='RdBu_r', vmin=-dv, vmax=dv, interpolation='nearest')
        a.set_title('deficit  GT $-$ model', fontsize=style.LEG_FS + 1)
        sh = np.mean(d > 0) * 100
        a.text(.02, .02, f'{sh:.0f}% short / {100-sh:.0f}% surplus', transform=a.transAxes,
               fontsize=style.LEG_FS - 1.5, va='bottom',
               bbox=dict(fc='white', ec='none', alpha=.75, pad=1.5))
    for a in axes[1]: a.set_xticks([]); a.set_yticks([])
    fig.colorbar(imd, ax=axes[1, 1:], shrink=.8, pad=.01).set_label('local energy deficit',
                                                                    fontsize=style.LEG_FS)
    fig.colorbar(im0, ax=axes[1, 0], shrink=.8, pad=.02)
    fig.suptitle(f'Downward transfer at Re=1000 (test triplet {i}): the chain, not guidance, '
                 'sets the dose', fontsize=style.SUP_FS)
    for d_, ext in (('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')):
        p = f'{d_}/chain_row_re1000.{ext}'; fig.savefig(p); print('  written', p)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--idx', type=int)
    make(ap.parse_args().idx)
