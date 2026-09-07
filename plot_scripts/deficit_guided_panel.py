"""Two rows, guided and unguided, one shared sample: the ground truth at the left
spanning both rows, then per row (base above, fine-tune below) the reconstruction
without and with the tapered dial and the two matching fine-scale deficit maps.

  JAX_PLATFORMS=cpu python plot_scripts/deficit_guided_panel.py [--triplet N]
"""
import os, sys, argparse
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'plot_scripts'); sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, style
import matplotlib.pyplot as plt
from viz_energy import local_hik_energy

CONFIG = dict(fields='base_results/fields/re1000', kcut=10, sigma=6.0, sig=4.7988,
              triplet=None)
GUIDE = os.environ.get('GUIDE', 'tapp0.2_3')
GLABEL = {'tapp0.2_3': '$+$ dial', 'v7bandgate': '$+$ gate',
          'v8lowband': '$+$ gate'}.get(GUIDE, '$+$ guide')
ROWS = [('base', 'base0__K3__mop0.2_0', f'base0__K3__{GUIDE}'),
        ('fine-tune', 'mt1k-0499__K3__mop0.2_0', f'mt1k-0499__K3__{GUIDE}')]
# COMPARE=1: the two variants per row are the tapered dial and the per-sample gate,
# instead of unguided and GUIDE
COMPARE = bool(os.environ.get('COMPARE'))
if COMPARE:
    LAB_U, GLABEL = '$+$ dial', '$+$ gate'
    ROWS = [('base', 'base0__K3__tapp0.2_3', 'base0__K3__v8lowband'),
            ('fine-tune', 'mt1k-0499__K3__tapp0.2_3', 'mt1k-0499__K3__v8lowband')]
else:
    LAB_U = 'unguided'

p = argparse.ArgumentParser(); p.add_argument('--triplet', type=int)
a = p.parse_args()
if a.triplet is not None: CONFIG['triplet'] = a.triplet

F, SIG, K = CONFIG['fields'], CONFIG['sig'], CONFIG['kcut']
load = lambda n: np.load(f'{F}/{n}.npz')['x'].astype(np.float32)
gt = load('GT')
fields = {tok: load(tok) for _, u, d in ROWS for tok in (u, d)}
for tok, x in fields.items():
    assert len(x) == len(gt), f'{tok}: {len(x)} vs GT {len(gt)}'
emap = lambda x: local_hik_energy(x[..., 1] * SIG, K, CONFIG['sigma'])
G = emap(gt)
D = {tok: G - emap(x) for tok, x in fields.items()}   # positive = model short of truth
DB = G - emap(load('base0__K3__mop0.2_0'))   # triplet choice always on the unguided base
i = (CONFIG['triplet'] if CONFIG['triplet'] is not None
     else int(np.argsort(DB.mean((1, 2)))[len(DB) // 2]))
print(f'triplet {i}, k>={K}')

style.apply()
fig = plt.figure(figsize=(16.6, 6.8), constrained_layout=True)
gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1])
w = lambda x: x[i, ..., 1] * SIG
vw = np.percentile(np.abs(w(gt)), 99.5)
vd = np.percentile(np.abs(np.concatenate([D[tok][i] for tok in D])), 99.5)

axg = fig.add_subplot(gs[:, 0])
axg.imshow(w(gt), cmap='RdBu_r', vmin=-vw, vmax=vw)
axg.set_title('ground truth', fontsize=style.TITLE_FS)
axg.set_xticks([]); axg.set_yticks([])

imw = imd = None
for ri, (lab, utok, dtok) in enumerate(ROWS):
    for ci, (tok, sub) in enumerate([(utok, LAB_U), (dtok, GLABEL)]):
        ax = fig.add_subplot(gs[ri, 1 + ci])
        imw = ax.imshow(w(fields[tok]), cmap='RdBu_r', vmin=-vw, vmax=vw)
        ax.set_title(f'{lab}, {sub}', fontsize=style.TITLE_FS - 1)
        ax.set_xticks([]); ax.set_yticks([])
    for ci, (tok, sub) in enumerate([(utok, LAB_U), (dtok, GLABEL)]):
        ax = fig.add_subplot(gs[ri, 3 + ci])
        imd = ax.imshow(D[tok][i], cmap='RdBu_r', vmin=-vd, vmax=vd)
        ax.set_title(f'deficit, {sub}', fontsize=style.TITLE_FS - 1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f'mean {D[tok][i].mean():+.2e},  '
                      f'{np.mean(D[tok][i] > 0) * 100:.0f}% short',
                      fontsize=style.LEG_FS - 1)
cb = fig.colorbar(imw, ax=axg, location='left', fraction=.05, pad=.02)
cb.set_label(r'vorticity $\omega$', fontsize=style.LEG_FS)
cb2 = fig.colorbar(imd, ax=[fig.axes[j] for j in range(len(fig.axes))
                            if fig.axes[j].get_title().startswith('deficit')][-2:],
                   fraction=.04, pad=.01)
cb2.set_label(f'GT $-$ model, local energy at $k\\geq{K}$', fontsize=style.LEG_FS)
fig.suptitle(f'Reconstructions and fine-scale deficit, {LAB_U} and {GLABEL} '
             f'— Re=1000, held-out triplet {i}  (red = missing, blue = excess)',
             fontsize=style.SUP_FS)
for d_, ext in (('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')):
    sfx = ('_dialvsgate' if COMPARE else
           '' if GUIDE == 'tapp0.2_3' else '_gate')
    p_ = f'{d_}/re1000_deficit_guided{sfx}.{ext}'; fig.savefig(p_); print('written', p_)
