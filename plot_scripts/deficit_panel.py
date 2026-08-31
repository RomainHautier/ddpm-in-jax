"""One row, five panels: the sample, both reconstructions, and the fine-scale energy each still owes.

  ground truth | base recon | fine-tune recon | deficit of the base | deficit of the fine-tune

The first three are vorticity. The last two are GT minus model in local fine-scale (k>=KCUT)
energy density, on a shared symmetric colour scale, so red is energy the model is MISSING and blue
is energy it has ADDED beyond the truth. Same triplet throughout.

  JAX_PLATFORMS=cpu python plot_scripts/deficit_panel.py [--triplet N] [--kcut 16]
"""
import os, sys, argparse
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'plot_scripts'); sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, style
import matplotlib.pyplot as plt
from viz_energy import local_hik_energy

CONFIG = dict(
    fields='base_results/fields/re1000',
    base=('base0__K3__none', 'base model'),
    ft=('mt1k-0499__K3__mo0', 'fine-tuned model'),
    triplet=None,          # None -> the triplet whose base deficit is the median one
    kcut=16, sigma=6.0, sig=4.7988,
    crop=None,             # e.g. 128 to zoom into a 128x128 corner
    out='plotting/figs/deficit_panel.pdf')
p = argparse.ArgumentParser()
p.add_argument('--triplet', type=int); p.add_argument('--kcut', type=int)
p.add_argument('--crop', type=int); p.add_argument('--out')
a = p.parse_args()
for k in ('triplet', 'kcut', 'crop', 'out'):
    if getattr(a, k) is not None: CONFIG[k] = getattr(a, k)

F, SIG, K = CONFIG['fields'], CONFIG['sig'], CONFIG['kcut']
load = lambda n: np.load(f'{F}/{n}.npz')['x'].astype(np.float32)
gt, yb, yf = load('GT'), load(CONFIG['base'][0]), load(CONFIG['ft'][0])
# ALIGNMENT GUARD: the stored GT is the FULL audit pool; at OOD regimes the model rows are the
# TEST subset only, and indexing both with the same integer pairs a reconstruction with another
# snapshot's truth. Equal lengths hold at Re=1000; refuse rather than plot nonsense elsewhere.
assert len(gt) == len(yb) == len(yf), (
    f'GT has {len(gt)} triplets, models {len(yb)}/{len(yf)} - mask the GT before indexing')
emap = lambda x: local_hik_energy(x[..., 1] * SIG, K, CONFIG['sigma'])
G, B, Ff = emap(gt), emap(yb), emap(yf)
DB, DF = G - B, G - Ff                       # positive = the model is short of the truth
i = CONFIG['triplet'] if CONFIG['triplet'] is not None else int(np.argsort(DB.mean((1, 2)))[len(DB) // 2])
sl = slice(0, CONFIG['crop']) if CONFIG['crop'] else slice(None)
print(f"triplet {i} of {len(gt)}; band k>={K}; base deficit {DB[i].mean():.3g}, "
      f"fine-tune deficit {DF[i].mean():.3g} (mean over the frame)")

style.apply()
fig, AX = plt.subplots(1, 5, figsize=(19, 4.3), constrained_layout=True)
w = lambda x: x[i, sl, sl, 1] * SIG
vw = np.percentile(np.abs(w(gt)), 99.5)
for ax, f, t in ((AX[0], gt, 'ground truth'), (AX[1], yb, CONFIG['base'][1]),
                 (AX[2], yf, CONFIG['ft'][1])):
    im = ax.imshow(w(f), cmap='RdBu_r', vmin=-vw, vmax=vw)
    ax.set_title(t, fontsize=style.TITLE_FS); ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(im, ax=AX[:3], fraction=.020, pad=.01)
cb.set_label(r'vorticity $\omega$', fontsize=style.LEG_FS)

vd = np.percentile(np.abs(np.concatenate([DB[i, sl, sl], DF[i, sl, sl]])), 99.5)
for ax, D, t, tot in ((AX[3], DB, f'energy the {CONFIG["base"][1]} is missing', DB[i].mean()),
                      (AX[4], DF, f'energy the {CONFIG["ft"][1]} is missing', DF[i].mean())):
    im2 = ax.imshow(D[i, sl, sl], cmap='RdBu_r', vmin=-vd, vmax=vd)
    ax.set_title(t, fontsize=style.TITLE_FS); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f'mean deficit {tot:.2e}', fontsize=style.LEG_FS)
cb2 = fig.colorbar(im2, ax=AX[3:], fraction=.020, pad=.01)
cb2.set_label(f'GT $-$ model, local energy at $k\\geq{K}$', fontsize=style.LEG_FS)
fig.suptitle(f'Where each model is still short of the truth — Re=1000, held-out triplet {i}, '
             f'$k\\geq{K}$   (red = missing, blue = excess)', fontsize=style.SUP_FS)
os.makedirs(os.path.dirname(CONFIG['out']), exist_ok=True)
fig.savefig(CONFIG['out'], bbox_inches='tight')
fig.savefig(CONFIG['out'].replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
print('wrote', CONFIG['out'])
for lab, D in ((CONFIG['base'][1], DB), (CONFIG['ft'][1], DF)):
    d = D[i]
    print(f"  {lab:<20} mean {d.mean():+.3e}   |deficit| {np.abs(d).mean():.3e}   "
          f"fraction of pixels short {np.mean(d>0)*100:.0f}%")
