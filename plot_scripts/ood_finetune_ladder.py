"""Cross-regime transfer of every finetune, with and without the tapered base dial.

Left: retention against Re. Middle: share of triplets within +-20% of their own truth.
Right: the same retention with the dial on. Colour = model (style.MODEL_COLOR), so each curve's
peak marks the regime the model serves best; the diagonal of peaks is the point.

  JAX_PLATFORMS=cpu python plot_scripts/ood_finetune_ladder.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib.pyplot as plt
import style

CONFIG = dict(regimes=[1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
              fig_w=13.6, fig_h=4.3, fontsize=style.BASE_FS, dpi=150,
              outdir='docs/figs_overleaf', pngdir='plotting/figs')
MODELS = [('base0', 'base', 'o'), ('mt1k-0499', 'Re=1000 ft', 's'),
          ('mt2k-0599', 'Re=2000 ft', '^'), ('r8kp02-0599', 'Re=8000 ft', 'D')]


def get(R, m, sg, f):
    p = 'base_results/re1000_audit.npz' if R == 1000 else f'base_results/regime_audit_re{R}.npz'
    S = np.load(p, allow_pickle=True)
    if f == 'inband':
        for ff in ('pst_ret', 'ps_ret_paired'):
            k = f'{R}|{m}|K3|{sg}||{ff}'
            if k in S.files: return float(np.mean(np.abs(np.asarray(S[k]) - 1) < .2)) * 100
        return np.nan
    k = f'{R}|{m}|K3|{sg}||{f}'
    return float(S[k]) if k in S.files else np.nan


def make(c):
    style.apply(c['fontsize'], c['dpi'])
    R = np.array(c['regimes'])
    fig, ax = plt.subplots(1, 3, figsize=(c['fig_w'], c['fig_h']), constrained_layout=True,
                           sharex=True)
    panels = [('mop0.2_0', 'ret', 'retention, unguided', 0),
              ('mop0.2_0', 'inband', 'in-band, unguided', 1),
              ('tapp0.2_3', 'ret', 'retention, tapered dial $\\lambda=3$', 2)]
    for sg, f, title, i in panels:
        a = ax[i]
        for m, lab, mk in MODELS:
            y = np.array([get(r, m, sg, f) for r in R])
            a.plot(R, y, '-', color=style.MODEL_COLOR[m], lw=1.9, marker=mk, ms=4.2,
                   label=lab if i == 0 else None)
        a.axhline(100 if f == 'inband' else 1, color=style.GT_COLOR, lw=1.5)
        style.re_axis(a); a.set_xlabel('Reynolds number')
        a.set_title(title, fontsize=style.TITLE_FS)
        if f == 'inband': a.set_ylabel('% of triplets within $\\pm20\\%$ of their own truth')
        elif i == 0: a.set_ylabel('$E_{[32,96)}/E_{\\mathrm{GT}}$')
        a.set_ylim((0, 105) if f == 'inband' else (0, 3.9))
    ax[0].legend(fontsize=style.LEG_FS, loc='upper right')
    fig.suptitle('Every finetune carried across every regime — each peaks where it was trained, '
                 'and nowhere else', fontsize=style.SUP_FS)
    for d, ext in ((c['outdir'], 'pdf'), (c['pngdir'], 'png')):
        os.makedirs(d, exist_ok=True)
        p = f'{d}/ood_finetune_ladder.{ext}'; fig.savefig(p); print('  written', p)


if __name__ == '__main__':
    argparse.ArgumentParser().parse_args()
    make(dict(CONFIG))
