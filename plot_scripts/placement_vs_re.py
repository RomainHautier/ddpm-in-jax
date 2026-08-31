"""Per-triplet placement against Reynolds number, for every model in the study.

Placement is the correlation of the local band-energy map with the ground truth's, computed ONE
CORRELATION PER TRIPLET and aggregated (median) - the pooled version folds frame-to-frame dose
error into an apparent spatial score.

Left:  does fine-tuning cost placement?  (all models, unguided)
Right: does steering cost placement?     (base and the Re=1000 fine-tune, with and without)

  JAX_PLATFORMS=cpu python plot_scripts/placement_vs_re.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib.pyplot as plt
import style

CONFIG = dict(regimes=[1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
              fig_w=8.2, fig_h=5.0, fontsize=style.BASE_FS, dpi=150,
              outdir='docs/figs_overleaf', pngdir='plotting/figs')
# colour = MODEL (style.MODEL_COLOR, shared across every figure)
# linestyle = STEERING: solid unguided, dashed matched dial, dotted gated dial
ROWS = [('base0', 'mop0.2_0', 'base', style.LS_UNGUIDED),
        ('base0', 'tapp0.2_3', 'base + dial', style.LS_DIAL),
        ('base0', 'v7bandgate', 'base + gated dial', style.LS_GATE),
        ('mt1k-0499', 'mop0.2_0', 'Re=1000 fine-tune', style.LS_UNGUIDED),
        ('mt1k-0499', 'tapp0.2_3', 'Re=1000 fine-tune + dial', style.LS_DIAL),
        ('mt2k-0599', 'mop0.2_0', 'Re=2000 fine-tune', style.LS_UNGUIDED),
        ('r8kp02-0599', 'mop0.2_0', 'Re=8000 fine-tune', style.LS_UNGUIDED)]
MARK = {'base0': 'o', 'mt1k-0499': 's', 'mt2k-0599': '^', 'mt8k-0549': 'v', 'r8kp02-0599': 'D'}


def place(R, m, sg):
    S = np.load('base_results/re1000_audit.npz' if R == 1000
                else f'base_results/regime_audit_re{R}.npz', allow_pickle=True)
    F = set(S.files)
    for f in ('pst_place', 'ps_place'):
        k = f'{R}|{m}|K3|{sg}||{f}'
        if k in F: return float(np.median(np.asarray(S[k])))
    return np.nan


def make(c):
    style.apply(c['fontsize'], c['dpi'])
    regs = [R for R in c['regimes'] if os.path.exists(
        'base_results/re1000_audit.npz' if R == 1000 else f'base_results/regime_audit_re{R}.npz')]
    fig, ax = plt.subplots(figsize=(c['fig_w'], c['fig_h']), constrained_layout=True)
    for m, sg, lab, ls in ROWS:
        y = np.array([place(R, m, sg) for R in regs])
        if np.all(np.isnan(y)): continue
        ax.plot(regs, y, ls, color=style.MODEL_COLOR[m], lw=1.9,
                marker=MARK.get(m, 'o'), ms=4.2, label=lab)
    style.re_axis(ax); ax.set_xlabel('Reynolds number')
    ax.set_ylabel('placement, per triplet (median)')
    ax.set_ylim(0.62, 0.95)
    ax.legend(fontsize=style.LEG_FS - 0.5, loc='lower left', ncol=2)
    fig.suptitle('Spatial placement of the small scales ($k\\geq32$ highpass, $\\sigma=6$), '
                 'against Reynolds number', fontsize=style.SUP_FS)
    for d, ext in ((c['outdir'], 'pdf'), (c['pngdir'], 'png')):
        os.makedirs(d, exist_ok=True)
        p = f'{d}/placement_vs_re.{ext}'; fig.savefig(p); print('  written', p)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--regimes'); a = ap.parse_args()
    c = dict(CONFIG)
    if a.regimes: c['regimes'] = [int(x) for x in a.regimes.split(',')]
    make(c)
