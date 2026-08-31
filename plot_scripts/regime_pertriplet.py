"""Per-triplet retention and placement vs Reynolds number, per model and sampling strategy.

Both metrics are computed once per reconstructed triplet against THAT triplet's own ground truth,
then aggregated across triplets - never pooled into a single global correlation (a pooled placement
number folds frame-to-frame dose error into what looks like a spatial score).

  retention  = E_model[32,96) / E_GT[32,96) for that triplet   (1.0 = correct dose)
  in-band    = share of triplets within +-20% of their own truth
  placement  = corr(local fine-scale energy map, GT's) for that triplet

  python plot_scripts/regime_pertriplet.py [--out FILE] [--models base0,r1k-449]
"""
import os, sys, argparse
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'plot_scripts')
import numpy as np, style
import matplotlib.pyplot as plt

CONFIG = dict(
    regimes=[1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
    models=['base0', 'r1k-449'],
    strats=[('none', 'unguided', '#9aa198'),
            ('rewardv2', 'dose dial v2', '#c22f4f'),
            ('all3v2', 'all three dials', '#d4770a'),
            ('v6gate', 'target-gated stack', '#0f9e78')],
    tol=0.2, band=(10, 90), out='plotting/figs/regime_pertriplet.pdf',
)
p = argparse.ArgumentParser(); p.add_argument('--out'); p.add_argument('--models')
a = p.parse_args()
if a.out: CONFIG['out'] = a.out
if a.models: CONFIG['models'] = a.models.split(',')

STORES = {}
for R in CONFIG['regimes']:
    f = 'base_results/re1000_audit.npz' if R == 1000 else f'base_results/regime_audit_re{R}.npz'
    if os.path.exists(f): STORES[R] = np.load(f, allow_pickle=True)


# per-FRAME metrics averaged over the triplet's three frames where the backfill has produced
# them; the fallback is the middle frame only, which they match to within 0.002
PREFER = {'ps_ret_paired': 'pst_ret', 'ps_place': 'pst_place'}


def ps(R, m, sg, field):
    A = STORES.get(R)
    if A is None: return None
    for f in (PREFER.get(field), field):
        if f and f'{R}|{m}|K3|{sg}||{f}' in A.files:
            return np.asarray(A[f'{R}|{m}|K3|{sg}||{f}'])
    return None


def series(m, sg, field, stat):
    """(regimes, values) for a per-triplet field reduced by `stat`, skipping regimes not yet run"""
    xs, ys = [], []
    for R in CONFIG['regimes']:
        v = ps(R, m, sg, field)
        if v is None: continue
        xs.append(R); ys.append(stat(v))
    return np.array(xs), np.array(ys)


style.apply()
M, S = CONFIG['models'], CONFIG['strats']
lo, hi = CONFIG['band']
fig, AX = plt.subplots(3, 3, figsize=(14.5, 11.2))

for r, m in enumerate(M):
    for sg, lab, c in S:
        # --- retention -------------------------------------------------------------------
        x, y = series(m, sg, 'ps_ret_paired', np.median)
        if len(x):
            _, yl = series(m, sg, 'ps_ret_paired', lambda v: np.percentile(v, lo))
            _, yh = series(m, sg, 'ps_ret_paired', lambda v: np.percentile(v, hi))
            AX[r][0].fill_between(x, yl, yh, color=c, alpha=.13, lw=0)
            AX[r][0].plot(x, y, 'o-', color=c, ms=4, lw=1.8, label=lab)
        # --- in-band ---------------------------------------------------------------------
        x, y = series(m, sg, 'ps_ret_paired',
                      lambda v: np.mean(np.abs(v - 1) < CONFIG['tol']) * 100)
        if len(x): AX[r][1].plot(x, y, 'o-', color=c, ms=4, lw=1.8, label=lab)
        # --- placement -------------------------------------------------------------------
        x, y = series(m, sg, 'ps_place', np.median)
        if len(x):
            _, yl = series(m, sg, 'ps_place', lambda v: np.percentile(v, lo))
            _, yh = series(m, sg, 'ps_place', lambda v: np.percentile(v, hi))
            AX[r][2].fill_between(x, yl, yh, color=c, alpha=.13, lw=0)
            AX[r][2].plot(x, y, 'o-', color=c, ms=4, lw=1.8, label=lab)
    AX[r][0].axhline(1, color=style.GT_COLOR, lw=2, zorder=0)
    AX[r][0].set_ylabel('per-triplet retention\nmedian, 10-90%'); AX[r][0].set_ylim(0.2, 1.9)
    AX[r][1].set_ylabel('triplets within +-20%  (%)'); AX[r][1].set_ylim(-3, 103)
    AX[r][2].set_ylabel('per-triplet placement\nmedian, 10-90%'); AX[r][2].set_ylim(0.45, 0.98)
    for cix, t in enumerate(['dose per triplet', 'agreement per triplet', 'placement per triplet']):
        AX[r][cix].set_title(f'{style.MODEL.get(m, m)} — {t}', fontsize=style.TITLE_FS)
        style.re_axis(AX[r][cix], CONFIG['regimes'])
        AX[r][cix].legend(fontsize=style.LEG_FS, loc='lower left')

# --- bottom row: fine-tune minus base, the same three metrics ------------------------------
if len(M) >= 2:
    mb, mf = M[0], M[1]
    for cix, (field, stat, lab, zero) in enumerate([
            ('ps_ret_paired', np.median, 'retention (median)', 0.0),
            ('ps_ret_paired', lambda v: np.mean(np.abs(v - 1) < CONFIG['tol']) * 100,
             'triplets in-band (percentage points)', 0.0),
            ('ps_place', np.median, 'placement (median)', 0.0)]):
        ax = AX[2][cix]
        for sg, slab, c in S:
            xb, yb = series(mb, sg, field, stat)
            xf, yf = series(mf, sg, field, stat)
            common = np.intersect1d(xb, xf)
            if not len(common): continue
            d = np.array([yf[list(xf).index(R)] - yb[list(xb).index(R)] for R in common])
            ax.plot(common, d, 'o-', color=c, ms=4, lw=1.8, label=slab)
        ax.axhline(zero, color=style.INK, lw=1.2, ls='--')
        ax.set_ylabel(f'fine-tune - base\n{lab}')
        ax.set_title(f'advantage of the fine-tune — {lab.split(" (")[0]}', fontsize=style.TITLE_FS)
        style.re_axis(ax, CONFIG['regimes']); ax.legend(fontsize=style.LEG_FS, loc='best')
        ax.set_xlabel('Reynolds number')

fig.suptitle('Per-triplet dose and placement across regimes — every metric computed against each '
             "triplet's own ground truth", fontsize=12, y=.997)
fig.tight_layout(rect=[0, 0, 1, .975])
os.makedirs(os.path.dirname(CONFIG['out']), exist_ok=True)
fig.savefig(CONFIG['out'], bbox_inches='tight')
fig.savefig(CONFIG['out'].replace('.pdf', '.png'), bbox_inches='tight', dpi=145)
print('wrote', CONFIG['out'])
