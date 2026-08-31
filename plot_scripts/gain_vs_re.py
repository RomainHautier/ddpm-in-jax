"""Why a fine-tune fails out of distribution, as one figure.

Left: the gain each mechanism applies, against the gain that would be REQUIRED to reach truth
(1 / base retention). A fine-tune's gain is flat - it applies the dose it learned - so it tracks
the required curve only where it was trained and falls away everywhere else. The dial's gain RISES
with Reynolds number, because it re-measures against each regime's anchor.
Right: the resulting retention, i.e. what that failure costs.

  JAX_PLATFORMS=cpu python plot_scripts/gain_vs_re.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib.pyplot as plt
import style

CONFIG = dict(regimes=[1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
              fig_w=11.6, fig_h=4.3, fontsize=style.BASE_FS, dpi=150,
              outdir='docs/figs_overleaf', pngdir='plotting/figs')
BASE, FT = ('base0', 'mop0.2_0'), ('mt1k-0499', 'mop0.2_0')
BASE_D, FT_D = ('base0', 'tapp0.2_3'), ('mt1k-0499', 'tapp0.2_3')


def ret(R, m, sg, band=(32, 96)):
    """Band retention computed from the stored mean spectra, so ANY band works - including the
    full spectrum. The stored `ret` scalar is [32,96) only."""
    S = np.load('base_results/re1000_audit.npz' if R == 1000
                else f'base_results/regime_audit_re{R}.npz', allow_pickle=True)
    k = f'{R}|{m}|K3|{sg}||E'
    if k not in S.files: return np.nan
    lo, hi = band
    return float(np.asarray(S[k])[lo:hi].sum() / np.asarray(S[f'{R}|GT||E'])[lo:hi].sum())


def make(c):
    style.apply(c['fontsize'], c['dpi'])
    R = np.array([r for r in c['regimes']]); bd = c['band']
    rb = np.array([ret(r, *BASE, bd) for r in R])
    rf = np.array([ret(r, *FT, bd) for r in R])
    rbd = np.array([ret(r, *BASE_D, bd) for r in R])
    rfd = np.array([ret(r, *FT_D, bd) for r in R])
    fig, ax = plt.subplots(1, 2, figsize=(c['fig_w'], c['fig_h']), constrained_layout=True)

    a = ax[0]
    a.plot(R, 1 / rb, '-', color=style.GT_COLOR, lw=2.4, marker='o', ms=4,
           label='required to reach truth  (1 / base retention)', zorder=6)
    a.plot(R, rf / rb, '--', color='#c22f4f', lw=1.9, marker='s', ms=4,
           label='fine-tune  (weights)')
    a.plot(R, rbd / rb, '-', color='#28658a', lw=1.9, marker='^', ms=4,
           label='dial on the base model')
    a.plot(R, rfd / rf, ':', color='#0f9e78', lw=2.1, marker='v', ms=4,
           label='dial on the fine-tune')
    # annotations placed RELATIVE TO THE DATA: over the full spectrum every gain is ~1 and
    # fixed coordinates would land them off-panel or on top of the curves.
    req, gf = 1 / rb, rf / rb
    ytop = max(float(np.nanmax(req)), float(np.nanmax(rbd / rb))) * 1.15
    ybot = min(0.95, float(np.nanmin(rfd / rf)) * 0.92)
    if req[-1] / gf[-1] > 1.25:                    # only mark a gap that actually exists
        a.annotate('', xy=(R[-1], req[-1]), xytext=(R[-1], gf[-1]),
                   arrowprops=dict(arrowstyle='<->', color='#5d645d', lw=1.4))
        a.text(R[-1] * 0.92, np.sqrt(req[-1] * gf[-1]), 'the gap a fixed\ndose cannot close',
               fontsize=style.LEG_FS - 0.5, color='#5d645d', ha='right', va='center')
    # only claim "calibrated here" where the fine-tune's gain actually MEETS the requirement -
    # over the full spectrum it never does, because the deficit is invisible in total energy
    if abs(gf[0] / req[0] - 1) < 0.06:
        a.plot([R[0]], [gf[0]], 'o', ms=11, mfc='none', mec='#5d645d', mew=1.4)
        a.text(R[0] * 1.08, ybot + (ytop - ybot) * 0.12, 'calibrated here,\nand only here',
               fontsize=style.LEG_FS - 0.5, color='#5d645d', ha='left', va='center')
    style.re_axis(a); a.set_xlabel('Reynolds number')
    a.set_ylabel('gain applied  ($E_{\\mathrm{cfg}}/E_{\\mathrm{base}}$)')
    a.set_title('what each mechanism multiplies the base model by', fontsize=style.TITLE_FS)
    a.set_ylim(ybot, ytop); a.legend(fontsize=style.LEG_FS, loc='upper left')

    b = ax[1]
    b.axhline(1, color=style.GT_COLOR, lw=1.8, zorder=5, label='ground truth')
    b.plot(R, rb, '--', color='#9aa198', lw=1.9, marker='o', ms=4, label='base, unguided')
    b.plot(R, rf, '--', color='#c22f4f', lw=1.9, marker='s', ms=4, label='fine-tune, unguided')
    b.plot(R, rbd, '-', color='#28658a', lw=1.9, marker='^', ms=4, label='base + dial')
    b.plot(R, rfd, '-', color='#0f9e78', lw=1.9, marker='v', ms=4, label='fine-tune + dial')
    style.re_axis(b); b.set_xlabel('Reynolds number')
    b.set_ylabel(f"retention  $E_{{{c['blabel']}}}/E_{{\\mathrm{{GT}}}}$")
    b.set_title('and what that leaves on the table', fontsize=style.TITLE_FS)
    b.set_ylim(0, max(1.2, float(np.nanmax(rfd)) * 1.1))
    b.legend(fontsize=style.LEG_FS, loc='lower left')

    fig.suptitle(f"A fine-tune applies a fixed dose; a dial re-measures  "
                 f"&mdash; over {c['btitle']}".replace('&mdash;', '\u2014'), fontsize=style.SUP_FS)
    for d, ext in ((c['outdir'], 'pdf'), (c['pngdir'], 'png')):
        os.makedirs(d, exist_ok=True)
        p = f"{d}/gain_vs_re{c['suffix']}.{ext}"; fig.savefig(p); print('  written', p)
    plt.close(fig)
    print('\n  Re   required   fine-tune   dial/base   dial/ft')
    for i, r in enumerate(R):
        print(f'{r:5d}{1/rb[i]:10.2f}{rf[i]/rb[i]:12.2f}{rbd[i]/rb[i]:12.2f}{rfd[i]/rf[i]:10.2f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--regimes')
    ap.add_argument('--band', default='32,96',
                    help="'32,96' (default), '16,96', or 'full' for the whole spectrum k>=1")
    a = ap.parse_args(); c = dict(CONFIG)
    if a.regimes: c['regimes'] = [int(x) for x in a.regimes.split(',')]
    if a.band == 'full':
        c['band'] = (1, 128); c['suffix'] = '_full'
        c['blabel'] = 'k\\geq 1'; c['btitle'] = 'the FULL spectrum'
    else:
        lo, hi = (int(x) for x in a.band.split(','))
        c['band'] = (lo, hi)
        c['suffix'] = '' if (lo, hi) == (32, 96) else f'_k{lo}'
        c['blabel'] = f'[{lo},{hi})'; c['btitle'] = f'$k\\in[{lo},{hi})$'
    make(c)
