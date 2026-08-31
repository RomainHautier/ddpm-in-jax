"""Where the injected energy goes, shell by shell, against where it is NEEDED.

For each shell k, deficit(k) = E_GT(k) - E_base(k) is what is missing, and
injection(k) = E_cfg(k) - E_base(k) is what a configuration put back. A point on the diagonal
means that shell was filled exactly. The FIT SLOPE is the fraction of the deficit a mechanism
supplies; the CORRELATION is whether it puts energy where it is needed at all.

  JAX_PLATFORMS=cpu python plot_scripts/deficit_targeting.py [--regimes 1000,4000,8000]
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib.pyplot as plt
import style

CONFIG = dict(panels=[1000, 4000, 8000], allregs=[1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
              klo=16, khi=96, panel_w=3.5, panel_h=3.4, fontsize=style.BASE_FS, dpi=150,
              outdir='docs/figs_overleaf', pngdir='plotting/figs')
ROWS = [('mt1k-0499', 'mop0.2_0', 'fine-tune', '#c22f4f', 's'),
        ('base0', 'tapp0.2_3', 'dial on base', '#28658a', '^')]


def spec(R, m, sg):
    S = np.load('base_results/re1000_audit.npz' if R == 1000
                else f'base_results/regime_audit_re{R}.npz', allow_pickle=True)
    return np.asarray(S[f'{R}|{m}|K3|{sg}||E'])


def fit(R, m, sg, klo, khi):
    Eb = spec(R, 'base0', 'mop0.2_0'); Eg = spec(R, 'GT', '') if False else None
    S = np.load('base_results/re1000_audit.npz' if R == 1000
                else f'base_results/regime_audit_re{R}.npz', allow_pickle=True)
    Eg = np.asarray(S[f'{R}|GT||E'])
    d = Eg[klo:khi] - Eb[klo:khi]; inj = spec(R, m, sg)[klo:khi] - Eb[klo:khi]
    return d, inj, np.polyfit(d, inj, 1)[0], np.corrcoef(d, inj)[0, 1]


def make(c):
    style.apply(c['fontsize'], c['dpi'])
    n = len(c['panels'])
    fig, ax = plt.subplots(1, n + 1, figsize=(c['panel_w'] * (n + 1), c['panel_h']),
                           constrained_layout=True)
    klo, khi = c['klo'], c['khi']
    for i, R in enumerate(c['panels']):
        a = ax[i]
        lims = None
        for m, sg, lab, col, mk in ROWS:
            d, inj, sl, r = fit(R, m, sg, klo, khi)
            a.scatter(d, inj, s=11, marker=mk, color=col, alpha=.55, lw=0,
                      label=f'{lab}: {sl:.2f}$\\times$, r={r:.2f}')
            xs = np.array([d.min(), d.max()])
            a.plot(xs, sl * xs, '-', color=col, lw=1.3, alpha=.8)
            lims = (min(d.min(), inj.min()), max(d.max(), inj.max()))
        a.plot(lims, lims, '-', color=style.GT_COLOR, lw=1.8, zorder=5,
               label='deficit filled exactly' if i == 0 else None)
        a.set(xscale='log', yscale='log')
        a.set_title(f'Re = {R}', fontsize=style.TITLE_FS)
        a.set_xlabel('deficit in shell $k$   $E_{\\mathrm{GT}}-E_{\\mathrm{base}}$')
        if i == 0: a.set_ylabel('injected   $E_{\\mathrm{cfg}}-E_{\\mathrm{base}}$')
        a.legend(fontsize=style.LEG_FS - 1, loc='upper left')

    a = ax[n]
    R = np.array(c['allregs'])
    for m, sg, lab, col, mk in ROWS:
        sl = np.array([fit(r, m, sg, klo, khi)[2] for r in R])
        rr = np.array([fit(r, m, sg, klo, khi)[3] for r in R])
        a.plot(R, sl, '-', color=col, lw=1.9, marker=mk, ms=4, label=f'{lab} — fraction filled')
        a.plot(R, rr, ':', color=col, lw=1.6, marker=mk, ms=3, alpha=.75,
               label=f'{lab} — correlation r')
    a.axhline(1, color=style.GT_COLOR, lw=1.4)
    style.re_axis(a); a.set_xlabel('Reynolds number'); a.set_ylim(0, 1.12)
    a.set_ylabel('fraction of the deficit filled  /  correlation')
    a.set_title('across every regime', fontsize=style.TITLE_FS)
    a.legend(fontsize=style.LEG_FS - 1, loc='center left')
    fig.suptitle(f'Does the injected energy go where it is missing?  '
                 f'Shells $k\\in[{klo},{khi})$, one point per shell', fontsize=style.SUP_FS)
    for d_, ext in ((c['outdir'], 'pdf'), (c['pngdir'], 'png')):
        os.makedirs(d_, exist_ok=True)
        p = f'{d_}/deficit_targeting.{ext}'; fig.savefig(p); print('  written', p)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--regimes'); a = ap.parse_args()
    c = dict(CONFIG)
    if a.regimes: c['panels'] = [int(x) for x in a.regimes.split(',')]
    make(c)
