"""How much MORE fine-scale energy is needed, as a function of wavenumber.

Left:  the gain required at each shell, E_GT(k) / E_base(k), one curve per regime. This is the
       target any mechanism has to hit. It grows with k AND with Re, so a single scalar gain
       cannot satisfy it.
Right: the same aggregated over the two bands the results are reported on, [16,96) and [32,96),
       against what the fine-tune and the dial actually deliver.

  JAX_PLATFORMS=cpu python plot_scripts/required_gain.py [--regimes 1000,2000,4000,8000]
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib.pyplot as plt
import style

CONFIG = dict(panels=[1000, 2000, 4000, 8000],
              allregs=[1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
              kmax=96, fig_w=15.0, fig_h=4.4, fontsize=style.BASE_FS, dpi=150,
              outdir='docs/figs_overleaf', pngdir='plotting/figs')
BANDS = [((16, 96), '$k\\in[16,96)$'), ((32, 96), '$k\\in[32,96)$')]
# Colour = MODEL everywhere (blue = base, red = fine-tune), as in every other figure in the set.
# On the left panel, where the curves are REGIMES rather than models, the same two hues are used
# as a cold->hot ramp so the palette stays consistent.
C_REQ = '#232823'
# models and regimes take their colours from style, so they match every other figure


def E(R, m, sg):
    S = np.load('base_results/re1000_audit.npz' if R == 1000
                else f'base_results/regime_audit_re{R}.npz', allow_pickle=True)
    k = f'{R}|GT||E' if m == 'GT' else f'{R}|{m}|K3|{sg}||E'
    return np.asarray(S[k]) if k in S.files else None


def make(c):
    style.apply(c['fontsize'], c['dpi'])
    fig, ax = plt.subplots(1, 3, figsize=(c['fig_w'], c['fig_h']), constrained_layout=True)
    kk = np.arange(1, c['kmax'] + 1)

    # ---- per shell, one curve per regime (cold -> hot) ----
    a = ax[0]
    style.shade_bands(a)
    n = len(c['panels'])
    for i, R in enumerate(c['panels']):
        col = style.REGIME_COLOR.get(R, '#5d645d')
        g = E(R, 'GT', '')[1:c['kmax'] + 1] / E(R, 'base0', 'mop0.2_0')[1:c['kmax'] + 1]
        a.loglog(kk, g, '-', color=col, lw=1.9, label=f'Re = {R}')
    a.axhline(1, color=style.GT_COLOR, lw=1.6, zorder=5)
    a.axvline(32, color='#5d645d', lw=1, ls='--')
    a.text(33.5, 1.06, 'coarse Nyquist', fontsize=style.LEG_FS - 1.5, color='#5d645d',
           rotation=90, va='bottom')
    a.set_xlabel('wavenumber $k$')
    a.set_ylabel('gain required   $E_{\\mathrm{GT}}(k)\\,/\\,E_{\\mathrm{base}}(k)$')
    a.set_title('what the BASE model is short by, per shell', fontsize=style.TITLE_FS)
    a.legend(fontsize=style.LEG_FS, loc='upper left', title='ground truth / base',
             title_fontsize=style.LEG_FS - 1)

    # ---- one panel per band: required vs what each MODEL delivers ----
    R = np.array(c['allregs'])
    def gain(m, sg, lo, hi):
        return np.array([E(r, m, sg)[lo:hi].sum() / E(r, 'base0', 'mop0.2_0')[lo:hi].sum()
                         for r in R])
    for pi, ((lo, hi), blab) in enumerate(BANDS):
        b = ax[pi + 1]
        req = np.array([E(r, 'GT', '')[lo:hi].sum() / E(r, 'base0', 'mop0.2_0')[lo:hi].sum()
                        for r in R])
        b.plot(R, req, '-', color=C_REQ, lw=2.6, marker='o', ms=4.5,
               label='required to reach truth')
        b.plot(R, gain('base0', 'tapp0.2_3', lo, hi), '-', color=style.MODEL_COLOR['base0'], lw=1.9, marker='^',
               ms=4, label='base + dial')
        b.plot(R, gain('mt1k-0499', 'mop0.2_0', lo, hi), '--', color=style.MODEL_COLOR['mt1k-0499'], lw=1.9, marker='s',
               ms=4, label='fine-tune (weights alone)')
        b.plot(R, gain('mt1k-0499', 'tapp0.2_3', lo, hi), '-', color=style.MODEL_COLOR['mt1k-0499'], lw=1.9, marker='v',
               ms=4, label='fine-tune + dial')
        b.axhline(1, color='#9aa198', lw=1.2)
        style.re_axis(b); b.set_xlabel('Reynolds number')
        if pi == 0: b.set_ylabel('gain over the base model')
        b.set_title(f'aggregated over {blab}', fontsize=style.TITLE_FS)
        b.set_ylim(0.9, float(req.max()) * 1.12)
        if pi == 0: b.legend(fontsize=style.LEG_FS - 0.5, loc='upper left')
    fig.suptitle('The gain required to reach ground truth, and what each model delivers',
                 fontsize=style.SUP_FS)
    for d, ext in ((c['outdir'], 'pdf'), (c['pngdir'], 'png')):
        os.makedirs(d, exist_ok=True)
        p = f'{d}/required_gain.{ext}'; fig.savefig(p); print('  written', p)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--regimes'); a = ap.parse_args()
    c = dict(CONFIG)
    if a.regimes: c['panels'] = [int(x) for x in a.regimes.split(',')]
    make(c)
