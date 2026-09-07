"""Three panel per-sample dose figure for Re=1000: where the energy lives, and how well
each configuration restores each triplet's own dose, over the full range and over k>=16.

Panel 1, the ground truth energy proportion per band, motivating the emphasis on a small
fraction of the total. Panels 2 and 3, per-triplet band energy against each triplet's
own truth, full range and k>=16, with the identity line and the +-20 percent tolerance.

  JAX_PLATFORMS=cpu python plot_scripts/dose_breakdown_panel.py
"""
import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'plot_scripts')
import numpy as np, style
import matplotlib.pyplot as plt

GUIDE = os.environ.get('GUIDE', 'tapp0.2_3')
GLAB = {'tapp0.2_3': 'dial', 'v7bandgate': 'gate', 'v8lowband': 'gate',
        'v7ensgate': 'ensemble gate'}.get(GUIDE, 'guide')
SFX = {'tapp0.2_3': '', 'v7ensgate': '_ensgate'}.get(GUIDE, '_gate')
EXTRA = [('gt1k-0199', 'tapp0.2_0', 'gated-dose fine-tune', '#b8399e', 's', 1.0)] \
    if os.environ.get('GUIDE') in ('v7bandgate', 'v8lowband') else []
ROWS = [('base0', 'mop0.2_0', 'base, unguided', '#28658a', 'o', .55),
        ('base0', GUIDE, f'base + {GLAB}', '#28658a', 'o', 1.0),
        ('mt1k-0499', 'mop0.2_0', 'fine-tune, unguided', '#c22f4f', '^', .55),
        ('mt1k-0499', GUIDE, f'fine-tune + {GLAB}', '#c22f4f', '^', 1.0)] + EXTRA
BAND_LABELS = ['$[1,5)$', '$[5,16)$', '$[16,32)$', '$[32,64)$', '$[64,96)$']
TOL = 0.2


def k16_standalone():
    """Only the per-triplet dose panel, as its own figure. KMIN=16 (default) or 32."""
    kmin = int(os.environ.get('KMIN', '16'))
    cols = slice(2, 5) if kmin == 16 else slice(3, 5)
    style.apply(style.BASE_FS, 150)
    S = np.load('base_results/re1000_audit.npz', allow_pickle=True)
    G = np.asarray(S['1000|GT||psEb'])
    fig, a_ = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    T = G[:, cols].sum(1)
    lo, hi = T.min() * .8, T.max() * 1.3
    xs = np.array([lo, hi])
    a_.fill_between(xs, xs * (1 - TOL), xs * (1 + TOL), color='#e6e2d8', zorder=1)
    a_.plot(xs, xs, '-', color=style.GT_COLOR, lw=1.4, zorder=2)
    idx = np.arange(0, len(T), 3)   # every third triplet, the cloud reads cleaner
    K16_ROWS = [
        ('base0',     'mop0.2_0',  'base, unguided',            '#9aa198', 'o'),
        ('base0',     'v8lowband', 'base + gate',               '#28658a', 'o'),
        ('mt1k-0499', 'mop0.2_0',  'fine-tune, unguided',       '#d8a0aa', '^'),
        ('mt1k-0499', 'v8lowband', 'fine-tune + gate',          '#c22f4f', '^'),
        ('gt1k-0199', 'tapp0.2_0', 'gated-dose fine-tune',      '#b8399e', 's'),
        ('base0',     'v7ensgate', 'base + ensemble gate',      '#28658a', 'D'),
        ('mt1k-0499', 'v7ensgate', 'fine-tune + ensemble gate', '#c22f4f', 'v'),
    ]
    for m, sg, lab, col, mk in K16_ROWS:
        key = f'1000|{m}|K3|{sg}||psEb'
        if key not in S.files: continue
        P = np.asarray(S[key])[:, cols].sum(1)
        a_.scatter(T[idx], P[idx], s=26, marker=mk, facecolors='none', edgecolors=col,
                   lw=1.1, zorder=3, label=lab)
    a_.set_xscale('log'); a_.set_yscale('log')
    a_.set_xlim(lo, hi); a_.set_ylim(lo * .5, hi * 1.2)
    a_.set_xlabel("triplet's own GT band energy")
    a_.set_ylabel('reconstructed band energy')
    a_.set_title(f'per-triplet dose, $k\\geq{kmin}$, Re=1000', fontsize=style.TITLE_FS)
    a_.legend(fontsize=style.LEG_FS - 1.5, loc='upper left', framealpha=.9)
    for d_, ext in (('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')):
        p_ = f'{d_}/re1000_dose_k{kmin}{SFX}_pertriplet.{ext}'
        fig.savefig(p_); print('written', p_)


def main():
    style.apply(style.BASE_FS, 150)
    S = np.load('base_results/re1000_audit.npz', allow_pickle=True)
    G = np.asarray(S['1000|GT||psEb'])
    fig, ax = plt.subplots(1, 3, figsize=(14.2, 4.6), constrained_layout=True)

    # ---- panel 1: where the energy lives ----
    share = G.sum(0) / G.sum() * 100
    ax[0].bar(np.arange(5), share, .7, color=[style.REGIME_COLOR[1000]] * 2 + ['#d4a017'] * 3,
              zorder=3)
    for i, v in enumerate(share):
        ax[0].text(i, v * 1.15, f'{v:.2f}%' if v < 1 else f'{v:.1f}%',
                   ha='center', fontsize=style.LEG_FS)
    ax[0].set_yscale('log'); ax[0].set_ylim(0.05, 300)
    ax[0].set_xticks(np.arange(5)); ax[0].set_xticklabels(BAND_LABELS)
    ax[0].set_xlabel('wavenumber band'); ax[0].set_ylabel('share of total energy (%)')
    ax[0].set_title('energy proportion per band (GT)', fontsize=style.TITLE_FS)
    ax[0].text(3, 120, f'$k\\geq16$: {share[2:].sum():.2f}%', fontsize=style.LEG_FS + 1,
               ha='center', color='#8c6d1f', fontweight='bold')
    ax[0].grid(axis='x', visible=False)

    # ---- panels 2 and 3: per-triplet dose, full range and k>=16 ----
    for pi, (cols, ttl) in enumerate([(slice(0, 5), 'per-triplet dose, full $k$ range'),
                                      (slice(2, 5), 'per-triplet dose, $k\\geq16$')]):
        a_ = ax[pi + 1]
        T = G[:, cols].sum(1)
        lo, hi = T.min() * .8, T.max() * 1.3
        xs = np.array([lo, hi])
        a_.fill_between(xs, xs * (1 - TOL), xs * (1 + TOL), color='#e6e2d8', zorder=1)
        a_.plot(xs, xs, '-', color=style.GT_COLOR, lw=1.4, zorder=2)
        for m, sg, lab, col, mk, al in ROWS:
            key = f'1000|{m}|K3|{sg}||psEb'
            if key not in S.files: continue
            P = np.asarray(S[key])[:, cols].sum(1)
            sl = np.polyfit(np.log(T), np.log(P), 1)[0]
            ib = np.mean(np.abs(P / T - 1) < TOL) * 100
            a_.scatter(T, P, s=15, marker=mk, color=col, alpha=al, lw=0, zorder=3,
                       label=f'{lab}  ({ib:.0f}% in, slope {sl:.2f})')
        a_.set_xscale('log'); a_.set_yscale('log')
        a_.set_xlim(lo, hi); a_.set_ylim(lo * .5, hi * 1.2)
        a_.set_xlabel("triplet's own GT band energy")
        if pi == 0: a_.set_ylabel('reconstructed band energy')
        a_.set_title(ttl, fontsize=style.TITLE_FS)
        a_.legend(fontsize=style.LEG_FS - 2.5, loc='upper left', framealpha=.9)
    fig.suptitle('Per-sample energy dose at Re=1000, and where that energy lives',
                 fontsize=style.SUP_FS)
    for d_, ext in (('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')):
        p_ = f'{d_}/re1000_dose_breakdown{SFX}.{ext}'; fig.savefig(p_); print('written', p_)


if __name__ == '__main__':
    k16_standalone() if os.environ.get('ONLY_K16') else main()
