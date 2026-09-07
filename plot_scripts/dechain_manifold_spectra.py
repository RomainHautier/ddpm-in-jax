"""Inference configurations, manifold view beside the corresponding spectra.

One row per downward target regime. Left, the single sample endpoints of the chain
config ladder on the spectral manifold (from the dechain trajectory dump). Right, the
pool mean spectrum ratio to ground truth of the same configurations, read from the
audit stores, in matching colors. Configs whose pool rows come from the validation
split are marked (val) in the right panel legend; the manifold endpoints are one median
test sample throughout. The per regime selected chain is black edged on both sides.

  SKIP_COMPUTE=1 JAX_PLATFORMS=cpu python plot_scripts/dechain_manifold_spectra.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import style

M = 'r8kp02-0599'
BASIS_REGS = [1000, 2000, 4000, 8000]
TARGETS = [1000, 2000, 4000, 8000]
SELECTED = {1000: '[100]', 2000: '[125]', 4000: '[160]',
            8000: '[150,100,50]'}
KMAX = 127
SP = ('/tmp/claude-2001/-home-rhautier-ddpm-jax/24ff525a-722f-4c10-b5d9-663bc835ccb1/'
      'scratchpad')
TGT_COL = '#b8399e'   # one fixed color for the target cloud, every row
CCOL = {'[150,100,50]': '#4a4e57', '[150,100]': '#5b8bd0', '[150]': '#2ba58f',
        '[160]': '#8a5cd6', '[125]': '#d4770a', '[120]': '#d4a017', '[100]': '#c2402f'}
# per regime pruning of the ladder, both panels
EXCLUDE = {1000: {'[150,100,50]'}, 2000: {'[120]', '[150]'},
           4000: {'[150,100]'}, 8000: {'[120]', '[150,100]'}}
# manifold ladder: label -> (traj source, prefix cut) in the dechain dump
LADDER = [('[150,100,50]', ('full', 30)), ('[150,100]', ('full', 20)),
          ('[160]', ('K160', None)), ('[150]', ('full', 10)),
          ('[125]', ('K125', None)), ('[120]', ('K120', None)),
          ('[100]', ('K100', None))]
# spectra rows per regime: label -> (chain token, tag)
SPECROWS = {
    1000: {'[150,100,50]': ('K3', 'mop0.2_0'), '[150,100]': ('K150-100', 'fcp0.2_0'),
           '[150]': ('K150', 'fcp0.2_0'), '[100]': ('K100', 'ccp0.2_0')},
    2000: {'[150,100,50]': ('K3', 'mop0.2_0'), '[150]': ('K150', 'clvp0.2_0'),
           '[125]': ('K125', 'cltp0.2_0'), '[120]': ('K120', 'cltp0.2_0')},
    4000: {'[150,100,50]': ('K3', 'mop0.2_0'), '[150,100]': ('K150-100', 'ccp0.2_0'),
           '[150]': ('K150', 'clvp0.2_0'), '[160]': ('K160', 'cltp0.2_0')},
    8000: {'[150,100,50]': ('K3', 'mop0.2_0'), '[150]': ('K150', 'cltp0.2_0'),
           '[100]': ('K100', 'cltp0.2_0')},
}


# the regime's OWN fine tune under its home configuration, as the reference the
# transfer competes with (at 8000 the ladder model IS the in regime model, so skip)
HOME_REF = {1000: 'mt1k-0499', 2000: 'mt2k-0599', 4000: 'r4kp02-0599'}
HOME_COL = '#0f9e78'

# the guided version of the regime's best configuration: chain token, tag, base config
# whose color it borrows, and the display suffix
GUIDED = {1000: ('K100', 'cltp0.2_3', '[100]', '+ dial'),
          2000: ('K125', 'cltp0.2_3', '[125]', '+ dial'),
          4000: ('K160', 'cltp0.2_3', '[160]', '+ dial'),
          8000: ('K3', 'tapp0.2_3', '[150,100,50]', '+ dial')}


def store(R):
    return ('base_results/re1000_audit.npz' if R == 1000
            else f'base_results/regime_audit_re{R}.npz')


def main():
    import jax.numpy as jnp
    from src.rewards import make_spectrum_fn
    style.apply(style.BASE_FS, 150)
    d = json.load(open(f'{SP}/dechain_data.json'))
    clouds = {int(k): np.array(v) for k, v in d['clouds'].items()}
    k = np.arange(1, KMAX + 1)
    sf = make_spectrum_fn(256)

    def spec(x):
        return np.log(np.maximum(np.concatenate(
            [np.asarray(sf(jnp.asarray(x[i:i + 32]))) for i in range(0, len(x), 32)]
        )[:, 1:96], 1e-20))

    gts_full = {R: spec(np.load(f'base_results/fields/re{R}/GT.npz')['x']
                        .astype(np.float32)) for R in BASIS_REGS}
    X = np.concatenate(list(gts_full.values())); mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P = lambda A: (A - mu) @ Vt[:2].T

    fig, axes = plt.subplots(len(TARGETS), 2, figsize=(13.2, 11.6),
                             gridspec_kw=dict(width_ratios=[1.45, 1]))
    for ri, R in enumerate(TARGETS):
        axm, axs = axes[ri]
        t = d['targets'][str(R)]
        # ---- left: manifold endpoints ----
        for Rb in BASIS_REGS:
            if Rb == R: continue
            v = clouds[Rb]
            axm.scatter(v[:, 0], v[:, 1], s=12, color=style.REGIME_COLOR[Rb],
                        alpha=.16, linewidths=0, zorder=2)
        v = clouds[R]
        axm.scatter(v[:, 0], v[:, 1], s=20, color=TGT_COL,
                    alpha=.65, linewidths=0, zorder=3,
                    label='target truth' if ri == 0 else None)
        rc = np.array(t['rc']); gt = np.array(t['gt'])
        pts = [rc[:2]]
        for lab, (src, cut) in reversed(LADDER):
            if src not in t['trajs'] or lab in EXCLUDE.get(R, set()): continue
            tr = np.array(t['trajs'][src])[:cut] if cut else np.array(t['trajs'][src])
            pts.append(tr[-1, :2])
        pts = np.array(pts)
        axm.plot(pts[:, 0], pts[:, 1], ls=':', color='#5d645d', lw=1.1, alpha=.7,
                 zorder=4)
        for lab, (src, cut) in LADDER:
            if src not in t['trajs'] or lab in EXCLUDE.get(R, set()): continue
            tr = np.array(t['trajs'][src])[:cut] if cut else np.array(t['trajs'][src])
            sel = SELECTED[R] == lab
            p_ = tr[-1, :2]
            axm.scatter(*p_, marker='D', s=120 if sel else 85, color=CCOL[lab],
                        zorder=7 if sel else 6,
                        edgecolors='black' if sel else 'white',
                        linewidths=1.2 if sel else .6)
            axm.annotate(lab, p_, textcoords='offset points', xytext=(0, 10),
                         fontsize=7.6, ha='center', color=CCOL[lab],
                         fontweight='bold')
        # the guided version of the selected configuration, same median sample where
        # the pools allow, otherwise that pool's own median sample
        gck, gsg, gbase, gsfx = GUIDED[R]
        gf = f'base_results/fields/re{R}/{M}__{gck}__{gsg}.npz'
        if os.path.exists(gf):
            gx = np.load(gf)['x'].astype(np.float32)
            i0 = t['i']; j = None
            npool = len(np.load(f'base_results/fields/re{R}/GT.npz')['x'])
            if len(gx) == npool:
                j = i0
            else:
                S_ = np.load(store(R), allow_pickle=True)
                sq = np.asarray(S_[f'{R}|EVAL||seq']); msk = sq >= 12
                if msk.sum() == len(gx) and msk[i0]:
                    j = int(msk[:i0].sum())
            if j is None:
                Eg_ = [np.exp(sp)[31:].sum() for sp in spec(gx)]
                j = int(np.argsort(Eg_)[len(Eg_) // 2])
            pg = P(spec(gx[j:j + 1]))[0]
            axm.scatter(*pg, marker='D', s=120, facecolors='none',
                        edgecolors=CCOL[gbase], linewidths=1.8, zorder=8)
            axm.annotate(f'{gbase} {gsfx}', pg, textcoords='offset points',
                         xytext=(0, -16), fontsize=7.6, ha='center',
                         color=CCOL[gbase], fontweight='bold')
        if R in HOME_REF:
            hm = HOME_REF[R]
            hx = np.load(f'base_results/fields/re{R}/{hm}__K3__mop0.2_0.npz')['x'] \
                .astype(np.float32)
            i0 = t['i']; j = None
            npool = len(np.load(f'base_results/fields/re{R}/GT.npz')['x'])
            if len(hx) == npool:
                j = i0
            else:
                S_ = np.load(store(R), allow_pickle=True)
                sq = np.asarray(S_[f'{R}|EVAL||seq']); msk = sq >= 12
                if msk.sum() == len(hx) and msk[i0]:
                    j = int(msk[:i0].sum())
            if j is None:
                Eh_ = [np.exp(sp)[31:].sum() for sp in spec(hx)]
                j = int(np.argsort(Eh_)[len(Eh_) // 2])
            ph = P(spec(hx[j:j + 1]))[0]
            axm.scatter(*ph, marker='P', s=130, color=HOME_COL, zorder=8,
                        edgecolors='white', linewidths=.7)
            axm.annotate('in-regime ft', ph, textcoords='offset points',
                         xytext=(0, -16), fontsize=7.6, ha='center',
                         color=HOME_COL, fontweight='bold')
        axm.scatter(*rc[:2], marker='s', s=70, color='#d4770a', zorder=8,
                    edgecolors='white', linewidths=.6)
        axm.scatter(*gt[:2], marker='*', s=280, color=TGT_COL,
                    zorder=9, edgecolors='black', linewidths=.8)
        axm.set_ylabel('PC2')
        axm.text(.012, .93, f'target Re={R}', transform=axm.transAxes, fontsize=10.5,
                 va='top', fontweight='bold')
        axm.set_ylim(-5.8, 4.9); axm.set_xlim(-32, 15)
        axm.grid(alpha=.25)
        if ri == len(TARGETS) - 1:
            axm.set_xlabel('PC1  (log-spectral shape, Reynolds axis)')
        # ---- right: spectrum ratios of the same configs ----
        S = np.load(store(R), allow_pickle=True); F = set(S.files)
        Eg = np.asarray(S[f'{R}|GT||E'])[1:KMAX + 1]
        style.shade_bands(axs)
        axs.axhline(1, color=style.GT_COLOR, lw=1.3, zorder=6)
        for lab, (ck, sg) in SPECROWS[R].items():
            if lab in EXCLUDE.get(R, set()):
                continue
            key = f'{R}|{M}|{ck}|{sg}||E'
            if key not in F:
                print(f'  missing {key}'); continue
            E = np.asarray(S[key])[1:KMAX + 1]
            sel = SELECTED[R] == lab
            val = 'clv' in sg
            axs.semilogx(k, E / Eg, '-', color=CCOL[lab], lw=1.9,
                         label=lab + (' (val)' if val else ''), zorder=5 if sel else 4)
        if R in HOME_REF:
            hkey = f'{R}|{HOME_REF[R]}|K3|mop0.2_0||E'
            if hkey in F:
                E = np.asarray(S[hkey])[1:KMAX + 1]
                axs.semilogx(k, E / Eg, '-', color=HOME_COL, lw=1.9,
                             label='in-regime ft', zorder=6)
        gkey = f'{R}|{M}|{gck}|{gsg}||E'
        if gkey in F:
            E = np.asarray(S[gkey])[1:KMAX + 1]
            axs.semilogx(k, E / Eg, '--', color=CCOL[gbase], lw=1.9,
                         label=f'{gbase} {gsfx}', zorder=6)
        axs.set(yscale='log', ylim=(0.05, 9), xscale='log')
        axs.set_yticks([0.1, 0.3, 1, 2, 4, 8])
        axs.set_yticklabels(['0.1', '0.3', '1', '2', '4', '8'])
        axs.axvline(96, color='#9aa198', lw=.8, ls=':', zorder=1)
        axs.legend(fontsize=style.LEG_FS - 3, loc='lower left', framealpha=.9)
        axs.set_ylabel('$E(k)\\,/\\,E_{\\mathrm{GT}}(k)$')
        axs.grid(alpha=.25)
        if ri == len(TARGETS) - 1:
            axs.set_xlabel('wavenumber $k$')
    fig.suptitle('The inference configuration ladder, manifold endpoints beside the '
                 'pool spectrum ratios (the Re=8000 specialist)',
                 y=.995)
    fig.tight_layout(rect=(0, 0, 1, .975))
    for d_, ext in (('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')):
        p_ = f'{d_}/dechain_manifold_spectra.{ext}'; fig.savefig(p_); print('written', p_)


if __name__ == '__main__':
    main()
