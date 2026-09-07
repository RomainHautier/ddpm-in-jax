"""How close can inference adaptation get? One sample per regime, the Re=8000 specialist
under the configurations that matter at that regime, on the manifold.

Below the home regime the panel shows the chain family and the dialed version of the
selected chain; the default K3 appears only at and above Re=6000, where full depth is
legitimately the right dose. All markers are read from the stored evaluation fields, so
each is the actual reconstruction that entered the tables; labels carry the POOL
retention and the closest-to-1 configuration is black-edged.

  JAX_PLATFORMS=cpu python plot_scripts/offmanifold_bestconfig.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import style

BASIS_REGS = [1000, 2000, 4000, 8000]
M = 'r8kp02-0599'
# per target regime: (display label, fields token) in ladder order
PANEL = {
    1000: [('[50]', 'K50__fcp0.2_0', '#5b8bd0'),
           ('[100]', 'K100__ccp0.2_0', '#c2402f'),
           ('[100] + dial', 'K100__cltp0.2_3', '#8a5cd6')],
    2000: [('[120]', 'K120__cltp0.2_0', '#d4a017'),
           ('[125]', 'K125__cltp0.2_0', '#c2402f'),
           ('[125] + dial', 'K125__cltp0.2_3', '#8a5cd6')],
    4000: [('[150,100]', 'K150-100__ccp0.2_0', '#5b8bd0'),
           ('[160]', 'K160__cltp0.2_0', '#c2402f'),
           ('[160] + dial', 'K160__cltp0.2_3', '#8a5cd6')],
    6000: [('K3', 'K3__mop0.2_0', '#4a4e57'),
           ('K3 + dial', 'K3__tapp0.2_3', '#5b8bd0'),
           ('K3 + gate', 'K3__v7bandgate', '#0f9e78')],
    8000: [('K3', 'K3__mop0.2_0', '#4a4e57'),
           ('K3 + dial', 'K3__tapp0.2_3', '#5b8bd0'),
           ('K3 + gate', 'K3__v7bandgate', '#0f9e78')],
}


def pool_ret(R, tok):
    S = np.load('base_results/re1000_audit.npz' if R == 1000
                else f'base_results/regime_audit_re{R}.npz', allow_pickle=True)
    ck, sg = tok.split('__', 1)
    k = f'{R}|{M}|{ck}|{sg}||ret'
    return float(S[k]) if k in S.files else None


def main():
    import jax.numpy as jnp
    from src.rewards import make_spectrum_fn
    style.apply(style.BASE_FS, 150)
    sf = make_spectrum_fn(256)

    def spec(x):
        return np.log(np.maximum(np.concatenate(
            [np.asarray(sf(jnp.asarray(x[i:i + 32]))) for i in range(0, len(x), 32)]
        )[:, 1:96], 1e-20))

    gts = {R: spec(np.load(f'base_results/fields/re{R}/GT.npz')['x'].astype(np.float32))
           for R in BASIS_REGS}
    X = np.concatenate(list(gts.values())); mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P = lambda A: (A - mu) @ Vt[:2].T
    tgt = dict(gts)
    for R in PANEL:
        if R not in tgt:
            tgt[R] = spec(np.load(f'base_results/fields/re{R}/GT.npz')['x']
                          .astype(np.float32))

    fig, axes = plt.subplots(len(PANEL), 1, figsize=(12.5, 12.6),
                             sharex=True, sharey=True)
    for ax, (R, cfgs) in zip(axes, PANEL.items()):
        for Rb in BASIS_REGS:
            if Rb == R: continue
            v = P(gts[Rb])
            ax.scatter(v[:, 0], v[:, 1], s=14, color=style.REGIME_COLOR[Rb],
                       alpha=.18, linewidths=0, zorder=2)
        g = tgt[R]; v = P(g)
        ax.scatter(v[:, 0], v[:, 1], s=24, color=style.vivid(style.REGIME_COLOR[R]),
                   alpha=.6, linewidths=0, zorder=3)
        base_f = np.load(f'base_results/fields/re{R}/{M}__K3__mop0.2_0.npz')['x'] \
            .astype(np.float32)
        n = len(base_f)
        if len(g) == n:
            gtm = g
        else:
            S_ = np.load(f'base_results/regime_audit_re{R}.npz', allow_pickle=True)
            sq = np.asarray(S_[f'{R}|EVAL||seq']); msk = sq >= 12
            gtm = g[msk] if msk.sum() == n else None
        E = np.asarray([np.exp(sp)[31:].sum() for sp in (gtm if gtm is not None
                                                          else spec(base_f))])
        i = int(np.argsort(E)[len(E) // 2])
        rets = {}
        for lab, tok, col in cfgs:
            f = f'base_results/fields/re{R}/{M}__{tok}.npz'
            if not os.path.exists(f):
                print(f'  MISSING {f}'); continue
            x = np.load(f)['x'].astype(np.float32)
            j = i if len(x) == n else int(np.argsort(
                [np.exp(sp)[31:].sum() for sp in spec(x)])[len(x) // 2])
            rets[lab] = (pool_ret(R, tok), P(spec(x[j:j + 1]))[0], col)
        best = min((l for l in rets if rets[l][0] is not None),
                   key=lambda l: abs(1 - rets[l][0]))
        for lab, (ret, p_, col) in rets.items():
            sel = lab == best
            ax.scatter(*p_, marker='D', s=150 if sel else 110, color=col,
                       zorder=8 if sel else 7,
                       edgecolors='black' if sel else 'white',
                       linewidths=1.3 if sel else .7)
            rtxt = f' {ret:.2f}' if ret is not None else ''
            ax.annotate(lab + rtxt, p_, textcoords='offset points', xytext=(0, 12),
                        fontsize=8.4, ha='center', color=col, fontweight='bold')
        if gtm is not None:
            ax.scatter(*P(gtm[i:i + 1])[0], marker='*', s=340,
                       color=style.vivid(style.REGIME_COLOR[R]), zorder=9,
                       edgecolors='black', linewidths=.9)
        ax.set_ylabel('PC2')
        ax.text(.008, .93, f'target Re={R}   best: {best} '
                f'(pool ret {rets[best][0]:.2f})',
                transform=ax.transAxes, fontsize=11, va='top', fontweight='bold')
        ax.set_ylim(-5.8, 5.2)
        ax.grid(alpha=.25)
        print(f'Re={R}: ' + '  '.join(f'{l}={v[0]:.2f}' for l, v in rets.items()
                                      if v[0] is not None) + f'  -> best {best}')
    axes[-1].set_xlabel('PC1  (log-spectral shape, Reynolds axis)')
    handles = [
        plt.Line2D([], [], ls='none', marker='D', color='#c2402f', ms=8,
                   label='selected chain'),
        plt.Line2D([], [], ls='none', marker='D', color='#8a5cd6', ms=8,
                   label='selected chain + dial'),
        plt.Line2D([], [], ls='none', marker='D', color='#0f9e78', ms=8,
                   label='+ gate'),
        plt.Line2D([], [], ls='none', marker='*', color='#777777', ms=13,
                   markeredgecolor='black', label='ground-truth sample'),
        plt.Line2D([], [], ls='none', marker='D', color='#9aa198', ms=8,
                   markeredgecolor='black', label='best config (pool ret)'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=5, fontsize=8.8,
               bbox_to_anchor=(.5, .975), framealpha=.92)
    fig.suptitle('The regime-appropriate inference family, the Re=8000 specialist '
                 '(labels carry pool retention)', y=.995)
    fig.tight_layout(rect=(0, 0, 1, .955))
    for d_, ext in (('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')):
        p_ = f'{d_}/offmanifold_bestconfig.{ext}'; fig.savefig(p_); print('written', p_)


if __name__ == '__main__':
    main()
