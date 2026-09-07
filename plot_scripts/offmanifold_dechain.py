"""From the default chain to a clean transfer: drop passes, then reduce the renoise depth.

One wide panel per downward target regime. The Re=8000 specialist runs on the same
median-energy sample with the real sampler (eta=1, temp=0.30) under a ladder of inference
configs: the default [150,100,50], the same chain with the last pass removed [150,100],
one pass only [150], then the renoise depth reduced to [120] and [100]. The same rng seed
makes the shorter-chain trajectories exact prefixes of the default one, so "taking out a
chain" is literally stopping earlier; the reduced-depth chains start from a shallower
renoise of the same reconstruction. The per-regime selected chain is the black-edged
endpoint.

  JAX_PLATFORMS=cpu python plot_scripts/offmanifold_dechain.py
  SKIP_COMPUTE=1 python plot_scripts/offmanifold_dechain.py
"""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, 'src/ddpo_ft')
os.environ.setdefault('BASE_CKPT', '/tmp/ema_ckpts/ema_base_0299.pkl')
import matplotlib.pyplot as plt
import numpy as np
import style

BASIS_REGS = [1000, 2000, 4000, 8000]
TARGETS = [1000, 1500, 2000, 3000]
SELECTED = {1000: '[100]', 1500: '[120]', 2000: '[125]', 3000: '[150]'}
NSTEP, KLO, KHI = 10, 1, 96
ETA, TEMP = 1.0, 0.30
CKPT = 'monitoring/ddpo_re8000_pdew02_ckpts/ddpo_re1000_iter0599.pkl'
SP = ('/tmp/claude-2001/-home-rhautier-ddpm-jax/24ff525a-722f-4c10-b5d9-663bc835ccb1/'
      'scratchpad')
DUMP = f'{SP}/dechain_data.json'
# (label, color, computed chain or prefix of 'full')
LADDER = [('[150,100,50]', '#4a4e57', ('full', 30)),
          ('[150,100]',    '#5b8bd0', ('full', 20)),
          ('[150]',        '#2ba58f', ('full', 10)),
          ('[125]',        '#d4770a', ('K125', None)),   # only computed at Re=2000
          ('[120]',        '#d4a017', ('K120', None)),
          ('[100]',        '#c2402f', ('K100', None))]


def run_chain(ddpm, params, ab, spec, P3, rc1, starts):
    import jax.numpy as jnp
    rng = np.random.default_rng(11)
    x_prev = jnp.asarray(rc1); tr = []
    for S in starts:
        noise = jnp.asarray(rng.normal(size=(1, 256, 256, 3)).astype(np.float32))
        zs = [jnp.asarray(rng.normal(size=(1, 256, 256, 3)).astype(np.float32))
              for _ in range(NSTEP)]
        ts = np.linspace(S, 0, NSTEP + 1).astype(int)
        x = float(np.sqrt(ab[S])) * x_prev + float(np.sqrt(1 - ab[S])) * noise
        for si, (a, b) in enumerate(zip(ts[:-1], ts[1:])):
            ta = jnp.full((1,), int(a), jnp.int32)
            e = ddpm.unet.apply({'params': params}, x, ta, train=False)
            xh = (x - float(np.sqrt(1 - ab[a])) * e) / float(np.sqrt(ab[a]))
            tr.append(P3(spec(np.asarray(xh)))[0].tolist())
            if b > 0:
                ab_c, ab_n = float(ab[a]), float(ab[b])
                sig = ETA * np.sqrt((1 - ab_n) / (1 - ab_c)) * np.sqrt(1 - ab_c / ab_n)
                x = (np.sqrt(ab_n) * xh + np.sqrt(max(1 - ab_n - sig ** 2, 0.0)) * e
                     + TEMP * sig * zs[si])
            else:
                x = xh
        x_prev = x
    return tr


def compute():
    import jax.numpy as jnp
    from train_claude import build_base_ddpm
    from src.rewards import make_spectrum_fn
    ddpm, _, _ = build_base_ddpm(); ab = ddpm.alpha_bar
    sf = make_spectrum_fn(256)

    def spec(x):
        return np.log(np.maximum(np.concatenate(
            [np.asarray(sf(jnp.asarray(x[i:i + 32]))) for i in range(0, len(x), 32)]
        )[:, KLO:KHI], 1e-20))

    d = pickle.load(open(CKPT, 'rb'))
    params = d.get('ema_params') if d.get('ema_params') is not None else d['params']
    gts = {R: spec(np.load(f'base_results/fields/re{R}/GT.npz')['x'].astype(np.float32))
           for R in BASIS_REGS}
    X = np.concatenate(list(gts.values())); mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P3 = lambda A: (A - mu) @ Vt[:3].T
    clouds = {R: P3(g).tolist() for R, g in gts.items()}
    for R in TARGETS:
        if R not in clouds:
            g = spec(np.load(f'base_results/fields/re{R}/GT.npz')['x'].astype(np.float32))
            clouds[R] = P3(g).tolist()

    out = {}
    for R in TARGETS:
        gt = np.load(f'base_results/fields/re{R}/GT.npz')['x'].astype(np.float32)
        rc = np.load(f'base_results/fields/re{R}/recon.npz')['x'].astype(np.float32)
        gs = spec(gt)
        E = np.asarray([np.exp(sp)[31:].sum() for sp in gs])
        i = int(np.argsort(E)[len(E) // 2])
        trajs = {'full': run_chain(ddpm, params, ab, spec, P3, rc[i:i + 1], [150, 100, 50]),
                 'K120': run_chain(ddpm, params, ab, spec, P3, rc[i:i + 1], [120]),
                 'K100': run_chain(ddpm, params, ab, spec, P3, rc[i:i + 1], [100])}
        out[str(R)] = dict(i=i, rc=P3(spec(rc[i:i + 1]))[0].tolist(),
                           gt=P3(gs[i:i + 1])[0].tolist(), trajs=trajs)
        print(f'  Re={R}: triplet {i} done')
    json.dump({'clouds': {str(R): v for R, v in clouds.items()},
               'targets': out, 'nstep': NSTEP}, open(DUMP, 'w'))
    print('  dumped', DUMP)


def render():
    style.apply()
    d = json.load(open(DUMP))
    clouds = {int(k): np.array(v) for k, v in d['clouds'].items()}
    nstep = d['nstep']

    fig, axes = plt.subplots(len(TARGETS), 1, figsize=(12.5, 10.4),
                             sharex=True, sharey=True)
    for ax, R in zip(axes, TARGETS):
        t = d['targets'][str(R)]
        for Rb in BASIS_REGS:
            if Rb == R:
                continue
            v = clouds[Rb]
            ax.scatter(v[:, 0], v[:, 1], s=14, color=style.REGIME_COLOR[Rb],
                       alpha=.22, linewidths=0, zorder=2)
        v = clouds[R]
        ax.scatter(v[:, 0], v[:, 1], s=24, color=style.vivid(style.REGIME_COLOR[R]),
                   alpha=.75, linewidths=0, zorder=3)
        for lab, col, (src, cut) in LADDER:
            if src not in t['trajs']:
                continue
            tr = np.array(t['trajs'][src])[:cut] if cut else np.array(t['trajs'][src])
            sel = SELECTED[R] == lab
            ax.plot(tr[:, 0], tr[:, 1], '-' if src != 'full' or cut == 30 else '-',
                    color=col, lw=2.2 if sel else 1.4,
                    alpha=.95 if sel else .6, zorder=6 if sel else 5)
            ax.scatter(*tr[-1, :2], marker='D', s=85 if sel else 55, color=col,
                       zorder=8 if sel else 7,
                       edgecolors='black' if sel else 'white',
                       linewidths=1.2 if sel else .7)
        # renoising events, marked once on the full chain
        ev = np.array(t['trajs']['full'])[::nstep]
        ax.scatter(ev[:, 0], ev[:, 1], s=100, color='#4a4e57', zorder=6,
                   edgecolors='white', linewidths=.8, alpha=.85)
        for j, p in enumerate(ev):
            ax.text(p[0], p[1], str(j + 1), color='white', fontsize=7,
                    ha='center', va='center', zorder=7, fontweight='bold')
        rc = np.array(t['rc']); gt = np.array(t['gt'])
        ax.scatter(*rc[:2], marker='s', s=70, color='#d4770a', zorder=8,
                   edgecolors='white', linewidths=.7)
        ax.scatter(*gt[:2], marker='*', s=320, color=style.vivid(style.REGIME_COLOR[R]),
                   zorder=9, edgecolors='black', linewidths=.9)
        ax.set_ylabel('PC2')
        ax.text(.008, .93, f'target Re={R}   selected chain {SELECTED[R]}',
                transform=ax.transAxes, fontsize=11, va='top', fontweight='bold')
        ax.set_ylim(-5.8, 4.4)
        ax.grid(alpha=.25)
    axes[-1].set_xlabel('PC1  (log-spectral shape, Reynolds axis)')

    handles = [plt.Line2D([], [], color=c, lw=2, label=l) for l, c, _ in LADDER]
    handles += [
        plt.Line2D([], [], ls='none', marker='s', color='#d4770a', ms=8,
                   label='standard recon (start)'),
        plt.Line2D([], [], ls='none', marker='*', color='#777777', ms=13,
                   markeredgecolor='black', label='ground-truth sample'),
        plt.Line2D([], [], ls='none', marker='D', color='#777777', ms=7,
                   markeredgecolor='black', label='selected endpoint'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=8, fontsize=8.7,
               bbox_to_anchor=(.5, .968), framealpha=.92)
    fig.suptitle('Dropping passes, then reducing the renoise depth: the Re=8000 '
                 'specialist stepped back onto each downward regime', y=.995)
    fig.tight_layout(rect=(0, 0, 1, .945))
    for d_, ext in [('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')]:
        os.makedirs(d_, exist_ok=True)
        p_ = f'{d_}/offmanifold_dechain.{ext}'
        fig.savefig(p_)
        print('written', p_)


def render_endpoints():
    style.apply()
    d = json.load(open(DUMP))
    clouds = {int(k): np.array(v) for k, v in d['clouds'].items()}

    fig, axes = plt.subplots(len(TARGETS), 1, figsize=(12.5, 10.4),
                             sharex=True, sharey=True)
    for ax, R in zip(axes, TARGETS):
        t = d['targets'][str(R)]
        for Rb in BASIS_REGS:
            if Rb == R:
                continue
            v = clouds[Rb]
            ax.scatter(v[:, 0], v[:, 1], s=14, color=style.REGIME_COLOR[Rb],
                       alpha=.18, linewidths=0, zorder=2)
        v = clouds[R]
        ax.scatter(v[:, 0], v[:, 1], s=24, color=style.vivid(style.REGIME_COLOR[R]),
                   alpha=.6, linewidths=0, zorder=3)
        rc = np.array(t['rc']); gt = np.array(t['gt'])
        pts = [rc[:2]]
        for lab, col, (src, cut) in reversed(LADDER):
            if src not in t['trajs']:
                continue
            tr = np.array(t['trajs'][src])[:cut] if cut else np.array(t['trajs'][src])
            pts.append(tr[-1, :2])
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], ls=':', color='#5d645d', lw=1.2, alpha=.7,
                zorder=4)
        for lab, col, (src, cut) in LADDER:
            if src not in t['trajs']:
                continue
            tr = np.array(t['trajs'][src])[:cut] if cut else np.array(t['trajs'][src])
            sel = SELECTED[R] == lab
            p_ = tr[-1, :2]
            ax.scatter(*p_, marker='D', s=150 if sel else 110, color=col,
                       zorder=7 if sel else 6,
                       edgecolors='black' if sel else 'white',
                       linewidths=1.3 if sel else .7)
            ax.annotate(lab, p_, textcoords='offset points', xytext=(0, 12),
                        fontsize=8.2, ha='center', color=col, fontweight='bold')
        ax.scatter(*rc[:2], marker='s', s=90, color='#d4770a', zorder=8,
                   edgecolors='white', linewidths=.7)
        ax.annotate('start (recon)', rc[:2], textcoords='offset points',
                    xytext=(0, -16), fontsize=8.2, ha='center', color='#d4770a')
        ax.scatter(*gt[:2], marker='*', s=340, color=style.vivid(style.REGIME_COLOR[R]),
                   zorder=9, edgecolors='black', linewidths=.9)
        ax.set_ylabel('PC2')
        ax.text(.008, .93, f'target Re={R}   selected chain {SELECTED[R]}',
                transform=ax.transAxes, fontsize=11, va='top', fontweight='bold')
        ax.set_ylim(-5.8, 4.9)
        ax.grid(alpha=.25)
    axes[-1].set_xlabel('PC1  (log-spectral shape, Reynolds axis)')
    handles = [plt.Line2D([], [], ls='none', marker='D', color=c, ms=8, label=l)
               for l, c, _ in LADDER]
    handles += [
        plt.Line2D([], [], ls='none', marker='s', color='#d4770a', ms=8,
                   label='standard recon (start)'),
        plt.Line2D([], [], ls='none', marker='*', color='#777777', ms=13,
                   markeredgecolor='black', label='ground-truth sample'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=7, fontsize=8.7,
               bbox_to_anchor=(.5, .968), framealpha=.92)
    fig.suptitle('One sample per regime: where each inference chain leaves it '
                 '(the Re=8000 specialist)', y=.995)
    fig.tight_layout(rect=(0, 0, 1, .945))
    for d_, ext in [('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')]:
        p_ = f'{d_}/offmanifold_dechain_endpoints.{ext}'
        fig.savefig(p_)
        print('written', p_)


if __name__ == '__main__':
    if not os.environ.get('SKIP_COMPUTE'):
        compute()
    if os.environ.get('ENDPOINTS'):
        render_endpoints()
    else:
        render()
