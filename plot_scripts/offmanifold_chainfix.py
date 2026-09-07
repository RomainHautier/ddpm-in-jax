"""What tuning the inference chain does on the spectral manifold, per downward regime.

Companion to offmanifold_topview.py. One wide panel per regime where the default K3
[150,100,50] chain had to be modified for the Re=8000 specialist to land. Each panel
shows the same r8kp02 model twice on the same start with the real sampler (eta=1,
temp=0.30): the default chain (overshoots) and the regime-tuned chain (lands). Basis is
the canonical four-regime log-spectral PCA; Re=1500/3000 truths are projected into it.

Tuned chains: [100] at Re=1000 (test-confirmed); [120]/[120]/[150] at Re=1500/2000/3000
(validation winners of the 2026-08-31 chain ladder).

  JAX_PLATFORMS=cpu python plot_scripts/offmanifold_chainfix.py            # compute+plot
  SKIP_COMPUTE=1 python plot_scripts/offmanifold_chainfix.py               # replot dump
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
TUNED = {1000: [100], 1500: [120], 2000: [120], 3000: [150]}
DEFAULT = [150, 100, 50]
NSTEP, KLO, KHI = 10, 1, 96
ETA, TEMP = 1.0, 0.30
CKPT = 'monitoring/ddpo_re8000_pdew02_ckpts/ddpo_re1000_iter0599.pkl'
SP = ('/tmp/claude-2001/-home-rhautier-ddpm-jax/24ff525a-722f-4c10-b5d9-663bc835ccb1/'
      'scratchpad')
DUMP = f'{SP}/chainfix_data.json'


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
        trajs = {}
        for name, starts in (('default', DEFAULT), ('tuned', TUNED[R])):
            # same seed per config so the first renoise draw is shared between them
            rng = np.random.default_rng(11)
            x_prev = jnp.asarray(rc[i:i + 1]); tr = []
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
                        sig = ETA * np.sqrt((1 - ab_n) / (1 - ab_c)) \
                            * np.sqrt(1 - ab_c / ab_n)
                        x = (np.sqrt(ab_n) * xh
                             + np.sqrt(max(1 - ab_n - sig ** 2, 0.0)) * e
                             + TEMP * sig * zs[si])
                    else:
                        x = xh
                x_prev = x
            trajs[name] = tr
        out[str(R)] = dict(i=i, rc=P3(spec(rc[i:i + 1]))[0].tolist(),
                           gt=P3(gs[i:i + 1])[0].tolist(), trajs=trajs)
        print(f'  Re={R}: triplet {i} done (default + tuned {TUNED[R]})')
    json.dump({'clouds': {str(R): v for R, v in clouds.items()},
               'targets': out, 'nstep': NSTEP}, open(DUMP, 'w'))
    print('  dumped', DUMP)


def render():
    style.apply()
    d = json.load(open(DUMP))
    clouds = {int(k): np.array(v) for k, v in d['clouds'].items()}
    nstep = d['nstep']
    col = style.MODEL_COLOR['r8kp02-0599']

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
        for name, ls, alpha, face in (('default', '--', .55, 'white'),
                                      ('tuned', '-', .95, col)):
            tr = np.array(t['trajs'][name])
            ax.plot(tr[:, 0], tr[:, 1], ls, color=col, lw=1.6, alpha=alpha, zorder=5)
            ev = tr[::nstep]
            ax.scatter(ev[:, 0], ev[:, 1], s=110, color=col, alpha=alpha, zorder=6,
                       edgecolors='white', linewidths=.8)
            for j, p in enumerate(ev):
                ax.text(p[0], p[1], str(j + 1), color='white', fontsize=7,
                        ha='center', va='center', zorder=7, fontweight='bold')
            ax.scatter(*tr[-1, :2], marker='D', s=60, facecolors=face,
                       edgecolors='black' if name == 'tuned' else col,
                       linewidths=1.1, zorder=7)
        rc = np.array(t['rc']); gt = np.array(t['gt'])
        ax.scatter(*rc[:2], marker='s', s=70, color='#d4770a', zorder=8,
                   edgecolors='white', linewidths=.7)
        ax.scatter(*gt[:2], marker='*', s=320, color=style.vivid(style.REGIME_COLOR[R]),
                   zorder=9, edgecolors='black', linewidths=.9)
        ax.set_ylabel('PC2')
        ch = ','.join(str(s) for s in TUNED[R])
        ax.text(.008, .93, f'target Re={R}   tuned chain [{ch}]',
                transform=ax.transAxes, fontsize=11, va='top', fontweight='bold')
        ax.set_ylim(-5.8, 4.4)
        ax.grid(alpha=.25)
    axes[-1].set_xlabel('PC1  (log-spectral shape, Reynolds axis)')

    handles = [
        plt.Line2D([], [], color=col, ls='--', lw=2, alpha=.6,
                   label='default chain [150,100,50]'),
        plt.Line2D([], [], color=col, lw=2, label='regime-tuned chain'),
        plt.Line2D([], [], ls='none', marker='s', color='#d4770a', ms=8,
                   label='standard recon (start)'),
        plt.Line2D([], [], ls='none', marker='*', color='#777777', ms=13,
                   markeredgecolor='black', label='ground-truth sample'),
        plt.Line2D([], [], ls='none', marker='o', color=col, ms=9,
                   markeredgecolor='white', label='renoising event (numbered)'),
        plt.Line2D([], [], ls='none', marker='D', color=col, ms=7,
                   markeredgecolor='black', label='chain endpoint'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=6, fontsize=9,
               bbox_to_anchor=(.5, .968), framealpha=.92)
    fig.suptitle('Tuning the chain to the regime: the Re=8000 specialist under the '
                 'default and the regime-tuned inference chain', y=.995)
    fig.tight_layout(rect=(0, 0, 1, .945))
    for d_, ext in [('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')]:
        os.makedirs(d_, exist_ok=True)
        p_ = f'{d_}/offmanifold_chainfix.{ext}'
        fig.savefig(p_)
        print('written', p_)


if __name__ == '__main__':
    if not os.environ.get('SKIP_COMPUTE'):
        compute()
    render()
