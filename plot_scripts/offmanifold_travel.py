"""Does re-noising provide the off-manifold travel, and does the required travel grow with Re?

Shared log-spectral PCA basis over the ground truths of Re=1000/2000/4000/8000. One test sample
per regime; each model runs the full K3 chain on it (deterministic DDIM within passes, shared
noise at each re-noise event), and every clean estimate is projected onto PC1. The target
regime's truth band is shaded; the LR input and the base reconstruction mark where the journey
starts.

  JAX_PLATFORMS=cpu python plot_scripts/offmanifold_travel.py
"""
import os, sys, argparse, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, 'src/ddpo_ft')
os.environ.setdefault('BASE_CKPT', '/tmp/ema_ckpts/ema_base_0299.pkl')
import numpy as np
import matplotlib.pyplot as plt
import style

REGS = [1000, 2000, 4000, 8000]
STARTS, NSTEP = [150, 100, 50], 10
KLO, KHI = 1, 96
MODELS = [('base', 'base', None),
          ('mt1k-0499', 'Re=1000 ft', 'monitoring/ddpo_re1000_match_ckpts/ddpo_re1000_iter0499.pkl'),
          ('mt2k-0599', 'Re=2000 ft', 'monitoring/ddpo_re2000_match_ckpts/ddpo_re1000_iter0599.pkl'),
          ('r4kp02-0599', 'Re=4000 ft', 'monitoring/ddpo_re4000_pdew02_ckpts/ddpo_re1000_iter0599.pkl'),
          ('r8kp02-0599', 'Re=8000 ft', 'monitoring/ddpo_re8000_pdew02_ckpts/ddpo_re1000_iter0599.pkl')]


def params_of(d): return d.get('ema_params') if d.get('ema_params') is not None else d['params']


def main():
    import jax.numpy as jnp
    from train_claude import build_base_ddpm
    from src.rewards import make_spectrum_fn
    style.apply(style.BASE_FS, 150)
    ddpm, bp, _ = build_base_ddpm(); ab = ddpm.alpha_bar
    sf = make_spectrum_fn(256)
    def spec(x):
        return np.log(np.maximum(np.concatenate(
            [np.asarray(sf(jnp.asarray(x[i:i+32]))) for i in range(0, len(x), 32)])[:, KLO:KHI], 1e-20))
    params = {n: (bp if p is None else params_of(pickle.load(open(p, 'rb')))) for n, _, p in MODELS}
    gts = {R: spec(np.load(f'base_results/fields/re{R}/GT.npz')['x'].astype(np.float32)) for R in REGS}
    X = np.concatenate(list(gts.values())); mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P3 = lambda A: (A - mu) @ Vt[:3].T

    var = None
    # ---- compute all trajectories once, with the REAL sampler (eta=1, temp=0.30) ----
    ETA, TEMP = 1.0, 0.30
    rng = np.random.default_rng(11)
    data = {}
    for R in REGS:
        gt = np.load(f'base_results/fields/re{R}/GT.npz')['x'].astype(np.float32)
        lr = np.load(f'base_results/fields/re{R}/LR.npz')['x'].astype(np.float32)
        rc = np.load(f'base_results/fields/re{R}/recon.npz')['x'].astype(np.float32)
        E = np.asarray([np.exp(sp)[31:].sum() for sp in spec(gt)])
        i = int(np.argsort(E)[len(E) // 2])
        # noise shared ACROSS MODELS: one renoise draw per pass, one z per step
        noises = [jnp.asarray(rng.normal(size=(1, 256, 256, 3)).astype(np.float32)) for _ in STARTS]
        zs = [[jnp.asarray(rng.normal(size=(1, 256, 256, 3)).astype(np.float32))
               for _ in range(NSTEP)] for _ in STARTS]
        trajs = {}
        for n, lab, _ in MODELS:
            x_prev = jnp.asarray(rc[i:i+1]); tr = []
            for ci, S in enumerate(STARTS):
                ts = np.linspace(S, 0, NSTEP + 1).astype(int)
                x = float(np.sqrt(ab[S])) * x_prev + float(np.sqrt(1 - ab[S])) * noises[ci]
                for si, (a, b) in enumerate(zip(ts[:-1], ts[1:])):
                    ta = jnp.full((1,), int(a), jnp.int32)
                    e = ddpm.unet.apply({'params': params[n]}, x, ta, train=False)
                    xh = (x - float(np.sqrt(1 - ab[a])) * e) / float(np.sqrt(ab[a]))
                    tr.append(P3(spec(np.asarray(xh)))[0])
                    if b > 0:
                        ab_c, ab_n = float(ab[a]), float(ab[b])
                        sig = ETA * np.sqrt((1 - ab_n) / (1 - ab_c)) * np.sqrt(1 - ab_c / ab_n)
                        x = (np.sqrt(ab_n) * xh + np.sqrt(max(1 - ab_n - sig ** 2, 0.0)) * e
                             + TEMP * sig * zs[ci][si])
                    else:
                        x = xh
                x_prev = x
            trajs[n] = np.array(tr)
        data[R] = dict(i=i, lr=P3(spec(lr[i:i+1]))[0], rc=P3(spec(rc[i:i+1]))[0],
                       gt=P3(spec(gt[i:i+1]))[0], trajs=trajs)
        print(f'  Re={R}: triplet {i} done (eta=1, temp={TEMP})')
    # dump everything the interactive 3D view needs
    import json
    SP = '/tmp/claude-2001/-home-rhautier-ddpm-jax/24ff525a-722f-4c10-b5d9-663bc835ccb1/scratchpad'
    dump = {'clouds': {str(R): P3(g).tolist() for R, g in gts.items()},
            'targets': {str(R): {'gt': data[R]['gt'].tolist(), 'rc': data[R]['rc'].tolist(),
                                 'lr': data[R]['lr'].tolist(),
                                 'trajs': {n: data[R]['trajs'][n].tolist() for n in data[R]['trajs']}}
                        for R in REGS},
            'nstep': NSTEP}
    json.dump(dump, open(f'{SP}/offmanifold_data.json', 'w'))
    print('  dumped offmanifold_data.json')

    # ---- 2D: one panel per target regime, PC1 x PC2, zoomed to the journey ----
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.2), constrained_layout=True)
    for pi, R in enumerate(REGS):
        ax = axes[pi // 2][pi % 2]; d = data[R]
        for Rb, g in gts.items():
            v = P3(g)
            ax.scatter(v[:, 0], v[:, 1], s=16, color=style.REGIME_COLOR[Rb],
                       alpha=.75 if Rb == R else .28, lw=0,
                       label=f'Re={Rb} truth' if (pi == 0) else None)
        ax.scatter(*d['lr'][:2], marker='v', s=70, color='#b8399e', zorder=6,
                   label='LR input' if pi == 0 else None)
        ax.scatter(*d['rc'][:2], marker='s', s=55, color='#d4770a', zorder=6,
                   label='base recon (start)' if pi == 0 else None)
        ax.scatter(*d['gt'][:2], marker='*', s=260, color=style.REGIME_COLOR[R],
                   edgecolors='black', linewidths=.9, zorder=7,
                   label="this sample's ground truth" if pi == 0 else None)
        for n, lab, _ in MODELS:
            t = d['trajs'][n]; col = style.MODEL_COLOR.get(n, '#232823')
            ax.plot(t[:, 0], t[:, 1], '-', color=col, lw=1.9,
                    label=lab if pi == 0 else None)
            for ci in range(len(STARTS)):                    # numbered dot at each pass start
                ax.scatter(*t[ci * NSTEP, :2], s=30, color=col, zorder=5)
                ax.annotate(str(ci + 1), t[ci * NSTEP, :2], textcoords='offset points',
                            xytext=(4, 4), fontsize=style.LEG_FS - 2, color=col,
                            fontweight='bold')
            ax.annotate('', xy=t[-1, :2], xytext=t[-5, :2],
                        arrowprops=dict(arrowstyle='-|>', color=col, lw=1.6))
        xs = np.concatenate([P3(gts[R])[:, 0], [d['rc'][0], d['lr'][0]]]
                            + [d['trajs'][n][:, 0] for n in d['trajs']])
        ys = np.concatenate([P3(gts[R])[:, 1], [d['rc'][1], d['lr'][1]]]
                            + [d['trajs'][n][:, 1] for n in d['trajs']])
        ax.set_xlim(xs.min() - 2, xs.max() + 2); ax.set_ylim(ys.min() - .8, ys.max() + .8)
        ax.set_title(f'target regime Re={R}', fontsize=style.TITLE_FS)
        if pi // 2: ax.set_xlabel('PC1 (Reynolds axis)')
        if pi % 2 == 0: ax.set_ylabel('PC2')
    axes[0][0].legend(fontsize=style.LEG_FS - 2.5, loc='lower right', ncol=2)
    fig.suptitle('The journey from the reconstruction to each regime, on the shared spectral '
                 'manifold (dots mark re-noising events)', fontsize=style.SUP_FS)
    for d_, ext in (('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')):
        p_ = f'{d_}/offmanifold_travel.{ext}'; fig.savefig(p_); print('  written', p_)

    # ---- 3D: the two extreme targets in one view ----
    fig2 = plt.figure(figsize=(11.5, 7.8), constrained_layout=True)
    ax3 = fig2.add_subplot(projection='3d')
    for Rb, g in gts.items():
        v = P3(g)
        ax3.scatter(v[:, 0], v[:, 1], v[:, 2], s=7, color=style.REGIME_COLOR[Rb], alpha=.25,
                    lw=0, label=f'Re={Rb} truth')
    for R, mk in ((1000, 'o'), (8000, '^')):
        d = data[R]
        ax3.scatter(*d['rc'], marker='s', s=60, color='#d4770a')
        for n in ('base', 'r8kp02-0599', 'mt1k-0499'):
            t = d['trajs'][n]
            ax3.plot(t[:, 0], t[:, 1], t[:, 2], '-', lw=2.2,
                     color=style.MODEL_COLOR.get(n, '#232823'))
    ax3.set_xlabel('PC1'); ax3.set_ylabel('PC2'); ax3.set_zlabel('PC3')
    ax3.legend(fontsize=style.LEG_FS - 1, loc='upper left')
    ax3.set_title('3D view: chains launched at Re=1000 and Re=8000 targets (squares = starts)',
                  fontsize=style.TITLE_FS)
    for d_, ext in (('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')):
        p_ = f'{d_}/offmanifold_travel_3d.{ext}'; fig2.savefig(p_); print('  written', p_)


if __name__ == '__main__':
    argparse.ArgumentParser().parse_args(); main()
