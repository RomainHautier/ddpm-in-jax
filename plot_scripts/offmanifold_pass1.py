"""The REAL trajectory of the sample through the first denoising pass.

The top-view figures plot clean estimates, and the hop from the reconstruction to the
first estimate looks like a straight jump. It is not. Renoising to t=150 moves the actual
sample to a noise-dominated spectral state far off the manifold (identical for every
model, the noise is shared), and each model then walks it back over 20 DDIM steps with
the real eta=1 / temp 0.30 sampler. This figure projects the actual states x_t at every
step of pass 1, next to the clean-estimate path, for the Re=1000 target sample of the
top-view figure.

  JAX_PLATFORMS=cpu python plot_scripts/offmanifold_pass1.py
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
R = 1000
S0, NSTEP = 150, 20
ALLPASSES = bool(os.environ.get('ALLPASSES'))
STARTS = [150, 100, 50] if ALLPASSES else [150]
ETA, TEMP = 1.0, 0.30
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
            [np.asarray(sf(jnp.asarray(x[i:i + 32]))) for i in range(0, len(x), 32)]
        )[:, KLO:KHI], 1e-20))

    params = {n: (bp if p is None else params_of(pickle.load(open(p, 'rb'))))
              for n, _, p in MODELS}
    gts = {Rb: spec(np.load(f'base_results/fields/re{Rb}/GT.npz')['x'].astype(np.float32))
           for Rb in BASIS_REGS}
    X = np.concatenate(list(gts.values())); mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P = lambda A: (A - mu) @ Vt[:2].T

    # the SAME sample the top view uses at this target: pool-median fine-band energy
    gt = np.load(f'base_results/fields/re{R}/GT.npz')['x'].astype(np.float32)
    rc = np.load(f'base_results/fields/re{R}/recon.npz')['x'].astype(np.float32)
    E = np.asarray([np.exp(sp)[31:].sum() for sp in gts[R]])
    i = int(np.argsort(E)[len(E) // 2])
    rc1 = jnp.asarray(rc[i:i + 1])
    rng = np.random.default_rng(11)
    noises = [jnp.asarray(rng.normal(size=(1, 256, 256, 3)).astype(np.float32))
              for _ in STARTS]
    zss = [[jnp.asarray(rng.normal(size=(1, 256, 256, 3)).astype(np.float32))
            for _ in range(NSTEP)] for _ in STARTS]

    ts = np.linspace(S0, 0, NSTEP + 1).astype(int)
    paths = {}
    for n, lab, _ in MODELS:
        p = params[n]
        x_prev = rc1; passes = []
        for ci, S in enumerate(STARTS):
            tss = np.linspace(S, 0, NSTEP + 1).astype(int)
            x = float(np.sqrt(ab[S])) * x_prev + float(np.sqrt(1 - ab[S])) * noises[ci]
            xt = [P(spec(np.asarray(x)))[0]]
            for si, (a, b) in enumerate(zip(tss[:-1], tss[1:])):
                ta = jnp.full((1,), int(a), jnp.int32)
                e = ddpm.unet.apply({'params': p}, x, ta, train=False)
                xh = (x - float(np.sqrt(1 - ab[a])) * e) / float(np.sqrt(ab[a]))
                if b > 0:
                    ab_c, ab_n = float(ab[a]), float(ab[b])
                    sig = ETA * np.sqrt((1 - ab_n) / (1 - ab_c)) * np.sqrt(1 - ab_c / ab_n)
                    x = (np.sqrt(ab_n) * xh + np.sqrt(max(1 - ab_n - sig ** 2, 0.0)) * e
                         + TEMP * sig * zss[ci][si])
                else:
                    x = xh
                xt.append(P(spec(np.asarray(x)))[0])
            passes.append(np.array(xt)); x_prev = x
        paths[n] = passes
        print(f'  {lab}: ' + ' | '.join(
            f'launch {ps[0][0]:.1f} -> {ps[-1][0]:.1f}' for ps in passes))

    prc = P(spec(rc[i:i + 1]))[0]; pgt = P(gts[R][i:i + 1])[0]
    fig, ax = plt.subplots(figsize=(10.6, 5.6), constrained_layout=True)
    for Rb, g in gts.items():
        v = P(g); tgt = Rb == R
        col = style.vivid(style.REGIME_COLOR[Rb]) if tgt else style.REGIME_COLOR[Rb]
        ax.scatter(v[:, 0], v[:, 1], s=22 if tgt else 13, color=col,
                   alpha=.75 if tgt else .22, linewidths=0, zorder=2)
    x0 = paths['base'][0][0]
    for n, lab, _ in MODELS:
        col = style.MODEL_COLOR.get({'base': 'base0'}.get(n, n), '#232823')
        prev = prc
        for ci, xt in enumerate(paths[n]):
            ax.plot([prev[0], xt[0, 0]], [prev[1], xt[0, 1]], ls=':',
                    color=col if ci else '#5d645d', lw=1.1, alpha=.65, zorder=4)
            ax.plot(xt[:, 0], xt[:, 1], '-', color=col, lw=1.7, alpha=.9, zorder=5,
                    label=lab if ci == 0 else None)
            ax.scatter(xt[1:-1, 0], xt[1:-1, 1], s=11, color=col, zorder=6,
                       linewidths=0)
            if ci:
                ax.scatter(*xt[0], marker='X', s=60, color=col, zorder=7,
                           edgecolors='white', linewidths=.6)
            prev = xt[-1]
        ax.scatter(*paths[n][-1][-1], marker='D', s=60, color=col, zorder=7,
                   edgecolors='black', linewidths=.8)
    ax.scatter(*x0, marker='X', s=130, color='#232823', zorder=8,
               edgecolors='white', linewidths=.8)
    ax.annotate('noised to $t=150$ (shared)', x0, textcoords='offset points',
                xytext=(-14, 16), fontsize=style.LEG_FS - 1, ha='right')
    for si in (2, 5, 10, 15):
        ax.annotate(f'$t={ts[si]}$', paths['base'][0][si], textcoords='offset points',
                    xytext=(4, -11), fontsize=style.LEG_FS - 3, color='#5d645d')
    ax.scatter(*prc, marker='s', s=80, color='#d4770a', zorder=8,
               edgecolors='white', linewidths=.7)
    ax.scatter(*pgt, marker='*', s=320, color=style.vivid(style.REGIME_COLOR[R]),
               zorder=9, edgecolors='black', linewidths=.9)
    ax.set_xlabel('PC1  (log-spectral shape, Reynolds axis)')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=style.LEG_FS - 1, loc='upper left', framealpha=.92)
    ax.set_title('The actual state $x_t$ through '
                 + ('all three passes' if ALLPASSES else 'the first pass')
                 + ', launches (X) and walks back', fontsize=style.TITLE_FS)
    ax.grid(alpha=.25)
    for d_, ext in (('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')):
        p_ = f'{d_}/offmanifold_pass{"es" if ALLPASSES else "1"}.{ext}'; fig.savefig(p_); print('  written', p_)


if __name__ == '__main__':
    main()
