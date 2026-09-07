"""How fast does each finetune move a sample per denoising step, in manifold and visual space?

The SAME input, noised to t=100 with the SAME noise, is denoised by a deterministic DDIM segment
(10 steps) under each model. Every intermediate clean estimate is projected into the log-spectral
PCA plane of the pooled ground truths (PC1 is effectively the Reynolds axis), giving one
trajectory per model; the right panel shows the distance travelled per step; the bottom row shows
where each model lands, visually, from the identical start.

  JAX_PLATFORMS=cpu python plot_scripts/step_trajectory.py [--idx 27]
"""
import os, sys, argparse, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, 'src/ddpo_ft')
os.environ.setdefault('BASE_CKPT', '/tmp/ema_ckpts/ema_base_0299.pkl')
import numpy as np
import matplotlib.pyplot as plt
import style

SIG, KLO, KHI = 4.7988, 1, 96
MODELS = [('base', 'base', None),
          ('mt1k-0499', 'Re=1000 ft', 'monitoring/ddpo_re1000_match_ckpts/ddpo_re1000_iter0499.pkl'),
          ('mt2k-0599', 'Re=2000 ft', 'monitoring/ddpo_re2000_match_ckpts/ddpo_re1000_iter0599.pkl'),
          ('r8kp02-0599', 'Re=8000 ft', 'monitoring/ddpo_re8000_pdew02_ckpts/ddpo_re1000_iter0599.pkl')]
NSTEP, T0 = 10, 100


def params_of(d): return d.get('ema_params') if d.get('ema_params') is not None else d['params']


def make(idx):
    import jax, jax.numpy as jnp
    from train_claude import build_base_ddpm
    from src.rewards import make_spectrum_fn
    style.apply(style.BASE_FS, 160)
    ddpm, bp, _ = build_base_ddpm(); ab = ddpm.alpha_bar
    sf = make_spectrum_fn(256)
    spec = lambda x: np.log(np.maximum(
        np.concatenate([np.asarray(sf(jnp.asarray(x[i:i+32]))) for i in range(0, len(x), 32)])[:, KLO:KHI], 1e-20))
    # --- the manifold basis: pooled GT of the three regimes (as manifold_pca) ---
    gts, cols = {}, {}
    for R in (1000, 2000, 8000):
        gts[R] = spec(np.load(f'base_results/fields/re{R}/GT.npz')['x'].astype(np.float32))
        cols[R] = style.REGIME_COLOR[R]
    X = np.concatenate(list(gts.values())); mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    proj = lambda A: (A - mu) @ Vt[:2].T

    params = {n: (bp if p is None else params_of(pickle.load(open(p, 'rb')))) for n, _, p in MODELS}
    x0 = np.load('base_results/fields/re1000/recon.npz')['x'].astype(np.float32)
    i = idx if idx is not None else 27
    x_in = jnp.asarray(x0[i:i + 1])
    eps0 = jnp.asarray(np.random.default_rng(3).normal(size=x_in.shape).astype(np.float32))
    ts = np.linspace(T0, 0, NSTEP + 1).astype(int)
    trajs, finals = {}, {}
    for n, lab, _ in MODELS:
        x = float(np.sqrt(ab[T0])) * x_in + float(np.sqrt(1 - ab[T0])) * eps0
        pts = []
        for a, b in zip(ts[:-1], ts[1:]):
            ta = jnp.full((1,), int(a), jnp.int32)
            e = ddpm.unet.apply({'params': params[n]}, x, ta, train=False)
            xh = (x - float(np.sqrt(1 - ab[a])) * e) / float(np.sqrt(ab[a]))
            pts.append(np.asarray(xh))
            x = float(np.sqrt(ab[b])) * xh + float(np.sqrt(1 - ab[b])) * e if b > 0 else xh
        trajs[n] = proj(np.concatenate([spec(np.asarray(p)) for p in pts]))
        finals[n] = np.asarray(x)[0, ..., 1] * SIG

    fig = plt.figure(figsize=(12.6, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1])

    # ---- position along the Reynolds axis, step by step, with the truth clouds as bands ----
    a = fig.add_subplot(gs[0, :2])
    for R, g in gts.items():
        p1 = proj(g)[:, 0]
        lo, hi = np.percentile(p1, [25, 75])
        a.axhspan(lo, hi, color=cols[R], alpha=.18, lw=0)
        a.text(NSTEP - 0.15, np.median(p1), f'Re={R} truth', ha='right', va='center',
               fontsize=style.LEG_FS - 1, color=cols[R])
    for n, lab, _ in MODELS:
        pc1 = trajs[n][:, 0]
        a.plot(range(1, NSTEP + 1), pc1, '-o', color=style.MODEL_COLOR.get(n, '#232823'),
               lw=1.9, ms=4, label=lab)
    a.set_xlabel('denoising step'); a.set_ylabel('position on PC1 (the Reynolds axis)')
    a.set_title('the Reynolds displacement is applied in the FIRST estimate,\nthen held', fontsize=style.TITLE_FS)
    a.legend(fontsize=style.LEG_FS - 1.5, loc='lower right')

    b = fig.add_subplot(gs[0, 2:])
    for n, lab, _ in MODELS:
        t = trajs[n]
        d = np.linalg.norm(np.diff(t, axis=0), axis=1)
        b.plot(range(1, NSTEP), np.cumsum(d), '-o', color=style.MODEL_COLOR.get(n, '#232823'),
               lw=1.9, ms=3.5, label=lab)
    b.set_xlabel('denoising step'); b.set_ylabel('cumulative distance travelled (PC plane)')
    b.set_title('later steps refine detail (motion off the Reynolds axis)', fontsize=style.TITLE_FS)
    b.legend(fontsize=style.LEG_FS - 1)

    v = float(np.percentile(np.abs(finals['base']), 99.5))
    for c, (n, lab, _) in enumerate(MODELS):
        a2 = fig.add_subplot(gs[1, c])
        a2.imshow(finals[n], cmap='RdBu_r', vmin=-v, vmax=v, interpolation='nearest')
        a2.set_title(lab, fontsize=style.LEG_FS + 1); a2.set_xticks([]); a2.set_yticks([])
    fig.suptitle(f'The same input, the same noise, {NSTEP} DDIM steps from $t={T0}$: each finetune '
                 'jumps a different distance immediately, in the same direction (test triplet %d)' % i,
                 fontsize=style.SUP_FS)
    for d_, ext in (('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')):
        p = f'{d_}/step_trajectory.{ext}'; fig.savefig(p); print('  written', p)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--idx', type=int)
    make(ap.parse_args().idx)
