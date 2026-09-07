"""What each finetune actually ADDS to a denoising step, as a map.

One test input, noised to several chain-relevant t with the same noise. Each panel is the middle
frame of dx0 = x0hat_ft - x0hat_base, the change a finetune makes to the one-step clean estimate,
on a scale shared across the three models at each t. Left column: the input reconstruction and,
below, its local fine-scale energy for orientation.

  JAX_PLATFORMS=cpu python plot_scripts/eps_correction_maps.py [--idx 27]
"""
import os, sys, argparse, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, 'src/ddpo_ft')
os.environ.setdefault('BASE_CKPT', '/tmp/ema_ckpts/ema_base_0299.pkl')
import numpy as np
import matplotlib.pyplot as plt
import style

SIG = 4.7988
TS = [150, 125, 100, 75, 50]
FT = [('mt1k-0499', 'Re=1000 finetune', 'monitoring/ddpo_re1000_match_ckpts/ddpo_re1000_iter0499.pkl'),
      ('mt2k-0599', 'Re=2000 finetune', 'monitoring/ddpo_re2000_match_ckpts/ddpo_re1000_iter0599.pkl'),
      ('r8kp02-0599', 'Re=8000 finetune', 'monitoring/ddpo_re8000_pdew02_ckpts/ddpo_re1000_iter0599.pkl')]


def params_of(d):
    return d.get('ema_params') if d.get('ema_params') is not None else d['params']


def make(idx):
    import jax.numpy as jnp
    from train_claude import build_base_ddpm
    from viz_energy import local_hik_energy
    style.apply(style.BASE_FS, 170)
    ddpm, bp, _ = build_base_ddpm(); ab = ddpm.alpha_bar
    fts = {n: params_of(pickle.load(open(p, 'rb'))) for n, _, p in FT}
    x0 = np.load('base_results/fields/re1000/recon.npz')['x'].astype(np.float32)
    i = idx if idx is not None else 27
    x = jnp.asarray(x0[i:i + 1])
    rng = np.random.default_rng(0)
    nrow, ncol = len(TS), 1 + len(FT)
    fig, ax = plt.subplots(nrow, ncol, figsize=(2.75 * ncol, 2.62 * nrow), constrained_layout=True)
    w_in = x0[i, ..., 1] * SIG
    vin = float(np.percentile(np.abs(w_in), 99.5))
    for r, t in enumerate(TS):
        eps = jnp.asarray(rng.normal(size=x.shape).astype(np.float32))
        sa, s1 = float(np.sqrt(ab[t])), float(np.sqrt(1 - ab[t]))
        xt = sa * x + s1 * eps
        ta = jnp.full((1,), t, jnp.int32)
        e_b = np.asarray(ddpm.unet.apply({'params': bp}, xt, ta, train=False))
        # dx0hat = -(s1/sa) * d_eps : the change to the one-step clean estimate, in field units
        d = {n: -(s1 / sa) * (np.asarray(ddpm.unet.apply({'params': p}, xt, ta, train=False)) - e_b)
             for n, p in fts.items()}
        dv = float(np.percentile(np.abs(np.stack([d[n][0, ..., 1] for n in d])), 99.0))
        a0 = ax[r, 0]
        if r == 0:
            a0.imshow(w_in, cmap='RdBu_r', vmin=-vin, vmax=vin, interpolation='nearest')
            a0.set_title('input (base recon)', fontsize=style.LEG_FS + 1)
        else:
            a0.imshow(local_hik_energy(w_in, 32, 6.0), cmap='magma', interpolation='nearest')
            if r == 1: a0.set_title('input fine-scale energy', fontsize=style.LEG_FS + 1)
        a0.set_ylabel(f'$t={t}$', fontsize=style.TITLE_FS)
        n_ref = np.linalg.norm(d['mt1k-0499'])
        for c, (n, lab, _) in enumerate(FT):
            a = ax[r, c + 1]
            im = a.imshow(d[n][0, ..., 1] * SIG, cmap='RdBu_r', vmin=-dv * SIG, vmax=dv * SIG,
                          interpolation='nearest')
            if r == 0: a.set_title(lab, fontsize=style.LEG_FS + 1)
            a.text(.02, .02, f'{np.linalg.norm(d[n]) / n_ref:.2f}$\\times$', transform=a.transAxes,
                   fontsize=style.LEG_FS, va='bottom',
                   bbox=dict(fc='white', ec='none', alpha=.8, pad=1.5))
        fig.colorbar(im, ax=ax[r, 1:], shrink=.85, pad=.01)
    for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
    fig.suptitle('The correction each finetune adds to one denoising step\n'
                 '$\\hat{x}_0^{ft}-\\hat{x}_0^{base}$, same input and noise; inset: norm vs the '
                 'Re=1000 finetune', fontsize=style.TITLE_FS)
    for d_, ext in (('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')):
        p = f'{d_}/eps_correction_maps.{ext}'; fig.savefig(p); print('  written', p)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--idx', type=int)
    make(ap.parse_args().idx)
