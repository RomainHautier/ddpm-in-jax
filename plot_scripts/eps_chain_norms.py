"""Does the model's epsilon correction change magnitude through the chain?

Runs the same K3 chain as chain_energy_trace (test triplet 27 at Re=1000, identical noise,
the real eta=1 / temp 0.30 sampler) and at every DDIM step records, on the model's own
state x_t, the norm of its prediction and of its correction against the base model
evaluated on the SAME x_t:

    ||eps_m(x_t, t)||     and     ||eps_m(x_t, t) - eps_base(x_t, t)||

The correction norm is the per-step magnitude of what fine-tuning added; the ratio to
||eps_base|| says how large that is relative to the score being corrected.

  JAX_PLATFORMS=cpu python plot_scripts/eps_chain_norms.py
"""
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

STARTS, NSTEP = [150, 100, 50], 20
ETA, TEMP = 1.0, 0.30
MODELS = [('mt1k-0499', 'Re=1000 ft', 'monitoring/ddpo_re1000_match_ckpts/ddpo_re1000_iter0499.pkl'),
          ('mt2k-0599', 'Re=2000 ft', 'monitoring/ddpo_re2000_match_ckpts/ddpo_re1000_iter0599.pkl'),
          ('r4kp02-0599', 'Re=4000 ft', 'monitoring/ddpo_re4000_pdew02_ckpts/ddpo_re1000_iter0599.pkl'),
          ('r8kp02-0599', 'Re=8000 ft', 'monitoring/ddpo_re8000_pdew02_ckpts/ddpo_re1000_iter0599.pkl')]


def params_of(d): return d.get('ema_params') if d.get('ema_params') is not None else d['params']


def main(idx=27):
    import jax.numpy as jnp
    from train_claude import build_base_ddpm
    style.apply(style.BASE_FS, 150)
    ddpm, bp, _ = build_base_ddpm(); ab = ddpm.alpha_bar
    params = {n: params_of(pickle.load(open(p, 'rb'))) for n, _, p in MODELS}
    rc = jnp.asarray(np.load('base_results/fields/re1000/recon.npz')['x']
                     .astype(np.float32)[idx:idx + 1])
    rng = np.random.default_rng(5)
    noises = [jnp.asarray(rng.normal(size=rc.shape).astype(np.float32)) for _ in STARTS]
    norms = {}
    for n, lab, _ in MODELS:
        zrng = np.random.default_rng(7)
        p = params[n]; x_prev = rc; rows = []
        for ci, S in enumerate(STARTS):
            ts = np.linspace(S, 0, NSTEP + 1).astype(int)
            x = float(np.sqrt(ab[S])) * x_prev + float(np.sqrt(1 - ab[S])) * noises[ci]
            for a, b in zip(ts[:-1], ts[1:]):
                ta = jnp.full((1,), int(a), jnp.int32)
                e = ddpm.unet.apply({'params': p}, x, ta, train=False)
                eb = ddpm.unet.apply({'params': bp}, x, ta, train=False)
                de = np.asarray(e - eb)
                rows.append((float(np.linalg.norm(np.asarray(e))),
                             float(np.linalg.norm(np.asarray(eb))),
                             float(np.linalg.norm(de))))
                xh = (x - float(np.sqrt(1 - ab[a])) * e) / float(np.sqrt(ab[a]))
                if b > 0:
                    ab_c, ab_n = float(ab[a]), float(ab[b])
                    sig = ETA * np.sqrt((1 - ab_n) / (1 - ab_c)) * np.sqrt(1 - ab_c / ab_n)
                    x = (np.sqrt(ab_n) * xh + np.sqrt(max(1 - ab_n - sig ** 2, 0.0)) * e
                         + TEMP * sig * jnp.asarray(zrng.normal(size=xh.shape)
                                                    .astype(np.float32)))
                else:
                    x = xh
            x_prev = x
        norms[n] = np.array(rows)   # (60, 3): ||e||, ||e_base||, ||de||
        m = norms[n]
        print(f'  {lab:12s} ||de|| pass means: '
              + '  '.join(f'{m[ci*NSTEP:(ci+1)*NSTEP, 2].mean():.1f}' for ci in range(3))
              + f'   rel to ||e_base||: '
              + '  '.join(f'{(m[ci*NSTEP:(ci+1)*NSTEP, 2]/m[ci*NSTEP:(ci+1)*NSTEP, 1]).mean()*100:.1f}%'
                          for ci in range(3)))
    np.savez('/tmp/claude-2001/-home-rhautier-ddpm-jax/24ff525a-722f-4c10-b5d9-663bc835ccb1/'
             'scratchpad/eps_chain_norms.npz', nstep=NSTEP, **norms)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10.6, 7.4), sharex=True,
                                  constrained_layout=True)
    xs = np.arange(1, 3 * NSTEP + 1)
    for n, lab, _ in MODELS:
        col = style.MODEL_COLOR.get(n, '#232823')
        ax.semilogy(xs, norms[n][:, 2], '-', color=col, lw=1.9, label=lab)
        ax2.plot(xs, 100 * norms[n][:, 2] / norms[n][:, 1], '-', color=col, lw=1.9)
    for a_ in (ax, ax2):
        for ci in range(3):
            a_.axvline(ci * NSTEP + 0.5, color='#9aa198', lw=1, ls='--')
        a_.grid(alpha=.25)
    for ci, S in enumerate(STARTS):
        ax.text(ci * NSTEP + 1, ax.get_ylim()[1] * 0.85, f'pass at $t={S}$',
                fontsize=style.LEG_FS - 1.5, color='#5d645d', va='top')
    ax.set_ylabel(r'$\|\epsilon_m - \epsilon_{\rm base}\|$  (same input)')
    ax.legend(fontsize=style.LEG_FS, loc='upper right')
    ax2.set_ylabel(r'correction / base score  (%)')
    ax2.set_xlabel('DDIM step through the K3 chain')
    fig.suptitle('The magnitude of the fine-tuned correction through the chain '
                 f'(test triplet {idx}, real sampler)', fontsize=style.SUP_FS)
    for d_, ext in (('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')):
        p_ = f'{d_}/eps_chain_norms.{ext}'; fig.savefig(p_); print('  written', p_)


if __name__ == '__main__':
    main()
