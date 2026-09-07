"""One sample's fine-band energy through the whole K3 chain, model by model.

Starts from the LR input's energy level, through the base-DDIM reconstruction, then the three
noise-and-denoise passes of the K3 chain [150,100,50]. At every DDIM step the clean estimate's
E[32,96) is recorded. The claim on display: energy jumps AT each re-noising pass and is flat
within one, so the dose is set by the passes, not by the steps.

  JAX_PLATFORMS=cpu python plot_scripts/chain_energy_trace.py [--idx 27]
"""
import os, sys, argparse, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, 'src/ddpo_ft')
os.environ.setdefault('BASE_CKPT', '/tmp/ema_ckpts/ema_base_0299.pkl')
import numpy as np
import matplotlib.pyplot as plt
import style

STARTS, NSTEP = [150, 100, 50], 20
ETA = float(os.environ.get('TRACE_ETA', '0'))
MODELS = [('base', 'base', None),
          ('mt1k-0499', 'Re=1000 ft', 'monitoring/ddpo_re1000_match_ckpts/ddpo_re1000_iter0499.pkl'),
          ('mt2k-0599', 'Re=2000 ft', 'monitoring/ddpo_re2000_match_ckpts/ddpo_re1000_iter0599.pkl'),
          ('r4kp02-0599', 'Re=4000 ft', 'monitoring/ddpo_re4000_pdew02_ckpts/ddpo_re1000_iter0599.pkl'),
          ('r8kp02-0599', 'Re=8000 ft', 'monitoring/ddpo_re8000_pdew02_ckpts/ddpo_re1000_iter0599.pkl')]


def params_of(d): return d.get('ema_params') if d.get('ema_params') is not None else d['params']


def make(idx):
    import jax.numpy as jnp
    from train_claude import build_base_ddpm
    from src.rewards import make_spectrum_fn
    style.apply(style.BASE_FS, 150)
    ddpm, bp, _ = build_base_ddpm(); ab = ddpm.alpha_bar
    sf = make_spectrum_fn(256)
    E = lambda x: float(np.asarray(sf(x))[0, 32:96].sum())
    params = {n: (bp if p is None else params_of(pickle.load(open(p, 'rb')))) for n, _, p in MODELS}
    DUMP = ('/tmp/claude-2001/-home-rhautier-ddpm-jax/24ff525a-722f-4c10-b5d9-663bc835ccb1/'
            f'scratchpad/chain_energy_traces{"_eta1" if ETA > 0 else ""}.npz')
    if os.environ.get('REPLOT') and os.path.exists(DUMP):
        d = np.load(DUMP)
        traces = {n: d[n] for n, _, _ in MODELS}
        Egt, Elr, Erc = float(d['E_gt']), float(d['E_lr']), float(d['E_rc'])
        i = idx if idx is not None else 27
        return draw(traces, Egt, Elr, Erc, i)
    F = 'base_results/fields/re1000'
    i = idx if idx is not None else 27
    gt = np.load(f'{F}/GT.npz')['x'].astype(np.float32)[i:i + 1]
    lr = np.load(f'{F}/LR.npz')['x'].astype(np.float32)[i:i + 1]
    rc = jnp.asarray(np.load(f'{F}/recon.npz')['x'].astype(np.float32)[i:i + 1])
    rng = np.random.default_rng(5)
    noises = [jnp.asarray(rng.normal(size=rc.shape).astype(np.float32)) for _ in STARTS]
    traces = {}
    for n, lab, _ in MODELS:
        zrng = np.random.default_rng(7)   # per-step noise shared across models
        p = params[n]; x_prev = rc; tr = []
        for ci, S in enumerate(STARTS):
            ts = np.linspace(S, 0, NSTEP + 1).astype(int)
            x = float(np.sqrt(ab[S])) * x_prev + float(np.sqrt(1 - ab[S])) * noises[ci]
            for a, b in zip(ts[:-1], ts[1:]):
                ta = jnp.full((1,), int(a), jnp.int32)
                e = ddpm.unet.apply({'params': p}, x, ta, train=False)
                xh = (x - float(np.sqrt(1 - ab[a])) * e) / float(np.sqrt(ab[a]))
                tr.append(E(xh))
                if b > 0:
                    if ETA > 0:
                        ab_c, ab_n = float(ab[a]), float(ab[b])
                        sig = ETA * np.sqrt((1 - ab_n) / (1 - ab_c)) * np.sqrt(1 - ab_c / ab_n)
                        x = (np.sqrt(ab_n) * xh + np.sqrt(max(1 - ab_n - sig ** 2, 0.0)) * e
                             + 0.30 * sig * jnp.asarray(
                                 zrng.normal(size=xh.shape).astype(np.float32)))
                    else:
                        x = float(np.sqrt(ab[b])) * xh + float(np.sqrt(1 - ab[b])) * e
                else:
                    x = xh
            x_prev = x
        traces[n] = np.array(tr)
        print(f'  {lab:12s} end E ratio to GT: {tr[-1] / E(jnp.asarray(gt)):.2f}')
    np.savez(DUMP, E_gt=E(jnp.asarray(gt)), E_lr=E(jnp.asarray(lr)), E_rc=E(rc), nstep=NSTEP,
             **{n: traces[n] for n, _, _ in MODELS})
    draw(traces, E(jnp.asarray(gt)), E(jnp.asarray(lr)), E(rc), i)


def draw(traces, Egt, Elr, Erc, i):
    fig, ax = plt.subplots(figsize=(10.6, 5.4), constrained_layout=True)
    xs = np.arange(1, 3 * NSTEP + 1)
    for n, lab, _ in MODELS:
        ax.semilogy(xs, traces[n], '-', lw=1.9, label=lab,
                    color=style.MODEL_COLOR.get({'base': 'base0'}.get(n, n), '#232823'))
    for lvl, lab, col, ls in ((Egt, "this sample's ground truth", style.GT_COLOR, '-'),
                              (Erc, 'base-DDIM reconstruction (chain input)', '#d4770a', ':'),
                              (Elr, 'LR input', '#b8399e', ':')):
        ax.axhline(lvl, color=col, lw=1.6, ls=ls)
        ax.text(3 * NSTEP + 0.7, lvl, lab, fontsize=style.LEG_FS - 1.5, color=col, va='center')
    for ci, S in enumerate(STARTS):
        ax.axvline(ci * NSTEP + 0.5, color='#9aa198', lw=1, ls='--')
        ax.text(ci * NSTEP + 1, ax.get_ylim()[1] * 0.92, f'renoise to $t={S}$',
                fontsize=style.LEG_FS - 1.5, color='#5d645d', va='top')
    ax.set_xlabel('DDIM step through the K3 chain')
    ax.set_ylabel("clean estimate's fine-band energy  $E_{[32,96)}$")
    ax.set_title('the energy level is set at each re-noising pass and holds within it'
                 if ETA == 0 else
                 'the real sampler ($\\eta=1$, temp $0.30$): same injections, plus the '
                 'stochastic contribution', fontsize=style.TITLE_FS)
    ax.legend(fontsize=style.LEG_FS, loc='center right')
    ax.set_xlim(0.5, 3 * NSTEP + 22)
    fig.suptitle(f'One sample (test triplet {i}) through the full K3 chain, every model, '
                 'identical noise', fontsize=style.SUP_FS)
    for d_, ext in (('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')):
        p_ = f'{d_}/chain_energy_trace{"_eta1" if ETA > 0 else ""}.{ext}'
        fig.savefig(p_); print('  written', p_)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--idx', type=int)
    make(ap.parse_args().idx)
