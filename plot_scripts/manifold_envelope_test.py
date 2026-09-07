"""Is the manifold gap of the Re=8000 reconstructions purely their spectral envelope?

True Re=8000 samples are rescaled shell by shell to carry the measured mean spectral
envelope of the model's reconstructions (unguided and gated), then projected into the
canonical basis beside the actual reconstruction clouds. If the open markers land on the
filled ones, the displacement of the reconstruction clouds from the true cloud is fully
explained by the per shell energy envelope, with no contribution from structure the
projection cannot see. A fixed attenuation ladder of the top octave calibrates the axes.

  JAX_PLATFORMS=cpu python plot_scripts/manifold_envelope_test.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import style

M = 'r8kp02-0599'


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
           for R in (1000, 2000, 4000, 8000)}
    X = np.concatenate(list(gts.values())); mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P = lambda A: (A - mu) @ Vt[:2].T

    S = np.load('base_results/regime_audit_re8000.npz', allow_pickle=True)
    Eg = np.asarray(S['8000|GT||E'])
    g8 = np.load('base_results/fields/re8000/GT.npz')['x'].astype(np.float32)
    fx = np.fft.fftfreq(256) * 256
    KX, KY = np.meshgrid(fx, fx, indexing='ij')
    SHELL = np.clip(np.round(np.sqrt(KX ** 2 + KY ** 2)).astype(int), 0, 127)

    def reshape_env(ratio):
        gain = np.sqrt(np.maximum(ratio, 1e-6))[SHELL]
        F = np.fft.fft2(g8, axes=(1, 2)) * gain[None, :, :, None]
        return np.real(np.fft.ifft2(F, axes=(1, 2))).astype(np.float32)

    fig, ax = plt.subplots(figsize=(10.6, 5.8), constrained_layout=True)
    for R, g in gts.items():
        v = P(g); tgt = R == 8000
        col = style.vivid(style.REGIME_COLOR[R]) if tgt else style.REGIME_COLOR[R]
        ax.scatter(v[:, 0], v[:, 1], s=20 if tgt else 13, color=col,
                   alpha=.55 if tgt else .2, linewidths=0, zorder=2,
                   label=f'Re={R} truth' if tgt else None)
    for lab, short, key, col, mk in (
            ('r8k unguided recons', 'unguided', '8000|r8kp02-0599|K3|mop0.2_0||E', '#8a5cd6', 'D'),
            ('r8k + gate recons', 'gate', '8000|r8kp02-0599|K3|v7bandgate||E', '#0f9e78', '^')):
        f = {'D': 'r8kp02-0599__K3__mop0.2_0', '^': 'r8kp02-0599__K3__v7bandgate'}[mk]
        rc = P(spec(np.load(f'base_results/fields/re8000/{f}.npz')['x']
                    .astype(np.float32)))
        ax.scatter(rc[:, 0], rc[:, 1], s=26, marker=mk, color=col, alpha=.5,
                   linewidths=0, zorder=4, label=lab)
        ratio = np.asarray(S[key]) / np.maximum(Eg, 1e-20)
        pe = P(spec(reshape_env(ratio)))
        ax.scatter(pe[:, 0], pe[:, 1], s=30, marker='o', facecolors='none',
                   edgecolors=col, linewidths=1.0, alpha=.8, zorder=5,
                   label=f'true Re=8000, {short} envelope')
        print(f'{lab}: recon median ({np.median(rc[:,0]):.2f},{np.median(rc[:,1]):.2f})'
              f'  envelope median ({np.median(pe[:,0]):.2f},{np.median(pe[:,1]):.2f})')
    for att in (0.5, 0.25, 0.1):
        ratio = np.ones(128); ratio[64:] = att
        p = np.median(P(spec(reshape_env(ratio))), axis=0)
        ax.scatter(*p, marker='*', s=200, color='#4a4e57', zorder=7,
                   edgecolors='white', linewidths=.6)
        ax.annotate(f'$k{{\\geq}}64\\times{att}$', p, textcoords='offset points',
                    xytext=(10, -12) if att == 0.1 else (8, 4), fontsize=8,
                    color='#4a4e57', fontweight='bold')
    ax.set_xlabel('PC1  (log-spectral shape, Reynolds axis)')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=style.LEG_FS - 2, loc='lower left', framealpha=.92, ncol=2)
    ax.set_title('True Re=8000 flows given the reconstruction envelopes land on the '
                 'reconstruction clouds', fontsize=style.TITLE_FS)
    ax.grid(alpha=.25)
    for d_, ext in (('docs/figs_overleaf', 'pdf'), ('plotting/figs', 'png')):
        p_ = f'{d_}/manifold_envelope_test.{ext}'; fig.savefig(p_); print('written', p_)


if __name__ == '__main__':
    main()
