"""Placement as a function of wavenumber, not just per band.

For each narrow shell [k, k+dk) the local energy density map is formed the same way the band
placement metric does it - bandpass, square, Gaussian-smooth (sigma=6) - and correlated against
the ground truth's, ONE CORRELATION PER TRIPLET, then aggregated. This resolves where in k a
configuration stops putting energy in the right place, which the five-band summary can only hint at.

  JAX_PLATFORMS=cpu python plot_scripts/placement_vs_k.py [--dk 4] [--out FILE]
"""
import os, sys, argparse
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'plot_scripts')
import numpy as np, style
import matplotlib.pyplot as plt

CONFIG = dict(
    fields='base_results/fields/re1000',
    # LR and recon are the two things that exist BEFORE any sampling: the 4x coarse observation
    # interpolated back to 256, and the base-DDIM reconstruction that is fed to the K3 chain and
    # used as the fine-tuning input pool. They set the floor every model result should be read
    # against - placement the model inherited rather than produced.
    # Colour = model, linestyle = dial, matching regime_spectra_dial.py and perband_vs_re.py.
    # The gated dial is deliberately NOT here: this figure is the MATCHED comparison, the
    # fine-tune's own reward optimised two ways, and the gate optimises something else.
    rows=[('LR', 'the 4x coarse observation (interpolated)', '#b8399e', ':'),
          ('recon', 'base-DDIM recon (the chain/fine-tune input)', '#d4770a', ':'),
          ('base0__K3__mop0.2_0', 'base, unguided', '#28658a', '--'),
          ('base0__K3__tapp0.2_3', 'base + matched dial', '#28658a', '-'),
          ('mt1k-0499__K3__mop0.2_0', 'fine-tune, unguided', '#c22f4f', '--'),
          ('mt1k-0499__K3__tapp0.2_3', 'fine-tune + matched dial', '#c22f4f', '-'),
          # plot=False: computed and written to the npz (so the report's per-k table can show
          # them) but kept off the figure, which is the MATCHED comparison only.
          ('base0__K3__v6gate', 'base + target-gated dial', '#0f9e78', '-', False),
          ('mt1k-0499__K3__v7bandgate', 'fine-tune + target-gated dial', '#0f9e78', '--', False)],
    dk=4, kmax=100, sigma=6.0, sig=4.7988, out='docs/figs_overleaf/re1000_placement_vs_k.pdf')
p = argparse.ArgumentParser(); p.add_argument('--dk', type=int); p.add_argument('--out')
p.add_argument('--rows'); a = p.parse_args()
if a.dk: CONFIG['dk'] = a.dk
if a.out: CONFIG['out'] = a.out
if a.rows: CONFIG['rows'] = [r for r in CONFIG['rows'] if r[0] in a.rows.split(',')]

CONFIG['rows'] = [(r + (True,))[:5] for r in CONFIG['rows']]
F, N, SIG = CONFIG['fields'], 256, CONFIG['sig']
fy = np.fft.fftfreq(N) * N
kmag = np.sqrt(fy[:, None] ** 2 + fy[None, :] ** 2)
gsm = np.exp(-2.0 * (np.pi * CONFIG['sigma']) ** 2 * ((fy[:, None] / N) ** 2 + (fy[None, :] / N) ** 2))
EDGES = np.arange(1, CONFIG['kmax'] + 1, CONFIG['dk'])
CENTRES = EDGES[:-1] + CONFIG['dk'] / 2.0


def shell_maps(w):
    """(nbands, n, 256, 256) local energy density per shell, w already in physical units"""
    Fw = np.fft.fft2(w)
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = ((kmag >= lo) & (kmag < hi)).astype(np.float32)
        bp = np.real(np.fft.ifft2(Fw * m))
        out.append(np.real(np.fft.ifft2(np.fft.fft2(bp ** 2) * gsm)))
    return np.stack(out)


gt = np.load(f'{F}/GT.npz')['x'].astype(np.float32)
G = shell_maps(gt[..., 1] * SIG)
print(f"{len(CENTRES)} shells of width {CONFIG['dk']}, {gt.shape[0]} triplets")

style.apply()
fig, AX = plt.subplots(1, 2, figsize=(13.5, 4.8))
curves = {}
for name, lab, c, ls, do_plot in CONFIG['rows']:
    f = f'{F}/{name}.npz'
    if not os.path.exists(f): print(f"  {name}: absent"); continue
    _x = np.load(f)['x'].astype(np.float32)
    # ALIGNMENT GUARD: see deficit_panel.py - GT and model rows must be the same pool
    assert len(_x) == gt.shape[0], (f'{name}: {len(_x)} triplets vs GT {gt.shape[0]} - '
                                    'mask the GT before comparing')
    Y = shell_maps(_x[..., 1] * SIG)
    v = np.array([[np.corrcoef(Y[b, i].ravel(), G[b, i].ravel())[0, 1]
                   for i in range(Y.shape[1])] for b in range(len(CENTRES))])
    curves[lab] = v
    med = np.median(v, 1)
    if do_plot:
        AX[0].plot(CENTRES, med, ls, color=c, lw=1.9, label=lab)
        AX[0].fill_between(CENTRES, np.percentile(v, 25, 1), np.percentile(v, 75, 1),
                           color=c, alpha=.13, lw=0)
    print(f"  {lab:<34} k=10: {med[np.argmin(abs(CENTRES-10))]:.3f}   "
          f"k=32: {med[np.argmin(abs(CENTRES-32))]:.3f}   k=64: {med[np.argmin(abs(CENTRES-64))]:.3f}"
          f"   k=88: {med[np.argmin(abs(CENTRES-88))]:.3f}")
AX[0].set_xlabel('wavenumber k'); AX[0].set_ylabel('placement, per triplet (median, IQR)')
AX[0].set_title('where the energy stops being in the right place', fontsize=style.TITLE_FS)
AX[0].axvline(96, color=style.INK, lw=1, ls=':'); style.shade_bands(AX[0])
AX[0].set_ylim(0, 1.02); AX[0].legend(fontsize=style.LEG_FS, loc='lower left')

BASELINE = 'base, unguided'
base = curves.get(BASELINE)
FLOORS = {'the 4x coarse observation (interpolated)', 'base-DDIM recon (the chain/fine-tune input)'}
for (name, lab, c, ls, do_plot) in CONFIG['rows']:
    if not do_plot or lab == BASELINE or lab in FLOORS or lab not in curves or base is None:
        continue
    d = np.median(curves[lab], 1) - np.median(base, 1)
    AX[1].plot(CENTRES, d, ls, color=c, lw=1.9, label=lab)
# the reference itself: every curve here is a difference FROM the unguided base, so the base is
# the zero line. Give it a legend entry or it looks like it is missing from the panel.
AX[1].axhline(0, color=style.GT_COLOR, lw=1.8, label=f'{BASELINE} (the reference)')
style.shade_bands(AX[1]); AX[1].axvline(96, color=style.INK, lw=1, ls=':')
AX[1].set_xlabel('wavenumber k'); AX[1].set_ylabel('placement minus the base model')
AX[1].set_title('difference from the unguided base', fontsize=style.TITLE_FS)
AX[1].legend(fontsize=style.LEG_FS, loc='lower left')
AX[1].set_ylim(-0.16, 0.06)
fig.suptitle(f'Per-wavenumber placement, Re=1000, {gt.shape[0]} held-out triplets '
             f'(shells of width {CONFIG["dk"]})', fontsize=12, y=1.0)
fig.tight_layout(rect=[0, 0, 1, .94])
os.makedirs(os.path.dirname(CONFIG['out']), exist_ok=True)
fig.savefig(CONFIG['out'], bbox_inches='tight')
fig.savefig(CONFIG['out'].replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
np.savez('base_results/placement_vs_k.npz', centres=CENTRES,
         **{l: v for l, v in curves.items()})
print('wrote', CONFIG['out'])
